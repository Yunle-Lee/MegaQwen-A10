# MegaQwen-A10

**A10 优化的 Qwen3-0.6B 融合解码内核**

基于 NVIDIA A10 (sm_86, 24GB, 72 SMs) 的纯 CUDA 手写推理内核，使用原子计数器实现跨块同步的单波次 decode 内核。

## 性能

| 方案 | 吞吐量 (tok/s) | 硬件 |
|------|----------------|------|
| **本内核** | **~353** | **A10** |
| MegaQwen v4 基线 | ~199 | A10 |
| llama.cpp BF16 | ~266 | A10 |
| llama.cpp Q4_K_M | ~438 | A10 |
| vLLM (Eager) | ~158 | A10 |
| HuggingFace | ~37 | A10 |

## 架构

```
输入 token → Embed → [ ×28 层 ] → Final RMSNorm → LM Head → 输出 token
                           │
                ┌──────────┼──────────┐
                ↓          ↓          ↓
         RMSNorm → QKV → RoPE+Cache → Attention → O Proj → PostNorm → MLP → Down Proj
```

**单波次设计**: 72 个 block 精确匹配 A10 的 72 个 SM，每 SM 一个 block，无跨波次调度开销。

## 内核流水线

| 阶段 | 函数 | 参与的 block | 同步点 |
|------|------|-------------|--------|
| 1 | `embed_lookup` | 全部 72 个 | 1 个 barrier |
| 2 | `rmsnorm_step` | block 0 做计算，其余空转 | 1 个 barrier |
| 3 | `qkv_proj` | 全部 72 个（分布式行） | 1 个 barrier |
| 4 | `qk_norm_rope_cache` | block 0-15 处理 Q，0-7 处理 K+缓存 | 1 个 barrier |
| 5 | `attention` | block 0-15 做注意力，16-71 做权重预取 | 1 个 barrier |
| 6 | `o_proj_postnorm_mlp` | 全部 72 个（O 投影分布式）→ block 0 PostNorm → 全部 72 个（MLP）→ 全部 72 个（Down） | 4 个 barrier |
| 7 | `final_rmsnorm` | block 0 仅 | 无 barrier |
| 8 | LM Head (phase1+2) | 独立内核启动 | 无 |

每个层共 5 个 barrier 点。28 层 + embed + final = 143 个 barrier 调用。

## 优化技术

### 已实现的优化

1. **72 块单波次发射** — `NUM_BLOCKS=72` 精确匹配 A10 的 72 SM，无跨波次干扰
2. **原子计数器 barrier** — 使用 `atomicAdd` 实现滚动计数同步，无需 Cooperative Groups 启动约束
3. **块交错式 Embedding** — `blockIdx.x * BLOCK_SIZE + threadIdx.x` 索引，消除写竞争
4. **Release-Acquire 内存屏障** — `__threadfence()` 在 barrier 前后确保跨 block 可见性：
   - 所有线程 `__threadfence()`（release）→ thread 0 `atomicAdd` → 忙等 → `__syncthreads()` → 所有线程 `__threadfence()`（acquire）
5. **uint4/float4 向量化加载** — 所有矩阵-向量乘使用 128-bit 加载（每次加载 8 个 bf16 权重 + 8 个 float 激活）
6. **共享内存激活缓存** — MLP gate+up 从共享内存 `s_act` 读取激活值，避免重复全局加载
7. **权重预取** — 空闲 block（16-71）在注意力阶段预取下一层的 O/gate/up 权重到 L2
8. **浮点融合** — `--use_fast_math` 启用快速数学运算
9. **块级工作分配** — 按 `ceil(n / num_blocks)` 切分行范围，负载均衡

### 待实现/修复的优化

1. **cp.async MLP 双缓冲** — `__pipeline_memcpy_async` 实现的门+上投影重叠加载与计算。**当前禁用**（导致非确定性），替换为直接 `__ldg` 全局内存加载。根本原因推测是 `__pipeline_wait_prior` + `__syncthreads` 在 sm_86 上共享内存可见性问题
2. **MLP_TILE_ROWS=8 和大共享内存** — 当前限制为 48 KB 默认共享内存。使用 `cudaFuncSetAttribute` 可启用 99 KB opt-in 共享内存，恢复 `MLP_TILE_ROWS=8`
3. **向量化 Embedding 和 RMSNorm** — 当前使用标量逐元素循环。可恢复为 uint4 向量化加载
4. **张量核心注意力** — 当前使用纯 warp 级归约，未使用 Tensor Core

## 关键文件

| 文件 | 说明 |
|------|------|
| `a10_decode_kernel.cu` | 主内核源码（融合解码 + LM head） |
| `config.cuh` | 配置常量、barrier 实现、工具函数 |
| `run_benchmark.py` | PyTorch `load_inline` 编译 + 模型加载 + 基准测试 |
| `a10_decode.py` | 简化的 benchmark 启动脚本 |
| `attention.cuh` | (参考) 注意力实现 |
| `matvec.cuh` | (参考) 矩阵-向量乘实现 |
| `rmsnorm.cuh` | (参考) RMSNorm 实现 |
| `rope.cuh` | (参考) RoPE 实现 |
| `minimal_test.py` | 最小 barrier 确定性测试 |
| `minimal_test2.py` | 扩展确定性测试 |

## 运行

```bash
# 确保模型路径正确（默认 /mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B）
# 运行 benchmark
python run_benchmark.py

# 或直接使用 decode 脚本
python a10_decode.py
```

## 依赖

- CUDA 12.8+
- PyTorch 2.10.0+ (cu128)
- Python 3.12+
- NVIDIA A10 (sm_86) 或兼容 GPU

## 确定性调试记录

内核曾因跨 block 非确定性卡住多日。排查路径：

1. ❌ `MINIMAL_BARRIER_TEST` (72 blocks, embed + barrier + double) → 确定性的 → barrier 本身没坏
2. ❌ 加 Release-Acquire fence 到 barrier → 仍然非确定性
3. ❌ 改 embed_lookup 为块交错式 → 仍然非确定性
4. ✅ **去掉 cp.async MLP 双缓冲** → 变为确定性 → 根因定位为 `__pipeline_memcpy_async` 的共享内存可见性问题
5. ✅ 验证正确性：1 层输出 1878、28 层输出 21806，均匹配 HF 参考

## 注意事项

- `cuda_pipeline.h` 仍被 `config.cuh` 包含（cp.async 相关），尽管当前未使用
- `async_load_tile` 函数保留为参考，可恢复 cp.async 时使用
- `.bak` 文件是调试过程中的备份，不包含在 git 中
- 模型权重需从 HuggingFace 下载并放置在 `MODEL_PATH` 中
- 当前仅支持 bfloat16 精度权重，int8/int4 量化未实现
- 内核使用 `__launch_bounds__(256, 1)` 限制每 SM 一个 block，共享内存占用约 44 KB
