# INT8 Decoder — Key Decisions & Findings

## Files
- `int8_decoder.py` — INT8 weight-only quantization decoder + benchmarks
- `a10_decode.py` / `a10_decode_kernel.cu` — existing A10 fused BF16 megakernel (reference)

## Approach
- **Triton INT8 GEMV kernels** (not `torch._int_mm`): `_int_mm` requires batch dim ≥ 16, but decode is batch=1 (GEMV). Two Triton kernels handle all weight shapes (up to 151K rows).
- **Per-output-channel symmetric quantization**: scale[j] = absmax(W[j,:]) / 127, applied to gate/up/down/o_proj + lm_head. QKV kept FP16 (small matrices, no benefit).
- **Standard split-half RoPE** matches HF transformers, verified 5/5 token agreement with the A10 fused kernel.

## Correctness
- FP16 Python decoder matches A10 fused kernel: 5/5 token agreement.
- INT8 per-matvec error: ~0.8–1.5% relative error, cos_sim > 0.99986 (high-fidelity).
- Token divergence after 28 layers: expected — 0.85% error compounds through 54 matvecs per step.

## Performance (NVIDIA A10)

### Raw matvec micro-benchmark (500 repeats)
| Matvec | Shape | FP16 | INT8 | Speedup |
|--------|-------|------|------|---------|
| gate_proj / up_proj | 3072×1024 | 24.7 µs | 19.6 µs | 1.26× |
| down_proj | 1024×3072 | 23.1 µs | 18.8 µs | 1.23× |
| o_proj | 1024×2048 | 23.6 µs | 18.6 µs | 1.27× |
| lm_head | 151936×1024 | 655 µs | 349 µs | **1.88×** |
| **Total per step (28 layers)** | | **3.35 ms** | **2.49 ms** | **1.34×** |

### Full decoder throughput (50 decode steps)
| Method | ms/step | tok/s | vs FP16 |
|--------|---------|-------|---------|
| FP16 (Python, unfused) | 178 | 5.6 | 1.00× |
| INT8 (Python, unfused) | 180 | 5.6 | 0.99× |
| A10 fused BF16 megakernel | 2.88 | 347 | **62×** |

## Key Finding
INT8 provides real matvec speedup (1.23–1.88×), but in an **unfused Python decoder it's invisible** — Python overhead (175 ms/step, 98% of runtime) swamps the 0.86 ms matvec savings. The benefit would only materialize inside a fused kernel like the A10 decoder, where matvecs dominate. The 40% memory bandwidth reduction (1192→713 MB per step) is the more impactful win for fused kernels.

## Running
```bash
python3 int8_decoder.py microbench   # raw matvec benchmarks only
python3 int8_decoder.py bench 50     # full benchmark (takes ~2 min)
python3 int8_decoder.py bench 10     # quick check
python3 int8_decoder.py correctness   # INT8 vs FP16 token agreement
```
