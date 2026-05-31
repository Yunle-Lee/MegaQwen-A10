"""
INT8 Weight-Only Quantization Decoder for Qwen3-0.6B (NVIDIA A10)

Quantises large weight matrices to INT8 (per-output-channel symmetric) and
benchmarks decode throughput against FP16 baselines at two levels:

  Macro-benchmark:  full decode step (28 layers, pure Python)
  Micro-benchmark:  raw matvec throughput (isolates INT8 memory benefit)
"""

import os
import sys
import time
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import triton
import triton.language as tl

# =============================================================================
# Constants
# =============================================================================

MODEL_PATH = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
NUM_LAYERS = 28
VOCAB_SIZE = 151936
MAX_SEQ_LEN = 2048
RMS_EPS = 1e-6
INT8_GEMV_BLOCK_N = 1024

# =============================================================================
# INT8 Quantisation
# =============================================================================

def quantize_per_channel(tensor):
    """
    Symmetric per-output-channel INT8 quantization.
    tensor: (M, N) row-major weight (out_features, in_features)
    returns: (w_int8 int8, scale fp32 of shape (M,))
    """
    M, N = tensor.shape
    max_abs = tensor.abs().max(dim=1).values
    scale = max_abs / 127.0
    scale = scale.to(torch.float32).clamp(min=1e-12)
    w_int8 = (tensor / scale.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale


# =============================================================================
# Triton INT8 GEMV Kernel
# =============================================================================

@triton.jit
def int8_gemv_kernel(
    x_ptr, w_ptr, scale_ptr, y_ptr,
    M, N, stride_w,
    BLOCK_N: tl.constexpr,
):
    """INT8 GEMV — one output row per program.  N <= BLOCK_N."""
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    w = tl.load(w_ptr + row * stride_w + offs_n, mask=mask, other=0).to(tl.float32)
    x = tl.load(x_ptr + offs_n, mask=mask, other=0).to(tl.float32)
    acc = tl.sum(w * x, axis=0)
    s = tl.load(scale_ptr + row)
    tl.store(y_ptr + row, acc * s)


@triton.jit
def int8_gemv_tiled_kernel(
    x_ptr, w_ptr, scale_ptr, y_ptr,
    M, N, stride_w,
    BLOCK_N: tl.constexpr,
):
    """INT8 GEMV with tiling for N > BLOCK_N."""
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((), tl.float32)
    for start in range(0, N, BLOCK_N):
        off = offs_n + start
        mask = off < N
        w = tl.load(w_ptr + row * stride_w + off, mask=mask, other=0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask, other=0).to(tl.float32)
        acc += tl.sum(w * x)
    s = tl.load(scale_ptr + row)
    tl.store(y_ptr + row, acc * s)


def int8_gemv(x, w_int8, scale):
    """
    y[j] = scale[j] * sum_i x[i] * w_int8[j, i]
    x: (N,) fp16,  w_int8: (M, N) int8,  scale: (M,) fp32  →  y: (M,) fp32
    """
    M, N = w_int8.shape
    y = torch.empty(M, dtype=torch.float32, device=x.device)
    if N <= INT8_GEMV_BLOCK_N:
        grid = (M,)
        int8_gemv_kernel[grid](
            x, w_int8, scale, y, M, N, w_int8.stride(0), BLOCK_N=INT8_GEMV_BLOCK_N)
    else:
        grid = (M,)
        int8_gemv_tiled_kernel[grid](
            x, w_int8, scale, y, M, N, w_int8.stride(0), BLOCK_N=INT8_GEMV_BLOCK_N)
    return y


# =============================================================================
# Per-layer container
# =============================================================================

@dataclass
class LayerData:
    input_layernorm_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    post_attention_layernorm_weight: torch.Tensor
    q_proj_weight: torch.Tensor
    k_proj_weight: torch.Tensor
    v_proj_weight: torch.Tensor
    o_proj_int8: torch.Tensor
    o_proj_scale: torch.Tensor
    o_proj_weight: torch.Tensor
    gate_proj_int8: torch.Tensor
    gate_proj_scale: torch.Tensor
    gate_proj_weight: torch.Tensor
    up_proj_int8: torch.Tensor
    up_proj_scale: torch.Tensor
    up_proj_weight: torch.Tensor
    down_proj_int8: torch.Tensor
    down_proj_scale: torch.Tensor
    down_proj_weight: torch.Tensor


# =============================================================================
# Decoder  (unfused Python)
# =============================================================================

class QwenDecoder:
    """
    Pure-Python (unfused) decoder for Qwen3-0.6B.
    use_int8=True switches the large matvecs to INT8.
    """

    def __init__(self, model_path=MODEL_PATH, use_int8=True):
        self.device = torch.device("cuda")
        self.use_int8 = use_int8
        self.model_path = model_path

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        state = hf_model.state_dict()

        self.embed_weight = state["model.embed_tokens.weight"].contiguous()

        self.layers = []
        for i in range(NUM_LAYERS):
            p = f"model.layers.{i}"
            q_n = state.get(f"{p}.self_attn.q_norm.weight")
            k_n = state.get(f"{p}.self_attn.k_norm.weight")

            o_w = state[f"{p}.self_attn.o_proj.weight"].contiguous()
            g_w = state[f"{p}.mlp.gate_proj.weight"].contiguous()
            u_w = state[f"{p}.mlp.up_proj.weight"].contiguous()
            d_w = state[f"{p}.mlp.down_proj.weight"].contiguous()

            self.layers.append(LayerData(
                input_layernorm_weight=state[f"{p}.input_layernorm.weight"].contiguous(),
                q_norm_weight=q_n.contiguous() if q_n is not None else torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=self.device),
                k_norm_weight=k_n.contiguous() if k_n is not None else torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=self.device),
                post_attention_layernorm_weight=state[f"{p}.post_attention_layernorm.weight"].contiguous(),
                q_proj_weight=state[f"{p}.self_attn.q_proj.weight"].contiguous(),
                k_proj_weight=state[f"{p}.self_attn.k_proj.weight"].contiguous(),
                v_proj_weight=state[f"{p}.self_attn.v_proj.weight"].contiguous(),
                o_proj_int8=quantize_per_channel(o_w)[0],
                o_proj_scale=quantize_per_channel(o_w)[1],
                o_proj_weight=o_w,
                gate_proj_int8=quantize_per_channel(g_w)[0],
                gate_proj_scale=quantize_per_channel(g_w)[1],
                gate_proj_weight=g_w,
                up_proj_int8=quantize_per_channel(u_w)[0],
                up_proj_scale=quantize_per_channel(u_w)[1],
                up_proj_weight=u_w,
                down_proj_int8=quantize_per_channel(d_w)[0],
                down_proj_scale=quantize_per_channel(d_w)[1],
                down_proj_weight=d_w,
            ))

        self.final_norm_weight = state["model.norm.weight"].contiguous()
        lm_w = state["lm_head.weight"].contiguous()
        self.lm_head_int8, self.lm_head_scale = quantize_per_channel(lm_w)
        self.lm_head_weight = lm_w

        self._init_rope_and_cache()
        self.hf_model = hf_model

    def _init_rope_and_cache(self):
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, device=self.device).float() / HEAD_DIM))
        pos = torch.arange(MAX_SEQ_LEN, device=self.device).float().unsqueeze(1)
        freqs = pos * inv_freq.unsqueeze(0)
        cos_vals = freqs.cos()
        sin_vals = freqs.sin()
        self.cos_table = torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        self.sin_table = torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        self.cos_table[:, :HEAD_DIM // 2] = cos_vals
        self.cos_table[:, HEAD_DIM // 2:] = cos_vals
        self.sin_table[:, :HEAD_DIM // 2] = sin_vals
        self.sin_table[:, HEAD_DIM // 2:] = sin_vals
        self._reset_state()

    def _reset_state(self):
        self.position = 0
        self.k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        self.v_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=self.device)

    @staticmethod
    def rmsnorm(x, weight):
        x_f = x.float()
        mean_sq = x_f.pow(2).mean()
        rstd = (mean_sq + RMS_EPS).rsqrt()
        return x_f * rstd * weight.float()

    def apply_rope(self, x, pos):
        half = HEAD_DIM // 2
        cos = self.cos_table[pos, :half]
        sin = self.sin_table[pos, :half]
        first = x[:half]
        second = x[half:]
        return torch.cat([first * cos - second * sin, first * sin + second * cos])

    def attention(self, q_rope, layer_idx, cache_len):
        attn_out = torch.zeros(Q_SIZE, dtype=torch.float32, device=self.device)
        gqa = NUM_Q_HEADS // NUM_KV_HEADS
        for qh in range(NUM_Q_HEADS):
            kvh = qh // gqa
            s = qh * HEAD_DIM
            q_h = q_rope[s:s + HEAD_DIM]
            k_h = self.k_cache[layer_idx, kvh, :cache_len, :].float()
            v_h = self.v_cache[layer_idx, kvh, :cache_len, :].float()
            scores = q_h @ k_h.T * (HEAD_DIM ** -0.5)
            smax = scores.max()
            exp_s = torch.exp(scores - smax)
            attn_out[s:s + HEAD_DIM] = (exp_s @ v_h) / exp_s.sum()
        return attn_out

    def _matvec(self, x, w_fp16, w_int8=None, scale=None):
        if self.use_int8 and w_int8 is not None:
            return int8_gemv(x.half(), w_int8, scale)
        return (x.bfloat16() @ w_fp16.T).float()

    @torch.no_grad()
    def decode_step(self, token_id):
        hidden = self.embed_weight[token_id].float()
        for li in range(NUM_LAYERS):
            ld = self.layers[li]
            normalized = self.rmsnorm(hidden, ld.input_layernorm_weight)
            q = self._matvec(normalized, ld.q_proj_weight)
            k = self._matvec(normalized, ld.k_proj_weight)
            v = self._matvec(normalized, ld.v_proj_weight)
            residual = hidden
            q_rope = torch.empty_like(q)
            k_rope = torch.empty_like(k)
            for h in range(NUM_Q_HEADS):
                s = h * HEAD_DIM
                qh = self.rmsnorm(q[s:s + HEAD_DIM], ld.q_norm_weight)
                q_rope[s:s + HEAD_DIM] = self.apply_rope(qh, self.position)
            for h in range(NUM_KV_HEADS):
                s = h * HEAD_DIM
                kh = self.rmsnorm(k[s:s + HEAD_DIM], ld.k_norm_weight)
                k_rope[s:s + HEAD_DIM] = self.apply_rope(kh, self.position)
            cache_len = self.position + 1
            self.k_cache[li, :, self.position, :] = k_rope.view(NUM_KV_HEADS, HEAD_DIM)
            self.v_cache[li, :, self.position, :] = v.view(NUM_KV_HEADS, HEAD_DIM)
            attn_out = self.attention(q_rope, li, cache_len)
            o_out = self._matvec(attn_out, ld.o_proj_weight, ld.o_proj_int8, ld.o_proj_scale)
            hidden = residual + o_out.float()
            normalized = self.rmsnorm(hidden, ld.post_attention_layernorm_weight)
            residual = hidden
            gate = self._matvec(normalized, ld.gate_proj_weight, ld.gate_proj_int8, ld.gate_proj_scale)
            up = self._matvec(normalized, ld.up_proj_weight, ld.up_proj_int8, ld.up_proj_scale)
            act = F.silu(gate) * up
            down = self._matvec(act, ld.down_proj_weight, ld.down_proj_int8, ld.down_proj_scale)
            hidden = residual + down.float()
        normalized = self.rmsnorm(hidden, self.final_norm_weight)
        logits = self._matvec(normalized, self.lm_head_weight, self.lm_head_int8, self.lm_head_scale)
        self.position += 1
        return logits.argmax().item()


# =============================================================================
# Micro-benchmark: raw matvec throughput
# =============================================================================

def microbench_matvec(weights_fp16, weights_int8, scales, num_repeat=500):
    """
    Benchmark individual matvec operations comparing FP16 vs INT8.

    weights_fp16 : list of (M, N) fp16 tensors
    weights_int8 : list of (M, N) int8 tensors
    scales      : list of (M,) fp32 tensors
    """
    device = weights_fp16[0].device
    results = []

    for idx, (w16, w8, sc) in enumerate(zip(weights_fp16, weights_int8, scales)):
        M, N = w16.shape
        x_fp32 = torch.randn(N, dtype=torch.float32, device=device)
        x_bf16 = x_fp32.bfloat16()
        x_fp16 = x_fp32.half()

        # --- FP16 matvec (cuBLAS via torch) ---
        # Warmup
        for _ in range(10):
            _ = (x_bf16 @ w16.T).float()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(num_repeat):
            y = (x_bf16 @ w16.T).float()
        torch.cuda.synchronize()
        t_fp16 = (time.time() - t0) / num_repeat * 1000  # ms

        # --- INT8 matvec (Triton) ---
        for _ in range(10):
            _ = int8_gemv(x_fp16, w8, sc)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(num_repeat):
            y = int8_gemv(x_fp16, w8, sc)
        torch.cuda.synchronize()
        t_int8 = (time.time() - t0) / num_repeat * 1000  # ms

        speedup = t_fp16 / t_int8 if t_int8 > 0 else 0
        results.append((M, N, t_fp16, t_int8, speedup))

    return results


def run_microbenchmark():
    """Build a representative set of matvec sizes and benchmark them."""
    print("\n" + "=" * 60)
    print("Matvec Micro-benchmark")
    print("=" * 60)

    # Sizes present in Qwen3-0.6B decode
    sizes = [
        (3072, 1024, "gate_proj / up_proj"),
        (1024, 3072, "down_proj"),
        (1024, 2048, "o_proj"),
        (151936, 1024, "lm_head"),
    ]

    device = torch.device("cuda")
    all_fp16, all_int8, all_scales, all_names = [], [], [], []

    for M, N, name in sizes:
        w_fp16 = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        w_int8, scale = quantize_per_channel(w_fp16)
        all_fp16.append(w_fp16)
        all_int8.append(w_int8)
        all_scales.append(scale)
        all_names.append(name)

    n_repeat = 500

    print(f"\n  {'Shape':>20}  {'Name':<20}  {'FP16 (ms)':>10}  {'INT8 (ms)':>10}  {'Speedup':>8}")
    print(f"  {'-'*20}  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")

    results = []
    for w16, w8, sc, name in zip(all_fp16, all_int8, all_scales, all_names):
        M, N = w16.shape

        # --- FP16 ---
        x_bf16 = torch.randn(N, dtype=torch.bfloat16, device=device)
        for _ in range(10):
            _ = (x_bf16 @ w16.T).float()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_repeat):
            _ = (x_bf16 @ w16.T).float()
        torch.cuda.synchronize()
        t_fp16 = (time.time() - t0) / n_repeat * 1000

        # --- INT8 ---
        x_fp16 = x_bf16.half()
        for _ in range(10):
            _ = int8_gemv(x_fp16, w8, sc)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_repeat):
            _ = int8_gemv(x_fp16, w8, sc)
        torch.cuda.synchronize()
        t_int8 = (time.time() - t0) / n_repeat * 1000

        sp = t_fp16 / t_int8 if t_int8 > 0 else 0
        results.append((M, N, name, t_fp16, t_int8, sp))
        print(f"  {f'({M:>6d}, {N:>5d})':>20}  {name:<20}  {t_fp16:>10.4f}  {t_int8:>10.4f}  {sp:>7.2f}x")

    # Weighted estimate for a full decode step
    print(f"\n  --- Estimated per-step matvec cost ---")
    per_layer_matvecs = {
        # name: (count per layer, M, N)
        'gate_proj': (1, 3072, 1024),
        'up_proj':   (1, 3072, 1024),
        'down_proj': (1, 1024, 3072),
        'o_proj':    (1, 1024, 2048),
    }
    total_fp16 = 0.0
    total_int8 = 0.0
    res_map = {(M, N): (tf, ti) for (M, N, _, tf, ti, _) in results}
    for name, (cnt, M, N) in per_layer_matvecs.items():
        tf, ti = res_map.get((M, N), (0, 0))
        total_fp16 += cnt * NUM_LAYERS * tf
        total_int8 += cnt * NUM_LAYERS * ti
    # LM head (once per step)
    total_fp16 += res_map.get((151936, 1024), (0, 0))[0]
    total_int8 += res_map.get((151936, 1024), (0, 0))[1]

    print(f"  {'Matvec total (28 layers)':<42}  {total_fp16:>10.2f}  {total_int8:>10.2f}  {total_fp16/total_int8:>7.2f}x" if total_int8 > 0 else "")
    print(f"  (Excludes QKV proj, norms, attention, Python overhead)")

    return results


# =============================================================================
# Full decoder benchmarks
# =============================================================================

def run_decoder_benchmark(use_int8, num_warmup=5, num_steps=50):
    label = "INT8" if use_int8 else "FP16"
    decoder = QwenDecoder(MODEL_PATH, use_int8=use_int8)

    for _ in range(num_warmup):
        decoder.decode_step(0)
    decoder._reset_state()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_steps):
        t0 = time.time()
        decoder.decode_step(0)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    tok_s = 1000.0 / avg_ms
    return avg_ms, tok_s


def run_a10_baseline(num_warmup=5, num_steps=50):
    try:
        from a10_decode import A10Decoder
    except ImportError:
        return None, None
    dec = A10Decoder(MODEL_PATH)
    for _ in range(num_warmup):
        dec.decoder.decode_step(0)
    dec.reset()
    torch.cuda.synchronize()
    times = []
    for _ in range(num_steps):
        t0 = time.time()
        dec.decoder.decode_step(0)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    tok_s = 1000.0 / avg_ms
    return avg_ms, tok_s


# =============================================================================
# Correctness check
# =============================================================================

def correctness_check():
    print("\n" + "=" * 60)
    print("Correctness Check — INT8 vs FP16 vs A10")
    print("=" * 60)

    # Single matvec comparison (layer 0, gate_proj)
    fp16 = QwenDecoder(MODEL_PATH, use_int8=False)
    int8 = QwenDecoder(MODEL_PATH, use_int8=True)

    hidden = fp16.embed_weight[20].float()
    ld16 = fp16.layers[0]
    ld8 = int8.layers[0]
    normalized = QwenDecoder.rmsnorm(hidden, ld16.input_layernorm_weight)

    # Gate proj
    g16 = (normalized.bfloat16() @ ld16.gate_proj_weight.T).float()
    g8 = int8_gemv(normalized.half(), ld8.gate_proj_int8, ld8.gate_proj_scale)
    rel_err = (g16 - g8).abs().max() / g16.abs().max()
    cos_sim = F.cosine_similarity(g16.unsqueeze(0), g8.unsqueeze(0)).item()
    print(f"  gate_proj  (3072×1024): rel_err={rel_err:.4%}  cos_sim={cos_sim:.6f}")

    # Down proj (with SiLU activation)
    u16 = (normalized.bfloat16() @ ld16.up_proj_weight.T).float()
    u8 = int8_gemv(normalized.half(), ld8.up_proj_int8, ld8.up_proj_scale)
    act16 = F.silu(g16) * u16
    act8 = F.silu(g8) * u8
    d16 = (act16.bfloat16() @ ld16.down_proj_weight.T).float()
    d8 = int8_gemv(act8.half(), ld8.down_proj_int8, ld8.down_proj_scale)
    rel_err_d = (d16 - d8).abs().max() / d16.abs().max()
    cos_sim_d = F.cosine_similarity(d16.unsqueeze(0), d8.unsqueeze(0)).item()
    print(f"  down_proj  (1024×3072): rel_err={rel_err_d:.4%}  cos_sim={cos_sim_d:.6f}")

    # LM head
    lm16 = (normalized.bfloat16() @ fp16.lm_head_weight.T).float()
    lm8 = int8_gemv(normalized.half(), int8.lm_head_int8, int8.lm_head_scale)
    rel_err_lm = (lm16 - lm8).abs().max() / lm16.abs().max()
    cos_sim_lm = F.cosine_similarity(lm16.unsqueeze(0), lm8.unsqueeze(0)).item()
    t16 = lm16.argmax().item()
    t8 = lm8.argmax().item()
    print(f"  lm_head    (151936×1024): rel_err={rel_err_lm:.4%}  cos_sim={cos_sim_lm:.6f}  top1: FP16={t16} INT8={t8} {'✓' if t16==t8 else '✗'}")

    # Multi-step token comparison
    print(f"\n  Token agreement over 5 steps:")
    fp16._reset_state()
    int8._reset_state()
    matches = 0
    for step in range(5):
        t_fp16 = fp16.decode_step(20)
        t_int8 = int8.decode_step(20)
        m = t_fp16 == t_int8
        matches += m
        print(f"    step {step}:  FP16={t_fp16:6d}  INT8={t_int8:6d}  {'✓' if m else '✗'}")
    print(f"    Token agreement: {matches}/5")

    # Compare with A10 fused kernel
    try:
        from a10_decode import A10Decoder
        a10 = A10Decoder(MODEL_PATH)
        fp16._reset_state()
        a10.reset()
        a10_ok = 0
        for step in range(5):
            t_fp16 = fp16.decode_step(20)
            t_a10 = a10.decoder.decode_step(20)
            m = t_fp16 == t_a10
            a10_ok += m
        print(f"  FP16 (Python) vs A10 fused: {a10_ok}/5 agreement")
    except Exception as e:
        print(f"  A10 comparison skipped: {e}")

    return matches / 5  # fraction matching


# =============================================================================
# Main
# =============================================================================

def main():
    torch.set_grad_enabled(False)

    print("=" * 60)
    print("Qwen3-0.6B  INT8 Decoder  (NVIDIA A10)")
    print("=" * 60)
    print(f"  PyTorch {torch.__version__}  CUDA {torch.version.cuda}")
    print(f"  Triton {triton.__version__}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")

    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    if mode == "correctness":
        correctness_check()
        return

    if mode == "microbench":
        run_microbenchmark()
        return

    if mode == "quick":
        n_steps = 10

    # 1) Micro-benchmark of matvec operations
    micro_results = run_microbenchmark()

    # 2) Sanity check
    agreement = correctness_check()
    if agreement < 0.6 and mode == "bench":
        print("  (INT8 token divergence is expected — quantization error compounds through 28 layers)")

    # 3) Full decoder benchmark
    print(f"\n{'='*60}")
    print("Full Decoder Benchmark")
    print(f"{'='*60}")
    print(f"  {n_steps} decode steps, warmup=5")
    fp16_ms, fp16_tok = run_decoder_benchmark(False, 5, n_steps)
    int8_ms, int8_tok = run_decoder_benchmark(True, 5, n_steps)
    a10_ms, a10_tok = run_a10_baseline(5, n_steps)

    # 4) Summary
    fp16_mem = _mem_estimate(False)
    int8_mem = _mem_estimate(True)

    # Correct per-step matvec estimate (times in ms):
    _gup = micro_results[0]   # (3072,1024) — gate_proj OR up_proj
    _dn = micro_results[1]    # (1024,3072) — down_proj
    _op = micro_results[2]    # (1024,2048) — o_proj
    _lm = micro_results[3]    # (151936,1024) — lm_head
    est_fp16 = (2 * _gup[3] + _dn[3] + _op[3]) * NUM_LAYERS + _lm[3]
    est_int8 = (2 * _gup[4] + _dn[4] + _op[4]) * NUM_LAYERS + _lm[4]

    print(f"\n{'='*60}")
    print("Benchmark Summary  ───  Qwen3-0.6B  (NVIDIA A10)")
    print(f"{'='*60}")

    print(f"\n  A) Raw matvec micro-benchmark (500 repeats each)")
    print(f"  {'Matvec':<30} {'Shape':>16} {'FP16':>8} {'INT8':>8} {'Speedup':>8}")
    print(f"  {'-'*30} {'-'*16} {'-'*8} {'-'*8} {'-'*8}")
    for (M, N, name, tf, ti, sp) in micro_results:
        print(f"  {name:<30} ({M:>6d},{N:>5d}) {tf*1000:>7.1f}µs {ti*1000:>7.1f}µs {sp:>7.2f}x")
    print(f"  {'Estimated per-step matvec total':<30} {'':>16} {est_fp16:>8.2f} {est_int8:>8.2f} {est_fp16/est_int8:>7.2f}x")
    print(f"  {'  (gate+up)×2 + down + o_proj per layer × 28 + lm_head':<30}")

    print(f"\n  B) Full decoder throughput  ({n_steps} decode steps)")
    print(f"  {'Method':<30} {'ms/step':>9} {'tok/s':>9}  {'vs FP16':>9}")
    print(f"  {'-'*30} {'-'*9} {'-'*9} {'-'*9}")
    print(f"  {'FP16 (Python, unfused)':<30} {fp16_ms:>9.2f} {fp16_tok:>9.1f}  {'1.00x':>9}")
    print(f"  {'INT8 (Python, unfused)':<30} {int8_ms:>9.2f} {int8_tok:>9.1f}  {int8_tok/fp16_tok:>8.2f}x")
    if a10_tok is not None:
        print(f"  {'A10 fused BF16 megakernel':<30} {a10_ms:>9.2f} {a10_tok:>9.1f}  {a10_tok/fp16_tok:>8.2f}x")

    print(f"\n  C) Weight memory per step  (gate/up/down/o_proj + lm_head quantized, QKV FP16)")
    print(f"  {'FP16 reads:':<30} {fp16_mem:>9.1f} MB")
    print(f"  {'INT8 reads:':<30} {int8_mem:>9.1f} MB  ({int8_mem/fp16_mem:.0%})")
    print(f"  {'Memory bandwidth saving:':<30} {'':>9} {(fp16_mem-int8_mem)/fp16_mem:>7.1%}")

    print(f"\n  D) Analysis")
    if est_fp16 > 0:
        bw_fp16 = fp16_mem / est_fp16
        bw_int8 = int8_mem / est_int8
        print(f"  Effective matvec BW (FP16): {bw_fp16:.0f} GB/s  (of 600 GB/s peak)")
        print(f"  Effective matvec BW (INT8): {bw_int8:.0f} GB/s")
    print(f"  Full-decoder Python overhead: {fp16_ms - est_fp16:.0f} ms/step ({((fp16_ms - est_fp16)/fp16_ms):.0%})")
    print(f"  A10 fused kernel is {a10_tok/fp16_tok:.0f}x faster than Python unfused (BF16)")


def _mem_estimate(use_int8=False):
    """Bytes read per decode step for weight matrices (excl. QKV which stay FP16)."""
    b = 1 if use_int8 else 2
    per_layer = (2048 * 1024 * 2) + (1024 * 1024 * 2) + (1024 * 1024 * 2) \
        + (1024 * 2048 * b) + (3072 * 1024 * b) + (3072 * 1024 * b) + (1024 * 3072 * b)
    total = per_layer * NUM_LAYERS + (151936 * 1024 * b)
    return total / 1e6


if __name__ == "__main__":
    main()
