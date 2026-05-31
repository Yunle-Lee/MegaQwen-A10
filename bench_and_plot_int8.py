"""Benchmark INT8+BF16 at 10 positions, merge with existing data, plot."""
import gc, json, os, sys, time, math
import torch
import warnings
warnings.filterwarnings("ignore")

PROJECT = os.path.dirname(os.path.abspath(__file__))
MEGA_PROJECT = "/mnt/workspace/DSW-GPU/MegaQwen"
MODEL_PATH = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
sys.path.insert(0, PROJECT)

TEST_POSITIONS = [1, 10, 25, 50, 75, 100, 128, 150, 175, 200]
WARMUP = 2
RUNS = 3

def clear():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def measure(fn):
    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)

# =============================================================================
# 1. Benchmark INT8
# =============================================================================

def benchmark_int8():
    from bench_int8 import compile_int8_kernel, make_rope, load_quantized_state
    print("=== INT8 Position Benchmark ===")
    device = torch.device("cuda")
    cos_t, sin_t = make_rope(2048, 128, device)
    hf, qstate = load_quantized_state(device)

    mod = compile_int8_kernel()

    layer_tensors = []
    for i in range(28):
        p = f"model.layers.{i}"
        layer_tensors.append(qstate[f"{p}.input_layernorm.weight"].contiguous())
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            layer_tensors.append(qstate[f"{p}.{proj}.weight"][0].contiguous())
        layer_tensors.append(qstate.get(f"{p}.self_attn.q_norm.weight",
            torch.ones(128, dtype=torch.bfloat16, device=device)).contiguous())
        layer_tensors.append(qstate.get(f"{p}.self_attn.k_norm.weight",
            torch.ones(128, dtype=torch.bfloat16, device=device)).contiguous())
        layer_tensors.append(qstate[f"{p}.self_attn.o_proj.weight"][0].contiguous())
        layer_tensors.append(qstate[f"{p}.post_attention_layernorm.weight"].contiguous())
        for proj in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            layer_tensors.append(qstate[f"{p}.{proj}.weight"][0].contiguous())

    layer_scales = []
    for i in range(28):
        p = f"model.layers.{i}"
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            layer_scales.append(qstate[f"{p}.{proj}.weight"][1].contiguous())

    lm_head = qstate["lm_head.weight"]
    final_norm = qstate["model.norm.weight"].contiguous()
    embed = qstate["model.embed_tokens.weight"].contiguous()

    print("Creating INT8 decoder...")
    decoder = mod.A10Int8MegakernelDecoder(
        embed, layer_tensors, layer_scales,
        lm_head[0].contiguous(), lm_head[1].contiguous(), final_norm,
        cos_t, sin_t, 28, 2048
    )

    results = {}
    for n in TEST_POSITIONS:
        def gen_fn(n=n):
            decoder.reset()
            for _ in range(n):
                decoder.decode_step(0)
        t = measure(gen_fn)
        results[n] = {"tok_s": n / t, "ms_tok": t * 1000 / n, "time_s": t}
        print(f"  INT8  pos {n:4d}: {n/t:8.1f} tok/s, {t*1000/n:.2f} ms/tok")
    del decoder; clear()
    return results

# =============================================================================
# 2. Benchmark BF16
# =============================================================================

def benchmark_bf16():
    from a10_decode import A10Decoder
    print("\n=== BF16 Position Benchmark ===")
    decoder = A10Decoder()
    results = {}
    for n in TEST_POSITIONS:
        def gen_fn(n=n):
            decoder.reset()
            for _ in range(n):
                decoder.decoder.decode_step(0)
        t = measure(gen_fn)
        results[n] = {"tok_s": n / t, "ms_tok": t * 1000 / n, "time_s": t}
        print(f"  BF16  pos {n:4d}: {n/t:8.1f} tok/s, {t*1000/n:.2f} ms/tok")
    del decoder; clear()
    return results

# =============================================================================
# 3. Merge + Save
# =============================================================================

def merge_and_save(int8_results, bf16_results):
    merged = {}
    mega_results_path = os.path.join(MEGA_PROJECT, "benchmark_results.json")
    if os.path.exists(mega_results_path):
        with open(mega_results_path) as f:
            mega = json.load(f)
        merged = mega["results"]
    else:
        merged = {}

    merged["A10 BF16"] = {str(k): v for k, v in bf16_results.items()}
    merged["A10 INT8"] = {str(k): v for k, v in int8_results.items()}
    merged["A10-Optimized"] = merged["A10 BF16"]  # alias

    combined_out = os.path.join(PROJECT, "combined_benchmark_results.json")
    with open(combined_out, "w") as f:
        json.dump({"test_positions": TEST_POSITIONS, "results": merged}, f, indent=2)
    print(f"\nCombined results saved: {combined_out}")
    return merged

# =============================================================================
# 4. Plot
# =============================================================================

def plot(merged):
    data = {"test_positions": TEST_POSITIONS, "results": merged}
    positions = TEST_POSITIONS
    out_dir = PROJECT

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ============== Color / style definitions ==============
    colors = {
        "A10 INT8":           "#e74c3c",
        "A10 BF16":           "#2ecc71",
        "A10-Optimized":      "#2ecc71",
        "Megakernel":         "#f39c12",
        "vLLM":               "#9b59b6",
        "HuggingFace":        "#95a5a6",
        "A10+CUDAGraph":      "#c0392b",
        "llama.cpp (BF16)":   "#3498db",
        "CUDAGraph Plugin":   "#16a085",
    }

    line_styles = {
        "A10 INT8":           ("o-", 3.5),
        "A10 BF16":           ("s-", 3),
        "A10-Optimized":      ("s-", 3),
        "Megakernel":         ("s--", 2.5),
        "vLLM":               ("x--", 2.5),
        "HuggingFace":        ("v:", 2.5),
        "A10+CUDAGraph":      ("o--", 2),
        "llama.cpp (BF16)":   ("^-.", 2.5),
        "CUDAGraph Plugin":   ("d--", 2.5),
    }

    llama_bf16 = {1:249.3,10:252.2,25:252.3,50:249.7,75:250.1,100:247.6,128:249.4,150:248.4,175:246.6,200:247.8}
    cudagraph_plugin = {p:154.3 for p in positions}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Qwen3-0.6B Inference on NVIDIA A10 — INT8 vs BF16",
                 fontsize=16, fontweight="bold", y=1.01)

    def plot_fw(ax, x, y, label):
        marker, lw = line_styles.get(label, ("o-", 2))
        c = colors.get(label, "#888")
        ax.plot(x, y, marker, label=label, color=c, linewidth=lw, markersize=7)

    # data frameworks present in merged results
    present_fws = list(data["results"].keys())

    # ===== (a) Throughput vs Position =====
    ax = axes[0, 0]
    for fw in present_fws:
        vals = [data["results"][fw][str(p)]["tok_s"] for p in positions]
        plot_fw(ax, positions, vals, fw)
    plot_fw(ax, positions, [llama_bf16[p] for p in positions], "llama.cpp (BF16)")
    plot_fw(ax, positions, [cudagraph_plugin[p] for p in positions], "CUDAGraph Plugin")
    ax.set_xlabel("Decode token position", fontsize=11)
    ax.set_ylabel("Throughput (tok/s)", fontsize=11)
    ax.set_title("(a) Throughput vs Position", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(positions) + 10)

    # ===== (b) Latency vs Position =====
    ax = axes[0, 1]
    for fw in present_fws:
        vals = [data["results"][fw][str(p)]["ms_tok"] for p in positions]
        plot_fw(ax, positions, vals, fw)
    plot_fw(ax, positions, [1000.0/llama_bf16[p] for p in positions], "llama.cpp (BF16)")
    plot_fw(ax, positions, [1000.0/cudagraph_plugin[p] for p in positions], "CUDAGraph Plugin")
    ax.set_xlabel("Decode token position", fontsize=11)
    ax.set_ylabel("Latency (ms/tok)", fontsize=11)
    ax.set_title("(b) Latency vs Position", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(positions) + 10)

    # ===== (c) Average Throughput bar =====
    ax = axes[1, 0]
    all_avg = {}
    for fw in present_fws:
        vals = [data["results"][fw][str(p)]["tok_s"] for p in positions]
        all_avg[fw] = sum(vals) / len(vals)
    all_avg["llama.cpp (BF16)"] = sum(llama_bf16[p] for p in positions) / len(positions)
    all_avg["CUDAGraph Plugin"] = sum(cudagraph_plugin[p] for p in positions) / len(positions)
    sorted_fws = sorted(all_avg.keys(), key=lambda x: all_avg[x], reverse=True)
    ba_vals = [all_avg[f] for f in sorted_fws]
    ba_colors = [colors.get(f, "#888") for f in sorted_fws]
    bars = ax.bar(range(len(sorted_fws)), ba_vals, color=ba_colors, width=0.6)
    for bar, val in zip(bars, ba_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 6,
                f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(sorted_fws)))
    ax.set_xticklabels([f"{f}" for f in sorted_fws], fontsize=8, rotation=12)
    ax.set_ylabel("Average tok/s", fontsize=11)
    ax.set_title("(c) Average Throughput (pos 1–200)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # ===== (d) Speedup vs HuggingFace at pos 100 =====
    ax = axes[1, 1]
    stable_pos = 100
    hf_val = data["results"]["HuggingFace"][str(stable_pos)]["tok_s"]
    speedups = {}
    for fw in present_fws:
        if fw != "HuggingFace":
            speedups[fw] = data["results"][fw][str(stable_pos)]["tok_s"] / hf_val
    speedups["llama.cpp (BF16)"] = llama_bf16[stable_pos] / hf_val
    speedups["CUDAGraph Plugin"] = cudagraph_plugin[stable_pos] / hf_val
    sorted_sp = sorted(speedups.keys(), key=lambda x: speedups[x], reverse=True)
    sp_vals = [speedups[f] for f in sorted_sp]
    sp_colors = [colors.get(f, "#888") for f in sorted_sp]
    bars = ax.bar(range(len(sorted_sp)), sp_vals, color=sp_colors, width=0.6)
    for bar, val in zip(bars, sp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}x", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(sorted_sp)))
    ax.set_xticklabels([f"{f}" for f in sorted_sp], fontsize=8, rotation=12)
    ax.set_ylabel("Speedup vs HuggingFace", fontsize=11)
    ax.set_title(f"(d) Speedup vs HF at position {stable_pos}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    for fmt in ["png", "svg"]:
        out_path = os.path.join(out_dir, f"int8_benchmarks.{fmt}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    print("Plot done.")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    int8_results = benchmark_int8()
    bf16_results = benchmark_bf16()
    merged = merge_and_save(int8_results, bf16_results)
    plot(merged)
