"""Plot all benchmark results: 4 charts in a square layout."""
import json, os

PROJECT = "/mnt/workspace/DSW-GPU/MegaQwen-A10"
OUT_DIR = "/mnt/workspace/DSW-GPU/MegaQwen-A10"

with open(os.path.join(PROJECT, "combined_benchmark_results.json")) as f:
    data = json.load(f)

positions = data["test_positions"]
frameworks = list(data["results"].keys())

llama_bf16 = {1: 249.30, 10: 252.17, 25: 252.34, 50: 249.69, 75: 250.11, 100: 247.60, 128: 249.38, 150: 248.39, 175: 246.57, 200: 247.76}
cudagraph_plugin = {1: 154.3, 10: 154.3, 25: 154.3, 50: 154.3, 75: 154.3, 100: 154.3, 128: 154.3, 150: 154.3, 175: 154.3, 200: 154.3}
# A10+CUDAGraph = A10-Optimized (overhead <0.5%, no measurable improvement)

colors = {
    "A10-Optimized": "#e74c3c",
    "A10+CUDAGraph": "#c0392b",
    "Megakernel": "#f39c12",
    "llama.cpp (BF16)": "#3498db",
    "CUDAGraph Plugin": "#16a085",
    "vLLM": "#9b59b6",
    "HuggingFace": "#95a5a6",
}

line_styles = {
    "A10-Optimized": ("o-", 3),
    "A10+CUDAGraph": ("o--", 2),
    "Megakernel": ("s--", 2.5),
    "llama.cpp (BF16)": ("^-.", 2.5),
    "CUDAGraph Plugin": ("d--", 2.5),
    "vLLM": ("x--", 2.5),
    "HuggingFace": ("v:", 2.5),
}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle("Qwen3-0.6B Inference Benchmark on NVIDIA A10 (24GB)",
             fontsize=16, fontweight="bold", y=1.01)

def plot_framework(ax, x, y, label):
    marker, lw = line_styles.get(label, ("o-", 2))
    ax.plot(x, y, marker, label=label, color=colors.get(label, "#888"),
            linewidth=lw, markersize=6)

# --- Chart 1: Throughput vs Position ---
ax = axes[0, 0]
for fw in frameworks:
    vals = [data["results"][fw][str(p)]["tok_s"] for p in positions]
    plot_framework(ax, positions, vals, fw)
vals = [llama_bf16[p] for p in positions]
plot_framework(ax, positions, vals, "llama.cpp (BF16)")
vals = [cudagraph_plugin[p] for p in positions]
plot_framework(ax, positions, vals, "CUDAGraph Plugin")
# A10+CUDAGraph line — identical to A10-Optimized (overhead < 0.5%)
a10cg_vals = [data["results"]["A10+CUDAGraph"][str(p)]["tok_s"] for p in positions]
plot_framework(ax, positions, a10cg_vals, "A10+CUDAGraph")
ax.set_xlabel("Decode token position", fontsize=11)
ax.set_ylabel("Throughput (tok/s)", fontsize=11)
ax.set_title("(a) Throughput vs Position", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(positions) + 10)
# Green arrow pointing to A10-Optimized
ax.annotate("KiLee", xy=(100, 336.8), xytext=(140, 370),
            arrowprops=dict(arrowstyle="->", color="green", lw=2.5),
            fontsize=12, fontweight="bold", color="green")

# --- Chart 2: Latency vs Position ---
ax = axes[0, 1]
for fw in frameworks:
    vals = [data["results"][fw][str(p)]["ms_tok"] for p in positions]
    plot_framework(ax, positions, vals, fw)
vals = [1000.0 / llama_bf16[p] for p in positions]
plot_framework(ax, positions, vals, "llama.cpp (BF16)")
vals = [1000.0 / cudagraph_plugin[p] for p in positions]
plot_framework(ax, positions, vals, "CUDAGraph Plugin")
a10cg_vals = [data["results"]["A10+CUDAGraph"][str(p)]["ms_tok"] for p in positions]
plot_framework(ax, positions, a10cg_vals, "A10+CUDAGraph")
ax.set_xlabel("Decode token position", fontsize=11)
ax.set_ylabel("Latency (ms/tok)", fontsize=11)
ax.set_title("(b) Latency vs Position", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(positions) + 10)
ax.annotate("KiLee", xy=(100, 2.97), xytext=(130, 3.8),
            arrowprops=dict(arrowstyle="->", color="green", lw=2.5),
            fontsize=12, fontweight="bold", color="green")

# --- Chart 3: Average Throughput bar chart ---
ax = axes[1, 0]
all_data = {}
for fw in frameworks:
    vals = [data["results"][fw][str(p)]["tok_s"] for p in positions]
    all_data[fw] = sum(vals) / len(vals)
all_data["llama.cpp (BF16)"] = sum(llama_bf16[p] for p in positions) / len(positions)
all_data["CUDAGraph Plugin"] = sum(cudagraph_plugin[p] for p in positions) / len(positions)
a10cg_vals = [data["results"]["A10+CUDAGraph"][str(p)]["tok_s"] for p in positions]
all_data["A10+CUDAGraph"] = sum(a10cg_vals) / len(a10cg_vals)
sorted_fws = sorted(all_data.keys(), key=lambda x: all_data[x], reverse=True)
ba_vals = [all_data[f] for f in sorted_fws]
ba_colors = [colors.get(f, "#888") for f in sorted_fws]
bars = ax.bar(range(len(sorted_fws)), ba_vals, color=ba_colors, width=0.55)
for bar, val in zip(bars, ba_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(sorted_fws)))
ax.set_xticklabels([f"{f}" for f in sorted_fws], fontsize=8.5, rotation=15)
ax.set_ylabel("Average tok/s", fontsize=11)
ax.set_title("(c) Average Throughput (pos 1-200)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
# Find A10 bar and add annotation
for i, fw in enumerate(sorted_fws):
    if "A10" in fw:
        a10_idx = i
        break
ax.annotate("KiLee", xy=(a10_idx, ba_vals[a10_idx]), xytext=(a10_idx + 0.5, ba_vals[a10_idx] + 30),
            arrowprops=dict(arrowstyle="->", color="green", lw=2.5),
            fontsize=12, fontweight="bold", color="green")

# --- Chart 4: Speedup vs HuggingFace bar chart ---
ax = axes[1, 1]
hf_base = "HuggingFace"
stable_pos = 100
hf_val = data["results"][hf_base][str(stable_pos)]["tok_s"]
speedups = {}
for fw in frameworks:
    if fw != hf_base:
        speedups[fw] = data["results"][fw][str(stable_pos)]["tok_s"] / hf_val
speedups["llama.cpp (BF16)"] = llama_bf16[stable_pos] / hf_val
speedups["CUDAGraph Plugin"] = cudagraph_plugin[stable_pos] / hf_val
speedups["A10+CUDAGraph"] = data["results"]["A10+CUDAGraph"][str(stable_pos)]["tok_s"] / hf_val
sorted_sp = sorted(speedups.keys(), key=lambda x: speedups[x], reverse=True)
sp_vals = [speedups[f] for f in sorted_sp]
sp_colors = [colors.get(f, "#888") for f in sorted_sp]
bars = ax.bar(range(len(sorted_sp)), sp_vals, color=sp_colors, width=0.55)
for bar, val in zip(bars, sp_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f"{val:.1f}x", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(sorted_sp)))
ax.set_xticklabels([f"{f}" for f in sorted_sp], fontsize=8.5, rotation=15)
ax.set_ylabel("Speedup vs HuggingFace", fontsize=11)
ax.set_title(f"(d) Speedup over HF at position {stable_pos}", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
for i, fw in enumerate(sorted_sp):
    if "A10" in fw:
        a10_idx = i
        break
ax.annotate("KiLee", xy=(a10_idx, sp_vals[a10_idx]), xytext=(a10_idx + 0.5, sp_vals[a10_idx] + 1.5),
            arrowprops=dict(arrowstyle="->", color="green", lw=2.5),
            fontsize=12, fontweight="bold", color="green")

plt.tight_layout()

out_png = os.path.join(OUT_DIR, "all_benchmarks_square.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Chart saved: {out_png}")

out_svg = os.path.join(OUT_DIR, "all_benchmarks_square.svg")
plt.savefig(out_svg, bbox_inches="tight")
print(f"Chart saved: {out_svg}")
