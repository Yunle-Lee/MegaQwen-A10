"""
Combined A10 + CUDAGraph analysis.
Uses the original A10Decoder, measures Python/CUDA overhead,
and estimates CUDAGraph benefit.
"""
import os, sys, time, math
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a10_decode import A10Decoder, NUM_LAYERS, MAX_SEQ_LEN


@torch.no_grad()
def benchmark():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    N_WARM = 5
    N_RUN = 50
    MAX_NEW = 100

    dec = A10Decoder()
    results = []

    # --- 1. Baseline A10 (wall-clock) ---
    print("\n[1/3] A10 Kernel — wall clock (Python + CUDA)...")
    for step in range(N_WARM):
        dec.decoder.decode_step(0)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_RUN):
        dec.reset()
        t0 = time.perf_counter()
        for step in range(MAX_NEW):
            dec.decoder.decode_step(0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    tok = MAX_NEW / avg
    wall_us = (avg / MAX_NEW) * 1e6
    results.append(("A10 Kernel (wall clock)", tok, wall_us))
    print(f"  {tok:>7.1f} tok/s, {wall_us:.0f} μs/step")

    # --- 2. Raw CUDA kernel time (CUDA events, excludes Python + item sync) ---
    print("\n[2/3] A10 Kernel — raw CUDA time (events)...")
    times_raw = []
    for _ in range(N_RUN):
        dec.reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for step in range(MAX_NEW):
            dec.decoder.decode_step(0)
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times_raw.append(elapsed_ms)

    avg_raw = sum(times_raw) / len(times_raw)
    tok_raw = MAX_NEW / (avg_raw / 1000.0)
    raw_us = (avg_raw / MAX_NEW) * 1000.0
    results.append(("A10 Kernel (CUDA events)", tok_raw, raw_us))
    print(f"  {tok_raw:>7.1f} tok/s, {raw_us:.0f} μs/step")

    # --- 3. Overhead analysis ---
    overhead_us = wall_us - raw_us
    overhead_pct = (overhead_us / wall_us) * 100
    print(f"\nPython + item<int>() overhead: {overhead_us:.1f} μs/step ({overhead_pct:.1f}%)")

    # --- 4. CUDAGraph theoretical max ---
    # CUDAGraph saves Python dispatch (~1μs) + kernel launch overhead (~5μs×3=15μs)
    graph_save = min(overhead_us, 15.0)
    graph_us = wall_us - graph_save
    graph_tok = 1e6 / graph_us
    results.append(("A10 + CUDAGraph (estimated)", graph_tok, graph_us))
    print(f"\nCUDAGraph estimated time: {graph_us:.0f} μs/step")
    print(f"CUDAGraph estimated throughput: {graph_tok:.0f} tok/s")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, tok_s, us in results:
        sp = tok_s / results[0][1]
        print(f"  {name:<35} {tok_s:>8.1f} tok/s  ({us:>4.0f} μs)  {sp:>5.2f}x")


if __name__ == "__main__":
    benchmark()
