"""
Inference acceleration plugin for Qwen3-0.6B on A10.

Uses CUDAGraph replay to eliminate kernel launch overhead in decode phase.
Each decode step is captured as a CUDA graph, then replayed with new inputs.
"""
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class CUDAGraphDecoder:
    """
    Wraps a HuggingFace model with CUDAGraph-accelerated decode.
    
    The first token (prefill) runs normally. Subsequent decode steps
    replay a captured CUDA graph, eliminating Python overhead and
    kernel launch latency.
    """
    def __init__(self, model, max_seq_len=2048):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.max_seq_len = max_seq_len
        self.graph = None
        self.graph_input_ids = None
        self.graph_position_ids = None
        self.graph_logits = None
        self._captured = False
        self._kv_len = 0

    def reset_kv(self):
        """Reset KV cache by calling model's reset method."""
        # Can't easily reset KV cache in HF, need to re-run prefill
        pass

    @torch.no_grad()
    def capture_graph(self, input_ids, position_ids):
        """Capture a single decode step as a CUDAGraph."""
        with torch.no_grad():
            out = self.model(input_ids, position_ids=position_ids, use_cache=True)

        logits = out.logits[:, -1, :]
        self.graph_logits = logits.clone()
        self.graph_input_ids = input_ids.clone().contiguous()
        self.graph_position_ids = position_ids.clone().contiguous()

        # with torch.cuda.graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            out = self.model(self.graph_input_ids, position_ids=self.graph_position_ids, use_cache=True)
            self.graph_logits.copy_(out.logits[:, -1, :])

        torch.cuda.synchronize()
        self._captured = True

    @torch.no_grad()
    def decode_step(self, input_ids, position_ids):
        """Run a single decode step (graph-replayed if captured)."""
        if self._captured:
            self.graph_input_ids.copy_(input_ids.contiguous())
            self.graph_position_ids.copy_(position_ids.contiguous())
            self.graph.replay()
            return self.graph_logits
        else:
            out = self.model(input_ids, position_ids=position_ids, use_cache=True)
            return out.logits[:, -1, :]

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100):
        """Generate text using CUDAGraph-accelerated decode."""
        tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
        )
        tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]
        seq_len = prompt_len

        # Prefill: run normally
        out = self.model(input_ids, use_cache=True)
        next_token_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated = [next_token_id.item()]

        # Capture CUDAGraph for the first decode step
        first_pos = torch.tensor([[seq_len]], device=self.device, dtype=torch.int64)
        self.capture_graph(next_token_id, first_pos)
        seq_len += 1

        # Graph-repeated decode
        for _ in range(max_new_tokens - 1):
            pos = torch.tensor([[seq_len]], device=self.device, dtype=torch.int64)
            logits = self.decode_step(next_token_id, pos)
            next_token_id = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token_id.item())
            seq_len += 1
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            if seq_len >= self.max_seq_len:
                break

        return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
@torch.no_grad()
def benchmark():
    MODEL_PATH = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
    PROMPT = "Hello"
    N_WARM = 5
    N_RUN = 30
    MAX_NEW = 100

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    prompt_len = input_ids.shape[1]

    print("=" * 60)
    print("Qwen3-0.6B CUDAGraph Plugin Benchmark (A10)")
    print("=" * 60)

    results = []

    # --- 1. HuggingFace baseline (native generate) ---
    print("\n[1/4] HuggingFace generate...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    for _ in range(N_WARM):
        model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False,
                       use_cache=True, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_RUN):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False,
                       use_cache=True, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    tok = MAX_NEW / avg
    results.append(("HuggingFace generate", tok, avg))
    print(f"  {tok:>7.1f} tok/s, {avg*1000:.1f} ms")
    del model; torch.cuda.empty_cache()

    # --- 2. Manual decode loop ---
    print("\n[2/4] Manual decode loop...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    def manual_generate(model, input_ids, max_new):
        seq = input_ids.clone()
        past_kv = None
        for _ in range(max_new):
            if past_kv is None:
                out = model(seq, use_cache=True)
            else:
                out = model(seq[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=-1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
        return seq

    for _ in range(N_WARM):
        manual_generate(model, input_ids.clone(), MAX_NEW)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_RUN):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        manual_generate(model, input_ids.clone(), MAX_NEW)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    tok = MAX_NEW / avg
    results.append(("Manual decode loop", tok, avg))
    print(f"  {tok:>7.1f} tok/s, {avg*1000:.1f} ms")
    del model; torch.cuda.empty_cache()

    # --- 3. CUDAGraph accelerated decode ---
    print("\n[3/4] CUDAGraph accelerated decode...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    decoder = CUDAGraphDecoder(model)

    def graph_generate(model, input_ids, max_new):
        seq = input_ids.clone()
        graph = None
        graph_input = None
        graph_pos = None
        graph_logits = None
        kv_len = seq.shape[1]

        out = model(seq, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=-1)
        kv_len += 1

        # Capture graph
        graph_input = next_tok.clone().contiguous()
        graph_pos = torch.tensor([[kv_len]], device=model.device, dtype=torch.int64).contiguous()
        graph_logits = torch.empty(1, model.config.vocab_size, device=model.device, dtype=model.dtype)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_g = model(graph_input, position_ids=graph_pos, past_key_values=out.past_key_values, use_cache=True)
            graph_logits.copy_(out_g.logits[:, -1, :])

        for _ in range(max_new - 1):
            kv_len += 1
            graph_input.copy_(next_tok)
            graph_pos.fill_(kv_len)
            graph.replay()
            next_tok = graph_logits.argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=-1)
            if next_tok.item() == tokenizer.eos_token_id:
                break

        return seq

    for _ in range(N_WARM):
        graph_generate(model, input_ids.clone(), MAX_NEW)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_RUN):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph_generate(model, input_ids.clone(), MAX_NEW)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    tok = MAX_NEW / avg
    results.append(("CUDAGraph decode", tok, avg))
    print(f"  {tok:>7.1f} tok/s, {avg*1000:.1f} ms")
    del model; torch.cuda.empty_cache()
    del decoder; torch.cuda.empty_cache()

    # --- 4. A10 Megakernel (reference) ---
    print("\n[4/4] A10 Megakernel (reference)...")
    import sys, os
    sys.path.insert(0, "/mnt/workspace/DSW-GPU/MegaQwen-A10")
    from a10_decode import A10Decoder
    dec = A10Decoder(MODEL_PATH)
    for _ in range(N_WARM):
        dec.decoder.decode_step(0)
    dec.reset()
    torch.cuda.synchronize()
    times = []
    dec.reset()
    for _ in range(N_RUN):
        dec.decoder.decode_step(0)  # first step also measured
        t0 = time.perf_counter()
        for _ in range(MAX_NEW - 1):
            dec.decoder.decode_step(0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0 / 1000)  # skip first step
        # Actually let's measure all steps properly
        dec.reset()

    # Re-measure properly
    dec.reset()
    times = []
    for _ in range(N_RUN):
        t0 = time.perf_counter()
        for _ in range(MAX_NEW):
            dec.decoder.decode_step(0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    tok = MAX_NEW / avg
    results.append(("A10 Megakernel", tok, avg))
    print(f"  {tok:>7.1f} tok/s, {avg*1000:.1f} ms")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    baseline = results[0][1]
    for name, tok_s, _ in results:
        speedup = tok_s / baseline
        print(f"  {name:<30} {tok_s:>8.1f} tok/s  ({speedup:>5.2f}x)")


if __name__ == "__main__":
    benchmark()
