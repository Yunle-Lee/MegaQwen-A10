"""Quick demo of the A10 kernel for Qwen3-0.6B."""

import time

from a10_chat import A10Chat

print("=" * 60)
print("MegaQwen-A10 Demo - Qwen3-0.6B A10-optimized Inference")
print("=" * 60)
print()

print("Loading model and compiling A10 kernel...")
chat = A10Chat()

print("Warming up...")
chat.generate("Hello", max_new_tokens=5, show_speed=False)
print("Ready!\n")

prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain quantum computing in one sentence:",
]

for prompt in prompts:
    print("-" * 60)
    print(f"Prompt: {prompt}")
    print()

    start = time.perf_counter()

    print("Response: ", end="", flush=True)
    response = chat.generate_stream(prompt, max_new_tokens=50)

    elapsed = time.perf_counter() - start
    response_len = len(chat.tokenizer.encode(response))
    print()
    print(f"\n[{response_len} tokens in {elapsed:.2f}s = {response_len/elapsed:.1f} tok/s]")
    print()

print("=" * 60)
print("Demo complete!")
