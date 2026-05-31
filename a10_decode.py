"""
A10-Optimized Fused Decode for Qwen3-0.6B
"""

import os
import time
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoTokenizer

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
LM_NUM_BLOCKS = 1184


def _get_cuda_source(filename: str) -> str:
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(kernel_dir, filename)) as f:
        return f.read()


class A10Decoder:
    def __init__(self, model_path: str = MODEL_PATH, max_seq_len: int = MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.device = torch.device("cuda")
        self._compile_kernel()
        self._load_model(model_path)
        self._allocate_buffers()
        self._setup_weights()

    def _compile_kernel(self):
        cuda_src = _get_cuda_source("a10_decode_kernel.cu")

        cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

struct LayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_a10_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache, void* v_cache,
    void* hidden_buffer,
    void* g_activations, void* g_residual,
    void* g_q, void* g_k, void* g_v,
    void* g_attn_out, void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals, void* block_max_idxs,
    int num_layers, int position, int cache_len,
    int max_seq_len, float attn_scale,
    cudaStream_t stream
);

class A10MegakernelDecoder {
public:
    A10MegakernelDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len) {

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;
        layer_weights_tensors_ = layer_weights_flat;

        layer_weights_.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();
        }

        d_layer_weights_ = torch::empty({num_layers * (int)sizeof(LayerWeights)},
                                         torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_weights_.data_ptr(), layer_weights_.data(),
                   num_layers * sizeof(LayerWeights), cudaMemcpyHostToDevice);

        int kv_heads = 8;
        int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        int hidden_size = 1024;
        int q_size = 16 * 128;
        int kv_size = 8 * 128;
        int intermediate_size = 3072;

        hidden_buffer_ = torch::empty({hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        g_activations_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_residual_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_q_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_k_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_v_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_attn_out_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_mlp_intermediate_ = torch::empty({intermediate_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_normalized_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        output_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));

        position_ = 0;
        attn_scale_ = 1.0f / sqrtf(128.0f);
    }

    int decode_step(int input_token_id) {
        int cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_a10_decode(
            input_token_id,
            (int*)output_token_.data_ptr(),
            embed_weight_.data_ptr(),
            (const LayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            position_,
            cache_len,
            max_seq_len_,
            attn_scale_,
            stream
        );

        position_++;
        return output_token_.item<int>();
    }

    int decode_step_with_logits(int input_token_id, torch::Tensor logits) {
        int cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_a10_decode(
            input_token_id,
            (int*)output_token_.data_ptr(),
            embed_weight_.data_ptr(),
            (const LayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            block_max_vals_.data_ptr(),
            block_max_idxs_.data_ptr(),
            num_layers_,
            position_,
            cache_len,
            max_seq_len_,
            attn_scale_,
            stream
        );

        position_++;
        return output_token_.item<int>();
    }

    void reset() {
        position_ = 0;
        k_cache_.zero_();
        v_cache_.zero_();
    }

    int position() const { return position_; }

private:
    int num_layers_, max_seq_len_, position_;
    float attn_scale_;
    torch::Tensor embed_weight_, final_norm_weight_, lm_head_weight_;
    torch::Tensor cos_table_, sin_table_, d_layer_weights_;
    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<LayerWeights> layer_weights_;
    torch::Tensor k_cache_, v_cache_;
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;
    torch::Tensor block_max_vals_, block_max_idxs_, output_token_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<A10MegakernelDecoder>(m, "A10MegakernelDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step", &A10MegakernelDecoder::decode_step)
        .def("reset", &A10MegakernelDecoder::reset)
        .def("position", &A10MegakernelDecoder::position);
}
"""

        kernel_dir = os.path.dirname(os.path.abspath(__file__))

        self._module = load_inline(
            name="a10_decode",
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-arch=sm_86",
                "--expt-relaxed-constexpr",
                "-I" + kernel_dir,
            ],
            verbose=False,
        )

    def _load_model(self, model_path: str):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        print("Model loaded.")

        # Extract weights
        state = hf_model.state_dict()
        self.embed_weight = state["model.embed_tokens.weight"].contiguous()

        # Layer weights: 11 per layer
        self.layer_tensors = []
        for i in range(NUM_LAYERS):
            prefix = f"model.layers.{i}"
            # Input layernorm
            self.layer_tensors.append(state[f"{prefix}.input_layernorm.weight"].contiguous())
            # QKV projections
            q_w = state[f"{prefix}.self_attn.q_proj.weight"].contiguous()
            k_w = state[f"{prefix}.self_attn.k_proj.weight"].contiguous()
            v_w = state[f"{prefix}.self_attn.v_proj.weight"].contiguous()
            self.layer_tensors.append(q_w)
            self.layer_tensors.append(k_w)
            self.layer_tensors.append(v_w)
            # QK norms
            q_norm_w = state.get(f"{prefix}.self_attn.q_norm.weight")
            k_norm_w = state.get(f"{prefix}.self_attn.k_norm.weight")
            if q_norm_w is None:
                q_norm_w = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=self.device)
            if k_norm_w is None:
                k_norm_w = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=self.device)
            self.layer_tensors.append(q_norm_w.contiguous())
            self.layer_tensors.append(k_norm_w.contiguous())
            # O projection
            self.layer_tensors.append(state[f"{prefix}.self_attn.o_proj.weight"].contiguous())
            # Post-attn layernorm
            self.layer_tensors.append(state[f"{prefix}.post_attention_layernorm.weight"].contiguous())
            # MLP
            self.layer_tensors.append(state[f"{prefix}.mlp.gate_proj.weight"].contiguous())
            self.layer_tensors.append(state[f"{prefix}.mlp.up_proj.weight"].contiguous())
            self.layer_tensors.append(state[f"{prefix}.mlp.down_proj.weight"].contiguous())

        self.final_norm_weight = state["model.norm.weight"].contiguous()
        self.lm_head_weight = state["lm_head.weight"].contiguous()

        self.hf_model = hf_model

    def _allocate_buffers(self):
        # RoPE tables
        self.cos_table = torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        self.sin_table = torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        for pos in range(MAX_SEQ_LEN):
            for d in range(0, HEAD_DIM, 2):
                theta = pos / (10000.0 ** (d / HEAD_DIM))
                self.cos_table[pos, d] = math.cos(theta)
                self.sin_table[pos, d] = math.sin(theta)
                self.cos_table[pos, d + 1] = math.cos(theta)
                self.sin_table[pos, d + 1] = math.sin(theta)

    def _setup_weights(self):
        print("Creating A10MegakernelDecoder...")
        self.decoder = self._module.A10MegakernelDecoder(
            self.embed_weight,
            self.layer_tensors,
            self.final_norm_weight,
            self.lm_head_weight,
            self.cos_table,
            self.sin_table,
            NUM_LAYERS,
            MAX_SEQ_LEN,
        )
        print("Decoder ready.")

    def reset(self):
        self.decoder.reset()

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_ids = input_ids[0].tolist()
        prompt_len = len(prompt_ids)
        print(f"Prompt: {prompt!r}  ({prompt_len} tokens)")

        self.reset()

        prefill_start = time.time()
        for position, token_id in enumerate(prompt_ids):
            self.decoder.decode_step(token_id)
        prefill_time = time.time() - prefill_start

        generated = list(prompt_ids)
        decode_times = []

        decode_start = time.time()
        current_token = prompt_ids[-1]
        for _ in range(max_new_tokens):
            t0 = time.time()
            next_token = self.decoder.decode_step(current_token)
            torch.cuda.synchronize()
            t1 = time.time()
            decode_times.append((t1 - t0) * 1000)
            generated.append(next_token)
            current_token = next_token
            if next_token == self.tokenizer.eos_token_id:
                break
        decode_time = time.time() - decode_start

        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        avg_ms = sum(decode_times) / len(decode_times) if decode_times else 0
        tok_s = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"  prefill: {prompt_len} tok @ {prompt_len/prefill_time:.0f} tok/s | "
              f"decode: {len(generated)-prompt_len} tok @ {tok_s:.0f} tok/s")
        return output_text, tok_s, decode_times


def benchmark(model_path: str = MODEL_PATH, num_warmup: int = 5, num_steps: int = 50):
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_path}")

    # Use existing model loading via the decoder
    decoder_wrapper = A10Decoder(model_path)

    # Prefill: decode the first token with KV cache
    print("Prefilling KV cache...")
    # For benchmark: we decode step-by-step starting from token 0
    # The first decode fills KV cache for position 0

    # Warmup
    print(f"Warmup ({num_warmup} steps)...")
    for i in range(num_warmup):
        decoder_wrapper.decoder.decode_step(0)
    decoder_wrapper.reset()

    # Benchmark
    print(f"Benchmark ({num_steps} steps)...")
    decode_times = []
    for i in range(num_steps):
        t0 = time.time()
        decoder_wrapper.decoder.decode_step(0)
        torch.cuda.synchronize()
        t1 = time.time()
        decode_times.append((t1 - t0) * 1000)

    avg_ms = sum(decode_times) / len(decode_times)
    tok_s = 1000.0 / avg_ms
    print(f"\nResults ({num_steps} decode steps):")
    print(f"  Average decode time: {avg_ms:.2f} ms")
    print(f"  Throughput: {tok_s:.1f} tok/s")
    return tok_s


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark()
    else:
        decoder = A10Decoder()
        text, tok_s, times = decoder.generate("Hello", max_new_tokens=32)
        print(f"\nGenerated: {text!r}")
        print(f"Throughput: {tok_s:.1f} tok/s")
