"""Benchmark the A10-optimized kernel vs baseline."""
import os
import sys
import time
import math

import torch
from torch.utils.cpp_extension import load_inline

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


def compile_module():
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(kernel_dir, "a10_decode_kernel.cu")) as f:
        cuda_src = f.read()

    cpp_src = '''
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
    void* global_sync_var,
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
        global_sync_var_ = torch::empty({1}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
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
            global_sync_var_.data_ptr(),
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

    // Debug getters
    torch::Tensor get_hidden() const { return hidden_buffer_; }
    torch::Tensor get_activations() const { return g_activations_; }
    torch::Tensor get_residual() const { return g_residual_; }
    torch::Tensor get_q() const { return g_q_; }
    torch::Tensor get_k() const { return g_k_; }
    torch::Tensor get_v() const { return g_v_; }
    torch::Tensor get_attn_out() const { return g_attn_out_; }
    torch::Tensor get_mlp_int() const { return g_mlp_intermediate_; }
    torch::Tensor get_normalized() const { return g_normalized_; }

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
    torch::Tensor global_sync_var_;
    torch::Tensor block_max_vals_, block_max_idxs_, output_token_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<A10MegakernelDecoder>(m, "A10MegakernelDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step", &A10MegakernelDecoder::decode_step)
        .def("reset", &A10MegakernelDecoder::reset)
        .def("position", &A10MegakernelDecoder::position)
        .def("get_hidden", &A10MegakernelDecoder::get_hidden)
        .def("get_activations", &A10MegakernelDecoder::get_activations)
        .def("get_residual", &A10MegakernelDecoder::get_residual)
        .def("get_q", &A10MegakernelDecoder::get_q)
        .def("get_k", &A10MegakernelDecoder::get_k)
        .def("get_v", &A10MegakernelDecoder::get_v)
        .def("get_attn_out", &A10MegakernelDecoder::get_attn_out)
        .def("get_mlp_int", &A10MegakernelDecoder::get_mlp_int)
        .def("get_normalized", &A10MegakernelDecoder::get_normalized);
}
'''

    print("Compiling A10 kernel...")
    mod = load_inline(
        name="a10_decode",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "-std=c++17", "-arch=sm_86",
            "--expt-relaxed-constexpr", "-I" + kernel_dir,
        ],
        verbose=False,
    )
    print("Compilation OK.")
    return mod


def create_rope_table(max_seq_len, head_dim, device):
    cos_t = torch.zeros(max_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    sin_t = torch.zeros(max_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    for pos in range(max_seq_len):
        for d in range(0, head_dim, 2):
            theta = pos / (10000.0 ** (d / head_dim))
            cv = math.cos(theta)
            sv = math.sin(theta)
            cos_t[pos, d] = cv
            cos_t[pos, d + 1] = cv
            sin_t[pos, d] = sv
            sin_t[pos, d + 1] = sv
    return cos_t, sin_t


def load_weights(model_path, device):
    from transformers import AutoModelForCausalLM
    print(f"Loading model from {model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    state = hf_model.state_dict()

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    layer_tensors = []
    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}"
        layer_tensors.append(state[f"{prefix}.input_layernorm.weight"].contiguous())
        layer_tensors.append(state[f"{prefix}.self_attn.q_proj.weight"].contiguous())
        layer_tensors.append(state[f"{prefix}.self_attn.k_proj.weight"].contiguous())
        layer_tensors.append(state[f"{prefix}.self_attn.v_proj.weight"].contiguous())
        qn = state.get(f"{prefix}.self_attn.q_norm.weight")
        kn = state.get(f"{prefix}.self_attn.k_norm.weight")
        if qn is None:
            qn = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)
        if kn is None:
            kn = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)
        layer_tensors.append(qn.contiguous())
        layer_tensors.append(kn.contiguous())
        layer_tensors.append(state[f"{prefix}.self_attn.o_proj.weight"].contiguous())
        layer_tensors.append(state[f"{prefix}.post_attention_layernorm.weight"].contiguous())
        layer_tensors.append(state[f"{prefix}.mlp.gate_proj.weight"].contiguous())
        layer_tensors.append(state[f"{prefix}.mlp.up_proj.weight"].contiguous())
        layer_tensors.append(state[f"{prefix}.mlp.down_proj.weight"].contiguous())

    final_norm = state["model.norm.weight"].contiguous()
    lm_head = state["lm_head.weight"].contiguous()

    del hf_model
    torch.cuda.empty_cache()

    return embed_weight, layer_tensors, final_norm, lm_head


def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"SM Count: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    mod = compile_module()

    cos_table, sin_table = create_rope_table(MAX_SEQ_LEN, HEAD_DIM, device)
    embed_weight, layer_tensors, final_norm, lm_head = load_weights(MODEL_PATH, device)

    print("Creating decoder...")
    decoder = mod.A10MegakernelDecoder(
        embed_weight, layer_tensors, final_norm, lm_head,
        cos_table, sin_table, NUM_LAYERS, MAX_SEQ_LEN,
    )
    print("Decoder ready.")

    # Warmup
    N_WARM = 10
    print(f"Warmup ({N_WARM} steps)...")
    for i in range(N_WARM):
        decoder.decode_step(0)
    decoder.reset()
    torch.cuda.synchronize()

    # Benchmark
    N_STEPS = 100
    print(f"Benchmark ({N_STEPS} steps)...")
    times = []
    for i in range(N_STEPS):
        t0 = time.time()
        tok = decoder.decode_step(0)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    tok_s = 1000.0 / avg_ms
    print(f"\nResults ({N_STEPS} decode steps):")
    print(f"  Average decode time: {avg_ms:.2f} ms")
    print(f"  Throughput: {tok_s:.1f} tok/s")
    print(f"  Min: {min(times):.2f} ms, Max: {max(times):.2f} ms")

    return tok_s


if __name__ == "__main__":
    main()
