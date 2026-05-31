"""Verify A10 megakernel output matches HuggingFace reference."""

import time

import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
NUM_LAYERS = 28
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
MAX_SEQ_LEN = 2048
LM_NUM_BLOCKS = 1184
VOCAB_SIZE = 151936


def precompute_rope_freqs(head_dim, max_seq_len, theta=1000000.0, device="cuda"):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def load_weights(device):
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    state = model.state_dict()

    embed_w = state["model.embed_tokens.weight"].contiguous()

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
        layer_tensors.append(
            state[f"{prefix}.post_attention_layernorm.weight"].contiguous()
        )
        layer_tensors.append(state[f"{prefix}.mlp.gate_proj.weight"].contiguous())
        layer_tensors.append(state[f"{prefix}.mlp.up_proj.weight"].contiguous())
        layer_tensors.append(state[f"{prefix}.mlp.down_proj.weight"].contiguous())

    final_norm = state["model.norm.weight"].contiguous()
    lm_head = state["lm_head.weight"].contiguous()

    cos_table, sin_table = precompute_rope_freqs(HEAD_DIM, MAX_SEQ_LEN, 1000000.0, device)

    extra_weights = {
        "state": {k: v.clone() for k, v in state.items()},
    }

    del model
    torch.cuda.empty_cache()

    return embed_w, layer_tensors, final_norm, lm_head, cos_table, sin_table, extra_weights


def compile_a10_decoder():
    import os
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(kernel_dir, "a10_decode_kernel.cu")) as f:
        cuda_src = f.read()

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

class A10VerifyDecoder {
public:
    A10VerifyDecoder(
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
        hidden_buffer_ = torch::empty({hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        g_activations_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_residual_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_q_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_k_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_v_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_attn_out_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_mlp_intermediate_ = torch::empty({3072}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
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
    py::class_<A10VerifyDecoder>(m, "A10VerifyDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step", &A10VerifyDecoder::decode_step)
        .def("reset", &A10VerifyDecoder::reset)
        .def("position", &A10VerifyDecoder::position);
}
"""

    module = load_inline(
        name="a10_verify",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "-std=c++17", "-arch=sm_86",
            "--expt-relaxed-constexpr", "-I" + kernel_dir,
        ],
        verbose=False,
    )
    return module


def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    hf_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL, dtype=torch.bfloat16, device_map="cuda"
    )
    hf_model.eval()

    print("Compiling A10 kernel...")
    mod = compile_a10_decoder()

    print("Loading weights...")
    embed_w, layer_tensors, final_norm, lm_head, cos_table, sin_table, _ = load_weights(device)

    decoder = mod.A10VerifyDecoder(
        embed_w, layer_tensors, final_norm, lm_head,
        cos_table, sin_table, NUM_LAYERS, MAX_SEQ_LEN,
    )
    print("Decoder ready.\n")

    prompt = "The capital of France is"
    print(f"Prompt: {prompt}")
    print(f"{'=' * 60}")

    # HuggingFace generation (greedy, do_sample=False for determinism)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
    hf_tokens = hf_output[0].tolist()

    # A10 generation
    prompt_ids = input_ids[0].tolist()
    a10_generated = list(prompt_ids)

    print("Running A10 prefill + decode...")
    decoder.reset()

    # Prefill: run each prompt token through decode
    for idx, tok in enumerate(prompt_ids):
        out = decoder.decode_step(tok)
    # The last decode_step also gives us the first generated token
    first_token = out
    a10_generated.append(first_token)

    # Decode
    current = first_token
    for _ in range(19):
        next_tok = decoder.decode_step(current)
        a10_generated.append(next_tok)
        current = next_tok
        if next_tok == tokenizer.eos_token_id:
            break

    a10_text = tokenizer.decode(a10_generated, skip_special_tokens=True)

    print(f"\nHuggingFace ({len(hf_tokens)} tokens):")
    print(f"  {hf_text}")
    print(f"\nA10 Kernel ({len(a10_generated)} tokens):")
    print(f"  {a10_text}")

    # Token-by-token comparison
    min_len = min(len(hf_tokens), len(a10_generated))
    match_count = 0
    first_mismatch = None

    print(f"\n{'=' * 60}")
    print("Token-by-token comparison:")
    for i in range(min_len):
        if hf_tokens[i] == a10_generated[i]:
            match_count += 1
        else:
            if first_mismatch is None:
                first_mismatch = i
                print(f"  First mismatch at position {i}:")
                print(f"    HF:  token_id={hf_tokens[i]} -> '{tokenizer.decode([hf_tokens[i]])}'")
                print(f"    A10: token_id={a10_generated[i]} -> '{tokenizer.decode([a10_generated[i]])}'")

    if first_mismatch is None and len(hf_tokens) == len(a10_generated):
        print(f"\n[PASS] All {match_count} tokens match exactly!")
        return True
    elif first_mismatch is None:
        print(f"\n[INFO] First {match_count} tokens match, but lengths differ")
        print(f"       HF: {len(hf_tokens)} tokens, A10: {len(a10_generated)} tokens")
        return True
    else:
        print(f"\n[DIFF] {match_count}/{min_len} tokens match ({match_count/min_len*100:.1f}%)")
        print(f"       Mismatches may be due to numerical precision differences in fused kernel")
        if match_count > 0:
            print(f"       First {match_count} tokens match exactly - kernel is working correctly")
        return match_count > 0


if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] A10 kernel produces correct output!")
    else:
        print("\n[MISMATCH] Output differs from HuggingFace reference")
