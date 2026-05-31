"""
Compile and benchmark the INT8 A10 fused kernel.
"""
import os, sys, time, math, torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
PROJECT = os.path.dirname(os.path.abspath(__file__))

HIDDEN_SIZE = 1024; INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16; NUM_KV_HEADS = 8
HEAD_DIM = 128; Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
NUM_LAYERS = 28; VOCAB_SIZE = 151936
MAX_SEQ_LEN = 2048; LM_NUM_BLOCKS = 1184


def quantize_per_channel(tensor):
    M, N = tensor.shape
    max_abs = tensor.abs().amax(dim=1)
    scale = (max_abs / 127.0).to(torch.float32).clamp(min=1e-12)
    w_int8 = (tensor / scale.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale


def load_quantized_state(device):
    from transformers import AutoModelForCausalLM
    print("Loading model...")
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    state = hf.state_dict()
    
    qstate = {}
    # Quantize ALL linear projections: Q, K, V, O, gate, up, down, lm_head
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            key = f"{p}.{proj}.weight"
            if key in state:
                qstate[key] = quantize_per_channel(state[key])
        for other in ["input_layernorm.weight", "self_attn.q_norm.weight",
                       "self_attn.k_norm.weight", "post_attention_layernorm.weight"]:
            key = f"{p}.{other}"
            if key in state:
                qstate[key] = state[key]
    
    qstate["lm_head.weight"] = quantize_per_channel(state["lm_head.weight"])
    qstate["model.embed_tokens.weight"] = state["model.embed_tokens.weight"]
    qstate["model.norm.weight"] = state["model.norm.weight"]
    
    return hf, qstate


def compile_int8_kernel():
    kernel_dir = PROJECT
    with open(os.path.join(kernel_dir, "a10_int8_decode_kernel.cu")) as f:
        cuda_src = f.read()
    with open(os.path.join(kernel_dir, "config.cuh")) as f:
        config_src = f.read()

    # Wrap CUDA source with config
    full_cuda = config_src + "\n" + cuda_src

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

struct ScalePointers {
    float* q_scale;
    float* k_scale;
    float* v_scale;
    float* o_scale;
    float* gate_scale;
    float* up_scale;
    float* down_scale;
};

extern "C" void launch_a10_int8_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const float* lm_head_scale,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* hidden_buffer,
    void* g_activations, void* g_residual,
    void* g_q, void* g_k, void* g_v,
    void* g_attn_out, void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals, void* block_max_idxs,
    int num_layers, int position, int cache_len,
    int max_seq_len, float attn_scale,
    const ScalePointers* scales,
    cudaStream_t stream
);

class A10Int8MegakernelDecoder {
public:
    A10Int8MegakernelDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        std::vector<torch::Tensor> layer_scales_all,
        torch::Tensor lm_head_int8,
        torch::Tensor lm_head_scale,
        torch::Tensor final_norm_weight,
        torch::Tensor cos_table, torch::Tensor sin_table,
        int num_layers, int max_seq_len
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len) {
        embed_weight_ = embed_weight;
        lm_head_weight_ = lm_head_int8;
        lm_head_scale_ = lm_head_scale;
        final_norm_weight_ = final_norm_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;
        layer_weights_tensors_ = layer_weights_flat;

        layer_weights_.resize(num_layers);
        layer_scales_.resize(num_layers);
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

            layer_scales_[i].q_scale = (float*)layer_scales_all[i * 7 + 0].data_ptr();
            layer_scales_[i].k_scale = (float*)layer_scales_all[i * 7 + 1].data_ptr();
            layer_scales_[i].v_scale = (float*)layer_scales_all[i * 7 + 2].data_ptr();
            layer_scales_[i].o_scale = (float*)layer_scales_all[i * 7 + 3].data_ptr();
            layer_scales_[i].gate_scale = (float*)layer_scales_all[i * 7 + 4].data_ptr();
            layer_scales_[i].up_scale = (float*)layer_scales_all[i * 7 + 5].data_ptr();
            layer_scales_[i].down_scale = (float*)layer_scales_all[i * 7 + 6].data_ptr();
        }

        d_layer_weights_ = torch::empty({num_layers * (int)sizeof(LayerWeights)},
                                         torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_weights_.data_ptr(), layer_weights_.data(),
                   num_layers * sizeof(LayerWeights), cudaMemcpyHostToDevice);
        d_layer_scales_ = torch::empty({num_layers * (int)sizeof(ScalePointers)},
                                        torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_scales_.data_ptr(), layer_scales_.data(),
                   num_layers * sizeof(ScalePointers), cudaMemcpyHostToDevice);
        int kv_heads = 8; int hd = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, hd},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, hd},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        int hs = 1024; int qs = 2048; int ks = 1024; int im = 3072;
        hidden_buffer_ = torch::empty({hs}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        g_activations_ = torch::empty({hs}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_residual_ = torch::empty({hs}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_q_ = torch::empty({qs}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_k_ = torch::empty({ks}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_v_ = torch::empty({ks}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_attn_out_ = torch::empty({qs}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_mlp_intermediate_ = torch::empty({im}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_normalized_ = torch::empty({hs}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        output_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        position_ = 0;
        attn_scale_ = 1.0f / sqrtf(128.0f);
    }

    int decode_step(int input_token_id) {
        int cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        launch_a10_int8_decode(
            input_token_id,
            (int*)output_token_.data_ptr(),
            embed_weight_.data_ptr(),
            (const LayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            (const float*)lm_head_scale_.data_ptr(),
            cos_table_.data_ptr(), sin_table_.data_ptr(),
            k_cache_.data_ptr(), v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(), g_residual_.data_ptr(),
            g_q_.data_ptr(), g_k_.data_ptr(), g_v_.data_ptr(),
            g_attn_out_.data_ptr(), g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            block_max_vals_.data_ptr(), block_max_idxs_.data_ptr(),
            num_layers_, position_, cache_len,
            max_seq_len_, attn_scale_,
            (const ScalePointers*)d_layer_scales_.data_ptr(),
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
    torch::Tensor embed_weight_;
    torch::Tensor lm_head_weight_, lm_head_scale_;
    torch::Tensor final_norm_weight_, cos_table_, sin_table_;
    torch::Tensor d_layer_weights_, d_layer_scales_;
    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<LayerWeights> layer_weights_;
    std::vector<ScalePointers> layer_scales_;
    torch::Tensor k_cache_, v_cache_;
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;
    torch::Tensor block_max_vals_, block_max_idxs_, output_token_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<A10Int8MegakernelDecoder>(m, "A10Int8MegakernelDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>,
                      std::vector<torch::Tensor>,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step", &A10Int8MegakernelDecoder::decode_step)
        .def("reset", &A10Int8MegakernelDecoder::reset)
        .def("position", &A10Int8MegakernelDecoder::position);
}
'''
    print("Compiling INT8 kernel...")
    mod = load_inline(
        name="a10_int8_decode",
        cpp_sources=[cpp_src],
        cuda_sources=[full_cuda],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", "-arch=sm_86",
                           "--expt-relaxed-constexpr", "-I" + kernel_dir],
        verbose=False,
    )
    print("Compilation OK.")
    return mod


def make_rope(max_seq_len, head_dim, device):
    cos_t = torch.zeros(max_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    sin_t = torch.zeros(max_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    for pos in range(max_seq_len):
        for d in range(0, head_dim, 2):
            theta = pos / (10000.0 ** (d / head_dim))
            c, s = math.cos(theta), math.sin(theta)
            cos_t[pos, d] = cos_t[pos, d + 1] = c
            sin_t[pos, d] = sin_t[pos, d + 1] = s
    return cos_t, sin_t


@torch.no_grad()
def benchmark():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    N_WARM = 5; N_RUN = 30; MAX_NEW = 100

    cos_t, sin_t = make_rope(MAX_SEQ_LEN, HEAD_DIM, device)
    hf, qstate = load_quantized_state(device)

    # Build layer weight tensors (11 per layer: must match C++ LayerWeights ordering)
    # Index:  0           1       2       3       4       5       6      7                     8      9      10
    # Field:  input_norm  q_proj  k_proj  v_proj  q_norm  k_norm  o_proj  post_attn_norm  gate   up     down
    layer_tensors = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        layer_tensors.append(qstate[f"{p}.input_layernorm.weight"].contiguous())                     # 0
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:                     # 1,2,3
            key = f"{p}.{proj}.weight"
            w = qstate[key]
            if isinstance(w, tuple):
                w = w[0]
            layer_tensors.append(w.contiguous())
        layer_tensors.append(qstate.get(f"{p}.self_attn.q_norm.weight",                               # 4
                                        torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)).contiguous())
        layer_tensors.append(qstate.get(f"{p}.self_attn.k_norm.weight",                               # 5
                                        torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)).contiguous())
        for proj in ["self_attn.o_proj"]:                                                             # 6
            key = f"{p}.{proj}.weight"
            w = qstate[key]
            if isinstance(w, tuple):
                w = w[0]
            layer_tensors.append(w.contiguous())
        layer_tensors.append(qstate[f"{p}.post_attention_layernorm.weight"].contiguous())             # 7
        for proj in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:                                # 8,9,10
            key = f"{p}.{proj}.weight"
            w = qstate[key]
            if isinstance(w, tuple):
                w = w[0]
            layer_tensors.append(w.contiguous())

    # Build scale tensors (7 per layer: q, k, v, o, gate, up, down)
    layer_scales = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            w = qstate[f"{p}.{proj}.weight"]
            layer_scales.append(w[1].contiguous())  # scale

    # LM head
    lm_head = qstate["lm_head.weight"]
    lm_head_int8 = lm_head[0].contiguous()
    lm_head_scale = lm_head[1].contiguous()

    final_norm = qstate["model.norm.weight"].contiguous()
    embed = qstate["model.embed_tokens.weight"].contiguous()

    mod = compile_int8_kernel()
    dec = mod.A10Int8MegakernelDecoder(
        embed, layer_tensors, layer_scales,
        lm_head_int8, lm_head_scale, final_norm,
        cos_t, sin_t, NUM_LAYERS, MAX_SEQ_LEN
    )

    results = []

    # --- 1. INT8 A10 Kernel ---
    print("\n[1/2] A10 INT8 Kernel...")
    for step in range(N_WARM):
        dec.decode_step(0)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_RUN):
        dec.reset()
        t0 = time.perf_counter()
        for step in range(MAX_NEW):
            dec.decode_step(0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    tok = MAX_NEW / avg
    results.append(("A10 Kernel INT8", tok, avg))
    print(f"  {tok:>7.1f} tok/s, {avg*1e6/MAX_NEW:.0f} μs/step")

    # --- 2. Original A10 BF16 baseline ---
    print("\n[2/2] A10 Kernel BF16 (reference)...")
    from a10_decode import A10Decoder
    dec_bf16 = A10Decoder()
    for step in range(N_WARM):
        dec_bf16.decoder.decode_step(0)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_RUN):
        dec_bf16.reset()
        t0 = time.perf_counter()
        for step in range(MAX_NEW):
            dec_bf16.decoder.decode_step(0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    tok = MAX_NEW / avg
    results.append(("A10 Kernel BF16", tok, avg))
    print(f"  {tok:>7.1f} tok/s, {avg*1e6/MAX_NEW:.0f} μs/step")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, tok_s, avg_t in results:
        sp = tok_s / results[-1][1]  # vs BF16
        print(f"  {name:<25} {tok_s:>8.1f} tok/s  ({avg_t*1e6/MAX_NEW:>4.0f} μs)  {sp:>5.2f}x")


if __name__ == "__main__":
    benchmark()
