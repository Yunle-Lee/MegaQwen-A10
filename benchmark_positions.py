"""Per-position benchmark for A10-optimized kernel."""
import gc, json, os, sys, time, warnings, math
warnings.filterwarnings("ignore")
import torch

PROJECT = os.path.dirname(os.path.abspath(__file__))
MEGA_PROJECT = "/mnt/workspace/DSW-GPU/MegaQwen"
MODEL_PATH = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
sys.path.insert(0, PROJECT)

TEST_TOKEN_POSITIONS = [1, 10, 25, 50, 75, 100, 128, 150, 175, 200]
WARMUP = 2
RUNS = 3

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

def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(1)

def measure(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times)/len(times)

def compile_module():
    from torch.utils.cpp_extension import load_inline
    kernel_dir = PROJECT
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
    void* block_max_vals, void* block_max_idxs,
    int num_layers, int position, int cache_len,
    int max_seq_len, float attn_scale,
    cudaStream_t stream
);

class A10BenchDecoder {
public:
    A10BenchDecoder(
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
        int kv_heads = 8; int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        hidden_buffer_ = torch::empty({1024}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        g_activations_ = torch::empty({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_residual_ = torch::empty({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_q_ = torch::empty({2048}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_k_ = torch::empty({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_v_ = torch::empty({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_attn_out_ = torch::empty({2048}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_mlp_intermediate_ = torch::empty({3072}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_normalized_ = torch::empty({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        output_token_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        position_ = 0;
        attn_scale_ = 1.0f / sqrtf(128.0f);
    }
    int decode_step(int input_token_id) {
        int cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        launch_a10_decode(input_token_id, (int*)output_token_.data_ptr(),
            embed_weight_.data_ptr(), (const LayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(), lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(), sin_table_.data_ptr(),
            k_cache_.data_ptr(), v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(), g_activations_.data_ptr(), g_residual_.data_ptr(),
            g_q_.data_ptr(), g_k_.data_ptr(), g_v_.data_ptr(),
            g_attn_out_.data_ptr(), g_mlp_intermediate_.data_ptr(), g_normalized_.data_ptr(),
            block_max_vals_.data_ptr(), block_max_idxs_.data_ptr(),
            num_layers_, position_, cache_len, max_seq_len_, attn_scale_, stream);
        position_++;
        return output_token_.item<int>();
    }
    void reset() { position_ = 0; k_cache_.zero_(); v_cache_.zero_(); }
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
    py::class_<A10BenchDecoder>(m, "A10BenchDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step", &A10BenchDecoder::decode_step)
        .def("reset", &A10BenchDecoder::reset)
        .def("position", &A10BenchDecoder::position);
}
'''
    print("Compiling A10 benchmark kernel...")
    mod = load_inline(
        name="a10_bench",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", "-arch=sm_86",
                           "--expt-relaxed-constexpr", "-I" + kernel_dir],
        verbose=False,
    )
    print("Compilation OK.")
    return mod

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
        if qn is None: qn = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)
        if kn is None: kn = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)
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

def benchmark_a10():
    print("\n=== A10-Optimized Kernel ===")
    device = torch.device("cuda")
    mod = compile_module()
    cos_table = torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=device)
    sin_table = torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=device)
    for pos in range(MAX_SEQ_LEN):
        for d in range(0, HEAD_DIM, 2):
            theta = pos / (10000.0 ** (d / HEAD_DIM))
            cos_table[pos, d] = cos_table[pos, d + 1] = math.cos(theta)
            sin_table[pos, d] = sin_table[pos, d + 1] = math.sin(theta)
    embed_weight, layer_tensors, final_norm, lm_head = load_weights(MODEL_PATH, device)
    decoder = mod.A10BenchDecoder(
        embed_weight, layer_tensors, final_norm, lm_head,
        cos_table, sin_table, NUM_LAYERS, MAX_SEQ_LEN,
    )
    results = {}
    for n in TEST_TOKEN_POSITIONS:
        def gen_fn(n=n):
            decoder.reset()
            for _ in range(n):
                decoder.decode_step(0)
        t = measure(gen_fn)
        results[n] = {"tok_s": n/t, "ms_tok": t*1000/n, "time_s": t}
        print(f"  {n:4d} tok: {n/t:8.1f} tok/s, {t*1000/n:.2f} ms/tok")
    del decoder; clear()
    return results

def main():
    all_results = {}
    all_results["A10-Optimized"] = benchmark_a10()
    a10_out = os.path.join(PROJECT, "a10_benchmark_results.json")
    with open(a10_out, "w") as f:
        json.dump({"test_positions": TEST_TOKEN_POSITIONS, "results": all_results}, f, indent=2)
    print(f"\nA10 results saved: {a10_out}")

    # Load existing MegaQwen results
    mega_results_path = os.path.join(MEGA_PROJECT, "benchmark_results.json")
    if os.path.exists(mega_results_path):
        with open(mega_results_path) as f:
            mega = json.load(f)
        merged = mega["results"]
        merged["A10-Optimized"] = all_results["A10-Optimized"]

        combined_out = os.path.join(PROJECT, "combined_benchmark_results.json")
        with open(combined_out, "w") as f:
            json.dump({"test_positions": TEST_TOKEN_POSITIONS, "results": merged}, f, indent=2)
        print(f"Combined results saved: {combined_out}")

    print("\nDone!")

if __name__ == "__main__":
    main()
