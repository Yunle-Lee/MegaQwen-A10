"""Test RMSNorm + QKV for determinism."""
import os, sys, torch, math
from torch.utils.cpp_extension import load_inline

MODEL_PATH = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
HIDDEN_SIZE, INTERMEDIATE_SIZE, HEAD_DIM = 1024, 3072, 128
NUM_Q_HEADS, NUM_KV_HEADS = 16, 8
Q_SIZE, KV_SIZE = NUM_Q_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
NUM_LAYERS = 28
device = torch.device('cuda')

from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
state = hf.state_dict()
del hf
torch.cuda.empty_cache()

cpp_src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void launch_test_rmsnorm_qkv(
    int token_id, int layer, int position,
    const void* embed_weight, const void* layer_weights,
    const void* cos_tbl, const void* sin_tbl,
    void* k_cache, void* v_cache,
    void* hidden_buffer, void* g_activations, void* g_residual,
    void* g_q, void* g_k, void* g_v,
    void* g_attn_out, void* g_mlp_int, void* g_normalized,
    void* sync_var,
    void* block_max_vals, void* block_max_idxs,
    int max_seq_len, float attn_scale,
    cudaStream_t stream
);

class TestRMSNormQKV {
public:
    TestRMSNormQKV(torch::Tensor embed_weight,
                   std::vector<torch::Tensor> layer_tensors,
                   torch::Tensor final_norm, torch::Tensor lm_head) {
        embed_weight_ = embed_weight;
        final_norm_ = final_norm;
        lm_head_ = lm_head;
        layer_tensors_ = layer_tensors;
        int max_seq_len = 2048;
        k_cache_ = torch::zeros({NUM_LAYERS, NUM_KV_HEADS, max_seq_len, HEAD_DIM},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({NUM_LAYERS, NUM_KV_HEADS, max_seq_len, HEAD_DIM},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        hidden_ = torch::empty({HIDDEN_SIZE}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        activations_ = torch::empty({HIDDEN_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        residual_ = torch::empty({HIDDEN_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        q_ = torch::empty({Q_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        k_ = torch::empty({KV_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        v_ = torch::empty({KV_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        attn_out_ = torch::empty({Q_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        mlp_int_ = torch::empty({INTERMEDIATE_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        normalized_ = torch::empty({HIDDEN_SIZE}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        sync_var_ = torch::empty({1}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
        block_max_vals_ = torch::empty({1184}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        block_max_idxs_ = torch::empty({1184}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        output_ = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        cos_tbl_ = torch::zeros({max_seq_len, HEAD_DIM}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        sin_tbl_ = torch::zeros({max_seq_len, HEAD_DIM}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    }
    
    int run(int token_id, int layer) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        // Build layer_weights struct with raw pointers
        launch_test_rmsnorm_qkv(
            token_id, layer, 0,
            embed_weight_.data_ptr(),
            nullptr, // layer_weights - skip for now
            cos_tbl_.data_ptr(), sin_tbl_.data_ptr(),
            k_cache_.data_ptr(), v_cache_.data_ptr(),
            hidden_.data_ptr(), activations_.data_ptr(), residual_.data_ptr(),
            q_.data_ptr(), k_.data_ptr(), v_.data_ptr(),
            attn_out_.data_ptr(), mlp_int_.data_ptr(), normalized_.data_ptr(),
            sync_var_.data_ptr(),
            block_max_vals_.data_ptr(), block_max_idxs_.data_ptr(),
            2048, 1.0f / math::sqrtf(128.0f),
            stream
        );
        cudaStreamSynchronize(stream);
        return 0;
    }
    
    torch::Tensor get_q() { return q_.cpu(); }
    
private:
    torch::Tensor embed_weight_, final_norm_, lm_head_, cos_tbl_, sin_tbl_;
    std::vector<torch::Tensor> layer_tensors_;
    torch::Tensor k_cache_, v_cache_, hidden_, activations_, residual_;
    torch::Tensor q_, k_, v_, attn_out_, mlp_int_, normalized_;
    torch::Tensor sync_var_, block_max_vals_, block_max_idxs_, output_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<TestRMSNormQKV>(m, "TestRMSNormQKV")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>())
        .def("run", &TestRMSNormQKV::run)
        .def("get_q", &TestRMSNormQKV::get_q);
}
'''

print("Compiling...")
with open(os.path.join(os.path.dirname(__file__), 'a10_decode_kernel.cu')) as f:
    cuda_src = f.read()

mod = load_inline(
    name="test_rmsnorm",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", "-arch=sm_86",
                       "-I" + os.path.dirname(os.path.abspath(__file__))],
    verbose=False,
)
print("Done")
