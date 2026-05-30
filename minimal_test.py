"""Minimal test to isolate the race condition."""
import os
import sys
import torch
from torch.utils.cpp_extension import load_inline
import math

MODEL_PATH = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
NUM_LAYERS = 1
VOCAB_SIZE = 151936
MAX_SEQ_LEN = 2048

device = torch.device('cuda')

# Load model weights
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device)
state = hf_model.state_dict()
embed_weight = state["model.embed_tokens.weight"].contiguous()

# Simple test: embed + write to output, using barrier
cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void launch_test_embed(
    int token_id,
    const void* embed_weight,
    void* output,
    void* sync_var,
    cudaStream_t stream
);

class TestEmbed {
public:
    TestEmbed(torch::Tensor embed_weight) {
        embed_weight_ = embed_weight;
        output_ = torch::empty({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        sync_var_ = torch::empty({1}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
    }

    void run(int token_id) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        launch_test_embed(
            token_id,
            embed_weight_.data_ptr(),
            output_.data_ptr(),
            sync_var_.data_ptr(),
            stream
        );
        cudaStreamSynchronize(stream);
    }

    torch::Tensor get_output() { return output_; }

private:
    torch::Tensor embed_weight_, output_, sync_var_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<TestEmbed>(m, "TestEmbed")
        .def(py::init<torch::Tensor>())
        .def("run", &TestEmbed::run)
        .def("get_output", &TestEmbed::get_output);
}
"""

cuda_src = """
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int HIDDEN_SIZE = 1024;
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr int NUM_BLOCKS = 72;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ void barrier(unsigned int* counter, int num_blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned int my_count = atomicAdd(counter, 1) + 1;
        unsigned int target = ((my_count - 1) / num_blocks + 1) * num_blocks;
        unsigned int current;
        do {
            current = atomicAdd(counter, 0);
        } while (current < target);
    }
    __syncthreads();
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
test_embed_kernel(
    int token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    float* __restrict__ output,
    unsigned int* __restrict__ sync_var
) {
    int num_blocks = gridDim.x;
    int stride = BLOCK_SIZE;
    
    // All blocks load the same embedding
    const __nv_bfloat16* embed_row = embed_weight + token_id * HIDDEN_SIZE;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += stride) {
        output[i] = __bfloat162float(__ldg(embed_row + i));
    }
    
    barrier(sync_var, num_blocks);
    
    // Double the values
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += stride) {
            output[i] = output[i] * 2.0f;
        }
    }
    
    barrier(sync_var, num_blocks);
}

extern "C" void launch_test_embed(
    int token_id,
    const void* embed_weight,
    void* output,
    void* sync_var,
    cudaStream_t stream
) {
    cudaMemsetAsync(sync_var, 0, sizeof(unsigned int), stream);
    test_embed_kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(
        token_id,
        (const __nv_bfloat16*)embed_weight,
        (float*)output,
        (unsigned int*)sync_var
    );
}
"""

print("Compiling minimal test...")
mod = load_inline(
    name="test_embed",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17", "-arch=sm_86"],
    verbose=False,
)
print("Compilation OK.")

# Test
tester = mod.TestEmbed(embed_weight)

# Run twice
tester.run(9707)
out1 = tester.get_output().clone()

tester.run(9707)
out2 = tester.get_output().clone()

diff = (out1 - out2).abs().max().item()
print(f"Max diff between runs: {diff:.6f}")
print(f"Deterministic: {diff < 1e-5}")
print(f"First few values: {out1[:5].tolist()}")
