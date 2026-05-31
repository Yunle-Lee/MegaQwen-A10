#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// =============================================================================
// A10-Optimized Tunable Parameters
// =============================================================================

// A10 has 72 SMs → 1 block per SM for single-wave
constexpr int NUM_BLOCKS = 72;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;
constexpr int WARP_SIZE = 32;

// MLP double-buffering: 8 rows per tile (one per warp)
constexpr int MLP_TILE_ROWS = 4;
constexpr int MLP_ROW_SIZE = 1024;

// LM head
constexpr int LM_NUM_BLOCKS = 1184;
constexpr int LM_BLOCK_SIZE = 256;
constexpr int VOCAB_SIZE = 151936;

// =============================================================================
// Model Dimensions (Qwen3-0.6B)
// =============================================================================

constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM;
constexpr float RMS_EPS = 1e-6f;

// Number of sync points: embed(1) + layers(5) + final(1) = 7 for 1-layer, 143 for 28
constexpr int MAX_SYNC_POINTS = 256;

struct LayerWeights {
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
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

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Atomic barrier with generation tracking – matches original atomic_barrier_v2
__device__ __forceinline__ void barrier(unsigned int* counter, int num_blocks) {
    __syncthreads();
    __threadfence();  // RELEASE: all threads flush prior stores globally
    if (threadIdx.x == 0) {
        unsigned int my_count = atomicAdd(counter, 1) + 1;
        unsigned int my_gen = (my_count - 1) / num_blocks;
        unsigned int target = (my_gen + 1) * num_blocks;
        unsigned int current;
        do {
            current = atomicAdd(counter, 0);
        } while (current < target);
    }
    __syncthreads();
    __threadfence();  // ACQUIRE: all threads see previously released stores
}

// Async cp.async load for weight tiles
__device__ __forceinline__ void async_load_tile(
    __nv_bfloat16* smem_dst,
    const __nv_bfloat16* gmem_base,
    int row_start, int num_rows, int row_size, int max_rows
) {
    constexpr int CHUNK_BYTES = 16;
    constexpr int CHUNK_ELEMS = 8;
    int chunks_per_row = row_size / CHUNK_ELEMS;
    int total_chunks = num_rows * chunks_per_row;
    for (int c = threadIdx.x; c < total_chunks; c += BLOCK_SIZE) {
        int row_in_tile = c / chunks_per_row;
        int chunk_in_row = c % chunks_per_row;
        int global_row = row_start + row_in_tile;
        if (global_row < max_rows) {
            const void* src = gmem_base + global_row * row_size + chunk_in_row * CHUNK_ELEMS;
            void* dst = smem_dst + row_in_tile * row_size + chunk_in_row * CHUNK_ELEMS;
            __pipeline_memcpy_async(dst, src, CHUNK_BYTES);
        }
    }
}
