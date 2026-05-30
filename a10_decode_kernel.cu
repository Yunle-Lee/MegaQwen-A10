/**
 * A10-Optimized Fused Decode Kernel for Qwen3-0.6B
 *
 * Optimizations:
 * 1. 72 blocks → matches A10's 72 SMs for one-wave execution
 * 2. Atomic barrier → avoids cooperative groups launch constraints
 * 3. uint4 (128-bit) loads throughout → maximizes memory bandwidth utilization
 * 4. cp.async double-buffering for MLP gate+up → overlaps compute with weight fetch
 * 5. Shared memory activation caching for MLP → saves redundant global reads
 * 6. Weight prefetch during attention → uses spare blocks to warm L2 for next layer
 * 7. float4 register-cached activations for matvec → reduces instruction count
 */

#include "config.cuh"

// =============================================================================
// Embedding Lookup
// =============================================================================

__device__ void embed_lookup(
    unsigned int* sync_var,
    int num_blocks,
    int token_id,
    const __nv_bfloat16* embed_weight,
    __nv_bfloat16* hidden_out,
    float* residual_out
) {
    const __nv_bfloat16* embed_row = embed_weight + token_id * HIDDEN_SIZE;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += gridDim.x * BLOCK_SIZE) {
        float v = __bfloat162float(__ldg(embed_row + i));
        hidden_out[i] = __float2bfloat16(v);
        residual_out[i] = v;
    }
    barrier(sync_var, num_blocks);
}

// =============================================================================
// RMSNorm: converts hidden (bf16) → float + saves residual + normalized
// =============================================================================

__device__ void rmsnorm_step(
    unsigned int* sync_var,
    int num_blocks,
    const __nv_bfloat16* hidden_in,
    const __nv_bfloat16* norm_weight,
    float* residual_out,
    float* normalized_out
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Load hidden → residual and compute sum_sq
    if (blockIdx.x == 0) {
        __shared__ float smem[HIDDEN_SIZE];
        __shared__ float smem_red[NUM_WARPS];

        float local_sum = 0.0f;
        int stride = NUM_WARPS * WARP_SIZE;  // 256
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += stride) {
            float v = __bfloat162float(__ldg(hidden_in + i));
            smem[i] = v;
            residual_out[i] = v;
            local_sum += v * v;
        }

        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) smem_red[warp_id] = local_sum;
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < NUM_WARPS) ? smem_red[lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) smem_red[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
        }
        __syncthreads();

        float rstd = smem_red[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += stride) {
            float w = __bfloat162float(__ldg(norm_weight + i));
            normalized_out[i] = smem[i] * rstd * w;
        }
    }
    barrier(sync_var, num_blocks);
}

// =============================================================================
// QKV Projection
// =============================================================================

__device__ void qkv_proj(
    unsigned int* sync_var,
    int num_blocks,
    const float* normalized,
    const __nv_bfloat16* q_w, const __nv_bfloat16* k_w, const __nv_bfloat16* v_w,
    float* q_out, float* k_out, float* v_out
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= row_end) continue;

        const __nv_bfloat16* w_row;
        float* out_ptr;
        if (m < Q_SIZE) {
            w_row = q_w + m * HIDDEN_SIZE;
            out_ptr = q_out + m;
        } else if (m < Q_SIZE + KV_SIZE) {
            w_row = k_w + (m - Q_SIZE) * HIDDEN_SIZE;
            out_ptr = k_out + (m - Q_SIZE);
        } else {
            w_row = v_w + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
            out_ptr = v_out + (m - Q_SIZE - KV_SIZE);
        }

        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
            float4 a1 = *reinterpret_cast<const float4*>(normalized + k);
            float4 a2 = *reinterpret_cast<const float4*>(normalized + k + 4);
            sum += __bfloat162float(w_ptr[0]) * a1.x
                 + __bfloat162float(w_ptr[1]) * a1.y
                 + __bfloat162float(w_ptr[2]) * a1.z
                 + __bfloat162float(w_ptr[3]) * a1.w
                 + __bfloat162float(w_ptr[4]) * a2.x
                 + __bfloat162float(w_ptr[5]) * a2.y
                 + __bfloat162float(w_ptr[6]) * a2.z
                 + __bfloat162float(w_ptr[7]) * a2.w;
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) *out_ptr = sum;
    }

    barrier(sync_var, num_blocks);
}

// =============================================================================
// QK Norm + RoPE + KV Cache
// =============================================================================

__device__ void qk_norm_rope_cache(
    unsigned int* sync_var,
    int num_blocks,
    float* q, float* k, const float* v,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    const __nv_bfloat16* cos_tbl, const __nv_bfloat16* sin_tbl,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int position, int max_seq_len
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_tbl + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_tbl + position * HEAD_DIM;

    // Process Q heads
    {
        int qh_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int qh_start = block_id * qh_per_block;
        int qh_end = min(qh_start + qh_per_block, NUM_Q_HEADS);

        for (int h = qh_start + warp_id; h < qh_end; h += NUM_WARPS) {
            float* qh = q + h * HEAD_DIM;

            float sum_sq = 0.0f;
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
                sum_sq += qh[i] * qh[i];
            sum_sq = warp_reduce_sum(sum_sq);
            float scale = rsqrtf(sum_sq / float(HEAD_DIM) + RMS_EPS);
            scale = __shfl_sync(0xffffffff, scale, 0);

            float qlocal[HEAD_DIM / WARP_SIZE];
            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
                qlocal[j] = qh[i] * scale * __bfloat162float(__ldg(q_norm_w + i));

            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
                float cos_v = __bfloat162float(__ldg(cos_pos + i));
                float sin_v = __bfloat162float(__ldg(sin_pos + i));
                int pair_off = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
                int pair_i = i + pair_off;
                int pair_j = pair_i / WARP_SIZE;
                float pair_v = __shfl_sync(0xffffffff, qlocal[pair_j], pair_i % WARP_SIZE);
                if (i < HEAD_DIM / 2)
                    qh[i] = qlocal[j] * cos_v - pair_v * sin_v;
                else
                    qh[i] = pair_v * sin_v + qlocal[j] * cos_v;
            }
        }
    }

    // Process K heads + cache
    {
        int kh_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
        int kh_start = block_id * kh_per_block;
        int kh_end = min(kh_start + kh_per_block, NUM_KV_HEADS);

        for (int h = kh_start + warp_id; h < kh_end; h += NUM_WARPS) {
            float* kh = k + h * HEAD_DIM;
            const float* vh = v + h * HEAD_DIM;
            __nv_bfloat16* kch = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
            __nv_bfloat16* vch = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

            float sum_sq = 0.0f;
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
                sum_sq += kh[i] * kh[i];
            sum_sq = warp_reduce_sum(sum_sq);
            float scale = rsqrtf(sum_sq / float(HEAD_DIM) + RMS_EPS);
            scale = __shfl_sync(0xffffffff, scale, 0);

            float k_local[HEAD_DIM / WARP_SIZE];
            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
                k_local[j] = kh[i] * scale * __bfloat162float(__ldg(k_norm_w + i));

            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
                float cos_v = __bfloat162float(__ldg(cos_pos + i));
                float sin_v = __bfloat162float(__ldg(sin_pos + i));
                int pair_off = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
                int pair_i = i + pair_off;
                int pair_j = pair_i / WARP_SIZE;
                float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_i % WARP_SIZE);
                float kf;
                if (i < HEAD_DIM / 2)
                    kf = k_local[j] * cos_v - pair_v * sin_v;
                else
                    kf = pair_v * sin_v + k_local[j] * cos_v;
                kh[i] = kf;
                kch[i] = __float2bfloat16(kf);
                vch[i] = __float2bfloat16(vh[i]);
            }
        }
    }

    barrier(sync_var, num_blocks);
}

// =============================================================================
// Attention with Weight Prefetch
// =============================================================================

__device__ void attention(
    unsigned int* sync_var,
    int num_blocks,
    const float* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    float* attn_out,
    int cache_len, int max_seq_len, float attn_scale,
    const __nv_bfloat16* o_w, const __nv_bfloat16* gate_w, const __nv_bfloat16* up_w
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    constexpr int ATTN_BLOCKS = NUM_Q_HEADS;  // 16 blocks do attention

    if (block_id >= ATTN_BLOCKS) {
        int pb = block_id - ATTN_BLOCKS;
        int npb = num_blocks - ATTN_BLOCKS;
        float dummy = 0.0f;
        if (pb < npb / 3) {
            int epp = (Q_SIZE * HIDDEN_SIZE) / (npb / 3 + 1);
            int s = pb * epp;
            for (int i = threadIdx.x; i < epp; i += BLOCK_SIZE * 4)
                dummy += __bfloat162float(__ldg(o_w + s + i));
        } else if (pb < 2 * npb / 3) {
            int a = pb - npb / 3;
            int epp = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (npb / 3 + 1);
            int s = a * epp;
            for (int i = threadIdx.x; i < epp; i += BLOCK_SIZE * 4)
                dummy += __bfloat162float(__ldg(gate_w + s + i));
        } else {
            int a = pb - 2 * npb / 3;
            int epp = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (npb / 3 + 1);
            int s = a * epp;
            for (int i = threadIdx.x; i < epp; i += BLOCK_SIZE * 4)
                dummy += __bfloat162float(__ldg(up_w + s + i));
        }
        __shared__ float s_dummy;
        if (threadIdx.x == 0) s_dummy = dummy;
        barrier(sync_var, num_blocks);
        return;
    }

    __shared__ float s_max_score[NUM_WARPS];
    __shared__ float s_sum_exp[NUM_WARPS];
    __shared__ float s_out_acc[NUM_WARPS][HEAD_DIM];

    int hpb = (NUM_Q_HEADS + ATTN_BLOCKS - 1) / ATTN_BLOCKS;
    int hs = block_id * hpb;
    int he = min(hs + hpb, NUM_Q_HEADS);

    for (int qh = hs; qh < he; qh++) {
        int kvh = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        const float* qh_ptr = q + qh * HEAD_DIM;
        float* oh = attn_out + qh * HEAD_DIM;

        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int pos = warp_id; pos < cache_len; pos += NUM_WARPS) {
            const __nv_bfloat16* kp = k_cache + kvh * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16* vp = v_cache + kvh * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            float score = 0.0f;
            #pragma unroll
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE)
                score += qh_ptr[d] * __bfloat162float(__ldg(kp + d));
            score = warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);

            float weight = expf(score - max_score);
            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                out_acc[j] = out_acc[j] * exp_diff + weight * __bfloat162float(__ldg(vp + d));
        }

        if (lane_id == 0) {
            s_max_score[warp_id] = max_score;
            s_sum_exp[warp_id] = sum_exp;
        }
        #pragma unroll
        for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
            s_out_acc[warp_id][d] = out_acc[j];
        __syncthreads();

        if (warp_id == 0) {
            float global_max = s_max_score[0];
            for (int w = 1; w < NUM_WARPS; w++)
                if (s_max_score[w] > -INFINITY)
                    global_max = fmaxf(global_max, s_max_score[w]);

            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int w = 0; w < NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    float sw = expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * sw;
                    #pragma unroll
                    for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                        final_out[j] += s_out_acc[w][d] * sw;
                }
            }
            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                oh[d] = final_out[j] / total_sum_exp;
        }
        __syncthreads();
    }

    barrier(sync_var, num_blocks);
}

// =============================================================================
// O Projection + Residual + PostNorm + MLP (pipelined)
// =============================================================================

__device__ void o_proj_postnorm_mlp(
    unsigned int* sync_var,
    int num_blocks,
    const __nv_bfloat16* o_w, const __nv_bfloat16* pn_w,
    const __nv_bfloat16* gate_w, const __nv_bfloat16* up_w, const __nv_bfloat16* down_w,
    const float* attn_out,
    float* residual, float* activations, float* mlp_int,
    __nv_bfloat16* hidden_out
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Shared memory for activation cache
    __shared__ float s_act[HIDDEN_SIZE];

    // O Projection + residual
    int hp_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hs = block_id * hp_block;
    int he = min(hs + hp_block, HIDDEN_SIZE);

    for (int m_base = hs; m_base < he; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= he) continue;

        const __nv_bfloat16* o_row = o_w + m * Q_SIZE;
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(o_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
            float4 a1 = *reinterpret_cast<const float4*>(attn_out + k);
            float4 a2 = *reinterpret_cast<const float4*>(attn_out + k + 4);
            sum += __bfloat162float(w_ptr[0]) * a1.x
                 + __bfloat162float(w_ptr[1]) * a1.y
                 + __bfloat162float(w_ptr[2]) * a1.z
                 + __bfloat162float(w_ptr[3]) * a1.w
                 + __bfloat162float(w_ptr[4]) * a2.x
                 + __bfloat162float(w_ptr[5]) * a2.y
                 + __bfloat162float(w_ptr[6]) * a2.z
                 + __bfloat162float(w_ptr[7]) * a2.w;
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) activations[m] = sum + residual[m];
    }
    barrier(sync_var, num_blocks);

    // Post-attention RMSNorm (block 0)
    if (block_id == 0) {
        __shared__ float smem_r[NUM_WARPS];
        float local_sum = 0.0f;
        for (int i = threadIdx.x * 4; i < HIDDEN_SIZE; i += BLOCK_SIZE * 4) {
            float4 v = *reinterpret_cast<const float4*>(activations + i);
            *reinterpret_cast<float4*>(residual + i) = v;
            local_sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) smem_r[warp_id] = local_sum;
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < NUM_WARPS) ? smem_r[lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) smem_r[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_r[0];
        for (int i = threadIdx.x * 8; i < HIDDEN_SIZE; i += BLOCK_SIZE * 8) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(pn_w + i));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
            float4 r1 = *reinterpret_cast<const float4*>(residual + i);
            float4 r2 = *reinterpret_cast<const float4*>(residual + i + 4);
            float4 o1, o2;
            o1.x = r1.x * rstd * __bfloat162float(w_ptr[0]);
            o1.y = r1.y * rstd * __bfloat162float(w_ptr[1]);
            o1.z = r1.z * rstd * __bfloat162float(w_ptr[2]);
            o1.w = r1.w * rstd * __bfloat162float(w_ptr[3]);
            o2.x = r2.x * rstd * __bfloat162float(w_ptr[4]);
            o2.y = r2.y * rstd * __bfloat162float(w_ptr[5]);
            o2.z = r2.z * rstd * __bfloat162float(w_ptr[6]);
            o2.w = r2.w * rstd * __bfloat162float(w_ptr[7]);
            *reinterpret_cast<float4*>(activations + i) = o1;
            *reinterpret_cast<float4*>(activations + i + 4) = o2;
        }
    }
    barrier(sync_var, num_blocks);

    // Gate + Up with direct global memory loads (no cp.async)
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
        s_act[i] = activations[i];
    __syncthreads();

    int ip_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int i_start = block_id * ip_block;
    int i_end = min(i_start + ip_block, INTERMEDIATE_SIZE);

    for (int gr = i_start + warp_id; gr < i_end; gr += NUM_WARPS) {
        float gs = 0.0f, us = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint4 gu4 = __ldg(reinterpret_cast<const uint4*>(gate_w + gr * HIDDEN_SIZE + k));
            uint4 uu4 = __ldg(reinterpret_cast<const uint4*>(up_w + gr * HIDDEN_SIZE + k));
            __nv_bfloat16* gp = reinterpret_cast<__nv_bfloat16*>(&gu4);
            __nv_bfloat16* up = reinterpret_cast<__nv_bfloat16*>(&uu4);
            float4 a1 = *reinterpret_cast<const float4*>(s_act + k);
            float4 a2 = *reinterpret_cast<const float4*>(s_act + k + 4);
            gs += __bfloat162float(gp[0]) * a1.x + __bfloat162float(gp[1]) * a1.y
                + __bfloat162float(gp[2]) * a1.z + __bfloat162float(gp[3]) * a1.w
                + __bfloat162float(gp[4]) * a2.x + __bfloat162float(gp[5]) * a2.y
                + __bfloat162float(gp[6]) * a2.z + __bfloat162float(gp[7]) * a2.w;
            us += __bfloat162float(up[0]) * a1.x + __bfloat162float(up[1]) * a1.y
                + __bfloat162float(up[2]) * a1.z + __bfloat162float(up[3]) * a1.w
                + __bfloat162float(up[4]) * a2.x + __bfloat162float(up[5]) * a2.y
                + __bfloat162float(up[6]) * a2.z + __bfloat162float(up[7]) * a2.w;
        }
        gs = warp_reduce_sum(gs);
        us = warp_reduce_sum(us);
        if (lane_id == 0) mlp_int[gr] = silu(gs) * us;
    }

    barrier(sync_var, num_blocks);

    // Down projection + residual
    for (int m_base = hs; m_base < he; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= he) continue;

        const __nv_bfloat16* d_row = down_w + m * INTERMEDIATE_SIZE;
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
            uint4 d_u4 = __ldg(reinterpret_cast<const uint4*>(d_row + k));
            __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u4);
            float4 m1 = *reinterpret_cast<const float4*>(mlp_int + k);
            float4 m2 = *reinterpret_cast<const float4*>(mlp_int + k + 4);
            sum += __bfloat162float(d_ptr[0]) * m1.x
                 + __bfloat162float(d_ptr[1]) * m1.y
                 + __bfloat162float(d_ptr[2]) * m1.z
                 + __bfloat162float(d_ptr[3]) * m1.w
                 + __bfloat162float(d_ptr[4]) * m2.x
                 + __bfloat162float(d_ptr[5]) * m2.y
                 + __bfloat162float(d_ptr[6]) * m2.z
                 + __bfloat162float(d_ptr[7]) * m2.w;
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) hidden_out[m] = __float2bfloat16(sum + residual[m]);
    }

    barrier(sync_var, num_blocks);
}

// =============================================================================
// Final RMSNorm
// =============================================================================

__device__ void final_rmsnorm(
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* norm_weight,
    float* normalized
) {
    if (blockIdx.x != 0) return;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_r[NUM_WARPS];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = __bfloat162float(hidden[i]);
        normalized[i] = v;
        local_sum += v * v;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) smem_r[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_r[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) smem_r[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_r[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = __bfloat162float(__ldg(norm_weight + i));
        normalized[i] = normalized[i] * rstd * w;
    }
}

// =============================================================================
// Main Decode Kernel
// =============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
a10_decode_kernel(
    int input_token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const LayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ hidden_buffer,
    float* __restrict__ g_activations,
    float* __restrict__ g_residual,
    float* __restrict__ g_q,
    float* __restrict__ g_k,
    float* __restrict__ g_v,
    float* __restrict__ g_attn_out,
    float* __restrict__ g_mlp_intermediate,
    float* __restrict__ g_normalized,
    unsigned int* __restrict__ global_sync_var,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    int num_blocks = gridDim.x;

    // Embedding lookup → hidden_buffer (bf16) + g_residual (float)
    embed_lookup(global_sync_var, num_blocks,
                 input_token_id, embed_weight,
                 hidden_buffer, g_residual);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const LayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_cache_layer_stride;

        // RMSNorm hidden_buffer → g_normalized (float), save residual
        rmsnorm_step(global_sync_var, num_blocks,
                     hidden_buffer, w.input_layernorm_weight,
                     g_residual, g_normalized);

        // QKV projection
        qkv_proj(global_sync_var, num_blocks,
                 g_normalized,
                 w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
                 g_q, g_k, g_v);

        // QK norm + RoPE + KV cache write
        qk_norm_rope_cache(global_sync_var, num_blocks,
                           g_q, g_k, g_v,
                           w.q_norm_weight, w.k_norm_weight,
                           cos_table, sin_table,
                           layer_k_cache, layer_v_cache,
                           position, max_seq_len);

        // Attention (with weight prefetch for next ops)
        attention(global_sync_var, num_blocks,
                  g_q, layer_k_cache, layer_v_cache, g_attn_out,
                  cache_len, max_seq_len, attn_scale,
                  w.o_proj_weight, w.gate_proj_weight, w.up_proj_weight);

        // O projection + PostNorm + MLP (pipelined)
        o_proj_postnorm_mlp(global_sync_var, num_blocks,
                            w.o_proj_weight, w.post_attn_layernorm_weight,
                            w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
                            g_attn_out, g_residual, g_activations, g_mlp_intermediate,
                            hidden_buffer);
    }

    // Final RMSNorm
    final_rmsnorm(hidden_buffer, final_norm_weight, g_normalized);
}

// =============================================================================
// LM Head Kernels (all-threads __threadfence in barrier)
// =============================================================================

__global__ void lm_head_phase1(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LM_BLOCK_SIZE)
        s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rpb = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb;
    int re = min(rs + rpb, VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = rs + warp_id; m < re; m += LM_BLOCK_SIZE / WARP_SIZE) {
        const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);
            sum += __bfloat162float(w_ptr[0]) * s_hidden[k]
                 + __bfloat162float(w_ptr[1]) * s_hidden[k+1]
                 + __bfloat162float(w_ptr[2]) * s_hidden[k+2]
                 + __bfloat162float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float w_max[LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int w_idx[LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        w_max[warp_id] = local_max;
        w_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float mv = (lane_id < LM_BLOCK_SIZE / WARP_SIZE) ? w_max[lane_id] : -INFINITY;
        int mi = (lane_id < LM_BLOCK_SIZE / WARP_SIZE) ? w_idx[lane_id] : -1;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float ov = __shfl_down_sync(0xffffffff, mv, offset);
            int oi = __shfl_down_sync(0xffffffff, mi, offset);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = mv;
            block_max_idxs[blockIdx.x] = mi;
        }
    }
}

__global__ void lm_head_phase2(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks_in
) {
    __shared__ float s_mv[1024];
    __shared__ int s_mi[1024];

    int tid = threadIdx.x;
    float lm = -INFINITY;
    int li = -1;
    for (int i = tid; i < num_blocks_in; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > lm) { lm = v; li = block_max_idxs[i]; }
    }
    s_mv[tid] = lm;
    s_mi[tid] = li;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_mv[tid + s] > s_mv[tid]) {
                s_mv[tid] = s_mv[tid + s];
                s_mi[tid] = s_mi[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *output_token = s_mi[0];
}

// =============================================================================
// Launch Function
// =============================================================================

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
) {
    cudaMemsetAsync(global_sync_var, 0, sizeof(unsigned int), stream);

    a10_decode_kernel<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(
        input_token_id,
        (const __nv_bfloat16*)embed_weight,
        layer_weights,
        (const __nv_bfloat16*)final_norm_weight,
        (const __nv_bfloat16*)cos_table,
        (const __nv_bfloat16*)sin_table,
        (__nv_bfloat16*)k_cache,
        (__nv_bfloat16*)v_cache,
        (__nv_bfloat16*)hidden_buffer,
        (float*)g_activations,
        (float*)g_residual,
        (float*)g_q,
        (float*)g_k,
        (float*)g_v,
        (float*)g_attn_out,
        (float*)g_mlp_intermediate,
        (float*)g_normalized,
        (unsigned int*)global_sync_var,
        num_layers, position, cache_len, max_seq_len, attn_scale
    );

    lm_head_phase1<<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LM_NUM_BLOCKS
    );
}
