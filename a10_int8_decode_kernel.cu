/**
 * A10-Optimized Fused Decode Kernel for Qwen3-0.6B
 *
 * Optimizations:
 * 1. 72 blocks → matches A10's 72 SMs for one-wave execution
 * 2. Atomic barrier with release-acquire fences → cross-block sync
 * 3. uint4 (128-bit) loads throughout → maximizes memory bandwidth utilization
 * 4. Direct __ldg MLP gate+up (cp.async disabled due to sm_86 shared mem issue)
 * 5. Shared memory activation caching for MLP → saves redundant global reads
 * 6. Weight prefetch during attention → uses spare blocks to warm L2 for next layer
 * 7. float4 register-cached activations for matvec → reduces instruction count
 * 8. Block-interleaved embedding → eliminates write-after-write race
 */

// config.cuh is prepended by bench_int8.py — do not include here

// =============================================================================
// Embedding Lookup
// =============================================================================

__device__ void embed_lookup(
    cg::grid_group& grid,
    int num_blocks,
    int token_id,
    const __nv_bfloat16* embed_weight,
    __nv_bfloat16* hidden_out
) {
    const __nv_bfloat16* embed_row = embed_weight + token_id * HIDDEN_SIZE;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += gridDim.x * BLOCK_SIZE) {
        hidden_out[i] = __ldg(embed_row + i);
    }
    grid.sync();
}

// =============================================================================
// Fused RMSNorm + QKV Projection (like 3090 version: saves 1 barrier per layer)
// =============================================================================

__device__ void rmsnorm_qkv(
    cg::grid_group& grid,
    int num_blocks,
    const __nv_bfloat16* hidden_in,
    const __nv_bfloat16* norm_weight,
    const int8_t* __restrict__ q_w, const int8_t* __restrict__ k_w, const int8_t* __restrict__ v_w,
    const float* __restrict__ q_scale, const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float* residual_out,
    float* normalized_out,
    float* q_out, float* k_out, float* v_out
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Block 0 does RMSNorm first
    if (block_id == 0) {
        __shared__ float smem[HIDDEN_SIZE];
        __shared__ float smem_red[NUM_WARPS];

        float local_sum = 0.0f;
        int stride = NUM_WARPS * WARP_SIZE;
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
    grid.sync();

    // QKV Projection (INT8 weights)
    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= row_end) continue;

        const int8_t* w_row;
        const float* w_scale;
        float* out_ptr;
        if (m < Q_SIZE) {
            w_row = q_w + m * HIDDEN_SIZE;
            w_scale = q_scale + m;
            out_ptr = q_out + m;
        } else if (m < Q_SIZE + KV_SIZE) {
            w_row = k_w + (m - Q_SIZE) * HIDDEN_SIZE;
            w_scale = k_scale + (m - Q_SIZE);
            out_ptr = k_out + (m - Q_SIZE);
        } else {
            w_row = v_w + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
            w_scale = v_scale + (m - Q_SIZE - KV_SIZE);
            out_ptr = v_out + (m - Q_SIZE - KV_SIZE);
        }

        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(w_row + k));
            int8_t* w_ptr = reinterpret_cast<int8_t*>(&w_u4);
            float4 a1 = *reinterpret_cast<const float4*>(normalized_out + k);
            float4 a2 = *reinterpret_cast<const float4*>(normalized_out + k + 4);
            float4 a3 = *reinterpret_cast<const float4*>(normalized_out + k + 8);
            float4 a4 = *reinterpret_cast<const float4*>(normalized_out + k + 12);
            sum += (float)(w_ptr[0]) * a1.x + (float)(w_ptr[1]) * a1.y
                 + (float)(w_ptr[2]) * a1.z + (float)(w_ptr[3]) * a1.w
                 + (float)(w_ptr[4]) * a2.x + (float)(w_ptr[5]) * a2.y
                 + (float)(w_ptr[6]) * a2.z + (float)(w_ptr[7]) * a2.w
                 + (float)(w_ptr[8]) * a3.x + (float)(w_ptr[9]) * a3.y
                 + (float)(w_ptr[10]) * a3.z + (float)(w_ptr[11]) * a3.w
                 + (float)(w_ptr[12]) * a4.x + (float)(w_ptr[13]) * a4.y
                 + (float)(w_ptr[14]) * a4.z + (float)(w_ptr[15]) * a4.w;
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) *out_ptr = sum * w_scale[0];
    }
    grid.sync();
}

// =============================================================================
// QK Norm + RoPE + KV Cache
// =============================================================================

__device__ void qk_norm_rope_cache(
    cg::grid_group& grid,
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

    grid.sync();
}

// =============================================================================
// Attention with Weight Prefetch
// =============================================================================

__device__ void attention(
    cg::grid_group& grid,
    int num_blocks,
    const float* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    float* attn_out,
    int cache_len, int max_seq_len, float attn_scale,
    const void* o_w, const void* gate_w, const void* up_w
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
                dummy += (float)(__ldg(((const int8_t*)o_w) + s + i));
        } else if (pb < 2 * npb / 3) {
            int a = pb - npb / 3;
            int epp = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (npb / 3 + 1);
            int s = a * epp;
            for (int i = threadIdx.x; i < epp; i += BLOCK_SIZE * 4)
                dummy += (float)(__ldg(((const int8_t*)gate_w) + s + i));
        } else {
            int a = pb - 2 * npb / 3;
            int epp = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (npb / 3 + 1);
            int s = a * epp;
            for (int i = threadIdx.x; i < epp; i += BLOCK_SIZE * 4)
                dummy += (float)(__ldg(((const int8_t*)up_w) + s + i));
        }
        __shared__ float s_dummy;
        if (threadIdx.x == 0) s_dummy = dummy;
        grid.sync();
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

    grid.sync();
}

// =============================================================================
// O Projection + Residual + PostNorm + MLP (pipelined)
// =============================================================================

__device__ void o_proj_postnorm_mlp(
    cg::grid_group& grid,
    int num_blocks,
    int layer,
    const int8_t* o_w, const __nv_bfloat16* pn_w,
    const int8_t* gate_w, const int8_t* up_w, const int8_t* down_w,
    const ScalePointers* layer_scales,
    const float* attn_out,
    float* residual, float* activations, float* mlp_int,
    __nv_bfloat16* hidden_out
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Shared memory for activation cache
    __shared__ float s_act[HIDDEN_SIZE];

    // O Projection + residual (INT8 weights)
    int hp_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hs = block_id * hp_block;
    int he = min(hs + hp_block, HIDDEN_SIZE);

    for (int m_base = hs; m_base < he; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= he) continue;

        const int8_t* o_row = o_w + m * Q_SIZE;
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 16; k < Q_SIZE; k += WARP_SIZE * 16) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(o_row + k));
            int8_t* w_ptr = reinterpret_cast<int8_t*>(&w_u4);
            float4 a1 = *reinterpret_cast<const float4*>(attn_out + k);
            float4 a2 = *reinterpret_cast<const float4*>(attn_out + k + 4);
            float4 a3 = *reinterpret_cast<const float4*>(attn_out + k + 8);
            float4 a4 = *reinterpret_cast<const float4*>(attn_out + k + 12);
            sum += (float)(w_ptr[0]) * a1.x + (float)(w_ptr[1]) * a1.y
                 + (float)(w_ptr[2]) * a1.z + (float)(w_ptr[3]) * a1.w
                 + (float)(w_ptr[4]) * a2.x + (float)(w_ptr[5]) * a2.y
                 + (float)(w_ptr[6]) * a2.z + (float)(w_ptr[7]) * a2.w
                 + (float)(w_ptr[8]) * a3.x + (float)(w_ptr[9]) * a3.y
                 + (float)(w_ptr[10]) * a3.z + (float)(w_ptr[11]) * a3.w
                 + (float)(w_ptr[12]) * a4.x + (float)(w_ptr[13]) * a4.y
                 + (float)(w_ptr[14]) * a4.z + (float)(w_ptr[15]) * a4.w;
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) activations[m] = sum * layer_scales->o_scale[m] + residual[m];
    }
    grid.sync();

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
    grid.sync();

    // Gate + Up: process 2 rows per warp to maximize memory bandwidth reuse
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
        s_act[i] = activations[i];
    __syncthreads();

    int ip_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int i_start = block_id * ip_block;
    int i_end = min(i_start + ip_block, INTERMEDIATE_SIZE);

    for (int gr = i_start + warp_id * 2; gr < i_end; gr += NUM_WARPS * 2) {
        float gs0 = 0.0f, us0 = 0.0f;
        float gs1 = 0.0f, us1 = 0.0f;
        int gr1 = gr + 1;

        #pragma unroll 4
        for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
            float4 a1 = *reinterpret_cast<const float4*>(s_act + k);
            float4 a2 = *reinterpret_cast<const float4*>(s_act + k + 4);
            float4 a3 = *reinterpret_cast<const float4*>(s_act + k + 8);
            float4 a4 = *reinterpret_cast<const float4*>(s_act + k + 12);

            uint4 gu4 = __ldg(reinterpret_cast<const uint4*>(gate_w + gr * HIDDEN_SIZE + k));
            uint4 uu4 = __ldg(reinterpret_cast<const uint4*>(up_w + gr * HIDDEN_SIZE + k));
            int8_t* gp = reinterpret_cast<int8_t*>(&gu4);
            int8_t* up = reinterpret_cast<int8_t*>(&uu4);
            gs0 += (float)(gp[0]) * a1.x + (float)(gp[1]) * a1.y
                 + (float)(gp[2]) * a1.z + (float)(gp[3]) * a1.w
                 + (float)(gp[4]) * a2.x + (float)(gp[5]) * a2.y
                 + (float)(gp[6]) * a2.z + (float)(gp[7]) * a2.w
                 + (float)(gp[8]) * a3.x + (float)(gp[9]) * a3.y
                 + (float)(gp[10]) * a3.z + (float)(gp[11]) * a3.w
                 + (float)(gp[12]) * a4.x + (float)(gp[13]) * a4.y
                 + (float)(gp[14]) * a4.z + (float)(gp[15]) * a4.w;
            us0 += (float)(up[0]) * a1.x + (float)(up[1]) * a1.y
                 + (float)(up[2]) * a1.z + (float)(up[3]) * a1.w
                 + (float)(up[4]) * a2.x + (float)(up[5]) * a2.y
                 + (float)(up[6]) * a2.z + (float)(up[7]) * a2.w
                 + (float)(up[8]) * a3.x + (float)(up[9]) * a3.y
                 + (float)(up[10]) * a3.z + (float)(up[11]) * a3.w
                 + (float)(up[12]) * a4.x + (float)(up[13]) * a4.y
                 + (float)(up[14]) * a4.z + (float)(up[15]) * a4.w;

            if (gr1 < i_end) {
                uint4 gu4_1 = __ldg(reinterpret_cast<const uint4*>(gate_w + gr1 * HIDDEN_SIZE + k));
                uint4 uu4_1 = __ldg(reinterpret_cast<const uint4*>(up_w + gr1 * HIDDEN_SIZE + k));
                int8_t* gp1 = reinterpret_cast<int8_t*>(&gu4_1);
                int8_t* up1 = reinterpret_cast<int8_t*>(&uu4_1);
                gs1 += (float)(gp1[0]) * a1.x + (float)(gp1[1]) * a1.y
                     + (float)(gp1[2]) * a1.z + (float)(gp1[3]) * a1.w
                     + (float)(gp1[4]) * a2.x + (float)(gp1[5]) * a2.y
                     + (float)(gp1[6]) * a2.z + (float)(gp1[7]) * a2.w
                     + (float)(gp1[8]) * a3.x + (float)(gp1[9]) * a3.y
                     + (float)(gp1[10]) * a3.z + (float)(gp1[11]) * a3.w
                     + (float)(gp1[12]) * a4.x + (float)(gp1[13]) * a4.y
                     + (float)(gp1[14]) * a4.z + (float)(gp1[15]) * a4.w;
                us1 += (float)(up1[0]) * a1.x + (float)(up1[1]) * a1.y
                     + (float)(up1[2]) * a1.z + (float)(up1[3]) * a1.w
                     + (float)(up1[4]) * a2.x + (float)(up1[5]) * a2.y
                     + (float)(up1[6]) * a2.z + (float)(up1[7]) * a2.w
                     + (float)(up1[8]) * a3.x + (float)(up1[9]) * a3.y
                     + (float)(up1[10]) * a3.z + (float)(up1[11]) * a3.w
                     + (float)(up1[12]) * a4.x + (float)(up1[13]) * a4.y
                     + (float)(up1[14]) * a4.z + (float)(up1[15]) * a4.w;
            }
        }
        gs0 = warp_reduce_sum(gs0);
        us0 = warp_reduce_sum(us0);
        if (lane_id == 0) mlp_int[gr] = silu(gs0 * layer_scales->gate_scale[gr]) * (us0 * layer_scales->up_scale[gr]);

        if (gr1 < i_end) {
            gs1 = warp_reduce_sum(gs1);
            us1 = warp_reduce_sum(us1);
            if (lane_id == 0) mlp_int[gr1] = silu(gs1 * layer_scales->gate_scale[gr1]) * (us1 * layer_scales->up_scale[gr1]);
        }
    }

    grid.sync();

    // Down projection + residual
    for (int m_base = hs; m_base < he; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= he) continue;

        const int8_t* d_row = down_w + m * INTERMEDIATE_SIZE;
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 16; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 16) {
            uint4 d_u4 = __ldg(reinterpret_cast<const uint4*>(d_row + k));
            int8_t* d_ptr = reinterpret_cast<int8_t*>(&d_u4);
            float4 m1 = *reinterpret_cast<const float4*>(mlp_int + k);
            float4 m2 = *reinterpret_cast<const float4*>(mlp_int + k + 4);
            float4 m3 = *reinterpret_cast<const float4*>(mlp_int + k + 8);
            float4 m4 = *reinterpret_cast<const float4*>(mlp_int + k + 12);
            sum += (float)(d_ptr[0]) * m1.x
                 + (float)(d_ptr[1]) * m1.y
                 + (float)(d_ptr[2]) * m1.z
                 + (float)(d_ptr[3]) * m1.w
                 + (float)(d_ptr[4]) * m2.x
                 + (float)(d_ptr[5]) * m2.y
                 + (float)(d_ptr[6]) * m2.z
                 + (float)(d_ptr[7]) * m2.w
                 + (float)(d_ptr[8]) * m3.x
                 + (float)(d_ptr[9]) * m3.y
                 + (float)(d_ptr[10]) * m3.z
                 + (float)(d_ptr[11]) * m3.w
                 + (float)(d_ptr[12]) * m4.x
                 + (float)(d_ptr[13]) * m4.y
                 + (float)(d_ptr[14]) * m4.z
                 + (float)(d_ptr[15]) * m4.w;
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) hidden_out[m] = __float2bfloat16(sum * layer_scales->down_scale[m] + residual[m]);
    }

    grid.sync();
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
    const ScalePointers* __restrict__ scales,
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
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    cg::grid_group grid = cg::this_grid();
    int num_blocks = gridDim.x;

    embed_lookup(grid, num_blocks,
                 input_token_id, embed_weight,
                 hidden_buffer);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const LayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_cache_layer_stride;

        rmsnorm_qkv(grid, num_blocks,
                    hidden_buffer, w.input_layernorm_weight,
                    (const int8_t*)w.q_proj_weight,
                    (const int8_t*)w.k_proj_weight,
                    (const int8_t*)w.v_proj_weight,
                    scales[layer].q_scale,
                    scales[layer].k_scale,
                    scales[layer].v_scale,
                    g_residual, g_normalized,
                    g_q, g_k, g_v);

        qk_norm_rope_cache(grid, num_blocks,
                           g_q, g_k, g_v,
                           w.q_norm_weight, w.k_norm_weight,
                           cos_table, sin_table,
                           layer_k_cache, layer_v_cache,
                           position, max_seq_len);

        attention(grid, num_blocks,
                  g_q, layer_k_cache, layer_v_cache, g_attn_out,
                  cache_len, max_seq_len, attn_scale,
                  w.o_proj_weight, w.gate_proj_weight, w.up_proj_weight);

        o_proj_postnorm_mlp(grid, num_blocks,
                            layer,
                            (const int8_t*)w.o_proj_weight, w.post_attn_layernorm_weight,
                            (const int8_t*)w.gate_proj_weight,
                            (const int8_t*)w.up_proj_weight,
                            (const int8_t*)w.down_proj_weight,
                            &scales[layer],
                            g_attn_out, g_residual, g_activations, g_mlp_intermediate,
                            hidden_buffer);
    }

    final_rmsnorm(hidden_buffer, final_norm_weight, g_normalized);
}

// =============================================================================
// LM Head Kernels (all-threads __threadfence in barrier)
// =============================================================================

__global__ void lm_head_phase1(
    const float* __restrict__ hidden,
    const int8_t* __restrict__ weight,
    const float* __restrict__ lm_head_scale,
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
        const int8_t* w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(w_row + k));
            int8_t* w_ptr = reinterpret_cast<int8_t*>(&w_u4);
            sum += (float)(w_ptr[0]) * s_hidden[k]
                 + (float)(w_ptr[1]) * s_hidden[k+1]
                 + (float)(w_ptr[2]) * s_hidden[k+2]
                 + (float)(w_ptr[3]) * s_hidden[k+3]
                 + (float)(w_ptr[4]) * s_hidden[k+4]
                 + (float)(w_ptr[5]) * s_hidden[k+5]
                 + (float)(w_ptr[6]) * s_hidden[k+6]
                 + (float)(w_ptr[7]) * s_hidden[k+7]
                 + (float)(w_ptr[8]) * s_hidden[k+8]
                 + (float)(w_ptr[9]) * s_hidden[k+9]
                 + (float)(w_ptr[10]) * s_hidden[k+10]
                 + (float)(w_ptr[11]) * s_hidden[k+11]
                 + (float)(w_ptr[12]) * s_hidden[k+12]
                 + (float)(w_ptr[13]) * s_hidden[k+13]
                 + (float)(w_ptr[14]) * s_hidden[k+14]
                 + (float)(w_ptr[15]) * s_hidden[k+15];
        }
        sum = warp_reduce_sum(sum);
        // Apply per-channel scale
        sum *= lm_head_scale[m];
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

extern "C" void launch_a10_int8_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const float* lm_head_scale,
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
    const ScalePointers* scales,
    cudaStream_t stream
) {
    void* args[] = {
        &input_token_id,
        (void*)&embed_weight,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&scales,
        (void*)&cos_table,
        (void*)&sin_table,
        &k_cache, &v_cache,
        &hidden_buffer,
        &g_activations, &g_residual,
        &g_q, &g_k, &g_v,
        &g_attn_out, &g_mlp_intermediate,
        &g_normalized,
        &num_layers, &position, &cache_len,
        &max_seq_len, (void*)&attn_scale
    };

    dim3 grid(NUM_BLOCKS);
    dim3 block(BLOCK_SIZE);

    cudaError_t err = cudaLaunchCooperativeKernel(
        (const void*)a10_decode_kernel,
        grid, block, args, 0, stream);

    lm_head_phase1<<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const int8_t*)lm_head_weight,
        lm_head_scale,
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
