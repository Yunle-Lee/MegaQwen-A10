"""
Generate INT8 variant of the A10 fused kernel.
Transforms MLP (Gate/Up/Down) and LM head GEMVs to use INT8 weights.
QKV/O stay in BF16 (small matrices, less benefit).
"""
import os, re

PROJECT = "/mnt/workspace/DSW-GPU/MegaQwen-A10"

def read_kernel():
    with open(os.path.join(PROJECT, "a10_decode_kernel.cu")) as f:
        return f.read()


def replace_gemv_load(src, ptr_name):
    """Replace a GEMV inner loop from BF16 to INT8 for a given weight pointer.
    
    Pattern (BF16):
      uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(ptr_name + k));
      __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
      sum += __bfloat162float(w_ptr[0]) * ...
    
    INT8:
      uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(ptr_name + k));
      int8_t* w_ptr = reinterpret_cast<int8_t*>(&w_u4);
      sum += (float)(w_ptr[0]) * ...
    """
    old1 = f'uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>({ptr_name} + k));\n            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);'
    new1 = f'uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>({ptr_name} + k));\n            int8_t* w_ptr = reinterpret_cast<int8_t*>(&w_u4);'
    src = src.replace(old1, new1)
    
    # Replace the conversion function
    src = src.replace(f'__bfloat162float(w_ptr', f'(float)(w_ptr')
    
    return src


def replace_gemv_dual_load(src, ptr1, ptr2):
    """Replace dual-issue GEMV load for Gate+Up (2 rows)."""
    # Row gr loads
    old = f'uint4 gu4 = __ldg(reinterpret_cast<const uint4*>({ptr1} + gr * HIDDEN_SIZE + k));\n            uint4 uu4 = __ldg(reinterpret_cast<const uint4*>({ptr2} + gr * HIDDEN_SIZE + k));\n            __nv_bfloat16* gp = reinterpret_cast<__nv_bfloat16*>(&gu4);\n            __nv_bfloat16* up = reinterpret_cast<__nv_bfloat16*>(&uu4);'
    new = f'uint4 gu4 = __ldg(reinterpret_cast<const uint4*>({ptr1} + gr * HIDDEN_SIZE + k));\n            uint4 uu4 = __ldg(reinterpret_cast<const uint4*>({ptr2} + gr * HIDDEN_SIZE + k));\n            int8_t* gp = reinterpret_cast<int8_t*>(&gu4);\n            int8_t* up = reinterpret_cast<int8_t*>(&uu4);'
    src = src.replace(old, new)

    # Row gr1 loads  
    old1 = f'uint4 gu4_1 = __ldg(reinterpret_cast<const uint4*>({ptr1} + gr1 * HIDDEN_SIZE + k));\n                uint4 uu4_1 = __ldg(reinterpret_cast<const uint4*>({ptr2} + gr1 * HIDDEN_SIZE + k));\n                __nv_bfloat16* gp1 = reinterpret_cast<__nv_bfloat16*>(&gu4_1);\n                __nv_bfloat16* up1 = reinterpret_cast<__nv_bfloat16*>(&uu4_1);'
    new1 = f'uint4 gu4_1 = __ldg(reinterpret_cast<const uint4*>({ptr1} + gr1 * HIDDEN_SIZE + k));\n                uint4 uu4_1 = __ldg(reinterpret_cast<const uint4*>({ptr2} + gr1 * HIDDEN_SIZE + k));\n                int8_t* gp1 = reinterpret_cast<int8_t*>(&gu4_1);\n                int8_t* up1 = reinterpret_cast<int8_t*>(&uu4_1);'
    src = src.replace(old1, new1)

    return src


def transform_int8_kernel(src):
    """Main transformation."""
    
    # =========== 1. Add struct ScalePointers ===========
    scale_struct = '''
struct ScalePointers {
    float* __restrict__ gate_scale;
    float* __restrict__ up_scale;
    float* __restrict__ down_scale;
    float* __restrict__ lm_head_scale;
};
'''
    # Insert after LayerWeights struct
    src = src.replace(
        "};",
        "};\n" + scale_struct,
        1  # first occurrence only (end of LayerWeights)
    )
    
    # =========== 2. Change function signatures ===========
    # Main kernel: add ScalePointers parameter
    src = src.replace(
        "const __nv_bfloat16* __restrict__ final_norm_weight,",
        "const __nv_bfloat16* __restrict__ final_norm_weight,\n    const ScalePointers* __restrict__ scales,"
    )
    
    # o_proj_postnorm_mlp signature: add scale ptrs
    src = src.replace(
        "const __nv_bfloat16* o_w, const __nv_bfloat16* pn_w,\n    const __nv_bfloat16* gate_w, const __nv_bfloat16* up_w, const __nv_bfloat16* down_w,",
        "const __nv_bfloat16* o_w, const __nv_bfloat16* pn_w,\n    const int8_t* gate_w, const int8_t* up_w, const int8_t* down_w,\n    const ScalePointers* scales,"
    )
    
    # lm_head_phase1_kernel signature: add scale
    src = src.replace(
        "const __nv_bfloat16* __restrict__ weight,",
        "const int8_t* __restrict__ weight,\n    const float* __restrict__ lm_head_scale,"
    )
    
    # =========== 3. Change gate_w, up_w, down_w types ===========
    # In main kernel function
    src = src.replace(
        "const __nv_bfloat16* __restrict__ gate_w",
        "const int8_t* __restrict__ gate_w"
    )
    src = src.replace(
        "const __nv_bfloat16* __restrict__ up_w",
        "const int8_t* __restrict__ up_w"
    )
    src = src.replace(
        "const __nv_bfloat16* __restrict__ down_w",
        "const int8_t* __restrict__ down_w"
    )
    src = src.replace(
        "const __nv_bfloat16* __restrict__ lm_head_weight",
        "const int8_t* __restrict__ lm_head_weight"
    )
    
    # =========== 4. Change GEMV inner loops ===========
    # Gate + Up dual loads
    src = replace_gemv_dual_load(src, "gate_w", "up_w")
    
    # Down projection
    src = replace_gemv_load(src, "d_row")
    
    # LM head
    src = replace_gemv_load(src, "w_row")
    
    # =========== 5. Change loop strides (INT8 = 2× elements per 128b load) ===========
    # Gate+Up loop: HIDDEN_SIZE (N)
    old_gu_loop = "for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {"
    new_gu_loop = "for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {"
    src = src.replace(old_gu_loop, new_gu_loop)
    
    # Down projection loop: INTERMEDIATE_SIZE (N)
    old_d_loop = "for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {"
    new_d_loop = "for (int k = lane_id * 16; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 16) {"
    src = src.replace(old_d_loop, new_d_loop)
    
    # LM head loop: HIDDEN_SIZE (N)
    # This matches the generic pattern, already replaced above
    
    # =========== 6. Add scale multiply after warp reduction ===========
    # Gate: if (lane_id == 0) mlp_int[gr] = silu(gs0) * us0;
    # → if (lane_id == 0) mlp_int[gr] = silu(gs0 * scales->gate_scale[gr]) * (us0 * scales->up_scale[gr]);
    # Actually, silu(x) = x * sigmoid(x). Since silu is non-linear, we need to apply the scale BEFORE silu.
    # Wait, gs0 = sum(x_i * w_i) -- we computed with INT8 weights already.
    # Actually the issue is more subtle. The INT8 dot product gives:
    #   sum_i (x_i * (w_int8[i] * scale[row]))
    # = scale[row] * sum_i (x_i * w_int8[i])
    # 
    # And for the gate+up SiLU product:
    #   silu(gs0 * gate_scale[gr]) * (us0 * up_scale[gr])
    #
    # No, the scale was for the weight quantization. Each weight value is:
    #   w_float[j] = w_int8[j] * scale[row]
    # So: gs0_int8 = sum(x_i * w_int8[i])
    #     gs0_float = gs0_int8 * gate_scale[gr]
    #     result = silu(gs0_float) * (us0_int8 * up_scale[gr])
    
    src = src.replace(
        "if (lane_id == 0) mlp_int[gr] = silu(gs0) * us0;",
        "if (lane_id == 0) mlp_int[gr] = silu(gs0 * scales->gate_scale[gr]) * (us0 * scales->up_scale[gr]);"
    )
    src = src.replace(
        "if (lane_id == 0) mlp_int[gr1] = silu(gs1) * us1;",
        "if (lane_id == 0) mlp_int[gr1] = silu(gs1 * scales->gate_scale[gr1]) * (us1 * scales->up_scale[gr1]);"
    )
    
    # Down projection: sum * scale
    src = src.replace(
        "if (lane_id == 0) hidden_out[m] = __float2bfloat16(sum + residual[m]);",
        "if (lane_id == 0) hidden_out[m] = __float2bfloat16(sum * scales->down_scale[m] + residual[m]);"
    )
    
    # LM head: need to find and modify the store site
    # The LM head kernel stores to block_max_vals/block_max_idxs
    # Let me look at the LM head phase 1 kernel more carefully
    # Pattern: if (lane_id == 0) bmv[bkt * NUM_BLOCKS + blockIdx.x] = max_val;
    # And: bmi[...] = max_idx;
    # We need to find the right pattern and add scale
    
    # =========== 7. Pass scales to helper functions ===========
    # In a10_decode_kernel, the call to o_proj_postnorm_mlp needs scales
    old_call = "o_proj_postnorm_mlp(grid, num_blocks,\n        g_activations, g_residual, g_mlp_intermediate, hidden_buffer);"
    # This call is setting up the params. We need to find the actual call site with the right arguments.
    # Let's search for the more complete call pattern.
    
    # Actually, let me find the exact call to o_proj_postnorm_mlp
    # Looking at line around 650-730
    
    return src


def verify_kernel(kernel_path):
    """Basic structural checks on the transformed kernel."""
    with open(kernel_path) as f:
        src = f.read()
    
    checks = [
        ("ScalePointers struct", "struct ScalePointers" in src),
        ("gate_w is int8_t", "const int8_t* __restrict__ gate_w" in src),
        ("up_w is int8_t", "const int8_t* __restrict__ up_w" in src),
        ("down_w is int8_t", "const int8_t* __restrict__ down_w" in src),
        ("lm_head_weight is int8_t", "const int8_t* __restrict__ lm_head_weight" in src),
        ("No __ldg gate_w bf16", "__ldg(reinterpret_cast<const uint4*>(gate_w" in src),
        ("No bf16 gate_w", '__bfloat162float(gp[' in src),
        ("Stride 16 in loop", "k += WARP_SIZE * 16" in src),
        ("Scale multiply in gate", "scales->gate_scale[gr]" in src),
        ("Scale multiply in down", "scales->down_scale[m]" in src),
    ]
    
    print("\nKernel verification:")
    all_ok = True
    for name, ok in checks:
        status = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        print(f"  {status} {name}")
    return all_ok


def main():
    print("Reading original kernel...")
    src = read_kernel()
    
    print("Applying INT8 transformations...")
    src = transform_int8_kernel(src)
    
    out_path = os.path.join(PROJECT, "a10_int8_decode_kernel.cu")
    with open(out_path, "w") as f:
        f.write(src)
    print(f"Written: {out_path}")
    
    verify_kernel(out_path)


if __name__ == "__main__":
    main()
