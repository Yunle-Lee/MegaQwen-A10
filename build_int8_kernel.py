"""
Build and benchmark INT8 quantized A10 kernel.
Takes the original a10_decode_kernel.cu, mechanically transforms
all GEMV sites to use INT8 weights + on-the-fly conversion + per-channel scales.
"""
import os, sys, re, time, math, json
import torch
from torch.utils.cpp_extension import load_inline

PROJECT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM  # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024
NUM_LAYERS = 28
VOCAB_SIZE = 151936
MAX_SEQ_LEN = 2048
LM_NUM_BLOCKS = 1184

L2P = 8   # lanes per load-bf16 (16 for int8)
L2P_INT8 = 16

# ---------------------------------------------------------------------------
# 1. Quantize weights
# ---------------------------------------------------------------------------
def quantize_weights(state, device):
    """Quantize all linear weight matrices to INT8 per-channel (per-output-row).
    
    Returns:
        qstate: dict with same keys as state, but W are (W_int8, scale) tuples
    """
    qstate = {}
    quant_keys = set()

    # All linear projection weights + embed + lm_head
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            quant_keys.add(f"{p}.{proj}.weight")
    quant_keys.add("model.embed_tokens.weight")
    quant_keys.add("lm_head.weight")

    for key, tensor in state.items():
        if key not in quant_keys:
            qstate[key] = tensor
            continue
        M, N = tensor.shape
        max_abs = tensor.abs().amax(dim=1)  # (M,)
        scale = (max_abs / 127.0).to(torch.float32).clamp(min=1e-12)
        w_int8 = (tensor / scale.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
        qstate[key] = (w_int8, scale)
    return qstate


# ---------------------------------------------------------------------------
# 2. Transform kernel source
# ---------------------------------------------------------------------------
def transform_kernel_source(src: str) -> str:
    """Apply mechanical INT8 transformations to the kernel source."""
    
    # For each GEMV site we need to:
    # 1. Change weight pointer type from const __nv_bfloat16* to const int8_t*
    # 2. Change per-thread load stride (lanes × elements_per_load)
    #     BF16: lane_id * 8, stride = WARP_SIZE * 8
    #     INT8: lane_id * 16, stride = WARP_SIZE * 16  (2x elements per 128b load)
    # 3. Replace __bfloat162float(w_ptr[i]) with float(w_ptr[i]) for int8
    # 4. Add scale factor multiply per output row
    
    # Pointer type changes
    weight_ptrs = [
        "const __nv_bfloat16* embed_row",
        "const __nv_bfloat16* q_w",
        "const __nv_bfloat16* k_w",
        "const __nv_bfloat16* v_w",
        "const __nv_bfloat16* o_w",
        "const __nv_bfloat16* gate_w",
        "const __nv_bfloat16* up_w",
        "const __nv_bfloat16* down_w",
        "const __nv_bfloat16* weight",  # LM head
        # MLP intermediate buffer is FP32, not a weight pointer - skip
    ]
    
    for ptr in weight_ptrs:
        src = src.replace(ptr, ptr.replace("const __nv_bfloat16*", "const int8_t*"))
    
    # Change w_ptr local variable type (used in inner loops as cast target)
    src = src.replace(
        "__nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);",
        "int8_t* w_ptr = reinterpret_cast<int8_t*>(&w_u4);"
    )
    
    # Change lane stride: lane_id * 8 → lane_id * 16 (in the inner loops)
    # These appear in for-loop initializers
    src = re.sub(
        r'for\s*\(int\s+k\s*=\s*lane_id\s*\*\s*8\s*;\s*k\s*<\s*HIDDEN_SIZE\s*;\s*k\s*\+=\s*WARP_SIZE\s*\*\s*8\s*\)',
        'for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16)',
        src
    )
    src = re.sub(
        r'for\s*\(int\s+k\s*=\s*lane_id\s*\*\s*8\s*;\s*k\s*<\s*Q_SIZE\s*;\s*k\s*\+=\s*WARP_SIZE\s*\*\s*8\s*\)',
        'for (int k = lane_id * 16; k < Q_SIZE; k += WARP_SIZE * 16)',
        src
    )
    src = re.sub(
        r'for\s*\(int\s+k\s*=\s*lane_id\s*\*\s*8\s*;\s*k\s*<\s*INTERMEDIATE_SIZE\s*;\s*k\s*\+=\s*WARP_SIZE\s*\*\s*8\s*\)',
        'for (int k = lane_id * 16; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 16)',
        src
    )
    # LM head uses a different loop pattern — check for the specific one
    # LM head loop: for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8)
    # But this might have already been replaced above. Let's replace any remaining.
    src = re.sub(
        r'for\s*\(int\s+k\s*=\s*lane_id\s*\*\s*8\s*;\s*k\s*<\s*HIDDEN_SIZE\s*;\s*k\s*\+=\s*WARP_SIZE\s*\*\s*8\s*\)',
        'for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16)',
        src
    )
    
    # Replace __bfloat162float(w_ptr[i]) → (float)(w_ptr[i])
    # Pattern: __bfloat162float(w_ptr[...])
    src = re.sub(
        r'__bfloat162float\(w_ptr\[(\d+)\]\)',
        r'(float)(w_ptr[\1])',
        src
    )
    # Also handle w_ptr[local_var] cases
    src = re.sub(
        r'__bfloat162float\(w_ptr\[(\w+)\]\)',
        r'(float)(w_ptr[\1])',
        src
    )
    
    # Replace the uint4 load address computation for INT8
    # For BF16: w_row + k (where k indexes BF16 elements, byte offset = k*2)
    # For INT8: w_row + k (where k indexes int8 elements, byte offset = k*1)
    # The stride in terms of bytes:
    #   BF16 load reads 8 BF16 = 16 bytes starting at byte (base + k*2)
    #   INT8 load reads 16 int8 = 16 bytes starting at byte (base + k*1)
    # Since k doubles for INT8 (lane_id * 16 vs lane_id * 8), the byte offset
    #   BF16: (lane_id*8)*2 = lane_id*16 bytes
    #   INT8: (lane_id*16)*1 = lane_id*16 bytes
    # Same! Wait, lane_id*8 for BF16 means byte offset lane_id*8*2 = lane_id*16
    # And lane_id*16 for INT8 means byte offset lane_id*16*1 = lane_id*16
    # ✓ Same byte offset for the starting thread. Good.
    
    # For the uint4 load, the BF16 version reads from w_row + k
    # where k is in element units. For BF16, w_row is __nv_bfloat16*,
    # so w_row + k = byte_base + k*2.
    # For INT8, w_row is int8_t*, so w_row + k = byte_base + k*1.
    # Since k in INT8 version is 2x larger (lane_id*16 vs lane_id*8),
    # the byte offset is the same: (lane_id*8)*2 = lane_id*16 = (lane_id*16)*1 ✓
    
    # The uint4 load expression itself doesn't change: __ldg(w_row + k) still works
    # because the pointer arithmetic handles the element size difference.
    
    # Now we need to add scale factor multiply after warp reduction.
    # Post-reduction stores:
    # Pattern: if (lane_id == 0) hidden[m] = sum;  →  if (lane_id == 0) hidden[m] = sum * scale_name[m];
    # Pattern: if (lane_id == 0) activations[m] = sum;  → similar
    # Pattern: if (lane_id == 0) *out_ptr = sum;  → similar
    
    # These are harder to replace mechanically. Let me handle each site.
    # Actually, let's add an #include and define a helper macro
    # And add scale pointer parameters to the function signature.
    
    # For now, let me modify the approach: instead of modifying the kernel,
    # let me scale the INT8 weights BEFORE storing them, so the dot product
    # naturally gives the correct answer without an extra multiply.
    #
    # Wait, that doesn't work because we need per-channel scaling applied
    # AFTER the dot product, not before. The weight values themselves
    # should be the INT8 quantized values; the scale is applied to the output.
    
    # Hmm wait, actually we CAN avoid the per-channel multiply if we modify
    # the kernel to do it implicitly. But that requires modifying the store sites.
    
    # Let me try a different approach: use a post-processing kernel or add
    # the scale multiply in each site.
    
    return src


def generate_int8_kernel():
    """Read original kernel, transform to INT8, return the transformed CUDA source."""
    with open(os.path.join(PROJECT, "a10_decode_kernel.cu")) as f:
        src = f.read()
    src = transform_kernel_source(src)
    return src


# ---------------------------------------------------------------------------
# 3. Generate C++ wrapper with INT8 support
# ---------------------------------------------------------------------------
def generate_cpp_wrapper():
    """Generate C++ wrapper that accepts INT8 weights + scales."""
    return '''
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

struct LayerScales {
    float* q_scale;
    float* k_scale;
    float* v_scale;
    float* o_scale;
    float* gate_scale;
    float* up_scale;
    float* down_scale;
};

// INT8 kernel launcher
extern "C" void launch_a10_int8_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight, float* embed_scale,
    const LayerWeights* layer_weights,
    const LayerScales* layer_scales,
    const void* final_norm_weight,
    const void* lm_head_weight, float* lm_head_scale,
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
    cudaStream_t stream
);

class A10Int8Decoder {
public:
    A10Int8Decoder(
        torch::Tensor embed_weight, torch::Tensor embed_scale,
        std::vector<torch::Tensor> layer_weights_flat,
        std::vector<torch::Tensor> layer_scales_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight, torch::Tensor lm_head_scale,
        torch::Tensor cos_table, torch::Tensor sin_table,
        int num_layers, int max_seq_len
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len) {
        embed_weight_ = embed_weight;
        embed_scale_ = embed_scale;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        lm_head_scale_ = lm_head_scale;
        cos_table_ = cos_table;
        sin_table_ = sin_table;
        layer_weights_tensors_ = layer_weights_flat;
        layer_scales_tensors_ = layer_scales_flat;

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

            layer_scales_[i].q_scale = (float*)layer_scales_flat[i * 8 + 0].data_ptr();
            layer_scales_[i].k_scale = (float*)layer_scales_flat[i * 8 + 1].data_ptr();
            layer_scales_[i].v_scale = (float*)layer_scales_flat[i * 8 + 2].data_ptr();
            layer_scales_[i].o_scale = (float*)layer_scales_flat[i * 8 + 3].data_ptr();
            layer_scales_[i].gate_scale = (float*)layer_scales_flat[i * 8 + 4].data_ptr();
            layer_scales_[i].up_scale = (float*)layer_scales_flat[i * 8 + 5].data_ptr();
            layer_scales_[i].down_scale = (float*)layer_scales_flat[i * 8 + 6].data_ptr();
        }

        d_layer_weights_ = torch::empty({num_layers * (int)sizeof(LayerWeights)},
                                         torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_weights_.data_ptr(), layer_weights_.data(),
                   num_layers * sizeof(LayerWeights), cudaMemcpyHostToDevice);
        d_layer_scales_ = torch::empty({num_layers * (int)sizeof(LayerScales)},
                                        torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_scales_.data_ptr(), layer_scales_.data(),
                   num_layers * sizeof(LayerScales), cudaMemcpyHostToDevice);

        int kv_heads = 8; int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
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
            embed_weight_.data_ptr(), (float*)embed_scale_.data_ptr(),
            (const LayerWeights*)d_layer_weights_.data_ptr(),
            (const LayerScales*)d_layer_scales_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(), (float*)lm_head_scale_.data_ptr(),
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
    torch::Tensor embed_weight_, embed_scale_;
    torch::Tensor final_norm_weight_, lm_head_weight_, lm_head_scale_;
    torch::Tensor cos_table_, sin_table_;
    torch::Tensor d_layer_weights_, d_layer_scales_;
    std::vector<torch::Tensor> layer_weights_tensors_, layer_scales_tensors_;
    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerScales> layer_scales_;
    torch::Tensor k_cache_, v_cache_;
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;
    torch::Tensor block_max_vals_, block_max_idxs_, output_token_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<A10Int8Decoder>(m, "A10Int8Decoder")
        .def(py::init<torch::Tensor, torch::Tensor,
                      std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step", &A10Int8Decoder::decode_step)
        .def("reset", &A10Int8Decoder::reset)
        .def("position", &A10Int8Decoder::position);
}
'''


# ---------------------------------------------------------------------------
# 4. Load model, quantize, compile, benchmark
# ---------------------------------------------------------------------------
def load_state(device):
    from transformers import AutoModelForCausalLM
    print("Loading model...")
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    state = hf.state_dict()
    return hf, state


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


def extract_weights(qstate, device):
    """Extract weights into the format needed by A10Int8Decoder."""
    embed = qstate["model.embed_tokens.weight"]
    embed_scale = qstate["model.embed_tokens.weight"]

    layer_weight_tensors = []
    layer_scale_tensors = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        # BF16 weights (layernorm, norms — not quantized)
        layer_weight_tensors.append(qstate[f"{p}.input_layernorm.weight"].contiguous())
        # INT8 weights: (W_int8, scale)
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            kw = f"{p}.{proj}.weight"
            w_int8, scale = qstate[kw]
            layer_weight_tensors.append(w_int8.contiguous())
            if proj.startswith("mlp"):
                layer_scale_tensors.append(scale.contiguous())
            else:
                layer_scale_tensors.append(scale.contiguous())
        # QK norms (not quantized)
        qn = qstate.get(f"{p}.self_attn.q_norm.weight", torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device))
        kn = qstate.get(f"{p}.self_attn.k_norm.weight", torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device))
        layer_weight_tensors.append(qn.contiguous())
        layer_weight_tensors.append(kn.contiguous())
        # Post-attn layernorm (not quantized)
        layer_weight_tensors.append(qstate[f"{p}.post_attention_layernorm.weight"].contiguous())

    final_norm = qstate["model.norm.weight"].contiguous()
    lm_head = qstate["lm_head.weight"]
    lm_head_scale = qstate["lm_head.weight"]

    return (embed, embed_scale, layer_weight_tensors, layer_scale_tensors,
            final_norm, lm_head, lm_head_scale)


@torch.no_grad()
def benchmark():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    N_WARM = 5
    N_RUN = 30
    MAX_NEW = 100

    # Load + quantize
    hf, state = load_state(device)
    print("Quantizing weights to INT8...")
    qstate = quantize_weights(state, device)
    del hf; torch.cuda.empty_cache()

    # Build scale arrays: for quantized matmuls, pass scales separately
    # The kernel will handle the rest internally
    # But wait — we need the LAUNCHER function to have the new signature
    # with scale parameters, AND the kernel source must be transformed.

    print("Transforming kernel source to INT8...")
    int8_src = generate_int8_kernel()

    # Write the transformed kernel
    int8_kernel_path = os.path.join(PROJECT, "a10_int8_kernel.cu")
    with open(int8_kernel_path, "w") as f:
        f.write(int8_src)
    print(f"Written: {int8_kernel_path}")

    # Now the challenge: the transformed kernel source references scales
    # that don't exist yet. We need to add scale pointer parameters and
    # the per-row scale multiply.
    #
    # This is getting complex. Let me instead take a simpler approach:
    # modify the kernel to include ADDITIONAL parameters for scales
    # and use them in the store sites.
    #
    # Actually, let me first check if the mechanical transform compiles.
    # If it does, great. If not, I'll add the scale support.

    # For now, compile the original kernel to get a BF16 baseline.
    print("Compiling original kernel for BF16 baseline...")
    from a10_decode import A10Decoder, benchmark as orig_benchmark
    # ... this is getting complex. Let me just run the measurement.

    print("\nFor INT8, we need to modify the kernel to add scale support.")
    print("The mechanical text transformations have been written to a10_int8_kernel.cu")
    print("Next step: add per-channel scale multiply to the kernel.")

    return None


if __name__ == "__main__":
    benchmark()
