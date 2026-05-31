"""Run INT8 kernel benchmark standalone."""
import os, sys, time, math, torch

MODEL_PATH = "/mnt/workspace/DSW-GPU/MegaQwen/Qwen3-0.6B"
PROJECT = "/mnt/workspace/DSW-GPU/MegaQwen-A10"
sys.path.insert(0, PROJECT)

from bench_int8 import compile_int8_kernel, quantize_per_channel, make_rope

HIDDEN_SIZE = 1024; INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16; NUM_KV_HEADS = 8
HEAD_DIM = 128; NUM_LAYERS = 28; MAX_SEQ_LEN = 2048

device = torch.device("cuda")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Compile first
mod = compile_int8_kernel()
print("Compilation OK")

cos_t, sin_t = make_rope(MAX_SEQ_LEN, HEAD_DIM, device)

from transformers import AutoModelForCausalLM
print("Loading model...")
hf = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device)
state = hf.state_dict()
del hf

qstate = {}
for i in range(NUM_LAYERS):
    p = f"model.layers.{i}"
    for proj in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        key = f"{p}.{proj}.weight"
        qstate[key] = quantize_per_channel(state[key])
    for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
        key = f"{p}.{proj}.weight"
        qstate[key] = state[key]
    for other in ["input_layernorm.weight", "self_attn.q_norm.weight",
                   "self_attn.k_norm.weight", "post_attention_layernorm.weight"]:
        key = f"{p}.{other}"
        if key in state:
            qstate[key] = state[key]

qstate["lm_head.weight"] = quantize_per_channel(state["lm_head.weight"])
qstate["model.embed_tokens.weight"] = state["model.embed_tokens.weight"]
qstate["model.norm.weight"] = state["model.norm.weight"]
del state
print("Quantization done")

layer_tensors = []
for i in range(NUM_LAYERS):
    p = f"model.layers.{i}"
    layer_tensors.append(qstate[f"{p}.input_layernorm.weight"].contiguous())                     # 0
    for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:                    # 1,2,3
        key = f"{p}.{proj}.weight"
        w = qstate[key]
        if isinstance(w, tuple):
            w = w[0]
        layer_tensors.append(w.contiguous())
    layer_tensors.append(qstate.get(f"{p}.self_attn.q_norm.weight",                              # 4
                                    torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)).contiguous())
    layer_tensors.append(qstate.get(f"{p}.self_attn.k_norm.weight",                              # 5
                                    torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)).contiguous())
    for proj in ["self_attn.o_proj"]:                                                            # 6
        key = f"{p}.{proj}.weight"
        w = qstate[key]
        if isinstance(w, tuple):
            w = w[0]
        layer_tensors.append(w.contiguous())
    layer_tensors.append(qstate[f"{p}.post_attention_layernorm.weight"].contiguous())            # 7
    for proj in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:                               # 8,9,10
        key = f"{p}.{proj}.weight"
        w = qstate[key]
        if isinstance(w, tuple):
            w = w[0]
        layer_tensors.append(w.contiguous())

mlp_scales = []
for i in range(NUM_LAYERS):
    p = f"model.layers.{i}"
    for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        w = qstate[f"{p}.{proj}.weight"]
        mlp_scales.append(w[1].contiguous())

lm_head = qstate["lm_head.weight"]
lm_head_int8 = lm_head[0].contiguous()
lm_head_scale = lm_head[1].contiguous()
final_norm = qstate["model.norm.weight"].contiguous()
embed = qstate["model.embed_tokens.weight"].contiguous()
del qstate
print("Tensors prepared")

torch.cuda.empty_cache()
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

dec = mod.A10Int8MegakernelDecoder(
    embed, layer_tensors, mlp_scales,
    lm_head_int8, lm_head_scale, final_norm,
    cos_t, sin_t, NUM_LAYERS, MAX_SEQ_LEN
)
print("Decoder created")

print("Running benchmark...")
N_WARM = 3; N_RUN = 30; MAX_NEW = 100

for step in range(N_WARM):
    dec.decode_step(0)
torch.cuda.synchronize()
print("Warmup done")

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
print(f"INT8 A10 Kernel: {tok:>7.1f} tok/s, {avg*1e6/MAX_NEW:.0f} us/step")
