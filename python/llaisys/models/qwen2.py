from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DataType, DeviceType
from ..libllaisys.qwen2 import LlaisysQwen2Meta

from pathlib import Path
import safetensors
import ctypes
import json


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config = json.loads((model_path / "config.json").read_text())
        dtype_name = config["torch_dtype"]
        dtype = {"float32": DataType.F32, "float16": DataType.F16, "bfloat16": DataType.BF16}[dtype_name]
        hidden_size = config["hidden_size"]
        heads = config["num_attention_heads"]
        self._meta = LlaisysQwen2Meta(
            dtype, config["num_hidden_layers"], hidden_size, heads, config["num_key_value_heads"],
            config.get("head_dim", hidden_size // heads), config["intermediate_size"],
            config["max_position_embeddings"], config["vocab_size"], config["rms_norm_eps"],
            config["rope_theta"], config["eos_token_id"],
        )
        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta), device, device_ids, 1
        )
        if not self._model:
            raise RuntimeError("failed to create Qwen2 backend model")
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        def load(name, handle, value):
            if tuple(value.shape) == ():
                raise ValueError(f"unsupported scalar weight: {name}")
            LIB_LLAISYS.tensorLoad(handle, ctypes.c_void_p(value.data_ptr()))

        globals_ = {
            "model.embed_tokens.weight": weights.in_embed,
            "lm_head.weight": weights.out_embed,
            "model.norm.weight": weights.out_norm_w,
        }
        layer_fields = {
            "input_layernorm.weight": weights.attn_norm_w, "self_attn.q_proj.weight": weights.attn_q_w,
            "self_attn.q_proj.bias": weights.attn_q_b, "self_attn.k_proj.weight": weights.attn_k_w,
            "self_attn.k_proj.bias": weights.attn_k_b, "self_attn.v_proj.weight": weights.attn_v_w,
            "self_attn.v_proj.bias": weights.attn_v_b, "self_attn.o_proj.weight": weights.attn_o_w,
            "post_attention_layernorm.weight": weights.mlp_norm_w, "mlp.gate_proj.weight": weights.mlp_gate_w,
            "mlp.up_proj.weight": weights.mlp_up_w, "mlp.down_proj.weight": weights.mlp_down_w,
        }

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as data_:
                for name_ in data_.keys():
                    value = data_.get_tensor(name_)
                    if name_ in globals_:
                        load(name_, globals_[name_], value)
                        continue
                    parts = name_.split(".")
                    if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
                        raise ValueError(f"unknown Qwen2 weight: {name_}")
                    layer = int(parts[2])
                    suffix = ".".join(parts[3:])
                    load(name_, layer_fields[suffix][layer], value)

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        if top_k != 1:
            raise ValueError("Qwen2 backend currently supports greedy generation only (top_k=1)")
        tokens = list(inputs)
        if not tokens:
            raise ValueError("generation requires at least one input token")
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        token_array = (ctypes.c_int64 * len(tokens))(*tokens)
        next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, token_array, len(tokens)))
        for _ in range(max_new_tokens or 0):
            tokens.append(next_token)
            if next_token == self._meta.end_token:
                break
            token_array = (ctypes.c_int64 * 1)(next_token)
            next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, token_array, 1))
        return tokens
