#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models {
class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);

    void reset();
    int64_t infer(const int64_t *token_ids, size_t ntoken);

    tensor_t in_embed, out_embed, out_norm_w;
    std::vector<tensor_t> attn_norm_w, attn_q_w, attn_q_b, attn_k_w, attn_k_b, attn_v_w, attn_v_b,
        attn_o_w, mlp_norm_w, mlp_gate_w, mlp_up_w, mlp_down_w;

private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    size_t _cache_len = 0;
    std::vector<tensor_t> _k_cache, _v_cache;
};
} // namespace llaisys::models
