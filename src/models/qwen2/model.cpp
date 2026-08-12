#include "model.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace llaisys::models {
namespace {
tensor_t make_tensor(const std::vector<size_t> &shape, llaisysDataType_t dtype,
                     llaisysDeviceType_t device, int device_id) {
    return Tensor::create(shape, dtype, device, device_id);
}
}

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    CHECK_ARGUMENT(device == LLAISYS_DEVICE_CPU || device == LLAISYS_DEVICE_NVIDIA,
                   "Qwen2 supports CPU and NVIDIA devices");
    CHECK_ARGUMENT(meta.hs == meta.nh * meta.dh && meta.nh % meta.nkvh == 0, "invalid Qwen2 head configuration");
    in_embed = make_tensor({meta.voc, meta.hs}, meta.dtype, device, device_id);
    out_embed = make_tensor({meta.voc, meta.hs}, meta.dtype, device, device_id);
    out_norm_w = make_tensor({meta.hs}, meta.dtype, device, device_id);
    const size_t qdim = meta.nh * meta.dh;
    const size_t kvdim = meta.nkvh * meta.dh;
    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        attn_norm_w.push_back(make_tensor({meta.hs}, meta.dtype, device, device_id));
        attn_q_w.push_back(make_tensor({qdim, meta.hs}, meta.dtype, device, device_id));
        attn_q_b.push_back(make_tensor({qdim}, meta.dtype, device, device_id));
        attn_k_w.push_back(make_tensor({kvdim, meta.hs}, meta.dtype, device, device_id));
        attn_k_b.push_back(make_tensor({kvdim}, meta.dtype, device, device_id));
        attn_v_w.push_back(make_tensor({kvdim, meta.hs}, meta.dtype, device, device_id));
        attn_v_b.push_back(make_tensor({kvdim}, meta.dtype, device, device_id));
        attn_o_w.push_back(make_tensor({meta.hs, qdim}, meta.dtype, device, device_id));
        mlp_norm_w.push_back(make_tensor({meta.hs}, meta.dtype, device, device_id));
        mlp_gate_w.push_back(make_tensor({meta.di, meta.hs}, meta.dtype, device, device_id));
        mlp_up_w.push_back(make_tensor({meta.di, meta.hs}, meta.dtype, device, device_id));
        mlp_down_w.push_back(make_tensor({meta.hs, meta.di}, meta.dtype, device, device_id));
    }
    _k_cache.resize(meta.nlayer);
    _v_cache.resize(meta.nlayer);
}

void Qwen2Model::reset() {
    _cache_len = 0;
    std::fill(_k_cache.begin(), _k_cache.end(), nullptr);
    std::fill(_v_cache.begin(), _v_cache.end(), nullptr);
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr && ntoken > 0, "Qwen2 inference needs at least one token");
    CHECK_ARGUMENT(_cache_len + ntoken <= _meta.maxseq, "Qwen2 context exceeds max sequence length");
    const size_t total_length = _cache_len + ntoken;
    const size_t qdim = _meta.nh * _meta.dh;
    const size_t kvdim = _meta.nkvh * _meta.dh;

    auto ids = make_tensor({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    ids->load(token_ids);
    auto x = make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
    ops::embedding(x, ids, in_embed);
    std::vector<int64_t> positions(ntoken);
    for (size_t i = 0; i < ntoken; ++i) positions[i] = static_cast<int64_t>(_cache_len + i);
    auto pos_ids = make_tensor({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    pos_ids->load(positions.data());

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        auto residual = x;
        auto norm = make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::rms_norm(norm, x, attn_norm_w[layer], _meta.epsilon);
        auto q_flat = make_tensor({ntoken, qdim}, _meta.dtype, _device, _device_id);
        auto k_flat = make_tensor({ntoken, kvdim}, _meta.dtype, _device, _device_id);
        auto v_flat = make_tensor({ntoken, kvdim}, _meta.dtype, _device, _device_id);
        ops::linear(q_flat, norm, attn_q_w[layer], attn_q_b[layer]);
        ops::linear(k_flat, norm, attn_k_w[layer], attn_k_b[layer]);
        ops::linear(v_flat, norm, attn_v_w[layer], attn_v_b[layer]);
        auto q = q_flat->view({ntoken, _meta.nh, _meta.dh});
        auto k = k_flat->view({ntoken, _meta.nkvh, _meta.dh});
        auto v = v_flat->view({ntoken, _meta.nkvh, _meta.dh});
        auto q_rotated = make_tensor({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
        auto k_rotated = make_tensor({ntoken, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        ops::rope(q_rotated, q, pos_ids, _meta.theta);
        ops::rope(k_rotated, k, pos_ids, _meta.theta);

        auto next_k = make_tensor({total_length, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        auto next_v = make_tensor({total_length, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        const size_t cached_bytes = _cache_len * kvdim * next_k->elementSize();
        auto runtime_api = llaisys::core::context().runtime().api();
        if (_cache_len != 0) {
            runtime_api->memcpy_sync(next_k->data(), _k_cache[layer]->data(), cached_bytes, LLAISYS_MEMCPY_D2D);
            runtime_api->memcpy_sync(next_v->data(), _v_cache[layer]->data(), cached_bytes, LLAISYS_MEMCPY_D2D);
        }
        const size_t current_bytes = ntoken * kvdim * next_k->elementSize();
        runtime_api->memcpy_sync(next_k->data() + cached_bytes, k_rotated->data(), current_bytes, LLAISYS_MEMCPY_D2D);
        runtime_api->memcpy_sync(next_v->data() + cached_bytes, v->data(), current_bytes, LLAISYS_MEMCPY_D2D);
        _k_cache[layer] = next_k;
        _v_cache[layer] = next_v;

        auto attention = make_tensor({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_id);
        ops::self_attention(attention, q_rotated, next_k, next_v, 1.0F / std::sqrt(static_cast<float>(_meta.dh)));
        auto attention_flat = attention->view({ntoken, qdim});
        auto projected = make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::linear(projected, attention_flat, attn_o_w[layer], nullptr);
        auto after_attention = make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::add(after_attention, residual, projected);

        residual = after_attention;
        norm = make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::rms_norm(norm, after_attention, mlp_norm_w[layer], _meta.epsilon);
        auto gate = make_tensor({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
        auto up = make_tensor({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
        ops::linear(gate, norm, mlp_gate_w[layer], nullptr);
        ops::linear(up, norm, mlp_up_w[layer], nullptr);
        auto activated = make_tensor({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
        ops::swiglu(activated, gate, up);
        auto mlp = make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::linear(mlp, activated, mlp_down_w[layer], nullptr);
        x = make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::add(x, residual, mlp);
    }
    _cache_len = total_length;
    auto norm = make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
    ops::rms_norm(norm, x, out_norm_w, _meta.epsilon);
    auto last = norm->slice(0, ntoken - 1, ntoken);
    auto logits = make_tensor({1, _meta.voc}, _meta.dtype, _device, _device_id);
    ops::linear(logits, last, out_embed, nullptr);
    auto next_id = make_tensor({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto max_value = make_tensor({1}, _meta.dtype, _device, _device_id);
    ops::argmax(next_id, max_value, logits->view({_meta.voc}));
    int64_t result = 0;
    llaisys::core::context().runtime().api()->memcpy_sync(&result, next_id->data(), sizeof(result), LLAISYS_MEMCPY_D2H);
    return result;
}
} // namespace llaisys::models
