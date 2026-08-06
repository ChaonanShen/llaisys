#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t query_length, size_t key_length,
                    size_t query_heads, size_t key_heads, size_t key_dim, size_t value_dim, float scale);
}

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "self_attention expects 3D tensors");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    const size_t query_length = q->shape()[0];
    const size_t key_length = k->shape()[0];
    const size_t query_heads = q->shape()[1];
    const size_t key_heads = k->shape()[1];
    const size_t key_dim = q->shape()[2];
    const size_t value_dim = v->shape()[2];
    CHECK_ARGUMENT(key_length >= query_length, "self_attention key length must be at least query length");
    CHECK_ARGUMENT(key_heads > 0 && query_heads % key_heads == 0, "self_attention requires an integral GQA head group");
    CHECK_ARGUMENT(k->shape()[2] == key_dim && v->shape()[0] == key_length && v->shape()[1] == key_heads,
                   "self_attention key/value shape mismatch");
    CHECK_ARGUMENT(attn_val->shape()[0] == query_length && attn_val->shape()[1] == query_heads &&
                       attn_val->shape()[2] == value_dim,
                   "self_attention output shape mismatch");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),
                                   query_length, key_length, query_heads, key_heads, key_dim, value_dim, scale);
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
