#include "../../cpu_utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {
template <typename T>
void self_attention_(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                     size_t query_length, size_t key_length, size_t query_heads, size_t key_heads,
                     size_t key_dim, size_t value_dim, float scale) {
    auto *result = reinterpret_cast<T *>(out);
    const auto *query = reinterpret_cast<const T *>(q);
    const auto *keys = reinterpret_cast<const T *>(k);
    const auto *values = reinterpret_cast<const T *>(v);
    const size_t group = query_heads / key_heads;
    std::vector<float> scores(key_length);
    std::vector<float> accumulator(value_dim);
    for (size_t qi = 0; qi < query_length; ++qi) {
        size_t visible = key_length - query_length + qi + 1;
        for (size_t qh = 0; qh < query_heads; ++qh) {
            size_t kh = qh / group;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t kj = 0; kj < visible; ++kj) {
                float dot = 0.0F;
                for (size_t dim = 0; dim < key_dim; ++dim) {
                    dot += cpu_detail::to_float(query[(qi * query_heads + qh) * key_dim + dim]) *
                           cpu_detail::to_float(keys[(kj * key_heads + kh) * key_dim + dim]);
                }
                scores[kj] = dot * scale;
                if (scores[kj] > max_score) max_score = scores[kj];
            }
            std::fill(accumulator.begin(), accumulator.end(), 0.0F);
            float denominator = 0.0F;
            for (size_t kj = 0; kj < visible; ++kj) {
                float probability = std::exp(scores[kj] - max_score);
                denominator += probability;
                for (size_t dim = 0; dim < value_dim; ++dim) {
                    accumulator[dim] += probability * cpu_detail::to_float(values[(kj * key_heads + kh) * value_dim + dim]);
                }
            }
            for (size_t dim = 0; dim < value_dim; ++dim) {
                result[(qi * query_heads + qh) * value_dim + dim] = cpu_detail::from_float<T>(accumulator[dim] / denominator);
            }
        }
    }
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t query_length, size_t key_length,
                    size_t query_heads, size_t key_heads, size_t key_dim, size_t value_dim, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return self_attention_<float>(out, q, k, v, query_length, key_length, query_heads, key_heads, key_dim, value_dim, scale);
    case LLAISYS_DTYPE_F16: return self_attention_<fp16_t>(out, q, k, v, query_length, key_length, query_heads, key_heads, key_dim, value_dim, scale);
    case LLAISYS_DTYPE_BF16: return self_attention_<bf16_t>(out, q, k, v, query_length, key_length, query_heads, key_heads, key_dim, value_dim, scale);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
