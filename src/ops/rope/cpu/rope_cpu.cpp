#include "../../cpu_utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
template <typename T>
void rope_(std::byte *out, const std::byte *in, const std::byte *pos_ids,
           size_t sequence_length, size_t heads, size_t head_dim, float theta) {
    auto *result = reinterpret_cast<T *>(out);
    const auto *input = reinterpret_cast<const T *>(in);
    const auto *positions = reinterpret_cast<const int64_t *>(pos_ids);
    const size_t half_dim = head_dim / 2;
    for (size_t token = 0; token < sequence_length; ++token) {
        for (size_t head = 0; head < heads; ++head) {
            size_t base = (token * heads + head) * head_dim;
            for (size_t j = 0; j < half_dim; ++j) {
                float angle = static_cast<float>(positions[token]) /
                              std::pow(theta, 2.0F * static_cast<float>(j) / static_cast<float>(head_dim));
                float a = cpu_detail::to_float(input[base + j]);
                float b = cpu_detail::to_float(input[base + half_dim + j]);
                float sine = std::sin(angle);
                float cosine = std::cos(angle);
                result[base + j] = cpu_detail::from_float<T>(a * cosine - b * sine);
                result[base + half_dim + j] = cpu_detail::from_float<T>(b * cosine + a * sine);
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t sequence_length, size_t heads, size_t head_dim, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return rope_<float>(out, in, pos_ids, sequence_length, heads, head_dim, theta);
    case LLAISYS_DTYPE_F16: return rope_<fp16_t>(out, in, pos_ids, sequence_length, heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16: return rope_<bf16_t>(out, in, pos_ids, sequence_length, heads, head_dim, theta);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
