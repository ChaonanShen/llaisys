#include "../../cpu_utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
template <typename T>
void rms_norm_(std::byte *out, const std::byte *in, const std::byte *weight, size_t rows, size_t width, float eps) {
    auto *result = reinterpret_cast<T *>(out);
    const auto *input = reinterpret_cast<const T *>(in);
    const auto *weights = reinterpret_cast<const T *>(weight);
    for (size_t row = 0; row < rows; ++row) {
        float sum_squares = 0.0F;
        for (size_t col = 0; col < width; ++col) {
            float value = cpu_detail::to_float(input[row * width + col]);
            sum_squares += value * value;
        }
        float scale = 1.0F / std::sqrt(sum_squares / static_cast<float>(width) + eps);
        for (size_t col = 0; col < width; ++col) {
            result[row * width + col] = cpu_detail::from_float<T>(
                cpu_detail::to_float(input[row * width + col]) * scale * cpu_detail::to_float(weights[col]));
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t width, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return rms_norm_<float>(out, in, weight, rows, width, eps);
    case LLAISYS_DTYPE_F16: return rms_norm_<fp16_t>(out, in, weight, rows, width, eps);
    case LLAISYS_DTYPE_BF16: return rms_norm_<bf16_t>(out, in, weight, rows, width, eps);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
