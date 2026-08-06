#include "../../cpu_utils.hpp"

namespace llaisys::ops::cpu {
template <typename T>
void argmax_(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t n) {
    const auto *input = reinterpret_cast<const T *>(vals);
    size_t best = 0;
    float best_value = cpu_detail::to_float(input[0]);
    for (size_t i = 1; i < n; ++i) {
        float value = cpu_detail::to_float(input[i]);
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    *reinterpret_cast<int64_t *>(max_idx) = static_cast<int64_t>(best);
    *reinterpret_cast<T *>(max_val) = cpu_detail::from_float<T>(best_value);
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t n) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(max_idx, max_val, vals, n);
    case LLAISYS_DTYPE_F16:
        return argmax_<fp16_t>(max_idx, max_val, vals, n);
    case LLAISYS_DTYPE_BF16:
        return argmax_<bf16_t>(max_idx, max_val, vals, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
