#include "../../cpu_utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
template <typename T>
void swiglu_(std::byte *out, const std::byte *gate, const std::byte *up, size_t n) {
    auto *result = reinterpret_cast<T *>(out);
    const auto *gate_values = reinterpret_cast<const T *>(gate);
    const auto *up_values = reinterpret_cast<const T *>(up);
    for (size_t i = 0; i < n; ++i) {
        float g = cpu_detail::to_float(gate_values[i]);
        result[i] = cpu_detail::from_float<T>(cpu_detail::to_float(up_values[i]) * g / (1.0F + std::exp(-g)));
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t n) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return swiglu_<float>(out, gate, up, n);
    case LLAISYS_DTYPE_F16: return swiglu_<fp16_t>(out, gate, up, n);
    case LLAISYS_DTYPE_BF16: return swiglu_<bf16_t>(out, gate, up, n);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
