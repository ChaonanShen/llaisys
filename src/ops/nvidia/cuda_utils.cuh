#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops::nvidia::detail {
inline void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

inline cudaStream_t stream() {
    return reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
}

template <typename T>
__device__ float load(const T *values, size_t index) { return static_cast<float>(values[index]); }
template <>
__device__ inline float load<fp16_t>(const fp16_t *values, size_t index) {
    return __half2float(__ushort_as_half(values[index]._v));
}
template <>
__device__ inline float load<bf16_t>(const bf16_t *values, size_t index) {
    return __uint_as_float(static_cast<unsigned int>(values[index]._v) << 16);
}
template <typename T>
__device__ void store(T *values, size_t index, float value) { values[index] = static_cast<T>(value); }
template <>
__device__ inline void store<fp16_t>(fp16_t *values, size_t index, float value) {
    values[index]._v = __half_as_ushort(__float2half_rn(value));
}
template <>
__device__ inline void store<bf16_t>(bf16_t *values, size_t index, float value) {
    const unsigned int bits = __float_as_uint(value);
    values[index]._v = static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
inline void checkLaunch(const char *operation) {
    check(cudaGetLastError(), operation);
}
} // namespace llaisys::ops::nvidia::detail
