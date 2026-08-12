#include "../../nvidia/cuda_utils.cuh"
#include <float.h>
namespace llaisys::ops::nvidia {
template <typename T>
__global__ void kernel(int64_t *out_index, T *out_value, const T *values, size_t n) {
    if (threadIdx.x) {
        return;
    }
    size_t best = 0;
    float best_value = detail::load(values, 0);
    for (size_t i = 1; i < n; ++i) {
        float value = detail::load(values, i);
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    *out_index = static_cast<int64_t>(best);
    detail::store(out_value, 0, best_value);
}
template <typename T>
void launch(std::byte *idx, std::byte *value, const std::byte *vals, size_t n) {
    kernel<<<1, 1, 0, detail::stream()>>>(reinterpret_cast<int64_t *>(idx), reinterpret_cast<T *>(value), reinterpret_cast<const T *>(vals), n);
    detail::checkLaunch("argmax kernel");
}
void argmax(std::byte *idx, std::byte *value, const std::byte *vals, llaisysDataType_t dtype, size_t n) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(idx, value, vals, n);
    case LLAISYS_DTYPE_F16:
        return launch<fp16_t>(idx, value, vals, n);
    case LLAISYS_DTYPE_BF16:
        return launch<bf16_t>(idx, value, vals, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
