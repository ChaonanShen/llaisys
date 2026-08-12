#include "../../nvidia/cuda_utils.cuh"
namespace llaisys::ops::nvidia {
template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = detail::load(gate, i);
        detail::store(out, i, detail::load(up, i) * g / (1.0F + expf(-g)));
    }
}
template <typename T>
void launchSwiglu(std::byte *out, const std::byte *gate, const std::byte *up, size_t n) {
    swigluKernel<<<(n + 255) / 256, 256, 0, detail::stream()>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate), reinterpret_cast<const T *>(up), n);
    detail::checkLaunch("swiglu kernel");
}
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t n) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchSwiglu<float>(out, gate, up, n);
    case LLAISYS_DTYPE_F16:
        return launchSwiglu<fp16_t>(out, gate, up, n);
    case LLAISYS_DTYPE_BF16:
        return launchSwiglu<bf16_t>(out, gate, up, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
