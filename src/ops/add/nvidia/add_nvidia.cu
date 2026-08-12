#include "../../nvidia/cuda_utils.cuh"
namespace llaisys::ops::nvidia {
template <typename T>
__global__ void addKernel(T *out, const T *a, const T *b, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        detail::store(out, i, detail::load(a, i) + detail::load(b, i));
    }
}
template <typename T>
void launchAdd(std::byte *out, const std::byte *a, const std::byte *b, size_t n) {
    addKernel<<<(n + 255) / 256, 256, 0, detail::stream()>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(a), reinterpret_cast<const T *>(b), n);
    detail::checkLaunch("add kernel");
}
void add(std::byte *out, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t n) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchAdd<float>(out, a, b, n);
    case LLAISYS_DTYPE_F16:
        return launchAdd<fp16_t>(out, a, b, n);
    case LLAISYS_DTYPE_BF16:
        return launchAdd<bf16_t>(out, a, b, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
