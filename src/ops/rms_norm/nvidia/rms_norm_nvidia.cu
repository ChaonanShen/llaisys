#include "../../nvidia/cuda_utils.cuh"
namespace llaisys::ops::nvidia {
template <typename T>
__global__ void kernel(T *out, const T *in, const T *weight, size_t width, float eps) {
    size_t row = blockIdx.x;
    float sum = 0;
    for (size_t i = threadIdx.x; i < width; i += blockDim.x) {
        float v = detail::load(in, row * width + i);
        sum += v * v;
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x / 2; n; n >>= 1) {
        if (threadIdx.x < n) {
            sums[threadIdx.x] += sums[threadIdx.x + n];
        }
        __syncthreads();
    }
    float scale = rsqrtf(sums[0] / width + eps);
    for (size_t i = threadIdx.x; i < width; i += blockDim.x) {
        detail::store(out, row * width + i, detail::load(in, row * width + i) * scale * detail::load(weight, i));
    }
}
template <typename T>
void launch(std::byte *out, const std::byte *in, const std::byte *weight, size_t rows, size_t width, float eps) {
    kernel<<<rows, 256, 0, detail::stream()>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), reinterpret_cast<const T *>(weight), width, eps);
    detail::checkLaunch("rms_norm kernel");
}
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t dtype, size_t rows, size_t width, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, in, weight, rows, width, eps);
    case LLAISYS_DTYPE_F16:
        return launch<fp16_t>(out, in, weight, rows, width, eps);
    case LLAISYS_DTYPE_BF16:
        return launch<bf16_t>(out, in, weight, rows, width, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
