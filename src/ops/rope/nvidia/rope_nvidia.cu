#include "../../nvidia/cuda_utils.cuh"
namespace llaisys::ops::nvidia {
template <typename T>
__global__ void kernel(T *out, const T *in, const int64_t *pos, size_t length, size_t heads, size_t dim, float theta) {
    size_t half = dim / 2;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < length * heads * half; i += blockDim.x * gridDim.x) {
        size_t j = i % half, base = (i / half) * dim;
        float angle = static_cast<float>(pos[(i / half) / heads]) / powf(theta, 2.0F * j / dim), a = detail::load(in, base + j), b = detail::load(in, base + half + j);
        detail::store(out, base + j, a * cosf(angle) - b * sinf(angle));
        detail::store(out, base + half + j, b * cosf(angle) + a * sinf(angle));
    }
}
template <typename T>
void launch(std::byte *out, const std::byte *in, const std::byte *pos, size_t l, size_t h, size_t d, float t) {
    size_t n = l * h * d / 2;
    kernel<<<(n + 255) / 256, 256, 0, detail::stream()>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), reinterpret_cast<const int64_t *>(pos), l, h, d, t);
    detail::checkLaunch("rope kernel");
}
void rope(std::byte *out, const std::byte *in, const std::byte *pos, llaisysDataType_t dtype, size_t l, size_t h, size_t d, float t) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, in, pos, l, h, d, t);
    case LLAISYS_DTYPE_F16:
        return launch<fp16_t>(out, in, pos, l, h, d, t);
    case LLAISYS_DTYPE_BF16:
        return launch<bf16_t>(out, in, pos, l, h, d, t);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
