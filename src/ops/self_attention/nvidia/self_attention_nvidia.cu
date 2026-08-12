#include "../../nvidia/cuda_utils.cuh"
#include <float.h>
namespace llaisys::ops::nvidia {
template <typename T>
__global__ void kernel(T *out, const T *q, const T *k, const T *v, size_t qlen, size_t klen, size_t qheads, size_t kheads, size_t kdim, size_t vdim, float scale) {
    size_t work = blockIdx.x;
    size_t qi = work / qheads, qh = work % qheads;
    if (qi >= qlen) {
        return;
    }
    size_t kh = qh / (qheads / kheads), visible = klen - qlen + qi + 1;
    float max_score = -FLT_MAX;
    for (size_t kj = 0; kj < visible; ++kj) {
        float score = 0;
        for (size_t d = 0; d < kdim; ++d) {
            score += detail::load(q, (qi * qheads + qh) * kdim + d) * detail::load(k, (kj * kheads + kh) * kdim + d);
        }
        max_score = fmaxf(max_score, score * scale);
    }
    for (size_t d = threadIdx.x; d < vdim; d += blockDim.x) {
        float total = 0, denom = 0;
        for (size_t kj = 0; kj < visible; ++kj) {
            float score = 0;
            for (size_t kd = 0; kd < kdim; ++kd) {
                score += detail::load(q, (qi * qheads + qh) * kdim + kd) * detail::load(k, (kj * kheads + kh) * kdim + kd);
            }
            float p = expf(score * scale - max_score);
            denom += p;
            total += p * detail::load(v, (kj * kheads + kh) * vdim + d);
        }
        detail::store(out, (qi * qheads + qh) * vdim + d, total / denom);
    }
}
template <typename T>
void launch(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, size_t ql, size_t kl, size_t qh, size_t kh, size_t kd, size_t vd, float s) {
    kernel<<<ql * qh, 256, 0, detail::stream()>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(q), reinterpret_cast<const T *>(k), reinterpret_cast<const T *>(v), ql, kl, qh, kh, kd, vd, s);
    detail::checkLaunch("self_attention kernel");
}
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t dtype, size_t ql, size_t kl, size_t qh, size_t kh, size_t kd, size_t vd, float s) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, q, k, v, ql, kl, qh, kh, kd, vd, s);
    case LLAISYS_DTYPE_F16:
        return launch<fp16_t>(out, q, k, v, ql, kl, qh, kh, kd, vd, s);
    case LLAISYS_DTYPE_BF16:
        return launch<bf16_t>(out, q, k, v, ql, kl, qh, kh, kd, vd, s);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
