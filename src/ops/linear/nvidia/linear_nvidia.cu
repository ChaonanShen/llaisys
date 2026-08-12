#include "../../nvidia/cuda_utils.cuh"
#include <cublas_v2.h>

namespace llaisys::ops::nvidia {
template <typename T>
__global__ void kernel(T *out, const T *in, const T *weight, const T *bias, size_t rows, size_t out_features, size_t in_features) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * out_features) {
        return;
    }
    size_t row = index / out_features, col = index % out_features;
    float sum = bias ? detail::load(bias, col) : 0.0F;
    for (size_t k = 0; k < in_features; ++k) {
        sum += detail::load(in, row * in_features + k) * detail::load(weight, col * in_features + k);
    }
    detail::store(out, index, sum);
}
template <typename T>
__global__ void biasKernel(T *values, const T *bias_values, size_t count, size_t columns) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        detail::store(values, index, detail::load(values, index) + detail::load(bias_values, index % columns));
    }
}
template <typename T>
void launch(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t rows, size_t o, size_t k) {
    size_t n = rows * o;
    kernel<<<(n + 255) / 256, 256, 0, detail::stream()>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(bias), rows, o, k);
    detail::checkLaunch("linear kernel");
}
inline void checkCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}
template <typename T>
void addBias(std::byte *out, const std::byte *bias, size_t n, size_t width) {
    if (bias == nullptr) {
        return;
    }
    biasKernel<<<(n + 255) / 256, 256, 0, detail::stream()>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(bias), n, width);
    detail::checkLaunch("linear bias kernel");
}
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype, size_t rows, size_t o, size_t k) {
    if (dtype == LLAISYS_DTYPE_BF16) {
        return launch<bf16_t>(out, in, weight, bias, rows, o, k);
    }
    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate");
    checkCublas(cublasSetStream(handle, detail::stream()), "cublasSetStream");
    const float alpha = 1.0F, beta = 0.0F;
    cublasStatus_t status;
    if (dtype == LLAISYS_DTYPE_F32) {
        status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(o), static_cast<int>(rows), static_cast<int>(k),
                             &alpha, reinterpret_cast<const float *>(weight), static_cast<int>(k), reinterpret_cast<const float *>(in),
                             static_cast<int>(k), &beta, reinterpret_cast<float *>(out), static_cast<int>(o));
        checkCublas(status, "cublasSgemm");
        addBias<float>(out, bias, rows * o, o);
    } else if (dtype == LLAISYS_DTYPE_F16) {
        status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(o), static_cast<int>(rows), static_cast<int>(k),
                              &alpha, weight, CUDA_R_16F, static_cast<int>(k), in, CUDA_R_16F, static_cast<int>(k), &beta,
                              out, CUDA_R_16F, static_cast<int>(o), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        checkCublas(status, "cublasGemmEx");
        addBias<fp16_t>(out, bias, rows * o, o);
    } else {
        cublasDestroy(handle);
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    checkCublas(cublasDestroy(handle), "cublasDestroy");
}
} // namespace llaisys::ops::nvidia
