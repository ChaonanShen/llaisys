#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <sstream>

namespace {
void check(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

cudaMemcpyKind copyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("unsupported CUDA memcpy kind");
    }
}
} // namespace

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) {
        return 0;
    }
    check(status, "cudaGetDeviceCount");
    return count;
}

void setDevice(int device_id) {
    check(cudaSetDevice(device_id), "cudaSetDevice");
}

void deviceSynchronize() {
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    // A regular stream preserves ordering with synchronous default-stream copies
    // used by the public RuntimeAPI and its Python test harness.
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    check(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
}
void streamSynchronize(llaisysStream_t stream) {
    check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    check(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    check(cudaFree(ptr), "cudaFree");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    check(cudaMallocHost(&ptr, size), "cudaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    check(cudaFreeHost(ptr), "cudaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    // CUDA synchronous copies do not necessarily order with PyTorch's per-thread
    // default stream. RuntimeAPI promises synchronous visibility to callers.
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize before cudaMemcpy");
    check(cudaMemcpy(dst, src, size, copyKind(kind)), "cudaMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    check(cudaMemcpyAsync(dst, src, size, copyKind(kind), reinterpret_cast<cudaStream_t>(stream)), "cudaMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
