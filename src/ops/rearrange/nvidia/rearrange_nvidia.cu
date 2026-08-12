#include "../../nvidia/cuda_utils.cuh"

namespace llaisys::ops::nvidia {
constexpr size_t MAX_DIMS = 8;

__global__ void kernel(std::byte *out, const std::byte *in, const size_t *shape,
                       const ptrdiff_t *strides, size_t ndim, size_t element_size, size_t n) {
    for (size_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < n;
         linear += blockDim.x * gridDim.x) {
        size_t remaining = linear;
        ptrdiff_t input_offset = 0;
        for (size_t i = ndim; i > 0; --i) {
            const size_t dim = i - 1;
            const size_t coordinate = remaining % shape[dim];
            remaining /= shape[dim];
            input_offset += static_cast<ptrdiff_t>(coordinate) * strides[dim];
        }
        for (size_t byte = 0; byte < element_size; ++byte) {
            out[linear * element_size + byte] = in[input_offset * static_cast<ptrdiff_t>(element_size) + byte];
        }
    }
}

void rearrange(std::byte *out, const std::byte *in, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &strides, size_t element_size) {
    CHECK_ARGUMENT(shape.size() <= MAX_DIMS, "NVIDIA rearrange supports at most 8 dimensions");
    size_t n = 1;
    for (size_t dimension : shape) n *= dimension;
    size_t *device_shape = nullptr;
    ptrdiff_t *device_strides = nullptr;
    detail::check(cudaMallocAsync(&device_shape, shape.size() * sizeof(size_t), detail::stream()),
                  "cudaMallocAsync rearrange shape");
    detail::check(cudaMallocAsync(&device_strides, strides.size() * sizeof(ptrdiff_t), detail::stream()),
                  "cudaMallocAsync rearrange strides");
    detail::check(cudaMemcpyAsync(device_shape, shape.data(), shape.size() * sizeof(size_t),
                                  cudaMemcpyHostToDevice, detail::stream()),
                  "cudaMemcpyAsync rearrange shape");
    detail::check(cudaMemcpyAsync(device_strides, strides.data(), strides.size() * sizeof(ptrdiff_t),
                                  cudaMemcpyHostToDevice, detail::stream()),
                  "cudaMemcpyAsync rearrange strides");
    kernel<<<(n + 255) / 256, 256, 0, detail::stream()>>>(out, in, device_shape, device_strides, shape.size(), element_size, n);
    detail::check(cudaFreeAsync(device_shape, detail::stream()), "cudaFreeAsync rearrange shape");
    detail::check(cudaFreeAsync(device_strides, detail::stream()), "cudaFreeAsync rearrange strides");
    detail::checkLaunch("rearrange kernel");
}
} // namespace llaisys::ops::nvidia
