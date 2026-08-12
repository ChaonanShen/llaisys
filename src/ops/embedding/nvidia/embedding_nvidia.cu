#include "../../nvidia/cuda_utils.cuh"
namespace llaisys::ops::nvidia {
/* Copy bytes, rather than interpreting values, so every tensor dtype is supported. */
__global__ void byteKernel(std::byte *out, const int64_t *indices, const std::byte *weight, size_t length, size_t vocab, size_t row_bytes) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < length * row_bytes; i += blockDim.x * gridDim.x) {
        size_t row = i / row_bytes;
        int64_t index = indices[row];
        if (index >= 0 && static_cast<size_t>(index) < vocab) {
            out[i] = weight[static_cast<size_t>(index) * row_bytes + i % row_bytes];
        }
    }
}
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t length, size_t vocab, size_t width, size_t element_size) {
    size_t bytes = length * width * element_size;
    byteKernel<<<(bytes + 255) / 256, 256, 0, detail::stream()>>>(out, reinterpret_cast<const int64_t *>(index), weight, length, vocab, width * element_size);
    detail::checkLaunch("embedding kernel");
}
} // namespace llaisys::ops::nvidia
