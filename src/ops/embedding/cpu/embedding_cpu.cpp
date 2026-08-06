#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t length, size_t vocab, size_t width, size_t element_size) {
    const auto *indices = reinterpret_cast<const int64_t *>(index);
    const size_t row_bytes = width * element_size;
    for (size_t row = 0; row < length; ++row) {
        int64_t id = indices[row];
        CHECK_ARGUMENT(id >= 0 && static_cast<size_t>(id) < vocab, "embedding index out of range");
        std::memcpy(out + row * row_bytes, weight + static_cast<size_t>(id) * row_bytes, row_bytes);
    }
}
} // namespace llaisys::ops::cpu
