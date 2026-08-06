#include "../../../utils.hpp"

#include <cstring>
#include <functional>
#include <numeric>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &strides, size_t element_size) {
    for (size_t linear = 0; linear < std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>()); ++linear) {
        size_t remaining = linear;
        ptrdiff_t input_offset = 0;
        for (size_t i = shape.size(); i > 0; --i) {
            size_t dim = i - 1;
            size_t coordinate = remaining % shape[dim];
            remaining /= shape[dim];
            input_offset += static_cast<ptrdiff_t>(coordinate) * strides[dim];
        }
        std::memcpy(out + linear * element_size, in + input_offset * static_cast<ptrdiff_t>(element_size), element_size);
    }
}
} // namespace llaisys::ops::cpu
