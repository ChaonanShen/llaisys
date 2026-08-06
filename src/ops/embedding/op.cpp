#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               size_t length, size_t vocab, size_t width, size_t element_size);
}

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1 && weight->ndim() == 2 && out->ndim() == 2,
                   "embedding expects index [L], weight [V,H], and out [L,H]");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding index must have int64 dtype");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "embedding output dtype must match weight dtype");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0] && out->shape()[1] == weight->shape()[1],
                   "embedding output shape mismatch");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), index->shape()[0],
                              weight->shape()[0], weight->shape()[1], out->elementSize());
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
