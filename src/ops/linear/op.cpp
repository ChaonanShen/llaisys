#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t rows, size_t out_features, size_t in_features);
}

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
                   "linear expects out/in/weight to be 2D");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "linear input and weight inner dimensions differ");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0],
                   "linear output shape mismatch");
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0], "linear bias shape mismatch");
        CHECK_ARGUMENT(bias->dtype() == out->dtype(), "linear bias dtype must match output dtype");
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && (!bias || bias->isContiguous()),
           "Linear: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr,
                           out->dtype(), in->shape()[0], weight->shape()[0], in->shape()[1]);
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
