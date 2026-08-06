#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t width, float eps);
}

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(eps > 0.0F, "rms_norm epsilon must be positive");
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 1,
                   "rms_norm expects out/in [M,H] and weight [H]");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "rms_norm weight shape mismatch");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RmsNorm: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(),
                             in->shape()[0], in->shape()[1], eps);
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
