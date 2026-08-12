#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &strides, size_t element_size);
}
#ifdef ENABLE_NVIDIA_API
namespace llaisys::ops::nvidia {
void rearrange(std::byte *, const std::byte *, const std::vector<size_t> &, const std::vector<ptrdiff_t> &, size_t);
}
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(out->isContiguous(), "rearrange output must be contiguous");
    for (ptrdiff_t stride : in->strides()) {
        CHECK_ARGUMENT(stride >= 0, "rearrange does not support negative strides");
    }
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), in->shape(), in->strides(), out->elementSize());
    }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::rearrange(out->data(), in->data(), in->shape(), in->strides(), out->elementSize());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
