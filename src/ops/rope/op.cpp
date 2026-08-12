#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t sequence_length, size_t heads, size_t head_dim, float theta);
}
#ifdef ENABLE_NVIDIA_API
namespace llaisys::ops::nvidia {
void rope(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t, size_t, size_t, float);
}
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(theta > 0.0F, "rope theta must be positive");
    CHECK_ARGUMENT(out->ndim() == 3 && in->ndim() == 3 && pos_ids->ndim() == 1,
                   "rope expects out/in [L,H,D] and pos_ids [L]");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope position ids must have int64 dtype");
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "rope position count mismatch");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "rope head dimension must be even");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(),
                         in->shape()[0], in->shape()[1], in->shape()[2], theta);
    }
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
