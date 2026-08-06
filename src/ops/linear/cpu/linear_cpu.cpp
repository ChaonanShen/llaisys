#include "../../cpu_utils.hpp"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#pragma push_macro("__C")
#undef __C
#include <Accelerate/Accelerate.h>
#pragma pop_macro("__C")
#endif

#include <vector>

namespace llaisys::ops::cpu {
template <typename T>
void linear_naive_(std::byte *out, const std::byte *in, const std::byte *weight,
                   const std::byte *bias, size_t rows, size_t out_features, size_t in_features) {
    auto *result = reinterpret_cast<T *>(out);
    const auto *input = reinterpret_cast<const T *>(in);
    const auto *weights = reinterpret_cast<const T *>(weight);
    const auto *bias_values = reinterpret_cast<const T *>(bias);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t feature = 0; feature < out_features; ++feature) {
            float sum = bias == nullptr ? 0.0F : cpu_detail::to_float(bias_values[feature]);
            for (size_t inner = 0; inner < in_features; ++inner) {
                sum += cpu_detail::to_float(input[row * in_features + inner]) *
                       cpu_detail::to_float(weights[feature * in_features + inner]);
            }
            result[row * out_features + feature] = cpu_detail::from_float<T>(sum);
        }
    }
}

template <typename T>
void linear_blas_(std::byte *out, const std::byte *in, const std::byte *weight,
                  const std::byte *bias, size_t rows, size_t out_features, size_t in_features) {
#if defined(__APPLE__)
    std::vector<float> input(rows * in_features);
    std::vector<float> weights(out_features * in_features);
    std::vector<float> result(rows * out_features);
    const auto *input_values = reinterpret_cast<const T *>(in);
    const auto *weight_values = reinterpret_cast<const T *>(weight);
    for (size_t i = 0; i < input.size(); ++i) input[i] = cpu_detail::to_float(input_values[i]);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = cpu_detail::to_float(weight_values[i]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(rows),
                static_cast<int>(out_features), static_cast<int>(in_features), 1.0F,
                input.data(), static_cast<int>(in_features), weights.data(),
                static_cast<int>(in_features), 0.0F, result.data(), static_cast<int>(out_features));
    const auto *bias_values = reinterpret_cast<const T *>(bias);
    auto *output_values = reinterpret_cast<T *>(out);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t feature = 0; feature < out_features; ++feature) {
            float value = result[row * out_features + feature];
            if (bias != nullptr) value += cpu_detail::to_float(bias_values[feature]);
            output_values[row * out_features + feature] = cpu_detail::from_float<T>(value);
        }
    }
#else
    linear_naive_<T>(out, in, weight, bias, rows, out_features, in_features);
#endif
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t rows, size_t out_features, size_t in_features) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return linear_blas_<float>(out, in, weight, bias, rows, out_features, in_features);
    case LLAISYS_DTYPE_F16: return linear_blas_<fp16_t>(out, in, weight, bias, rows, out_features, in_features);
    case LLAISYS_DTYPE_BF16: return linear_blas_<bf16_t>(out, in, weight, bias, rows, out_features, in_features);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
