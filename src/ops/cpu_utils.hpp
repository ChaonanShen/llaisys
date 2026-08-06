#pragma once

#include "../utils.hpp"

#include <type_traits>

namespace llaisys::ops::cpu_detail {
template <typename T>
float to_float(T value) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<float>(value);
    } else {
        return value;
    }
}

template <typename T>
T from_float(float value) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<T>(value);
    } else {
        return value;
    }
}
} // namespace llaisys::ops::cpu_detail
