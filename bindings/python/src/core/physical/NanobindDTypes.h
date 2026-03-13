#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace nanobind::detail {

template <>
struct dtype_traits<half> {
    static constexpr dlpack::dtype value{(uint8_t)dlpack::dtype_code::Float, 16, 1};
    static constexpr auto name = const_name("float16");
};

template <>
struct dtype_traits<__nv_bfloat16> {
    static constexpr dlpack::dtype value{(uint8_t)dlpack::dtype_code::Bfloat, 16, 1};
    static constexpr auto name = const_name("bfloat16");
};

template <>
struct dtype_traits<__nv_fp8_e4m3> {
    static constexpr dlpack::dtype value{(uint8_t)dlpack::dtype_code::Float8_E4M3FN, 8, 1};
    static constexpr auto name = const_name("float8_e4m3");
};

template <>
struct dtype_traits<__nv_fp8_e5m2> {
    static constexpr dlpack::dtype value{(uint8_t)dlpack::dtype_code::Float8_E5M2, 8, 1};
    static constexpr auto name = const_name("float8_e5m2");
};

}  // namespace nanobind::detail
