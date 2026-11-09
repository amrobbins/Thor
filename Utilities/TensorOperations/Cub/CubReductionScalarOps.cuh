#pragma once

// 1265: the scalar device helpers shared by the dense reducers and the ragged reducers, moved verbatim out of
// CubReductionOperation.cuh (which includes this header, so its users see no change). This header stays lean (no CUB,
// Thrust, cooperative groups or pipeline includes) so the ragged translation units compile quickly
// (plans/Thor_Ragged_Reduction_Plan_2026-11-06.md, phase 2). Everything here is inline, so moving it changes no
// generated code.
#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"

#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ThorImplementation::CubReductionInternal {

struct PropagatingMinimumFp32 {
    __host__ __device__ float operator()(float lhs, float rhs) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // PTX min.NaN.f32 exactly matches Thor's VALUE reduction requirement that any NaN poisons the aggregate,
        // but does so as one native instruction instead of two data-dependent isnan branches plus fminf. The PTX
        // instruction canonicalizes the NaN payload; VALUE reductions require NaN propagation, not payload identity.
        float result;
        asm("min.NaN.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
        return result;
#else
        if (lhs != lhs) {
            return lhs;
        }
        if (rhs != rhs) {
            return rhs;
        }
        return ::fminf(lhs, rhs);
#endif
    }
};

struct PropagatingMaximumFp32 {
    __host__ __device__ float operator()(float lhs, float rhs) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // See PropagatingMinimumFp32. max.NaN.f32 removes the branchy NaN arbitration from the hot reduction chain.
        float result;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
        return result;
#else
        if (lhs != lhs) {
            return lhs;
        }
        if (rhs != rhs) {
            return rhs;
        }
        return ::fmaxf(lhs, rhs);
#endif
    }
};

// Retained FP8 OneWindow and RuntimeK policy: the combine is a compile-time
// specialization. No runtime operation dispatch is generated in the hot loop.
// 0=SUM, 1=PRODUCT, 2=MIN, 3=MAX. FP32 semantics match the staged combine.
template <int FixedOp>
struct Fp8TypedReductionOp {
    static_assert(FixedOp >= 0 && FixedOp <= 3);

    __host__ __device__ float identity() const {
        if constexpr (FixedOp == 0) return 0.0f;
        else if constexpr (FixedOp == 1) return 1.0f;
        else if constexpr (FixedOp == 2) return cuda::std::numeric_limits<float>::infinity();
        else return -cuda::std::numeric_limits<float>::infinity();
    }
    __host__ __device__ float operator()(float a, float b) const {
        if constexpr (FixedOp == 0) return a + b;
        else if constexpr (FixedOp == 1) return a * b;
        else if constexpr (FixedOp == 2) return PropagatingMinimumFp32{}(a, b);
        else return PropagatingMaximumFp32{}(a, b);
    }
};

// P5-F: FP8 MIN and MAX share one compiled kernel (PTX min.NaN.f32) in OneWindow and the RuntimeK PrefixReserved rotated
// first stage. The input transform and the MAX sign are folded into the packed FP8 bytes before decode, one LOP3 per
// four values:
//   MIN identity  bytes & 0xff           MAX identity  bytes ^ 0x80          (max x   = -min(-x))
//   MIN |x|       bytes & 0x7f           MAX |x|       (bytes & 0x7f) ^ 0x80 (max |x| = -min(-|x|))
//   MIN/MAX x^2   as |x|, then the extremum is squared (x^2 is monotone in |x|, and FP8 squares are exact in FP32,
//                 so (max |x|)^2 == max(x^2) bit for bit).
// The fold is undone once per result: Complete kernels square in-kernel and the host negates output_scale for MAX
// (unfoldFp8ExtremumResult); the RuntimeK rotated first stage negates or squares each FP32 partial before it is stored
// (unfoldFp8ExtremumPartial), so staged partials and the continuation are unchanged. FP8 sign flips keep NaN as NaN
// and swap +-Inf and +-0; min.NaN orders -0 < +0, so NaN propagation and signed zeros match a separate MAX kernel.
// SignFold == false folds only the transform (byte AND mask, no XOR operand) for kernels that keep separate MIN and MAX
// specializations: the RuntimeK Exact rotated kernel sits at the 64-register cliff, and the sign XOR mask cost it a
// block per SM (1084, 64 -> 67 registers, 1.47x slower).
struct Fp8ExtremumFold {
    uint32_t byte_and_mask;
    uint32_t byte_xor_mask;
    bool square_result;
};

inline constexpr Fp8ExtremumFold FP8_EXTREMUM_FOLD_NONE{0xffffffffU, 0U, false};

// FixedOp vocabulary of Fp8TypedReductionOp: MIN (2) and MAX (3) launch the MIN kernel.
[[nodiscard]] constexpr bool fp8SharedExtremumOp(int fixed_op) { return fixed_op == 2 || fixed_op == 3; }
[[nodiscard]] constexpr int fp8PhysicalExtremumOp(int fixed_op) { return fixed_op == 3 ? 2 : fixed_op; }

[[nodiscard]] inline Fp8ExtremumFold fp8ExtremumFold(bool maximum, CubReductionStageInputTransform input_transform) {
    bool magnitude = false;
    bool square = false;
    switch (input_transform) {
        case CubReductionStageInputTransform::Identity:
            break;
        case CubReductionStageInputTransform::AbsoluteValue:
            magnitude = true;
            break;
        case CubReductionStageInputTransform::Square:
            magnitude = true;
            square = true;
            break;
        default:
            throw std::logic_error("FP8 MIN/MAX fold received an unsupported input transform.");
    }
    return {magnitude ? 0x7f7f7f7fU : 0xffffffffU, maximum ? 0x80808080U : 0U, square};
}

// Complete kernels: the fold for FixedOp, with the MAX negation moved into output_scale (not needed when squared).
[[nodiscard]] inline Fp8ExtremumFold fp8CompleteExtremumFold(int fixed_op,
                                                             CubReductionStageInputTransform input_transform,
                                                             float& output_scale) {
    if (!fp8SharedExtremumOp(fixed_op)) {
        return FP8_EXTREMUM_FOLD_NONE;
    }
    const Fp8ExtremumFold fold = fp8ExtremumFold(fixed_op == 3, input_transform);
    if (fixed_op == 3 && !fold.square_result) {
        output_scale = -output_scale;
    }
    return fold;
}

// 1179: the same fold for FP32 input, one element per word (retained KR kernels shared with FP8): |x| clears the sign
// bit, MAX flips it, and the MAX negation moves into output_scale unless squared.
[[nodiscard]] inline Fp8ExtremumFold fp32CompleteExtremumFold(int fixed_op,
                                                              CubReductionStageInputTransform input_transform,
                                                              float& output_scale) {
    if (!fp8SharedExtremumOp(fixed_op)) {
        return FP8_EXTREMUM_FOLD_NONE;
    }
    const Fp8ExtremumFold bytes = fp8ExtremumFold(fixed_op == 3, input_transform);
    const bool magnitude = bytes.byte_and_mask != 0xffffffffU;
    const Fp8ExtremumFold fold{magnitude ? 0x7fffffffU : 0xffffffffU, fixed_op == 3 ? 0x80000000U : 0U,
                               bytes.square_result};
    if (fixed_op == 3 && !fold.square_result) {
        output_scale = -output_scale;
    }
    return fold;
}

// 1233: the same fold for FP16 / BF16 input, two elements per word (retained KR kernels shared with FP8 and FP32):
// |x| clears each half-word's sign bit, MAX flips it, and the MAX negation moves into output_scale unless squared.
// Both formats keep the sign in bit 15, so one fold serves both; a sign flip keeps NaN as NaN.
[[nodiscard]] inline Fp8ExtremumFold fp16CompleteExtremumFold(int fixed_op,
                                                              CubReductionStageInputTransform input_transform,
                                                              float& output_scale) {
    if (!fp8SharedExtremumOp(fixed_op)) {
        return FP8_EXTREMUM_FOLD_NONE;
    }
    const Fp8ExtremumFold bytes = fp8ExtremumFold(fixed_op == 3, input_transform);
    const bool magnitude = bytes.byte_and_mask != 0xffffffffU;
    const Fp8ExtremumFold fold{magnitude ? 0x7fff7fffU : 0xffffffffU, fixed_op == 3 ? 0x80008000U : 0U,
                               bytes.square_result};
    if (fixed_op == 3 && !fold.square_result) {
        output_scale = -output_scale;
    }
    return fold;
}

template <bool Fold, bool SignFold = true>
__device__ __forceinline__ uint32_t foldFp8ExtremumBytes(uint32_t packed_fp8, const Fp8ExtremumFold& fold) {
    if constexpr (Fold && !SignFold) {
        return packed_fp8 & fold.byte_and_mask;
    } else if constexpr (Fold) {
        return (packed_fp8 & fold.byte_and_mask) ^ fold.byte_xor_mask;
    } else {
        static_cast<void>(fold);
        return packed_fp8;
    }
}

template <bool Fold>
__device__ __forceinline__ float unfoldFp8ExtremumResult(float value, const Fp8ExtremumFold& fold) {
    if constexpr (Fold) {
        return fold.square_result ? value * value : value;
    } else {
        static_cast<void>(fold);
        return value;
    }
}

template <bool Fold, bool SignFold = true>
__device__ __forceinline__ float unfoldFp8ExtremumPartial(float value, const Fp8ExtremumFold& fold) {
    if constexpr (Fold && !SignFold) {
        return fold.square_result ? value * value : value;
    } else if constexpr (Fold) {
        if (fold.square_result) {
            return value * value;
        }
        return fold.byte_xor_mask != 0U ? -value : value;
    } else {
        static_cast<void>(fold);
        return value;
    }
}

inline __host__ __device__ void storeFp32AsRuntimeDType(void* output,
                                                         DataType output_dtype,
                                                         uint64_t index,
                                                         float value) {
    switch (output_dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            static_cast<__nv_fp8_e4m3*>(output)[index] = ThorLowPrecision::toFp8E4M3Satfinite(value);
            return;
        case DataType::FP8_E5M2:
            static_cast<__nv_fp8_e5m2*>(output)[index] = ThorLowPrecision::toFp8E5M2Satfinite(value);
            return;
#endif
        case DataType::FP16:
            static_cast<__half*>(output)[index] = __float2half_rn(value);
            return;
        case DataType::BF16:
            static_cast<__nv_bfloat16*>(output)[index] = __float2bfloat16_rn(value);
            return;
        case DataType::FP32:
            static_cast<float*>(output)[index] = value;
            return;
        default:
            return;
    }
}

#if THOR_CUB_ENABLE_FP8_TYPES
template <typename InputT>
[[nodiscard]] __device__ __forceinline__ float2 decodeFp8x2ToFp32(uint16_t packed_fp8) {
    static_assert(std::is_same_v<InputT, __nv_fp8_e4m3> || std::is_same_v<InputT, __nv_fp8_e5m2>);

    // Let CUDA own the packed FP8 -> float2 lowering.  In particular, do not force an explicit FP16x2
    // intermediate in Thor source: the native vector type exposes operator float2() directly and gives nvcc
    // freedom to select/schedule the target conversion sequence.
    if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3>) {
        __nv_fp8x2_e4m3 values;
        values.__x = static_cast<__nv_fp8x2_storage_t>(packed_fp8);
        return static_cast<float2>(values);
    } else {
        __nv_fp8x2_e5m2 values;
        values.__x = static_cast<__nv_fp8x2_storage_t>(packed_fp8);
        return static_cast<float2>(values);
    }
}

template <typename InputT>
[[nodiscard]] __device__ __forceinline__ float4 decodeFp8x4ToFp32(uint32_t packed_fp8) {
    const float2 low = decodeFp8x2ToFp32<InputT>(static_cast<uint16_t>(packed_fp8));
    const float2 high = decodeFp8x2ToFp32<InputT>(static_cast<uint16_t>(packed_fp8 >> 16));
    return make_float4(low.x, low.y, high.x, high.y);
}
#endif

}  // namespace ThorImplementation::CubReductionInternal
