#pragma once

// What the dense kernel files in this directory (RK*.cu, KR.cu) share: the narrow and shallow launch shapes and their
// occupancy query, the FP8 Complete finalizer and output checks, the decoded-packet input transform, the FP16 / BF16
// widening, the runtime-dtype result store, and the FP8 ARG outputs with their stores.

#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Kernels/ReductionOperation.cuh"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgKey.h"

#include <cstdint>
#include <stdexcept>

namespace ThorImplementation::ReductionInternal {

// The grid cap of the OneWindow, PacketComplete and FP8 ARG continuation launches (persistent grids).
constexpr uint64_t PERSISTENT_MAX_GRID_BLOCKS = 65535;

// The shallow-R Complete families (OneWindow and RuntimeK PacketComplete) launch eight-warp CTAs. OneWindow buckets
// runtime K into fixed units; PacketComplete tiles runtime K.
constexpr int SHALLOW_COMPLETE_WARPS = 8;
constexpr int SHALLOW_COMPLETE_THREADS = SHALLOW_COMPLETE_WARPS * TILED_REDUCTION_WARP_THREADS;
static_assert(SHALLOW_COMPLETE_THREADS == 256);

template <typename KernelT>
[[nodiscard]] KernelOccupancyInfo queryShallowCompleteOccupancy(KernelT kernel) {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, SHALLOW_COMPLETE_THREADS, 0));
    return KernelOccupancyInfo{attributes.numRegs, attributes.sharedSizeBytes, 0, active_blocks,
                               active_blocks * SHALLOW_COMPLETE_WARPS};
}

template <typename InputT>
inline constexpr bool isF16InputType =
    std::is_same_v<InputT, __half> || std::is_same_v<InputT, __nv_bfloat16>;

// The additive finalizer of an FP8 Complete pass, from its stage semantics. The input transform applies to each decoded
// value; the finalizer applies once, to the complete FP32 aggregate, before the runtime output conversion.
[[nodiscard]] inline AdditiveFinalizeFp32 fp8CompleteFinalizerFromSemantics(
    const ReductionStageSemantics& semantics) {
    if (semantics.combine != ReductionStageCombine::Sum &&
        semantics.finalize != ReductionStageFinalize::Identity)
        throw std::logic_error("Nonadditive FP8 Complete cannot have an additive finalizer.");
    switch (semantics.input_transform) {
        case ReductionStageInputTransform::Identity:
        case ReductionStageInputTransform::AbsoluteValue:
        case ReductionStageInputTransform::Square:
            break;
        default:
            throw std::logic_error("FP8 Complete received an unsupported input transform.");
    }
    switch (semantics.finalize) {
        case ReductionStageFinalize::Identity:
            return {1.0f, false};
        case ReductionStageFinalize::Divide:
            if (semantics.finalize_divisor == 0) {
                throw std::logic_error("FP8 Complete DIVIDE requires a nonzero original reduction size.");
            }
            return {static_cast<float>(semantics.finalize_divisor), false};
        case ReductionStageFinalize::SquareRoot:
            return {1.0f, true};
        default:
            throw std::logic_error("FP8 Complete received an unsupported finalization mode.");
    }
}

// Transform a decoded packet under one launch-uniform branch, before the
// per-component tail guards. This avoids a separate transform-selection branch
// inside each valid-value guard. OneWindow passes FixedInputTransform=-1 (runtime
// transform operand); RuntimeK packet-complete compiles the three fixed transforms.
template <int FixedInputTransform = -1>
__device__ __forceinline__ float4 transformFp8CompleteDecodedPacket(
    float4 values, ReductionStageInputTransform transform) {
    if constexpr (FixedInputTransform >= 0) {
        static_cast<void>(transform);
    }
    if constexpr (FixedInputTransform == 0) {
        return values;
    } else if constexpr (FixedInputTransform == 1) {
        values.x = ::fabsf(values.x);
        values.y = ::fabsf(values.y);
        values.z = ::fabsf(values.z);
        values.w = ::fabsf(values.w);
    } else if constexpr (FixedInputTransform == 2) {
        values.x *= values.x;
        values.y *= values.y;
        values.z *= values.z;
        values.w *= values.w;
    } else {
        if (transform == ReductionStageInputTransform::AbsoluteValue) {
            values.x = ::fabsf(values.x);
            values.y = ::fabsf(values.y);
            values.z = ::fabsf(values.z);
            values.w = ::fabsf(values.w);
        } else if (transform == ReductionStageInputTransform::Square) {
            values.x *= values.x;
            values.y *= values.y;
            values.z *= values.z;
            values.w *= values.w;
        }
    }
    return values;
}

// The FP8 Complete kernels share one runtime-dtype store. Unsupported destinations are rejected on the host: the device
// store's default arm writes nothing.
[[nodiscard]] inline bool fp8CompleteSupportsOutputStorage(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

// The Complete families share a uniform runtime-dtype store. Finalization and
// scaling happen once, on the final mathematical reduction pass only.
// AdditiveFinalize is FixedOp == 0: fp8CompleteFinalizerFromSemantics() only
// hands PRODUCT/MIN/MAX the identity finalizer, so their kernels compile none.
// SUM keeps the identity test inline and calls the shared out-of-line divide / square-root finalizer only for MEAN and
// the square-root operations (see finalizeFp32Epilogue).
template <bool AdditiveFinalize>
__device__ __forceinline__ void storeFp8CompleteResult(void* output,
                                                        DataType output_dtype,
                                                        uint64_t index,
                                                        float value,
                                                        AdditiveFinalizeFp32 finalizer,
                                                        float output_scale) {
    if constexpr (AdditiveFinalize) {
        value = finalizeFp32Epilogue(finalizer, value);
    } else {
        static_cast<void>(finalizer);
    }
    storeFp32AsRuntimeDType(output, output_dtype, index, value * output_scale);
}

// One FP16 / BF16 element (its 16 bits in the low half of `bits`) widened to FP32, exactly.
template <typename InputT>
__device__ __forceinline__ float widenF16Bits(uint32_t bits) {
    static_assert(isF16InputType<InputT>);
    if constexpr (std::is_same_v<InputT, __half>) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xffffU)));
    } else {
        return __uint_as_float(bits << 16);
    }
}

__device__ __forceinline__ float decodeFp8ArgByte(uint32_t byte, bool e5m2) {
    if (e5m2) {
        __nv_fp8_e5m2 value;
        value.__x = static_cast<__nv_fp8_storage_t>(byte);
        return static_cast<float>(value);
    }
    __nv_fp8_e4m3 value;
    value.__x = static_cast<__nv_fp8_storage_t>(byte);
    return static_cast<float>(value);
}

struct Fp8ArgOutputs {
    void* value;  // null: no value output
    DataType value_dtype;
    void* index;  // null: no index output
    DataType index_dtype;
    uint64_t domain_stride;  // the reduced run's original ARG-domain stride: index = row * domain_stride
    Fp8ArgFold fold;
    bool e5m2;
    // Staged passes may instead write the raw shard-relative lane candidate, 4 bytes per partial (key << 24 | shard
    // row << 1 | negative zero); fp8ArgPackedContinuationKernel finishes the reduction.
    uint32_t* packed = nullptr;
};

// MayPack: only the staged kernels (DeepRSmallK, rotated RuntimeK) test `packed`; Complete kernels (OneWindow,
// PacketComplete) compile without the branch (with it, the OneWindow O1 c16 twin measured 1.12x slower, 63 -> 64
// registers).
template <bool MayPack = false>
__device__ __forceinline__ void storeFp8ArgResult(const Fp8ArgOutputs& outputs, uint64_t output_index,
                                                  uint32_t candidate, uint64_t row_base = 0) {
    if constexpr (MayPack) {
        if (outputs.packed != nullptr) {
            outputs.packed[output_index] = candidate;
            return;
        }
    }
    if (outputs.value != nullptr) {
        const uint32_t byte = fp8ArgValueByte(candidate >> 24, candidate & 1u, outputs.fold);
        storeFp32AsRuntimeDType(outputs.value, outputs.value_dtype, output_index,
                                decodeFp8ArgByte(byte, outputs.e5m2));
    }
    if (outputs.index != nullptr) {
        const uint64_t index =
            (row_base + ((candidate >> 1) & (FP8_ARG_MAX_LANE_ROWS - 1u))) * outputs.domain_stride;
        if (outputs.index_dtype == DataType::UINT32) {
            static_cast<uint32_t*>(outputs.index)[output_index] = static_cast<uint32_t>(index);
        } else {
            static_cast<uint64_t*>(outputs.index)[output_index] = index;
        }
    }
}

// Writes one reduced packed candidate (widened, 55-bit index) to the requested outputs at `output`.
__device__ __forceinline__ void storeFp8ArgPackedResult(const Fp8ArgOutputs& outputs, uint64_t output, uint64_t best) {
    if (outputs.value != nullptr) {
        const uint32_t byte = fp8ArgValueByte(fp8ArgWideKey(best), fp8ArgWideNegativeZero(best), outputs.fold);
        storeFp32AsRuntimeDType(outputs.value, outputs.value_dtype, output, decodeFp8ArgByte(byte, outputs.e5m2));
    }
    if (outputs.index != nullptr) {
        const uint64_t index = fp8ArgWideIndex(best) * outputs.domain_stride;
        if (outputs.index_dtype == DataType::UINT32) {
            static_cast<uint32_t*>(outputs.index)[output] = static_cast<uint32_t>(index);
        } else {
            static_cast<uint64_t*>(outputs.index)[output] = index;
        }
    }
}

}  // namespace ThorImplementation::ReductionInternal
