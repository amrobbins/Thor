#pragma once

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Reduction/Kernels/ScalarOps.cuh"
#include "Utilities/TensorOperations/Reduction/Reducer.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StageSemantics.h"

#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ThorImplementation::ReductionInternal {

template <typename Fn>
decltype(auto) dispatchReductionInputDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            return fn.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return fn.template operator()<__nv_fp8_e5m2>();
#endif
        case DataType::FP16:
            return fn.template operator()<__half>();
        case DataType::BF16:
            return fn.template operator()<__nv_bfloat16>();
        case DataType::FP32:
            return fn.template operator()<float>();
        default:
            throw std::invalid_argument("Unsupported tensor reduction input dtype value "
                                        + std::to_string(static_cast<int>(dtype)) + ".");
    }
}

template <typename T>
struct ToFp32 {
    __host__ __device__ float operator()(T value) const { return static_cast<float>(value); }
};

template <>
struct ToFp32<__half> {
    __host__ __device__ float operator()(__half value) const { return __half2float(value); }
};

template <>
struct ToFp32<__nv_bfloat16> {
    __host__ __device__ float operator()(__nv_bfloat16 value) const { return __bfloat162float(value); }
};

struct IdentityFp32 {
    __host__ __device__ float operator()(float value) const { return value; }
};

struct AbsoluteValueFp32 {
    __host__ __device__ float operator()(float value) const { return ::fabsf(value); }
};

struct SquareFp32 {
    __host__ __device__ float operator()(float value) const { return value * value; }
};

struct AdditiveFinalizeFp32 {
    float divisor;
    bool square_root;

    // Sum, mean, sum-squares, and L2 intentionally share this finalizer type so optional division/square-root
    // semantics do not multiply the complete additive reduction kernel matrix. These choices are paid once per
    // final aggregate rather than in the input transform or reduction loop.
    __host__ __device__ float operator()(float value) const {
        float finalized = value;
        if (divisor == 0.0f) {
            finalized = 0.0f;
        } else if (divisor != 1.0f) {
            finalized /= divisor;
        }
        return square_root ? ::sqrtf(finalized) : finalized;
    }
};

// One out-of-line additive finalizer for the FP8 Complete epilogues.
//
// nvcc hoists the launch-uniform output-dtype switch out of unrolled store loops, so an inline AdditiveFinalizeFp32
// was copied once per (store slot x output dtype): every copy carried the IEEE divide and sqrt fast paths and its own
// calls to the __cuda_sm3x_div_rn / __cuda_sm20_sqrt_rn slowpath subroutines, which were in turn duplicated in every
// kernel (a P8/CTA64 AlignedK Complete kernel had 48 such copies in about 31 KB of its 40 KB of SASS). Identity
// finalization (SUM, and every nonadditive operation) stays inline as one uniform test; MEAN and the square-root
// operations call this single noinline copy per translation unit (the Cub objects are not relocatable device code).
// It runs exactly AdditiveFinalizeFp32's arithmetic, so results are bit-identical.
inline __device__ __noinline__ float finalizeAdditiveFp32OutOfLine(float value, float divisor, bool square_root) {
    return AdditiveFinalizeFp32{divisor, square_root}(value);
}

// Finalize one complete FP32 aggregate before scaling and output conversion. Other finalizer types are applied as is.
template <typename OutputFinalizeT>
__device__ __forceinline__ float finalizeFp32Epilogue(const OutputFinalizeT& finalize, float value) {
    return finalize(value);
}

__device__ __forceinline__ float finalizeFp32Epilogue(const AdditiveFinalizeFp32& finalize, float value) {
    if (finalize.divisor == 1.0f && !finalize.square_root) {
        return value;
    }
    return finalizeAdditiveFp32OutOfLine(value, finalize.divisor, finalize.square_root);
}

// PropagatingMinimum / MaximumFp32, Fp8TypedReductionOp, the extremum folds, storeFp32AsRuntimeDType and the FP8
// decodes live in ScalarOps.cuh (included above), so lean translation units (the ragged reducers) can use them without
// this header's CUB and Thrust includes.

inline __host__ __device__ uint64_t tiledReductionOutputIndex(uint64_t outer_index,
                                                              uint64_t inner_index,
                                                              uint64_t output_outer_stride,
                                                              uint64_t output_inner_stride) {
    // The supported output layouts are natural dense [outer,inner] (inner stride 1) or dense [inner,outer] retained
    // order (outer stride 1). Keep both common cases to one multiply plus one uniform branch rather than paying two
    // general stride multiplies at every final store.
    if (output_inner_stride == 1) {
        return outer_index * output_outer_stride + inner_index;
    }
    if (output_outer_stride == 1) {
        return inner_index * output_inner_stride + outer_index;
    }
    return outer_index * output_outer_stride + inner_index * output_inner_stride;
}

// The warp width of the RK kernels.
constexpr int TILED_REDUCTION_WARP_THREADS = 32;

[[nodiscard]] inline __host__ __device__ uint64_t ceilDivideU64(uint64_t numerator, uint64_t denominator) {
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

[[nodiscard]] inline __host__ __device__ uint64_t minU64(uint64_t lhs, uint64_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

// Division-free reducer work grids.
//
// A sharded reducer stage owns a (shard, component tile, outer) work space. The launch maps it directly onto the grid:
// shards on x, component tiles on y, outer rows on z. Shards stay on the fastest-dispatched axis so the blocks resident
// at any moment keep walking consecutive rows of one outer slab (shard-fastest dispatch order). x has the 2^31-1
// hardware limit and y/z have 65535. Kernels visit their work with three 32-bit stride loops,
//
//     for (uint32_t outer_index = blockIdx.z; outer_index < outer_size; outer_index += gridDim.z)
//     for (uint32_t component_tile = blockIdx.y; component_tile < component_tiles; component_tile += gridDim.y)
//     for (uint32_t shard = blockIdx.x; shard < shards; shard += gridDim.x) { ... }
//
// so geometries beyond the grid limits stay correct, while normally every block owns exactly one work item and no
// device code divides by a runtime work-space extent. Every extent is limited to 2^31-1 so that index + stride can
// never wrap a 32-bit loop variable; address arithmetic widens to 64 bits at the multiply. 2^31 outer rows of even
// one reduced row pair of 2-byte K would already be an 8 GiB FP8 tensor, and the DeepRSmallK / rotated families
// require R > 32, so the limit is not reachable by those families on current devices.
constexpr uint64_t REDUCTION_WORK_GRID_MAX_EXTENT = 2147483647ULL;
constexpr uint64_t REDUCTION_WORK_GRID_MAX_X = 2147483647ULL;
constexpr uint64_t REDUCTION_WORK_GRID_MAX_YZ = 65535ULL;

// Some stages own work items so small that block launch and teardown dominate their loads. Measured on RTX 5090 against
// a persistent 65535-block linear grid, one block per work item was 11% slower for AlignedK Complete items of <= 8 rows
// and <= 512 bytes and 15% slower for DeepRSmallK tiny-K items of <= 256 bytes, while every other measured stage was
// neutral or faster with a full grid. Those launches pass this cap to reductionWorkGrid() and the stride loops give
// each block several work items.
constexpr uint64_t REDUCTION_PERSISTENT_MAX_BLOCKS = 65535;

// max_blocks == 0 launches one block per work item (up to the hardware axis limits). A nonzero max_blocks caps the
// total block count, shrinking outer (z) first, then component tiles (y), then shards (x), so the shard-fastest
// dispatch order is preserved and each block strides over several work items.
[[nodiscard]] inline dim3 reductionWorkGrid(uint64_t shards,
                                            uint64_t component_tiles,
                                            uint64_t outer_size,
                                            uint64_t max_blocks = 0) {
    if (shards == 0 || component_tiles == 0 || outer_size == 0) {
        throw std::logic_error("Reduction work grid requires non-zero shards, component tiles and outer size.");
    }
    if (shards > REDUCTION_WORK_GRID_MAX_EXTENT || component_tiles > REDUCTION_WORK_GRID_MAX_EXTENT ||
        outer_size > REDUCTION_WORK_GRID_MAX_EXTENT) {
        throw std::invalid_argument("Reduction work grid extent exceeds the 32-bit work-index range (2^31 - 1).");
    }
    uint64_t x = std::min(shards, REDUCTION_WORK_GRID_MAX_X);
    uint64_t y = std::min(component_tiles, REDUCTION_WORK_GRID_MAX_YZ);
    uint64_t z = std::min(outer_size, REDUCTION_WORK_GRID_MAX_YZ);
    if (max_blocks != 0) {
        x = std::min(x, max_blocks);
        y = std::min(y, std::max<uint64_t>(1, max_blocks / x));
        z = std::min(z, std::max<uint64_t>(1, max_blocks / (x * y)));
    }
    return dim3(static_cast<unsigned int>(x), static_cast<unsigned int>(y), static_cast<unsigned int>(z));
}

// Balanced R split, precomputed on the host: shards [0, extra_rows) own base_rows + 1 rows and the remaining shards own
// base_rows rows. Both fields are 32-bit: the host guarantees base_rows + 1 fits the kernels' 32-bit per-shard row
// counter, and extra_rows < shards <= 2^31 - 1 (reductionWorkGrid). The row offset widens to 64 bits.
struct ReductionShardRows {
    uint32_t base_rows;
    uint32_t extra_rows;
};

[[nodiscard]] inline ReductionShardRows balancedReductionShardRows(uint64_t reduction_size, uint64_t shards) {
    if (shards == 0 || shards > REDUCTION_WORK_GRID_MAX_EXTENT) {
        throw std::logic_error("Balanced reduction shard rows require 1 <= shards <= 2^31 - 1.");
    }
    const uint64_t base_rows = reduction_size / shards;
    const uint64_t extra_rows = reduction_size % shards;
    if (base_rows + static_cast<uint64_t>(extra_rows != 0) > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Reduction shard owns more rows than its 32-bit row counter supports.");
    }
    return ReductionShardRows{static_cast<uint32_t>(base_rows), static_cast<uint32_t>(extra_rows)};
}

[[nodiscard]] __host__ __device__ __forceinline__ uint64_t reductionShardRowBegin(const ReductionShardRows& rows,
                                                                                  uint32_t shard) {
    return static_cast<uint64_t>(shard) * rows.base_rows + (shard < rows.extra_rows ? shard : rows.extra_rows);
}

[[nodiscard]] __host__ __device__ __forceinline__ uint32_t reductionShardRowCount(const ReductionShardRows& rows,
                                                                                  uint32_t shard) {
    return rows.base_rows + static_cast<uint32_t>(shard < rows.extra_rows);
}

// Direct vector packets use CUDA's native 2/4/8/16-byte vector-load widths, extending to 32/64/128 bytes as a short
// compile-time sequence of uint4 loads (the RK kernels' input packets).
template <size_t Bytes>
struct alignas(16) RawVectorPacket {
    static_assert(Bytes >= 16 && Bytes % 16 == 0);
    uint4 words[Bytes / 16];
};

template <>
struct alignas(1) RawVectorPacket<1> {
    uint8_t value;
};

template <>
struct alignas(2) RawVectorPacket<2> {
    uint16_t value;
};

template <>
struct alignas(4) RawVectorPacket<4> {
    uint32_t value;
};

template <>
struct alignas(8) RawVectorPacket<8> {
    uint2 value;
};

template <>
struct alignas(16) RawVectorPacket<16> {
    uint4 value;
};

template <typename InputT, int ItemsPerLane>
struct PackedInputValues {
    InputT values[ItemsPerLane];
};

template <typename InputT, int ItemsPerLane>
[[nodiscard]] __device__ inline PackedInputValues<InputT, ItemsPerLane> loadVectorizedInputPacket(
    const InputT* source) {
    constexpr size_t bytes = sizeof(InputT) * ItemsPerLane;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16 || bytes == 32
                  || bytes == 64 || bytes == 128);
    using RawT = RawVectorPacket<bytes>;
    using ValuesT = PackedInputValues<InputT, ItemsPerLane>;
    static_assert(sizeof(RawT) == sizeof(ValuesT));
    const RawT raw = *reinterpret_cast<const RawT*>(source);
    return cuda::std::bit_cast<ValuesT>(raw);
}

// The extremum fold applied to a loaded FP8 packet (see Fp8ExtremumFold), one LOP3 per 32-bit word.
template <bool Fold, bool SignFold, typename InputT, int ItemsPerLane>
[[nodiscard]] __device__ __forceinline__ PackedInputValues<InputT, ItemsPerLane> foldFp8ExtremumPacket(
    const PackedInputValues<InputT, ItemsPerLane>& values, const Fp8ExtremumFold& fold) {
    if constexpr (Fold) {
        static_assert(sizeof(InputT) == 1 && ItemsPerLane % 4 == 0);
        struct Words { uint32_t word[ItemsPerLane / 4]; };
        Words words = cuda::std::bit_cast<Words>(values);
#pragma unroll
        for (int i = 0; i < ItemsPerLane / 4; ++i) {
            words.word[i] = foldFp8ExtremumBytes<true, SignFold>(words.word[i], fold);
        }
        return cuda::std::bit_cast<PackedInputValues<InputT, ItemsPerLane>>(words);
    } else {
        static_cast<void>(fold);
        return values;
    }
}

#if THOR_CUB_ENABLE_FP8_TYPES
template <typename InputT, int ItemsPerLane, typename ReductionOpT, typename InputTransformT>
__device__ __forceinline__ void reducePackedFp8ValuesToFp32(
    float (&local)[ItemsPerLane],
    const PackedInputValues<InputT, ItemsPerLane>& values,
    const ReductionOpT& reduction_op,
    const InputTransformT& input_transform) {
    static_assert(std::is_same_v<InputT, __nv_fp8_e4m3> || std::is_same_v<InputT, __nv_fp8_e5m2>);
    static_assert(ItemsPerLane >= 2 && ItemsPerLane % 2 == 0);

#pragma unroll
    for (int item = 0; item < ItemsPerLane; item += 2) {
        const uint16_t packed = static_cast<uint16_t>(values.values[item].__x)
                                | (static_cast<uint16_t>(values.values[item + 1].__x) << 8);
        const float2 decoded = decodeFp8x2ToFp32<InputT>(packed);
        local[item] = reduction_op(local[item], input_transform(decoded.x));
        local[item + 1] = reduction_op(local[item + 1], input_transform(decoded.y));
    }
}
#endif

// Arbitrary row strides can shift an otherwise contiguous per-thread packet away from a 16-byte boundary. Never issue
// a misaligned 8/16-byte vector access: align the global window down to 16 bytes, load one additional uint4 when the
// logical packet is shifted, and select the requested values from that register window. Tensor backing allocations
// carry 128 bytes of trailing padding, so the final logical packet may use the same fixed-width load as every other
// packet without a scalar tail path. Packet starts differ by 16 InputT values between block threads, hence every thread
// in a CTA sees the same ElementOffset for a given reduction row and the dispatch below is warp-uniform.
template <typename InputT, int ItemsPerLane, int ElementOffset>
[[nodiscard]] __device__ inline PackedInputValues<InputT, ItemsPerLane> loadAlignedWindowInputPacket(
    const InputT* source) {
    constexpr int alignment_elements = 16 / static_cast<int>(sizeof(InputT));
    constexpr size_t packet_bytes = sizeof(InputT) * ItemsPerLane;
    static_assert(16 % sizeof(InputT) == 0);
    static_assert(ElementOffset >= 0 && ElementOffset < alignment_elements);
    static_assert(packet_bytes >= 16 && packet_bytes % 16 == 0);

    if constexpr (ElementOffset == 0) {
        return loadVectorizedInputPacket<InputT, ItemsPerLane>(source);
    } else {
        using RawWindowT = RawVectorPacket<packet_bytes + 16>;
        using WindowValuesT = PackedInputValues<InputT, ItemsPerLane + alignment_elements>;
        static_assert(sizeof(RawWindowT) == sizeof(WindowValuesT));

        const uintptr_t source_address = reinterpret_cast<uintptr_t>(source);
        const auto* aligned_source = reinterpret_cast<const InputT*>(source_address & ~uintptr_t{15});
        const RawWindowT raw = *reinterpret_cast<const RawWindowT*>(aligned_source);
        const WindowValuesT window = cuda::std::bit_cast<WindowValuesT>(raw);

        PackedInputValues<InputT, ItemsPerLane> values;
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            values.values[item] = window.values[ElementOffset + item];
        }
        return values;
    }
}

template <typename InputT, int ItemsPerLane>
[[nodiscard]] __device__ inline PackedInputValues<InputT, ItemsPerLane> loadAlignmentSafeInputPacket(
    const InputT* source) {
    static_assert(sizeof(InputT) == 1 || sizeof(InputT) == 2 || sizeof(InputT) == 4 || sizeof(InputT) == 8);
    const int element_offset =
        static_cast<int>((reinterpret_cast<uintptr_t>(source) & uintptr_t{15}) / sizeof(InputT));

    // The selected path is uniform across the CTA: contiguous packets are separated by 16 * sizeof(InputT) bytes.
    if constexpr (sizeof(InputT) == 8) {
        switch (element_offset) {
            case 0:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 0>(source);
            default:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 1>(source);
        }
    } else if constexpr (sizeof(InputT) == 4) {
        switch (element_offset) {
            case 0:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 0>(source);
            case 1:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 1>(source);
            case 2:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 2>(source);
            default:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 3>(source);
        }
    } else if constexpr (sizeof(InputT) == 2) {
        switch (element_offset) {
            case 0:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 0>(source);
            case 1:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 1>(source);
            case 2:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 2>(source);
            case 3:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 3>(source);
            case 4:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 4>(source);
            case 5:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 5>(source);
            case 6:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 6>(source);
            default:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 7>(source);
        }
    } else {
        switch (element_offset) {
            case 0:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 0>(source);
            case 1:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 1>(source);
            case 2:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 2>(source);
            case 3:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 3>(source);
            case 4:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 4>(source);
            case 5:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 5>(source);
            case 6:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 6>(source);
            case 7:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 7>(source);
            case 8:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 8>(source);
            case 9:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 9>(source);
            case 10:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 10>(source);
            case 11:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 11>(source);
            case 12:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 12>(source);
            case 13:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 13>(source);
            case 14:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 14>(source);
            default:
                return loadAlignedWindowInputPacket<InputT, ItemsPerLane, 15>(source);
        }
    }
}

}  // namespace ThorImplementation::ReductionInternal
