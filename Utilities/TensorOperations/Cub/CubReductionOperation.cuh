#pragma once

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ThorImplementation::CubReductionInternal {

namespace cg = cooperative_groups;

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
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
            return fn.template operator()<double>();
#endif
        default:
            throw std::invalid_argument("Unsupported CUB tensor reduction input dtype value "
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
            static_cast<__nv_fp8_e5m2*>(output)[index] = ThorLowPrecision::toFp8E5M2Nosat(value);
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
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::FP64:
            static_cast<double*>(output)[index] = static_cast<double>(value);
            return;
#endif
        default:
            return;
    }
}

inline __host__ __device__ uint64_t tiledReductionOutputIndex(uint64_t outer_index,
                                                              uint64_t inner_index,
                                                              uint64_t output_outer_stride,
                                                              uint64_t output_inner_stride) {
    // The supported tiled layouts are natural dense [outer,inner] (inner stride 1)
    // or dense [inner,outer] retained order (outer stride 1). Keep both common
    // cases to one multiply plus one uniform branch rather than paying two general
    // stride multiplies at every final store.
    if (output_inner_stride == 1) {
        return outer_index * output_outer_stride + inner_index;
    }
    if (output_outer_stride == 1) {
        return inner_index * output_inner_stride + outer_index;
    }
    return outer_index * output_outer_stride + inner_index * output_inner_stride;
}

// Keep output storage dtype out of the CUB iterator type. The dtype is uniform for the launch and is selected only
// when each final aggregate is stored. This avoids instantiating every CUB kernel for InputT x OutputT x operation x
// path while preserving fused FP32 finalization and storage conversion.
template <typename OutputFinalizeT>
struct FinalizeAndStoreRuntimeFp32 {
    void* output;
    DataType output_dtype;
    OutputFinalizeT finalize;
    float output_scale;

    template <typename IndexT>
    __host__ __device__ void operator()(IndexT index, float value) const {
        storeFp32AsRuntimeDType(
            output, output_dtype, static_cast<uint64_t>(index), finalize(value) * output_scale);
    }
};

template <typename OutputFinalizeT>
auto makeRuntimeFp32OutputIterator(void* output,
                                   DataType output_dtype,
                                   OutputFinalizeT output_finalize,
                                   float output_scale = 1.0f) {
    return cuda::make_tabulate_output_iterator(FinalizeAndStoreRuntimeFp32<OutputFinalizeT>{
        output, output_dtype, output_finalize, output_scale});
}

template <typename InputT, typename InputTransformT>
struct ConvertAndTransformInputToFp32 {
    InputTransformT transform;

    __host__ __device__ float operator()(InputT value) const { return transform(ToFp32<InputT>{}(value)); }
};

template <typename InputT, typename InputTransformT>
auto makeContiguousFp32Iterator(const InputT* input, InputTransformT input_transform) {
    return thrust::make_transform_iterator(
        input, ConvertAndTransformInputToFp32<InputT, InputTransformT>{input_transform});
}

template <typename InputT, typename InputTransformT>
struct AffineStridedToFp32 {
    const InputT* input;
    uint64_t stride;
    InputTransformT transform;

    __host__ __device__ float operator()(int64_t logical_index) const {
        const uint64_t physical_index = static_cast<uint64_t>(logical_index) * stride;
        return transform(ToFp32<InputT>{}(input[physical_index]));
    }
};

template <typename InputT, typename InputTransformT>
auto makeAffineStridedFp32Iterator(const InputT* input, uint64_t stride, InputTransformT input_transform) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        AffineStridedToFp32<InputT, InputTransformT>{input, stride, input_transform});
}

constexpr int TILED_REDUCTION_BLOCK_THREADS = 256;
constexpr int TILED_REDUCTION_WARP_THREADS = static_cast<int>(CubReductionTiledPolicy::WARP_THREADS);
constexpr int TILED_REDUCTION_WARPS_PER_BLOCK =
    static_cast<int>(CubReductionTiledPolicy::WARPS_PER_BLOCK);
static_assert(TILED_REDUCTION_WARPS_PER_BLOCK == TILED_REDUCTION_BLOCK_THREADS / TILED_REDUCTION_WARP_THREADS);
constexpr uint64_t TILED_REDUCTION_TARGET_ACTIVE_WARPS = CubReductionTiledPolicy::TARGET_ACTIVE_WARPS;
constexpr uint64_t TILED_REDUCTION_MAX_GRID_BLOCKS = 65535;

// Sixteen FP32 accumulators per lane is the current proven-fast register tile. It is a benchmarked design point, not a
// claim that sixteen is the universal optimum: Thor keeps the complete kernel near a ~48-register/thread design budget,
// leaving the remaining registers for indexing, pipeline state, vector packets, operation temporaries, and compiler live
// ranges. For wider trailing vectors, 2/4/8 physical warps cooperate on one output while keeping the same <=16
// accumulators/thread. That scales the full-row engine through D=4096 without increasing per-thread accumulator pressure.
constexpr int FULL_ROW_MAX_COMPONENTS_PER_LANE =
    static_cast<int>(CubReductionTiledPolicy::FULL_ROW_MAX_COMPONENTS_PER_LANE);
constexpr uint64_t FULL_ROW_COMPONENTS_PER_WARP =
    CubReductionTiledPolicy::FULL_ROW_COMPONENTS_PER_WARP;
constexpr int FULL_ROW_MAX_WARPS_PER_OUTPUT =
    static_cast<int>(CubReductionTiledPolicy::FULL_ROW_MAX_WARPS_PER_OUTPUT);
constexpr uint64_t FULL_ROW_GROUP_MAX_INNER_SIZE =
    CubReductionTiledPolicy::FULL_ROW_GROUP_MAX_INNER_SIZE;
// Once one output consumes a full 8-warp block, wider D values are sharded across independent blocks. Each block owns
// at most this many output components, preserving the same <=16 FP32 accumulators/thread. Component shards never need
// to communicate because every trailing component is an independent reduction across the reduction axis.
constexpr uint64_t FULL_ROW_COMPONENTS_PER_BLOCK = CubReductionTiledPolicy::FULL_ROW_COMPONENTS_PER_BLOCK;

// Permuted retained output is produced without a global-memory transpose. Eight warps reduce adjacent physical outer
// rows for one contiguous retained-component tile, keep all reduction state in registers, stage only the finalized FP32
// values through a padded shared-memory tile, and then write dense [inner,outer] with adjacent outer coordinates next to
// one another. The largest tile is 8 x 512 finalized values (~16 KiB); no input/reduction intermediate is materialized.
// Keep the shared-transpose packet width tied to the proven full-row register tile so the two paths cannot silently
// diverge as the tuned reduction family evolves.
constexpr int SHARED_TRANSPOSE_OUTER_ROWS = TILED_REDUCTION_WARPS_PER_BLOCK;
constexpr int SHARED_TRANSPOSE_MAX_ITEMS_PER_LANE = FULL_ROW_MAX_COMPONENTS_PER_LANE;

// Shared-transpose output is an epilogue of the tuned one-warp-per-output reduction families. Producers keep their
// native register ownership and stage only finalized values. The packet layout mirrors the original producer mapping
// while padding shared-memory stripes so the component-major consumer pass can emit adjacent outer coordinates.
template <int ItemsPerLane>
struct SharedTransposeContiguousPacketLayout {
    static_assert(ItemsPerLane == 1 || ItemsPerLane == 2 || ItemsPerLane == 4 || ItemsPerLane == 8
                  || ItemsPerLane == 16);
    static constexpr uint64_t stripe_pitch = ItemsPerLane == 2 ? 34 : 33;
    static constexpr uint64_t row_unpadded = static_cast<uint64_t>(ItemsPerLane) * stripe_pitch;
    static constexpr uint64_t row_pitch =
        row_unpadded + ((4 + 32 - (row_unpadded % 32)) % 32);
    static constexpr size_t shared_bytes = static_cast<size_t>(SHARED_TRANSPOSE_OUTER_ROWS)
                                           * static_cast<size_t>(row_pitch) * sizeof(float);
};

template <int ItemsPerLane>
struct SharedTransposeStridedPacketLayout {
    static_assert(ItemsPerLane == 2 || ItemsPerLane == 4 || ItemsPerLane == 8 || ItemsPerLane == 16);
    static constexpr uint64_t stripe_pitch = 33;
    static constexpr uint64_t row_pitch = static_cast<uint64_t>(ItemsPerLane) * stripe_pitch;
    static constexpr size_t shared_bytes = static_cast<size_t>(SHARED_TRANSPOSE_OUTER_ROWS)
                                           * static_cast<size_t>(row_pitch) * sizeof(float);
};

constexpr uint64_t SHARED_TRANSPOSE_NARROW_ROW_PITCH = TILED_REDUCTION_WARP_THREADS;
constexpr size_t SHARED_TRANSPOSE_NARROW_SHARED_BYTES =
    static_cast<size_t>(SHARED_TRANSPOSE_OUTER_ROWS) * SHARED_TRANSPOSE_NARROW_ROW_PITCH * sizeof(float);

// Each physical warp owns a private two-stage global->shared pipeline. Keeping the staging footprint fixed in bytes
// makes occupancy independent of InputT while still allowing narrow dtypes to stage more reduction rows per batch.
constexpr int ASYNC_TILED_REDUCTION_PIPELINE_STAGES = 2;
constexpr size_t ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP =
    static_cast<size_t>(CubReductionTiledPolicy::ASYNC_STAGE_BYTES_PER_WARP);
constexpr size_t ASYNC_TILED_REDUCTION_SHARED_BYTES =
    TILED_REDUCTION_WARPS_PER_BLOCK * ASYNC_TILED_REDUCTION_PIPELINE_STAGES
    * ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP;

[[nodiscard]] inline __host__ __device__ uint64_t ceilDivideU64(uint64_t numerator, uint64_t denominator) {
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

[[nodiscard]] inline __host__ __device__ uint64_t minU64(uint64_t lhs, uint64_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

[[nodiscard]] inline __host__ __device__ uint64_t largestRowsWithByteAlignment(uint64_t max_rows,
                                                                              size_t row_bytes,
                                                                              size_t alignment) {
    // Alignment is a power of two <= 16, so at most 15 smaller row counts need to be considered.
    for (uint64_t delta = 0; delta < static_cast<uint64_t>(alignment) && delta < max_rows; ++delta) {
        const uint64_t rows = max_rows - delta;
        if ((static_cast<size_t>(rows) * row_bytes) % alignment == 0) {
            return rows;
        }
    }
    return 0;
}

struct AsyncFullRowStagePlan {
    uint64_t rows_per_stage = 0;
    size_t bulk_copy_alignment = 0;
};

// Prefer a stage row count whose total byte count preserves the strongest useful cuda::memcpy_async alignment. Odd
// row widths such as 33 FP16 values otherwise tend to choose the absolute largest stage (31 rows = 2046 bytes), which
// prevents the hardware cp.async path even though a slightly smaller 24-row stage is 16-byte aligned. Returning the
// alignment together with the row count is important: the source of each outer row may need a tiny direct prologue
// before the first async stage can truthfully make that same alignment promise.
template <typename InputT>
[[nodiscard]] inline __host__ __device__ AsyncFullRowStagePlan chooseAsyncFullRowStagePlan(
    uint64_t row_elements,
    uint64_t reduction_size,
    size_t stage_bytes = ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP) {
    const uint64_t stage_capacity_elements = static_cast<uint64_t>(stage_bytes / sizeof(InputT));
    uint64_t max_rows = stage_capacity_elements / row_elements;
    max_rows = minU64(max_rows, reduction_size);
    if (max_rows == 0) {
        return {};
    }

    const size_t row_bytes = static_cast<size_t>(row_elements) * sizeof(InputT);
    if (const uint64_t rows = largestRowsWithByteAlignment(max_rows, row_bytes, 16); rows != 0) {
        return {rows, 16};
    }
    if (const uint64_t rows = largestRowsWithByteAlignment(max_rows, row_bytes, 8); rows != 0) {
        return {rows, 8};
    }
    if (const uint64_t rows = largestRowsWithByteAlignment(max_rows, row_bytes, 4); rows != 0) {
        return {rows, 4};
    }
    return {max_rows, 0};
}

template <typename InputT>
[[nodiscard]] __device__ inline uint64_t chooseAsyncFullRowHeadRows(const InputT* outer_input,
                                                                     uint64_t reduction_size,
                                                                     uint64_t inner_size,
                                                                     size_t bulk_copy_alignment) {
    if (bulk_copy_alignment < 4 || reduction_size == 0) {
        return reduction_size;
    }

    const size_t row_bytes = static_cast<size_t>(inner_size) * sizeof(InputT);
    const uintptr_t initial_source_address = reinterpret_cast<uintptr_t>(outer_input);
    for (size_t alignment = bulk_copy_alignment; alignment >= 4; alignment >>= 1U) {
        const uintptr_t alignment_mask = static_cast<uintptr_t>(alignment - 1);
        uintptr_t source_address = initial_source_address;
        const uint64_t search_rows = minU64(reduction_size, static_cast<uint64_t>(alignment));
        for (uint64_t head_rows = 0; head_rows < search_rows; ++head_rows) {
            if ((source_address & alignment_mask) == 0U) {
                return head_rows;
            }
            source_address += row_bytes;
        }
    }

    // A power-of-two alignment has a period no longer than its byte alignment. If even four-byte alignment cannot be
    // reached in the bounded searches above, this outer row cannot use the hardware async-copy path safely.
    return reduction_size;
}

template <typename InputT>
[[nodiscard]] __device__ inline uint64_t chooseAsyncFullRowBulkRows(uint64_t remaining_rows, uint64_t inner_size) {
    // The head prologue guarantees that the bulk source is aligned by at least four bytes. Keep only the largest prefix
    // whose byte count is also four-byte aligned; at most three rows are left for the direct suffix. Full stages retain
    // the stronger alignment encoded in AsyncFullRowStagePlan.
    const size_t row_bytes = static_cast<size_t>(inner_size) * sizeof(InputT);
    return largestRowsWithByteAlignment(remaining_rows, row_bytes, 4);
}

template <typename GroupT, typename PipelineT>
__device__ inline void memcpyAsyncPreferAligned(GroupT group,
                                                void* destination,
                                                const void* source,
                                                size_t bytes,
                                                PipelineT& pipeline) {
    const uintptr_t alignment_bits = reinterpret_cast<uintptr_t>(destination)
                                     | reinterpret_cast<uintptr_t>(source)
                                     | static_cast<uintptr_t>(bytes);
    if ((alignment_bits & 15U) == 0U) {
        cuda::memcpy_async(group, destination, source, cuda::aligned_size_t<16>(bytes), pipeline);
    } else if ((alignment_bits & 7U) == 0U) {
        cuda::memcpy_async(group, destination, source, cuda::aligned_size_t<8>(bytes), pipeline);
    } else if ((alignment_bits & 3U) == 0U) {
        cuda::memcpy_async(group, destination, source, cuda::aligned_size_t<4>(bytes), pipeline);
    } else {
        // Defensive correctness fallback. The full-row async planner peels under-aligned heads/tails before enqueueing,
        // so production full-row stages should never reach this generic libcudacxx path.
        cuda::memcpy_async(group, destination, source, bytes, pipeline);
    }
}

template <typename InputT, typename GroupT, typename PipelineT>
__device__ inline void enqueueAsyncFullRowReductionStage(GroupT warp,
                                                         PipelineT& pipeline,
                                                         InputT* stage,
                                                         const InputT* input,
                                                         uint64_t outer_index,
                                                         uint64_t row_begin,
                                                         uint64_t row_count,
                                                         uint64_t reduction_size,
                                                         uint64_t inner_size) {
    pipeline.producer_acquire();
    const InputT* source = input + (outer_index * reduction_size + row_begin) * inner_size;
    const size_t bytes = static_cast<size_t>(row_count * inner_size) * sizeof(InputT);
    memcpyAsyncPreferAligned(warp, stage, source, bytes, pipeline);
    pipeline.producer_commit();
}

// Directly consume a bounded head/tail range while preserving the exact component ownership used by the async wide
// and grouped consumers. This is intentionally only a prologue/epilogue helper: the normal bulk remains double-buffered.
template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          int ComponentThreads,
          int ItemsPerLane>
__device__ inline void reduceDirectFullRowComponentRange(const InputT* outer_input,
                                                         uint64_t row_begin,
                                                         uint64_t row_end,
                                                         uint64_t inner_size,
                                                         int component_lane,
                                                         ReductionOpT reduction_op,
                                                         InputTransformT input_transform,
                                                         float (&local)[ItemsPerLane]) {
    if (row_begin >= row_end) {
        return;
    }

    const InputT* row_source = outer_input + row_begin * inner_size;
    for (uint64_t row = row_begin; row < row_end; ++row) {
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = static_cast<uint64_t>(component_lane)
                                       + static_cast<uint64_t>(item * ComponentThreads);
            if (component < inner_size) {
                local[item] =
                    reduction_op(local[item], input_transform(ToFp32<InputT>{}(row_source[component])));
            }
        }
        row_source += inner_size;
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, int RowLanes>
__device__ inline void reduceDirectNarrowFullRowRange(const InputT* outer_input,
                                                      uint64_t row_begin,
                                                      uint64_t row_end,
                                                      uint64_t inner_size,
                                                      uint64_t component,
                                                      int row_lane,
                                                      ReductionOpT reduction_op,
                                                      InputTransformT input_transform,
                                                      float& local) {
    const uint64_t first_row = row_begin + static_cast<uint64_t>(row_lane);
    if (first_row >= row_end) {
        return;
    }

    const InputT* source = outer_input + first_row * inner_size + component;
    const uint64_t row_stride = static_cast<uint64_t>(RowLanes) * inner_size;
    for (uint64_t row = first_row; row < row_end; row += static_cast<uint64_t>(RowLanes)) {
        local = reduction_op(local, input_transform(ToFp32<InputT>{}(*source)));
        source += row_stride;
    }
}

// Defined later with the other direct Tiled backend helpers. Async launchers use it when a geometry cannot form even a
// four-byte-aligned async stage, avoiding a forced generic cuda::memcpy_async path.
template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int RowLanes>
void launchDirectTiledFixedSegmentReductionForRowLanes(const InputT* input,
                                                       void* output,
                                                       DataType output_dtype,
                                                       const CubReductionGeometry& geometry,
                                                       ReductionOpT reduction_op,
                                                       float init,
                                                       InputTransformT input_transform,
                                                       OutputFinalizeT output_finalize,
                                                       float output_scale,
                                                       cudaStream_t stream);

// The narrow async kernel handles non-exact widths below 32: the whole trailing row fits in one physical warp tile,
// while otherwise-unused lanes split reduction rows for inner_size <= 16. Exact K=32 uses the direct full-row kernel
// instead: all lanes are already useful, each reduction row is a naturally coalesced warp load, and async pipeline
// setup only adds fixed overhead for short reductions.
// For 33..511 trailing components whose complete row fits in one async stage, one physical warp owns the complete output
// vector. Each lane owns up to ItemsPerLane components spaced 32 apart, so every shared-memory read round is a contiguous
// 32-component transaction.
// The global->shared pipeline therefore always copies complete consecutive rows, including awkward widths such as 33,
// 65, and 129, instead of submitting per-row component-strip copies.
// Once the trailing vector exceeds one warp's <=512-component register tile, keep the same full-row architecture and
// scale ownership horizontally. A 2/4/8-warp cooperative group owns one output vector while every thread still carries
// at most ItemsPerLane FP32 accumulators. Each added warp contributes another 2 KiB to each async stage, so the existing
// 32 KiB/block allocation is simply repartitioned as 4x2-warp, 2x4-warp, or 1x8-warp output groups. The global copy is
// still a complete contiguous row slab; no warp copies a strided component strip.
// Direct vector packets use CUDA's native 2/4/8/16-byte vector-load widths, extending to 32/64/128 bytes as a
// short compile-time sequence of uint4 loads. These exact-width kernels have no shared memory and no synchronization.
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

// Exact large widths can bypass shared memory entirely. A 2/4/8-warp output group partitions one complete row into
// contiguous ItemsPerLane packets, one packet per thread. The groups are independent and no synchronization is needed:
// every thread repeatedly loads its packet from each reduction row, accumulates in registers, and stores disjoint outputs.
// Beyond one full block/output, exact multiples of 4096 scale D without increasing per-thread registers by assigning
// independent component shards to independent blocks. Full shards retain the exact vector-direct x16 memory path.
// Arbitrary widths use the alignment-safe shaped backend below.
// For arbitrary large D, preserve complete 4096-component shards unless a tiny remainder would leave too little
// aggregate warp parallelism. When a sub-half-block remainder has enough outer work, borrow only the final full shard
// and split that 4096+remainder tail into two ~half-sized launches. Otherwise keep the simpler 4096*N + remainder
// geometry. This decision is host-side only; every CUDA launch still uses one fixed shard width and a branch-free
// reduction loop.
constexpr uint64_t ALIGNMENT_SAFE_TAIL_REBALANCE_MIN_ACTIVE_WARPS = TILED_REDUCTION_TARGET_ACTIVE_WARPS / 2;

// For an awkward large row, choose the smallest power-of-two output group that satisfies both register ownership and
// async-stage capacity. Each warp contributes 512 components of register capacity and 2 KiB to one pipeline stage.
// Returning zero means even a full 8-warp block cannot stage and own the complete row with the current fixed resources.
// Original Patch-2 direct component-tiled backend retained as the final fallback for rows that cannot be owned and
// staged by one 256-thread block under the fixed full-row resource budget.
template <int RowLanes>
[[nodiscard]] int chooseDirectTiledReductionWarpsPerTile(const CubReductionGeometry& geometry) {
    static_assert(RowLanes == 1 || RowLanes == 2 || RowLanes == 4 || RowLanes == 8 || RowLanes == 16);
    constexpr uint64_t components_per_warp = TILED_REDUCTION_WARP_THREADS / RowLanes;
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_warp);
    const uint64_t output_tiles = geometry.outer_size * component_tiles;
    const uint64_t useful_warps_from_rows = ceilDivideU64(geometry.reduction_size, static_cast<uint64_t>(RowLanes));
    const uint64_t desired_warps_per_tile =
        ceilDivideU64(TILED_REDUCTION_TARGET_ACTIVE_WARPS, std::max<uint64_t>(output_tiles, 1));

    int warps_per_tile = 1;
    while (warps_per_tile < TILED_REDUCTION_WARPS_PER_BLOCK
           && static_cast<uint64_t>(warps_per_tile) < desired_warps_per_tile
           && static_cast<uint64_t>(warps_per_tile) < useful_warps_from_rows) {
        warps_per_tile *= 2;
    }
    while (warps_per_tile > 1 && static_cast<uint64_t>(warps_per_tile) > useful_warps_from_rows) {
        warps_per_tile /= 2;
    }
    return warps_per_tile;
}

// Adaptive cooperative tiled-value primitive. One CTA owns one retained-component tile while a geometry-selected
// power-of-two number of physical warps split the reduction rows. Unlike the older full-row exact-width kernels, this
// primitive derives its cooperative width from actual output-tile parallelism and may use up to a full 32-warp CUDA
// block when those extra warps all cooperate on the same reduction. When fewer than 8 warps cooperate, independent
// output tiles share one <=256-thread CTA so the small-group regime does not devolve into tiny one-warp blocks. The
// read-side packet is a compile-time 4/8/16-byte width; the production-control launcher remains fixed at 16 bytes/thread.
//
// Tail handling deliberately relies on Thor's tensor allocation contract: every tensor has 128 bytes of physical
// backing padding after its logical storage. A lane whose packet *starts* inside the retained row therefore issues the
// same fixed-width alignment-safe load even when that packet crosses the logical row end. For non-final reduction rows
// the harmless over-read lands in the next packed row; for the final row it lands in the 128-byte backing pad. Values
// beyond inner_size are accumulated but never stored. Lanes whose packet starts outside the logical row do not load,
// which bounds the maximum final-tensor over-read to one alignment-safe packet window (<=32 bytes here). Output stores
// remain predicated because output rows are logically adjacent and an over-store could corrupt the next result row.
constexpr size_t ADAPTIVE_COOPERATIVE_TILED_MAX_VECTOR_BYTES = 16;
constexpr size_t ADAPTIVE_COOPERATIVE_TILED_MIN_VECTOR_BYTES = 4;
constexpr int ADAPTIVE_COOPERATIVE_TILED_MAX_WARPS = 32;
constexpr uint64_t ADAPTIVE_COOPERATIVE_TILED_TARGET_ACTIVE_WARPS =
    TILED_REDUCTION_TARGET_ACTIVE_WARPS * CubReductionTiledPolicy::DENSE_RK_TARGET_SM_WAVES;

template <typename InputT, size_t PacketBytes>
[[nodiscard]] constexpr int adaptiveCooperativeTiledItemsPerLane() {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(PacketBytes >= sizeof(InputT));
    static_assert(PacketBytes % sizeof(InputT) == 0);
    constexpr int items = static_cast<int>(PacketBytes / sizeof(InputT));
    static_assert(items >= 1 && items <= 8,
                  "adaptive cooperative tiled reduction currently targets FP16/BF16/FP32 storage");
    return items;
}

template <typename InputT, size_t PacketBytes>
[[nodiscard]] inline int chooseAdaptiveCooperativeTiledWarpsPerTile(const CubReductionGeometry& geometry) {
    constexpr int items_per_lane = adaptiveCooperativeTiledItemsPerLane<InputT, PacketBytes>();
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(items_per_lane);
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_tile);
    uint64_t output_tiles = std::numeric_limits<uint64_t>::max();
    if (component_tiles == 0) {
        output_tiles = 1;
    } else if (geometry.outer_size <= std::numeric_limits<uint64_t>::max() / component_tiles) {
        output_tiles = std::max<uint64_t>(1, geometry.outer_size * component_tiles);
    }

    const uint64_t desired_warps = ceilDivideU64(
        ADAPTIVE_COOPERATIVE_TILED_TARGET_ACTIVE_WARPS, std::max<uint64_t>(output_tiles, 1));
    int warps = 1;
    while (warps < ADAPTIVE_COOPERATIVE_TILED_MAX_WARPS
           && static_cast<uint64_t>(warps) < desired_warps
           && static_cast<uint64_t>(warps) < geometry.reduction_size) {
        warps <<= 1;
    }

    // When a warp revisits multiple rows, prefer a cooperative width whose row-to-row byte stride preserves the
    // 16-byte packet alignment. This lets every warp see one stable row-head alignment through its reduction slice.
    // If the reduction is already no deeper than the cooperative group, every warp owns at most one row and no such
    // stride constraint exists. loadAlignmentSafeInputPacket remains the correctness fallback either way.
    while (warps < ADAPTIVE_COOPERATIVE_TILED_MAX_WARPS
           && geometry.reduction_size > static_cast<uint64_t>(warps)
           && (geometry.inner_size * static_cast<uint64_t>(sizeof(InputT)) * static_cast<uint64_t>(warps))
                  % PacketBytes
                  != 0) {
        warps <<= 1;
    }
    return warps;
}

// Shared cooperative tiled combine used by both the terminal packet-adaptive reducer and the staged FP32-partial
// first stage. Every participating physical warp owns the same lane/items coordinates for a different subset of
// reduction rows; this routine performs only the inter-warp associative combine. Epilogues remain separate so the
// same mechanics can either finalize/store public output or emit FP32 shard partials for another ordinary reduction.
template <int BlockWarps, int WarpsPerTile, int ItemsPerLane, typename ReductionOpT>
__device__ inline void cooperativeTiledCombineWarpPartials(
    float (&partials)[BlockWarps][ItemsPerLane][TILED_REDUCTION_WARP_THREADS],
    const float (&local)[ItemsPerLane],
    int physical_warp,
    int warp_in_tile,
    int lane,
    ReductionOpT reduction_op) {
    static_assert(WarpsPerTile > 1);
    static_assert(BlockWarps >= WarpsPerTile);
    static_assert(BlockWarps % WarpsPerTile == 0);
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        partials[physical_warp][item][lane] = local[item];
    }
    __syncthreads();

#pragma unroll
    for (int stride = WarpsPerTile / 2; stride > 0; stride >>= 1) {
        if (warp_in_tile < stride) {
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                partials[physical_warp][item][lane] = reduction_op(
                    partials[physical_warp][item][lane], partials[physical_warp + stride][item][lane]);
            }
        }
        __syncthreads();
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int WarpsPerTile,
          int ItemsPerLane,
          bool NaturallyAlignedRows>
__global__ void adaptiveCooperativeTiledFixedSegmentReductionKernel(const InputT* input,
                                                                    void* output,
                                                                    DataType output_dtype,
                                                                    uint64_t outer_size,
                                                                    uint64_t reduction_size,
                                                                    uint64_t inner_size,
                                                                    uint64_t output_outer_stride,
                                                                    uint64_t output_inner_stride,
                                                                    ReductionOpT reduction_op,
                                                                    float init,
                                                                    InputTransformT input_transform,
                                                                    OutputFinalizeT output_finalize,
                                                                    float output_scale) {
    static_assert(WarpsPerTile == 1 || WarpsPerTile == 2 || WarpsPerTile == 4 || WarpsPerTile == 8
                  || WarpsPerTile == 16 || WarpsPerTile == 32);
    constexpr size_t packet_bytes = sizeof(InputT) * ItemsPerLane;
    static_assert(packet_bytes == 4 || packet_bytes == 8 || packet_bytes == 16);
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    constexpr int block_warps = WarpsPerTile < TILED_REDUCTION_WARPS_PER_BLOCK
                                    ? TILED_REDUCTION_WARPS_PER_BLOCK
                                    : WarpsPerTile;
    constexpr int groups_per_block = block_warps / WarpsPerTile;
    static_assert(block_warps <= ADAPTIVE_COOPERATIVE_TILED_MAX_WARPS);
    static_assert(block_warps % WarpsPerTile == 0);

    __shared__ float partials[block_warps][ItemsPerLane][TILED_REDUCTION_WARP_THREADS];

    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int group_in_block = physical_warp / WarpsPerTile;
    const int warp_in_tile = physical_warp % WarpsPerTile;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t component_tiles = ceilDivideU64(inner_size, components_per_tile);
    const uint64_t total_work = outer_size * component_tiles;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(groups_per_block);

    for (uint64_t block_work_begin =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(groups_per_block);
         block_work_begin < total_work;
         block_work_begin += grid_stride) {
        const uint64_t work_index = block_work_begin + static_cast<uint64_t>(group_in_block);
        const bool active_group = work_index < total_work;
        const uint64_t outer_index = active_group ? work_index / component_tiles : 0;
        const uint64_t component_tile = active_group ? work_index - outer_index * component_tiles : 0;
        const uint64_t packet_component_begin =
            component_tile * components_per_tile
            + static_cast<uint64_t>(lane) * static_cast<uint64_t>(ItemsPerLane);
        const bool owns_packet_start = active_group && packet_component_begin < inner_size;

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = init;
        }

        if (owns_packet_start) {
            const uint64_t first_row = static_cast<uint64_t>(warp_in_tile);
            if (first_row < reduction_size) {
                uint64_t input_index =
                    (outer_index * reduction_size + first_row) * inner_size + packet_component_begin;
                const uint64_t row_stride = static_cast<uint64_t>(WarpsPerTile) * inner_size;
                for (uint64_t row = first_row; row < reduction_size; row += static_cast<uint64_t>(WarpsPerTile)) {
                    const PackedInputValues<InputT, ItemsPerLane> values = [&]() {
                        if constexpr (NaturallyAlignedRows) {
                            return loadVectorizedInputPacket<InputT, ItemsPerLane>(input + input_index);
                        } else {
                            return loadAlignmentSafeInputPacket<InputT, ItemsPerLane>(input + input_index);
                        }
                    }();
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item) {
                        local[item] = reduction_op(
                            local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
                    }
                    input_index += row_stride;
                }
            }
        }

        if constexpr (WarpsPerTile == 1) {
            if (owns_packet_start) {
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item) {
                    const uint64_t component = packet_component_begin + static_cast<uint64_t>(item);
                    if (component < inner_size) {
                        const float finalized = output_finalize(local[item]) * output_scale;
                        storeFp32AsRuntimeDType(output,
                                                output_dtype,
                                                tiledReductionOutputIndex(outer_index,
                                                                          component,
                                                                          output_outer_stride,
                                                                          output_inner_stride),
                                                finalized);
                    }
                }
            }
        } else {
            cooperativeTiledCombineWarpPartials<block_warps, WarpsPerTile, ItemsPerLane>(
                partials, local, physical_warp, warp_in_tile, lane, reduction_op);

            if (warp_in_tile == 0 && owns_packet_start) {
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item) {
                    const uint64_t component = packet_component_begin + static_cast<uint64_t>(item);
                    if (component < inner_size) {
                        const float finalized =
                            output_finalize(partials[group_in_block * WarpsPerTile][item][lane]) * output_scale;
                        storeFp32AsRuntimeDType(output,
                                                output_dtype,
                                                tiledReductionOutputIndex(outer_index,
                                                                          component,
                                                                          output_outer_stride,
                                                                          output_inner_stride),
                                                finalized);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int WarpsPerTile,
          size_t PacketBytes,
          bool NaturallyAlignedRows>
void launchAdaptiveCooperativeTiledFixedSegmentReductionForWarps(const InputT* input,
                                                                 void* output,
                                                                 DataType output_dtype,
                                                                 const CubReductionGeometry& geometry,
                                                                 ReductionOpT reduction_op,
                                                                 float init,
                                                                 InputTransformT input_transform,
                                                                 OutputFinalizeT output_finalize,
                                                                 float output_scale,
                                                                 cudaStream_t stream) {
    constexpr int items_per_lane = adaptiveCooperativeTiledItemsPerLane<InputT, PacketBytes>();
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(items_per_lane);
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_tile);
    const uint64_t total_work = geometry.outer_size * component_tiles;
    constexpr int block_warps = WarpsPerTile < TILED_REDUCTION_WARPS_PER_BLOCK
                                    ? TILED_REDUCTION_WARPS_PER_BLOCK
                                    : WarpsPerTile;
    constexpr int groups_per_block = block_warps / WarpsPerTile;
    const uint64_t required_blocks = ceilDivideU64(total_work, static_cast<uint64_t>(groups_per_block));
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));
    constexpr int block_threads = block_warps * TILED_REDUCTION_WARP_THREADS;

    adaptiveCooperativeTiledFixedSegmentReductionKernel<InputT,
                                                        ReductionOpT,
                                                        InputTransformT,
                                                        OutputFinalizeT,
                                                        WarpsPerTile,
                                                        items_per_lane,
                                                        NaturallyAlignedRows>
        <<<grid_blocks, block_threads, 0, stream>>>(input,
                                                    output,
                                                    output_dtype,
                                                    geometry.outer_size,
                                                    geometry.reduction_size,
                                                    geometry.inner_size,
                                                    geometry.tiled_output_outer_stride,
                                                    geometry.tiled_output_inner_stride,
                                                    reduction_op,
                                                    init,
                                                    input_transform,
                                                    output_finalize,
                                                    output_scale);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          size_t PacketBytes>
void launchAdaptiveCooperativeTiledFixedSegmentReductionForPacketBytes(const InputT* input,
                                                                        void* output,
                                                                        DataType output_dtype,
                                                                        const CubReductionGeometry& geometry,
                                                                        ReductionOpT reduction_op,
                                                                        float init,
                                                                        InputTransformT input_transform,
                                                                        OutputFinalizeT output_finalize,
                                                                        float output_scale,
                                                                        cudaStream_t stream) {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    const int warps = chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, PacketBytes>(geometry);
    const bool naturally_aligned_rows =
        (geometry.inner_size * static_cast<uint64_t>(sizeof(InputT))) % PacketBytes == 0;

    // 8- and 4-byte packets are only selected by the packet-adaptive host policy when every row begins naturally
    // aligned to that packet width. The 16-byte path remains the general awkward-row path and can use Thor's existing
    // alignment-safe 16/32-byte window loader plus the 128-byte backing pad.
    if constexpr (PacketBytes < 16) {
        if (!naturally_aligned_rows) {
            throw std::logic_error("Adaptive cooperative tiled sub-16-byte packet selected for an unaligned row stride.");
        }
    }

    const auto launch_for_alignment = [&]<bool NaturallyAlignedRows>() {
        switch (warps) {
            case 1:
                launchAdaptiveCooperativeTiledFixedSegmentReductionForWarps<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 1, PacketBytes, NaturallyAlignedRows>(
                    input, output, output_dtype, geometry, reduction_op, init, input_transform, output_finalize, output_scale, stream);
                return;
            case 2:
                launchAdaptiveCooperativeTiledFixedSegmentReductionForWarps<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 2, PacketBytes, NaturallyAlignedRows>(
                    input, output, output_dtype, geometry, reduction_op, init, input_transform, output_finalize, output_scale, stream);
                return;
            case 4:
                launchAdaptiveCooperativeTiledFixedSegmentReductionForWarps<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 4, PacketBytes, NaturallyAlignedRows>(
                    input, output, output_dtype, geometry, reduction_op, init, input_transform, output_finalize, output_scale, stream);
                return;
            case 8:
                launchAdaptiveCooperativeTiledFixedSegmentReductionForWarps<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 8, PacketBytes, NaturallyAlignedRows>(
                    input, output, output_dtype, geometry, reduction_op, init, input_transform, output_finalize, output_scale, stream);
                return;
            case 16:
                launchAdaptiveCooperativeTiledFixedSegmentReductionForWarps<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 16, PacketBytes, NaturallyAlignedRows>(
                    input, output, output_dtype, geometry, reduction_op, init, input_transform, output_finalize, output_scale, stream);
                return;
            case 32:
                launchAdaptiveCooperativeTiledFixedSegmentReductionForWarps<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 32, PacketBytes, NaturallyAlignedRows>(
                    input, output, output_dtype, geometry, reduction_op, init, input_transform, output_finalize, output_scale, stream);
                return;
            default:
                throw std::logic_error("Adaptive cooperative tiled reduction selected an invalid warp count.");
        }
    };

    if constexpr (PacketBytes == 16) {
        if (naturally_aligned_rows) {
            launch_for_alignment.template operator()<true>();
        } else {
            launch_for_alignment.template operator()<false>();
        }
    } else {
        // The host packet chooser guarantees this for narrow packets, avoiding any scalar/misaligned fallback.
        launch_for_alignment.template operator()<true>();
    }
}

template <typename InputT>
[[nodiscard]] inline size_t chooseAdaptiveCooperativeTiledPacketBytes(const CubReductionGeometry& geometry) {
    constexpr uint64_t warp_lanes = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS);
    constexpr uint64_t items_16 = 16 / sizeof(InputT);
    constexpr uint64_t items_8 = 8 / sizeof(InputT);
    constexpr uint64_t components_16 = warp_lanes * items_16;
    constexpr uint64_t components_8 = warp_lanes * items_8;

    const uint64_t row_bytes = geometry.inner_size * static_cast<uint64_t>(sizeof(InputT));

    // Keep the widest packet unless a single retained tile would use no more than half of a physical warp's
    // 16-byte component capacity. In that under-filled regime, halve the packet so all 32 lanes can participate.
    // Repeat once more for 4-byte packets, which is the production hard lower bound. Never choose a narrow
    // packet when the packed row stride is not naturally aligned to that width; awkward rows stay on the proven
    // 16-byte alignment-safe path.
    size_t packet_bytes = ADAPTIVE_COOPERATIVE_TILED_MAX_VECTOR_BYTES;
    if (geometry.inner_size <= components_16 / 2 && row_bytes % 8 == 0) {
        packet_bytes = 8;
        if (geometry.inner_size <= components_8 / 2 && row_bytes % 4 == 0) {
            packet_bytes = ADAPTIVE_COOPERATIVE_TILED_MIN_VECTOR_BYTES;
        }
    }
    return packet_bytes;
}

template <typename InputT>
[[nodiscard]] constexpr bool packetAdaptiveCooperativeInputTypeIsProductionSupported() {
    return std::is_same_v<InputT, __half> || std::is_same_v<InputT, __nv_bfloat16>
           || std::is_same_v<InputT, float>;
}

template <typename ReductionOpT>
[[nodiscard]] constexpr bool packetAdaptiveCooperativeReductionOpIsProductionSupported() {
    // The release census currently covers the additive value-reduction family. This includes Sum/Mean directly and
    // L1/L2/SumSquares through their existing FP32 input transforms/finalizers. Product/Min/Max remain on their
    // previously ordained tiled kernels until they receive equivalent calibration.
    return std::is_same_v<std::decay_t<ReductionOpT>, cuda::std::plus<float>>;
}

template <typename InputT>
[[nodiscard]] inline int choosePacketAdaptiveCooperativeTiledWarpsPerTile(const CubReductionGeometry& geometry) {
    switch (chooseAdaptiveCooperativeTiledPacketBytes<InputT>(geometry)) {
        case 16:
            return chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, 16>(geometry);
        case 8:
            return chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, 8>(geometry);
        case 4:
            return chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, 4>(geometry);
        default:
            throw std::logic_error("Packet-adaptive cooperative tiled reduction selected an invalid packet width.");
    }
}

template <typename InputT, typename ReductionOpT>
[[nodiscard]] bool shouldUsePacketAdaptiveCooperativeTiledReduction(const CubReductionGeometry& geometry) {
    if constexpr (!packetAdaptiveCooperativeInputTypeIsProductionSupported<InputT>()
                  || !packetAdaptiveCooperativeReductionOpIsProductionSupported<ReductionOpT>()) {
        return false;
    } else {
        // Keep production on the exact dense/natural-output contract measured by the release census. View/permuted,
        // pitched, payload-transpose, and shared-transpose cases retain their dedicated ownership/store policies.
        if (geometry.path != CubReductionPath::TiledFixedSegment
            || geometry.permutation_aware_tiled_geometry.has_value()
            || geometry.payload_transpose_tiled_geometry.has_value()
            || geometry.pitched_tiled_geometry.has_value()
            || geometry.tiled_output_permuted
            || geometry.tiled_output_shared_transpose
            || geometry.tiled_output_outer_stride != geometry.inner_size
            || geometry.tiled_output_inner_stride != 1) {
            return false;
        }

        using namespace CubReductionTiledPolicy;
        if (geometry.reduction_size < PACKET_COOPERATIVE_MIN_REDUCTION_SIZE
            || geometry.inner_size < PACKET_COOPERATIVE_MIN_INNER_SIZE
            || geometry.inner_size > PACKET_COOPERATIVE_MAX_INNER_SIZE) {
            return false;
        }

        // The many-output sweep produced a clean crossover: every measured 16/8/4-byte case whose packet-aware
        // geometry wanted >=4 physical warps per retained tile beat the existing direct production reducer, while
        // W=1/2 contains the direct/cooperative crossover. Keep that crossover on the established direct path rather
        // than encoding dtype/shape-specific thresholds.
        return static_cast<uint64_t>(choosePacketAdaptiveCooperativeTiledWarpsPerTile<InputT>(geometry))
               >= PACKET_COOPERATIVE_MIN_WARPS_PER_TILE;
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchAdaptiveCooperativeTiledFixedSegmentReduction(const InputT* input,
                                                         void* output,
                                                         DataType output_dtype,
                                                         const CubReductionGeometry& geometry,
                                                         ReductionOpT reduction_op,
                                                         float init,
                                                         InputTransformT input_transform,
                                                         OutputFinalizeT output_finalize,
                                                         float output_scale,
                                                         cudaStream_t stream) {
    launchAdaptiveCooperativeTiledFixedSegmentReductionForPacketBytes<InputT,
                                                                      ReductionOpT,
                                                                      InputTransformT,
                                                                      OutputFinalizeT,
                                                                      16>(input,
                                                                          output,
                                                                          output_dtype,
                                                                          geometry,
                                                                          reduction_op,
                                                                          init,
                                                                          input_transform,
                                                                          output_finalize,
                                                                          output_scale,
                                                                          stream);
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchPacketAdaptiveCooperativeTiledFixedSegmentReduction(const InputT* input,
                                                               void* output,
                                                               DataType output_dtype,
                                                               const CubReductionGeometry& geometry,
                                                               ReductionOpT reduction_op,
                                                               float init,
                                                               InputTransformT input_transform,
                                                               OutputFinalizeT output_finalize,
                                                               float output_scale,
                                                               cudaStream_t stream) {
    // launchTiledFixedSegmentReduction is instantiated for every storage type / reduction operation supported by
    // CubReduction. A normal runtime `if (shouldUse...)` at that call site does not prevent the compiler from
    // instantiating this function template. Keep the packet-aware kernel family behind the same compile-time support
    // boundary as the selector so FP8, Product, Min, Max, and other unsupported combinations never instantiate the
    // 16/8/4-byte packet geometry templates. Their established production paths remain unchanged.
    if constexpr (!packetAdaptiveCooperativeInputTypeIsProductionSupported<InputT>()
                  || !packetAdaptiveCooperativeReductionOpIsProductionSupported<ReductionOpT>()) {
        throw std::logic_error(
            "Packet-adaptive cooperative tiled reduction was launched for an unsupported input/op combination.");
    } else {
        switch (chooseAdaptiveCooperativeTiledPacketBytes<InputT>(geometry)) {
            case 16:
                launchAdaptiveCooperativeTiledFixedSegmentReductionForPacketBytes<InputT,
                                                                                  ReductionOpT,
                                                                                  InputTransformT,
                                                                                  OutputFinalizeT,
                                                                                  16>(input, output, output_dtype, geometry,
                                                                                      reduction_op, init, input_transform,
                                                                                      output_finalize, output_scale, stream);
                return;
            case 8:
                launchAdaptiveCooperativeTiledFixedSegmentReductionForPacketBytes<InputT,
                                                                                  ReductionOpT,
                                                                                  InputTransformT,
                                                                                  OutputFinalizeT,
                                                                                  8>(input, output, output_dtype, geometry,
                                                                                     reduction_op, init, input_transform,
                                                                                     output_finalize, output_scale, stream);
                return;
            case 4:
                launchAdaptiveCooperativeTiledFixedSegmentReductionForPacketBytes<InputT,
                                                                                  ReductionOpT,
                                                                                  InputTransformT,
                                                                                  OutputFinalizeT,
                                                                                  4>(input, output, output_dtype, geometry,
                                                                                     reduction_op, init, input_transform,
                                                                                     output_finalize, output_scale, stream);
                return;
            default:
                throw std::logic_error("Adaptive cooperative tiled reduction selected an invalid packet width.");
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int RowLanes>
__global__ void directTiledFixedSegmentReductionKernel(const InputT* input,
                                                       void* output,
                                                       DataType output_dtype,
                                                       uint64_t outer_size,
                                                       uint64_t reduction_size,
                                                       uint64_t inner_size,
                                                  uint64_t output_outer_stride,
                                                  uint64_t output_inner_stride,
                                                       int warps_per_tile,
                                                       ReductionOpT reduction_op,
                                                       float init,
                                                       InputTransformT input_transform,
                                                       OutputFinalizeT output_finalize,
                                                       float output_scale) {
    static_assert(RowLanes == 1 || RowLanes == 2 || RowLanes == 4 || RowLanes == 8 || RowLanes == 16);
    constexpr int components_per_warp = TILED_REDUCTION_WARP_THREADS / RowLanes;
    using WarpReduceT = cub::WarpReduce<float, RowLanes>;

    __shared__ typename WarpReduceT::TempStorage
        logical_warp_storage[TILED_REDUCTION_BLOCK_THREADS / RowLanes];
    __shared__ float warp_partials[TILED_REDUCTION_WARPS_PER_BLOCK * components_per_warp];

    const int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / warps_per_tile;
    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int group_in_block = physical_warp / warps_per_tile;
    const int warp_in_tile = physical_warp % warps_per_tile;
    const int component_in_tile = lane / RowLanes;
    const int row_lane = lane % RowLanes;
    const int logical_warp_in_block = static_cast<int>(threadIdx.x) / RowLanes;

    const uint64_t component_tiles = ceilDivideU64(inner_size, static_cast<uint64_t>(components_per_warp));
    const uint64_t total_work = outer_size * component_tiles;
    const uint64_t block_work_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(groups_per_block);

    for (uint64_t block_work_base =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(groups_per_block);
         block_work_base < total_work;
         block_work_base += block_work_stride) {
        const uint64_t work_index = block_work_base + static_cast<uint64_t>(group_in_block);
        const bool work_active = work_index < total_work;
        const uint64_t outer_index = work_active ? work_index / component_tiles : 0;
        const uint64_t component_tile = work_active ? work_index - outer_index * component_tiles : 0;
        const uint64_t component =
            component_tile * static_cast<uint64_t>(components_per_warp) + static_cast<uint64_t>(component_in_tile);
        const bool component_active = work_active && component < inner_size;

        float local = init;
        if (component_active) {
            const uint64_t first_row = static_cast<uint64_t>(warp_in_tile * RowLanes + row_lane);
            const uint64_t row_stride = static_cast<uint64_t>(warps_per_tile * RowLanes);
            if (first_row < reduction_size) {
                uint64_t input_index =
                    ((outer_index * reduction_size + first_row) * inner_size) + component;
                const uint64_t input_stride = row_stride * inner_size;
                for (uint64_t row = first_row; row < reduction_size; row += row_stride) {
                    local = reduction_op(local, input_transform(ToFp32<InputT>{}(input[input_index])));
                    input_index += input_stride;
                }
            }
        }

        float warp_partial = local;
        if constexpr (RowLanes > 1) {
            warp_partial = WarpReduceT(logical_warp_storage[logical_warp_in_block]).Reduce(local, reduction_op);
        } else {
            (void)logical_warp_storage;
            (void)logical_warp_in_block;
        }

        if (warps_per_tile == 1) {
            if (component_active && row_lane == 0) {
                const float finalized = output_finalize(warp_partial) * output_scale;
                storeFp32AsRuntimeDType(
                    output, output_dtype, tiledReductionOutputIndex(outer_index, component, output_outer_stride, output_inner_stride), finalized);
            }
            if constexpr (RowLanes > 1) {
                __syncwarp();
            }
        } else {
            if (row_lane == 0) {
                warp_partials[physical_warp * components_per_warp + component_in_tile] =
                    component_active ? warp_partial : init;
            }
            __syncthreads();

            if (warp_in_tile == 0 && row_lane == 0 && component_active) {
                float aggregate = init;
                const int first_warp_in_group = group_in_block * warps_per_tile;
                for (int cooperating_warp = 0; cooperating_warp < warps_per_tile; ++cooperating_warp) {
                    aggregate = reduction_op(
                        aggregate,
                        warp_partials[(first_warp_in_group + cooperating_warp) * components_per_warp
                                      + component_in_tile]);
                }
                const float finalized = output_finalize(aggregate) * output_scale;
                storeFp32AsRuntimeDType(
                    output, output_dtype, tiledReductionOutputIndex(outer_index, component, output_outer_stride, output_inner_stride), finalized);
            }
            __syncthreads();
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int RowLanes>
void launchDirectTiledFixedSegmentReductionForRowLanes(const InputT* input,
                                                       void* output,
                                                       DataType output_dtype,
                                                       const CubReductionGeometry& geometry,
                                                       ReductionOpT reduction_op,
                                                       float init,
                                                       InputTransformT input_transform,
                                                       OutputFinalizeT output_finalize,
                                                       float output_scale,
                                                       cudaStream_t stream) {
    constexpr uint64_t components_per_warp = TILED_REDUCTION_WARP_THREADS / RowLanes;
    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_warp);
    const int warps_per_tile = chooseDirectTiledReductionWarpsPerTile<RowLanes>(geometry);
    const int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / warps_per_tile;
    const uint64_t total_work = geometry.outer_size * component_tiles;
    const uint64_t required_blocks =
        ceilDivideU64(total_work, static_cast<uint64_t>(groups_per_block));
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    directTiledFixedSegmentReductionKernel<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, RowLanes>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                   output,
                                                                   output_dtype,
                                                                   geometry.outer_size,
                                                                   geometry.reduction_size,
                                                                   geometry.inner_size,
                                                                   geometry.tiled_output_outer_stride,
                                                                   geometry.tiled_output_inner_stride,
                                                                   warps_per_tile,
                                                                   reduction_op,
                                                                   init,
                                                                   input_transform,
                                                                   output_finalize,
                                                                   output_scale);
    CUDA_CHECK(cudaGetLastError());
}


// VIEW-PITCHED-TILED is intentionally a separate kernel family from the sealed dense TiledFixedSegment kernels above.
// The planner proves that every source address is affine in exactly three flattened coordinates:
//
//   outer * outer_stride + reduction * reduction_stride + inner
//
// Inner components are physically contiguous, so adjacent component groups issue coalesced global loads. RowLanes is a
// runtime power-of-two subwarp width rather than a template parameter: this avoids adding a new dtype/op/RowLanes kernel
// specialization matrix solely for pitched views. No logical-axis arrays, coordinate division/modulo, or per-element
// mixed-radix mapping is used.
template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
__global__ void pitchedTiledFixedSegmentReductionKernel(const InputT* input,
                                                        void* output,
                                                        DataType output_dtype,
                                                        uint64_t outer_size,
                                                        uint64_t reduction_size,
                                                        uint64_t inner_size,
                                                        uint64_t outer_stride,
                                                        uint64_t reduction_stride,
                                                        uint64_t output_outer_stride,
                                                        uint64_t output_inner_stride,
                                                        int row_lanes,
                                                        ReductionOpT reduction_op,
                                                        float init,
                                                        InputTransformT input_transform,
                                                        OutputFinalizeT output_finalize,
                                                        float output_scale) {
    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int component_in_warp = lane / row_lanes;
    const int row_lane = lane - component_in_warp * row_lanes;
    const int components_per_warp = TILED_REDUCTION_WARP_THREADS / row_lanes;

    const uint64_t component_tiles =
        ceilDivideU64(inner_size, static_cast<uint64_t>(components_per_warp));
    const uint64_t total_work = outer_size * component_tiles;
    const uint64_t first_work =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK)
        + static_cast<uint64_t>(physical_warp);
    const uint64_t work_stride =
        static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);

    for (uint64_t work_index = first_work; work_index < total_work; work_index += work_stride) {
        const uint64_t outer_index = work_index / component_tiles;
        const uint64_t component_tile = work_index - outer_index * component_tiles;
        const uint64_t component =
            component_tile * static_cast<uint64_t>(components_per_warp) + static_cast<uint64_t>(component_in_warp);
        const bool component_active = component < inner_size;

        float local = init;
        if (component_active && static_cast<uint64_t>(row_lane) < reduction_size) {
            uint64_t row = static_cast<uint64_t>(row_lane);
            uint64_t input_index = outer_index * outer_stride + row * reduction_stride + component;
            for (;;) {
                local = reduction_op(local, input_transform(ToFp32<InputT>{}(input[input_index])));
                if (reduction_size - row <= static_cast<uint64_t>(row_lanes)) {
                    break;
                }
                row += static_cast<uint64_t>(row_lanes);
                // The planner proved every reachable row offset fits uint64_t. Only form the row-step product when a
                // next row actually exists, so even extreme but valid pitches cannot overflow on an unused increment.
                input_index += static_cast<uint64_t>(row_lanes) * reduction_stride;
            }
        }

        // row_lanes is uniform for the launch and always a power of two <= warpSize. CUDA's width argument partitions
        // the physical warp into independent logical subwarps, so reductions never exchange values between adjacent
        // retained components even though every lane participates in the shuffle instruction.
        for (int offset = row_lanes / 2; offset > 0; offset /= 2) {
            const float other = __shfl_down_sync(0xffffffffu, local, offset, row_lanes);
            if (row_lane < offset) {
                local = reduction_op(local, other);
            }
        }

        if (component_active && row_lane == 0) {
            const float finalized = output_finalize(local) * output_scale;
            storeFp32AsRuntimeDType(
                output,
                output_dtype,
                tiledReductionOutputIndex(
                    outer_index, component, output_outer_stride, output_inner_stride),
                finalized);
        }
    }
}

[[nodiscard]] inline int choosePitchedTiledRowLanes(const CubReductionGeometry& geometry) {
    // Use the largest useful logical subwarp that still leaves enough independent component groups in each physical
    // warp. Wide payloads stay at RowLanes=1 for maximally coalesced component traffic; narrow payloads expose row
    // parallelism when the reduction domain is large enough to use it.
    if (geometry.inner_size <= 1 && geometry.reduction_size >= 32) {
        return 32;
    }
    if (geometry.inner_size <= 2 && geometry.reduction_size >= 16) {
        return 16;
    }
    if (geometry.inner_size <= 4 && geometry.reduction_size >= 8) {
        return 8;
    }
    if (geometry.inner_size <= 8 && geometry.reduction_size >= 4) {
        return 4;
    }
    if (geometry.inner_size <= 16 && geometry.reduction_size >= 2) {
        return 2;
    }
    return 1;
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchPitchedTiledFixedSegmentReduction(const InputT* input,
                                             void* output,
                                             DataType output_dtype,
                                             const CubReductionGeometry& geometry,
                                             ReductionOpT reduction_op,
                                             float init,
                                             InputTransformT input_transform,
                                             OutputFinalizeT output_finalize,
                                             float output_scale,
                                             cudaStream_t stream) {
    if (!geometry.pitched_tiled_geometry.has_value() || geometry.tiled_output_permuted
        || geometry.tiled_output_shared_transpose || geometry.tiled_output_outer_stride != geometry.inner_size
        || geometry.tiled_output_inner_stride != 1) {
        throw std::logic_error(
            "Pitched tiled CUB reduction requires natural dense retained-output geometry.");
    }
    const CubReductionPitchedTiledGeometry& pitched = geometry.pitched_tiled_geometry.value();
    const int row_lanes = choosePitchedTiledRowLanes(geometry);
    const int components_per_warp = TILED_REDUCTION_WARP_THREADS / row_lanes;
    const uint64_t component_tiles =
        ceilDivideU64(geometry.inner_size, static_cast<uint64_t>(components_per_warp));
    const uint64_t total_work = geometry.outer_size * component_tiles;
    const uint64_t required_blocks =
        ceilDivideU64(total_work, static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK));
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    pitchedTiledFixedSegmentReductionKernel<InputT, ReductionOpT, InputTransformT, OutputFinalizeT>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                   output,
                                                                   output_dtype,
                                                                   geometry.outer_size,
                                                                   geometry.reduction_size,
                                                                   geometry.inner_size,
                                                                   pitched.outer_stride,
                                                                   pitched.reduction_stride,
                                                                   geometry.tiled_output_outer_stride,
                                                                   geometry.tiled_output_inner_stride,
                                                                   row_lanes,
                                                                   reduction_op,
                                                                   init,
                                                                   input_transform,
                                                                   output_finalize,
                                                                   output_scale);
    CUDA_CHECK(cudaGetLastError());
}

// VIEW-DIRECT-2B is a separate compact-permutation strategy. The planner has already flattened the physical source to
// [A,reduction,B,payload] and proved that Thor's logical dense output is [B,A,payload]. Each producer warp owns one A
// row at a time and keeps adjacent payload lanes contiguous on every global read. Finalized values are staged through a
// fixed 32x33 FP32 shared-memory tile; the +1 padding deliberately favors coalesced global traffic even if an awkward
// payload width still creates a small shared-memory bank-conflict pattern in the consumer pass. No global transpose
// intermediate or logical-coordinate mapper is involved.
constexpr uint64_t PAYLOAD_TRANSPOSE_TILE_ROWS = 32;
constexpr uint64_t PAYLOAD_TRANSPOSE_TILE_COLUMNS = 32;
constexpr uint64_t PAYLOAD_TRANSPOSE_SHARED_PITCH = 33;
static_assert(PAYLOAD_TRANSPOSE_TILE_ROWS % TILED_REDUCTION_WARPS_PER_BLOCK == 0);
static_assert(PAYLOAD_TRANSPOSE_TILE_COLUMNS == TILED_REDUCTION_WARP_THREADS);

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
__global__ void payloadTranspose32x33ReductionKernel(const InputT* input,
                                                     void* output,
                                                     DataType output_dtype,
                                                     uint64_t a_size,
                                                     uint64_t reduction_size,
                                                     uint64_t b_size,
                                                     uint64_t payload_size,
                                                     uint64_t a_tiles,
                                                     uint64_t payload_tiles,
                                                     ReductionOpT reduction_op,
                                                     float init,
                                                     InputTransformT input_transform,
                                                     OutputFinalizeT output_finalize,
                                                     float output_scale) {
    __shared__ float retained_tile[PAYLOAD_TRANSPOSE_TILE_ROWS][PAYLOAD_TRANSPOSE_SHARED_PITCH];

    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    constexpr uint64_t a_rows_per_warp = PAYLOAD_TRANSPOSE_TILE_ROWS / TILED_REDUCTION_WARPS_PER_BLOCK;

    const uint64_t total_work = b_size * a_tiles * payload_tiles;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);
    const uint64_t reduction_row_stride = b_size * payload_size;
    const uint64_t a_stride = reduction_size * reduction_row_stride;

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work;
         work_index += grid_stride) {
        const uint64_t payload_tile_index = work_index % payload_tiles;
        const uint64_t outer_work = work_index / payload_tiles;
        const uint64_t a_tile_index = outer_work % a_tiles;
        const uint64_t b_index = outer_work / a_tiles;
        const uint64_t a_base = a_tile_index * PAYLOAD_TRANSPOSE_TILE_ROWS;
        const uint64_t payload_base = payload_tile_index * PAYLOAD_TRANSPOSE_TILE_COLUMNS;
        const uint64_t payload_index = payload_base + static_cast<uint64_t>(lane);
        const bool payload_active = payload_index < payload_size;

#pragma unroll
        for (uint64_t a_pass = 0; a_pass < a_rows_per_warp; ++a_pass) {
            const uint64_t a_local =
                static_cast<uint64_t>(physical_warp) + a_pass * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
            const uint64_t a_index = a_base + a_local;
            if (a_index >= a_size || !payload_active) {
                continue;
            }

            float local = init;
            uint64_t input_index = a_index * a_stride + b_index * payload_size + payload_index;
            for (uint64_t reduction_index = 0; reduction_index < reduction_size; ++reduction_index) {
                local = reduction_op(local, input_transform(ToFp32<InputT>{}(input[input_index])));
                input_index += reduction_row_stride;
            }
            retained_tile[a_local][lane] = output_finalize(local) * output_scale;
        }
        __syncthreads();

        const uint64_t valid_a_rows = minU64(PAYLOAD_TRANSPOSE_TILE_ROWS, a_size - a_base);
        const uint64_t valid_payload = minU64(PAYLOAD_TRANSPOSE_TILE_COLUMNS, payload_size - payload_base);
        const uint64_t staged_values = valid_a_rows * valid_payload;

        // When payload_size <= 32 this consumer walk is exactly one dense [A,payload] output interval for fixed B,
        // so consecutive threads issue fully coalesced writes even across A-row boundaries. Wider payloads retain
        // coalescing within each 32-value payload packet while reusing the same fixed shared-memory footprint.
        for (uint64_t linear = static_cast<uint64_t>(threadIdx.x); linear < staged_values;
             linear += static_cast<uint64_t>(blockDim.x)) {
            const uint64_t a_local = linear / valid_payload;
            const uint64_t payload_local = linear - a_local * valid_payload;
            const uint64_t output_index =
                ((b_index * a_size + (a_base + a_local)) * payload_size) + payload_base + payload_local;
            storeFp32AsRuntimeDType(
                output, output_dtype, output_index, retained_tile[a_local][payload_local]);
        }
        __syncthreads();
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchPayloadTranspose32x33Reduction(const InputT* input,
                                          void* output,
                                          DataType output_dtype,
                                          const CubReductionGeometry& geometry,
                                          ReductionOpT reduction_op,
                                          float init,
                                          InputTransformT input_transform,
                                          OutputFinalizeT output_finalize,
                                          float output_scale,
                                          cudaStream_t stream) {
    if (!geometry.payload_transpose_tiled_geometry.has_value() || geometry.pitched_tiled_geometry.has_value()) {
        throw std::logic_error("VIEW-DIRECT-2B requires exclusive payload-transpose tiled geometry.");
    }
    const CubReductionPayloadTransposeTiledGeometry& transposed =
        geometry.payload_transpose_tiled_geometry.value();
    if (transposed.a_size <= 1 || transposed.b_size <= 1 || transposed.payload_size == 0
        || transposed.reduction_size != geometry.reduction_size) {
        throw std::logic_error("VIEW-DIRECT-2B payload-transpose geometry is internally inconsistent.");
    }

    const uint64_t a_tiles = ceilDivideU64(transposed.a_size, PAYLOAD_TRANSPOSE_TILE_ROWS);
    const uint64_t payload_tiles = ceilDivideU64(transposed.payload_size, PAYLOAD_TRANSPOSE_TILE_COLUMNS);
    const uint64_t total_work = transposed.b_size * a_tiles * payload_tiles;
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));

    payloadTranspose32x33ReductionKernel<InputT, ReductionOpT, InputTransformT, OutputFinalizeT>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                   output,
                                                                   output_dtype,
                                                                   transposed.a_size,
                                                                   transposed.reduction_size,
                                                                   transposed.b_size,
                                                                   transposed.payload_size,
                                                                   a_tiles,
                                                                   payload_tiles,
                                                                   reduction_op,
                                                                   init,
                                                                   input_transform,
                                                                   output_finalize,
                                                                   output_scale);
    CUDA_CHECK(cudaGetLastError());
}

// Wide fallback for permuted retained output when the tuned full-row family assigns multiple warps (or an entire CTA)
// to one outer row. Those CTAs do not own adjacent outer rows, so shared-memory transposition cannot be expressed as a
// pure epilogue without changing work ownership. Keep that exception isolated here; <=512 components use the tuned
// reducer itself with a shared-transpose output epilogue.
template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
__global__ void wideComponentTiledSharedTransposeReductionKernel(const InputT* input,
                                                                        void* output,
                                                                        DataType output_dtype,
                                                                        uint64_t outer_size,
                                                                        uint64_t reduction_size,
                                                                        uint64_t inner_size,
                                                                        uint64_t component_tiles,
                                                                        uint64_t outer_groups,
                                                                        ReductionOpT reduction_op,
                                                                        float init,
                                                                        InputTransformT input_transform,
                                                                        OutputFinalizeT output_finalize,
                                                                        float output_scale) {
    static_assert(ItemsPerLane == 1 || ItemsPerLane == 2 || ItemsPerLane == 4 || ItemsPerLane == 8
                  || ItemsPerLane == SHARED_TRANSPOSE_MAX_ITEMS_PER_LANE);
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    // Producers own contiguous ItemsPerLane packets. Store those packets as [item][lane] stripes so a warp writes 32
    // distinct shared-memory banks instead of lane*ItemsPerLane bank strides. The stripe/row paddings are selected so
    // the transposed consumer mapping (four components x eight outer rows per warp instruction) is also bank-conflict
    // free for each supported ItemsPerLane.
    constexpr uint64_t shared_stripe_pitch = ItemsPerLane == 2 ? 34 : 33;
    constexpr uint64_t shared_row_unpadded = static_cast<uint64_t>(ItemsPerLane) * shared_stripe_pitch;
    constexpr uint64_t shared_row_pitch =
        shared_row_unpadded + ((4 + 32 - (shared_row_unpadded % 32)) % 32);
    constexpr size_t packet_bytes = sizeof(InputT) * ItemsPerLane;

    extern __shared__ float retained_tile[];

    const int outer_lane = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t total_tile_work = component_tiles * outer_groups;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t tile_work = static_cast<uint64_t>(blockIdx.x); tile_work < total_tile_work;
         tile_work += grid_stride) {
        const uint64_t component_tile = tile_work / outer_groups;
        const uint64_t outer_group = tile_work - component_tile * outer_groups;
        const uint64_t outer_index =
            outer_group * static_cast<uint64_t>(SHARED_TRANSPOSE_OUTER_ROWS) + static_cast<uint64_t>(outer_lane);
        const uint64_t tile_component_begin = component_tile * components_per_tile;
        const uint64_t component_begin =
            tile_component_begin + static_cast<uint64_t>(lane) * static_cast<uint64_t>(ItemsPerLane);
        const bool outer_active = outer_index < outer_size;
        const bool packet_fully_active = outer_active && component_begin < inner_size
                                         && static_cast<uint64_t>(ItemsPerLane) <= inner_size - component_begin;

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = init;
        }

        if (outer_active && component_begin < inner_size) {
            uint64_t row_base = outer_index * reduction_size * inner_size + component_begin;
            for (uint64_t row = 0; row < reduction_size; ++row) {
                if (packet_fully_active) {
                    if constexpr (packet_bytes >= 16) {
                        // Arbitrary retained widths can change packet alignment from one reduction row to the next.
                        // The existing aligned-window loader preserves vectorized input traffic without introducing a
                        // scalar tail path; all requested values remain within the current physical row.
                        const PackedInputValues<InputT, ItemsPerLane> values =
                            loadAlignmentSafeInputPacket<InputT, ItemsPerLane>(input + row_base);
#pragma unroll
                        for (int item = 0; item < ItemsPerLane; ++item) {
                            local[item] = reduction_op(
                                local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
                        }
                    } else {
                        const InputT* source = input + row_base;
                        if constexpr (packet_bytes == 2 || packet_bytes == 4 || packet_bytes == 8) {
                            using RawT = RawVectorPacket<packet_bytes>;
                            if ((reinterpret_cast<uintptr_t>(source) & (alignof(RawT) - 1U)) == 0) {
                                const PackedInputValues<InputT, ItemsPerLane> values =
                                    loadVectorizedInputPacket<InputT, ItemsPerLane>(source);
#pragma unroll
                                for (int item = 0; item < ItemsPerLane; ++item) {
                                    local[item] = reduction_op(
                                        local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
                                }
                            } else {
#pragma unroll
                                for (int item = 0; item < ItemsPerLane; ++item) {
                                    local[item] = reduction_op(
                                        local[item], input_transform(ToFp32<InputT>{}(source[item])));
                                }
                            }
                        } else {
                            static_assert(packet_bytes == 1);
                            local[0] = reduction_op(local[0], input_transform(ToFp32<InputT>{}(source[0])));
                        }
                    }
                } else {
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item) {
                        const uint64_t component = component_begin + static_cast<uint64_t>(item);
                        if (component < inner_size) {
                            local[item] = reduction_op(
                                local[item], input_transform(ToFp32<InputT>{}(input[row_base + item])));
                        }
                    }
                }
                row_base += inner_size;
            }
        }

        // Stage exactly the finalized retained values. Reduction state never leaves registers and there is no global
        // [outer,inner] temporary. The [item][lane] shared stripes make producer writes conflict-free; the consumer
        // mapping below reads those stripes as a retained-axis transpose and emits adjacent outer coordinates
        // contiguously in dense [inner,outer].
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component_offset =
                static_cast<uint64_t>(lane) * static_cast<uint64_t>(ItemsPerLane) + static_cast<uint64_t>(item);
            const uint64_t component = tile_component_begin + component_offset;
            if (outer_active && component < inner_size) {
                retained_tile[static_cast<uint64_t>(outer_lane) * shared_row_pitch
                              + static_cast<uint64_t>(item) * shared_stripe_pitch + static_cast<uint64_t>(lane)] =
                    output_finalize(local[item]) * output_scale;
            }
        }
        __syncthreads();

        const uint64_t valid_components = minU64(components_per_tile, inner_size - tile_component_begin);
        const uint64_t staged_values =
            valid_components * static_cast<uint64_t>(SHARED_TRANSPOSE_OUTER_ROWS);
        for (uint64_t linear = static_cast<uint64_t>(threadIdx.x); linear < staged_values;
             linear += static_cast<uint64_t>(blockDim.x)) {
            const uint64_t component_offset =
                linear / static_cast<uint64_t>(SHARED_TRANSPOSE_OUTER_ROWS);
            const uint64_t output_outer_offset =
                linear - component_offset * static_cast<uint64_t>(SHARED_TRANSPOSE_OUTER_ROWS);
            const uint64_t output_outer =
                outer_group * static_cast<uint64_t>(SHARED_TRANSPOSE_OUTER_ROWS) + output_outer_offset;
            if (output_outer < outer_size) {
                const uint64_t component = tile_component_begin + component_offset;
                storeFp32AsRuntimeDType(output,
                                        output_dtype,
                                        component * outer_size + output_outer,
                                        retained_tile[output_outer_offset * shared_row_pitch
                                                      + (component_offset % static_cast<uint64_t>(ItemsPerLane))
                                                            * shared_stripe_pitch
                                                      + component_offset / static_cast<uint64_t>(ItemsPerLane)]);
            }
        }

        // A grid-stride iteration may reuse the same shared tile; all consumers must finish before producers overwrite
        // it on the next iteration.
        __syncthreads();
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
void launchWideComponentTiledSharedTransposeReductionForItems(const InputT* input,
                                                                     void* output,
                                                                     DataType output_dtype,
                                                                     const CubReductionGeometry& geometry,
                                                                     ReductionOpT reduction_op,
                                                                     float init,
                                                                     InputTransformT input_transform,
                                                                     OutputFinalizeT output_finalize,
                                                                     float output_scale,
                                                                     cudaStream_t stream) {
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    constexpr uint64_t shared_stripe_pitch = ItemsPerLane == 2 ? 34 : 33;
    constexpr uint64_t shared_row_unpadded = static_cast<uint64_t>(ItemsPerLane) * shared_stripe_pitch;
    constexpr uint64_t shared_row_pitch =
        shared_row_unpadded + ((4 + 32 - (shared_row_unpadded % 32)) % 32);
    constexpr size_t shared_bytes =
        static_cast<size_t>(SHARED_TRANSPOSE_OUTER_ROWS) * static_cast<size_t>(shared_row_pitch) * sizeof(float);

    const uint64_t component_tiles = ceilDivideU64(geometry.inner_size, components_per_tile);
    const uint64_t outer_groups =
        ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(SHARED_TRANSPOSE_OUTER_ROWS));
    if (component_tiles != 0 && outer_groups > std::numeric_limits<uint64_t>::max() / component_tiles) {
        throw std::invalid_argument("Shared-transpose tiled reduction work geometry overflows uint64_t.");
    }
    const uint64_t total_tile_work = component_tiles * outer_groups;
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_tile_work, TILED_REDUCTION_MAX_GRID_BLOCKS));

    wideComponentTiledSharedTransposeReductionKernel<InputT,
                                                            ReductionOpT,
                                                            InputTransformT,
                                                            OutputFinalizeT,
                                                            ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, shared_bytes, stream>>>(input,
                                                                              output,
                                                                              output_dtype,
                                                                              geometry.outer_size,
                                                                              geometry.reduction_size,
                                                                              geometry.inner_size,
                                                                              component_tiles,
                                                                              outer_groups,
                                                                              reduction_op,
                                                                              init,
                                                                              input_transform,
                                                                              output_finalize,
                                                                              output_scale);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchWideComponentTiledSharedTransposeReduction(const InputT* input,
                                                              void* output,
                                                              DataType output_dtype,
                                                              const CubReductionGeometry& geometry,
                                                              ReductionOpT reduction_op,
                                                              float init,
                                                              InputTransformT input_transform,
                                                              OutputFinalizeT output_finalize,
                                                              float output_scale,
                                                              cudaStream_t stream) {
    if (!geometry.tiled_output_permuted || !geometry.tiled_output_shared_transpose) {
        throw std::logic_error("Shared-transpose tiled reduction requires permuted retained-output geometry.");
    }
    if (geometry.tiled_output_outer_stride != 1 || geometry.tiled_output_inner_stride != geometry.outer_size) {
        throw std::logic_error("Shared-transpose tiled reduction requires dense [inner,outer] output strides.");
    }
    if (geometry.inner_size <= 1
        || (!geometry.reduced_axes_are_contiguous && !geometry.permutation_aware_tiled_geometry.has_value())) {
        throw std::logic_error(
            "Shared-transpose tiled reduction requires a dense physical [outer,reduction,inner] traversal.");
    }

    // Match the existing full-row register budget: each lane owns at most sixteen adjacent retained components. Wider
    // outputs are tiled in 512-component chunks, so a very wide retained dimension increases independent CTA work
    // rather than per-thread register pressure.
    if (geometry.inner_size <= 32) {
        launchWideComponentTiledSharedTransposeReductionForItems<InputT,
                                                                        ReductionOpT,
                                                                        InputTransformT,
                                                                        OutputFinalizeT,
                                                                        1>(input,
                                                                           output,
                                                                           output_dtype,
                                                                           geometry,
                                                                           reduction_op,
                                                                           init,
                                                                           input_transform,
                                                                           output_finalize,
                                                                           output_scale,
                                                                           stream);
    } else if (geometry.inner_size <= 64) {
        launchWideComponentTiledSharedTransposeReductionForItems<InputT,
                                                                        ReductionOpT,
                                                                        InputTransformT,
                                                                        OutputFinalizeT,
                                                                        2>(input,
                                                                           output,
                                                                           output_dtype,
                                                                           geometry,
                                                                           reduction_op,
                                                                           init,
                                                                           input_transform,
                                                                           output_finalize,
                                                                           output_scale,
                                                                           stream);
    } else if (geometry.inner_size <= 128) {
        launchWideComponentTiledSharedTransposeReductionForItems<InputT,
                                                                        ReductionOpT,
                                                                        InputTransformT,
                                                                        OutputFinalizeT,
                                                                        4>(input,
                                                                           output,
                                                                           output_dtype,
                                                                           geometry,
                                                                           reduction_op,
                                                                           init,
                                                                           input_transform,
                                                                           output_finalize,
                                                                           output_scale,
                                                                           stream);
    } else if (geometry.inner_size <= 256) {
        launchWideComponentTiledSharedTransposeReductionForItems<InputT,
                                                                        ReductionOpT,
                                                                        InputTransformT,
                                                                        OutputFinalizeT,
                                                                        8>(input,
                                                                           output,
                                                                           output_dtype,
                                                                           geometry,
                                                                           reduction_op,
                                                                           init,
                                                                           input_transform,
                                                                           output_finalize,
                                                                           output_scale,
                                                                           stream);
    } else {
        launchWideComponentTiledSharedTransposeReductionForItems<InputT,
                                                                        ReductionOpT,
                                                                        InputTransformT,
                                                                        OutputFinalizeT,
                                                                        SHARED_TRANSPOSE_MAX_ITEMS_PER_LANE>(
            input,
            output,
            output_dtype,
            geometry,
            reduction_op,
            init,
            input_transform,
            output_finalize,
            output_scale,
            stream);
    }
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchTiledFixedSegmentReduction(const InputT* input,
                                      void* output,
                                      DataType output_dtype,
                                      const CubReductionGeometry& geometry,
                                      ReductionOpT reduction_op,
                                      float init,
                                      InputTransformT input_transform,
                                      OutputFinalizeT output_finalize,
                                      float output_scale,
                                      cudaStream_t stream) {
    // geometry.path == TiledFixedSegment guarantees a dense physical [outer,reduction,inner] source traversal.
    // For ordinary dense tensors that follows from contiguous logical reduction axes; for zero-copy permutation views
    // it follows from permutation_aware_tiled_geometry instead. The CUDA kernels only consume the physical extents.
    if (geometry.inner_size <= 1
        || (!geometry.reduced_axes_are_contiguous && !geometry.permutation_aware_tiled_geometry.has_value())) {
        throw std::logic_error(
            "Tiled CUB reduction requires a dense physical [outer,reduction,inner] traversal with trailing width > 1.");
    }

    if (geometry.tiled_output_shared_transpose) {
        if (!geometry.tiled_output_permuted || geometry.tiled_output_outer_stride != 1
            || geometry.tiled_output_inner_stride != geometry.outer_size) {
            throw std::logic_error(
                "Shared-transpose tiled reduction requires dense permuted [inner,outer] output geometry.");
        }

        // Up through one-warp-per-output widths, shared transpose is strictly an epilogue of the same tuned reducer
        // selected below. The only <=512 exception is an awkward 257..511 row that does not fit the async staging
        // capacity for a wide InputT; that shape already misses the tuned full-row family, so use the component-tiled
        // shared fallback rather than accidentally dropping back to uncoalesced stores. Wider tuned kernels can assign
        // multiple warps (or a whole CTA) to one outer row, so a CTA no longer owns enough adjacent outer rows to form
        // a useful transpose tile. Keep that inherently different ownership regime isolated in the fallback.
        const uint64_t async_stage_capacity =
            ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP / static_cast<uint64_t>(sizeof(InputT));
        const bool one_warp_tuned_shared_epilogue =
            geometry.inner_size <= 256 || geometry.inner_size == FULL_ROW_COMPONENTS_PER_WARP
            || (geometry.inner_size < FULL_ROW_COMPONENTS_PER_WARP
                && geometry.inner_size <= async_stage_capacity);
        if (!one_warp_tuned_shared_epilogue) {
            launchWideComponentTiledSharedTransposeReduction<InputT>(input,
                                                                      output,
                                                                      output_dtype,
                                                                      geometry,
                                                                      reduction_op,
                                                                      init,
                                                                      input_transform,
                                                                      output_finalize,
                                                                      output_scale,
                                                                      stream);
            return;
        }
    }

    if (shouldUsePacketAdaptiveCooperativeTiledReduction<InputT, ReductionOpT>(geometry)) {
        launchPacketAdaptiveCooperativeTiledFixedSegmentReduction<InputT>(input,
                                                                           output,
                                                                           output_dtype,
                                                                           geometry,
                                                                           reduction_op,
                                                                           init,
                                                                           input_transform,
                                                                           output_finalize,
                                                                           output_scale,
                                                                           stream);
        return;
    }

    // Ordinary dense RK is stamped through ReducersDenseRK before reaching this function. Keep only the generic
    // component-tiled fallback for ordained view geometries and defensive direct callers. Historical direct/full-row,
    // grouped-full-row, block-sharded, and row-split VALUE kernels are deliberately not referenced here, so they are
    // no longer instantiated into every VALUE-operation translation unit.
    launchDirectTiledFixedSegmentReductionForRowLanes<InputT,
                                                      ReductionOpT,
                                                      InputTransformT,
                                                      OutputFinalizeT,
                                                      1>(input,
                                                         output,
                                                         output_dtype,
                                                         geometry,
                                                         reduction_op,
                                                         init,
                                                         input_transform,
                                                         output_finalize,
                                                         output_scale,
                                                         stream);
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
size_t queryReductionBytesForInput(const InputT* input,
                                   uint64_t input_elements,
                                   void* output,
                                   DataType output_dtype,
                                   const CubReductionGeometry& geometry,
                                   ReductionOpT reduction_op,
                                   float init,
                                   InputTransformT input_transform,
                                   OutputFinalizeT output_finalize,
                                   float output_scale,
                                   int gpu_num,
                                   cudaStream_t stream) {
    using AccumulatorT =
        std::decay_t<decltype(std::declval<ReductionOpT>()(std::declval<float>(), std::declval<float>()))>;
    static_assert(std::is_same_v<AccumulatorT, float>, "CUB tensor reductions must accumulate in FP32.");
    static_assert(std::is_same_v<decltype(std::declval<InputTransformT>()(std::declval<float>())), float>,
                  "CUB tensor reduction input transforms must produce FP32.");
    static_assert(std::is_same_v<decltype(std::declval<OutputFinalizeT>()(std::declval<float>())), float>,
                  "CUB tensor reduction output finalizers must produce FP32.");

    size_t queried_bytes = 0;
    auto output_iterator =
        makeRuntimeFp32OutputIterator(output, output_dtype, output_finalize, output_scale);
    const ConvertAndTransformInputToFp32<InputT, InputTransformT> device_input_transform{input_transform};

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce:
            if (geometry.device_transform_uses_affine_stride) {
                auto input_iterator = makeAffineStridedFp32Iterator<InputT>(
                    input, geometry.affine_input_stride, input_transform);
                CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr,
                                                     queried_bytes,
                                                     input_iterator,
                                                     output_iterator,
                                                     static_cast<int64_t>(input_elements),
                                                     reduction_op,
                                                     init,
                                                     stream));
            } else {
                CUDA_CHECK(cub::DeviceReduce::TransformReduce(nullptr,
                                                              queried_bytes,
                                                              input,
                                                              output_iterator,
                                                              static_cast<int64_t>(input_elements),
                                                              reduction_op,
                                                              device_input_transform,
                                                              init,
                                                              stream));
            }
            break;
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousFp32Iterator(input, input_transform);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                          queried_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::TiledFixedSegment:
            // Ordinary dense RK workspace is planned by CubReduction::queryPhysicalStageWorkspace(). The retained
            // generic/view Tiled path itself owns no temporary tensor.
            queried_bytes = 1;
            break;
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB reduction received offset-segmented geometry.");
        case CubReductionPath::ComposedDense:
            throw std::logic_error(
                "Composed dense VALUE geometry must be materialized into direct physical passes before workspace query.");
    }

    return std::max<size_t>(queried_bytes, 1);
}

template <typename InputT, typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchReductionForInput(const Tensor& temp_storage,
                             size_t temp_storage_bytes,
                             const Tensor& input,
                             Tensor& output,
                             const CubReductionGeometry& geometry,
                             ReductionOpT reduction_op,
                             float init,
                             InputTransformT input_transform,
                             OutputFinalizeT output_finalize,
                             float output_scale,
                             cudaStream_t stream) {
    using AccumulatorT =
        std::decay_t<decltype(std::declval<ReductionOpT>()(std::declval<float>(), std::declval<float>()))>;
    static_assert(std::is_same_v<AccumulatorT, float>, "CUB tensor reductions must accumulate in FP32.");

    void* temp_storage_ptr =
        const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
    auto output_iterator =
        makeRuntimeFp32OutputIterator(output.getMemPtr<void>(), output.getDataType(), output_finalize, output_scale);
    const ConvertAndTransformInputToFp32<InputT, InputTransformT> device_input_transform{input_transform};

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce:
            if (geometry.device_transform_uses_affine_stride) {
                auto input_iterator = makeAffineStridedFp32Iterator<InputT>(
                    input.getMemPtr<InputT>(), geometry.affine_input_stride, input_transform);
                CUDA_CHECK(cub::DeviceReduce::Reduce(temp_storage_ptr,
                                                     temp_storage_bytes,
                                                     input_iterator,
                                                     output_iterator,
                                                     static_cast<int64_t>(input.getTotalNumElements()),
                                                     reduction_op,
                                                     init,
                                                     stream));
            } else {
                CUDA_CHECK(cub::DeviceReduce::TransformReduce(temp_storage_ptr,
                                                              temp_storage_bytes,
                                                              input.getMemPtr<InputT>(),
                                                              output_iterator,
                                                              static_cast<int64_t>(input.getTotalNumElements()),
                                                              reduction_op,
                                                              device_input_transform,
                                                              init,
                                                              stream));
            }
            break;
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousFp32Iterator(input.getMemPtr<InputT>(), input_transform);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::TiledFixedSegment:
            if (geometry.payload_transpose_tiled_geometry.has_value()) {
                launchPayloadTranspose32x33Reduction<InputT>(input.getMemPtr<InputT>(),
                                                              output.getMemPtr<void>(),
                                                              output.getDataType(),
                                                              geometry,
                                                              reduction_op,
                                                              init,
                                                              input_transform,
                                                              output_finalize,
                                                              output_scale,
                                                              stream);
            } else if (geometry.pitched_tiled_geometry.has_value()) {
                launchPitchedTiledFixedSegmentReduction<InputT>(input.getMemPtr<InputT>(),
                                                                 output.getMemPtr<void>(),
                                                                 output.getDataType(),
                                                                 geometry,
                                                                 reduction_op,
                                                                 init,
                                                                 input_transform,
                                                                 output_finalize,
                                                                 output_scale,
                                                                 stream);
            } else {
                launchTiledFixedSegmentReduction<InputT>(input.getMemPtr<InputT>(),
                                                          output.getMemPtr<void>(),
                                                          output.getDataType(),
                                                          geometry,
                                                          reduction_op,
                                                          init,
                                                          input_transform,
                                                          output_finalize,
                                                          output_scale,
                                                          stream);
            }
            break;
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB reduction received offset-segmented geometry.");
        case CubReductionPath::ComposedDense:
            throw std::logic_error(
                "Composed dense VALUE geometry must be materialized into direct physical passes before launch.");
    }
}

template <typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
size_t queryOperationReductionBytes(DataType input_dtype,
                                    const void* input,
                                    uint64_t input_elements,
                                    DataType output_dtype,
                                    void* output,
                                    const CubReductionGeometry& geometry,
                                    ReductionOpT reduction_op,
                                    float init,
                                    InputTransformT input_transform,
                                    OutputFinalizeT output_finalize,
                                    float output_scale,
                                    const Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> size_t {
        return queryReductionBytesForInput<InputT>(static_cast<const InputT*>(input),
                                                   input_elements,
                                                   output,
                                                   output_dtype,
                                                   geometry,
                                                   reduction_op,
                                                   init,
                                                   input_transform,
                                                   output_finalize,
                                                   output_scale,
                                                   stream.getGpuNum(),
                                                   stream.getStream());
    };
    return dispatchReductionInputDType(input_dtype, dispatch_input);
}

template <typename ReductionOpT, typename InputTransformT, typename OutputFinalizeT>
void launchOperationReduction(const Tensor& temp_storage,
                              size_t temp_storage_bytes,
                              const Tensor& input,
                              Tensor& output,
                              const CubReductionGeometry& geometry,
                              ReductionOpT reduction_op,
                              float init,
                              InputTransformT input_transform,
                              OutputFinalizeT output_finalize,
                              float output_scale,
                              Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> void {
        launchReductionForInput<InputT>(temp_storage,
                                        temp_storage_bytes,
                                        input,
                                        output,
                                        geometry,
                                        reduction_op,
                                        init,
                                        input_transform,
                                        output_finalize,
                                        output_scale,
                                        stream.getStream());
    };
    dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

}  // namespace ThorImplementation::CubReductionInternal
