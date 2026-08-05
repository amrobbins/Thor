#pragma once

#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReductionIndexing.cuh"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda/std/bit>
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

    // Sum and mean intentionally share this finalizer type so the complete additive reduction kernel family is
    // instantiated only once. The distinction is paid once per final aggregate rather than in the reduction loop.
    __host__ __device__ float operator()(float value) const {
        if (divisor == 0.0f) {
            return 0.0f;
        }
        if (divisor == 1.0f) {
            return value;
        }
        return value / divisor;
    }
};

struct SquareRootFinalizeFp32 {
    __host__ __device__ float operator()(float value) const { return ::sqrtf(value); }
};

struct PropagatingMinimumFp32 {
    __host__ __device__ float operator()(float lhs, float rhs) const {
        if (lhs != lhs) {
            return lhs;
        }
        if (rhs != rhs) {
            return rhs;
        }
        return ::fminf(lhs, rhs);
    }
};

struct PropagatingMaximumFp32 {
    __host__ __device__ float operator()(float lhs, float rhs) const {
        if (lhs != lhs) {
            return lhs;
        }
        if (rhs != rhs) {
            return rhs;
        }
        return ::fmaxf(lhs, rhs);
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
        case DataType::FP8_E5M2: {
            __nv_fp8_e5m2 converted;
            converted.__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
            static_cast<__nv_fp8_e5m2*>(output)[index] = converted;
            return;
        }
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
struct LogicalAxesToFp32 {
    const InputT* input;
    uint64_t reduction_size;
    CubReductionDeviceIndexing indexing;
    InputTransformT transform;

    __host__ __device__ float operator()(int64_t logical_index) const {
        const uint64_t unsigned_logical_index = static_cast<uint64_t>(logical_index);
        const uint64_t output_index = unsigned_logical_index / reduction_size;
        const uint64_t reduction_index = unsigned_logical_index - output_index * reduction_size;
        const uint64_t physical_index = mapLogicalReductionIndex(indexing, output_index, reduction_index);
        return transform(ToFp32<InputT>{}(input[physical_index]));
    }
};

template <typename InputT, typename InputTransformT>
auto makeContiguousFp32Iterator(const InputT* input, InputTransformT input_transform) {
    return thrust::make_transform_iterator(
        input, ConvertAndTransformInputToFp32<InputT, InputTransformT>{input_transform});
}

template <typename InputT, typename InputTransformT>
auto makeStridedFp32Iterator(const InputT* input,
                             const CubReductionGeometry& geometry,
                             InputTransformT input_transform) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        LogicalAxesToFp32<InputT, InputTransformT>{
            input, geometry.reduction_size, geometry.device_indexing, input_transform});
}

constexpr int TILED_REDUCTION_BLOCK_THREADS = 256;
constexpr int TILED_REDUCTION_WARP_THREADS = 32;
constexpr int TILED_REDUCTION_WARPS_PER_BLOCK =
    TILED_REDUCTION_BLOCK_THREADS / TILED_REDUCTION_WARP_THREADS;
constexpr uint64_t TILED_REDUCTION_TARGET_ACTIVE_WARPS = 1024;
constexpr uint64_t TILED_REDUCTION_MAX_GRID_BLOCKS = 65535;

// Sixteen FP32 accumulators per lane is the current proven-fast register tile. It is a benchmarked design point, not a
// claim that sixteen is the universal optimum: Thor keeps the complete kernel near a ~48-register/thread design budget,
// leaving the remaining registers for indexing, pipeline state, vector packets, operation temporaries, and compiler live
// ranges. For wider trailing vectors, 2/4/8 physical warps cooperate on one output while keeping the same <=16
// accumulators/thread. That scales the full-row engine through D=4096 without increasing per-thread accumulator pressure.
constexpr int FULL_ROW_MAX_COMPONENTS_PER_LANE = 16;
constexpr uint64_t FULL_ROW_COMPONENTS_PER_WARP =
    TILED_REDUCTION_WARP_THREADS * FULL_ROW_MAX_COMPONENTS_PER_LANE;
constexpr int FULL_ROW_MAX_WARPS_PER_OUTPUT = TILED_REDUCTION_WARPS_PER_BLOCK;
constexpr uint64_t FULL_ROW_GROUP_MAX_INNER_SIZE =
    FULL_ROW_COMPONENTS_PER_WARP * FULL_ROW_MAX_WARPS_PER_OUTPUT;
// Once one output consumes a full 8-warp block, wider D values are sharded across independent blocks. Each block owns
// at most this many output components, preserving the same <=16 FP32 accumulators/thread. Component shards never need
// to communicate because every trailing component is an independent reduction across the reduction axis.
constexpr uint64_t FULL_ROW_COMPONENTS_PER_BLOCK = FULL_ROW_GROUP_MAX_INNER_SIZE;

// Each physical warp owns a private two-stage global->shared pipeline. Keeping the staging footprint fixed in bytes
// makes occupancy independent of InputT while still allowing narrow dtypes to stage more reduction rows per batch.
constexpr int ASYNC_TILED_REDUCTION_PIPELINE_STAGES = 2;
constexpr size_t ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP = 2048;
constexpr size_t ASYNC_TILED_REDUCTION_SHARED_BYTES =
    TILED_REDUCTION_WARPS_PER_BLOCK * ASYNC_TILED_REDUCTION_PIPELINE_STAGES
    * ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP;

[[nodiscard]] inline __host__ __device__ uint64_t ceilDivideU64(uint64_t numerator, uint64_t denominator) {
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

[[nodiscard]] inline __host__ __device__ uint64_t minU64(uint64_t lhs, uint64_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

[[nodiscard]] __device__ inline uint64_t largestRowsWithByteAlignment(uint64_t max_rows,
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

// Prefer a stage row count whose total byte count carries a 16-byte alignment promise into cuda::memcpy_async. Odd
// row widths such as 33 FP16 values otherwise tend to choose the absolute largest stage (31 rows = 2046 bytes), which
// prevents the hardware cp.async path even though a slightly smaller 24-row stage is 16-byte aligned.
template <typename InputT>
[[nodiscard]] __device__ inline uint64_t chooseAlignedAsyncRowsPerStage(uint64_t row_elements,
                                                                        uint64_t reduction_size,
                                                                        size_t stage_bytes =
                                                                            ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP) {
    const uint64_t stage_capacity_elements = static_cast<uint64_t>(stage_bytes / sizeof(InputT));
    uint64_t max_rows = stage_capacity_elements / row_elements;
    max_rows = minU64(max_rows, reduction_size);
    if (max_rows == 0) {
        return 1;
    }

    const size_t row_bytes = static_cast<size_t>(row_elements) * sizeof(InputT);
    if (const uint64_t rows = largestRowsWithByteAlignment(max_rows, row_bytes, 16); rows != 0) {
        return rows;
    }
    if (const uint64_t rows = largestRowsWithByteAlignment(max_rows, row_bytes, 8); rows != 0) {
        return rows;
    }
    if (const uint64_t rows = largestRowsWithByteAlignment(max_rows, row_bytes, 4); rows != 0) {
        return rows;
    }
    return max_rows;
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
        // Odd-byte tails and genuinely under-aligned rows are still correct. libcudacxx may use a synchronous
        // fallback for those bytes, while aligned bulk copies use the hardware-accelerated global->shared path.
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

// The narrow async kernel keeps the successful <=32 policy from the first staging experiment: the whole trailing row
// fits in one physical warp tile, while otherwise-unused lanes split reduction rows for inner_size <= 16.
template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int RowLanes>
__global__ void asyncNarrowFullRowReductionKernel(const InputT* input,
                                                  void* output,
                                                  DataType output_dtype,
                                                  uint64_t outer_size,
                                                  uint64_t reduction_size,
                                                  uint64_t inner_size,
                                                  ReductionOpT reduction_op,
                                                  float init,
                                                  InputTransformT input_transform,
                                                  OutputFinalizeT output_finalize,
                                                  float output_scale) {
    static_assert(RowLanes == 1 || RowLanes == 2 || RowLanes == 4 || RowLanes == 8 || RowLanes == 16);
    static_assert(ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP % sizeof(InputT) == 0);

    constexpr uint64_t stage_capacity_elements = ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP / sizeof(InputT);
    using WarpReduceT = cub::WarpReduce<float, RowLanes>;

    __shared__ typename WarpReduceT::TempStorage
        logical_warp_storage[TILED_REDUCTION_BLOCK_THREADS / RowLanes];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, ASYNC_TILED_REDUCTION_PIPELINE_STAGES>
        pipeline_states[TILED_REDUCTION_WARPS_PER_BLOCK];
    extern __shared__ __align__(16) unsigned char async_shared_bytes[];

    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<TILED_REDUCTION_WARP_THREADS>(block);
    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const int component = lane / RowLanes;
    const int row_lane = lane % RowLanes;
    const int logical_warp_in_block = static_cast<int>(threadIdx.x) / RowLanes;

    auto pipeline = cuda::make_pipeline(warp, &pipeline_states[physical_warp]);
    InputT* warp_shared = reinterpret_cast<InputT*>(
        async_shared_bytes
        + static_cast<size_t>(physical_warp * ASYNC_TILED_REDUCTION_PIPELINE_STAGES)
              * ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP);

    const uint64_t total_work = outer_size;
    const uint64_t block_work_stride =
        static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
    const uint64_t rows_per_stage = chooseAlignedAsyncRowsPerStage<InputT>(inner_size, reduction_size);

    for (uint64_t block_work_base =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
         block_work_base < total_work;
         block_work_base += block_work_stride) {
        const uint64_t outer_index = block_work_base + static_cast<uint64_t>(physical_warp);
        if (outer_index >= total_work) {
            continue;
        }

        const bool component_active = static_cast<uint64_t>(component) < inner_size;
        float local = init;
        if (reduction_size != 0) {
            uint64_t current_rows = minU64(rows_per_stage, reduction_size);
            int current_stage = 0;

            enqueueAsyncFullRowReductionStage(
                warp, pipeline, warp_shared, input, outer_index, 0, current_rows, reduction_size, inner_size);

            uint64_t next_row = current_rows;
            while (true) {
                uint64_t next_rows = 0;
                if (next_row < reduction_size) {
                    next_rows = minU64(rows_per_stage, reduction_size - next_row);
                    InputT* next_stage = warp_shared
                                         + static_cast<uint64_t>(current_stage ^ 1) * stage_capacity_elements;
                    enqueueAsyncFullRowReductionStage(warp,
                                                      pipeline,
                                                      next_stage,
                                                      input,
                                                      outer_index,
                                                      next_row,
                                                      next_rows,
                                                      reduction_size,
                                                      inner_size);
                }

                pipeline.consumer_wait();
                if (component_active) {
                    const InputT* stage = warp_shared
                                          + static_cast<uint64_t>(current_stage) * stage_capacity_elements;
                    for (uint64_t stage_row = static_cast<uint64_t>(row_lane);
                         stage_row < current_rows;
                         stage_row += static_cast<uint64_t>(RowLanes)) {
                        const uint64_t shared_index = stage_row * inner_size + static_cast<uint64_t>(component);
                        local = reduction_op(local, input_transform(ToFp32<InputT>{}(stage[shared_index])));
                    }
                }
                pipeline.consumer_release();

                if (next_rows == 0) {
                    break;
                }
                current_rows = next_rows;
                next_row += next_rows;
                current_stage ^= 1;
            }
        }

        float warp_partial = local;
        if constexpr (RowLanes > 1) {
            warp_partial = WarpReduceT(logical_warp_storage[logical_warp_in_block]).Reduce(local, reduction_op);
        } else {
            (void)logical_warp_storage;
            (void)logical_warp_in_block;
        }

        if (component_active && row_lane == 0) {
            const float finalized = output_finalize(warp_partial) * output_scale;
            storeFp32AsRuntimeDType(
                output, output_dtype, outer_index * inner_size + static_cast<uint64_t>(component), finalized);
        }
        if constexpr (RowLanes > 1) {
            __syncwarp();
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int RowLanes>
void launchAsyncNarrowFullRowReduction(const InputT* input,
                                       void* output,
                                       DataType output_dtype,
                                       const CubReductionGeometry& geometry,
                                       ReductionOpT reduction_op,
                                       float init,
                                       InputTransformT input_transform,
                                       OutputFinalizeT output_finalize,
                                       float output_scale,
                                       cudaStream_t stream) {
    const uint64_t required_blocks =
        ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK));
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    asyncNarrowFullRowReductionKernel<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, RowLanes>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, ASYNC_TILED_REDUCTION_SHARED_BYTES, stream>>>(
            input,
            output,
            output_dtype,
            geometry.outer_size,
            geometry.reduction_size,
            geometry.inner_size,
            reduction_op,
            init,
            input_transform,
            output_finalize,
            output_scale);
    CUDA_CHECK(cudaGetLastError());
}

// For 33..511 trailing components whose complete row fits in one async stage, one physical warp owns the complete output
// vector. Each lane owns up to ItemsPerLane components spaced 32 apart, so every shared-memory read round is a contiguous
// 32-component transaction.
// The global->shared pipeline therefore always copies complete consecutive rows, including awkward widths such as 33,
// 65, and 129, instead of submitting per-row component-strip copies.
template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
__global__ void asyncWideFullRowReductionKernel(const InputT* input,
                                                void* output,
                                                DataType output_dtype,
                                                uint64_t outer_size,
                                                uint64_t reduction_size,
                                                uint64_t inner_size,
                                                ReductionOpT reduction_op,
                                                float init,
                                                InputTransformT input_transform,
                                                OutputFinalizeT output_finalize,
                                                float output_scale) {
    static_assert(ItemsPerLane == 2 || ItemsPerLane == 4 || ItemsPerLane == 8 || ItemsPerLane == 16);
    static_assert(ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP % sizeof(InputT) == 0);
    static_assert(TILED_REDUCTION_WARP_THREADS * ItemsPerLane <= FULL_ROW_COMPONENTS_PER_WARP);

    constexpr uint64_t stage_capacity_elements = ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP / sizeof(InputT);
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, ASYNC_TILED_REDUCTION_PIPELINE_STAGES>
        pipeline_states[TILED_REDUCTION_WARPS_PER_BLOCK];
    extern __shared__ __align__(16) unsigned char async_shared_bytes[];

    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<TILED_REDUCTION_WARP_THREADS>(block);
    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;

    auto pipeline = cuda::make_pipeline(warp, &pipeline_states[physical_warp]);
    InputT* warp_shared = reinterpret_cast<InputT*>(
        async_shared_bytes
        + static_cast<size_t>(physical_warp * ASYNC_TILED_REDUCTION_PIPELINE_STAGES)
              * ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP);

    const uint64_t block_work_stride =
        static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
    const uint64_t rows_per_stage = chooseAlignedAsyncRowsPerStage<InputT>(inner_size, reduction_size);

    for (uint64_t block_work_base =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
         block_work_base < outer_size;
         block_work_base += block_work_stride) {
        const uint64_t outer_index = block_work_base + static_cast<uint64_t>(physical_warp);
        if (outer_index >= outer_size) {
            continue;
        }

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = init;
        }

        if (reduction_size != 0) {
            uint64_t current_rows = minU64(rows_per_stage, reduction_size);
            int current_stage = 0;
            enqueueAsyncFullRowReductionStage(
                warp, pipeline, warp_shared, input, outer_index, 0, current_rows, reduction_size, inner_size);

            uint64_t next_row = current_rows;
            while (true) {
                uint64_t next_rows = 0;
                if (next_row < reduction_size) {
                    next_rows = minU64(rows_per_stage, reduction_size - next_row);
                    InputT* next_stage = warp_shared
                                         + static_cast<uint64_t>(current_stage ^ 1) * stage_capacity_elements;
                    enqueueAsyncFullRowReductionStage(warp,
                                                      pipeline,
                                                      next_stage,
                                                      input,
                                                      outer_index,
                                                      next_row,
                                                      next_rows,
                                                      reduction_size,
                                                      inner_size);
                }

                pipeline.consumer_wait();
                const InputT* stage =
                    warp_shared + static_cast<uint64_t>(current_stage) * stage_capacity_elements;
                for (uint64_t stage_row = 0; stage_row < current_rows; ++stage_row) {
                    const uint64_t row_base = stage_row * inner_size;
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item) {
                        const uint64_t component =
                            static_cast<uint64_t>(lane) + static_cast<uint64_t>(item * TILED_REDUCTION_WARP_THREADS);
                        if (component < inner_size) {
                            local[item] = reduction_op(
                                local[item], input_transform(ToFp32<InputT>{}(stage[row_base + component])));
                        }
                    }
                }
                pipeline.consumer_release();

                if (next_rows == 0) {
                    break;
                }
                current_rows = next_rows;
                next_row += next_rows;
                current_stage ^= 1;
            }
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component =
                static_cast<uint64_t>(lane) + static_cast<uint64_t>(item * TILED_REDUCTION_WARP_THREADS);
            if (component < inner_size) {
                const float finalized = output_finalize(local[item]) * output_scale;
                storeFp32AsRuntimeDType(output, output_dtype, outer_index * inner_size + component, finalized);
            }
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
void launchAsyncWideFullRowReduction(const InputT* input,
                                     void* output,
                                     DataType output_dtype,
                                     const CubReductionGeometry& geometry,
                                     ReductionOpT reduction_op,
                                     float init,
                                     InputTransformT input_transform,
                                     OutputFinalizeT output_finalize,
                                     float output_scale,
                                     cudaStream_t stream) {
    const uint64_t required_blocks =
        ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK));
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    asyncWideFullRowReductionKernel<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, ASYNC_TILED_REDUCTION_SHARED_BYTES, stream>>>(
            input,
            output,
            output_dtype,
            geometry.outer_size,
            geometry.reduction_size,
            geometry.inner_size,
            reduction_op,
            init,
            input_transform,
            output_finalize,
            output_scale);
    CUDA_CHECK(cudaGetLastError());
}

// Once the trailing vector exceeds one warp's <=512-component register tile, keep the same full-row architecture and
// scale ownership horizontally. A 2/4/8-warp cooperative group owns one output vector while every thread still carries
// at most ItemsPerLane FP32 accumulators. Each added warp contributes another 2 KiB to each async stage, so the existing
// 32 KiB/block allocation is simply repartitioned as 4x2-warp, 2x4-warp, or 1x8-warp output groups. The global copy is
// still a complete contiguous row slab; no warp copies a strided component strip.
template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int WarpsPerOutput,
          int ItemsPerLane>
__global__ void asyncGroupedFullRowReductionKernel(const InputT* input,
                                                   void* output,
                                                   DataType output_dtype,
                                                   uint64_t outer_size,
                                                   uint64_t reduction_size,
                                                   uint64_t inner_size,
                                                   ReductionOpT reduction_op,
                                                   float init,
                                                   InputTransformT input_transform,
                                                   OutputFinalizeT output_finalize,
                                                   float output_scale) {
    static_assert(WarpsPerOutput == 2 || WarpsPerOutput == 4 || WarpsPerOutput == 8);
    static_assert(ItemsPerLane == FULL_ROW_MAX_COMPONENTS_PER_LANE);
    static_assert(TILED_REDUCTION_WARPS_PER_BLOCK % WarpsPerOutput == 0);

    constexpr int group_threads = TILED_REDUCTION_WARP_THREADS * WarpsPerOutput;
    constexpr int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / WarpsPerOutput;
    constexpr size_t stage_bytes_per_group =
        ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP * static_cast<size_t>(WarpsPerOutput);
    constexpr uint64_t stage_capacity_elements = stage_bytes_per_group / sizeof(InputT);
    constexpr uint64_t component_capacity =
        static_cast<uint64_t>(group_threads) * static_cast<uint64_t>(ItemsPerLane);
    static_assert(component_capacity <= FULL_ROW_GROUP_MAX_INNER_SIZE);

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, ASYNC_TILED_REDUCTION_PIPELINE_STAGES>
        pipeline_states[groups_per_block];
    extern __shared__ __align__(16) unsigned char async_shared_bytes[];

    const auto block = cg::this_thread_block();
    const auto output_group = cg::tiled_partition<group_threads>(block);
    const int group_index = static_cast<int>(threadIdx.x) / group_threads;
    const int group_lane = static_cast<int>(threadIdx.x) % group_threads;

    auto pipeline = cuda::make_pipeline(output_group, &pipeline_states[group_index]);
    InputT* group_shared = reinterpret_cast<InputT*>(
        async_shared_bytes
        + static_cast<size_t>(group_index * ASYNC_TILED_REDUCTION_PIPELINE_STAGES) * stage_bytes_per_group);

    const uint64_t rows_per_stage =
        chooseAlignedAsyncRowsPerStage<InputT>(inner_size, reduction_size, stage_bytes_per_group);
    const uint64_t block_work_stride =
        static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(groups_per_block);

    for (uint64_t block_work_base =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(groups_per_block);
         block_work_base < outer_size;
         block_work_base += block_work_stride) {
        const uint64_t outer_index = block_work_base + static_cast<uint64_t>(group_index);
        if (outer_index >= outer_size) {
            continue;
        }

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = init;
        }

        if (reduction_size != 0) {
            uint64_t current_rows = minU64(rows_per_stage, reduction_size);
            int current_stage = 0;
            enqueueAsyncFullRowReductionStage(output_group,
                                              pipeline,
                                              group_shared,
                                              input,
                                              outer_index,
                                              0,
                                              current_rows,
                                              reduction_size,
                                              inner_size);

            uint64_t next_row = current_rows;
            while (true) {
                uint64_t next_rows = 0;
                if (next_row < reduction_size) {
                    next_rows = minU64(rows_per_stage, reduction_size - next_row);
                    InputT* next_stage = group_shared
                                         + static_cast<uint64_t>(current_stage ^ 1) * stage_capacity_elements;
                    enqueueAsyncFullRowReductionStage(output_group,
                                                      pipeline,
                                                      next_stage,
                                                      input,
                                                      outer_index,
                                                      next_row,
                                                      next_rows,
                                                      reduction_size,
                                                      inner_size);
                }

                pipeline.consumer_wait();
                const InputT* stage =
                    group_shared + static_cast<uint64_t>(current_stage) * stage_capacity_elements;
                for (uint64_t stage_row = 0; stage_row < current_rows; ++stage_row) {
                    const uint64_t row_base = stage_row * inner_size;
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item) {
                        const uint64_t component = static_cast<uint64_t>(group_lane)
                                                   + static_cast<uint64_t>(item * group_threads);
                        if (component < inner_size) {
                            local[item] = reduction_op(
                                local[item], input_transform(ToFp32<InputT>{}(stage[row_base + component])));
                        }
                    }
                }
                pipeline.consumer_release();

                if (next_rows == 0) {
                    break;
                }
                current_rows = next_rows;
                next_row += next_rows;
                current_stage ^= 1;
            }
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component =
                static_cast<uint64_t>(group_lane) + static_cast<uint64_t>(item * group_threads);
            if (component < inner_size) {
                const float finalized = output_finalize(local[item]) * output_scale;
                storeFp32AsRuntimeDType(output, output_dtype, outer_index * inner_size + component, finalized);
            }
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int WarpsPerOutput,
          int ItemsPerLane>
void launchAsyncGroupedFullRowReduction(const InputT* input,
                                        void* output,
                                        DataType output_dtype,
                                        const CubReductionGeometry& geometry,
                                        ReductionOpT reduction_op,
                                        float init,
                                        InputTransformT input_transform,
                                        OutputFinalizeT output_finalize,
                                        float output_scale,
                                        cudaStream_t stream) {
    static_assert(TILED_REDUCTION_WARPS_PER_BLOCK % WarpsPerOutput == 0);
    constexpr int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / WarpsPerOutput;
    constexpr uint64_t component_capacity =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS * WarpsPerOutput * ItemsPerLane);
    constexpr uint64_t stage_capacity_elements =
        static_cast<uint64_t>(ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP * WarpsPerOutput / sizeof(InputT));
    if (geometry.inner_size > component_capacity || geometry.inner_size > stage_capacity_elements) {
        throw std::logic_error("Grouped async full-row reduction launch exceeds its component or stage capacity.");
    }
    const uint64_t required_blocks =
        ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(groups_per_block));
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    asyncGroupedFullRowReductionKernel<InputT,
                                       ReductionOpT,
                                       InputTransformT,
                                       OutputFinalizeT,
                                       WarpsPerOutput,
                                       ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, ASYNC_TILED_REDUCTION_SHARED_BYTES, stream>>>(
            input,
            output,
            output_dtype,
            geometry.outer_size,
            geometry.reduction_size,
            geometry.inner_size,
            reduction_op,
            init,
            input_transform,
            output_finalize,
            output_scale);
    CUDA_CHECK(cudaGetLastError());
}

// Direct vector packets use CUDA's native 2/4/8/16-byte vector-load widths, extending to 32/64/128 bytes as a
// short compile-time sequence of uint4 loads. These exact-width kernels have no shared memory and no synchronization.
template <size_t Bytes>
struct alignas(16) RawVectorPacket {
    static_assert(Bytes >= 16 && Bytes % 16 == 0);
    uint4 words[Bytes / 16];
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
    static_assert(bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16 || bytes == 32 || bytes == 64 || bytes == 128);
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

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
__global__ void vectorizedDirectFullRowReductionKernel(const InputT* input,
                                                       void* output,
                                                       DataType output_dtype,
                                                       uint64_t outer_size,
                                                       uint64_t reduction_size,
                                                       uint64_t inner_size,
                                                       ReductionOpT reduction_op,
                                                       float init,
                                                       InputTransformT input_transform,
                                                       OutputFinalizeT output_finalize,
                                                       float output_scale) {
    static_assert(ItemsPerLane == 2 || ItemsPerLane == 4 || ItemsPerLane == 8 || ItemsPerLane == 16);
    constexpr uint64_t expected_inner_size = TILED_REDUCTION_WARP_THREADS * ItemsPerLane;

    const int physical_warp = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t block_work_stride =
        static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);

    if (inner_size != expected_inner_size) {
        return;
    }

    for (uint64_t block_work_base =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK);
         block_work_base < outer_size;
         block_work_base += block_work_stride) {
        const uint64_t outer_index = block_work_base + static_cast<uint64_t>(physical_warp);
        if (outer_index >= outer_size) {
            continue;
        }

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = init;
        }

        uint64_t row_base = outer_index * reduction_size * inner_size;
        const uint64_t lane_component_begin = static_cast<uint64_t>(lane * ItemsPerLane);
        for (uint64_t row = 0; row < reduction_size; ++row) {
            const InputT* source = input + row_base + lane_component_begin;
            const PackedInputValues<InputT, ItemsPerLane> values =
                loadVectorizedInputPacket<InputT, ItemsPerLane>(source);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = lane_component_begin + static_cast<uint64_t>(item);
            const float finalized = output_finalize(local[item]) * output_scale;
            storeFp32AsRuntimeDType(output, output_dtype, outer_index * inner_size + component, finalized);
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
void launchVectorizedDirectFullRowReduction(const InputT* input,
                                            void* output,
                                            DataType output_dtype,
                                            const CubReductionGeometry& geometry,
                                            ReductionOpT reduction_op,
                                            float init,
                                            InputTransformT input_transform,
                                            OutputFinalizeT output_finalize,
                                            float output_scale,
                                            cudaStream_t stream) {
    const uint64_t required_blocks =
        ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(TILED_REDUCTION_WARPS_PER_BLOCK));
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    vectorizedDirectFullRowReductionKernel<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                   output,
                                                                   output_dtype,
                                                                   geometry.outer_size,
                                                                   geometry.reduction_size,
                                                                   geometry.inner_size,
                                                                   reduction_op,
                                                                   init,
                                                                   input_transform,
                                                                   output_finalize,
                                                                   output_scale);
    CUDA_CHECK(cudaGetLastError());
}

// Exact large widths can bypass shared memory entirely. A 2/4/8-warp output group partitions one complete row into
// contiguous ItemsPerLane packets, one packet per thread. The groups are independent and no synchronization is needed:
// every thread repeatedly loads its packet from each reduction row, accumulates in registers, and stores disjoint outputs.
template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int WarpsPerOutput,
          int ItemsPerLane>
__global__ void vectorizedDirectGroupedFullRowReductionKernel(const InputT* input,
                                                              void* output,
                                                              DataType output_dtype,
                                                              uint64_t outer_size,
                                                              uint64_t reduction_size,
                                                              uint64_t inner_size,
                                                              ReductionOpT reduction_op,
                                                              float init,
                                                              InputTransformT input_transform,
                                                              OutputFinalizeT output_finalize,
                                                              float output_scale) {
    static_assert(WarpsPerOutput == 2 || WarpsPerOutput == 4 || WarpsPerOutput == 8);
    static_assert(ItemsPerLane == FULL_ROW_MAX_COMPONENTS_PER_LANE);
    static_assert(TILED_REDUCTION_WARPS_PER_BLOCK % WarpsPerOutput == 0);

    constexpr int group_threads = TILED_REDUCTION_WARP_THREADS * WarpsPerOutput;
    constexpr int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / WarpsPerOutput;
    constexpr uint64_t expected_inner_size =
        static_cast<uint64_t>(group_threads) * static_cast<uint64_t>(ItemsPerLane);

    if (inner_size != expected_inner_size) {
        return;
    }

    const int group_index = static_cast<int>(threadIdx.x) / group_threads;
    const int group_lane = static_cast<int>(threadIdx.x) % group_threads;
    const uint64_t block_work_stride =
        static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(groups_per_block);

    for (uint64_t block_work_base =
             static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(groups_per_block);
         block_work_base < outer_size;
         block_work_base += block_work_stride) {
        const uint64_t outer_index = block_work_base + static_cast<uint64_t>(group_index);
        if (outer_index >= outer_size) {
            continue;
        }

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = init;
        }

        uint64_t row_base = outer_index * reduction_size * inner_size;
        const uint64_t component_begin = static_cast<uint64_t>(group_lane * ItemsPerLane);
        for (uint64_t row = 0; row < reduction_size; ++row) {
            const InputT* source = input + row_base + component_begin;
            const PackedInputValues<InputT, ItemsPerLane> values =
                loadVectorizedInputPacket<InputT, ItemsPerLane>(source);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = component_begin + static_cast<uint64_t>(item);
            const float finalized = output_finalize(local[item]) * output_scale;
            storeFp32AsRuntimeDType(output, output_dtype, outer_index * inner_size + component, finalized);
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int WarpsPerOutput,
          int ItemsPerLane>
void launchVectorizedDirectGroupedFullRowReduction(const InputT* input,
                                                   void* output,
                                                   DataType output_dtype,
                                                   const CubReductionGeometry& geometry,
                                                   ReductionOpT reduction_op,
                                                   float init,
                                                   InputTransformT input_transform,
                                                   OutputFinalizeT output_finalize,
                                                   float output_scale,
                                                   cudaStream_t stream) {
    static_assert(TILED_REDUCTION_WARPS_PER_BLOCK % WarpsPerOutput == 0);
    constexpr int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / WarpsPerOutput;
    const uint64_t required_blocks =
        ceilDivideU64(geometry.outer_size, static_cast<uint64_t>(groups_per_block));
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(required_blocks, TILED_REDUCTION_MAX_GRID_BLOCKS));

    vectorizedDirectGroupedFullRowReductionKernel<InputT,
                                                  ReductionOpT,
                                                  InputTransformT,
                                                  OutputFinalizeT,
                                                  WarpsPerOutput,
                                                  ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                   output,
                                                                   output_dtype,
                                                                   geometry.outer_size,
                                                                   geometry.reduction_size,
                                                                   geometry.inner_size,
                                                                   reduction_op,
                                                                   init,
                                                                   input_transform,
                                                                   output_finalize,
                                                                   output_scale);
    CUDA_CHECK(cudaGetLastError());
}

// Beyond one full block/output, exact multiples of 4096 scale D without increasing per-thread registers by assigning
// independent component shards to independent blocks. Full shards retain the exact vector-direct x16 memory path.
// Arbitrary widths use the alignment-safe shaped backend below.
template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
__global__ void vectorizedDirectBlockShardedReductionKernel(const InputT* input,
                                                            void* output,
                                                            DataType output_dtype,
                                                            uint64_t outer_size,
                                                            uint64_t reduction_size,
                                                            uint64_t inner_size,
                                                            ReductionOpT reduction_op,
                                                            float init,
                                                            InputTransformT input_transform,
                                                            OutputFinalizeT output_finalize,
                                                            float output_scale) {
    static_assert(ItemsPerLane == FULL_ROW_MAX_COMPONENTS_PER_LANE);
    constexpr uint64_t components_per_block =
        static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    static_assert(components_per_block == FULL_ROW_COMPONENTS_PER_BLOCK);

    if (inner_size % components_per_block != 0) {
        return;
    }

    const uint64_t component_shards = inner_size / components_per_block;
    const uint64_t total_work = outer_size * component_shards;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t outer_index = work_index / component_shards;
        const uint64_t shard_index = work_index - outer_index * component_shards;
        const uint64_t shard_begin = shard_index * components_per_block;
        const uint64_t component_begin =
            shard_begin + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = init;
        }

        uint64_t row_base = outer_index * reduction_size * inner_size + component_begin;
        for (uint64_t row = 0; row < reduction_size; ++row) {
            const PackedInputValues<InputT, ItemsPerLane> values =
                loadVectorizedInputPacket<InputT, ItemsPerLane>(input + row_base);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
            }
            row_base += inner_size;
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t component = component_begin + static_cast<uint64_t>(item);
            const float finalized = output_finalize(local[item]) * output_scale;
            storeFp32AsRuntimeDType(output, output_dtype, outer_index * inner_size + component, finalized);
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
void launchVectorizedDirectBlockShardedReduction(const InputT* input,
                                                 void* output,
                                                 DataType output_dtype,
                                                 const CubReductionGeometry& geometry,
                                                 ReductionOpT reduction_op,
                                                 float init,
                                                 InputTransformT input_transform,
                                                 OutputFinalizeT output_finalize,
                                                 float output_scale,
                                                 cudaStream_t stream) {
    static_assert(ItemsPerLane == FULL_ROW_MAX_COMPONENTS_PER_LANE);
    constexpr uint64_t components_per_block =
        static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS) * static_cast<uint64_t>(ItemsPerLane);
    if (geometry.inner_size % components_per_block != 0) {
        throw std::logic_error("Vectorized block-sharded reduction requires an exact 4096-component shard width.");
    }
    const uint64_t component_shards = geometry.inner_size / components_per_block;
    const uint64_t total_work = geometry.outer_size * component_shards;
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));

    vectorizedDirectBlockShardedReductionKernel<InputT,
                                                ReductionOpT,
                                                InputTransformT,
                                                OutputFinalizeT,
                                                ItemsPerLane>
        <<<grid_blocks, TILED_REDUCTION_BLOCK_THREADS, 0, stream>>>(input,
                                                                   output,
                                                                   output_dtype,
                                                                   geometry.outer_size,
                                                                   geometry.reduction_size,
                                                                   geometry.inner_size,
                                                                   reduction_op,
                                                                   init,
                                                                   input_transform,
                                                                   output_finalize,
                                                                   output_scale);
    CUDA_CHECK(cudaGetLastError());
}

// For arbitrary large D, preserve complete 4096-component shards unless a tiny remainder would leave too little
// aggregate warp parallelism. When a sub-half-block remainder has enough outer work, borrow only the final full shard
// and split that 4096+remainder tail into two ~half-sized launches. Otherwise keep the simpler 4096*N + remainder
// geometry. This decision is host-side only; every CUDA launch still uses one fixed shard width and a branch-free
// reduction loop.
constexpr uint64_t ALIGNMENT_SAFE_TAIL_REBALANCE_MIN_ACTIVE_WARPS = TILED_REDUCTION_TARGET_ACTIVE_WARPS / 2;

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
__global__ void alignmentSafeVectorizedShardRangeReductionKernel(const InputT* input,
                                                                 void* output,
                                                                 DataType output_dtype,
                                                                 uint64_t outer_size,
                                                                 uint64_t reduction_size,
                                                                 uint64_t inner_size,
                                                                 uint64_t first_shard_begin,
                                                                 uint64_t shard_width,
                                                                 uint64_t shard_stride,
                                                                 uint64_t shard_count,
                                                                 ReductionOpT reduction_op,
                                                                 float init,
                                                                 InputTransformT input_transform,
                                                                 OutputFinalizeT output_finalize,
                                                                 float output_scale) {
    static_assert(ItemsPerLane == FULL_ROW_MAX_COMPONENTS_PER_LANE);

    const uint64_t total_work = outer_size * shard_count;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t outer_index = work_index / shard_count;
        const uint64_t shard_index = work_index - outer_index * shard_count;
        const uint64_t shard_begin = first_shard_begin + shard_index * shard_stride;
        const uint64_t component_in_shard =
            static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
        const uint64_t component_begin = shard_begin + component_in_shard;

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = init;
        }

        // The launch uses exactly ceil(shard_width / ItemsPerLane) threads, so every launched thread owns a real
        // packet start. The last packet may cross the logical row end, but the tensor's 128-byte backing padding makes
        // that fixed-width read safe even on the final reduction row. No width predicate exists in the reduction loop.
        uint64_t row_base = outer_index * reduction_size * inner_size + component_begin;
        for (uint64_t row = 0; row < reduction_size; ++row) {
            const PackedInputValues<InputT, ItemsPerLane> values =
                loadAlignmentSafeInputPacket<InputT, ItemsPerLane>(input + row_base);
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                local[item] = reduction_op(local[item], input_transform(ToFp32<InputT>{}(values.values[item])));
            }
            row_base += inner_size;
        }

        // Input padding can make the read side completely tail-free, but output rows are logically adjacent. Only the
        // final packet of an awkward D therefore needs predicated stores so it cannot overwrite the next outer row.
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            const uint64_t item_in_shard = component_in_shard + static_cast<uint64_t>(item);
            if (item_in_shard < shard_width) {
                const uint64_t component = shard_begin + item_in_shard;
                const float finalized = output_finalize(local[item]) * output_scale;
                storeFp32AsRuntimeDType(output, output_dtype, outer_index * inner_size + component, finalized);
            }
        }
    }
}

template <typename InputT,
          typename ReductionOpT,
          typename InputTransformT,
          typename OutputFinalizeT,
          int ItemsPerLane>
void launchAlignmentSafeVectorizedShardRange(const InputT* input,
                                             void* output,
                                             DataType output_dtype,
                                             const CubReductionGeometry& geometry,
                                             uint64_t first_shard_begin,
                                             uint64_t shard_width,
                                             uint64_t shard_stride,
                                             uint64_t shard_count,
                                             ReductionOpT reduction_op,
                                             float init,
                                             InputTransformT input_transform,
                                             OutputFinalizeT output_finalize,
                                             float output_scale,
                                             cudaStream_t stream) {
    static_assert(ItemsPerLane == FULL_ROW_MAX_COMPONENTS_PER_LANE);
    if (shard_count == 0) {
        return;
    }

    const uint64_t block_threads_u64 = ceilDivideU64(shard_width, static_cast<uint64_t>(ItemsPerLane));
    if (block_threads_u64 == 0 || block_threads_u64 > static_cast<uint64_t>(TILED_REDUCTION_BLOCK_THREADS)) {
        throw std::logic_error("Alignment-safe vectorized reduction shard width exceeds one block.");
    }
    const unsigned int block_threads = static_cast<unsigned int>(block_threads_u64);
    const uint64_t total_work = geometry.outer_size * shard_count;
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));

    alignmentSafeVectorizedShardRangeReductionKernel<InputT,
                                                      ReductionOpT,
                                                      InputTransformT,
                                                      OutputFinalizeT,
                                                      ItemsPerLane>
        <<<grid_blocks, block_threads, 0, stream>>>(input,
                                                    output,
                                                    output_dtype,
                                                    geometry.outer_size,
                                                    geometry.reduction_size,
                                                    geometry.inner_size,
                                                    first_shard_begin,
                                                    shard_width,
                                                    shard_stride,
                                                    shard_count,
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
          int ItemsPerLane>
void launchAlignmentSafeVectorizedShapedBlockShardedReduction(const InputT* input,
                                                              void* output,
                                                              DataType output_dtype,
                                                              const CubReductionGeometry& geometry,
                                                              ReductionOpT reduction_op,
                                                              float init,
                                                              InputTransformT input_transform,
                                                              OutputFinalizeT output_finalize,
                                                              float output_scale,
                                                              cudaStream_t stream) {
    static_assert(ItemsPerLane == FULL_ROW_MAX_COMPONENTS_PER_LANE);
    constexpr uint64_t components_per_block = FULL_ROW_COMPONENTS_PER_BLOCK;
    constexpr uint64_t half_block = components_per_block / 2;
    constexpr uint64_t components_per_warp = FULL_ROW_COMPONENTS_PER_WARP;

    const uint64_t full_shards = geometry.inner_size / components_per_block;
    const uint64_t remainder = geometry.inner_size % components_per_block;

    uint64_t preserved_full_shards = full_shards;
    uint64_t first_balanced_tail_width = 0;
    uint64_t second_balanced_tail_width = 0;

    // A tiny final shard is cheap when there are many preceding 4096-component shards, but expensive when outer_size
    // is large enough that borrowing one full shard creates two well-populated ~half-block launches. Choose between
    // those geometries from aggregate useful warps, not from D-specific cases. Requiring the smaller balanced launch to
    // contribute at least half the normal target active-warps keeps both launches well populated.
    if (remainder != 0 && remainder < half_block && full_shards != 0) {
        const uint64_t balanced_tail_width = components_per_block + remainder;
        const uint64_t half_tail = balanced_tail_width / 2;
        first_balanced_tail_width =
            ((half_tail + components_per_warp / 2) / components_per_warp) * components_per_warp;
        first_balanced_tail_width = std::max<uint64_t>(first_balanced_tail_width, half_block);
        first_balanced_tail_width = std::min<uint64_t>(first_balanced_tail_width, components_per_block);
        second_balanced_tail_width = balanced_tail_width - first_balanced_tail_width;

        const uint64_t first_tail_warps = ceilDivideU64(first_balanced_tail_width, components_per_warp);
        const uint64_t second_tail_warps = ceilDivideU64(second_balanced_tail_width, components_per_warp);
        const uint64_t smaller_tail_active_warps =
            geometry.outer_size * std::min(first_tail_warps, second_tail_warps);

        if (smaller_tail_active_warps >= ALIGNMENT_SAFE_TAIL_REBALANCE_MIN_ACTIVE_WARPS) {
            preserved_full_shards = full_shards - 1;
        } else {
            first_balanced_tail_width = 0;
            second_balanced_tail_width = 0;
        }
    }

    // All preserved 4096-component shards share one launch. Their starts are packet-aligned; only the row stride can
    // make their global addresses awkward, which is handled entirely by loadAlignmentSafeInputPacket().
    launchAlignmentSafeVectorizedShardRange<InputT,
                                            ReductionOpT,
                                            InputTransformT,
                                            OutputFinalizeT,
                                            ItemsPerLane>(input,
                                                          output,
                                                          output_dtype,
                                                          geometry,
                                                          0,
                                                          components_per_block,
                                                          components_per_block,
                                                          preserved_full_shards,
                                                          reduction_op,
                                                          init,
                                                          input_transform,
                                                          output_finalize,
                                                          output_scale,
                                                          stream);

    if (first_balanced_tail_width != 0) {
        const uint64_t tail_begin = preserved_full_shards * components_per_block;
        launchAlignmentSafeVectorizedShardRange<InputT,
                                                ReductionOpT,
                                                InputTransformT,
                                                OutputFinalizeT,
                                                ItemsPerLane>(input,
                                                              output,
                                                              output_dtype,
                                                              geometry,
                                                              tail_begin,
                                                              first_balanced_tail_width,
                                                              0,
                                                              1,
                                                              reduction_op,
                                                              init,
                                                              input_transform,
                                                              output_finalize,
                                                              output_scale,
                                                              stream);
        launchAlignmentSafeVectorizedShardRange<InputT,
                                                ReductionOpT,
                                                InputTransformT,
                                                OutputFinalizeT,
                                                ItemsPerLane>(input,
                                                              output,
                                                              output_dtype,
                                                              geometry,
                                                              tail_begin + first_balanced_tail_width,
                                                              second_balanced_tail_width,
                                                              0,
                                                              1,
                                                              reduction_op,
                                                              init,
                                                              input_transform,
                                                              output_finalize,
                                                              output_scale,
                                                              stream);
        return;
    }

    // Otherwise keep every complete 4096-component shard and put the entire remainder in one final launch. Its thread
    // count is exactly ceil(remainder / 16), and tensor padding makes its final fixed-width packet safe to read in full.
    if (remainder != 0) {
        launchAlignmentSafeVectorizedShardRange<InputT,
                                                ReductionOpT,
                                                InputTransformT,
                                                OutputFinalizeT,
                                                ItemsPerLane>(input,
                                                              output,
                                                              output_dtype,
                                                              geometry,
                                                              full_shards * components_per_block,
                                                              remainder,
                                                              0,
                                                              1,
                                                              reduction_op,
                                                              init,
                                                              input_transform,
                                                              output_finalize,
                                                              output_scale,
                                                              stream);
    }
}

// For an awkward large row, choose the smallest power-of-two output group that satisfies both register ownership and
// async-stage capacity. Each warp contributes 512 components of register capacity and 2 KiB to one pipeline stage.
// Returning zero means even a full 8-warp block cannot stage and own the complete row with the current fixed resources.
template <typename InputT>
[[nodiscard]] int chooseGroupedFullRowWarpsPerOutput(uint64_t inner_size) {
    const uint64_t component_warps = ceilDivideU64(inner_size, FULL_ROW_COMPONENTS_PER_WARP);
    const uint64_t row_bytes = inner_size * static_cast<uint64_t>(sizeof(InputT));
    const uint64_t stage_warps = ceilDivideU64(row_bytes, ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP);
    const uint64_t required_warps = std::max(component_warps, stage_warps);

    int warps = 1;
    while (warps < FULL_ROW_MAX_WARPS_PER_OUTPUT && static_cast<uint64_t>(warps) < required_warps) {
        warps *= 2;
    }
    return static_cast<uint64_t>(warps) >= required_warps ? warps : 0;
}

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
                    output, output_dtype, outer_index * inner_size + component, finalized);
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
                    output, output_dtype, outer_index * inner_size + component, finalized);
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
                                                                   warps_per_tile,
                                                                   reduction_op,
                                                                   init,
                                                                   input_transform,
                                                                   output_finalize,
                                                                   output_scale);
    CUDA_CHECK(cudaGetLastError());
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
    if (!geometry.reduced_axes_are_contiguous || geometry.inner_size <= 1) {
        throw std::logic_error("Tiled CUB reduction requires contiguous reduced axes and a trailing vector width > 1.");
    }

    if (geometry.inner_size <= 2) {
        launchAsyncNarrowFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 16>(
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
    } else if (geometry.inner_size <= 4) {
        launchAsyncNarrowFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 8>(
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
    } else if (geometry.inner_size <= 8) {
        launchAsyncNarrowFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 4>(
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
    } else if (geometry.inner_size <= 16) {
        launchAsyncNarrowFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 2>(
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
    } else if (geometry.inner_size <= 32) {
        launchAsyncNarrowFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 1>(
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
    } else if (geometry.inner_size == 64) {
        launchVectorizedDirectFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 2>(
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
    } else if (geometry.inner_size < 64) {
        launchAsyncWideFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 2>(
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
    } else if (geometry.inner_size == 128) {
        launchVectorizedDirectFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 4>(
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
    } else if (geometry.inner_size < 128) {
        launchAsyncWideFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 4>(
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
    } else if (geometry.inner_size == 256) {
        launchVectorizedDirectFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 8>(
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
    } else if (geometry.inner_size < 256) {
        launchAsyncWideFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 8>(
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
    } else if (geometry.inner_size == 512) {
        launchVectorizedDirectFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 16>(
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
    } else if (geometry.inner_size < 512
               && geometry.inner_size
                      <= ASYNC_TILED_REDUCTION_STAGE_BYTES_PER_WARP / static_cast<uint64_t>(sizeof(InputT))) {
        launchAsyncWideFullRowReduction<InputT, ReductionOpT, InputTransformT, OutputFinalizeT, 16>(
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
    } else if (geometry.inner_size == 1024) {
        launchVectorizedDirectGroupedFullRowReduction<InputT,
                                                      ReductionOpT,
                                                      InputTransformT,
                                                      OutputFinalizeT,
                                                      2,
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
    } else if (geometry.inner_size == 2048) {
        launchVectorizedDirectGroupedFullRowReduction<InputT,
                                                      ReductionOpT,
                                                      InputTransformT,
                                                      OutputFinalizeT,
                                                      4,
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
    } else if (geometry.inner_size == 4096) {
        launchVectorizedDirectGroupedFullRowReduction<InputT,
                                                      ReductionOpT,
                                                      InputTransformT,
                                                      OutputFinalizeT,
                                                      8,
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
    } else if (geometry.inner_size > 512 && geometry.inner_size < FULL_ROW_GROUP_MAX_INNER_SIZE) {
        const int warps_per_output = chooseGroupedFullRowWarpsPerOutput<InputT>(geometry.inner_size);
        if (warps_per_output == 2) {
            launchAsyncGroupedFullRowReduction<InputT,
                                               ReductionOpT,
                                               InputTransformT,
                                               OutputFinalizeT,
                                               2,
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
            return;
        }
        if (warps_per_output == 4) {
            launchAsyncGroupedFullRowReduction<InputT,
                                               ReductionOpT,
                                               InputTransformT,
                                               OutputFinalizeT,
                                               4,
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
            return;
        }
        if (warps_per_output == 8) {
            launchAsyncGroupedFullRowReduction<InputT,
                                               ReductionOpT,
                                               InputTransformT,
                                               OutputFinalizeT,
                                               8,
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
            return;
        }

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
    } else if (geometry.inner_size > FULL_ROW_GROUP_MAX_INNER_SIZE) {
        if (geometry.inner_size % FULL_ROW_COMPONENTS_PER_BLOCK == 0) {
            launchVectorizedDirectBlockShardedReduction<InputT,
                                                        ReductionOpT,
                                                        InputTransformT,
                                                        OutputFinalizeT,
                                                        FULL_ROW_MAX_COMPONENTS_PER_LANE>(input,
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
            launchAlignmentSafeVectorizedShapedBlockShardedReduction<InputT,
                                                                      ReductionOpT,
                                                                      InputTransformT,
                                                                      OutputFinalizeT,
                                                                      FULL_ROW_MAX_COMPONENTS_PER_LANE>(input,
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
    } else {
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
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(nullptr,
                                                          queried_bytes,
                                                          input,
                                                          output_iterator,
                                                          static_cast<int64_t>(input_elements),
                                                          reduction_op,
                                                          device_input_transform,
                                                          init,
                                                          stream));
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
            // Tiled backends use only launch-owned registers/static-or-dynamic shared memory and no stamped temporary tensor.
            // Keep the one-byte workspace convention so the allocation-free run contract remains uniform across backends.
            queried_bytes = 1;
            break;
        case CubReductionPath::StridedFixedSegment: {
            auto input_iterator = makeStridedFp32Iterator<InputT>(input, geometry, input_transform);
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
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB reduction received offset-segmented geometry.");
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
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input.getMemPtr<InputT>(),
                                                          output_iterator,
                                                          static_cast<int64_t>(input.getTotalNumElements()),
                                                          reduction_op,
                                                          device_input_transform,
                                                          init,
                                                          stream));
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
            break;
        case CubReductionPath::StridedFixedSegment: {
            auto input_iterator = makeStridedFp32Iterator<InputT>(input.getMemPtr<InputT>(), geometry, input_transform);
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
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB reduction received offset-segmented geometry.");
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
