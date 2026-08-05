#include "Utilities/TensorOperations/Cub/CubArgReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {

constexpr uint32_t VECTOR_SEGMENTED_ARG_THREADS = 256;
constexpr uint64_t VECTOR_SEGMENTED_ARG_MAX_GRID_BLOCKS = 65535;

template <typename Fn>
decltype(auto) dispatchOffsetDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
#if THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
#endif
        default:
            throw std::invalid_argument("Unsupported CUB segmented arg-reduction offset dtype.");
    }
}

[[nodiscard]] uint64_t vectorElementsPerValue(const Tensor& input) {
    const std::vector<uint64_t>& dimensions = input.getDimensions();
    if (dimensions.size() <= 1) {
        return 1;
    }
    uint64_t elements_per_value = 1;
    for (size_t axis = 1; axis < dimensions.size(); ++axis) {
        if (dimensions[axis] != 0
            && elements_per_value > std::numeric_limits<uint64_t>::max() / dimensions[axis]) {
            throw std::invalid_argument("Vector segmented arg-reduction trailing value size overflows uint64_t.");
        }
        elements_per_value *= dimensions[axis];
    }
    if (elements_per_value == 0) {
        throw std::invalid_argument("Vector segmented arg reduction requires a non-zero trailing value size.");
    }
    return elements_per_value;
}

template <typename InputT, typename OffsetT, typename ReductionOpT>
size_t queryScalarForTypes(const Tensor& input,
                           Tensor& index_output,
                           const Tensor& segment_offsets,
                           uint64_t num_segments,
                           ReductionOpT reduction_op,
                           ArgReductionCandidateFp32 init,
                           cudaStream_t stream) {
    size_t queried_bytes = 0;
    auto input_iterator = makeDeviceArgCandidateIterator<InputT>(input);
    auto output_iterator = makeRuntimeArgReductionOutputIterator(nullptr, &index_output);
    const OffsetT* offsets = segment_offsets.getMemPtr<OffsetT>();
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                  queried_bytes,
                                                  input_iterator,
                                                  output_iterator,
                                                  static_cast<int64_t>(num_segments),
                                                  offsets,
                                                  offsets + 1,
                                                  reduction_op,
                                                  init,
                                                  stream));
    return std::max<size_t>(queried_bytes, 1);
}

template <typename InputT, typename OffsetT, typename ReductionOpT>
void launchScalarForTypes(const Tensor& temp_storage,
                          size_t temp_storage_bytes,
                          const Tensor& input,
                          Tensor& index_output,
                          const Tensor& segment_offsets,
                          uint64_t num_segments,
                          ReductionOpT reduction_op,
                          ArgReductionCandidateFp32 init,
                          cudaStream_t stream) {
    void* temp_storage_ptr = const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
    auto input_iterator = makeDeviceArgCandidateIterator<InputT>(input);
    auto output_iterator = makeRuntimeArgReductionOutputIterator(nullptr, &index_output);
    const OffsetT* offsets = segment_offsets.getMemPtr<OffsetT>();
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                  temp_storage_bytes,
                                                  input_iterator,
                                                  output_iterator,
                                                  static_cast<int64_t>(num_segments),
                                                  offsets,
                                                  offsets + 1,
                                                  reduction_op,
                                                  init,
                                                  stream));
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename IndexT>
[[nodiscard]] __device__ inline DenseArgReductionCandidateFp32<IndexT> makeVectorSegmentedArgInit(
    ArgReductionCandidateFp32 init) {
    return DenseArgReductionCandidateFp32<IndexT>{static_cast<IndexT>(~IndexT{0}), init.value};
}

template <typename IndexT>
__device__ inline void storeVectorSegmentedArgWinner(void* index_output,
                                                     DataType index_output_dtype,
                                                     uint64_t output_index,
                                                     uint64_t elements_per_value,
                                                     uint64_t component,
                                                     DenseArgReductionCandidateFp32<IndexT> candidate) {
    if (candidate.index == static_cast<IndexT>(~IndexT{0})) {
        const uint64_t sentinel =
            index_output_dtype == DataType::UINT32 ? static_cast<uint64_t>(~uint32_t{0}) : ~uint64_t{0};
        storeArgIndexAsRuntimeDType(index_output, index_output_dtype, output_index, sentinel);
        return;
    }
    const uint64_t row = static_cast<uint64_t>(candidate.index);
    const uint64_t packed_index = row * elements_per_value + component;
    storeArgIndexAsRuntimeDType(index_output, index_output_dtype, output_index, packed_index);
}

// For narrow vector values, one physical warp owns one segment. Power-of-two logical warps split the segment rows for
// each trailing component, preserving coalesced row-major reads while keeping enough lanes busy for long segments.
template <typename InputT, typename OffsetT, typename ReductionOpT, typename IndexT, int RowLanes>
__global__ void narrowVectorOffsetSegmentedArgReductionKernel(const InputT* input,
                                                               const OffsetT* offsets,
                                                               void* index_output,
                                                               DataType index_output_dtype,
                                                               uint64_t num_segments,
                                                               uint64_t elements_per_value,
                                                               ReductionOpT reduction_op,
                                                               ArgReductionCandidateFp32 init) {
    static_assert(RowLanes == 2 || RowLanes == 4 || RowLanes == 8 || RowLanes == 16 || RowLanes == 32);
    using CandidateT = DenseArgReductionCandidateFp32<IndexT>;
    using WarpReduceT = cub::WarpReduce<CandidateT, RowLanes>;
    __shared__ typename WarpReduceT::TempStorage logical_warp_storage[VECTOR_SEGMENTED_ARG_THREADS / RowLanes];

    constexpr uint32_t physical_warps_per_block = VECTOR_SEGMENTED_ARG_THREADS / 32;
    const uint32_t physical_warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t component = lane / RowLanes;
    const uint32_t row_lane = lane % RowLanes;
    const uint32_t logical_warp = threadIdx.x / RowLanes;
    const uint64_t block_stride = static_cast<uint64_t>(gridDim.x) * physical_warps_per_block;

    for (uint64_t segment = static_cast<uint64_t>(blockIdx.x) * physical_warps_per_block + physical_warp;
         segment < num_segments;
         segment += block_stride) {
        const uint64_t begin = static_cast<uint64_t>(offsets[segment]);
        const uint64_t end = static_cast<uint64_t>(offsets[segment + 1]);
        CandidateT local = makeVectorSegmentedArgInit<IndexT>(init);
        if (component < elements_per_value) {
            for (uint64_t row = begin + row_lane; row < end; row += RowLanes) {
                const CandidateT candidate{
                    static_cast<IndexT>(row),
                    ToFp32<InputT>{}(input[row * elements_per_value + component])};
                local = reduction_op(local, candidate);
            }
        }

        CandidateT reduced = WarpReduceT(logical_warp_storage[logical_warp]).Reduce(local, reduction_op);
        if (row_lane == 0 && component < elements_per_value) {
            const uint64_t output_index = segment * elements_per_value + component;
            storeVectorSegmentedArgWinner(index_output,
                                          index_output_dtype,
                                          output_index,
                                          elements_per_value,
                                          component,
                                          reduced);
        }
        __syncwarp();
    }
}

template <typename InputT, typename OffsetT, typename ReductionOpT, typename IndexT, int RowLanes>
void launchNarrowVectorForTypes(const Tensor& input,
                                Tensor& index_output,
                                const Tensor& segment_offsets,
                                uint64_t num_segments,
                                uint64_t elements_per_value,
                                ReductionOpT reduction_op,
                                ArgReductionCandidateFp32 init,
                                cudaStream_t stream) {
    constexpr uint64_t physical_warps_per_block = VECTOR_SEGMENTED_ARG_THREADS / 32;
    const uint64_t required_blocks =
        (num_segments + physical_warps_per_block - 1) / physical_warps_per_block;
    const uint32_t grid_blocks = static_cast<uint32_t>(
        std::min<uint64_t>(required_blocks, VECTOR_SEGMENTED_ARG_MAX_GRID_BLOCKS));
    narrowVectorOffsetSegmentedArgReductionKernel<InputT, OffsetT, ReductionOpT, IndexT, RowLanes>
        <<<grid_blocks, VECTOR_SEGMENTED_ARG_THREADS, 0, stream>>>(input.getMemPtr<InputT>(),
                                                                   segment_offsets.getMemPtr<OffsetT>(),
                                                                   index_output.getMemPtr<void>(),
                                                                   index_output.getDataType(),
                                                                   num_segments,
                                                                   elements_per_value,
                                                                   reduction_op,
                                                                   init);
    CUDA_CHECK(cudaGetLastError());
}

// Wider vector values need no inter-thread reduction: one thread owns one trailing component and walks the rows in
// that segment. Adjacent threads therefore read adjacent components from each row. Candidate state keeps only the
// winning row index; the global packed winner is formed once when the result is stored.
template <typename InputT, typename OffsetT, typename ReductionOpT, typename IndexT>
__global__ void vectorOffsetSegmentedArgReductionKernel(const InputT* input,
                                                         const OffsetT* offsets,
                                                         void* index_output,
                                                         DataType index_output_dtype,
                                                         uint64_t num_segments,
                                                         uint64_t elements_per_value,
                                                         uint64_t component_tiles,
                                                         ReductionOpT reduction_op,
                                                         ArgReductionCandidateFp32 init) {
    using CandidateT = DenseArgReductionCandidateFp32<IndexT>;
    const uint64_t total_work = num_segments * component_tiles;
    for (uint64_t work = static_cast<uint64_t>(blockIdx.x); work < total_work; work += gridDim.x) {
        const uint64_t segment = work / component_tiles;
        const uint64_t tile = work - segment * component_tiles;
        const uint64_t component = tile * VECTOR_SEGMENTED_ARG_THREADS + static_cast<uint64_t>(threadIdx.x);
        if (component >= elements_per_value) {
            continue;
        }

        const uint64_t begin = static_cast<uint64_t>(offsets[segment]);
        const uint64_t end = static_cast<uint64_t>(offsets[segment + 1]);
        CandidateT local = makeVectorSegmentedArgInit<IndexT>(init);
        for (uint64_t row = begin; row < end; ++row) {
            const CandidateT candidate{
                static_cast<IndexT>(row),
                ToFp32<InputT>{}(input[row * elements_per_value + component])};
            local = reduction_op(local, candidate);
        }

        const uint64_t output_index = segment * elements_per_value + component;
        storeVectorSegmentedArgWinner(index_output,
                                      index_output_dtype,
                                      output_index,
                                      elements_per_value,
                                      component,
                                      local);
    }
}

template <typename InputT, typename OffsetT, typename ReductionOpT, typename IndexT>
void launchWideVectorForTypes(const Tensor& input,
                              Tensor& index_output,
                              const Tensor& segment_offsets,
                              uint64_t num_segments,
                              uint64_t elements_per_value,
                              ReductionOpT reduction_op,
                              ArgReductionCandidateFp32 init,
                              cudaStream_t stream) {
    const uint64_t component_tiles =
        (elements_per_value + VECTOR_SEGMENTED_ARG_THREADS - 1) / VECTOR_SEGMENTED_ARG_THREADS;
    if (num_segments > std::numeric_limits<uint64_t>::max() / component_tiles) {
        throw std::invalid_argument("Vector segmented arg-reduction launch work count overflows uint64_t.");
    }
    const uint64_t total_work = num_segments * component_tiles;
    const uint32_t grid_blocks = static_cast<uint32_t>(
        std::min<uint64_t>(total_work, VECTOR_SEGMENTED_ARG_MAX_GRID_BLOCKS));
    vectorOffsetSegmentedArgReductionKernel<InputT, OffsetT, ReductionOpT, IndexT>
        <<<grid_blocks, VECTOR_SEGMENTED_ARG_THREADS, 0, stream>>>(input.getMemPtr<InputT>(),
                                                                   segment_offsets.getMemPtr<OffsetT>(),
                                                                   index_output.getMemPtr<void>(),
                                                                   index_output.getDataType(),
                                                                   num_segments,
                                                                   elements_per_value,
                                                                   component_tiles,
                                                                   reduction_op,
                                                                   init);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename OffsetT, typename ReductionOpT, typename IndexT>
void launchVectorForTypes(const Tensor& input,
                          Tensor& index_output,
                          const Tensor& segment_offsets,
                          uint64_t num_segments,
                          uint64_t elements_per_value,
                          ReductionOpT reduction_op,
                          ArgReductionCandidateFp32 init,
                          cudaStream_t stream) {
    if (elements_per_value <= 1) {
        launchNarrowVectorForTypes<InputT, OffsetT, ReductionOpT, IndexT, 32>(
            input, index_output, segment_offsets, num_segments, elements_per_value, reduction_op, init, stream);
    } else if (elements_per_value <= 2) {
        launchNarrowVectorForTypes<InputT, OffsetT, ReductionOpT, IndexT, 16>(
            input, index_output, segment_offsets, num_segments, elements_per_value, reduction_op, init, stream);
    } else if (elements_per_value <= 4) {
        launchNarrowVectorForTypes<InputT, OffsetT, ReductionOpT, IndexT, 8>(
            input, index_output, segment_offsets, num_segments, elements_per_value, reduction_op, init, stream);
    } else if (elements_per_value <= 8) {
        launchNarrowVectorForTypes<InputT, OffsetT, ReductionOpT, IndexT, 4>(
            input, index_output, segment_offsets, num_segments, elements_per_value, reduction_op, init, stream);
    } else if (elements_per_value <= 16) {
        launchNarrowVectorForTypes<InputT, OffsetT, ReductionOpT, IndexT, 2>(
            input, index_output, segment_offsets, num_segments, elements_per_value, reduction_op, init, stream);
    } else {
        launchWideVectorForTypes<InputT, OffsetT, ReductionOpT, IndexT>(
            input, index_output, segment_offsets, num_segments, elements_per_value, reduction_op, init, stream);
    }
}

template <typename Fn>
decltype(auto) dispatchOperation(CubArgReductionOp op, Fn&& fn) {
    switch (op) {
        case CubArgReductionOp::ArgMin:
            return fn(ArgMinimumCandidateFp32{},
                      ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(),
                                                std::numeric_limits<float>::infinity()});
        case CubArgReductionOp::ArgMax:
            return fn(ArgMaximumCandidateFp32{},
                      ArgReductionCandidateFp32{std::numeric_limits<uint64_t>::max(),
                                                -std::numeric_limits<float>::infinity()});
    }
    throw std::invalid_argument("Unsupported segmented arg-reduction operation.");
}

}  // namespace

size_t queryOffsetSegmentedArgReductionBytes(CubArgReductionOp op,
                                             const Tensor& input,
                                             Tensor& index_output,
                                             const Tensor& segment_offsets,
                                             uint64_t num_segments,
                                             const Stream& stream) {
    if (input.getDimensions().size() > 1) {
        // The vector backend carries all reduction state in registers/shared warp-reduce storage.
        return 1;
    }

    auto dispatch_offset = [&]<typename OffsetT>() -> size_t {
        auto dispatch_input = [&]<typename InputT>() -> size_t {
            auto run_op = [&](auto reduction_op, ArgReductionCandidateFp32 init) -> size_t {
                return queryScalarForTypes<InputT, OffsetT>(input,
                                                            index_output,
                                                            segment_offsets,
                                                            num_segments,
                                                            reduction_op,
                                                            init,
                                                            stream.getStream());
            };
            return dispatchOperation(op, run_op);
        };
        return dispatchReductionInputDType(input.getDataType(), dispatch_input);
    };
    return dispatchOffsetDType(segment_offsets.getDataType(), dispatch_offset);
}

void launchOffsetSegmentedArgReduction(CubArgReductionOp op,
                                       const Tensor& temp_storage,
                                       size_t temp_storage_bytes,
                                       const Tensor& input,
                                       Tensor& index_output,
                                       const Tensor& segment_offsets,
                                       uint64_t num_segments,
                                       Stream& stream) {
    const uint64_t elements_per_value = vectorElementsPerValue(input);
    auto dispatch_offset = [&]<typename OffsetT>() -> void {
        auto dispatch_input = [&]<typename InputT>() -> void {
            auto run_op = [&](auto reduction_op, ArgReductionCandidateFp32 init) -> void {
                if (input.getDimensions().size() == 1) {
                    launchScalarForTypes<InputT, OffsetT>(temp_storage,
                                                          temp_storage_bytes,
                                                          input,
                                                          index_output,
                                                          segment_offsets,
                                                          num_segments,
                                                          reduction_op,
                                                          init,
                                                          stream.getStream());
                    return;
                }

                auto dispatch_row_index = [&]<typename RowIndexT>() -> void {
                    launchVectorForTypes<InputT, OffsetT, decltype(reduction_op), RowIndexT>(input,
                                                                                             index_output,
                                                                                             segment_offsets,
                                                                                             num_segments,
                                                                                             elements_per_value,
                                                                                             reduction_op,
                                                                                             init,
                                                                                             stream.getStream());
                };
                if (input.getDimensions()[0] <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                    dispatch_row_index.template operator()<uint32_t>();
                } else {
                    dispatch_row_index.template operator()<uint64_t>();
                }
            };
            dispatchOperation(op, run_op);
        };
        dispatchReductionInputDType(input.getDataType(), dispatch_input);
    };
    dispatchOffsetDType(segment_offsets.getDataType(), dispatch_offset);
}

}  // namespace ThorImplementation::CubReductionInternal
