#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda/iterator>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {

constexpr uint32_t VECTOR_SEGMENTED_THREADS = 256;
constexpr uint64_t VECTOR_SEGMENTED_MAX_GRID_BLOCKS = 65535;

enum class VectorSegmentedReductionFamily : uint8_t { Additive, Min, Max };

struct RuntimeOffsetSegmentedReductionFp32 {
    CubReductionOp op;

    __host__ __device__ float operator()(float lhs, float rhs) const {
        switch (op) {
            case CubReductionOp::Sum:
            case CubReductionOp::Mean:
                return lhs + rhs;
            case CubReductionOp::Min:
                return PropagatingMinimumFp32{}(lhs, rhs);
            case CubReductionOp::Max:
                return PropagatingMaximumFp32{}(lhs, rhs);
            default:
                return lhs;
        }
    }
};

[[nodiscard]] float offsetSegmentedInit(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
        case CubReductionOp::Mean:
            return 0.0f;
        case CubReductionOp::Min:
            return std::numeric_limits<float>::infinity();
        case CubReductionOp::Max:
            return -std::numeric_limits<float>::infinity();
        default:
            throw std::invalid_argument("Offset-segmented CUB reduction supports sum, mean, min, and max.");
    }
}

template <typename OffsetT>
struct FinalizeAndStoreOffsetSegmentedFp32 {
    void* output;
    DataType output_dtype;
    const OffsetT* offsets;
    CubReductionOp op;

    template <typename IndexT>
    __host__ __device__ void operator()(IndexT raw_index, float value) const {
        const uint64_t index = static_cast<uint64_t>(raw_index);
        if (op == CubReductionOp::Mean) {
            const OffsetT begin = offsets[index];
            const OffsetT end = offsets[index + 1];
            const uint64_t count = end >= begin ? static_cast<uint64_t>(end - begin) : 0;
            value = count == 0 ? 0.0f : value / static_cast<float>(count);
        }
        storeFp32AsRuntimeDType(output, output_dtype, index, value);
    }
};

template <typename OffsetT>
auto makeOffsetSegmentedOutputIterator(Tensor& output,
                                       const Tensor& segment_offsets,
                                       CubReductionOp op) {
    return cuda::make_tabulate_output_iterator(FinalizeAndStoreOffsetSegmentedFp32<OffsetT>{
        output.getMemPtr<void>(), output.getDataType(), segment_offsets.getMemPtr<OffsetT>(), op});
}

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
            throw std::invalid_argument("Unsupported CUB segmented-reduction offset dtype value "
                                        + std::to_string(static_cast<int>(dtype)) + ".");
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
            throw std::invalid_argument("Vector segmented-reduction trailing value size overflows uint64_t.");
        }
        elements_per_value *= dimensions[axis];
    }
    if (elements_per_value == 0) {
        throw std::invalid_argument("Vector segmented reduction requires a non-zero trailing value size.");
    }
    return elements_per_value;
}

template <VectorSegmentedReductionFamily Family>
__device__ inline float combineVectorSegmentedFp32(float lhs, float rhs) {
    if constexpr (Family == VectorSegmentedReductionFamily::Additive) {
        return lhs + rhs;
    } else if constexpr (Family == VectorSegmentedReductionFamily::Min) {
        return PropagatingMinimumFp32{}(lhs, rhs);
    } else {
        static_assert(Family == VectorSegmentedReductionFamily::Max);
        return PropagatingMaximumFp32{}(lhs, rhs);
    }
}

template <VectorSegmentedReductionFamily Family>
struct VectorSegmentedReductionFp32 {
    __device__ float operator()(float lhs, float rhs) const {
        return combineVectorSegmentedFp32<Family>(lhs, rhs);
    }
};

template <VectorSegmentedReductionFamily Family>
__device__ inline float vectorSegmentedInitFp32() {
    if constexpr (Family == VectorSegmentedReductionFamily::Additive) {
        return 0.0f;
    } else if constexpr (Family == VectorSegmentedReductionFamily::Min) {
        return CUDART_INF_F;
    } else {
        static_assert(Family == VectorSegmentedReductionFamily::Max);
        return -CUDART_INF_F;
    }
}

// Narrow vector rows use otherwise-idle lanes to split the segment rows. A physical warp owns one segment and is
// partitioned into power-of-two logical warps, one per trailing component. Across the physical warp, each iteration
// touches a compact row slab instead of one D-strided component stream.
template <typename InputT, typename OffsetT, VectorSegmentedReductionFamily Family, int RowLanes>
__global__ void narrowVectorOffsetSegmentedReductionKernel(const InputT* input,
                                                            const OffsetT* offsets,
                                                            void* output,
                                                            DataType output_dtype,
                                                            uint64_t num_segments,
                                                            uint64_t elements_per_value,
                                                            bool compute_mean) {
    static_assert(RowLanes == 2 || RowLanes == 4 || RowLanes == 8 || RowLanes == 16 || RowLanes == 32);
    using WarpReduceT = cub::WarpReduce<float, RowLanes>;
    __shared__ typename WarpReduceT::TempStorage logical_warp_storage[VECTOR_SEGMENTED_THREADS / RowLanes];

    constexpr uint32_t physical_warps_per_block = VECTOR_SEGMENTED_THREADS / 32;
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
        float local = vectorSegmentedInitFp32<Family>();
        if (component < elements_per_value) {
            for (uint64_t row = begin + row_lane; row < end; row += RowLanes) {
                const float value = ToFp32<InputT>{}(input[row * elements_per_value + component]);
                local = combineVectorSegmentedFp32<Family>(local, value);
            }
        }

        float reduced = local;
        if constexpr (RowLanes > 1) {
            reduced = WarpReduceT(logical_warp_storage[logical_warp]).Reduce(
                local, VectorSegmentedReductionFp32<Family>{});
        }
        if (row_lane == 0 && component < elements_per_value) {
            if constexpr (Family == VectorSegmentedReductionFamily::Additive) {
                if (compute_mean) {
                    const uint64_t count = end - begin;
                    reduced = count == 0 ? 0.0f : reduced / static_cast<float>(count);
                }
            }
            storeFp32AsRuntimeDType(output,
                                    output_dtype,
                                    segment * elements_per_value + component,
                                    reduced);
        }
        // CUB logical WarpReduce may use its TempStorage through the end of Reduce. Keep the physical warp together
        // before any logical group reuses the same storage for its next grid-stride segment.
        __syncwarp();
    }
}

template <typename InputT, typename OffsetT, VectorSegmentedReductionFamily Family, int RowLanes>
void launchNarrowVectorForTypes(const Tensor& input,
                                Tensor& output,
                                const Tensor& segment_offsets,
                                uint64_t num_segments,
                                uint64_t elements_per_value,
                                bool compute_mean,
                                cudaStream_t stream) {
    constexpr uint64_t physical_warps_per_block = VECTOR_SEGMENTED_THREADS / 32;
    const uint64_t required_blocks =
        (num_segments + physical_warps_per_block - 1) / physical_warps_per_block;
    const uint32_t grid_blocks = static_cast<uint32_t>(
        std::min<uint64_t>(required_blocks, VECTOR_SEGMENTED_MAX_GRID_BLOCKS));
    narrowVectorOffsetSegmentedReductionKernel<InputT, OffsetT, Family, RowLanes>
        <<<grid_blocks, VECTOR_SEGMENTED_THREADS, 0, stream>>>(input.getMemPtr<InputT>(),
                                                               segment_offsets.getMemPtr<OffsetT>(),
                                                               output.getMemPtr<void>(),
                                                               output.getDataType(),
                                                               num_segments,
                                                               elements_per_value,
                                                               compute_mean);
    CUDA_CHECK(cudaGetLastError());
}

// Vector-valued segmented reductions are a row-major [N,D...] problem. One thread owns one trailing component and
// walks the rows in its segment. Adjacent threads therefore consume adjacent components from every row, unlike the
// former Expression kernel where threads walked rows for one component and issued D-strided memory accesses. No
// inter-thread reduction is needed because every trailing component is an independent segmented reduction.
template <typename InputT, typename OffsetT, VectorSegmentedReductionFamily Family>
__global__ void vectorOffsetSegmentedReductionKernel(const InputT* input,
                                                      const OffsetT* offsets,
                                                      void* output,
                                                      DataType output_dtype,
                                                      uint64_t num_segments,
                                                      uint64_t elements_per_value,
                                                      uint64_t component_tiles,
                                                      bool compute_mean) {
    const uint64_t total_work = num_segments * component_tiles;
    for (uint64_t work = static_cast<uint64_t>(blockIdx.x); work < total_work; work += gridDim.x) {
        const uint64_t segment = work / component_tiles;
        const uint64_t tile = work - segment * component_tiles;
        const uint64_t component = tile * VECTOR_SEGMENTED_THREADS + static_cast<uint64_t>(threadIdx.x);
        if (component >= elements_per_value) {
            continue;
        }

        const uint64_t begin = static_cast<uint64_t>(offsets[segment]);
        const uint64_t end = static_cast<uint64_t>(offsets[segment + 1]);
        float local = vectorSegmentedInitFp32<Family>();
        for (uint64_t row = begin; row < end; ++row) {
            const float value = ToFp32<InputT>{}(input[row * elements_per_value + component]);
            local = combineVectorSegmentedFp32<Family>(local, value);
        }

        if constexpr (Family == VectorSegmentedReductionFamily::Additive) {
            if (compute_mean) {
                const uint64_t count = end - begin;
                local = count == 0 ? 0.0f : local / static_cast<float>(count);
            }
        }
        storeFp32AsRuntimeDType(output,
                                output_dtype,
                                segment * elements_per_value + component,
                                local);
    }
}

template <typename InputT, typename OffsetT, VectorSegmentedReductionFamily Family>
void launchWideVectorForTypes(const Tensor& input,
                              Tensor& output,
                              const Tensor& segment_offsets,
                              uint64_t num_segments,
                              uint64_t elements_per_value,
                              bool compute_mean,
                              cudaStream_t stream) {
    const uint64_t component_tiles =
        (elements_per_value + VECTOR_SEGMENTED_THREADS - 1) / VECTOR_SEGMENTED_THREADS;
    if (num_segments > std::numeric_limits<uint64_t>::max() / component_tiles) {
        throw std::invalid_argument("Vector segmented-reduction launch work count overflows uint64_t.");
    }
    const uint64_t total_work = num_segments * component_tiles;
    const uint32_t grid_blocks = static_cast<uint32_t>(
        std::min<uint64_t>(total_work, VECTOR_SEGMENTED_MAX_GRID_BLOCKS));
    vectorOffsetSegmentedReductionKernel<InputT, OffsetT, Family>
        <<<grid_blocks, VECTOR_SEGMENTED_THREADS, 0, stream>>>(input.getMemPtr<InputT>(),
                                                               segment_offsets.getMemPtr<OffsetT>(),
                                                               output.getMemPtr<void>(),
                                                               output.getDataType(),
                                                               num_segments,
                                                               elements_per_value,
                                                               component_tiles,
                                                               compute_mean);
    CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename OffsetT, VectorSegmentedReductionFamily Family>
void launchVectorForOperationSpecialized(const Tensor& input,
                                         Tensor& output,
                                         const Tensor& segment_offsets,
                                         uint64_t num_segments,
                                         uint64_t elements_per_value,
                                         bool compute_mean,
                                         cudaStream_t stream) {
    if (elements_per_value <= 1) {
        launchNarrowVectorForTypes<InputT, OffsetT, Family, 32>(
            input, output, segment_offsets, num_segments, elements_per_value, compute_mean, stream);
    } else if (elements_per_value <= 2) {
        launchNarrowVectorForTypes<InputT, OffsetT, Family, 16>(
            input, output, segment_offsets, num_segments, elements_per_value, compute_mean, stream);
    } else if (elements_per_value <= 4) {
        launchNarrowVectorForTypes<InputT, OffsetT, Family, 8>(
            input, output, segment_offsets, num_segments, elements_per_value, compute_mean, stream);
    } else if (elements_per_value <= 8) {
        launchNarrowVectorForTypes<InputT, OffsetT, Family, 4>(
            input, output, segment_offsets, num_segments, elements_per_value, compute_mean, stream);
    } else if (elements_per_value <= 16) {
        launchNarrowVectorForTypes<InputT, OffsetT, Family, 2>(
            input, output, segment_offsets, num_segments, elements_per_value, compute_mean, stream);
    } else {
        launchWideVectorForTypes<InputT, OffsetT, Family>(
            input, output, segment_offsets, num_segments, elements_per_value, compute_mean, stream);
    }
}

template <typename InputT, typename OffsetT>
void launchVectorForOperation(CubReductionOp op,
                              const Tensor& input,
                              Tensor& output,
                              const Tensor& segment_offsets,
                              uint64_t num_segments,
                              uint64_t elements_per_value,
                              cudaStream_t stream) {
    switch (op) {
        case CubReductionOp::Sum:
            launchVectorForOperationSpecialized<InputT, OffsetT, VectorSegmentedReductionFamily::Additive>(
                input, output, segment_offsets, num_segments, elements_per_value, false, stream);
            return;
        case CubReductionOp::Mean:
            launchVectorForOperationSpecialized<InputT, OffsetT, VectorSegmentedReductionFamily::Additive>(
                input, output, segment_offsets, num_segments, elements_per_value, true, stream);
            return;
        case CubReductionOp::Min:
            launchVectorForOperationSpecialized<InputT, OffsetT, VectorSegmentedReductionFamily::Min>(
                input, output, segment_offsets, num_segments, elements_per_value, false, stream);
            return;
        case CubReductionOp::Max:
            launchVectorForOperationSpecialized<InputT, OffsetT, VectorSegmentedReductionFamily::Max>(
                input, output, segment_offsets, num_segments, elements_per_value, false, stream);
            return;
        default:
            throw std::invalid_argument("Vector offset-segmented reduction supports sum, mean, min, and max.");
    }
}

template <typename InputT, typename OffsetT>
size_t queryScalarForTypes(CubReductionOp op,
                           const Tensor& input,
                           Tensor& output,
                           const Tensor& segment_offsets,
                           uint64_t num_segments,
                           cudaStream_t stream) {
    size_t queried_bytes = 0;
    auto input_iterator = makeContiguousFp32Iterator(input.getMemPtr<InputT>(), IdentityFp32{});
    auto output_iterator = makeOffsetSegmentedOutputIterator<OffsetT>(output, segment_offsets, op);
    const OffsetT* offsets = segment_offsets.getMemPtr<OffsetT>();
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                  queried_bytes,
                                                  input_iterator,
                                                  output_iterator,
                                                  static_cast<int64_t>(num_segments),
                                                  offsets,
                                                  offsets + 1,
                                                  RuntimeOffsetSegmentedReductionFp32{op},
                                                  offsetSegmentedInit(op),
                                                  stream));
    return std::max<size_t>(queried_bytes, 1);
}

template <typename InputT, typename OffsetT>
void launchScalarForTypes(CubReductionOp op,
                          const Tensor& temp_storage,
                          size_t temp_storage_bytes,
                          const Tensor& input,
                          Tensor& output,
                          const Tensor& segment_offsets,
                          uint64_t num_segments,
                          cudaStream_t stream) {
    void* temp_storage_ptr =
        const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
    auto input_iterator = makeContiguousFp32Iterator(input.getMemPtr<InputT>(), IdentityFp32{});
    auto output_iterator = makeOffsetSegmentedOutputIterator<OffsetT>(output, segment_offsets, op);
    const OffsetT* offsets = segment_offsets.getMemPtr<OffsetT>();
    CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                  temp_storage_bytes,
                                                  input_iterator,
                                                  output_iterator,
                                                  static_cast<int64_t>(num_segments),
                                                  offsets,
                                                  offsets + 1,
                                                  RuntimeOffsetSegmentedReductionFp32{op},
                                                  offsetSegmentedInit(op),
                                                  stream));
    CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace

size_t queryOffsetSegmentedReductionBytes(CubReductionOp op,
                                          const Tensor& input,
                                          Tensor& output,
                                          const Tensor& segment_offsets,
                                          uint64_t,
                                          uint64_t num_segments,
                                          const Stream& stream) {
    if (input.getDimensions().size() > 1) {
        // The vector backend is allocation-free at run time. Stamped operations retain a one-byte placeholder because
        // Tensor descriptors do not represent a zero-byte workspace allocation.
        static_cast<void>(vectorElementsPerValue(input));
        return 1;
    }

    auto dispatch_input = [&]<typename InputT>() -> size_t {
        auto dispatch_offset = [&]<typename OffsetT>() -> size_t {
            return queryScalarForTypes<InputT, OffsetT>(
                op, input, output, segment_offsets, num_segments, stream.getStream());
        };
        return dispatchOffsetDType(segment_offsets.getDataType(), dispatch_offset);
    };
    return dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

void launchOffsetSegmentedReduction(CubReductionOp op,
                                    const Tensor& temp_storage,
                                    size_t temp_storage_bytes,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    uint64_t,
                                    uint64_t num_segments,
                                    Stream& stream) {
    const uint64_t elements_per_value = vectorElementsPerValue(input);
    auto dispatch_input = [&]<typename InputT>() -> void {
        auto dispatch_offset = [&]<typename OffsetT>() -> void {
            if (elements_per_value == 1 && input.getDimensions().size() == 1) {
                launchScalarForTypes<InputT, OffsetT>(op,
                                                      temp_storage,
                                                      temp_storage_bytes,
                                                      input,
                                                      output,
                                                      segment_offsets,
                                                      num_segments,
                                                      stream.getStream());
                return;
            }
            launchVectorForOperation<InputT, OffsetT>(op,
                                                      input,
                                                      output,
                                                      segment_offsets,
                                                      num_segments,
                                                      elements_per_value,
                                                      stream.getStream());
        };
        dispatchOffsetDType(segment_offsets.getDataType(), dispatch_offset);
    };
    dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

}  // namespace ThorImplementation::CubReductionInternal
