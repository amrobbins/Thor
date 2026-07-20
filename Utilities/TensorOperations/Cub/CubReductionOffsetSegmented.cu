#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/iterator>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionInternal {
namespace {

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
#if defined(__CUDA_ARCH__)
                asm("trap;");
#endif
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
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
#endif
        default:
            throw std::invalid_argument("Unsupported CUB segmented-reduction offset dtype value "
                                        + std::to_string(static_cast<int>(dtype)) + ".");
    }
}

template <typename InputT, typename OffsetT>
size_t queryForTypes(CubReductionOp op,
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
void launchForTypes(CubReductionOp op,
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
    auto dispatch_input = [&]<typename InputT>() -> size_t {
        auto dispatch_offset = [&]<typename OffsetT>() -> size_t {
            return queryForTypes<InputT, OffsetT>(
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
    auto dispatch_input = [&]<typename InputT>() -> void {
        auto dispatch_offset = [&]<typename OffsetT>() -> void {
            launchForTypes<InputT, OffsetT>(op,
                                            temp_storage,
                                            temp_storage_bytes,
                                            input,
                                            output,
                                            segment_offsets,
                                            num_segments,
                                            stream.getStream());
        };
        dispatchOffsetDType(segment_offsets.getDataType(), dispatch_offset);
    };
    dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

}  // namespace ThorImplementation::CubReductionInternal
