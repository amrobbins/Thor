#include "Utilities/TensorOperations/Cub/CubArgReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"

#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ThorImplementation::CubReductionInternal {
namespace {

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

template <typename InputT, typename OffsetT, typename ReductionOpT>
size_t queryForTypes(const Tensor& input,
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
void launchForTypes(const Tensor& temp_storage,
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
    auto dispatch_offset = [&]<typename OffsetT>() -> size_t {
        auto dispatch_input = [&]<typename InputT>() -> size_t {
            auto run_op = [&](auto reduction_op, ArgReductionCandidateFp32 init) -> size_t {
                return queryForTypes<InputT, OffsetT>(input,
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
    auto dispatch_offset = [&]<typename OffsetT>() -> void {
        auto dispatch_input = [&]<typename InputT>() -> void {
            auto run_op = [&](auto reduction_op, ArgReductionCandidateFp32 init) -> void {
                launchForTypes<InputT, OffsetT>(temp_storage,
                                                temp_storage_bytes,
                                                input,
                                                index_output,
                                                segment_offsets,
                                                num_segments,
                                                reduction_op,
                                                init,
                                                stream.getStream());
            };
            dispatchOperation(op, run_op);
        };
        dispatchReductionInputDType(input.getDataType(), dispatch_input);
    };
    dispatchOffsetDType(segment_offsets.getDataType(), dispatch_offset);
}

}  // namespace ThorImplementation::CubReductionInternal
