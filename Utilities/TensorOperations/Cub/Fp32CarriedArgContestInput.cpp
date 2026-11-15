#include "Utilities/TensorOperations/Cub/Fp32CarriedArgContestInput.h"

#include "Utilities/TensorOperations/Cub/ReductionAutotunerCore.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionInternal {

StridedSpan Fp32CarriedArgContestInput::values(const StridedSpan& slot, const std::vector<uint64_t>& dims) const {
    return slot.bytesAs(0, DataType::FP32, dims);
}

StridedSpan Fp32CarriedArgContestInput::carriedIndex(const StridedSpan& slot, const std::vector<uint64_t>& dims) const {
    return slot.bytesAs(index_offset, index_dtype, dims);
}

Fp32CarriedArgContestInput fp32CarriedArgContestInput(uint64_t elements, DataType index_dtype) {
    if (elements == 0 || elements > (uint64_t{1} << 60) ||
        (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)) {
        throw std::invalid_argument("FP32 carried ARG contest input needs elements and a UINT32 / UINT64 index");
    }
    Fp32CarriedArgContestInput layout;
    layout.elements = elements;
    layout.index_dtype = index_dtype;
    layout.index_offset = ReductionAutotuning::nextSliceOffset(elements * 4);
    layout.bytes = layout.index_offset + elements * (index_dtype == DataType::UINT64 ? 8 : 4);
    return layout;
}

namespace {
void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("FP32 carried ARG contest input: ") + what + ": " + cudaGetErrorString(status));
    }
}
}  // namespace

Tensor makeFp32CarriedArgContestInput(const Fp32CarriedArgContestInput& layout, const Stream& stream,
                                      const Tensor* values, const Tensor* carried_index) {
    if ((values == nullptr) != (carried_index == nullptr) ||
        (values != nullptr &&
         (values->getDataType() != DataType::FP32 || values->getTotalNumElements() != layout.elements ||
          !values->isDenseContiguous() || carried_index->getDataType() != layout.index_dtype ||
          carried_index->getTotalNumElements() != layout.elements || !carried_index->isDenseContiguous()))) {
        throw std::invalid_argument("FP32 carried ARG contest input received mismatched values or carried index");
    }
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    Tensor input(placement, TensorDescriptor(DataType::UINT8, {layout.bytes}));
    Stream copy_stream = stream;
    const StridedSpan value_slice = layout.values(input, {layout.elements});
    const StridedSpan index_slice = layout.carriedIndex(input, {layout.elements});
    if (values != nullptr) {
        checkCuda(cudaMemcpyAsync(value_slice.getMemPtr<void>(), values->getMemPtr<void>(), value_slice.getArraySizeInBytes(),
                                  cudaMemcpyDeviceToDevice, copy_stream.getStream()),
                  "copy values");
        checkCuda(cudaMemcpyAsync(index_slice.getMemPtr<void>(), carried_index->getMemPtr<void>(),
                                  index_slice.getArraySizeInBytes(), cudaMemcpyDeviceToDevice, copy_stream.getStream()),
                  "copy carried index");
    } else {
        // Values 1.0 (a dense FP32 tensor of its own, copied in) and index 0.
        input.memsetAsync(copy_stream, 0);
        Tensor ones(placement, TensorDescriptor(DataType::FP32, {layout.elements}));
        ones.fill(1.0, copy_stream);
        checkCuda(cudaMemcpyAsync(value_slice.getMemPtr<void>(), ones.getMemPtr<void>(), value_slice.getArraySizeInBytes(),
                                  cudaMemcpyDeviceToDevice, copy_stream.getStream()),
                  "copy representative values");
    }
    copy_stream.synchronize();
    return input;
}

}  // namespace ThorImplementation::CubReductionInternal
