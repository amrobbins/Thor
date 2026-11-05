#include "Utilities/TensorOperations/Cub/Fp32CarriedArgContestInput.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal {

Tensor Fp32CarriedArgContestInput::values(const Tensor& slot, const std::vector<uint64_t>& dims) const {
    return slot.aliasByteStorageAs(DataType::FP32, dims, 0);
}

Tensor Fp32CarriedArgContestInput::carriedIndex(const Tensor& slot, const std::vector<uint64_t>& dims) const {
    return slot.aliasByteStorageAs(index_dtype, dims, index_offset);
}

Fp32CarriedArgContestInput fp32CarriedArgContestInput(uint64_t elements, DataType index_dtype) {
    if (elements == 0 || elements > (uint64_t{1} << 60) ||
        (index_dtype != DataType::UINT32 && index_dtype != DataType::UINT64)) {
        throw std::invalid_argument("FP32 carried ARG contest input needs elements and a UINT32 / UINT64 index");
    }
    Fp32CarriedArgContestInput layout;
    layout.elements = elements;
    layout.index_dtype = index_dtype;
    layout.index_offset = (elements * 4 + 255) & ~uint64_t{255};
    layout.bytes = layout.index_offset + elements * (index_dtype == DataType::UINT64 ? 8 : 4);
    return layout;
}

Tensor makeFp32CarriedArgContestInput(const Fp32CarriedArgContestInput& layout, const Stream& stream,
                                      const Tensor* values, const Tensor* carried_index) {
    if ((values == nullptr) != (carried_index == nullptr) ||
        (values != nullptr &&
         (values->getDataType() != DataType::FP32 || values->getTotalNumElements() != layout.elements ||
          carried_index->getDataType() != layout.index_dtype ||
          carried_index->getTotalNumElements() != layout.elements))) {
        throw std::invalid_argument("FP32 carried ARG contest input received mismatched values or carried index");
    }
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    Tensor input(placement, TensorDescriptor(DataType::UINT8, {layout.bytes}));
    Stream copy_stream = stream;
    Tensor value_view = layout.values(input, {layout.elements});
    Tensor index_view = layout.carriedIndex(input, {layout.elements});
    if (values != nullptr) {
        Tensor value_source = *values;
        value_source.reshape({layout.elements});
        Tensor index_source = *carried_index;
        index_source.reshape({layout.elements});
        value_view.copyFromAsync(value_source, copy_stream);
        index_view.copyFromAsync(index_source, copy_stream);
    } else {
        input.memsetAsync(copy_stream, 0);
        value_view.fill(1.0, copy_stream);
    }
    copy_stream.synchronize();
    return input;
}

}  // namespace ThorImplementation::CubReductionInternal
