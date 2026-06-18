#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"

#include <sstream>

namespace ThorImplementation {

std::string RowPartitionDescriptor::toString() const {
    std::ostringstream out;
    out << "RowPartitionDescriptor(batch_size=" << batchSize << ", max_total_values=" << maxTotalValues
        << ", offsets_data_type=" << TensorDescriptor::getElementTypeName(offsetsDataType) << ")";
    return out.str();
}

std::vector<uint64_t> RaggedTensorDescriptor::getTrailingDimensions() const {
    std::vector<uint64_t> valuesDimensions = valuesDescriptor.getDimensions();
    THOR_THROW_IF_FALSE(!valuesDimensions.empty());
    return std::vector<uint64_t>(valuesDimensions.begin() + 1, valuesDimensions.end());
}

std::string RaggedTensorDescriptor::toString() const {
    std::ostringstream out;
    out << "RaggedTensorDescriptor(ragged_rank=" << raggedRank << ", batch_size=" << getBatchSize()
        << ", max_total_values=" << getMaxTotalValues() << ", values_data_type="
        << TensorDescriptor::getElementTypeName(getValuesDataType()) << ", offsets_data_type="
        << TensorDescriptor::getElementTypeName(getOffsetsDataType()) << ", values_dimensions=[";
    const std::vector<uint64_t> dimensions = valuesDescriptor.getDimensions();
    for (uint32_t i = 0; i < dimensions.size(); ++i) {
        out << dimensions[i];
        if (i + 1 < dimensions.size())
            out << ' ';
    }
    out << "])";
    return out.str();
}

}  // namespace ThorImplementation
