#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"

#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"

#include <cstdint>

namespace Thor {

namespace {

std::vector<uint64_t> makeLogicalValuesDimensions(uint64_t maxValuesPerRow,
                                                  const std::vector<uint64_t>& trailingDimensions) {
    THOR_THROW_IF_FALSE(maxValuesPerRow > 0);
    std::vector<uint64_t> dimensions;
    dimensions.reserve(trailingDimensions.size() + 1);
    dimensions.push_back(maxValuesPerRow);
    for (uint64_t dim : trailingDimensions) {
        THOR_THROW_IF_FALSE(dim > 0);
        dimensions.push_back(dim);
    }
    return dimensions;
}

}  // namespace

RaggedTensor RaggedNetworkInput::Builder::build() {
    THOR_THROW_IF_FALSE(network_.has_value());
    THOR_THROW_IF_FALSE(name_.has_value());
    THOR_THROW_IF_FALSE(!name_.value().empty());
    THOR_THROW_IF_FALSE(valuesDataType_.has_value());

    const std::string valuesInputName = name_.value() + ".values";

    if (partition_.has_value()) {
        THOR_THROW_IF_FALSE(partition_->isInitialized());
        // Shared-partition construction has one structural source of truth.
        // Reject duplicate metadata instead of accepting values that merely
        // happen to match the referenced partition today.
        THOR_THROW_IF_FALSE(!offsetsDataType_.has_value());
        THOR_THROW_IF_FALSE(!maxValuesPerRow_.has_value());

        const RaggedTensor& partition = partition_.value();
        THOR_THROW_IF_FALSE(partition.hasMaxValuesPerRow());
        THOR_THROW_IF_FALSE(partition.getValuesDimensions().front() == partition.getMaxValuesPerRow());
        THOR_THROW_IF_FALSE(partition.getOffsetsDimensions() == std::vector<uint64_t>{1});
        (void)partition.getDescriptorForPlacement(1);
        std::optional<RaggedNetworkInputReference> partitionSource;
        for (const RaggedNetworkInputReference& candidate : network_.value()->getExternalRaggedNetworkInputs()) {
            if (candidate.raggedTensor.sharesPartitionWith(partition)) {
                partitionSource = candidate;
                break;
            }
        }
        THOR_THROW_IF_FALSE(partitionSource.has_value());
        const std::string canonicalPartitionInputName =
            partitionSource->partitionInputName.value_or(partitionSource->name);

        NetworkInput valuesInput = NetworkInput::Builder()
                                       .network(*network_.value())
                                       .name(valuesInputName)
                                       .dimensions(makeLogicalValuesDimensions(partition.getMaxValuesPerRow(), trailingDimensions_))
                                       .dataType(valuesDataType_.value())
                                       .dimensionsIncludeBatch(true)
                                       .build();

        RaggedTensor raggedTensor = partition.withValues(valuesInput.getFeatureOutput().value());
        network_.value()->registerRaggedNetworkInput(
            name_.value(), raggedTensor, valuesInputName, canonicalPartitionInputName);
        return raggedTensor;
    }

    const DataType offsetsDataType = offsetsDataType_.value_or(DataType::UINT32);
    THOR_THROW_IF_FALSE(ThorImplementation::RowPartitionDescriptor::isValidOffsetsDataType(offsetsDataType));

    const std::string partitionTokenInputName = "__thor_row_partition." + name_.value();

    // Public API geometry is logical. Independent boundaries require a
    // semantic maximum per row; physical capacity is chosen at placement.
    THOR_THROW_IF_FALSE(maxValuesPerRow_.has_value());
    THOR_THROW_IF_FALSE(maxValuesPerRow_.value() > 0);

    NetworkInput valuesInput = NetworkInput::Builder()
                                   .network(*network_.value())
                                   .name(valuesInputName)
                                   .dimensions(makeLogicalValuesDimensions(maxValuesPerRow_.value(), trailingDimensions_))
                                   .dataType(valuesDataType_.value())
                                   .dimensionsIncludeBatch(true)
                                   .build();

    // This token is graph topology only. It is deliberately non-external and
    // Network::place() never stamps it. Requirement-driven managed inputs are the
    // only device partition allocations for this logical boundary.
    NetworkInput partitionTokenInput = NetworkInput::Builder()
                                           .network(*network_.value())
                                           .name(partitionTokenInputName)
                                           .dimensions({1})
                                           .dataType(offsetsDataType)
                                           .dimensionsIncludeBatch(true)
                                           .external(false)
                                           .build();

    RaggedTensor raggedTensor = RaggedTensor::logical(
        valuesInput.getFeatureOutput().value(), partitionTokenInput.getFeatureOutput().value(), maxValuesPerRow_.value());
    network_.value()->registerRaggedNetworkInput(name_.value(), raggedTensor, valuesInputName);
    return raggedTensor;
}

RaggedNetworkInput::Builder& RaggedNetworkInput::Builder::network(Network& network) {
    THOR_THROW_IF_FALSE(!network_.has_value());
    network_ = &network;
    return *this;
}

RaggedNetworkInput::Builder& RaggedNetworkInput::Builder::name(const std::string& name) {
    THOR_THROW_IF_FALSE(!name.empty());
    THOR_THROW_IF_FALSE(!name_.has_value());
    name_ = name;
    return *this;
}

RaggedNetworkInput::Builder& RaggedNetworkInput::Builder::valuesDataType(DataType dataType) {
    THOR_THROW_IF_FALSE(Tensor::dataTypeValid(dataType));
    valuesDataType_ = dataType;
    return *this;
}

RaggedNetworkInput::Builder& RaggedNetworkInput::Builder::offsetsDataType(DataType dataType) {
    THOR_THROW_IF_FALSE(ThorImplementation::RowPartitionDescriptor::isValidOffsetsDataType(dataType));
    THOR_THROW_IF_FALSE(!partition_.has_value());
    offsetsDataType_ = dataType;
    return *this;
}

RaggedNetworkInput::Builder& RaggedNetworkInput::Builder::trailingDimensions(const std::vector<uint64_t>& dimensions) {
    trailingDimensions_ = dimensions;
    for (uint64_t dim : trailingDimensions_) {
        THOR_THROW_IF_FALSE(dim > 0);
    }
    return *this;
}

RaggedNetworkInput::Builder& RaggedNetworkInput::Builder::maxValuesPerRow(uint64_t maxValuesPerRow) {
    THOR_THROW_IF_FALSE(maxValuesPerRow > 0);
    THOR_THROW_IF_FALSE(!partition_.has_value());
    maxValuesPerRow_ = maxValuesPerRow;
    return *this;
}

RaggedNetworkInput::Builder& RaggedNetworkInput::Builder::partition(const RaggedTensor& partition) {
    THOR_THROW_IF_FALSE(partition.isInitialized());
    THOR_THROW_IF_FALSE(!partition_.has_value());
    THOR_THROW_IF_FALSE(!offsetsDataType_.has_value());
    THOR_THROW_IF_FALSE(!maxValuesPerRow_.has_value());
    partition_ = partition;
    return *this;
}

}  // namespace Thor
