#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"

#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"

#include <cstdint>

namespace Thor {

namespace {

std::vector<uint64_t> makeValuesDimensions(uint64_t maxTotalValues, const std::vector<uint64_t>& trailingDimensions) {
    THOR_THROW_IF_FALSE(maxTotalValues > 0);
    std::vector<uint64_t> dimensions;
    dimensions.reserve(trailingDimensions.size() + 1);
    dimensions.push_back(maxTotalValues);
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
    THOR_THROW_IF_FALSE(maxTotalValues_.has_value());
    THOR_THROW_IF_FALSE(batchSize_.has_value());
    THOR_THROW_IF_FALSE(maxTotalValues_.value() > 0);
    THOR_THROW_IF_FALSE(batchSize_.value() <= UINT64_MAX - 1);
    THOR_THROW_IF_FALSE(ThorImplementation::RowPartitionDescriptor::isValidOffsetsDataType(offsetsDataType_));

    const std::string valuesInputName = name_.value() + ".values";
    const std::string offsetsInputName = name_.value() + ".offsets";

    NetworkInput valuesInput = NetworkInput::Builder()
                                   .network(*network_.value())
                                   .name(valuesInputName)
                                   .dimensions(makeValuesDimensions(maxTotalValues_.value(), trailingDimensions_))
                                   .dataType(valuesDataType_.value())
                                   .dimensionsIncludeBatch(true)
                                   .build();

    NetworkInput offsetsInput = NetworkInput::Builder()
                                    .network(*network_.value())
                                    .name(offsetsInputName)
                                    .dimensions({batchSize_.value() + 1})
                                    .dataType(offsetsDataType_)
                                    .dimensionsIncludeBatch(true)
                                    .build();

    RaggedTensor raggedTensor(valuesInput.getFeatureOutput().value(), offsetsInput.getFeatureOutput().value());
    network_.value()->registerRaggedNetworkInput(name_.value(), raggedTensor, valuesInputName, offsetsInputName);
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

RaggedNetworkInput::Builder& RaggedNetworkInput::Builder::maxTotalValues(uint64_t maxTotalValues) {
    THOR_THROW_IF_FALSE(maxTotalValues > 0);
    maxTotalValues_ = maxTotalValues;
    return *this;
}

RaggedNetworkInput::Builder& RaggedNetworkInput::Builder::batchSize(uint64_t batchSize) {
    THOR_THROW_IF_FALSE(batchSize <= UINT64_MAX - 1);
    batchSize_ = batchSize;
    return *this;
}

}  // namespace Thor
