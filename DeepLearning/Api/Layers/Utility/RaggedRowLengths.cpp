#include "DeepLearning/Api/Layers/Utility/RaggedRowLengths.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <limits>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {

namespace {

std::vector<uint64_t> contiguousStrides(const std::vector<uint64_t>& dimensions) {
    std::vector<uint64_t> strides(dimensions.size(), 1);
    uint64_t stride = 1;
    for (size_t axis = dimensions.size(); axis-- > 0;) {
        strides[axis] = stride;
        if (dimensions[axis] != 0 && stride > std::numeric_limits<uint64_t>::max() / dimensions[axis]) {
            throw std::overflow_error("RaggedRowLengths stride overflow.");
        }
        stride *= dimensions[axis];
    }
    return strides;
}

}  // namespace

RaggedRowLengths RaggedRowLengths::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value()) {
        throw std::runtime_error("RaggedRowLengths requires network and featureInput.");
    }
    if (!_featureInput->isPlacementNeutral() &&
        (_featureInput->getBatchSize() == 0 || _featureInput->getBatchSize() > std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("RaggedRowLengths requires a nonzero batch size within uint32 capacity.");
    }

    RaggedRowLengths layer;
    layer.raggedFeatureInput = _featureInput.value();
    // Row lengths are a property of the partition only. Deliberately depend on
    // offsets, not packed values, so cloning/serialization and graph liveness do
    // not invent a mathematical dependency on values.
    layer.featureInput = _featureInput->getOffsets();
    layer.featureOutput = Tensor(DataType::INT32, {1});
    layer.initialized = true;
    layer.addToNetwork(_network.value());
    return layer;
}

uint64_t RaggedRowLengths::getOutputTensorBytes(uint32_t batchSize) const {
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    const uint64_t perRow = featureOutput->getTotalSizeInBytes();
    const uint64_t physicalBatchSize = raggedFeatureInput.isPlacementNeutral()
        ? static_cast<uint64_t>(batchSize)
        : raggedFeatureInput.getBatchSize();
    if (physicalBatchSize != 0 && perRow > std::numeric_limits<uint64_t>::max() / physicalBatchSize) {
        throw std::overflow_error("RaggedRowLengths output size overflows uint64_t.");
    }
    return perRow * physicalBatchSize;
}

uint64_t RaggedRowLengths::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    THOR_THROW_IF_FALSE(featureInput.has_value());

    if (raggedFeatureInput.isPlacementNeutral()) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const ThorImplementation::RaggedTensorDescriptor physicalInputDescriptor =
            raggedFeatureInput.getDescriptorForPlacement(batchSize);
        const uint64_t offsetsBytes = physicalInputDescriptor.getOffsetsDescriptor().getArraySizeInBytes();
        const uint64_t outputBytes = getOutputTensorBytes(batchSize);
        if (offsetsBytes > std::numeric_limits<uint64_t>::max() - outputBytes) {
            throw std::overflow_error("RaggedRowLengths neutral memory requirement overflows uint64_t.");
        }
        return offsetsBytes + outputBytes;
    }

    const uint64_t outputBytes = getOutputTensorBytes(0);
    const uint64_t inputBytes = featureInput->getTotalSizeInBytes();
    if (outputBytes > std::numeric_limits<uint64_t>::max() - inputBytes) {
        throw std::overflow_error("RaggedRowLengths memory requirement overflows uint64_t.");
    }
    return outputBytes + inputBytes;
}

std::shared_ptr<ThorImplementation::Layer> RaggedRowLengths::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value() && featureOutput.has_value());
    THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());

    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;

    const ThorImplementation::RaggedTensorDescriptor physicalInputDescriptor =
        SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
            raggedFeatureInput, drivingLayer, "RaggedRowLengths");
    const uint64_t physicalBatchSize = physicalInputDescriptor.getBatchSize();
    if (physicalBatchSize == 0 || physicalBatchSize > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("RaggedRowLengths placement batch size exceeds Thor's uint32 placement capacity.");
    }
    const uint32_t placementBatchSize = static_cast<uint32_t>(physicalBatchSize);

    Expression offsets = Expression::input("offsets");
    const std::vector<uint64_t> rowShape{placementBatchSize, 1};
    const std::vector<uint64_t> rowStrides = contiguousStrides(rowShape);
    Expression starts = offsets.stridedView(rowShape, rowStrides, 0);
    Expression ends = offsets.stridedView(rowShape, rowStrides, 1);
    Expression lengths = (ends - starts).cast(ThorImplementation::DataType::INT32);
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
        Expression::outputs({{"feature_output", lengths}}));

    auto physical = std::make_shared<ThorImplementation::CustomLayer>(
        DynamicExpression::fromExpressionDefinition(definition),
        std::vector<std::string>{"offsets"},
        std::vector<std::string>{"feature_output"},
        placement,
        std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{},
        inferenceOnly,
        getId(),
        std::vector<ThorImplementation::CustomLayer::DeclaredOutputDescriptor>{
            {ThorImplementation::DataType::INT32, {1}, false}},
        false,
        false,
        std::vector<bool>{true},
        placementBatchSize);
    physical->setLayerName(getLayerType());
    return physical;
}

json RaggedRowLengths::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value() && featureOutput.has_value());
    const bool placementNeutral = raggedFeatureInput.isPlacementNeutral();
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", placementNeutral ? "1.1.0" : getLayerVersion()},
                {"layer_type", "ragged_row_lengths"},
                {"ragged_feature_input",
                 SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureInput)},
                {"offsets_input", featureInput->architectureJson()},
                {"feature_output", featureOutput->architectureJson()}};
}

void RaggedRowLengths::deserialize(const json& j, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported version in RaggedRowLengths::deserialize: " + version);
    }
    if (j.at("layer_type").get<std::string>() != "ragged_row_lengths") {
        throw std::runtime_error("Layer type mismatch in RaggedRowLengths::deserialize: " + j.at("layer_type").get<std::string>());
    }

    const json& inputJson = j.at("ragged_feature_input");
    RaggedTensor ragged = SegmentedPrimitiveDetail::reconstructInput(inputJson, network, "RaggedRowLengths");
    if ((version == "1.0.0" && ragged.isPlacementNeutral()) ||
        (version == "1.1.0" && !ragged.isPlacementNeutral())) {
        throw std::runtime_error("RaggedRowLengths serialized layer version does not match its ragged geometry mode.");
    }

    const json& offsetsJson = j.at("offsets_input");
    const uint64_t serializedOffsetsId = offsetsJson.at("id").get<uint64_t>();
    if (serializedOffsetsId != inputJson.at("offsets").at("id").get<uint64_t>()) {
        throw std::runtime_error("RaggedRowLengths serialized offsets_input must match its ragged input partition token.");
    }
    if (offsetsJson.at("dimensions").get<std::vector<uint64_t>>() != ragged.getOffsetsDimensions() ||
        offsetsJson.at("data_type").get<DataType>() != ragged.getOffsetsDataType()) {
        throw std::runtime_error("RaggedRowLengths serialized offsets_input descriptor must match its ragged input partition token.");
    }
    Tensor offsets = network->getApiTensorByOriginalId(serializedOffsetsId);
    if (offsets != ragged.getOffsets()) {
        throw std::runtime_error("RaggedRowLengths reconstructed offsets_input must match its ragged input partition token.");
    }

    Tensor output = Tensor::deserialize(j.at("feature_output"));
    if (output.getDataType() != DataType::INT32 || output.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::runtime_error("RaggedRowLengths serialized output must be INT32 logical [1].");
    }

    RaggedRowLengths layer;
    layer.raggedFeatureInput = ragged;
    layer.featureInput = offsets;
    layer.featureOutput = output;
    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("ragged_row_lengths", &Thor::RaggedRowLengths::deserialize);
    return true;
}();
}
