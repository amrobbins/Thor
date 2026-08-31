#include "DeepLearning/Api/Layers/Utility/SegmentedBroadcast.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace Thor {

SegmentedBroadcast SegmentedBroadcast::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value() || !_partitionInput.has_value()) {
        throw std::runtime_error("SegmentedBroadcast requires network, featureInput, and partitionInput.");
    }
    SegmentedPrimitiveDetail::requireSupportedValueDataType(_featureInput->getDataType(), "SegmentedBroadcast");
    if (_partitionInput->getBatchSize() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("SegmentedBroadcast batch size exceeds Thor's uint32 placement capacity.");
    }

    const std::vector<uint64_t> featureInputDimensions = _featureInput->getDimensions();
    std::vector<uint64_t> outputDimensions;
    outputDimensions.reserve(featureInputDimensions.size() + 1);
    outputDimensions.push_back(_partitionInput->getMaxTotalValues());
    outputDimensions.insert(outputDimensions.end(), featureInputDimensions.begin(), featureInputDimensions.end());

    SegmentedBroadcast layer;
    layer.denseFeatureInput = _featureInput.value();
    layer.partitionInput = _partitionInput.value();
    layer.raggedFeatureOutput = _partitionInput->withValues(Tensor(_featureInput->getDataType(), outputDimensions));
    layer.featureInputs = {_featureInput.value(), _partitionInput->getOffsets()};
    layer.featureOutputs = {layer.raggedFeatureOutput.getValues()};
    layer.initialized = true;
    layer.addToNetwork(_network.value());
    return layer;
}

std::vector<Tensor> SegmentedBroadcast::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs.front()};
}

void SegmentedBroadcast::informThatInputConnectionMade(Tensor inputTensor) {
    connectedInputPortIndices.insert(static_cast<uint32_t>(getConnectionType(inputTensor)));
}

void SegmentedBroadcast::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int SegmentedBroadcast::getConnectionType(Tensor connectingTensor) const {
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (connectingTensor == featureInputs[i]) return static_cast<int>(i);
    }
    if (!featureOutputs.empty() && connectingTensor == featureOutputs.front()) return 0;
    throw std::runtime_error("Tensor is not connected to this SegmentedBroadcast layer.");
}

uint64_t SegmentedBroadcast::getOutputTensorBytes(uint32_t batchSize) const {
    (void)batchSize;
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    return featureOutputs.front().getTotalSizeInBytes();
}

std::shared_ptr<ThorImplementation::Layer> SegmentedBroadcast::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized);

    using ThorImplementation::CustomLayer;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;
    using ThorImplementation::RaggedExpression;

    Expression perSegmentValues = Expression::input("per_segment_values", std::nullopt, denseFeatureInput.getDataType());
    // The values side of this RaggedExpression is intentionally unreachable from
    // the output graph. It is only a convenient carrier for the canonical offsets
    // and partition capacity metadata used by SEGMENTED_BROADCAST.
    RaggedExpression partition =
        RaggedExpression::input("__unused_partition_values", "feature_offsets", partitionInput.getDescriptor());
    RaggedExpression output = partition.segment_broadcast(perSegmentValues, raggedFeatureOutput.getDescriptor());
    if (output.getDescriptor() != raggedFeatureOutput.getDescriptor()) {
        throw std::runtime_error("SegmentedBroadcast expression output descriptor does not match the API output.");
    }

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
        Expression::outputs({{"feature_output", output.getValues()}}));
    auto physical = std::make_shared<CustomLayer>(
        DynamicExpression::fromExpressionDefinition(definition),
        std::vector<std::string>{"per_segment_values", "feature_offsets"},
        std::vector<std::string>{"feature_output"},
        placement,
        std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{},
        inferenceOnly,
        getId(),
        std::vector<CustomLayer::DeclaredOutputDescriptor>{
            {raggedFeatureOutput.getValuesDataType(), raggedFeatureOutput.getValuesDimensions(), true}},
        false,
        false,
        std::vector<bool>{false, true},
        static_cast<uint32_t>(partitionInput.getBatchSize()));
    physical->setLayerName(getLayerType());
    return physical;
}

json SegmentedBroadcast::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", getLayerVersion()},
                {"layer_type", "segmented_broadcast"},
                {"feature_input", denseFeatureInput.architectureJson()},
                {"partition_input", partitionInput.architectureJson()},
                {"ragged_feature_output", raggedFeatureOutput.architectureJson()}};
}

void SegmentedBroadcast::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in SegmentedBroadcast::deserialize: " + j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "segmented_broadcast") {
        throw std::runtime_error("Layer type mismatch in SegmentedBroadcast::deserialize: " + j.at("layer_type").get<std::string>());
    }

    Tensor denseInput = network->getApiTensorByOriginalId(j.at("feature_input").at("id").get<uint64_t>());
    SegmentedPrimitiveDetail::requireSupportedValueDataType(denseInput.getDataType(), "SegmentedBroadcast");
    RaggedTensor partition = SegmentedPrimitiveDetail::reconstructInput(j.at("partition_input"), network, "SegmentedBroadcast");
    if (partition.getBatchSize() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("SegmentedBroadcast serialized batch size exceeds Thor's uint32 placement capacity.");
    }

    const json& outputJson = j.at("ragged_feature_output");
    SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
        outputJson, j.at("partition_input"), partition, "SegmentedBroadcast");
    Tensor outputValues = Tensor::deserialize(outputJson.at("values"));
    const std::vector<uint64_t> denseInputDimensions = denseInput.getDimensions();
    std::vector<uint64_t> expectedOutputDimensions;
    expectedOutputDimensions.reserve(denseInputDimensions.size() + 1);
    expectedOutputDimensions.push_back(partition.getMaxTotalValues());
    expectedOutputDimensions.insert(expectedOutputDimensions.end(), denseInputDimensions.begin(), denseInputDimensions.end());
    if (outputValues.getDataType() != denseInput.getDataType() || outputValues.getDimensions() != expectedOutputDimensions) {
        throw std::runtime_error("SegmentedBroadcast serialized output descriptor does not match the dense input and partition capacity.");
    }
    RaggedTensor output = partition.withValues(outputValues);

    SegmentedBroadcast layer;
    layer.denseFeatureInput = denseInput;
    layer.partitionInput = partition;
    layer.raggedFeatureOutput = output;
    layer.featureInputs = {denseInput, partition.getOffsets()};
    layer.featureOutputs = {output.getValues()};
    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("segmented_broadcast", &Thor::SegmentedBroadcast::deserialize);
    return true;
}();
}  // namespace
