#include "DeepLearning/Api/Layers/Utility/SegmentedSoftmax.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include <limits>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

namespace Thor {

SegmentedSoftmax SegmentedSoftmax::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value()) {
        throw std::runtime_error("SegmentedSoftmax requires network and featureInput.");
    }
    SegmentedPrimitiveDetail::requireSupportedValueDataType(_featureInput->getValuesDataType(), "SegmentedSoftmax");

    SegmentedSoftmax layer;
    layer.raggedFeatureInput = _featureInput.value();
    layer.raggedFeatureOutput = _featureInput->withValues(
        Tensor(_featureInput->getValuesDataType(), _featureInput->getValuesDimensions()));
    layer.featureInputs = {_featureInput->getValues(), _featureInput->getOffsets()};
    layer.featureOutputs = {layer.raggedFeatureOutput.getValues()};
    layer.initialized = true;
    layer.addToNetwork(_network.value());
    return layer;
}

std::vector<Tensor> SegmentedSoftmax::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs.front()};
}

void SegmentedSoftmax::informThatInputConnectionMade(Tensor inputTensor) {
    connectedInputPortIndices.insert(static_cast<uint32_t>(getConnectionType(inputTensor)));
}

void SegmentedSoftmax::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int SegmentedSoftmax::getConnectionType(Tensor connectingTensor) const {
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (connectingTensor == featureInputs[i]) return static_cast<int>(i);
    }
    if (!featureOutputs.empty() && connectingTensor == featureOutputs.front()) return 0;
    throw std::runtime_error("Tensor is not connected to this SegmentedSoftmax layer.");
}

uint64_t SegmentedSoftmax::getOutputTensorBytes(uint32_t batchSize) const {
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    const uint64_t logicalBytes = featureOutputs.front().getTotalSizeInBytes();
    if (!raggedFeatureOutput.isPlacementNeutral()) return logicalBytes;
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(logicalBytes <= std::numeric_limits<uint64_t>::max() / batchSize);
    return logicalBytes * batchSize;
}

std::shared_ptr<ThorImplementation::Layer> SegmentedSoftmax::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized);

    using ThorImplementation::DynamicExpression;
    using ThorImplementation::Expression;
    using ThorImplementation::ExpressionDefinition;
    using ThorImplementation::RaggedExpression;

    const ThorImplementation::RaggedTensorDescriptor physicalInputDescriptor =
        SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
            raggedFeatureInput, drivingLayer, "SegmentedSoftmax");
    const ThorImplementation::RaggedTensorDescriptor physicalOutputDescriptor =
        raggedFeatureOutput.isPlacementNeutral()
        ? raggedFeatureOutput.getDescriptorForPlacement(physicalInputDescriptor.getBatchSize())
        : raggedFeatureOutput.getDescriptor();

    RaggedExpression input =
        RaggedExpression::input("feature_input", "feature_offsets", physicalInputDescriptor);
    RaggedExpression output = input.segment_softmax();
    if (output.getDescriptor() != physicalOutputDescriptor) {
        throw std::runtime_error("SegmentedSoftmax expression output descriptor does not match the API output.");
    }

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
        Expression::outputs({{"feature_output", output.getValues()}}));
    const uint64_t elementsPerValue =
        SegmentedPrimitiveDetail::elementsPerValue(raggedFeatureInput.getTrailingDimensions(), "SegmentedSoftmax");
    auto physical = std::make_shared<ThorImplementation::RaggedCustomLayer>(
        DynamicExpression::fromExpressionDefinition(definition),
        std::vector<std::string>{"feature_input", "feature_offsets"},
        std::vector<std::string>{"feature_output"},
        placement,
        inferenceOnly,
        physicalInputDescriptor.getMaxTotalValues(),
        elementsPerValue,
        elementsPerValue,
        0,
        1,
        getId());
    physical->setLayerName(getLayerType());
    return physical;
}

json SegmentedSoftmax::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    const bool placementNeutral = raggedFeatureInput.isPlacementNeutral();
    THOR_THROW_IF_FALSE(placementNeutral == raggedFeatureOutput.isPlacementNeutral());
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", placementNeutral ? "1.1.0" : getLayerVersion()},
                {"layer_type", "segmented_softmax"},
                {"ragged_feature_input",
                 SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureInput)},
                {"ragged_feature_output",
                 SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureOutput)}};
}

void SegmentedSoftmax::deserialize(const json& j, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported version in SegmentedSoftmax::deserialize: " + version);
    }
    if (j.at("layer_type").get<std::string>() != "segmented_softmax") {
        throw std::runtime_error("Layer type mismatch in SegmentedSoftmax::deserialize: " + j.at("layer_type").get<std::string>());
    }

    RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_feature_input"), network, "SegmentedSoftmax");
    if ((version == "1.0.0" && input.isPlacementNeutral()) ||
        (version == "1.1.0" && !input.isPlacementNeutral())) {
        throw std::runtime_error("SegmentedSoftmax serialized layer version does not match its ragged geometry mode.");
    }
    SegmentedPrimitiveDetail::requireSupportedValueDataType(input.getValuesDataType(), "SegmentedSoftmax");
    const json& outputJson = j.at("ragged_feature_output");
    SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
        outputJson, j.at("ragged_feature_input"), input, "SegmentedSoftmax");
    Tensor outputValues = Tensor::deserialize(outputJson.at("values"));
    if (outputValues.getDataType() != input.getValuesDataType() || outputValues.getDimensions() != input.getValuesDimensions()) {
        throw std::runtime_error("SegmentedSoftmax serialized output values descriptor does not match its input.");
    }
    RaggedTensor output = input.withValues(outputValues);

    SegmentedSoftmax layer;
    layer.raggedFeatureInput = input;
    layer.raggedFeatureOutput = output;
    layer.featureInputs = {input.getValues(), input.getOffsets()};
    layer.featureOutputs = {output.getValues()};
    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("segmented_softmax", &Thor::SegmentedSoftmax::deserialize);
    return true;
}();
}  // namespace
