#include "DeepLearning/Api/Layers/Utility/RaggedToPaddedDense.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace Thor {

RaggedToPaddedDense RaggedToPaddedDense::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value()) {
        throw std::runtime_error("RaggedToPaddedDense requires network and featureInput.");
    }
    if (!_featureInput->hasMaxValuesPerRow()) {
        throw std::invalid_argument("RaggedToPaddedDense requires featureInput.max_values_per_row.");
    }
    if (_featureInput->getBatchSize() == 0 || _featureInput->getBatchSize() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("RaggedToPaddedDense requires a non-zero batch size within uint32 placement capacity.");
    }

    std::vector<uint64_t> outputDimensions;
    outputDimensions.reserve(_featureInput->getTrailingDimensions().size() + 1);
    outputDimensions.push_back(_featureInput->getMaxValuesPerRow());
    const std::vector<uint64_t> trailing = _featureInput->getTrailingDimensions();
    outputDimensions.insert(outputDimensions.end(), trailing.begin(), trailing.end());

    RaggedToPaddedDense layer;
    layer.raggedFeatureInput = _featureInput.value();
    layer.paddingValue = _paddingValue;
    layer.featureInputs = {_featureInput->getValues(), _featureInput->getOffsets()};
    layer.featureOutputs = {Tensor(_featureInput->getValuesDataType(), outputDimensions)};
    layer.initialized = true;
    layer.addToNetwork(_network.value());
    return layer;
}

std::vector<Tensor> RaggedToPaddedDense::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs.front()};
}

void RaggedToPaddedDense::informThatInputConnectionMade(Tensor inputTensor) {
    connectedInputPortIndices.insert(static_cast<uint32_t>(getConnectionType(inputTensor)));
}

void RaggedToPaddedDense::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int RaggedToPaddedDense::getConnectionType(Tensor connectingTensor) const {
    if (connectingTensor == raggedFeatureInput.getValues()) return 0;
    if (connectingTensor == raggedFeatureInput.getOffsets()) return 1;
    if (!featureOutputs.empty() && connectingTensor == featureOutputs.front()) return 0;
    throw std::runtime_error("Tensor is not connected to this RaggedToPaddedDense layer.");
}

std::optional<std::string> RaggedToPaddedDense::getInputPortName(const Tensor& inputTensor) const {
    if (inputTensor == raggedFeatureInput.getValues()) return "values";
    if (inputTensor == raggedFeatureInput.getOffsets()) return "offsets";
    return std::nullopt;
}

std::optional<std::string> RaggedToPaddedDense::getOutputPortName(const Tensor& outputTensor) const {
    if (!featureOutputs.empty() && outputTensor == featureOutputs.front()) return "padded_values";
    return std::nullopt;
}

uint64_t RaggedToPaddedDense::getOutputTensorBytes(uint32_t batchSize) const {
    (void)batchSize;
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    const uint64_t perExampleBytes = featureOutputs.front().getTotalSizeInBytes();
    if (raggedFeatureInput.getBatchSize() != 0 &&
        perExampleBytes > std::numeric_limits<uint64_t>::max() / raggedFeatureInput.getBatchSize()) {
        throw std::overflow_error("RaggedToPaddedDense output size overflows uint64_t.");
    }
    return perExampleBytes * raggedFeatureInput.getBatchSize();
}

uint64_t RaggedToPaddedDense::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    return getOutputTensorBytes(batchSize) + sizeof(uint32_t);
}

std::shared_ptr<ThorImplementation::Layer> RaggedToPaddedDense::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)placement;
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized && featureOutputs.size() == 1);

    std::vector<uint64_t> physicalOutputDimensions;
    physicalOutputDimensions.reserve(featureOutputs.front().getDimensions().size() + 1);
    physicalOutputDimensions.push_back(raggedFeatureInput.getBatchSize());
    const std::vector<uint64_t> outputDimensions = featureOutputs.front().getDimensions();
    physicalOutputDimensions.insert(physicalOutputDimensions.end(), outputDimensions.begin(), outputDimensions.end());

    auto physical = std::make_shared<ThorImplementation::RaggedToPaddedDense>(
        raggedFeatureInput.getDescriptor(),
        ThorImplementation::TensorDescriptor(featureOutputs.front().getDataType(), physicalOutputDimensions),
        paddingValue);
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedToPaddedDense::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized && featureOutputs.size() == 1);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", getLayerVersion()},
                {"layer_type", "ragged_to_padded_dense"},
                {"ragged_feature_input", raggedFeatureInput.architectureJson()},
                {"padding_value", paddingValue},
                {"feature_output", featureOutputs.front().architectureJson()}};
}

void RaggedToPaddedDense::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in RaggedToPaddedDense::deserialize: " + j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "ragged_to_padded_dense") {
        throw std::runtime_error("Layer type mismatch in RaggedToPaddedDense::deserialize: " + j.at("layer_type").get<std::string>());
    }

    RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_feature_input"), network, "RaggedToPaddedDense");
    if (!input.hasMaxValuesPerRow()) {
        throw std::runtime_error("RaggedToPaddedDense serialized input must carry max_values_per_row.");
    }
    Tensor output = Tensor::deserialize(j.at("feature_output"));
    std::vector<uint64_t> expectedDimensions;
    expectedDimensions.push_back(input.getMaxValuesPerRow());
    const std::vector<uint64_t> trailing = input.getTrailingDimensions();
    expectedDimensions.insert(expectedDimensions.end(), trailing.begin(), trailing.end());
    if (output.getDataType() != input.getValuesDataType() || output.getDimensions() != expectedDimensions) {
        throw std::runtime_error("RaggedToPaddedDense serialized output descriptor does not match the ragged input.");
    }

    RaggedToPaddedDense layer;
    layer.raggedFeatureInput = input;
    layer.paddingValue = j.at("padding_value").get<double>();
    layer.featureInputs = {input.getValues(), input.getOffsets()};
    layer.featureOutputs = {output};
    layer.initialized = true;
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("ragged_to_padded_dense", &Thor::RaggedToPaddedDense::deserialize);
    return true;
}();
}  // namespace
