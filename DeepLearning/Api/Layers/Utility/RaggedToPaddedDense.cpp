#include "DeepLearning/Api/Layers/Utility/RaggedToPaddedDense.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace Thor {
namespace {
// For canonical logical references, every field is durable
// semantic state. Look up by saved original ID, then verify the nested record.
void validateSemanticTensorReference(const json& serialized, const Tensor& live) {
    if (!serialized.is_object() || serialized.size() != 4 ||
        serialized.at("version").get<std::string>() != "1.0.0" ||
        serialized.at("id").get<uint64_t>() != live.getOriginalId() ||
        serialized.at("data_type").get<DataType>() != live.getDataType() ||
        serialized.at("dimensions").get<std::vector<uint64_t>>() != live.getDimensions()) {
        throw std::runtime_error("RaggedToPaddedDense semantic tensor reference disagrees with its original graph tensor.");
    }
}

void validateSemanticInputReference(const json& serialized, const RaggedTensor& live) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(serialized) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(serialized, "RaggedToPaddedDense") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        serialized.at("ragged_rank").get<uint32_t>() != 1 ||
        serialized.at("max_values_per_row").get<uint64_t>() != live.getMaxValuesPerRow() ||
        serialized.contains("batch_size") || serialized.contains("max_total_values")) {
        throw std::runtime_error("RaggedToPaddedDense semantic partition reference is malformed.");
    }
    validateSemanticTensorReference(serialized.at("values"), live.getValues());
    validateSemanticTensorReference(serialized.at("offsets"), live.getRowPartitionToken());
}
}  // namespace


RaggedToPaddedDense RaggedToPaddedDense::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value()) {
        throw std::runtime_error("RaggedToPaddedDense requires network and featureInput.");
    }
    if (!_featureInput->hasMaxValuesPerRow()) {
        throw std::invalid_argument("RaggedToPaddedDense requires featureInput.max_values_per_row.");
    }
    if (_featureInput->getMaxValuesPerRow() == 0 || _featureInput->getRaggedRank() != 1) {
        throw std::invalid_argument("RaggedToPaddedDense requires rank-1 ragged input with positive semantic row width.");
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
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    const uint64_t perExampleBytes = featureOutputs.front().getTotalSizeInBytes();
    const uint64_t b = batchSize;
    if (b == 0 || perExampleBytes > std::numeric_limits<uint64_t>::max() / b) {
        throw std::overflow_error("RaggedToPaddedDense output size has invalid batch or overflows uint64_t.");
    }
    return perExampleBytes * b;
}

uint64_t RaggedToPaddedDense::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    const auto descriptor = raggedFeatureInput.getDescriptorForPlacement(batchSize);
    const uint64_t inputValues = descriptor.getValuesDescriptor().getArraySizeInBytes();
    const uint64_t inputOffsets = descriptor.getOffsetsDescriptor().getArraySizeInBytes();
    const uint64_t output = getOutputTensorBytes(batchSize);
    if (inputValues > std::numeric_limits<uint64_t>::max() - inputOffsets ||
        inputValues + inputOffsets > std::numeric_limits<uint64_t>::max() - output) {
        throw std::overflow_error("RaggedToPaddedDense memory count overflows uint64_t.");
    }
    return inputValues + inputOffsets + output;
}

std::shared_ptr<ThorImplementation::Layer> RaggedToPaddedDense::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)placement;
    (void)drivingApiLayer;
    const int drivingPort = getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized && featureOutputs.size() == 1);

    ThorImplementation::RaggedTensorDescriptor physicalInputDescriptor;
    if (drivingPort == 0) {
        physicalInputDescriptor = SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
            raggedFeatureInput, drivingLayer, "RaggedToPaddedDense");
    } else {
        // DEVICE_OFFSETS is a real [B+1] input. It is not an active-count
        // scalar: infer B from the physical offsets descriptor when this port
        // is the first encountered connection. Managed partition metadata,
        // when available, must agree with that physical descriptor.
        const auto physicalOffsets = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(
            drivingLayer, "RaggedToPaddedDense");
        const auto dims = physicalOffsets.getDimensions();
        if (physicalOffsets.getDataType() != raggedFeatureInput.getOffsetsDataType() ||
            dims.size() != 1 || dims.front() < 2) {
            throw std::runtime_error("RaggedToPaddedDense requires a real [B+1] physical offsets driver.");
        }
        const uint64_t b = dims.front() - 1;
        physicalInputDescriptor = raggedFeatureInput.getDescriptorForPlacement(b);
        if (physicalOffsets != physicalInputDescriptor.getOffsetsDescriptor()) {
            throw std::runtime_error("RaggedToPaddedDense physical offsets do not match placement.");
        }
        const auto managed = SegmentedPrimitivePlacementDetail::rowPartitionDescriptorForStamp(drivingLayer);
        if (managed.has_value() && managed.value() != physicalInputDescriptor.getRowPartition()) {
            throw std::runtime_error("RaggedToPaddedDense managed partition metadata disagrees with offsets driver.");
        }
    }
    const uint64_t runtimeB = physicalInputDescriptor.getBatchSize();
    if (runtimeB == 0 || runtimeB > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("RaggedToPaddedDense placement batch size is out of uint32 range.");
    }
    if (physicalInputDescriptor != raggedFeatureInput.getDescriptorForPlacement(runtimeB)) {
        throw std::runtime_error("RaggedToPaddedDense physical partition descriptor disagrees with placement.");
    }
    const auto actualDriver = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(
        drivingLayer, "RaggedToPaddedDense");
    const auto expectedDriver = drivingPort == 0 ? physicalInputDescriptor.getValuesDescriptor()
                                                 : physicalInputDescriptor.getOffsetsDescriptor();
    if (actualDriver != expectedDriver) {
        throw std::runtime_error("RaggedToPaddedDense driving physical values/offsets descriptor mismatches placement.");
    }

    std::vector<uint64_t> physicalOutputDimensions;
    physicalOutputDimensions.reserve(featureOutputs.front().getDimensions().size() + 1);
    physicalOutputDimensions.push_back(runtimeB);
    const std::vector<uint64_t> outputDimensions = featureOutputs.front().getDimensions();
    physicalOutputDimensions.insert(physicalOutputDimensions.end(), outputDimensions.begin(), outputDimensions.end());

    auto physical = std::make_shared<ThorImplementation::RaggedToPaddedDense>(
        physicalInputDescriptor,
        ThorImplementation::TensorDescriptor(featureOutputs.front().getDataType(), physicalOutputDimensions),
        paddingValue);
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedToPaddedDense::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized && featureOutputs.size() == 1);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", "1.1.0"},
                {"layer_type", "ragged_to_padded_dense"},
                {"ragged_feature_input", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureInput)},
                {"padding_value", paddingValue},
                {"feature_output", featureOutputs.front().architectureJson()}};
}

void RaggedToPaddedDense::deserialize(const json& j, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported version in RaggedToPaddedDense::deserialize: " + version);
    }
    if (j.at("layer_type").get<std::string>() != "ragged_to_padded_dense") {
        throw std::runtime_error("Layer type mismatch in RaggedToPaddedDense::deserialize: " + j.at("layer_type").get<std::string>());
    }
    if (version == "1.1.0" && (!j.is_object() || j.size() != 6 ||
                               j.at("factory") != Layer::Factory::Layer.value())) {
        throw std::runtime_error("RaggedToPaddedDense neutral layer schema has unexpected or missing fields.");
    }

    const json& inputJson = j.at("ragged_feature_input");
    RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(inputJson, network, "RaggedToPaddedDense");
    // The layer version describes an archive grammar, not the model's
    // placement geometry. Validate semantic references whenever that grammar
    // is present, including when an older layer record embeds it.
    if (SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(inputJson, "RaggedToPaddedDense") ==
        SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL) {
        validateSemanticInputReference(inputJson, input);
    }
    if (!input.hasMaxValuesPerRow()) {
        throw std::runtime_error("RaggedToPaddedDense serialized input must carry max_values_per_row.");
    }
    const json& outputJson = j.at("feature_output");
    Tensor output = Tensor::deserialize(outputJson);
    if (SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(inputJson, "RaggedToPaddedDense") ==
        SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL) {
        validateSemanticTensorReference(outputJson, output);
        if (output.getOriginalId() == input.getValues().getOriginalId() ||
            output.getOriginalId() == input.getOffsets().getOriginalId()) {
            throw std::runtime_error("RaggedToPaddedDense output must have its own values tensor identity.");
        }
    }
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
