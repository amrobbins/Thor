#include "DeepLearning/Api/Layers/Utility/RaggedGather.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace Thor {
namespace {

bool validIndexDataType(DataType dataType) {
    return dataType == DataType::UINT32 || dataType == DataType::UINT64;
}

}  // namespace

RaggedGather RaggedGather::makeLayer(const RaggedTensor& sourceInput,
                                     const RaggedTensor& indicesInput,
                                     const std::optional<RaggedTensor>& serializedOutput) {
    if (!sourceInput.isInitialized() || !indicesInput.isInitialized()) {
        throw std::invalid_argument("RaggedGather inputs must be initialized RaggedTensor objects.");
    }
    if (sourceInput.getBatchSize() == 0 || sourceInput.getBatchSize() != indicesInput.getBatchSize()) {
        throw std::invalid_argument("RaggedGather source and indices must have the same non-zero logical batch size.");
    }
    if (sourceInput.getBatchSize() > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("RaggedGather batch size exceeds Thor's uint32 placement capacity.");
    }
    if (!validIndexDataType(indicesInput.getValuesDataType())) {
        throw std::invalid_argument("RaggedGather indices values must use UINT32 or UINT64 dtype.");
    }
    if (!indicesInput.getTrailingDimensions().empty()) {
        throw std::invalid_argument("RaggedGather indices must contain exactly one row-local scalar index per ragged token.");
    }
    if (sourceInput.getValues() == indicesInput.getValues()) {
        throw std::invalid_argument("RaggedGather source and indices values must be distinct graph tensors.");
    }

    std::vector<uint64_t> outputValuesDimensions;
    outputValuesDimensions.reserve(sourceInput.getTrailingDimensions().size() + 1);
    outputValuesDimensions.push_back(indicesInput.getMaxTotalValues());
    const std::vector<uint64_t> trailingDimensions = sourceInput.getTrailingDimensions();
    outputValuesDimensions.insert(outputValuesDimensions.end(), trailingDimensions.begin(), trailingDimensions.end());

    RaggedTensor output;
    if (serializedOutput.has_value()) {
        output = serializedOutput.value();
    } else {
        output = indicesInput.withValues(Tensor(sourceInput.getValuesDataType(), outputValuesDimensions));
    }

    if (output.getValuesDataType() != sourceInput.getValuesDataType() ||
        output.getTrailingDimensions() != sourceInput.getTrailingDimensions() ||
        output.getBatchSize() != indicesInput.getBatchSize() ||
        output.getMaxTotalValues() != indicesInput.getMaxTotalValues() ||
        output.getOffsets() != indicesInput.getOffsets() ||
        output.hasMaxValuesPerRow() != indicesInput.hasMaxValuesPerRow() ||
        (indicesInput.hasMaxValuesPerRow() && output.getMaxValuesPerRow() != indicesInput.getMaxValuesPerRow())) {
        throw std::runtime_error("RaggedGather serialized output must use source value geometry and preserve indices partition Q exactly.");
    }

    RaggedGather layer;
    layer.raggedSourceInput = sourceInput;
    layer.raggedIndicesInput = indicesInput;
    layer.raggedFeatureOutput = output;
    layer.sharedOffsets = sourceInput.getOffsets() == indicesInput.getOffsets();
    layer.indicesOffsetsInputPort = layer.sharedOffsets ? 2U : 3U;
    layer.featureInputs = {sourceInput.getValues(), indicesInput.getValues(), sourceInput.getOffsets()};
    if (!layer.sharedOffsets) layer.featureInputs.push_back(indicesInput.getOffsets());
    layer.featureOutputs = {output.getValues()};
    layer.initialized = true;
    return layer;
}

RaggedGather RaggedGather::Builder::build() {
    if (!_network.has_value() || !_sourceInput.has_value() || !_indicesInput.has_value()) {
        throw std::runtime_error("RaggedGather requires network, sourceInput, and indicesInput.");
    }
    RaggedGather layer = RaggedGather::makeLayer(_sourceInput.value(), _indicesInput.value());
    layer.addToNetwork(_network.value());
    return layer;
}

std::vector<Tensor> RaggedGather::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedOutputsAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedOutputsAfterAllInputsConnected = true;
    return featureOutputs;
}

void RaggedGather::informThatInputConnectionMade(Tensor inputTensor) {
    const int connectionType = getConnectionType(inputTensor);
    THOR_THROW_IF_FALSE(connectionType >= 0);
    connectedInputPortIndices.insert(static_cast<uint32_t>(connectionType));
}

void RaggedGather::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedOutputsAfterAllInputsConnected = false;
}

int RaggedGather::getConnectionType(Tensor connectingTensor) const {
    if (connectingTensor == raggedSourceInput.getValues()) return 0;
    if (connectingTensor == raggedIndicesInput.getValues()) return 1;
    if (connectingTensor == raggedSourceInput.getOffsets()) return 2;
    if (!sharedOffsets && connectingTensor == raggedIndicesInput.getOffsets()) return 3;
    if (connectingTensor == raggedFeatureOutput.getValues()) return 0;
    throw std::runtime_error("Tensor is not connected to this RaggedGather layer.");
}

std::optional<std::string> RaggedGather::getInputPortName(const Tensor& inputTensor) const {
    if (inputTensor == raggedSourceInput.getValues()) return "source_values";
    if (inputTensor == raggedIndicesInput.getValues()) return "indices_values";
    if (inputTensor == raggedSourceInput.getOffsets()) return sharedOffsets ? "shared_offsets" : "source_offsets";
    if (!sharedOffsets && inputTensor == raggedIndicesInput.getOffsets()) return "indices_offsets";
    return std::nullopt;
}

std::optional<std::string> RaggedGather::getOutputPortName(const Tensor& outputTensor) const {
    if (outputTensor == raggedFeatureOutput.getValues()) return "values";
    return std::nullopt;
}

bool RaggedGather::outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const {
    if (outputTensor != raggedFeatureOutput.getValues()) {
        throw std::invalid_argument("Tensor is not an output of this RaggedGather layer.");
    }
    return true;
}

uint64_t RaggedGather::getOutputTensorBytes(uint32_t batchSize) const {
    (void)batchSize;
    // Q is the indices partition and is not copied or newly allocated by this layer.
    return raggedFeatureOutput.getValues().getTotalSizeInBytes();
}

uint64_t RaggedGather::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    return getOutputTensorBytes(batchSize);
}

std::shared_ptr<ThorImplementation::Layer> RaggedGather::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)placement;
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized);
    auto physical = std::make_shared<ThorImplementation::RaggedGather>(
        raggedSourceInput.getDescriptor(), raggedIndicesInput.getDescriptor(), raggedFeatureOutput.getDescriptor(), sharedOffsets);
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedGather::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", getLayerVersion()},
                {"layer_type", "ragged_gather"},
                {"ragged_source", raggedSourceInput.architectureJson()},
                {"ragged_indices", raggedIndicesInput.architectureJson()},
                {"ragged_output", raggedFeatureOutput.architectureJson()}};
}

void RaggedGather::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in RaggedGather::deserialize: " + j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "ragged_gather") {
        throw std::runtime_error("Layer type mismatch in RaggedGather::deserialize: " + j.at("layer_type").get<std::string>());
    }

    RaggedTensor source = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_source"), network, "RaggedGather");
    RaggedTensor indices = SegmentedPrimitiveDetail::reconstructInput(j.at("ragged_indices"), network, "RaggedGather");

    const json& outputJson = j.at("ragged_output");
    SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
        outputJson, j.at("ragged_indices"), indices, "RaggedGather");
    Tensor outputValues = Tensor::deserialize(outputJson.at("values"));
    std::vector<uint64_t> expectedDimensions;
    expectedDimensions.push_back(indices.getMaxTotalValues());
    const std::vector<uint64_t> trailing = source.getTrailingDimensions();
    expectedDimensions.insert(expectedDimensions.end(), trailing.begin(), trailing.end());
    if (outputValues.getDataType() != source.getValuesDataType() || outputValues.getDimensions() != expectedDimensions) {
        throw std::runtime_error("RaggedGather serialized output values do not match source geometry and indices capacity.");
    }
    RaggedTensor output = indices.withValues(outputValues);

    RaggedGather layer = makeLayer(source, indices, output);
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("ragged_gather", &Thor::RaggedGather::deserialize);
    return true;
}();
}  // namespace
