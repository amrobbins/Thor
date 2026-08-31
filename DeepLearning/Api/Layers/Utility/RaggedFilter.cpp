#include "DeepLearning/Api/Layers/Utility/RaggedFilter.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {
namespace {

RaggedTensor reconstructInput(const json& raggedJson, Network* network, const char* context) {
    Tensor values = network->getApiTensorByOriginalId(raggedJson.at("values").at("id").get<uint64_t>());
    Tensor offsets = network->getApiTensorByOriginalId(raggedJson.at("offsets").at("id").get<uint64_t>());
    RaggedTensor input = raggedJson.contains("max_values_per_row")
        ? RaggedTensor(values, offsets, raggedJson.at("max_values_per_row").get<uint64_t>())
        : RaggedTensor(values, offsets);
    if (raggedJson.at("ragged_rank").get<uint32_t>() != 1 ||
        input.getBatchSize() != raggedJson.at("batch_size").get<uint64_t>() ||
        input.getMaxTotalValues() != raggedJson.at("max_total_values").get<uint64_t>()) {
        throw std::runtime_error(std::string(context) +
                                 " serialized ragged input metadata does not match reconstructed tensors.");
    }
    return input;
}

}  // namespace

RaggedFilter RaggedFilter::makeLayer(const RaggedTensor& featureInput,
                                     const RaggedTensor& maskInput,
                                     const std::optional<RaggedTensor>& serializedOutput) {
    if (!featureInput.isInitialized() || !maskInput.isInitialized()) {
        throw std::invalid_argument("RaggedFilter inputs must be initialized RaggedTensor objects.");
    }
    if (featureInput.getBatchSize() > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("RaggedFilter batch size exceeds Thor's uint32 placement capacity.");
    }
    if (maskInput.getValuesDataType() != DataType::BOOLEAN) {
        throw std::invalid_argument("RaggedFilter mask values must use BOOLEAN dtype.");
    }
    if (!maskInput.getTrailingDimensions().empty()) {
        throw std::invalid_argument("RaggedFilter mask must contain exactly one scalar predicate per ragged token.");
    }
    if (featureInput.getOffsets() != maskInput.getOffsets()) {
        throw std::invalid_argument("RaggedFilter feature and mask inputs must share the exact same offsets tensor.");
    }
    if (featureInput.getDescriptor().getRowPartition() != maskInput.getDescriptor().getRowPartition()) {
        throw std::invalid_argument("RaggedFilter feature and mask inputs must share the same row-partition descriptor.");
    }

    RaggedTensor output = serializedOutput.has_value()
        ? serializedOutput.value()
        : (featureInput.hasMaxValuesPerRow()
               ? RaggedTensor(featureInput.getValuesDataType(),
                              featureInput.getTrailingDimensions(),
                              featureInput.getBatchSize(),
                              featureInput.getMaxTotalValues(),
                              featureInput.getMaxValuesPerRow(),
                              featureInput.getOffsetsDataType())
               : RaggedTensor(featureInput.getValuesDataType(),
                              featureInput.getTrailingDimensions(),
                              featureInput.getBatchSize(),
                              featureInput.getMaxTotalValues(),
                              featureInput.getOffsetsDataType()));

    if (output.getValuesDataType() != featureInput.getValuesDataType() ||
        output.getOffsetsDataType() != featureInput.getOffsetsDataType() ||
        output.getBatchSize() != featureInput.getBatchSize() ||
        output.getMaxTotalValues() != featureInput.getMaxTotalValues() ||
        output.getTrailingDimensions() != featureInput.getTrailingDimensions() ||
        output.hasMaxValuesPerRow() != featureInput.hasMaxValuesPerRow() ||
        (featureInput.hasMaxValuesPerRow() && output.getMaxValuesPerRow() != featureInput.getMaxValuesPerRow())) {
        throw std::runtime_error("RaggedFilter serialized output descriptor does not match its feature input.");
    }
    if (output.getOffsets().getOriginalId() == featureInput.getOffsets().getOriginalId()) {
        throw std::runtime_error("RaggedFilter must own a newly produced offsets tensor.");
    }

    RaggedFilter layer;
    layer.raggedFeatureInput = featureInput;
    layer.raggedMaskInput = maskInput;
    layer.raggedFeatureOutput = output;
    // The feature and mask share one exact structural partition, so expose that
    // offsets tensor only once as a physical graph port.
    layer.featureInputs = {featureInput.getValues(), maskInput.getValues(), featureInput.getOffsets()};
    layer.featureOutputs = {output.getValues(), output.getOffsets()};
    layer.initialized = true;
    return layer;
}

RaggedFilter RaggedFilter::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value() || !_maskInput.has_value()) {
        throw std::runtime_error("RaggedFilter requires network, featureInput, and maskInput.");
    }
    RaggedFilter layer = RaggedFilter::makeLayer(_featureInput.value(), _maskInput.value());
    layer.addToNetwork(_network.value());
    return layer;
}

std::vector<Tensor> RaggedFilter::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedOutputsAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedOutputsAfterAllInputsConnected = true;
    return featureOutputs;
}

void RaggedFilter::informThatInputConnectionMade(Tensor inputTensor) {
    const int connectionType = getConnectionType(inputTensor);
    THOR_THROW_IF_FALSE(connectionType >= 0);
    connectedInputPortIndices.insert(static_cast<uint32_t>(connectionType));
}

void RaggedFilter::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedOutputsAfterAllInputsConnected = false;
}

int RaggedFilter::getConnectionType(Tensor connectingTensor) const {
    if (connectingTensor == raggedFeatureInput.getValues()) return 0;
    if (connectingTensor == raggedMaskInput.getValues()) return 1;
    if (connectingTensor == raggedFeatureInput.getOffsets()) return 2;
    if (connectingTensor == raggedFeatureOutput.getValues()) return 0;
    if (connectingTensor == raggedFeatureOutput.getOffsets()) return 1;
    throw std::runtime_error("Tensor is not connected to this RaggedFilter layer.");
}

std::optional<std::string> RaggedFilter::getInputPortName(const Tensor& inputTensor) const {
    if (inputTensor == raggedFeatureInput.getValues()) return "values";
    if (inputTensor == raggedMaskInput.getValues()) return "mask";
    if (inputTensor == raggedFeatureInput.getOffsets()) return "offsets";
    return std::nullopt;
}

std::optional<std::string> RaggedFilter::getOutputPortName(const Tensor& outputTensor) const {
    if (outputTensor == raggedFeatureOutput.getValues()) return "values";
    if (outputTensor == raggedFeatureOutput.getOffsets()) return "offsets";
    return std::nullopt;
}

bool RaggedFilter::outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const {
    if (outputTensor != raggedFeatureOutput.getValues() && outputTensor != raggedFeatureOutput.getOffsets()) {
        throw std::invalid_argument("Tensor is not an output of this RaggedFilter layer.");
    }
    return true;
}

uint64_t RaggedFilter::getOutputTensorBytes(uint32_t batchSize) const {
    (void)batchSize;
    return raggedFeatureOutput.getValues().getTotalSizeInBytes() + raggedFeatureOutput.getOffsets().getTotalSizeInBytes();
}

uint64_t RaggedFilter::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    const uint64_t outputBytes = getOutputTensorBytes(batchSize);
    const uint64_t rowLengthsBytes = raggedFeatureInput.getBatchSize() *
        ThorImplementation::TensorDescriptor::getElementSizeInBytes(raggedFeatureInput.getOffsetsDataType());
    if (outputBytes > std::numeric_limits<uint64_t>::max() - rowLengthsBytes) {
        throw std::overflow_error("RaggedFilter memory requirement overflow.");
    }
    return outputBytes + rowLengthsBytes;
}

std::shared_ptr<ThorImplementation::Layer> RaggedFilter::stamp(
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
    auto physical = std::make_shared<ThorImplementation::RaggedFilter>(
        raggedFeatureInput.getDescriptor(), raggedMaskInput.getDescriptor(), raggedFeatureOutput.getDescriptor());
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedFilter::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", getLayerVersion()},
                {"layer_type", "ragged_filter"},
                {"ragged_input", raggedFeatureInput.architectureJson()},
                {"ragged_mask", raggedMaskInput.architectureJson()},
                {"ragged_output", raggedFeatureOutput.architectureJson()}};
}

void RaggedFilter::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in RaggedFilter::deserialize: " +
                                 j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "ragged_filter") {
        throw std::runtime_error("Layer type mismatch in RaggedFilter::deserialize: " +
                                 j.at("layer_type").get<std::string>());
    }

    RaggedTensor featureInput = reconstructInput(j.at("ragged_input"), network, "RaggedFilter");
    RaggedTensor maskInput = reconstructInput(j.at("ragged_mask"), network, "RaggedFilter");

    const json& outputJson = j.at("ragged_output");
    Tensor outputValues = Tensor::deserialize(outputJson.at("values"));
    Tensor outputOffsets = Tensor::deserialize(outputJson.at("offsets"));
    RaggedTensor output = outputJson.contains("max_values_per_row")
        ? RaggedTensor(outputValues, outputOffsets, outputJson.at("max_values_per_row").get<uint64_t>())
        : RaggedTensor(outputValues, outputOffsets);
    if (outputJson.at("ragged_rank").get<uint32_t>() != 1 ||
        output.getBatchSize() != outputJson.at("batch_size").get<uint64_t>() ||
        output.getMaxTotalValues() != outputJson.at("max_total_values").get<uint64_t>()) {
        throw std::runtime_error("RaggedFilter serialized ragged output metadata does not match reconstructed tensors.");
    }

    RaggedFilter layer = makeLayer(featureInput, maskInput, output);
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("ragged_filter", &Thor::RaggedFilter::deserialize);
    return true;
}();
}  // namespace
