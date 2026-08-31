#include "DeepLearning/Api/Layers/Utility/RaggedSequenceSlice.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace Thor {
namespace {

RaggedTensor reconstructInput(const json& raggedJson, Network* network) {
    Tensor values = network->getApiTensorByOriginalId(raggedJson.at("values").at("id").get<uint64_t>());
    Tensor offsets = network->getApiTensorByOriginalId(raggedJson.at("offsets").at("id").get<uint64_t>());
    RaggedTensor input = raggedJson.contains("max_values_per_row")
        ? RaggedTensor(values, offsets, raggedJson.at("max_values_per_row").get<uint64_t>())
        : RaggedTensor(values, offsets);
    if (raggedJson.at("ragged_rank").get<uint32_t>() != 1 ||
        input.getBatchSize() != raggedJson.at("batch_size").get<uint64_t>() ||
        input.getMaxTotalValues() != raggedJson.at("max_total_values").get<uint64_t>()) {
        throw std::runtime_error("RaggedSequenceSlice serialized ragged input metadata does not match reconstructed tensors.");
    }
    return input;
}

std::pair<uint64_t, uint64_t> deriveOutputBounds(const RaggedTensor& input, uint64_t start, uint64_t length) {
    const uint64_t batchSize = input.getBatchSize();
    if (batchSize == 0) throw std::invalid_argument("RaggedSequenceSlice requires a non-empty logical batch descriptor.");
    if (length == 0) throw std::invalid_argument("RaggedSequenceSlice length must be greater than zero.");

    // Even when no explicit per-row bound is attached, one row can never contain
    // more than the full packed capacity. This gives sequence slice a finite
    // placement-time output bound without inspecting runtime offsets.
    const uint64_t sourceRowUpperBound =
        input.hasMaxValuesPerRow() ? input.getMaxValuesPerRow() : input.getMaxTotalValues();
    const uint64_t remainingUpperBound = start >= sourceRowUpperBound ? 0 : sourceRowUpperBound - start;
    const uint64_t exactRowUpperBound = std::min<uint64_t>(length, remainingUpperBound);

    // RaggedTensor uses zero as "max_values_per_row unspecified", so an
    // always-empty result is represented by the conservative positive bound 1.
    const uint64_t outputMaxValuesPerRow = std::max<uint64_t>(exactRowUpperBound, 1);

    uint64_t aggregateRowCapacity = input.getMaxTotalValues();
    if (exactRowUpperBound == 0) {
        aggregateRowCapacity = 1;
    } else if (batchSize <= input.getMaxTotalValues() / exactRowUpperBound) {
        aggregateRowCapacity = batchSize * exactRowUpperBound;
    }
    const uint64_t outputMaxTotalValues =
        exactRowUpperBound == 0 ? 1 : std::min<uint64_t>(input.getMaxTotalValues(), aggregateRowCapacity);
    THOR_THROW_IF_FALSE(outputMaxValuesPerRow <= outputMaxTotalValues);
    return {outputMaxTotalValues, outputMaxValuesPerRow};
}

}  // namespace

RaggedSequenceSlice RaggedSequenceSlice::makeLayer(const RaggedTensor& input,
                                                   uint64_t start,
                                                   uint64_t length,
                                                   const std::optional<RaggedTensor>& serializedOutput) {
    if (!input.isInitialized()) throw std::invalid_argument("RaggedSequenceSlice input must be initialized.");
    if (input.getBatchSize() > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("RaggedSequenceSlice batch size exceeds Thor's uint32 placement capacity.");
    }

    const auto [outputMaxTotalValues, outputMaxValuesPerRow] = deriveOutputBounds(input, start, length);
    if (!ThorImplementation::canonicalRowPartitionOffsetCanRepresent(input.getOffsetsDataType(), outputMaxTotalValues)) {
        throw std::invalid_argument(
            "RaggedSequenceSlice output max_total_values cannot be represented by the selected offsets dtype.");
    }

    RaggedTensor output = serializedOutput.has_value()
        ? serializedOutput.value()
        : RaggedTensor(input.getValuesDataType(),
                       input.getTrailingDimensions(),
                       input.getBatchSize(),
                       outputMaxTotalValues,
                       outputMaxValuesPerRow,
                       input.getOffsetsDataType());

    if (output.getValuesDataType() != input.getValuesDataType() ||
        output.getOffsetsDataType() != input.getOffsetsDataType() || output.getBatchSize() != input.getBatchSize() ||
        output.getMaxTotalValues() != outputMaxTotalValues || output.getTrailingDimensions() != input.getTrailingDimensions() ||
        !output.hasMaxValuesPerRow() || output.getMaxValuesPerRow() != outputMaxValuesPerRow) {
        throw std::runtime_error("RaggedSequenceSlice serialized output descriptor does not match its input and slice window.");
    }
    if (output.getOffsets().getOriginalId() == input.getOffsets().getOriginalId()) {
        throw std::runtime_error("RaggedSequenceSlice must own a newly produced offsets tensor.");
    }

    RaggedSequenceSlice layer;
    layer.start = start;
    layer.length = length;
    layer.raggedFeatureInput = input;
    layer.raggedFeatureOutput = output;
    layer.featureInputs = {input.getValues(), input.getOffsets()};
    layer.featureOutputs = {output.getValues(), output.getOffsets()};
    layer.initialized = true;
    return layer;
}

RaggedSequenceSlice RaggedSequenceSlice::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value() || !_start.has_value() || !_length.has_value()) {
        throw std::runtime_error("RaggedSequenceSlice requires network, featureInput, start, and length.");
    }
    RaggedSequenceSlice layer =
        RaggedSequenceSlice::makeLayer(_featureInput.value(), _start.value(), _length.value());
    layer.addToNetwork(_network.value());
    return layer;
}

std::vector<Tensor> RaggedSequenceSlice::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedOutputsAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedOutputsAfterAllInputsConnected = true;
    return featureOutputs;
}

void RaggedSequenceSlice::informThatInputConnectionMade(Tensor inputTensor) {
    const int connectionType = getConnectionType(inputTensor);
    THOR_THROW_IF_FALSE(connectionType >= 0);
    connectedInputPortIndices.insert(static_cast<uint32_t>(connectionType));
}

void RaggedSequenceSlice::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedOutputsAfterAllInputsConnected = false;
}

int RaggedSequenceSlice::getConnectionType(Tensor connectingTensor) const {
    if (connectingTensor == raggedFeatureInput.getValues()) return 0;
    if (connectingTensor == raggedFeatureInput.getOffsets()) return 1;
    if (connectingTensor == raggedFeatureOutput.getValues()) return 0;
    if (connectingTensor == raggedFeatureOutput.getOffsets()) return 1;
    throw std::runtime_error("Tensor is not connected to this RaggedSequenceSlice layer.");
}

std::optional<std::string> RaggedSequenceSlice::getInputPortName(const Tensor& inputTensor) const {
    if (inputTensor == raggedFeatureInput.getValues()) return "values";
    if (inputTensor == raggedFeatureInput.getOffsets()) return "offsets";
    return std::nullopt;
}

std::optional<std::string> RaggedSequenceSlice::getOutputPortName(const Tensor& outputTensor) const {
    if (outputTensor == raggedFeatureOutput.getValues()) return "values";
    if (outputTensor == raggedFeatureOutput.getOffsets()) return "offsets";
    return std::nullopt;
}

bool RaggedSequenceSlice::outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const {
    if (outputTensor != raggedFeatureOutput.getValues() && outputTensor != raggedFeatureOutput.getOffsets()) {
        throw std::invalid_argument("Tensor is not an output of this RaggedSequenceSlice layer.");
    }
    return true;
}

uint64_t RaggedSequenceSlice::getOutputTensorBytes(uint32_t batchSize) const {
    (void)batchSize;
    return raggedFeatureOutput.getValues().getTotalSizeInBytes() + raggedFeatureOutput.getOffsets().getTotalSizeInBytes();
}

uint64_t RaggedSequenceSlice::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    const uint64_t outputBytes = getOutputTensorBytes(batchSize);
    const uint64_t rowLengthsBytes = raggedFeatureInput.getBatchSize() *
        ThorImplementation::TensorDescriptor::getElementSizeInBytes(raggedFeatureInput.getDescriptor().getOffsetsDataType());
    if (outputBytes > std::numeric_limits<uint64_t>::max() - rowLengthsBytes) {
        throw std::overflow_error("RaggedSequenceSlice memory requirement overflow.");
    }
    // CUB scan workspace is execution-local and allocated by the stamped layer.
    // The public scheduler accounting includes the deterministic row-length
    // tensor here; CUB's implementation-specific scratch is intentionally not
    // encoded into the serialized API contract.
    return outputBytes + rowLengthsBytes;
}

std::shared_ptr<ThorImplementation::Layer> RaggedSequenceSlice::stamp(
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
    auto physical = std::make_shared<ThorImplementation::RaggedSequenceSlice>(
        start, length, raggedFeatureInput.getDescriptor(), raggedFeatureOutput.getDescriptor());
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedSequenceSlice::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", getLayerVersion()},
                {"layer_type", "ragged_sequence_slice"},
                {"start", start},
                {"length", length},
                {"ragged_input", raggedFeatureInput.architectureJson()},
                {"ragged_output", raggedFeatureOutput.architectureJson()}};
}

void RaggedSequenceSlice::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in RaggedSequenceSlice::deserialize: " +
                                 j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "ragged_sequence_slice") {
        throw std::runtime_error("Layer type mismatch in RaggedSequenceSlice::deserialize: " +
                                 j.at("layer_type").get<std::string>());
    }

    RaggedTensor input = reconstructInput(j.at("ragged_input"), network);
    const uint64_t start = j.at("start").get<uint64_t>();
    const uint64_t length = j.at("length").get<uint64_t>();

    const json& outputJson = j.at("ragged_output");
    Tensor outputValues = Tensor::deserialize(outputJson.at("values"));
    Tensor outputOffsets = Tensor::deserialize(outputJson.at("offsets"));
    RaggedTensor output = outputJson.contains("max_values_per_row")
        ? RaggedTensor(outputValues, outputOffsets, outputJson.at("max_values_per_row").get<uint64_t>())
        : RaggedTensor(outputValues, outputOffsets);
    if (outputJson.at("ragged_rank").get<uint32_t>() != 1 ||
        output.getBatchSize() != outputJson.at("batch_size").get<uint64_t>() ||
        output.getMaxTotalValues() != outputJson.at("max_total_values").get<uint64_t>()) {
        throw std::runtime_error("RaggedSequenceSlice serialized ragged output metadata does not match reconstructed tensors.");
    }

    RaggedSequenceSlice layer = makeLayer(input, start, length, output);
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("ragged_sequence_slice", &Thor::RaggedSequenceSlice::deserialize);
    return true;
}();
}  // namespace
