#include "DeepLearning/Api/Layers/Utility/RaggedSequenceSlice.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"

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

// The original graph ID is authoritative. A matching ID alone does not
// establish that the saved nested dtype/dimensions or logical geometry are valid.
void validateCanonicalTensor(const json& saved, const Tensor& live) {
    if (!saved.is_object() || saved.size() != 4 ||
        saved.at("version").get<std::string>() != "1.0.0" ||
        saved.at("id").get<uint64_t>() != live.getOriginalId() ||
        saved.at("data_type").get<DataType>() != live.getDataType() ||
        saved.at("dimensions").get<std::vector<uint64_t>>() != live.getDimensions()) {
        throw std::runtime_error("RaggedSequenceSlice canonical tensor reference mismatch.");
    }
}

void validateCanonicalRagged(const json& saved, const RaggedTensor& live) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(saved) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(saved, "RaggedSequenceSlice") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        saved.at("ragged_rank").get<uint32_t>() != 1 ||
        saved.at("max_values_per_row").get<uint64_t>() != live.getMaxValuesPerRow() ||
        saved.contains("batch_size") || saved.contains("max_total_values")) {
        throw std::runtime_error("RaggedSequenceSlice canonical ragged reference mismatch.");
    }
    validateCanonicalTensor(saved.at("values"), live.getValues());
    validateCanonicalTensor(saved.at("offsets"), live.getRowPartitionToken());
}

uint64_t deriveOutputM(const RaggedTensor& input, uint64_t start, uint64_t length) {
    if (length == 0) throw std::invalid_argument("RaggedSequenceSlice length must be greater than zero.");
    if (!input.hasMaxValuesPerRow() || input.getMaxValuesPerRow() == 0)
        throw std::invalid_argument("RaggedSequenceSlice requires positive semantic M.");
    const uint64_t m = input.getMaxValuesPerRow();
    const uint64_t remaining = start >= m ? 0 : m - start;
    return std::max<uint64_t>(1, std::min(length, remaining));
}
}  // namespace

RaggedSequenceSlice RaggedSequenceSlice::makeLayer(const RaggedTensor& input,
                                                   uint64_t start,
                                                   uint64_t length,
                                                   const std::optional<RaggedTensor>& serializedOutput) {
    if (!input.isInitialized()) throw std::invalid_argument("RaggedSequenceSlice input must be initialized.");
    if (input.getOffsetsDimensions() != std::vector<uint64_t>{1} ||
        input.getValuesDimensions().front() != input.getMaxValuesPerRow())
        throw std::invalid_argument("RaggedSequenceSlice requires logical [M,...] values and a [1] partition token.");
    const uint64_t outputMaxValuesPerRow = deriveOutputM(input, start, length);
    std::vector<uint64_t> outputShape{outputMaxValuesPerRow};
    const auto trailing = input.getTrailingDimensions();
    outputShape.insert(outputShape.end(), trailing.begin(), trailing.end());
    RaggedTensor output = serializedOutput.has_value() ? serializedOutput.value() :
        RaggedTensor::logical(Tensor(input.getValuesDataType(), outputShape),
                                       Tensor(input.getOffsetsDataType(), {1}), outputMaxValuesPerRow);

    if (output.getValuesDataType() != input.getValuesDataType() ||
        output.getOffsetsDataType() != input.getOffsetsDataType() ||
        output.getTrailingDimensions() != trailing ||
        !output.hasMaxValuesPerRow() || output.getMaxValuesPerRow() != outputMaxValuesPerRow ||
        output.getValuesDimensions() != outputShape ||
        output.getOffsetsDimensions() != std::vector<uint64_t>{1}) {
        throw std::runtime_error("RaggedSequenceSlice output descriptor disagrees with slice semantics.");
    }
    if (output.sharesPartitionWith(input) ||
        output.getValues() == input.getValues() || output.getValues() == input.getOffsets() ||
        output.getOffsets() == input.getValues() || output.getOffsets() == input.getOffsets()) {
        throw std::runtime_error("RaggedSequenceSlice must create distinct values and a new row partition.");
    }
    if (output.getValues().getOriginalId() == input.getValues().getOriginalId() ||
                    output.getValues().getOriginalId() == input.getOffsets().getOriginalId() ||
                    output.getOffsets().getOriginalId() == input.getValues().getOriginalId() ||
                    output.getOffsets().getOriginalId() == input.getOffsets().getOriginalId() ||
                    output.getValues().getOriginalId() == output.getOffsets().getOriginalId()) {
        throw std::runtime_error("RaggedSequenceSlice logical output identities must be distinct.");
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
    const uint64_t logicalBytes = raggedFeatureOutput.getValues().getTotalSizeInBytes();
    if (batchSize == 0 || logicalBytes > std::numeric_limits<uint64_t>::max() / batchSize)
        throw std::overflow_error("RaggedSequenceSlice output size overflows uint64_t.");
    return logicalBytes * batchSize;
}

uint64_t RaggedSequenceSlice::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    const uint64_t valuesBytes = getOutputTensorBytes(batchSize);
    // The new partition has its own real [B+1] offsets. Input buffers are
    // already owned by their producers and must not be double-counted here.
    const auto offsets = raggedFeatureOutput.getDescriptorForPlacement(batchSize).getOffsetsDescriptor();
    const uint64_t offsetsBytes = offsets.getArraySizeInBytes();
    if (valuesBytes > std::numeric_limits<uint64_t>::max() - offsetsBytes)
        throw std::overflow_error("RaggedSequenceSlice memory count overflows uint64_t.");
    return valuesBytes + offsetsBytes;
}

std::shared_ptr<ThorImplementation::Layer> RaggedSequenceSlice::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)placement;
    (void)drivingApiLayer;
    const int port = getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized);
    ThorImplementation::RaggedTensorDescriptor inputDescriptor;
    if (port == 0) {
        inputDescriptor = SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
            raggedFeatureInput, drivingLayer, "RaggedSequenceSlice");
    } else {
        const auto driver = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(
            drivingLayer, "RaggedSequenceSlice");
        const auto dims = driver.getDimensions();
        if (driver.getDataType() != raggedFeatureInput.getOffsetsDataType() ||
            dims.size() != 1 || dims.front() < 2) {
            throw std::runtime_error("RaggedSequenceSlice needs real [B+1] offsets to drive placement.");
        }
        inputDescriptor = raggedFeatureInput.getDescriptorForPlacement(dims.front() - 1);
        if (driver != inputDescriptor.getOffsetsDescriptor()) {
            throw std::runtime_error("RaggedSequenceSlice physical offsets disagree with placement.");
        }
        const auto managed = SegmentedPrimitivePlacementDetail::rowPartitionDescriptorForStamp(drivingLayer);
        if (managed.has_value() && managed.value() != inputDescriptor.getRowPartition()) {
            throw std::runtime_error("RaggedSequenceSlice managed partition metadata mismatch.");
        }
    }
    const uint64_t b = inputDescriptor.getBatchSize();
    if (b == 0 || b > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("RaggedSequenceSlice runtime B outside uint32 range.");
    const auto driver = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(
        drivingLayer, "RaggedSequenceSlice");
    const auto expected = port == 0 ? inputDescriptor.getValuesDescriptor() : inputDescriptor.getOffsetsDescriptor();
    if (driver != expected)
        throw std::runtime_error("RaggedSequenceSlice driver descriptor disagrees with placement.");
    const auto managed = SegmentedPrimitivePlacementDetail::rowPartitionDescriptorForStamp(drivingLayer);
    if (managed.has_value() && managed.value() != inputDescriptor.getRowPartition())
        throw std::runtime_error("RaggedSequenceSlice driver partition metadata mismatch.");
    const auto outputDescriptor = raggedFeatureOutput.getDescriptorForPlacement(b);
    if (!ThorImplementation::canonicalRowPartitionOffsetCanRepresent(
            raggedFeatureOutput.getOffsetsDataType(), outputDescriptor.getMaxTotalValues())) {
        throw std::runtime_error("RaggedSequenceSlice output physical capacity exceeds offsets dtype.");
    }
    auto physical = std::make_shared<ThorImplementation::RaggedSequenceSlice>(
        start, length, inputDescriptor, outputDescriptor);
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedSequenceSlice::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", "1.1.0"},
                {"layer_type", "ragged_sequence_slice"},
                {"start", start},
                {"length", length},
                {"ragged_input", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureInput)},
                {"ragged_output", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureOutput)}};
}

void RaggedSequenceSlice::deserialize(const json& j, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported version in RaggedSequenceSlice::deserialize: " + version);
    }
    if (j.at("layer_type").get<std::string>() != "ragged_sequence_slice") {
        throw std::runtime_error("Layer type mismatch in RaggedSequenceSlice::deserialize.");
    }
    if (version == "1.1.0" && (!j.is_object() || j.size() != 7 ||
                               j.at("factory") != Layer::Factory::Layer.value())) {
        throw std::runtime_error("RaggedSequenceSlice canonical layer has unexpected or missing fields.");
    }
    const json& inputJson = j.at("ragged_input");
    RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(inputJson, network, "RaggedSequenceSlice");
    if (version == "1.1.0") validateCanonicalRagged(inputJson, input);
    const uint64_t start = j.at("start").get<uint64_t>();
    const uint64_t length = j.at("length").get<uint64_t>();

    const json& outputJson = j.at("ragged_output");
    const uint64_t outputM = deriveOutputM(input, start, length);
    RaggedTensor output = SegmentedPrimitiveDetail::reconstructNewPartitionOutput(
        outputJson, outputM, input.getValuesDataType(), input.getOffsetsDataType(),
        input.getTrailingDimensions(), "RaggedSequenceSlice");
    if (version == "1.1.0") validateCanonicalRagged(outputJson, output);
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
