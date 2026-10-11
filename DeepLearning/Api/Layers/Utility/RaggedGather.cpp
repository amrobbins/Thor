#include "DeepLearning/Api/Layers/Utility/RaggedGather.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cstdint>
#include <string>
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

// Canonical layer references must match the original graph IDs and all semantic
// fields, not merely resolve to some live tensor after deserialization.
void checkCanonicalTensorRef(const json& saved, const Tensor& live) {
    if (!saved.is_object() || saved.size() != 4 ||
        saved.at("version").get<std::string>() != "1.0.0" ||
        saved.at("id").get<uint64_t>() != live.getOriginalId() ||
        saved.at("data_type").get<DataType>() != live.getDataType() ||
        saved.at("dimensions").get<std::vector<uint64_t>>() != live.getDimensions()) {
        throw std::runtime_error("RaggedGather semantic tensor reference mismatch.");
    }
}

void checkCanonicalRaggedRef(const json& saved, const RaggedTensor& live) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(saved) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(saved, "RaggedGather") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        saved.at("ragged_rank").get<uint32_t>() != 1 ||
        saved.at("max_values_per_row").get<uint64_t>() != live.getMaxValuesPerRow() ||
        saved.contains("batch_size") || saved.contains("max_total_values")) {
        throw std::runtime_error("RaggedGather canonical ragged reference mismatch.");
    }
    checkCanonicalTensorRef(saved.at("values"), live.getValues());
    checkCanonicalTensorRef(saved.at("offsets"), live.getRowPartitionToken());
}

}  // namespace

RaggedGather RaggedGather::makeLayer(const RaggedTensor& sourceInput,
                                     const RaggedTensor& indicesInput,
                                     const std::optional<RaggedTensor>& serializedOutput) {
    if (!sourceInput.isInitialized() || !indicesInput.isInitialized()) {
        throw std::invalid_argument("RaggedGather inputs must be initialized RaggedTensor objects.");
    }
    if (sourceInput.getRaggedRank() != 1 || indicesInput.getRaggedRank() != 1) {
        throw std::invalid_argument("RaggedGather requires rank-1 source and indices partitions.");
    }
    if (!sourceInput.hasMaxValuesPerRow() || !indicesInput.hasMaxValuesPerRow() ||
        sourceInput.getMaxValuesPerRow() == 0 || indicesInput.getMaxValuesPerRow() == 0 ||
        sourceInput.getOffsetsDimensions() != std::vector<uint64_t>{1} ||
        indicesInput.getOffsetsDimensions() != std::vector<uint64_t>{1} ||
        sourceInput.getValuesDimensions().front() != sourceInput.getMaxValuesPerRow() ||
        indicesInput.getValuesDimensions().front() != indicesInput.getMaxValuesPerRow()) {
        throw std::invalid_argument("RaggedGather inputs require logical [M,...] values and [1] partition tokens.");
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
    outputValuesDimensions.push_back(indicesInput.getMaxValuesPerRow());
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
        output.getValuesDimensions() != outputValuesDimensions ||
        !output.sharesPartitionWith(indicesInput) ||
        output.getRowPartitionToken() != indicesInput.getRowPartitionToken() ||
        output.getValues() == sourceInput.getValues() ||
        output.getValues() == indicesInput.getValues() ||
        output.getValues() == indicesInput.getOffsets() ||
        output.getValues() == sourceInput.getOffsets() ||
        output.hasMaxValuesPerRow() != indicesInput.hasMaxValuesPerRow() ||
        (indicesInput.hasMaxValuesPerRow() && output.getMaxValuesPerRow() != indicesInput.getMaxValuesPerRow())) {
        throw std::runtime_error("RaggedGather serialized output must use source value geometry and preserve indices partition Q exactly.");
    }

    RaggedGather layer;
    layer.raggedSourceInput = sourceInput;
    layer.raggedIndicesInput = indicesInput;
    layer.raggedFeatureOutput = output;
    layer.sharedPartition = sourceInput.sharesPartitionWith(indicesInput);
    layer.indicesOffsetsInputPort = layer.sharedPartition ? 2U : 3U;
    layer.featureInputs = {sourceInput.getValues(), indicesInput.getValues(), sourceInput.getOffsets()};
    if (!layer.sharedPartition) layer.featureInputs.push_back(indicesInput.getOffsets());
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
    if (!sharedPartition && connectingTensor == raggedIndicesInput.getOffsets()) return 3;
    if (connectingTensor == raggedFeatureOutput.getValues()) return 0;
    throw std::runtime_error("Tensor is not connected to this RaggedGather layer.");
}

std::optional<std::string> RaggedGather::getInputPortName(const Tensor& inputTensor) const {
    if (inputTensor == raggedSourceInput.getValues()) return "source_values";
    if (inputTensor == raggedIndicesInput.getValues()) return "indices_values";
    if (inputTensor == raggedSourceInput.getOffsets()) return sharedPartition ? "shared_offsets" : "source_offsets";
    if (!sharedPartition && inputTensor == raggedIndicesInput.getOffsets()) return "indices_offsets";
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
    // Q is the indices partition and is not copied or newly allocated by this layer.
    const uint64_t perExample = raggedFeatureOutput.getValues().getTotalSizeInBytes();
    if (batchSize == 0) throw std::invalid_argument("RaggedGather output requires positive B.");
    if (perExample > std::numeric_limits<uint64_t>::max() / batchSize)
        throw std::overflow_error("RaggedGather output allocation overflows uint64_t.");
    return perExample * batchSize;
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
    (void)drivingApiLayer;
    const int port = getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized);
    const auto sourceDescriptor = raggedSourceInput.getDescriptorForPlacement([&]() -> uint32_t {
              const auto driver = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(drivingLayer, "RaggedGather");
              uint64_t b = 0;
              if (port == 0 || port == 1) {
                  const RaggedTensor& input = port == 0 ? raggedSourceInput : raggedIndicesInput;
                  const auto placed = SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(
                      input, drivingLayer, "RaggedGather");
                  b = placed.getBatchSize();
              } else {
                  const RaggedTensor& partition = port == 2 ? raggedSourceInput : raggedIndicesInput;
                  const auto dims = driver.getDimensions();
                  if (driver.getDataType() != partition.getOffsetsDataType() ||
                      dims.size() != 1 || dims.front() < 2) {
                      throw std::runtime_error("RaggedGather driving offsets must have physical shape [B+1].");
                  }
                  b = dims.front() - 1;
              }
              if (b == 0 || b > std::numeric_limits<uint32_t>::max())
                  throw std::runtime_error("RaggedGather placement B is outside uint32 range.");
              return static_cast<uint32_t>(b);
          }());
    const auto indicesDescriptor = raggedIndicesInput.getDescriptorForPlacement(sourceDescriptor.getBatchSize());
    const auto outputDescriptor = raggedFeatureOutput.getDescriptorForPlacement(sourceDescriptor.getBatchSize());
    {
        const auto driving = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(drivingLayer, "RaggedGather");
        const auto expected = port == 0 ? sourceDescriptor.getValuesDescriptor() :
                              port == 1 ? indicesDescriptor.getValuesDescriptor() :
                              port == 2 ? sourceDescriptor.getOffsetsDescriptor() : indicesDescriptor.getOffsetsDescriptor();
        if (driving != expected)
            throw std::runtime_error("RaggedGather physical driving descriptor disagrees with runtime B and port geometry.");
        const auto managed = SegmentedPrimitivePlacementDetail::rowPartitionDescriptorForStamp(drivingLayer);
        if (managed.has_value()) {
            const auto expectedPartition = (port == 0 || port == 2)
                ? sourceDescriptor.getRowPartition() : indicesDescriptor.getRowPartition();
            if (managed.value() != expectedPartition)
                throw std::runtime_error("RaggedGather managed driver partition metadata disagrees with its logical partition.");
        }
    }
    auto physical = std::make_shared<ThorImplementation::RaggedGather>(
        sourceDescriptor, indicesDescriptor, outputDescriptor, sharedPartition);
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedGather::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", "1.1.0"},
                {"layer_type", "ragged_gather"},
                {"ragged_source", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedSourceInput)},
                {"ragged_indices", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedIndicesInput)},
                {"ragged_output", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureOutput)}};
}

void RaggedGather::deserialize(const json& j, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported version in RaggedGather::deserialize: " + version);
    }
    if (j.at("layer_type").get<std::string>() != "ragged_gather") {
        throw std::runtime_error("Layer type mismatch in RaggedGather::deserialize.");
    }
    if (version == "1.1.0" && (!j.is_object() || j.size() != 6 ||
                               j.at("factory") != Layer::Factory::Layer.value())) {
        throw std::runtime_error("RaggedGather canonical layer schema has unexpected fields.");
    }

    const json& sourceJson = j.at("ragged_source");
    const json& indicesJson = j.at("ragged_indices");
    RaggedTensor source = SegmentedPrimitiveDetail::reconstructInput(sourceJson, network, "RaggedGather");
    RaggedTensor indices = SegmentedPrimitiveDetail::reconstructInput(indicesJson, network, "RaggedGather");
    if (version == "1.1.0") {
        checkCanonicalRaggedRef(sourceJson, source);
        checkCanonicalRaggedRef(indicesJson, indices);
    }

    const json& outputJson = j.at("ragged_output");
    SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
        outputJson, indicesJson, indices, "RaggedGather");
    Tensor outputValues = SegmentedPrimitiveDetail::reconstructRaggedOutputValues(
        outputJson, indices, "RaggedGather");
    if (version == "1.1.0") {
        if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(outputJson) ||
            outputJson.at("ragged_rank").get<uint32_t>() != 1 ||
            outputJson.at("max_values_per_row").get<uint64_t>() != indices.getMaxValuesPerRow()) {
            throw std::runtime_error("RaggedGather output partition reference is malformed.");
        }
        checkCanonicalTensorRef(outputJson.at("values"), outputValues);
        checkCanonicalTensorRef(outputJson.at("offsets"), indices.getRowPartitionToken());
    }
    // Output identity is semantic in both archive grammars. A deserialized
    // Tensor receives a new runtime ID even if its *original* ID aliases an
    // input; the ordinary Tensor equality checks would not catch that.
    if (outputValues.getOriginalId() == source.getValues().getOriginalId() ||
        outputValues.getOriginalId() == indices.getValues().getOriginalId() ||
        outputValues.getOriginalId() == source.getOffsets().getOriginalId() ||
        outputValues.getOriginalId() == indices.getOffsets().getOriginalId()) {
        throw std::runtime_error("RaggedGather output values must have a distinct graph identity.");
    }
    std::vector<uint64_t> expectedDimensions;
    expectedDimensions.push_back(indices.getMaxValuesPerRow());
    const auto trailing = source.getTrailingDimensions();
    expectedDimensions.insert(expectedDimensions.end(), trailing.begin(), trailing.end());
    if (outputValues.getDataType() != source.getValuesDataType() ||
        outputValues.getDimensions() != expectedDimensions) {
        throw std::runtime_error("RaggedGather output values disagree with source geometry and Q capacity.");
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
