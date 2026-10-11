#include "DeepLearning/Api/Layers/Utility/PaddedDenseToRagged.h"

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
// Compare every durable semantic field after resolving graph references by
// their saved original IDs. Runtime tensor IDs may differ after load.
void checkSemanticTensorReference(const json& saved, const Tensor& live) {
    if (!saved.is_object() || saved.size() != 4 ||
        saved.at("version").get<std::string>() != "1.0.0" ||
        saved.at("id").get<uint64_t>() != live.getOriginalId() ||
        saved.at("data_type").get<DataType>() != live.getDataType() ||
        saved.at("dimensions").get<std::vector<uint64_t>>() != live.getDimensions()) {
        throw std::runtime_error("PaddedDenseToRagged semantic tensor reference mismatch.");
    }
}
void checkSemanticPartitionReference(const json& saved, const RaggedTensor& live) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(saved) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(saved, "PaddedDenseToRagged") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        saved.at("ragged_rank").get<uint32_t>() != 1 ||
        saved.at("max_values_per_row").get<uint64_t>() != live.getMaxValuesPerRow() ||
        saved.contains("batch_size") || saved.contains("max_total_values")) {
        throw std::runtime_error("PaddedDenseToRagged semantic partition reference mismatch.");
    }
    checkSemanticTensorReference(saved.at("values"), live.getValues());
    checkSemanticTensorReference(saved.at("offsets"), live.getRowPartitionToken());
}
}  // namespace

PaddedDenseToRagged PaddedDenseToRagged::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value() || !_partitionInput.has_value()) {
        throw std::runtime_error("PaddedDenseToRagged requires network, featureInput, and partitionInput.");
    }
    if (!_partitionInput->hasMaxValuesPerRow()) {
        throw std::invalid_argument("PaddedDenseToRagged requires partitionInput.max_values_per_row.");
    }
    if (_partitionInput->getRaggedRank() != 1 || _partitionInput->getMaxValuesPerRow() == 0) {
        throw std::invalid_argument("PaddedDenseToRagged requires a positive rank-1 semantic row bound.");
    }
    const std::vector<uint64_t> denseDimensions = _featureInput->getDimensions();
    if (denseDimensions.empty()) {
        throw std::invalid_argument("PaddedDenseToRagged feature input must have logical shape [padded_width, ...].");
    }
    if (denseDimensions.front() < _partitionInput->getMaxValuesPerRow()) {
        throw std::invalid_argument("PaddedDenseToRagged padded width must be at least partitionInput.max_values_per_row.");
    }

    std::vector<uint64_t> outputValuesDimensions;
    outputValuesDimensions.reserve(denseDimensions.size());
    outputValuesDimensions.push_back(_partitionInput->getMaxValuesPerRow());
    outputValuesDimensions.insert(outputValuesDimensions.end(), denseDimensions.begin() + 1, denseDimensions.end());

    PaddedDenseToRagged layer;
    layer.denseFeatureInput = _featureInput.value();
    layer.partitionInput = _partitionInput.value();
    layer.raggedFeatureOutput = _partitionInput->withValues(Tensor(_featureInput->getDataType(), outputValuesDimensions));
    layer.featureInputs = {_featureInput.value(), _partitionInput->getOffsets()};
    layer.featureOutputs = {layer.raggedFeatureOutput.getValues()};
    layer.initialized = true;
    layer.addToNetwork(_network.value());
    return layer;
}

std::vector<Tensor> PaddedDenseToRagged::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs.front()};
}

void PaddedDenseToRagged::informThatInputConnectionMade(Tensor inputTensor) {
    connectedInputPortIndices.insert(static_cast<uint32_t>(getConnectionType(inputTensor)));
}

void PaddedDenseToRagged::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
}

int PaddedDenseToRagged::getConnectionType(Tensor connectingTensor) const {
    if (connectingTensor == denseFeatureInput) return 0;
    if (connectingTensor == partitionInput.getOffsets()) return 1;
    if (connectingTensor == raggedFeatureOutput.getValues()) return 0;
    throw std::runtime_error("Tensor is not connected to this PaddedDenseToRagged layer.");
}

std::optional<std::string> PaddedDenseToRagged::getInputPortName(const Tensor& inputTensor) const {
    if (inputTensor == denseFeatureInput) return "padded_values";
    if (inputTensor == partitionInput.getOffsets()) return "offsets";
    return std::nullopt;
}

std::optional<std::string> PaddedDenseToRagged::getOutputPortName(const Tensor& outputTensor) const {
    if (outputTensor == raggedFeatureOutput.getValues()) return "values";
    return std::nullopt;
}

bool PaddedDenseToRagged::outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const {
    if (outputTensor != raggedFeatureOutput.getValues()) {
        throw std::invalid_argument("Tensor is not an output of this PaddedDenseToRagged layer.");
    }
    return true;
}

uint64_t PaddedDenseToRagged::getOutputTensorBytes(uint32_t batchSize) const {
    if (batchSize == 0) throw std::invalid_argument("PaddedDenseToRagged requires positive placement B.");
    // [M, ...F] is logical, whereas [B*M, ...F] is allocated at placement.
    const uint64_t perRow = raggedFeatureOutput.getValues().getTotalSizeInBytes();
    if (perRow > std::numeric_limits<uint64_t>::max() / batchSize)
        throw std::overflow_error("PaddedDenseToRagged output memory overflows uint64_t.");
    return perRow * batchSize;
}

uint64_t PaddedDenseToRagged::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    const uint64_t output = getOutputTensorBytes(batchSize);
    // The partition's [B+1] offsets are already owned by the producer; this
    // layer allocates one new packed-values output, not copied partition state.
    return output;
}

std::shared_ptr<ThorImplementation::Layer> PaddedDenseToRagged::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)placement;
    (void)drivingApiLayer;
    const int port = getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized);

    const uint32_t b = SegmentedPrimitivePlacementDetail::placementBatchSizeForStamp(
        denseFeatureInput, partitionInput, drivingLayer, connectingApiTensor, "PaddedDenseToRagged");
    const auto partitionDescriptor = partitionInput.getDescriptorForPlacement(b);
    const auto outputDescriptor = raggedFeatureOutput.getDescriptorForPlacement(b);

    std::vector<uint64_t> physicalDenseDimensions;
    physicalDenseDimensions.reserve(denseFeatureInput.getDimensions().size() + 1);
    physicalDenseDimensions.push_back(b);
    const std::vector<uint64_t> denseDimensions = denseFeatureInput.getDimensions();
    physicalDenseDimensions.insert(physicalDenseDimensions.end(), denseDimensions.begin(), denseDimensions.end());
    const ThorImplementation::TensorDescriptor physicalDense(
        denseFeatureInput.getDataType(), physicalDenseDimensions);

    const auto driver = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(
        drivingLayer, "PaddedDenseToRagged");
    const auto expected = port == 0 ? physicalDense : partitionDescriptor.getOffsetsDescriptor();
    if (driver != expected) {
        throw std::runtime_error("PaddedDenseToRagged physical driving port disagrees with placement.");
    }
    const auto managed = SegmentedPrimitivePlacementDetail::rowPartitionDescriptorForStamp(drivingLayer);
    if (managed.has_value() && managed.value() != partitionDescriptor.getRowPartition()) {
        throw std::runtime_error("PaddedDenseToRagged managed partition metadata disagrees with canonical partition.");
    }
    auto physical = std::make_shared<ThorImplementation::PaddedDenseToRagged>(
        physicalDense, partitionDescriptor, outputDescriptor);
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json PaddedDenseToRagged::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", "1.1.0"},
                {"layer_type", "padded_dense_to_ragged"},
                {"feature_input", denseFeatureInput.architectureJson()},
                {"partition_input", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(partitionInput)},
                {"ragged_feature_output", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureOutput)}};
}

void PaddedDenseToRagged::deserialize(const json& j, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported version in PaddedDenseToRagged::deserialize: " + version);
    }
    if (j.at("layer_type").get<std::string>() != "padded_dense_to_ragged") {
        throw std::runtime_error("Layer type mismatch in PaddedDenseToRagged::deserialize.");
    }
    if (version == "1.1.0" && (!j.is_object() || j.size() != 6 ||
                               j.at("factory") != Layer::Factory::Layer.value())) {
        throw std::runtime_error("PaddedDenseToRagged neutral layer schema has unexpected fields.");
    }
    const json& inputJson = j.at("feature_input");
    Tensor denseInput = network->getApiTensorByOriginalId(inputJson.at("id").get<uint64_t>());
    const json& partitionJson = j.at("partition_input");
    RaggedTensor partition = SegmentedPrimitiveDetail::reconstructInput(partitionJson, network, "PaddedDenseToRagged");
    // Resolve the historical and current grammars through the same logical
    // graph contract. Schema validation follows the reference grammar, not
    // the containing layer's historical version number.
    const bool semanticReference =
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(partitionJson, "PaddedDenseToRagged") ==
        SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL;
    if (semanticReference) {
        checkSemanticTensorReference(inputJson, denseInput);
        checkSemanticPartitionReference(partitionJson, partition);
    }
    if (!partition.hasMaxValuesPerRow() || partition.getMaxValuesPerRow() == 0 || partition.getRaggedRank() != 1) {
        throw std::runtime_error("PaddedDenseToRagged requires a positive rank-1 row bound.");
    }
    const auto denseDimensions = denseInput.getDimensions();
    if (denseDimensions.empty() || denseDimensions.front() < partition.getMaxValuesPerRow()) {
        throw std::runtime_error("PaddedDenseToRagged dense input is narrower than partition.max_values_per_row.");
    }

    const json& outputJson = j.at("ragged_feature_output");
    SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
        outputJson, partitionJson, partition, "PaddedDenseToRagged");
    // Historical outputs recorded B*M. Only the producer's graph partition
    // supplies semantic M; never divide or trust an archived physical extent.
    Tensor outputValues = SegmentedPrimitiveDetail::reconstructRaggedOutputValues(
        outputJson, partition, "PaddedDenseToRagged");
    if (SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(outputJson, "PaddedDenseToRagged") ==
        SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL) {
        if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(outputJson))
            throw std::runtime_error("PaddedDenseToRagged output reference has unexpected fields.");
        checkSemanticTensorReference(outputJson.at("values"), outputValues);
        checkSemanticTensorReference(outputJson.at("offsets"), partition.getRowPartitionToken());
        if (outputJson.at("ragged_rank").get<uint32_t>() != 1 ||
            outputJson.at("max_values_per_row").get<uint64_t>() != partition.getMaxValuesPerRow()) {
            throw std::runtime_error("PaddedDenseToRagged output must preserve the canonical partition.");
        }
    }
    if (outputValues.getOriginalId() == denseInput.getOriginalId() ||
        outputValues.getOriginalId() == partition.getValues().getOriginalId() ||
        outputValues.getOriginalId() == partition.getOffsets().getOriginalId()) {
        throw std::runtime_error("PaddedDenseToRagged output must own a distinct values tensor.");
    }
    std::vector<uint64_t> expectedDimensions;
    expectedDimensions.push_back(partition.getMaxValuesPerRow());
    expectedDimensions.insert(expectedDimensions.end(), denseDimensions.begin() + 1, denseDimensions.end());
    if (outputValues.getDataType() != denseInput.getDataType() ||
        outputValues.getDimensions() != expectedDimensions) {
        throw std::runtime_error("PaddedDenseToRagged output values disagree with dense input geometry and partition capacity.");
    }
    RaggedTensor output = partition.withValues(outputValues);

    PaddedDenseToRagged layer;
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
    Thor::Layer::register_layer("padded_dense_to_ragged", &Thor::PaddedDenseToRagged::deserialize);
    return true;
}();
}  // namespace
