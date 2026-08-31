#include "DeepLearning/Api/Layers/Utility/PaddedDenseToRagged.h"

#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace Thor {

PaddedDenseToRagged PaddedDenseToRagged::Builder::build() {
    if (!_network.has_value() || !_featureInput.has_value() || !_partitionInput.has_value()) {
        throw std::runtime_error("PaddedDenseToRagged requires network, featureInput, and partitionInput.");
    }
    if (!_partitionInput->hasMaxValuesPerRow()) {
        throw std::invalid_argument("PaddedDenseToRagged requires partitionInput.max_values_per_row.");
    }
    if (_partitionInput->getBatchSize() == 0 || _partitionInput->getBatchSize() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("PaddedDenseToRagged requires a non-zero partition batch size within uint32 placement capacity.");
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
    outputValuesDimensions.push_back(_partitionInput->getMaxTotalValues());
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
    (void)batchSize;
    return raggedFeatureOutput.getValues().getTotalSizeInBytes();
}

uint64_t PaddedDenseToRagged::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    return getOutputTensorBytes(batchSize) + sizeof(uint32_t);
}

std::shared_ptr<ThorImplementation::Layer> PaddedDenseToRagged::stamp(
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

    std::vector<uint64_t> physicalDenseDimensions;
    physicalDenseDimensions.reserve(denseFeatureInput.getDimensions().size() + 1);
    physicalDenseDimensions.push_back(partitionInput.getBatchSize());
    const std::vector<uint64_t> denseDimensions = denseFeatureInput.getDimensions();
    physicalDenseDimensions.insert(physicalDenseDimensions.end(), denseDimensions.begin(), denseDimensions.end());

    auto physical = std::make_shared<ThorImplementation::PaddedDenseToRagged>(
        ThorImplementation::TensorDescriptor(denseFeatureInput.getDataType(), physicalDenseDimensions),
        partitionInput.getDescriptor(),
        raggedFeatureOutput.getDescriptor());
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json PaddedDenseToRagged::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", getLayerVersion()},
                {"layer_type", "padded_dense_to_ragged"},
                {"feature_input", denseFeatureInput.architectureJson()},
                {"partition_input", partitionInput.architectureJson()},
                {"ragged_feature_output", raggedFeatureOutput.architectureJson()}};
}

void PaddedDenseToRagged::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in PaddedDenseToRagged::deserialize: " + j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "padded_dense_to_ragged") {
        throw std::runtime_error("Layer type mismatch in PaddedDenseToRagged::deserialize: " + j.at("layer_type").get<std::string>());
    }

    Tensor denseInput = network->getApiTensorByOriginalId(j.at("feature_input").at("id").get<uint64_t>());
    RaggedTensor partition = SegmentedPrimitiveDetail::reconstructInput(j.at("partition_input"), network, "PaddedDenseToRagged");
    if (!partition.hasMaxValuesPerRow()) {
        throw std::runtime_error("PaddedDenseToRagged serialized partition must carry max_values_per_row.");
    }
    const std::vector<uint64_t> denseDimensions = denseInput.getDimensions();
    if (denseDimensions.empty() || denseDimensions.front() < partition.getMaxValuesPerRow()) {
        throw std::runtime_error("PaddedDenseToRagged serialized dense input is narrower than partition.max_values_per_row.");
    }

    const json& outputJson = j.at("ragged_feature_output");
    SegmentedPrimitiveDetail::validateSerializedPreservedPartition(
        outputJson, j.at("partition_input"), partition, "PaddedDenseToRagged");
    Tensor outputValues = Tensor::deserialize(outputJson.at("values"));
    std::vector<uint64_t> expectedOutputDimensions;
    expectedOutputDimensions.push_back(partition.getMaxTotalValues());
    expectedOutputDimensions.insert(expectedOutputDimensions.end(), denseDimensions.begin() + 1, denseDimensions.end());
    if (outputValues.getDataType() != denseInput.getDataType() || outputValues.getDimensions() != expectedOutputDimensions) {
        throw std::runtime_error("PaddedDenseToRagged serialized output values do not match dense input geometry and partition capacity.");
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
