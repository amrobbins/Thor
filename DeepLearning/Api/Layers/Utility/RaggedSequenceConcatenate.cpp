#include "DeepLearning/Api/Layers/Utility/RaggedSequenceConcatenate.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitiveCommon.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

using json = nlohmann::json;

namespace Thor {
namespace {

void checkCanonicalTensor(const json& saved, const Tensor& live) {
    if (!saved.is_object() || saved.size() != 4 ||
        saved.at("version").get<std::string>() != "1.0.0" ||
        saved.at("id").get<uint64_t>() != live.getOriginalId() ||
        saved.at("data_type").get<DataType>() != live.getDataType() ||
        saved.at("dimensions").get<std::vector<uint64_t>>() != live.getDimensions())
        throw std::runtime_error("RaggedSequenceConcatenate canonical tensor reference mismatch.");
}

void checkCanonicalRagged(const json& saved, const RaggedTensor& live) {
    if (!SegmentedPrimitiveDetail::hasCanonicalLayerRaggedReferenceFields(saved) ||
        SegmentedPrimitiveDetail::layerRaggedReferenceGrammar(saved, "RaggedSequenceConcatenate") !=
            SegmentedPrimitiveDetail::LayerRaggedReferenceGrammar::CANONICAL_LOGICAL ||
        saved.at("ragged_rank").get<uint32_t>() != 1 ||
        saved.at("max_values_per_row").get<uint64_t>() != live.getMaxValuesPerRow() ||
        saved.contains("batch_size") || saved.contains("max_total_values"))
        throw std::runtime_error("RaggedSequenceConcatenate canonical ragged reference mismatch.");
    checkCanonicalTensor(saved.at("values"), live.getValues());
    checkCanonicalTensor(saved.at("offsets"), live.getRowPartitionToken());
}

uint64_t checkedAdd(uint64_t a, uint64_t b, const char* what) {
    if (a > std::numeric_limits<uint64_t>::max() - b) throw std::overflow_error(what);
    return a + b;
}

}  // namespace

RaggedSequenceConcatenate RaggedSequenceConcatenate::makeLayer(
    const std::vector<RaggedTensor>& inputs, const std::optional<RaggedTensor>& serializedOutput) {
    if (inputs.size() < 2) throw std::invalid_argument("RaggedSequenceConcatenate requires at least two inputs.");
    if (inputs.size() > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("RaggedSequenceConcatenate input count exceeds uint32 capacity.");
    }

    const RaggedTensor& reference = inputs.front();
    if (!reference.isInitialized() || reference.getRaggedRank() != 1)
        throw std::invalid_argument("RaggedSequenceConcatenate requires initialized rank-1 inputs.");
    const DataType valuesDataType = reference.getValuesDataType();
    const DataType offsetsDataType = reference.getOffsetsDataType();
    const std::vector<uint64_t> trailingDimensions = reference.getTrailingDimensions();
    uint64_t outputMaxValuesPerRow = 0;
    std::set<Tensor> uniqueValues;
    std::vector<Tensor> uniqueOffsets;
    std::set<RowPartitionId> uniquePartitions;

    for (uint32_t i = 0; i < inputs.size(); ++i) {
        const RaggedTensor& input = inputs[i];
        if (!input.isInitialized()) throw std::invalid_argument("RaggedSequenceConcatenate inputs must be initialized.");
        if (input.getRaggedRank() != 1) {
            throw std::invalid_argument("RaggedSequenceConcatenate inputs must have rank one.");
        }
        if (!input.hasMaxValuesPerRow() || input.getMaxValuesPerRow() == 0 ||
            input.getOffsetsDimensions() != std::vector<uint64_t>{1} ||
            input.getValuesDimensions().front() != input.getMaxValuesPerRow())
            throw std::invalid_argument("RaggedSequenceConcatenate requires logical [M,...] values and [1] partition tokens.");
        if (input.getValuesDataType() != valuesDataType) {
            throw std::invalid_argument("RaggedSequenceConcatenate inputs must have the same values dtype.");
        }
        if (input.getOffsetsDataType() != offsetsDataType) {
            throw std::invalid_argument("RaggedSequenceConcatenate inputs must use the same UINT32/UINT64 offsets dtype.");
        }
        if (input.getTrailingDimensions() != trailingDimensions) {
            throw std::invalid_argument("RaggedSequenceConcatenate inputs must have identical trailing value dimensions.");
        }
        if (!uniqueValues.insert(input.getValues()).second) {
            throw std::invalid_argument(
                "RaggedSequenceConcatenate values inputs must be distinct graph tensors; duplicate values ports are not supported.");
        }

        outputMaxValuesPerRow = checkedAdd(outputMaxValuesPerRow,
                                           input.getMaxValuesPerRow(),
                                           "RaggedSequenceConcatenate output max_values_per_row overflow.");

        if (uniquePartitions.insert(input.getRowPartitionId()).second) {
            uniqueOffsets.push_back(input.getOffsets());
        }
    }

    std::vector<uint64_t> outputDimensions{outputMaxValuesPerRow};
    outputDimensions.insert(outputDimensions.end(), trailingDimensions.begin(), trailingDimensions.end());
    RaggedTensor output = serializedOutput.has_value() ? serializedOutput.value() :
        RaggedTensor::logical(Tensor(valuesDataType, outputDimensions),
                                       Tensor(offsetsDataType, {1}), outputMaxValuesPerRow);

    if (output.getValuesDataType() != valuesDataType || output.getOffsetsDataType() != offsetsDataType ||
        output.getTrailingDimensions() != trailingDimensions ||
        !output.hasMaxValuesPerRow() || output.getMaxValuesPerRow() != outputMaxValuesPerRow ||
        output.getValuesDimensions() != outputDimensions ||
        output.getOffsetsDimensions() != std::vector<uint64_t>{1}) {
        throw std::runtime_error("RaggedSequenceConcatenate serialized output descriptor does not match its inputs.");
    }
    {
        const Tensor& values = output.getValues();
        const Tensor& offsets = output.getOffsets();
        if (values.getOriginalId() == offsets.getOriginalId())
            throw std::runtime_error("RaggedSequenceConcatenate output values/partition IDs must differ.");
        for (const auto& input : inputs) {
            for (const auto& id : {input.getValues().getOriginalId(), input.getOffsets().getOriginalId()})
                if (values.getOriginalId() == id || offsets.getOriginalId() == id)
                    throw std::runtime_error("RaggedSequenceConcatenate output must have new tensor identities.");
        }
    }
    for (const RaggedTensor& input : inputs) {
        if (output.sharesPartitionWith(input)) {
            throw std::runtime_error("RaggedSequenceConcatenate must own a newly created logical row partition.");
        }
    }

    RaggedSequenceConcatenate layer;
    layer.raggedFeatureInputs = inputs;
    layer.raggedFeatureOutput = output;
    layer.uniqueOffsetsInputs = std::move(uniqueOffsets);
    layer.featureInputs.reserve(inputs.size() + layer.uniqueOffsetsInputs.size());
    for (const RaggedTensor& input : inputs) layer.featureInputs.push_back(input.getValues());
    for (const Tensor& offsets : layer.uniqueOffsetsInputs) layer.featureInputs.push_back(offsets);
    layer.featureOutputs = {output.getValues(), output.getOffsets()};
    layer.initialized = true;
    return layer;
}

RaggedSequenceConcatenate RaggedSequenceConcatenate::Builder::build() {
    if (!_network.has_value()) throw std::runtime_error("RaggedSequenceConcatenate requires a network.");
    RaggedSequenceConcatenate layer = RaggedSequenceConcatenate::makeLayer(_featureInputs);
    layer.addToNetwork(_network.value());
    return layer;
}

std::vector<Tensor> RaggedSequenceConcatenate::getOutputsFromInput(Tensor inputTensor) {
    (void)getConnectionType(inputTensor);
    if (emittedOutputsAfterAllInputsConnected || connectedInputPortIndices.size() != featureInputs.size()) return {};
    emittedOutputsAfterAllInputsConnected = true;
    return featureOutputs;
}

void RaggedSequenceConcatenate::informThatInputConnectionMade(Tensor inputTensor) {
    const int connectionType = getConnectionType(inputTensor);
    THOR_THROW_IF_FALSE(connectionType >= 0);
    connectedInputPortIndices.insert(static_cast<uint32_t>(connectionType));
}

void RaggedSequenceConcatenate::resetGraphTraversalState() {
    connectedInputPortIndices.clear();
    emittedOutputsAfterAllInputsConnected = false;
}

int RaggedSequenceConcatenate::getConnectionType(Tensor connectingTensor) const {
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        if (connectingTensor == featureInputs[i]) return static_cast<int>(i);
    }
    if (featureOutputs.size() == 2) {
        if (connectingTensor == featureOutputs[0]) return 0;
        if (connectingTensor == featureOutputs[1]) return 1;
    }
    throw std::runtime_error("Tensor is not connected to this RaggedSequenceConcatenate layer.");
}

std::optional<std::string> RaggedSequenceConcatenate::getInputPortName(const Tensor& inputTensor) const {
    for (uint32_t i = 0; i < raggedFeatureInputs.size(); ++i) {
        if (inputTensor == raggedFeatureInputs[i].getValues()) return "values[" + std::to_string(i) + "]";
    }
    for (uint32_t i = 0; i < uniqueOffsetsInputs.size(); ++i) {
        if (inputTensor == uniqueOffsetsInputs[i]) return "offsets[" + std::to_string(i) + "]";
    }
    return std::nullopt;
}

std::optional<std::string> RaggedSequenceConcatenate::getOutputPortName(const Tensor& outputTensor) const {
    if (outputTensor == raggedFeatureOutput.getValues()) return "values";
    if (outputTensor == raggedFeatureOutput.getOffsets()) return "offsets";
    return std::nullopt;
}

bool RaggedSequenceConcatenate::outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const {
    if (outputTensor != raggedFeatureOutput.getValues() && outputTensor != raggedFeatureOutput.getOffsets()) {
        throw std::invalid_argument("Tensor is not an output of this RaggedSequenceConcatenate layer.");
    }
    return true;
}

uint64_t RaggedSequenceConcatenate::getOutputTensorBytes(uint32_t batchSize) const {
    const uint64_t logicalBytes = raggedFeatureOutput.getValues().getTotalSizeInBytes();
    if (batchSize == 0 || logicalBytes > std::numeric_limits<uint64_t>::max() / batchSize)
        throw std::overflow_error("RaggedSequenceConcatenate output allocation overflow.");
    return logicalBytes * batchSize;
}

uint64_t RaggedSequenceConcatenate::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    const uint64_t valuesBytes = getOutputTensorBytes(batchSize);
    const uint64_t offsetBytes = raggedFeatureOutput.getDescriptorForPlacement(batchSize)
        .getOffsetsDescriptor().getArraySizeInBytes();
    // The physical copy plan reserves at most N spans per row; storage includes pinned host and device.
    const uint64_t spanBytes = raggedFeatureOutput.getOffsetsDataType() == DataType::UINT32 ?
        sizeof(ThorImplementation::RaggedSequenceCopySpan32) : sizeof(ThorImplementation::RaggedSequenceCopySpan64);
    const uint64_t n = raggedFeatureInputs.size();
    if (n > std::numeric_limits<uint64_t>::max() / batchSize ||
        batchSize * n > std::numeric_limits<uint64_t>::max() / (2 * spanBytes))
        throw std::overflow_error("RaggedSequenceConcatenate copy-plan allocation overflow.");
    const uint64_t copyPlanBytes = 2 * batchSize * n * spanBytes; // pinned host and device
    if (valuesBytes > std::numeric_limits<uint64_t>::max() - offsetBytes ||
        valuesBytes + offsetBytes > std::numeric_limits<uint64_t>::max() - copyPlanBytes)
        throw std::overflow_error("RaggedSequenceConcatenate total allocation overflow.");
    return valuesBytes + offsetBytes + copyPlanBytes;
}

std::shared_ptr<ThorImplementation::Layer> RaggedSequenceConcatenate::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    bool inferenceOnly) const {
    (void)placement;
    (void)drivingApiLayer;
    const int port = getConnectionType(connectingApiTensor);
    THOR_THROW_IF_FALSE(initialized);
    ThorImplementation::RaggedTensorDescriptor outputDescriptor;
    {
        const auto driver = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(
            drivingLayer, "RaggedSequenceConcatenate");
        uint64_t b = 0;
        const uint32_t numValues = static_cast<uint32_t>(raggedFeatureInputs.size());
        const RaggedTensor* input = nullptr;
        if (port < static_cast<int>(numValues)) input = &raggedFeatureInputs[port];
        else {
            const Tensor& token = uniqueOffsetsInputs[port - numValues];
            for (const auto& candidate : raggedFeatureInputs)
                if (candidate.getOffsets() == token) { input = &candidate; break; }
        }
        if (input == nullptr) throw std::runtime_error("RaggedSequenceConcatenate unrecognized structural port.");
        const bool structuralPort = port >= static_cast<int>(numValues);
        const auto managed = SegmentedPrimitivePlacementDetail::rowPartitionDescriptorForStamp(drivingLayer);
        if (managed.has_value()) b = managed->getBatchSize();
        else {
            const auto dims = driver.getDimensions();
            const auto trailing = input->getTrailingDimensions();
            if (!structuralPort) {
                if (driver.getDataType() != input->getValuesDataType() ||
                    dims.size() != trailing.size() + 1 || dims.empty() ||
                    !std::equal(trailing.begin(), trailing.end(), dims.begin() + 1) ||
                    dims[0] == 0 || dims[0] % input->getMaxValuesPerRow() != 0)
                    throw std::runtime_error("RaggedSequenceConcatenate values driver geometry mismatch.");
                b = dims[0] / input->getMaxValuesPerRow();
            } else if (driver.getDataType() == input->getOffsetsDataType() &&
                       dims.size() == 1 && dims[0] >= 2) {
                b = dims[0] - 1;
            } else if (!dims.empty() && dims[0] > 0 &&
                       dims[0] % input->getMaxValuesPerRow() == 0) {
                // HOST_EXTENT is an opaque values carrier for the *partition*.
                // It need not be this operation's values tensor: multiple
                // same-partition branches can have different trailing shapes.
                b = dims[0] / input->getMaxValuesPerRow();
            } else {
                throw std::runtime_error("RaggedSequenceConcatenate invalid HOST_EXTENT carrier.");
            }
        }
        if (b == 0 || b > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("RaggedSequenceConcatenate placement B outside uint32 range.");
        const auto placed = input->getDescriptorForPlacement(b);
        const auto physicalDims = driver.getDimensions();
        if (!structuralPort && driver != placed.getValuesDescriptor())
            throw std::runtime_error("RaggedSequenceConcatenate values driver descriptor mismatch.");
        if (structuralPort && driver != placed.getOffsetsDescriptor() &&
            (physicalDims.empty() || physicalDims[0] != placed.getMaxTotalValues()))
            throw std::runtime_error("RaggedSequenceConcatenate HOST_EXTENT carrier capacity mismatch.");
        if (managed.has_value() && managed.value() != placed.getRowPartition())
            throw std::runtime_error("RaggedSequenceConcatenate managed partition metadata mismatch.");
        for (const auto& input : raggedFeatureInputs) (void)input.getDescriptorForPlacement(b);
        outputDescriptor = raggedFeatureOutput.getDescriptorForPlacement(b);
        if (!ThorImplementation::canonicalRowPartitionOffsetCanRepresent(
                raggedFeatureOutput.getOffsetsDataType(), outputDescriptor.getMaxTotalValues()))
            throw std::runtime_error("RaggedSequenceConcatenate output capacity exceeds offset dtype.");
    }
    auto physical = std::make_shared<ThorImplementation::RaggedSequenceConcatenate>(
        static_cast<uint32_t>(raggedFeatureInputs.size()),
        static_cast<uint32_t>(uniqueOffsetsInputs.size()), outputDescriptor);
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedSequenceConcatenate::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    json inputs = json::array();
    for (const RaggedTensor& input : raggedFeatureInputs)
        inputs.push_back(SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(input));
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", "1.1.0"},
                {"layer_type", "ragged_sequence_concatenate"},
                {"ragged_inputs", inputs},
                {"ragged_output", SegmentedPrimitiveDetail::architectureJsonForLayerRaggedReference(raggedFeatureOutput)}};
}

void RaggedSequenceConcatenate::deserialize(const json& j, Network* network) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0")
        throw std::runtime_error("Unsupported version in RaggedSequenceConcatenate::deserialize: " + version);
    if (version == "1.1.0" && (!j.is_object() || j.size() != 5 ||
                               j.at("factory") != Layer::Factory::Layer.value()))
        throw std::runtime_error("RaggedSequenceConcatenate canonical layer has unexpected fields.");
    if (j.at("layer_type").get<std::string>() != "ragged_sequence_concatenate") {
        throw std::runtime_error("Layer type mismatch in RaggedSequenceConcatenate::deserialize: " +
                                 j.at("layer_type").get<std::string>());
    }

    std::vector<RaggedTensor> inputs;
    if (!j.at("ragged_inputs").is_array() || j.at("ragged_inputs").size() < 2)
        throw std::runtime_error("RaggedSequenceConcatenate needs at least two serialized inputs.");
    uint64_t outputM = 0;
    for (const json& inputJson : j.at("ragged_inputs")) {
        RaggedTensor input = SegmentedPrimitiveDetail::reconstructInput(
            inputJson, network, "RaggedSequenceConcatenate");
        if (version == "1.1.0") checkCanonicalRagged(inputJson, input);
        outputM = checkedAdd(outputM, input.getMaxValuesPerRow(),
                             "RaggedSequenceConcatenate semantic output M overflow.");
        inputs.push_back(input);
    }

    const json& outputJson = j.at("ragged_output");
    RaggedTensor output = SegmentedPrimitiveDetail::reconstructNewPartitionOutput(
        outputJson, outputM, inputs.front().getValuesDataType(),
        inputs.front().getOffsetsDataType(), inputs.front().getTrailingDimensions(),
        "RaggedSequenceConcatenate");
    if (version == "1.1.0") checkCanonicalRagged(outputJson, output);
    RaggedSequenceConcatenate layer = makeLayer(inputs, output);
    layer.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Layer::register_layer("ragged_sequence_concatenate", &Thor::RaggedSequenceConcatenate::deserialize);
    return true;
}();
}  // namespace
