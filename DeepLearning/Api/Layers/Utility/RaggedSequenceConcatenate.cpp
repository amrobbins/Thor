#include "DeepLearning/Api/Layers/Utility/RaggedSequenceConcatenate.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

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
        throw std::runtime_error(
            "RaggedSequenceConcatenate serialized ragged input metadata does not match reconstructed tensors.");
    }
    return input;
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
    const uint64_t batchSize = reference.getBatchSize();
    const DataType valuesDataType = reference.getValuesDataType();
    const DataType offsetsDataType = reference.getOffsetsDataType();
    const std::vector<uint64_t> trailingDimensions = reference.getTrailingDimensions();
    if (batchSize > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("RaggedSequenceConcatenate batch size exceeds Thor's uint32 placement capacity.");
    }

    uint64_t outputMaxTotalValues = 0;
    uint64_t outputMaxValuesPerRow = 0;
    bool allInputsHaveMaxValuesPerRow = true;
    std::set<Tensor> uniqueValues;
    std::vector<Tensor> uniqueOffsets;
    std::map<Tensor, uint32_t> offsetPortByTensor;
    std::vector<uint32_t> offsetPortForInput;
    offsetPortForInput.reserve(inputs.size());

    for (uint32_t i = 0; i < inputs.size(); ++i) {
        const RaggedTensor& input = inputs[i];
        if (!input.isInitialized()) throw std::invalid_argument("RaggedSequenceConcatenate inputs must be initialized.");
        if (input.getBatchSize() != batchSize) {
            throw std::invalid_argument("RaggedSequenceConcatenate inputs must have the same logical batch size.");
        }
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

        outputMaxTotalValues = checkedAdd(outputMaxTotalValues,
                                          input.getMaxTotalValues(),
                                          "RaggedSequenceConcatenate output max_total_values overflow.");
        if (input.hasMaxValuesPerRow()) {
            outputMaxValuesPerRow = checkedAdd(outputMaxValuesPerRow,
                                               input.getMaxValuesPerRow(),
                                               "RaggedSequenceConcatenate output max_values_per_row overflow.");
        } else {
            allInputsHaveMaxValuesPerRow = false;
        }

        auto foundOffset = offsetPortByTensor.find(input.getOffsets());
        if (foundOffset == offsetPortByTensor.end()) {
            const uint32_t newPort = static_cast<uint32_t>(uniqueOffsets.size());
            uniqueOffsets.push_back(input.getOffsets());
            offsetPortByTensor.emplace(input.getOffsets(), newPort);
            offsetPortForInput.push_back(newPort);
        } else {
            offsetPortForInput.push_back(foundOffset->second);
        }
    }

    if (!ThorImplementation::canonicalRowPartitionOffsetCanRepresent(offsetsDataType, outputMaxTotalValues)) {
        throw std::invalid_argument(
            "RaggedSequenceConcatenate output max_total_values cannot be represented by the selected offsets dtype.");
    }

    RaggedTensor output = serializedOutput.has_value()
        ? serializedOutput.value()
        : (allInputsHaveMaxValuesPerRow
               ? RaggedTensor(valuesDataType,
                              trailingDimensions,
                              batchSize,
                              outputMaxTotalValues,
                              outputMaxValuesPerRow,
                              offsetsDataType)
               : RaggedTensor(valuesDataType,
                              trailingDimensions,
                              batchSize,
                              outputMaxTotalValues,
                              offsetsDataType));

    if (output.getValuesDataType() != valuesDataType || output.getOffsetsDataType() != offsetsDataType ||
        output.getBatchSize() != batchSize || output.getMaxTotalValues() != outputMaxTotalValues ||
        output.getTrailingDimensions() != trailingDimensions || output.hasMaxValuesPerRow() != allInputsHaveMaxValuesPerRow ||
        (allInputsHaveMaxValuesPerRow && output.getMaxValuesPerRow() != outputMaxValuesPerRow)) {
        throw std::runtime_error("RaggedSequenceConcatenate serialized output descriptor does not match its inputs.");
    }
    for (const RaggedTensor& input : inputs) {
        if (output.getOffsets().getOriginalId() == input.getOffsets().getOriginalId()) {
            throw std::runtime_error("RaggedSequenceConcatenate must own a newly produced offsets tensor.");
        }
    }

    RaggedSequenceConcatenate layer;
    layer.raggedFeatureInputs = inputs;
    layer.raggedFeatureOutput = output;
    layer.uniqueOffsetsInputs = std::move(uniqueOffsets);
    layer.offsetPortForInput = std::move(offsetPortForInput);
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
    (void)batchSize;
    return raggedFeatureOutput.getValues().getTotalSizeInBytes() + raggedFeatureOutput.getOffsets().getTotalSizeInBytes();
}

uint64_t RaggedSequenceConcatenate::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    return getOutputTensorBytes(batchSize);
}

std::shared_ptr<ThorImplementation::Layer> RaggedSequenceConcatenate::stamp(
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
    auto physical = std::make_shared<ThorImplementation::RaggedSequenceConcatenate>(
        static_cast<uint32_t>(raggedFeatureInputs.size()),
        static_cast<uint32_t>(uniqueOffsetsInputs.size()),
        offsetPortForInput,
        raggedFeatureOutput.getDescriptor());
    physical->setConstructForInferenceOnly(inferenceOnly);
    physical->setName(getLayerType());
    return physical;
}

json RaggedSequenceConcatenate::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    json inputs = json::array();
    for (const RaggedTensor& input : raggedFeatureInputs) inputs.push_back(input.architectureJson());
    return json{{"factory", Layer::Factory::Layer.value()},
                {"version", getLayerVersion()},
                {"layer_type", "ragged_sequence_concatenate"},
                {"ragged_inputs", inputs},
                {"ragged_output", raggedFeatureOutput.architectureJson()}};
}

void RaggedSequenceConcatenate::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in RaggedSequenceConcatenate::deserialize: " +
                                 j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "ragged_sequence_concatenate") {
        throw std::runtime_error("Layer type mismatch in RaggedSequenceConcatenate::deserialize: " +
                                 j.at("layer_type").get<std::string>());
    }

    std::vector<RaggedTensor> inputs;
    for (const json& inputJson : j.at("ragged_inputs")) inputs.push_back(reconstructInput(inputJson, network));

    const json& outputJson = j.at("ragged_output");
    Tensor outputValues = Tensor::deserialize(outputJson.at("values"));
    Tensor outputOffsets = Tensor::deserialize(outputJson.at("offsets"));
    RaggedTensor output = outputJson.contains("max_values_per_row")
        ? RaggedTensor(outputValues, outputOffsets, outputJson.at("max_values_per_row").get<uint64_t>())
        : RaggedTensor(outputValues, outputOffsets);
    if (outputJson.at("ragged_rank").get<uint32_t>() != 1 ||
        output.getBatchSize() != outputJson.at("batch_size").get<uint64_t>() ||
        output.getMaxTotalValues() != outputJson.at("max_total_values").get<uint64_t>()) {
        throw std::runtime_error(
            "RaggedSequenceConcatenate serialized ragged output metadata does not match reconstructed tensors.");
    }

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
