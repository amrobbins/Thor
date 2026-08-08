#include "DeepLearning/Api/Layers/Utility/Einsum.h"

#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/TensorOperations/Einsum/EinsumPlanner.h"

#include <limits>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

namespace Thor {

bool Einsum::isSupportedStorageDType(DataType dataType) {
    return dataType == DataType::FP16 || dataType == DataType::BF16 || dataType == DataType::FP32;
}

void Einsum::validateAndResolve(const std::string& equation,
                                const std::vector<Tensor>& featureInputs,
                                ThorImplementation::ResolvedEinsumEquation* resolved) {
    if (equation.empty()) {
        throw std::invalid_argument("Einsum equation must not be empty.");
    }
    if (featureInputs.empty()) {
        throw std::invalid_argument("Einsum requires at least one feature input.");
    }
    if (featureInputs.size() > ThorImplementation::EinsumPlanner::MAX_SOURCE_OPERANDS) {
        throw std::invalid_argument("Einsum supports at most " +
                                    std::to_string(ThorImplementation::EinsumPlanner::MAX_SOURCE_OPERANDS) +
                                    " operand occurrences.");
    }

    const ThorImplementation::EinsumEquation parsed = ThorImplementation::EinsumParser::parse(equation);
    if (parsed.inputs.size() != featureInputs.size()) {
        throw std::invalid_argument("Einsum equation declares " + std::to_string(parsed.inputs.size()) +
                                    " operands but the builder received " + std::to_string(featureInputs.size()) + ".");
    }

    const DataType dataType = featureInputs.front().getDataType();
    if (!isSupportedStorageDType(dataType)) {
        throw std::invalid_argument("Einsum supports FP16, BF16, and FP32 feature tensors.");
    }

    std::vector<std::vector<uint64_t>> inputDimensions;
    inputDimensions.reserve(featureInputs.size());
    for (uint32_t i = 0; i < featureInputs.size(); ++i) {
        const Tensor& input = featureInputs[i];
        if (!input.isInitialized()) {
            throw std::invalid_argument("Einsum operand[" + std::to_string(i) + "] is uninitialized.");
        }
        if (input.getDataType() != dataType) {
            throw std::invalid_argument("Einsum requires all feature inputs to have the same storage data type.");
        }
        inputDimensions.push_back(input.getDimensions());
    }

    ThorImplementation::ResolvedEinsumEquation localResolved =
        ThorImplementation::EinsumParser::resolve(parsed, inputDimensions);
    if (resolved != nullptr) {
        *resolved = std::move(localResolved);
    }
}

void Einsum::rebuildInputBindings() {
    inputOperandBindingsByTensorOriginalId.clear();
    for (uint32_t operandIndex = 0; operandIndex < featureInputs.size(); ++operandIndex) {
        inputOperandBindingsByTensorOriginalId[featureInputs[operandIndex].getOriginalId()].push_back(operandIndex);
    }
}

Tensor Einsum::getFeatureOutput(Tensor inputTensor) const {
    if (!inputOperandBindingsByTensorOriginalId.contains(inputTensor.getOriginalId())) {
        throw std::runtime_error("Tensor is not an input to this Einsum layer.");
    }
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    return featureOutputs[0];
}

Tensor Einsum::getFeatureInput(Tensor outputTensor) const {
    (void)outputTensor;
    // A single einsum output generally depends on several operands, so there
    // is no unique inverse output->input mapping.
    THOR_UNREACHABLE();
}

int Einsum::getConnectionType(Tensor connectingTensor) const {
    const uint64_t originalId = connectingTensor.getOriginalId();
    auto inputIt = inputOperandBindingsByTensorOriginalId.find(originalId);
    if (inputIt != inputOperandBindingsByTensorOriginalId.end()) {
        const std::vector<uint32_t>& bindings = inputIt->second;
        THOR_THROW_IF_FALSE(!bindings.empty());
        uint32_t& cursor = nextInputBindingConnectionCursorByTensorOriginalId[originalId];
        const uint32_t operandIndex = bindings[cursor % bindings.size()];
        ++cursor;
        return static_cast<int>(operandIndex);
    }

    if (featureOutputs.size() == 1 && connectingTensor == featureOutputs[0]) {
        return 0;
    }
    throw std::runtime_error("Tensor is not connected to this Einsum layer.");
}

std::optional<std::string> Einsum::getInputPortName(const Tensor& inputTensor) const {
    auto it = inputOperandBindingsByTensorOriginalId.find(inputTensor.getOriginalId());
    if (it == inputOperandBindingsByTensorOriginalId.end()) {
        return std::nullopt;
    }
    if (it->second.size() == 1) {
        return "operand[" + std::to_string(it->second.front()) + "]";
    }
    std::string name = "operand[";
    for (size_t i = 0; i < it->second.size(); ++i) {
        if (i != 0) {
            name += ',';
        }
        name += std::to_string(it->second[i]);
    }
    name += ']';
    return name;
}

std::optional<std::string> Einsum::getOutputPortName(const Tensor& outputTensor) const {
    if (featureOutputs.size() == 1 && outputTensor == featureOutputs[0]) {
        return "output";
    }
    return std::nullopt;
}

void Einsum::informThatInputConnectionMade(Tensor inputTensor) {
    auto it = inputOperandBindingsByTensorOriginalId.find(inputTensor.getOriginalId());
    if (it == inputOperandBindingsByTensorOriginalId.end()) {
        throw std::runtime_error("Einsum informed of a connection for an unknown input tensor.");
    }
    for (uint32_t operandIndex : it->second) {
        connectedInputOperandIndices.insert(operandIndex);
    }
}

std::vector<Tensor> Einsum::getOutputsFromInput(Tensor inputTensor) {
    if (!inputOperandBindingsByTensorOriginalId.contains(inputTensor.getOriginalId())) {
        throw std::runtime_error("Einsum asked for outputs from an unknown input tensor.");
    }
    if (emittedFeatureOutputAfterAllInputsConnected || connectedInputOperandIndices.size() != featureInputs.size()) {
        return {};
    }
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
    emittedFeatureOutputAfterAllInputsConnected = true;
    return {featureOutputs[0]};
}

void Einsum::resetGraphTraversalState() {
    connectedInputOperandIndices.clear();
    emittedFeatureOutputAfterAllInputsConnected = false;
    nextInputBindingConnectionCursorByTensorOriginalId.clear();
}

std::shared_ptr<ThorImplementation::Layer> Einsum::stamp(ThorImplementation::TensorPlacement placement,
                                                         std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                         std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                         Thor::Tensor connectingApiTensor,
                                                         const bool inferenceOnly) const {
    (void)placement;
    (void)drivingLayer;
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    if (!inputOperandBindingsByTensorOriginalId.contains(connectingApiTensor.getOriginalId())) {
        throw std::runtime_error("Einsum::stamp called with a tensor that is not one of its declared operands.");
    }

    auto physicalLayer = std::make_shared<ThorImplementation::EinsumLayer>(equation);
    physicalLayer->setConstructForInferenceOnly(inferenceOnly);
    return physicalLayer;
}

uint64_t Einsum::getFirstInstanceMemRequirementInBytes(
    uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);

    // The physical layer owns one dense operand-gradient tensor per logical
    // operand occurrence on a live training path plus its feature output. Count
    // duplicate operand occurrences separately because their product-rule terms
    // are distinct until TensorFanout accumulates them upstream.
    uint64_t bytesPerExample = featureOutputs[0].getTotalSizeInBytes();
    for (const Tensor& input : featureInputs) {
        if (bytesPerExample > std::numeric_limits<uint64_t>::max() - input.getTotalSizeInBytes()) {
            throw std::overflow_error("Einsum API memory requirement overflows uint64_t.");
        }
        bytesPerExample += input.getTotalSizeInBytes();
    }
    if (batchSize != 0 && bytesPerExample > std::numeric_limits<uint64_t>::max() / batchSize) {
        throw std::overflow_error("Einsum API memory requirement overflows uint64_t.");
    }
    return bytesPerExample * batchSize;
}

json Einsum::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(!featureInputs.empty());
    THOR_THROW_IF_FALSE(featureOutputs.size() == 1);

    json j;
    j["factory"] = Layer::Factory::Layer.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = to_snake_case(getLayerType());
    j["equation"] = equation;

    json inputs = json::array();
    for (const Tensor& input : featureInputs) {
        inputs.push_back(input.architectureJson());
    }
    j["inputs"] = std::move(inputs);

    json outputs = json::array();
    outputs.push_back(featureOutputs[0].architectureJson());
    j["outputs"] = std::move(outputs);
    return j;
}

void Einsum::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in Einsum::deserialize: " + j.at("version").get<std::string>());
    }
    if (j.at("layer_type").get<std::string>() != "einsum") {
        throw std::runtime_error("Layer type mismatch in Einsum::deserialize: " + j.at("layer_type").get<std::string>());
    }
    if (network == nullptr) {
        throw std::invalid_argument("Einsum::deserialize requires a non-null network.");
    }

    const std::string equation = j.at("equation").get<std::string>();
    const std::vector<json> inputJsons = j.at("inputs").get<std::vector<json>>();
    if (inputJsons.empty()) {
        throw std::runtime_error("Einsum::deserialize requires at least one input tensor.");
    }

    std::vector<Tensor> featureInputs;
    featureInputs.reserve(inputJsons.size());
    for (const json& input : inputJsons) {
        const uint64_t originalTensorId = input.at("id").get<uint64_t>();
        featureInputs.push_back(network->getApiTensorByOriginalId(originalTensorId));
    }

    ThorImplementation::ResolvedEinsumEquation resolved;
    validateAndResolve(equation, featureInputs, &resolved);

    const std::vector<json> outputJsons = j.at("outputs").get<std::vector<json>>();
    if (outputJsons.size() != 1) {
        throw std::runtime_error("Einsum::deserialize requires exactly one output tensor.");
    }
    Tensor featureOutput = Tensor::deserialize(outputJsons[0]);
    if (featureOutput.getDataType() != featureInputs.front().getDataType()) {
        throw std::runtime_error("Einsum::deserialize output dtype does not match the input dtype.");
    }
    if (featureOutput.getDimensions() != resolved.output_dimensions) {
        throw std::runtime_error("Einsum::deserialize output dimensions do not match the resolved equation.");
    }

    Einsum einsum;
    einsum.equation = equation;
    einsum.featureInputs = std::move(featureInputs);
    einsum.featureOutputs = {featureOutput};
    einsum.rebuildInputBindings();
    for (const Tensor& input : einsum.featureInputs) {
        einsum.outputTensorFromInputTensor[input] = featureOutput;
    }
    einsum.initialized = true;
    einsum.addToNetwork(network);
}

Einsum Einsum::Builder::build() {
    if (!_network.has_value()) {
        throw std::runtime_error("Einsum requires a network.");
    }
    if (!_equation.has_value()) {
        throw std::runtime_error("Einsum requires an equation.");
    }
    if (_featureInputs.empty()) {
        throw std::runtime_error("Einsum requires at least one feature input.");
    }

    ThorImplementation::ResolvedEinsumEquation resolved;
    Einsum::validateAndResolve(_equation.value(), _featureInputs, &resolved);

    Einsum einsum;
    einsum.equation = _equation.value();
    einsum.featureInputs = _featureInputs;
    einsum.featureOutputs.push_back(Tensor(_featureInputs.front().getDataType(), resolved.output_dimensions));
    einsum.rebuildInputBindings();
    for (const Tensor& input : einsum.featureInputs) {
        einsum.outputTensorFromInputTensor[input] = einsum.featureOutputs[0];
    }
    einsum.initialized = true;
    einsum.addToNetwork(_network.value());
    return einsum;
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Layer::register_layer("einsum", &Thor::Einsum::deserialize);
    return true;
}();
}  // namespace
