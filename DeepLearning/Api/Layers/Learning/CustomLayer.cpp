#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

using DynamicExpression = ThorImplementation::DynamicExpression;

namespace Thor {

CustomLayer::CustomLayer(DynamicExpression expr,
                         const std::vector<TensorMap>& inputInterfaces,
                         std::vector<std::shared_ptr<Parameter>> parameters,
                         bool inferenceOnly,
                         bool useFastMath)
    : CustomLayer(std::move(expr), {}, {}, inputInterfaces, {}, std::move(parameters), inferenceOnly, useFastMath) {}

CustomLayer::CustomLayer(DynamicExpression expr,
                         std::vector<std::string> inputNames,
                         std::vector<std::string> outputNames,
                         const std::vector<TensorMap>& inputInterfaces,
                         const std::vector<TensorMap>& outputInterfaces,
                         std::vector<std::shared_ptr<Parameter>> parameters,
                         bool inferenceOnly,
                         bool useFastMath)
    : expr(std::move(expr)), inferenceOnly(inferenceOnly), useFastMath(useFastMath), parameters(std::move(parameters)) {
    if (inputNames.empty())
        inputNames = this->expr.getExpectedInputNames();
    if (outputNames.empty())
        outputNames = this->expr.getExpectedOutputNames();

    this->inputNames = std::move(inputNames);
    this->outputNames = std::move(outputNames);

    if (inputInterfaces.empty())
        throw runtime_error("Cannot create a CustomLayer with zero input interfaces.");

    validateInputInterfacesMatchExpression();
    validateOutputInterfacesMatchExpression();
    validateExpressionCanBindFeatureAndParameterInputs();

    for (const TensorMap& inputInterface : inputInterfaces) {
        validateTensorInterface(inputInterface, "input");
        validateInterfaceNames(inputInterface, this->inputNames, "input");
    }

    // Ensure no two interfaces are exactly the same. Aliasing a tensor across otherwise-distinct interfaces is allowed,
    // but exact duplicate interfaces make getOutputInterface(inputInterface) ambiguous.
    for (uint32_t i = 0; i < inputInterfaces.size() - 1; ++i) {
        for (uint32_t j = i + 1; j < inputInterfaces.size(); ++j) {
            if (interfaceMatches(inputInterfaces[i], inputInterfaces[j])) {
                throw runtime_error("CustomLayer: An input interface was connected more than once.");
            }
        }
    }

    // Ensure shape and dtype equivalence between all corresponding tensors across all interfaces.
    const TensorMap& referenceInterface = inputInterfaces.front();
    for (const std::string& name : this->inputNames) {
        const Tensor& referenceTensor = referenceInterface.at(name);
        const auto referenceDataType = referenceTensor.getDataType();
        const auto referenceDimensions = referenceTensor.getDimensions();

        for (uint32_t interfaceIndex = 1; interfaceIndex < inputInterfaces.size(); ++interfaceIndex) {
            const Tensor& tensor = inputInterfaces[interfaceIndex].at(name);

            if (tensor.getDataType() != referenceDataType || tensor.getDimensions() != referenceDimensions) {
                std::ostringstream oss;
                oss << "CustomLayer input tensor '" << name << "' must have the same shape and dtype across all "
                    << "input interfaces. Interface 0 has " << referenceTensor.getDescriptorString() << ", but interface " << interfaceIndex
                    << " has " << tensor.getDescriptorString() << ".";
                throw runtime_error(oss.str());
            }
        }
    }

    assignInputInterfaces(inputInterfaces);
    if (outputInterfaces.empty())
        materializeOutputInterfacesFromInputInterfaces();
    else
        assignOutputInterfaces(outputInterfaces);
    initialized = true;
}

std::string CustomLayer::joinNames(const std::set<std::string>& names) {
    if (names.empty())
        return "<none>";

    std::ostringstream oss;
    bool first = true;
    for (const auto& name : names) {
        if (!first)
            oss << ", ";
        oss << name;
        first = false;
    }
    return oss.str();
}

uint32_t CustomLayer::encodeInputConnection(uint32_t interfaceIndex, uint32_t inputPortIndex) const {
    return interfaceIndex * static_cast<uint32_t>(inputNames.size()) + inputPortIndex;
}

uint32_t CustomLayer::encodeOutputConnection(uint32_t interfaceIndex, uint32_t outputPortIndex) const {
    return interfaceIndex * static_cast<uint32_t>(outputNames.size()) + outputPortIndex;
}

void CustomLayer::validateTensorInterface(const TensorMap& tensorInterface, const std::string& what) {
    if (tensorInterface.empty()) {
        throw runtime_error("CustomLayer requires at least one tensor in each " + what + " interface.");
    }

    for (const auto& [name, tensor] : tensorInterface) {
        if (name.empty()) {
            throw runtime_error("CustomLayer " + what + " name cannot be empty.");
        }
        if (!tensor.isInitialized()) {
            throw runtime_error("CustomLayer " + what + " tensor for name '" + name + "' is not initialized.");
        }
    }
}

void CustomLayer::validateInterfaceNames(const TensorMap& tensorInterface,
                                         const std::vector<std::string>& expectedNames,
                                         const std::string& what) {
    std::set<std::string> actualNames;
    for (const auto& [name, tensor] : tensorInterface) {
        (void)tensor;
        actualNames.insert(name);
    }

    std::set<std::string> expectedNameSet(expectedNames.begin(), expectedNames.end());
    if (actualNames != expectedNameSet) {
        throw runtime_error("CustomLayer " + what + " interface name mismatch. Expected {" + joinNames(expectedNameSet) + "}, got {" +
                            joinNames(actualNames) + "}.");
    }
}

void CustomLayer::validateInputInterfacesMatchExpression() const {
    if (inputNames.empty()) {
        throw runtime_error("CustomLayer must declare at least one feature input name.");
    }
}

void CustomLayer::validateOutputInterfacesMatchExpression() const {
    if (outputNames.empty()) {
        throw runtime_error("CustomLayer must declare at least one output name.");
    }
}

void CustomLayer::validateExpressionCanBindFeatureAndParameterInputs() const {
    const std::vector<std::string>& expectedInputNames = expr.getExpectedInputNames();
    if (expectedInputNames.empty())
        return;

    std::set<std::string> expected(expectedInputNames.begin(), expectedInputNames.end());
    for (const std::string& name : inputNames) {
        if (!expected.contains(name)) {
            throw runtime_error("CustomLayer expression expected input names do not include feature input '" + name + "'.");
        }
    }
    for (const auto& parameter : parameters) {
        if (parameter == nullptr)
            throw runtime_error("CustomLayer contains a null Parameter.");
        const std::string& name = parameter->getName();
        if (!expected.contains(name)) {
            throw runtime_error("CustomLayer expression expected input names do not include parameter '" + name + "'.");
        }
    }

    const std::vector<std::string>& expectedOutputNames = expr.getExpectedOutputNames();
    if (!expectedOutputNames.empty()) {
        std::set<std::string> expectedOutputs(expectedOutputNames.begin(), expectedOutputNames.end());
        for (const std::string& name : outputNames) {
            if (!expectedOutputs.contains(name)) {
                throw runtime_error("CustomLayer expression expected output names do not include output '" + name + "'.");
            }
        }
    }
}

void CustomLayer::assignInputInterfaces(const std::vector<TensorMap>& inputInterfaces) {
    if (inputInterfaces.empty()) {
        throw runtime_error("CustomLayer requires at least one input interface.");
    }

    featureInputs.clear();
    this->inputInterfaces.clear();
    connectedInputPortIndicesByInterface.clear();
    emittedOutputInterface.clear();
    inputBindingsByTensorOriginalId.clear();
    nextInputBindingConnectionCursorByTensorOriginalId.clear();

    for (uint32_t interfaceIndex = 0; interfaceIndex < inputInterfaces.size(); ++interfaceIndex) {
        const TensorMap& inputInterface = inputInterfaces[interfaceIndex];
        validateTensorInterface(inputInterface, "input");
        validateInterfaceNames(inputInterface, inputNames, "input");

        this->inputInterfaces.push_back(inputInterface);
        connectedInputPortIndicesByInterface.emplace_back();
        emittedOutputInterface.push_back(false);

        for (uint32_t inputPortIndex = 0; inputPortIndex < inputNames.size(); ++inputPortIndex) {
            const std::string& name = inputNames[inputPortIndex];
            const Tensor& tensor = inputInterface.at(name);

            featureInputs.push_back(tensor);
            inputBindingsByTensorOriginalId[tensor.getOriginalId()].push_back(InputBinding{interfaceIndex, inputPortIndex, name});
        }
    }
}

Tensor CustomLayer::defaultOutputTensorForInterface(const TensorMap& inputInterface, const std::string& outputName) const {
    auto sameNameInput = inputInterface.find(outputName);
    if (sameNameInput != inputInterface.end()) {
        return sameNameInput->second.clone();
    }

    // DynamicExpression can infer exact output tensors only after physical GPU tensors/streams exist. The API graph still
    // needs placeholder edge tensors, so default to the first input port's descriptor. This matches the common elementwise /
    // broadcast custom-layer case; physical stamping remains the source of truth for the real output descriptor.
    return inputInterface.at(inputNames.front()).clone();
}

void CustomLayer::materializeOutputInterfacesFromInputInterfaces() {
    std::vector<TensorMap> materialized;
    materialized.reserve(inputInterfaces.size());

    for (const TensorMap& inputInterface : inputInterfaces) {
        TensorMap outputInterface;
        for (const std::string& outputName : outputNames) {
            outputInterface[outputName] = defaultOutputTensorForInterface(inputInterface, outputName);
        }
        materialized.push_back(std::move(outputInterface));
    }

    assignOutputInterfaces(materialized);
}

void CustomLayer::assignOutputInterfaces(const std::vector<TensorMap>& outputInterfaces) {
    featureOutputs.clear();
    this->outputInterfaces.clear();

    for (const TensorMap& outputInterface : outputInterfaces) {
        validateTensorInterface(outputInterface, "output");
        validateInterfaceNames(outputInterface, outputNames, "output");
        this->outputInterfaces.push_back(outputInterface);

        for (const std::string& name : outputNames) {
            featureOutputs.push_back(outputInterface.at(name));
        }
    }
}

bool CustomLayer::interfaceMatches(const TensorMap& subset, const TensorMap& superset) {
    if (subset.size() != superset.size())
        return false;

    for (const auto& [name, tensor] : subset) {
        const auto& found = superset.find(name);
        if (found == superset.end() || found->second != tensor)
            return false;
    }
    return true;
}

CustomLayer::TensorMap CustomLayer::getOutputInterface(const TensorMap& inputInterface) const {
    validateInterfaceNames(inputInterface, inputNames, "input");

    bool foundMatch = false;
    uint32_t matchedIndex = 0;
    for (uint32_t i = 0; i < inputInterfaces.size(); ++i) {
        if (!interfaceMatches(inputInterfaces[i], inputInterface)) {
            continue;
        }
        if (foundMatch) {
            throw runtime_error("Cannot get output interface because the input interface matches more than one CustomLayer interface.");
        }
        foundMatch = true;
        matchedIndex = i;
    }

    if (!foundMatch) {
        throw runtime_error(
            "Cannot get output interface from the inputInterface that was sent,"
            " because the input interface that was sent is not in the list of connected input interfaces.");
    }

    if (matchedIndex >= outputInterfaces.size()) {
        throw runtime_error("CustomLayer output interface has not been materialized yet for the requested input interface.");
    }

    return outputInterfaces[matchedIndex];
}

int CustomLayer::getConnectionType(Tensor connectingTensor) const {
    const uint64_t originalId = connectingTensor.getOriginalId();
    auto inputIt = inputBindingsByTensorOriginalId.find(originalId);
    if (inputIt != inputBindingsByTensorOriginalId.end()) {
        const std::vector<InputBinding>& bindings = inputIt->second;
        assert(!bindings.empty());

        uint32_t& cursor = nextInputBindingConnectionCursorByTensorOriginalId[originalId];
        const InputBinding& binding = bindings[cursor % bindings.size()];
        ++cursor;
        return static_cast<int>(encodeInputConnection(binding.interfaceIndex, binding.inputPortIndex));
    }

    for (uint32_t interfaceIndex = 0; interfaceIndex < outputInterfaces.size(); ++interfaceIndex) {
        for (uint32_t outputPortIndex = 0; outputPortIndex < outputNames.size(); ++outputPortIndex) {
            const std::string& name = outputNames[outputPortIndex];
            if (outputInterfaces[interfaceIndex].at(name) == connectingTensor) {
                return static_cast<int>(encodeOutputConnection(interfaceIndex, outputPortIndex));
            }
        }
    }

    throw runtime_error("Tensor is not connected to this CustomLayer.");
}

void CustomLayer::informThatInputConnectionMade(Tensor inputTensor) {
    auto it = inputBindingsByTensorOriginalId.find(inputTensor.getOriginalId());
    if (it == inputBindingsByTensorOriginalId.end()) {
        throw runtime_error("CustomLayer informed of connection for unknown input tensor.");
    }

    for (const InputBinding& binding : it->second) {
        assert(binding.interfaceIndex < connectedInputPortIndicesByInterface.size());
        connectedInputPortIndicesByInterface[binding.interfaceIndex].insert(binding.inputPortIndex);
    }
}

std::vector<Tensor> CustomLayer::getOutputsFromInput(Tensor inputTensor) {
    auto it = inputBindingsByTensorOriginalId.find(inputTensor.getOriginalId());
    if (it == inputBindingsByTensorOriginalId.end()) {
        throw runtime_error("CustomLayer asked for outputs from unknown input tensor.");
    }

    std::vector<Tensor> outputs;

    std::set<uint32_t> candidateInterfaces;
    for (const InputBinding& binding : it->second) {
        candidateInterfaces.insert(binding.interfaceIndex);
    }

    for (uint32_t interfaceIndex : candidateInterfaces) {
        if (interfaceIndex >= connectedInputPortIndicesByInterface.size()) {
            continue;
        }
        if (emittedOutputInterface[interfaceIndex]) {
            continue;
        }
        if (connectedInputPortIndicesByInterface[interfaceIndex].size() != inputNames.size()) {
            continue;
        }
        if (interfaceIndex >= outputInterfaces.size()) {
            continue;
        }

        emittedOutputInterface[interfaceIndex] = true;
        outputs.reserve(outputs.size() + outputNames.size());
        for (const std::string& name : outputNames) {
            outputs.push_back(outputInterfaces[interfaceIndex].at(name));
        }
    }

    return outputs;
}

uint64_t CustomLayer::getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;

    uint64_t totalBytes = 0;
    std::set<uint64_t> countedInputOriginalIds;
    for (const Tensor& tensor : featureInputs) {
        if (countedInputOriginalIds.insert(tensor.getOriginalId()).second) {
            totalBytes += tensor.getTotalSizeInBytes();
        }
    }
    for (const Tensor& tensor : featureOutputs)
        totalBytes += tensor.getTotalSizeInBytes();
    return totalBytes * std::max<uint32_t>(1, batchSize);
}

std::shared_ptr<ThorImplementation::Layer> CustomLayer::stamp(ThorImplementation::TensorPlacement placement,
                                                              std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                              std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                              Thor::Tensor connectingApiTensor) const {
    (void)drivingLayer;
    (void)drivingApiLayer;

    if (!inputBindingsByTensorOriginalId.contains(connectingApiTensor.getOriginalId())) {
        throw runtime_error("CustomLayer::stamp called with a tensor that is not one of its declared inputs.");
    }

    std::vector<std::shared_ptr<ThorImplementation::Parameter>> physicalParameters;
    physicalParameters.reserve(parameters.size());
    bool hasTrainableParameter = false;
    for (const auto& parameter : parameters) {
        if (parameter == nullptr)
            throw runtime_error("CustomLayer contains a null Parameter.");
        hasTrainableParameter |= parameter->isTrainable();
        physicalParameters.push_back(parameter->stamp());
    }

    auto physicalLayer = std::make_shared<ThorImplementation::CustomLayer>(
        expr, inputNames, outputNames, placement, physicalParameters, inferenceOnly, Layer::getId(), useFastMath);
    physicalLayer->setLayerName(getLayerType());
    if (hasTrainableParameter)
        stampOptimizer(physicalLayer);
    return physicalLayer;
}

json CustomLayer::architectureJson() const {
    json j;
    j["factory"] = Layer::Factory::Learning.value();
    j["version"] = "1.0.0";
    j["layer_type"] = "custom_layer";
    j["inference_only"] = inferenceOnly;
    j["input_names"] = inputNames;
    j["output_names"] = outputNames;
    j["input_interfaces"] = json::array();
    j["output_interfaces"] = json::array();
    j["parameters"] = json::array();

    for (const TensorMap& inputInterface : inputInterfaces) {
        json interfaceJson;
        for (const std::string& name : inputNames) {
            interfaceJson[name] = inputInterface.at(name).architectureJson();
        }
        j["input_interfaces"].push_back(interfaceJson);
    }

    for (const TensorMap& outputInterface : outputInterfaces) {
        json interfaceJson;
        for (const std::string& name : outputNames) {
            interfaceJson[name] = outputInterface.at(name).architectureJson();
        }
        j["output_interfaces"].push_back(interfaceJson);
    }

    for (const auto& parameter : parameters) {
        if (parameter != nullptr)
            j["parameters"].push_back(parameter->architectureJson());
    }

    // FIXME: Will need to serialize the expression.

    return j;
}

json CustomLayer::serialize(thor_file::TarWriter& archiveWriter,
                            Stream stream,
                            bool saveOptimizerState,
                            ThorImplementation::StampedNetwork& stampedNetwork) const {
    (void)archiveWriter;
    (void)stream;
    (void)saveOptimizerState;
    (void)stampedNetwork;

    // FIXME: Will need to serialize the parameters and sometimes optimizer parameters.

    return architectureJson();
}

}  // namespace Thor
