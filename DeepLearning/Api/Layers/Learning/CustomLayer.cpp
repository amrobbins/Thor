#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "DeepLearning/Implementation/Tensor/Tensor.h"

using namespace std;
using json = nlohmann::json;

using DynamicExpression = ThorImplementation::DynamicExpression;
using DataType = ThorImplementation::TensorDescriptor::DataType;
using PhysicalTensor = ThorImplementation::Tensor;
using PhysicalTensorMap = std::unordered_map<std::string, PhysicalTensor>;
using CompiledExecutionStage = ThorImplementation::CompiledExecutionStage;
using CompiledOutputs = ThorImplementation::CompiledOutputs;
using CompiledStageOutput = ThorImplementation::CompiledStageOutput;

namespace {

Optional<DataType> stageOutputDType(const CompiledExecutionStage& stage, size_t outputIdx) {
    switch (stage.kind) {
        case CompiledExecutionStage::Kind::FusedKernel:
            return stage.flat->output_dtypes.at(outputIdx);
        case CompiledExecutionStage::Kind::Reduction:
            return stage.reduction->output_dtype;
        case CompiledExecutionStage::Kind::ArgMinMax:
            return stage.arg_minmax->output_dtype;
        case CompiledExecutionStage::Kind::Matmul:
            return stage.matmul->output_dtype;
        case CompiledExecutionStage::Kind::Convolution:
            return stage.convolution->output_dtype;
        case CompiledExecutionStage::Kind::ConvolutionBackward:
            return stage.convolution_backward->output_dtype;
        case CompiledExecutionStage::Kind::ReduceMinMaxBackward:
            return stage.reduce_minmax_backward->output_dtype;
        case CompiledExecutionStage::Kind::Transpose:
            return stage.transpose->output_dtypes.at(outputIdx);
    }
    return Optional<DataType>::empty();
}

PhysicalTensor makeFakePlacedTensor(const Thor::Tensor& apiTensor) {
    std::vector<uint64_t> fakeDims;
    fakeDims.reserve(apiTensor.getDimensions().size() + 1);
    fakeDims.push_back(1);
    for (uint64_t dim : apiTensor.getDimensions()) {
        fakeDims.push_back(dim);
    }

    ThorImplementation::TensorPlacement placement(ThorImplementation::TensorPlacement::MemDevices::CPU, 0);
    ThorImplementation::TensorDescriptor descriptor(apiTensor.getDataType(), fakeDims);
    return PhysicalTensor(placement, descriptor);
}

Thor::Tensor logicalTensorFromFakeOutput(const std::vector<uint64_t>& fakeOutputDims, DataType dtype) {
    std::vector<uint64_t> logicalDims;
    if (!fakeOutputDims.empty()) {
        logicalDims.assign(fakeOutputDims.begin() + 1, fakeOutputDims.end());
    }
    return Thor::Tensor(dtype, logicalDims);
}

}  // namespace

namespace Thor {

CustomLayer::CustomLayer(DynamicExpression expr,
                         const std::vector<TensorMap>& inputInterfaces,
                         std::vector<std::shared_ptr<Parameter>> parameters,
                         bool useFastMath)
    : CustomLayer(std::move(expr), {}, {}, inputInterfaces, {}, std::move(parameters), useFastMath) {}

CustomLayer::CustomLayer(DynamicExpression expr,
                         std::vector<std::string> inputNames,
                         std::vector<std::string> outputNames,
                         const std::vector<TensorMap>& inputInterfaces,
                         const std::vector<TensorMap>& outputInterfaces,
                         std::vector<std::shared_ptr<Parameter>> parameters,
                         bool useFastMath)
    : expr(std::move(expr)), useFastMath(useFastMath), parameters(std::move(parameters)) {
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
    validateParameterNames();
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

void CustomLayer::validateParameterNames() const {
    std::set<std::string> seen;
    std::set<std::string> featureInputNames(inputNames.begin(), inputNames.end());
    for (const auto& parameter : parameters) {
        if (parameter == nullptr) {
            throw runtime_error("CustomLayer contains a null Parameter.");
        }

        const std::string& name = parameter->getName();
        if (!seen.insert(name).second) {
            throw runtime_error("CustomLayer received duplicate Parameter name '" + name + "'.");
        }
        if (featureInputNames.contains(name)) {
            throw runtime_error("CustomLayer Parameter name '" + name + "' conflicts with a feature input name.");
        }
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

CustomLayer::TensorMap CustomLayer::inferOutputInterfaceFromInputInterface(const TensorMap& inputInterface) const {
    PhysicalTensorMap fakeFeatureInputs;
    for (const auto& [name, apiTensor] : inputInterface) {
        fakeFeatureInputs.emplace(name, makeFakePlacedTensor(apiTensor));
    }

    ThorImplementation::Parameter::StorageContext fakeStorageContext(fakeFeatureInputs);
    PhysicalTensorMap fakeParameterTensors;
    for (const auto& apiParameter : parameters) {
        std::shared_ptr<ThorImplementation::Parameter> physicalParameter = apiParameter->stamp();
        physicalParameter->compileStorageAndOptimizer(fakeStorageContext, Optional<Stream>::empty(), true);
        Optional<PhysicalTensor> storage = physicalParameter->getStorage();
        if (!storage.isPresent()) {
            throw std::runtime_error("CustomLayer failed to infer parameter storage for '" + apiParameter->getName() + "'.");
        }
        fakeParameterTensors.emplace(apiParameter->getName(), storage.get());
    }

    PhysicalTensorMap fakeAllInputs = fakeFeatureInputs;
    for (const auto& [name, tensor] : fakeParameterTensors) {
        auto [it, inserted] = fakeAllInputs.emplace(name, tensor);
        if (!inserted) {
            throw std::runtime_error("CustomLayer Parameter name '" + name + "' conflicts with a feature input name.");
        }
    }

    Stream fakeStream(0, Stream::Priority::REGULAR);
    ThorImplementation::DynamicExpressionBuild build = expr.build(fakeAllInputs, {}, fakeStream);

    std::unordered_map<std::string, std::vector<uint64_t>> fakeOutputShapes = build.equation->getOutputShapes(build.stamp_inputs);
    std::shared_ptr<CompiledOutputs> compiledOutputs = build.equation->compileForInputs(build.stamp_inputs, {}, build.tensor_scalar_inputs);

    std::unordered_map<uint32_t, DataType> outputDTypeByValueId;
    for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
        for (size_t outputIdx = 0; outputIdx < stage.outputs.size(); ++outputIdx) {
            outputDTypeByValueId.emplace(stage.outputs[outputIdx].value_id, stageOutputDType(stage, outputIdx).get());
        }
    }

    TensorMap inferredOutputs;
    for (const CompiledStageOutput& finalOutput : compiledOutputs->final_outputs) {
        auto shapeIt = fakeOutputShapes.find(finalOutput.name);
        if (shapeIt == fakeOutputShapes.end()) {
            throw std::runtime_error("CustomLayer failed to infer output shape for '" + finalOutput.name + "'.");
        }
        auto dtypeIt = outputDTypeByValueId.find(finalOutput.value_id);
        if (dtypeIt == outputDTypeByValueId.end()) {
            throw std::runtime_error("CustomLayer failed to infer output dtype for '" + finalOutput.name + "'.");
        }
        inferredOutputs.emplace(finalOutput.name, logicalTensorFromFakeOutput(shapeIt->second, dtypeIt->second));
    }

    return inferredOutputs;
}

void CustomLayer::materializeOutputInterfacesFromInputInterfaces() {
    TensorMap inferredOutputs = inferOutputInterfaceFromInputInterface(inputInterfaces.front());

    std::vector<TensorMap> materialized;
    materialized.reserve(inputInterfaces.size());
    for (size_t interfaceIndex = 0; interfaceIndex < inputInterfaces.size(); ++interfaceIndex) {
        (void)interfaceIndex;
        TensorMap outputInterface;
        for (const std::string& outputName : outputNames) {
            auto it = inferredOutputs.find(outputName);
            if (it == inferredOutputs.end()) {
                throw std::runtime_error("CustomLayer failed to infer output tensor for '" + outputName + "'.");
            }
            outputInterface[outputName] = it->second.clone();
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

CustomLayer::TensorMap CustomLayer::getInputInterface(uint32_t interfaceIndex) const {
    if (interfaceIndex >= inputInterfaces.size()) {
        throw runtime_error("CustomLayer input interface index out of range.");
    }
    return inputInterfaces[interfaceIndex];
}

CustomLayer::TensorMap CustomLayer::getOutputInterfaceByIndex(uint32_t interfaceIndex) const {
    if (interfaceIndex >= outputInterfaces.size()) {
        throw runtime_error("CustomLayer output interface index out of range.");
    }
    return outputInterfaces[interfaceIndex];
}

Tensor CustomLayer::getOutput(const std::string& outputName, uint32_t interfaceIndex) const {
    if (interfaceIndex >= outputInterfaces.size()) {
        throw runtime_error("CustomLayer output interface index out of range.");
    }

    const TensorMap& outputInterface = outputInterfaces[interfaceIndex];
    auto found = outputInterface.find(outputName);
    if (found == outputInterface.end()) {
        throw runtime_error("CustomLayer has no output named '" + outputName + "'.");
    }
    return found->second;
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
                                                              Thor::Tensor connectingApiTensor,
                                                              const bool inferenceOnly) const {
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
