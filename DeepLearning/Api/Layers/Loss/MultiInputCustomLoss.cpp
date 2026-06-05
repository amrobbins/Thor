#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"

#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/FusedEquation.h"

#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace std;
using json = nlohmann::json;

namespace Thor {

using CompiledOutputs = ThorImplementation::CompiledOutputs;
using CompiledExecutionStage = ThorImplementation::CompiledExecutionStage;
using CompiledStageOutput = ThorImplementation::CompiledStageOutput;

MultiInputCustomLoss::MultiInputCustomLoss(ThorImplementation::DynamicExpression lossExpression,
                                           ThorImplementation::DynamicExpression gradientExpression,
                                           vector<InputSpec> inputs,
                                           string lossName,
                                           optional<Tensor> lossTensor,
                                           optional<DataType> requestedLossDataType)
    : lossExpression(std::move(lossExpression)),
      gradientExpression(std::move(gradientExpression)),
      inputs(std::move(inputs)),
      lossName(std::move(lossName)) {
    validateName(this->lossName, "loss output");
    validateInputSpecs();

    set<string> gradientNames;
    for (const InputSpec& input : this->inputs)
        gradientNames.insert(input.gradientName);

    validateExpressionNames(this->lossExpression, {this->lossName}, "loss");
    validateExpressionNames(this->gradientExpression, gradientNames, "gradient");

    Tensor inferredLossTensor = inferLossTensor();
    lossDataType = requestedLossDataType.value_or(inferredLossTensor.getDataType());
    THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32);
    if (inferredLossTensor.getDataType() != lossDataType) {
        throw runtime_error("MultiInputCustomLoss loss_data_type must match the expression output dtype. Expected " +
                            inferredLossTensor.getDescriptorString() + ".");
    }

    if (lossTensor.has_value()) {
        if (lossTensor.value().getDataType() != inferredLossTensor.getDataType() ||
            lossTensor.value().getDimensions() != inferredLossTensor.getDimensions()) {
            throw runtime_error("MultiInputCustomLoss loss tensor must match the expression output descriptor. Expected " +
                                inferredLossTensor.getDescriptorString() + ", got " + lossTensor.value().getDescriptorString() + ".");
        }
        this->lossTensor = lossTensor.value();
    } else {
        this->lossTensor = inferredLossTensor;
    }
    lossShaperInput = this->lossTensor;
    lossShape = LossShape::RAW;
    validateGradientTensors();
    initialized = true;
}

void MultiInputCustomLoss::validateName(const string& name, const string& what) {
    if (name.empty())
        throw runtime_error("MultiInputCustomLoss " + what + " name cannot be empty.");
    if (name.length() >= 2 && name[0] == '_' && name[1] == '_')
        throw runtime_error("MultiInputCustomLoss " + what + " names cannot start with __ that is reserved. Name " + name + " is illegal.");
}

set<string> MultiInputCustomLoss::toNameSet(const vector<string>& names) { return set<string>(names.begin(), names.end()); }

string MultiInputCustomLoss::joinNames(const set<string>& names) {
    if (names.empty())
        return "<none>";

    ostringstream oss;
    bool first = true;
    for (const string& name : names) {
        if (!first)
            oss << ", ";
        oss << name;
        first = false;
    }
    return oss.str();
}

MultiInputCustomLoss::PhysicalTensor MultiInputCustomLoss::makeFakePlacedTensor(const Tensor& apiTensor) {
    vector<uint64_t> fakeDims;
    fakeDims.reserve(apiTensor.getDimensions().size() + 1);
    fakeDims.push_back(1);
    for (uint64_t dim : apiTensor.getDimensions())
        fakeDims.push_back(dim);

    ThorImplementation::TensorPlacement placement(ThorImplementation::TensorPlacement::MemDevices::CPU, 0);
    ThorImplementation::TensorDescriptor descriptor(apiTensor.getDataType(), fakeDims);
    return PhysicalTensor(placement, descriptor);
}

Tensor MultiInputCustomLoss::logicalLossTensorFromFakeOutput(const vector<uint64_t>& fakeOutputDims, DataType dtype) {
    vector<uint64_t> logicalDims = fakeOutputDims;
    if (logicalDims.size() >= 2 && logicalDims.front() == 1)
        logicalDims.erase(logicalDims.begin());
    if (logicalDims.empty())
        logicalDims.push_back(1);
    return Tensor(dtype, logicalDims);
}

DataType MultiInputCustomLoss::findOutputDType(const shared_ptr<CompiledOutputs>& compiledOutputs, const string& outputName) {
    optional<DataType> outputDType;
    for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
        for (size_t outputIndex = 0; outputIndex < stage.outputs.size(); ++outputIndex) {
            const CompiledStageOutput& output = stage.outputs[outputIndex];
            if (output.name == outputName) {
                outputDType = stage.outputDType(outputIndex);
                break;
            }
        }
        if (outputDType.has_value())
            break;
    }

    if (!outputDType.has_value()) {
        for (const CompiledStageOutput& finalOutput : compiledOutputs->final_outputs) {
            if (finalOutput.name != outputName)
                continue;
            for (const CompiledExecutionStage& stage : compiledOutputs->stages) {
                for (size_t outputIndex = 0; outputIndex < stage.outputs.size(); ++outputIndex) {
                    if (stage.outputs[outputIndex].value_id == finalOutput.value_id) {
                        outputDType = stage.outputDType(outputIndex);
                        break;
                    }
                }
                if (outputDType.has_value())
                    break;
            }
        }
    }

    if (!outputDType.has_value())
        throw runtime_error("MultiInputCustomLoss expression did not infer output dtype for '" + outputName + "'.");
    return outputDType.value();
}

void MultiInputCustomLoss::validateInputSpecs() const {
    if (inputs.empty())
        throw runtime_error("MultiInputCustomLoss requires at least one differentiable input.");

    set<string> names;
    set<string> gradientNames;
    set<Tensor> tensors;
    for (const InputSpec& input : inputs) {
        validateName(input.name, "input");
        validateName(input.gradientName, "gradient output");
        if (!input.tensor.isInitialized())
            throw runtime_error("MultiInputCustomLoss input '" + input.name + "' tensor is not initialized.");
        if (!names.insert(input.name).second)
            throw runtime_error("MultiInputCustomLoss input name '" + input.name + "' is duplicated.");
        if (!gradientNames.insert(input.gradientName).second)
            throw runtime_error("MultiInputCustomLoss gradient output name '" + input.gradientName + "' is duplicated.");
        if (!tensors.insert(input.tensor).second)
            throw runtime_error("MultiInputCustomLoss input tensor is used by more than one named input; duplicate tensors are ambiguous.");
    }
}

void MultiInputCustomLoss::validateExpressionNames(const ThorImplementation::DynamicExpression& expression,
                                                   const set<string>& outputNames,
                                                   const string& what) const {
    const vector<string>& expectedInputs = expression.getExpectedInputNames();
    if (!expectedInputs.empty()) {
        set<string> expected;
        for (const InputSpec& input : inputs)
            expected.insert(input.name);
        set<string> actual(expectedInputs.begin(), expectedInputs.end());
        if (actual != expected) {
            throw runtime_error("MultiInputCustomLoss " + what + " expression input name mismatch. Expected {" + joinNames(expected) +
                                "}, got {" + joinNames(actual) + "}.");
        }
    }

    const vector<string>& expectedOutputs = expression.getExpectedOutputNames();
    if (!expectedOutputs.empty()) {
        set<string> actual(expectedOutputs.begin(), expectedOutputs.end());
        if (actual != outputNames) {
            throw runtime_error("MultiInputCustomLoss " + what + " expression output name mismatch. Expected {" + joinNames(outputNames) +
                                "}, got {" + joinNames(actual) + "}.");
        }
    }
}

Tensor MultiInputCustomLoss::inferExpressionTensor(const ThorImplementation::DynamicExpression& expression,
                                                   const string& outputName,
                                                   const string& what) const {
    PhysicalTensorMap fakeInputs;
    for (const InputSpec& input : inputs)
        fakeInputs.emplace(input.name, makeFakePlacedTensor(input.tensor));

    Stream fakeStream(0, Stream::Priority::REGULAR);
    ThorImplementation::DynamicExpressionBuild build = expression.build(fakeInputs, {}, fakeStream);

    const set<string> actualOutputNames = toNameSet(build.equation->getOutputNames());
    if (actualOutputNames.count(outputName) == 0) {
        throw runtime_error("MultiInputCustomLoss " + what + " expression did not provide output '" + outputName + "'. Got {" +
                            joinNames(actualOutputNames) + "}.");
    }

    unordered_map<string, vector<uint64_t>> fakeOutputShapes = build.equation->getOutputShapes(build.stamp_inputs, build.tensor_scalar_inputs);
    auto shapeIt = fakeOutputShapes.find(outputName);
    if (shapeIt == fakeOutputShapes.end())
        throw runtime_error("MultiInputCustomLoss failed to infer output shape for '" + outputName + "'.");

    shared_ptr<CompiledOutputs> compiledOutputs = build.equation->compileForInputs(build.stamp_inputs, {}, build.tensor_scalar_inputs);
    return logicalLossTensorFromFakeOutput(shapeIt->second, findOutputDType(compiledOutputs, outputName));
}

Tensor MultiInputCustomLoss::inferLossTensor() const { return inferExpressionTensor(lossExpression, lossName, "loss"); }

void MultiInputCustomLoss::validateGradientTensors() const {
    for (const InputSpec& input : inputs) {
        Tensor inferredGradientTensor = inferExpressionTensor(gradientExpression, input.gradientName, "gradient");
        if (inferredGradientTensor.getDataType() != input.tensor.getDataType() ||
            inferredGradientTensor.getDimensions() != input.tensor.getDimensions()) {
            throw runtime_error("MultiInputCustomLoss gradient expression output '" + input.gradientName +
                                "' must match input '" + input.name + "'. Expected " + input.tensor.getDescriptorString() + ", got " +
                                inferredGradientTensor.getDescriptorString() + ".");
        }
    }
}

vector<Tensor> MultiInputCustomLoss::getLossInputTensors() const {
    vector<Tensor> tensors;
    tensors.reserve(inputs.size());
    for (const InputSpec& input : inputs)
        tensors.push_back(input.tensor);
    return tensors;
}

Tensor MultiInputCustomLoss::getPredictions() const {
    THOR_THROW_IF_FALSE(!inputs.empty());
    return inputs.front().tensor;
}

Tensor MultiInputCustomLoss::getLabels() const {
    throw runtime_error("MultiInputCustomLoss does not have labels. Use getInputs() / getLossInputTensors() instead.");
}

optional<Tensor> MultiInputCustomLoss::getFeatureInput() const {
    THOR_THROW_IF_FALSE(!inputs.empty());
    return inputs.front().tensor;
}

int MultiInputCustomLoss::getConnectionType(Tensor connectingTensor) const {
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        if (connectingTensor == inputs[i].tensor)
            return static_cast<int>(i);
    }
    if (connectingTensor == lossTensor)
        return 0;
    throw runtime_error("Tensor is not connected to this MultiInputCustomLoss.");
}

shared_ptr<ThorImplementation::Layer> MultiInputCustomLoss::stamp(ThorImplementation::TensorPlacement placement,
                                                                  shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                                  shared_ptr<Thor::Layer> drivingApiLayer,
                                                                  Thor::Tensor connectingApiTensor,
                                                                  const bool inferenceOnly) const {
    (void)placement;
    (void)drivingLayer;
    (void)drivingApiLayer;
    THOR_THROW_IF_FALSE(initialized);
    bool isInputTensor = false;
    for (const InputSpec& input : inputs)
        isInputTensor = isInputTensor || connectingApiTensor == input.tensor;
    THOR_THROW_IF_FALSE(isInputTensor);

    vector<string> inputNames;
    vector<string> gradientNames;
    inputNames.reserve(inputs.size());
    gradientNames.reserve(inputs.size());
    for (const InputSpec& input : inputs) {
        inputNames.push_back(input.name);
        gradientNames.push_back(input.gradientName);
    }

    shared_ptr<ThorImplementation::MultiInputCustomLoss> customLoss = make_shared<ThorImplementation::MultiInputCustomLoss>(
        lossExpression, gradientExpression, inputNames, gradientNames, lossName, lossDataType);
    customLoss->setConstructForInferenceOnly(inferenceOnly);
    return customLoss;
}

void MultiInputCustomLoss::buildSupportLayersAndAddToNetwork() {
    MultiInputCustomLoss::Builder builder;
    builder.network(*network)
        .lossExpression(lossExpression)
        .gradientExpression(gradientExpression)
        .lossName(lossName)
        .lossDataType(lossDataType)
        .reportsRawLoss();
    for (const InputSpec& input : inputs)
        builder.input(input.name, input.tensor, input.gradientName);

    MultiInputCustomLoss rawLoss = builder.build();
    lossShaperInput = rawLoss.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

uint64_t MultiInputCustomLoss::getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                                     ThorImplementation::TensorPlacement tensorPlacement) const {
    (void)tensorPlacement;
    uint64_t bytes = 4;
    for (const InputSpec& input : inputs) {
        bytes += batchSize * input.tensor.getTotalSizeInBytes();
        bytes += batchSize * input.tensor.getTotalSizeInBytes();
    }
    bytes += batchSize * lossTensor.getTotalSizeInBytes();
    return bytes;
}

json MultiInputCustomLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "multi_input_custom_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["loss_name"] = lossName;
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();

    j["inputs"] = json::array();
    for (const InputSpec& input : inputs) {
        json inputJson;
        inputJson["name"] = input.name;
        inputJson["gradient_name"] = input.gradientName;
        inputJson["tensor"] = input.tensor.architectureJson();
        j["inputs"].push_back(inputJson);
    }

    auto serializedLossDefinition = lossExpression.getSerializedDefinition();
    if (serializedLossDefinition == nullptr) {
        throw runtime_error(
            "MultiInputCustomLoss loss expression is not serializable. Construct it from a ThorImplementation::ExpressionDefinition or "
            "DynamicExpression::fromExpressionDefinition(...). Arbitrary DynamicExpression builders cannot be saved.");
    }
    auto serializedGradientDefinition = gradientExpression.getSerializedDefinition();
    if (serializedGradientDefinition == nullptr) {
        throw runtime_error(
            "MultiInputCustomLoss gradient expression is not serializable. Construct it from a ThorImplementation::ExpressionDefinition or "
            "DynamicExpression::fromExpressionDefinition(...). Arbitrary DynamicExpression builders cannot be saved.");
    }
    j["loss_expression"] = serializedLossDefinition->architectureJson();
    j["gradient_expression"] = serializedGradientDefinition->architectureJson();
    return j;
}

void MultiInputCustomLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in MultiInputCustomLoss::deserialize: " + j["version"].get<string>());
    if (j.at("layer_type").get<string>() != "multi_input_custom_loss")
        throw runtime_error("Layer type mismatch in MultiInputCustomLoss::deserialize: " + j.at("layer_type").get<string>());

    THOR_THROW_IF_FALSE(j.at("loss_shape").get<LossShape>() == LossShape::RAW);

    vector<InputSpec> inputs;
    for (const json& inputJson : j.at("inputs")) {
        uint64_t originalTensorId = inputJson.at("tensor").at("id").get<uint64_t>();
        Tensor tensor = network->getApiTensorByOriginalId(originalTensorId);
        inputs.push_back(InputSpec{inputJson.at("name").get<string>(), tensor, inputJson.at("gradient_name").get<string>()});
    }

    Tensor rawLossTensor = Tensor::deserialize(j["loss_shaper_input_tensor"]);

    ThorImplementation::ExpressionDefinition lossDefinition = ThorImplementation::ExpressionDefinition::deserialize(
        j.at("loss_expression"),
        network != nullptr && network->allowUnsafeLoadedCudaKernelSourceCompilation(),
        network != nullptr ? network->trustedLoadedCudaKernelPublicKey() : string{},
        network != nullptr ? network->trustedLoadedCudaKernelSourceDecryptionKey() : string{});
    ThorImplementation::ExpressionDefinition gradientDefinition = ThorImplementation::ExpressionDefinition::deserialize(
        j.at("gradient_expression"),
        network != nullptr && network->allowUnsafeLoadedCudaKernelSourceCompilation(),
        network != nullptr ? network->trustedLoadedCudaKernelPublicKey() : string{},
        network != nullptr ? network->trustedLoadedCudaKernelSourceDecryptionKey() : string{});

    MultiInputCustomLoss customLoss(ThorImplementation::DynamicExpression::fromExpressionDefinition(lossDefinition),
                                    ThorImplementation::DynamicExpression::fromExpressionDefinition(gradientDefinition),
                                    inputs,
                                    j.value("loss_name", string("loss")),
                                    rawLossTensor,
                                    j.at("loss_data_type").get<DataType>());
    customLoss.lossShape = LossShape::RAW;
    customLoss.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Loss::register_layer("multi_input_custom_loss", &Thor::MultiInputCustomLoss::deserialize);
    return true;
}();
}  // namespace
