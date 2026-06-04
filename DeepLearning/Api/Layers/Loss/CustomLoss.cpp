#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"

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

CustomLoss::CustomLoss(ThorImplementation::DynamicExpression lossExpression,
                       ThorImplementation::DynamicExpression gradientExpression,
                       Tensor predictions,
                       Tensor labels,
                       std::string predictionsName,
                       std::string labelsName,
                       std::string lossName,
                       std::string gradientName,
                       std::optional<Tensor> lossTensor,
                       std::optional<DataType> requestedLossDataType)
    : lossExpression(std::move(lossExpression)),
      gradientExpression(std::move(gradientExpression)),
      predictionsName(std::move(predictionsName)),
      labelsName(std::move(labelsName)),
      lossName(std::move(lossName)),
      gradientName(std::move(gradientName)) {
    validateName(this->predictionsName, "predictions input");
    validateName(this->labelsName, "labels input");
    validateName(this->lossName, "loss output");
    validateName(this->gradientName, "gradient output");
    if (this->predictionsName == this->labelsName)
        throw runtime_error("CustomLoss predictions and labels input names must be distinct.");

    predictionsTensor = std::move(predictions);
    labelsTensor = std::move(labels);

    validateExpressionNames(this->lossExpression, this->lossName, "loss");
    validateExpressionNames(this->gradientExpression, this->gradientName, "gradient");

    Tensor inferredLossTensor = inferLossTensor();
    lossDataType = requestedLossDataType.value_or(inferredLossTensor.getDataType());
    THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32);
    if (inferredLossTensor.getDataType() != lossDataType) {
        throw runtime_error("CustomLoss loss_data_type must match the expression output dtype. Expected " +
                            inferredLossTensor.getDescriptorString() + ".");
    }

    if (lossTensor.has_value()) {
        if (lossTensor.value().getDataType() != inferredLossTensor.getDataType() ||
            lossTensor.value().getDimensions() != inferredLossTensor.getDimensions()) {
            throw runtime_error("CustomLoss loss tensor must match the expression output descriptor. Expected " +
                                inferredLossTensor.getDescriptorString() + ", got " + lossTensor.value().getDescriptorString() + ".");
        }
        this->lossTensor = lossTensor.value();
    } else {
        this->lossTensor = inferredLossTensor;
    }
    lossShaperInput = this->lossTensor;
    lossShape = LossShape::RAW;
    validateGradientTensor();
    initialized = true;
}

void CustomLoss::validateName(const std::string& name, const std::string& what) {
    if (name.empty())
        throw runtime_error("CustomLoss " + what + " name cannot be empty.");
    if (name.length() >= 2 && name[0] == '_' && name[1] == '_')
        throw runtime_error("CustomLoss " + what + " names cannot start with __ that is reserved. Name " + name + " is illegal.");
}

std::set<std::string> CustomLoss::toNameSet(const std::vector<std::string>& names) { return std::set<std::string>(names.begin(), names.end()); }

std::string CustomLoss::joinNames(const std::set<std::string>& names) {
    if (names.empty())
        return "<none>";

    std::ostringstream oss;
    bool first = true;
    for (const std::string& name : names) {
        if (!first)
            oss << ", ";
        oss << name;
        first = false;
    }
    return oss.str();
}

CustomLoss::PhysicalTensor CustomLoss::makeFakePlacedTensor(const Tensor& apiTensor) {
    std::vector<uint64_t> fakeDims;
    fakeDims.reserve(apiTensor.getDimensions().size() + 1);
    fakeDims.push_back(1);
    for (uint64_t dim : apiTensor.getDimensions())
        fakeDims.push_back(dim);

    ThorImplementation::TensorPlacement placement(ThorImplementation::TensorPlacement::MemDevices::CPU, 0);
    ThorImplementation::TensorDescriptor descriptor(apiTensor.getDataType(), fakeDims);
    return PhysicalTensor(placement, descriptor);
}

Tensor CustomLoss::logicalLossTensorFromFakeOutput(const std::vector<uint64_t>& fakeOutputDims, DataType dtype) {
    std::vector<uint64_t> logicalDims = fakeOutputDims;
    if (logicalDims.size() >= 2 && logicalDims.front() == 1)
        logicalDims.erase(logicalDims.begin());
    if (logicalDims.empty())
        logicalDims.push_back(1);
    return Tensor(dtype, logicalDims);
}

DataType CustomLoss::findOutputDType(const std::shared_ptr<CompiledOutputs>& compiledOutputs, const std::string& outputName) {
    std::optional<DataType> outputDType;
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
        throw runtime_error("CustomLoss expression did not infer output dtype for '" + outputName + "'.");
    return outputDType.value();
}

void CustomLoss::validateExpressionNames(const ThorImplementation::DynamicExpression& expression,
                                         const std::string& outputName,
                                         const std::string& what) const {
    const std::vector<std::string>& expectedInputs = expression.getExpectedInputNames();
    if (!expectedInputs.empty()) {
        std::set<std::string> expected{predictionsName, labelsName};
        std::set<std::string> actual(expectedInputs.begin(), expectedInputs.end());
        if (actual != expected) {
            throw runtime_error("CustomLoss " + what + " expression input name mismatch. Expected {" + joinNames(expected) +
                                "}, got {" + joinNames(actual) + "}.");
        }
    }

    const std::vector<std::string>& expectedOutputs = expression.getExpectedOutputNames();
    if (!expectedOutputs.empty()) {
        std::set<std::string> expected{outputName};
        std::set<std::string> actual(expectedOutputs.begin(), expectedOutputs.end());
        if (actual != expected) {
            throw runtime_error("CustomLoss " + what + " expression output name mismatch. Expected {" + joinNames(expected) +
                                "}, got {" + joinNames(actual) + "}.");
        }
    }
}

Tensor CustomLoss::inferExpressionTensor(const ThorImplementation::DynamicExpression& expression,
                                         const std::string& outputName,
                                         const std::string& what) const {
    PhysicalTensorMap fakeInputs;
    fakeInputs.emplace(predictionsName, makeFakePlacedTensor(getPredictions()));
    fakeInputs.emplace(labelsName, makeFakePlacedTensor(getLabels()));

    Stream fakeStream(0, Stream::Priority::REGULAR);
    ThorImplementation::DynamicExpressionBuild build = expression.build(fakeInputs, {}, fakeStream);

    const std::set<std::string> actualOutputNames = toNameSet(build.equation->getOutputNames());
    const std::set<std::string> expectedOutputNames{outputName};
    if (actualOutputNames != expectedOutputNames) {
        throw runtime_error("CustomLoss " + what + " expression output name mismatch. Expected {" + joinNames(expectedOutputNames) +
                            "}, got {" + joinNames(actualOutputNames) + "}.");
    }

    std::unordered_map<std::string, std::vector<uint64_t>> fakeOutputShapes =
        build.equation->getOutputShapes(build.stamp_inputs, build.tensor_scalar_inputs);
    auto shapeIt = fakeOutputShapes.find(outputName);
    if (shapeIt == fakeOutputShapes.end())
        throw runtime_error("CustomLoss failed to infer output shape for '" + outputName + "'.");

    std::shared_ptr<CompiledOutputs> compiledOutputs = build.equation->compileForInputs(build.stamp_inputs, {}, build.tensor_scalar_inputs);
    return logicalLossTensorFromFakeOutput(shapeIt->second, findOutputDType(compiledOutputs, outputName));
}

Tensor CustomLoss::inferLossTensor() const { return inferExpressionTensor(lossExpression, lossName, "loss"); }

void CustomLoss::validateGradientTensor() const {
    Tensor inferredGradientTensor = inferExpressionTensor(gradientExpression, gradientName, "gradient");
    if (inferredGradientTensor.getDataType() != predictionsTensor.getDataType() ||
        inferredGradientTensor.getDimensions() != predictionsTensor.getDimensions()) {
        throw runtime_error("CustomLoss gradient expression output must match predictions. Expected " +
                            predictionsTensor.getDescriptorString() + ", got " + inferredGradientTensor.getDescriptorString() + ".");
    }
}

std::shared_ptr<ThorImplementation::Layer> CustomLoss::stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor,
                                                             const bool inferenceOnly) const {
    (void)placement;
    (void)drivingLayer;
    (void)drivingApiLayer;
    (void)inferenceOnly;
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(connectingApiTensor == predictionsTensor || connectingApiTensor == labelsTensor);

    return std::make_shared<ThorImplementation::CustomLoss>(
        lossExpression, gradientExpression, predictionsName, labelsName, lossName, gradientName, lossDataType);
}

void CustomLoss::buildSupportLayersAndAddToNetwork() {
    CustomLoss rawLoss = CustomLoss::Builder()
                             .network(*network)
                             .lossExpression(lossExpression)
                             .gradientExpression(gradientExpression)
                             .predictions(predictionsTensor)
                             .labels(labelsTensor)
                             .predictionsName(predictionsName)
                             .labelsName(labelsName)
                             .lossName(lossName)
                             .gradientName(gradientName)
                             .lossDataType(lossDataType)
                             .reportsRawLoss()
                             .build();

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

uint64_t CustomLoss::getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                           ThorImplementation::TensorPlacement tensorPlacement) const {
    uint64_t lossShaperBytes = 0;
    if (isMultiLayer()) {
        lossShaperBytes = LossShaper::Builder()
                              .lossInput(lossTensor)
                              .reportsBatchLoss()
                              .getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }

    uint64_t standardLossBytes = Loss::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    return standardLossBytes + lossShaperBytes;
}

json CustomLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "custom_loss";
    j["predictions_name"] = predictionsName;
    j["labels_name"] = labelsName;
    j["loss_name"] = lossName;
    j["gradient_name"] = gradientName;

    auto serializedLossDefinition = lossExpression.getSerializedDefinition();
    if (serializedLossDefinition == nullptr) {
        throw runtime_error(
            "CustomLoss loss expression is not serializable. Construct it from a ThorImplementation::ExpressionDefinition or "
            "DynamicExpression::fromExpressionDefinition(...). Arbitrary DynamicExpression builders cannot be saved.");
    }
    auto serializedGradientDefinition = gradientExpression.getSerializedDefinition();
    if (serializedGradientDefinition == nullptr) {
        throw runtime_error(
            "CustomLoss gradient expression is not serializable. Construct it from a ThorImplementation::ExpressionDefinition or "
            "DynamicExpression::fromExpressionDefinition(...). Arbitrary DynamicExpression builders cannot be saved.");
    }
    j["loss_expression"] = serializedLossDefinition->architectureJson();
    j["gradient_expression"] = serializedGradientDefinition->architectureJson();
    return j;
}

void CustomLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CustomLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "custom_loss")
        throw runtime_error("Layer type mismatch in CustomLoss::deserialize: " + j.at("layer_type").get<std::string>());

    THOR_THROW_IF_FALSE(j.at("loss_shape").get<LossShape>() == LossShape::RAW);

    const std::string predictionsName = j.value("predictions_name", std::string("predictions"));
    const std::string labelsName = j.value("labels_name", std::string("labels"));
    const std::string lossName = j.value("loss_name", std::string("loss"));
    const std::string gradientName = j.value("gradient_name", predictionsName + "_grad");

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    Tensor rawLossTensor = Tensor::deserialize(j["loss_shaper_input_tensor"]);

    ThorImplementation::ExpressionDefinition lossDefinition = ThorImplementation::ExpressionDefinition::deserialize(
        j.at("loss_expression"),
        network != nullptr && network->allowUnsafeLoadedCudaKernelSourceCompilation(),
        network != nullptr ? network->trustedLoadedCudaKernelPublicKey() : std::string{},
        network != nullptr ? network->trustedLoadedCudaKernelSourceDecryptionKey() : std::string{});
    ThorImplementation::ExpressionDefinition gradientDefinition = ThorImplementation::ExpressionDefinition::deserialize(
        j.at("gradient_expression"),
        network != nullptr && network->allowUnsafeLoadedCudaKernelSourceCompilation(),
        network != nullptr ? network->trustedLoadedCudaKernelPublicKey() : std::string{},
        network != nullptr ? network->trustedLoadedCudaKernelSourceDecryptionKey() : std::string{});

    CustomLoss customLoss(ThorImplementation::DynamicExpression::fromExpressionDefinition(lossDefinition),
                          ThorImplementation::DynamicExpression::fromExpressionDefinition(gradientDefinition),
                          predictions,
                          labels,
                          predictionsName,
                          labelsName,
                          lossName,
                          gradientName,
                          rawLossTensor,
                          j.at("loss_data_type").get<DataType>());
    customLoss.lossShape = LossShape::RAW;
    customLoss.addToNetwork(network);
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Loss::register_layer("custom_loss", &Thor::CustomLoss::deserialize);
    return true;
}();
}  // namespace
