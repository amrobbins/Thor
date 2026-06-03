#include "DeepLearning/Implementation/Layers/Loss/BinaryCrossEntropy.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <stdexcept>
#include <unordered_map>

using namespace ThorImplementation;
using namespace std;

namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";

}  // namespace

BinaryCrossEntropy::BinaryCrossEntropy(DataType lossDataType) : Loss(lossDataType) {}

void BinaryCrossEntropy::validateLabelsDType(DataType dtype) {
    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::UINT8:
        case DataType::UINT16:
        case DataType::UINT32:
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::FP16:
        case DataType::FP32:
            return;
        default:
            throw runtime_error("Unsupported BinaryCrossEntropy label dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

Outputs BinaryCrossEntropy::makeForwardOutputs(DataType lossDataType) {
    Expression logits = Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    Expression labels = Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    Expression zero(0.0);

    // Numerically stable BCE-with-logits:
    //   max(x, 0) - x * y + log1p(exp(-abs(x)))
    Expression loss = (logits.max(zero) - (logits * labels) + (-logits.abs()).exp().log1p()).withOutputDType(lossDataType);
    return Expression::outputs({{kLossName, loss}});
}

Outputs BinaryCrossEntropy::makeGradientOutputs(DataType predictionDataType) {
    Expression logits = Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    Expression labels = Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    Expression grad = ((logits.sigmoid() - labels) * Expression(Loss::getLossScalingFactor())).withOutputDType(predictionDataType);
    return Expression::outputs({{kGradientName, grad}});
}

void BinaryCrossEntropy::compileImpl() {
    Layer::compileImpl();

    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.value().isInitialized());
    THOR_THROW_IF_FALSE(featureOutput.value().isInitialized());
    THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureOutput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == featureOutput.value().getPlacement());
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == labelsInput.value().getPlacement());
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == featureOutput.value().getDescriptor().getDimensions());
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == labelsInput.value().getDescriptor().getDimensions());

    const DataType predictionDType = featureInput.value().getDescriptor().getDataType();
    THOR_THROW_IF_FALSE(predictionDType == DataType::FP16 || predictionDType == DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == DataType::FP16 ||
           featureOutput.value().getDescriptor().getDataType() == DataType::FP32);
    validateLabelsDType(labelsInput.value().getDescriptor().getDataType());

    const int gpuNum = featureInput.value().getPlacement().getDeviceNum();
    const unordered_map<string, Tensor> inputs{{kPredictionsName, featureInput.value()}, {kLabelsName, labelsInput.value()}};

    FusedEquation forwardEquation = FusedEquation::compile(makeForwardOutputs(lossDataType).physicalOutputs(), gpuNum);
    forwardPlan = make_unique<StampedExecutionPlan>(
        forwardEquation.stamp(inputs, stream, {}, unordered_map<string, Tensor>{{kLossName, featureOutput.value()}}));

    if (!isInferenceOnly()) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement() == featureInput.value().getPlacement());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor().getDimensions() == featureInput.value().getDescriptor().getDimensions());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor().getDataType() == predictionDType);

        FusedEquation gradientEquation = FusedEquation::compile(makeGradientOutputs(predictionDType).physicalOutputs(), gpuNum);
        gradientPlan = make_unique<StampedExecutionPlan>(
            gradientEquation.stamp(inputs, stream, {}, unordered_map<string, Tensor>{{kGradientName, errorOutput.value()}}));
    } else {
        gradientPlan.reset();
    }
}

void BinaryCrossEntropy::cleanup() {
    forwardPlan.reset();
    gradientPlan.reset();
    Layer::cleanup();
}

void BinaryCrossEntropy::infer(std::optional<Tensor> predictions, std::optional<Tensor> loss, Stream runStream) {
    THOR_THROW_IF_FALSE(predictions.has_value());
    THOR_THROW_IF_FALSE(loss.has_value());
    THOR_THROW_IF_FALSE(predictions.value() == featureInput.value());
    THOR_THROW_IF_FALSE(loss.value() == featureOutput.value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(forwardPlan != nullptr);

    runStream.waitEvent(labelsStream.putEvent());
    forwardPlan->run();
    if (gradientPlan != nullptr) {
        // Loss layers originate backpropagation, so make the prediction gradient ready during forward.
        gradientPlan->run();
    }
    labelsStream.waitEvent(runStream.putEvent());
}

void BinaryCrossEntropy::backProp(std::optional<Tensor> labels,
                                  std::optional<Tensor> predictions,
                                  std::optional<Tensor> lossGradient,
                                  Stream runStream) {
    THOR_THROW_IF_FALSE(labels.has_value());
    THOR_THROW_IF_FALSE(predictions.has_value());
    THOR_THROW_IF_FALSE(lossGradient.has_value());
    THOR_THROW_IF_FALSE(labels.value() == labelsInput.value());
    THOR_THROW_IF_FALSE(predictions.value() == featureInput.value());
    THOR_THROW_IF_FALSE(lossGradient.value() == errorOutput.value());

    (void)runStream;
    // The loss gradient is precomputed during infer(), matching the other loss implementations.
    THOR_THROW_IF_FALSE(gradientPlan != nullptr);
}
