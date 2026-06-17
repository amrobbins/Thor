#include "DeepLearning/Implementation/Layers/Loss/CtcLoss.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

#include <limits>
#include <utility>

using namespace std;

namespace ThorImplementation {

namespace {

uint32_t checkedUint32(uint64_t value, const char* what) {
    (void)what;
    THOR_THROW_IF_FALSE(value <= numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(value);
}

bool isBatchLengthVector(const Tensor& tensor, uint64_t batchSize) {
    const vector<uint64_t> dims = tensor.getDescriptor().getDimensions();
    return (dims.size() == 1 && dims[0] == batchSize) || (dims.size() == 2 && dims[0] == batchSize && dims[1] == 1);
}

bool isPaddedLabelsMatrix(const Tensor& tensor, uint64_t batchSize, uint64_t maxLabelLength) {
    const vector<uint64_t> dims = tensor.getDescriptor().getDimensions();
    return dims.size() == 2 && dims[0] == batchSize && dims[1] == maxLabelLength;
}

}  // namespace

CtcLoss::CtcLoss(uint32_t maxLabelLength, CtcLossOobGradientMode oobGradientMode, optional<float> lossWeight)
    : Loss(DataType::FP32),
      maxLabelLength(maxLabelLength),
      oobGradientMode(oobGradientMode),
      lossWeight(normalizeLossWeight(lossWeight)) {
    THOR_THROW_IF_FALSE(maxLabelLength > 0);
}

vector<uint64_t> CtcLoss::rawLossDimensionsForProbabilities(const vector<uint64_t>& probabilityDimensions) {
    THOR_THROW_IF_FALSE(probabilityDimensions.size() == 3);
    THOR_THROW_IF_FALSE(probabilityDimensions[0] > 0);
    THOR_THROW_IF_FALSE(probabilityDimensions[1] > 0);
    THOR_THROW_IF_FALSE(probabilityDimensions[2] > 1);
    return {probabilityDimensions[0], 1};
}

optional<Tensor> CtcLoss::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return Tensor(featureInput.value().getPlacement(),
                  TensorDescriptor(DataType::FP32, rawLossDimensionsForProbabilities(featureInput.value().getDescriptor().getDimensions())));
}

optional<Tensor> CtcLoss::createErrorOutputTensor(bool backPropagateError) {
    if (backPropagateError && !isInferenceOnly()) {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone(DataType::FP32);
    }
    return nullopt;
}

optional<Tensor> CtcLoss::connectToPreviousLayer(Layer* previousLayer,
                                                 optional<Tensor> featureInput,
                                                 Stream stream,
                                                 bool backPropagateError,
                                                 int connectionType) {
    if (connectionType == static_cast<int>(ConnectionType::FORWARD_BACKWARD)) {
        return connectToPredictionsInputLayer(previousLayer, featureInput, stream, backPropagateError);
    } else if (connectionType == static_cast<int>(ConnectionType::LABELS)) {
        return connectToLabelsInputLayer(previousLayer, featureInput, stream);
    } else if (connectionType == LABEL_LENGTHS_CONNECTION_TYPE) {
        return connectToLabelLengthsInputLayer(previousLayer, featureInput, stream);
    } else if (connectionType == INPUT_LENGTHS_CONNECTION_TYPE) {
        return connectToInputLengthsInputLayer(previousLayer, featureInput, stream);
    }
    THOR_UNREACHABLE();
}

optional<Tensor> CtcLoss::connectToLabelLengthsInputLayer(Layer* labelLengthsLayer,
                                                          optional<Tensor> labelLengths,
                                                          Stream labelLengthsStream) {
    (void)labelLengthsLayer;
    THOR_THROW_IF_FALSE(!this->labelLengthsInput.has_value());
    THOR_THROW_IF_FALSE(labelLengths.has_value());

    if (featureInput.has_value()) {
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == labelLengths.value().getPlacement());
    }
    if (labelsInput.has_value()) {
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == labelLengths.value().getPlacement());
    }
    if (inputLengthsInput.has_value()) {
        THOR_THROW_IF_FALSE(inputLengthsInput.value().getPlacement() == labelLengths.value().getPlacement());
    }

    this->labelLengthsInput = labelLengths;
    this->labelLengthsStream = labelLengthsStream;
    return nullopt;
}

optional<Tensor> CtcLoss::connectToInputLengthsInputLayer(Layer* inputLengthsLayer,
                                                          optional<Tensor> inputLengths,
                                                          Stream inputLengthsStream) {
    (void)inputLengthsLayer;
    THOR_THROW_IF_FALSE(!this->inputLengthsInput.has_value());
    THOR_THROW_IF_FALSE(inputLengths.has_value());

    if (featureInput.has_value()) {
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == inputLengths.value().getPlacement());
    }
    if (labelsInput.has_value()) {
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == inputLengths.value().getPlacement());
    }
    if (labelLengthsInput.has_value()) {
        THOR_THROW_IF_FALSE(labelLengthsInput.value().getPlacement() == inputLengths.value().getPlacement());
    }

    this->inputLengthsInput = inputLengths;
    this->inputLengthsStream = inputLengthsStream;
    return nullopt;
}

void CtcLoss::initialize() {
    Loss::initialize();
    labelLengthsReceived = false;
    inputLengthsReceived = false;
}

void CtcLoss::cleanup() {
    ctcPlan.reset();
    workspace.reset();
    inferenceGradientScratch.reset();
    packedLabels.reset();
    maxTimeSteps = 0;
    ctcBatchSize = 0;
    numClasses = 0;
    Layer::cleanup();
}

void CtcLoss::validateConnectedDescriptors() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(labelLengthsInput.has_value());
    THOR_THROW_IF_FALSE(inputLengthsInput.has_value());

    THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getPlacement() == featureInput.value().getPlacement());

    const vector<uint64_t> probabilityDimensions = featureInput.value().getDescriptor().getDimensions();
    THOR_THROW_IF_FALSE(probabilityDimensions.size() == 3);
    THOR_THROW_IF_FALSE(probabilityDimensions[0] > 0);
    THOR_THROW_IF_FALSE(probabilityDimensions[1] > 0);
    THOR_THROW_IF_FALSE(probabilityDimensions[2] > 1);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDimensions() == rawLossDimensionsForProbabilities(probabilityDimensions));

    ctcBatchSize = checkedUint32(probabilityDimensions[0], "batchSize");
    maxTimeSteps = checkedUint32(probabilityDimensions[1], "maxTimeSteps");
    numClasses = checkedUint32(probabilityDimensions[2], "numClasses");

    THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
    THOR_THROW_IF_FALSE(labelLengthsInput.value().isInitialized());
    THOR_THROW_IF_FALSE(inputLengthsInput.value().isInitialized());
    THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == featureInput.value().getPlacement());
    THOR_THROW_IF_FALSE(labelLengthsInput.value().getPlacement() == featureInput.value().getPlacement());
    THOR_THROW_IF_FALSE(inputLengthsInput.value().getPlacement() == featureInput.value().getPlacement());

    THOR_THROW_IF_FALSE(labelsInput.value().getDescriptor().getDataType() == DataType::INT32);
    THOR_THROW_IF_FALSE(labelLengthsInput.value().getDescriptor().getDataType() == DataType::INT32);
    THOR_THROW_IF_FALSE(inputLengthsInput.value().getDescriptor().getDataType() == DataType::INT32);
    THOR_THROW_IF_FALSE(isPaddedLabelsMatrix(labelsInput.value(), ctcBatchSize, maxLabelLength));
    THOR_THROW_IF_FALSE(isBatchLengthVector(labelLengthsInput.value(), ctcBatchSize));
    THOR_THROW_IF_FALSE(isBatchLengthVector(inputLengthsInput.value(), ctcBatchSize));

    THOR_THROW_IF_FALSE(errorOutput.has_value() || isInferenceOnly());
    if (errorOutput.has_value()) {
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement() == featureInput.value().getPlacement());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor() == featureInput.value().getDescriptor());
    }
}

void CtcLoss::compileImpl() {
    Layer::compileImpl();
    validateConnectedDescriptors();

    CudnnCtcLossConfig config;
    config.maxTimeSteps = maxTimeSteps;
    config.batchSize = ctcBatchSize;
    config.numClasses = numClasses;
    config.maxLabelLength = maxLabelLength;
    config.dataType = DataType::FP32;
    config.algorithm = CtcLossAlgorithm::DETERMINISTIC;
    config.normalization = CtcLossNormalization::SOFTMAX;
    config.oobGradientMode = oobGradientMode;

    ctcPlan = make_unique<CudnnCtcLossPlan>(config, stream);
    if (ctcPlan->getWorkspaceSizeInBytes() > 0) {
        workspace = Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::UINT8, {ctcPlan->getWorkspaceSizeInBytes()}));
    } else {
        workspace.reset();
    }

    packedLabels = Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::INT32, {static_cast<uint64_t>(ctcBatchSize) * maxLabelLength}));

    if (isInferenceOnly()) {
        inferenceGradientScratch = featureInput.value().clone(DataType::FP32);
    } else {
        inferenceGradientScratch.reset();
    }
}

void CtcLoss::runCudnn(Stream stream) {
    THOR_THROW_IF_FALSE(ctcPlan != nullptr);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(labelLengthsInput.has_value());
    THOR_THROW_IF_FALSE(inputLengthsInput.has_value());

    Tensor& gradientTensor = errorOutput.has_value() ? errorOutput.value() : inferenceGradientScratch.value();
    THOR_THROW_IF_FALSE(gradientTensor.isInitialized());

    THOR_THROW_IF_FALSE(packedLabels.has_value());
    launchCompactPaddedCtcLabels(labelsInput.value().getMemPtr<int>(),
                                 labelLengthsInput.value().getMemPtr<int>(),
                                 packedLabels.value().getMemPtr<int>(),
                                 ctcBatchSize,
                                 maxLabelLength,
                                 stream);

    const size_t workspaceSizeBytes = ctcPlan->getWorkspaceSizeInBytes();
    void* workspacePtr = workspace.has_value() ? workspace.value().getMemPtr() : nullptr;

    ctcPlan->run(featureInput.value().getMemPtr(),
                 packedLabels.value().getMemPtr<int>(),
                 labelLengthsInput.value().getMemPtr<int>(),
                 inputLengthsInput.value().getMemPtr<int>(),
                 featureOutput.value().getMemPtr(),
                 gradientTensor.getMemPtr(),
                 workspacePtr,
                 workspaceSizeBytes,
                 stream);

    const float materializedLossWeight = materializeLossWeight(lossWeight);
    const float gradientScale = static_cast<float>(lossScalingFactor) * materializedLossWeight;
    launchScaleCtcLossOutputs(featureOutput.value().getMemPtr<float>(),
                              gradientTensor.getMemPtr<float>(),
                              inputLengthsInput.value().getMemPtr<int>(),
                              ctcBatchSize,
                              maxTimeSteps,
                              numClasses,
                              featureOutput.value().getTotalNumElements(),
                              errorOutput.has_value(),
                              materializedLossWeight,
                              gradientScale,
                              stream);
}

void CtcLoss::infer(optional<Tensor> probabilities, optional<Tensor> loss, Stream stream) {
    THOR_THROW_IF_FALSE(probabilities.has_value());
    THOR_THROW_IF_FALSE(loss.has_value());
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(probabilities.value() == featureInput.value());
    THOR_THROW_IF_FALSE(loss.value() == featureOutput.value());

    ScopedGpu scopedGpu(probabilities.value().getPlacement().getDeviceNum());
    stream.waitEvent(labelsStream.putEvent());
    stream.waitEvent(labelLengthsStream.putEvent());
    stream.waitEvent(inputLengthsStream.putEvent());

    runCudnn(stream);

    labelsStream.waitEvent(stream.putEvent());
    labelLengthsStream.waitEvent(stream.putEvent());
    inputLengthsStream.waitEvent(stream.putEvent());
}

void CtcLoss::backProp(optional<Tensor> labels, optional<Tensor> probabilities, optional<Tensor> lossGradient, Stream stream) {
    (void)labels;
    (void)probabilities;
    (void)stream;
    // CTC gradients are produced during infer() because cuDNN returns costs and gradients together.
    THOR_THROW_IF_FALSE(lossGradient.has_value());
    THOR_THROW_IF_FALSE(lossGradient.value().getDescriptor().getDataType() == DataType::FP32);
}

void CtcLoss::forward(optional<Tensor> inputTensor, bool validationPass, uint32_t batchSize) {
    THOR_THROW_IF_FALSE(running);
    THOR_THROW_IF_FALSE(labelsStream.isInitialized());
    THOR_THROW_IF_FALSE(labelLengthsStream.isInitialized());
    THOR_THROW_IF_FALSE(inputLengthsStream.isInitialized());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(labelLengthsInput.has_value());
    THOR_THROW_IF_FALSE(inputLengthsInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(featureInput.has_value());
    if (!isInferenceOnly()) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
    }

    if (inputTensor.has_value()) {
        if (batchSize != 0)
            currentBatchSize = batchSize;
        if (inputTensor.value() == featureInput.value()) {
            forwardFeatures(inputTensor.value(), validationPass);
            return;
        }
        if (inputTensor.value() == labelsInput.value()) {
            forwardLabels(inputTensor.value(), validationPass);
            return;
        }
        if (inputTensor.value() == labelLengthsInput.value()) {
            THOR_THROW_IF_FALSE(labelLengthsReceived == false);
            labelLengthsReceived = true;
            advanceDataIfReady(validationPass);
            return;
        }
        if (inputTensor.value() == inputLengthsInput.value()) {
            THOR_THROW_IF_FALSE(inputLengthsReceived == false);
            inputLengthsReceived = true;
            advanceDataIfReady(validationPass);
            return;
        }
        THOR_UNREACHABLE();
    }

    THOR_THROW_IF_FALSE(!inputTensor.has_value());
    THOR_THROW_IF_FALSE(featureInputReceived);
    THOR_THROW_IF_FALSE(labelsReceived);
    THOR_THROW_IF_FALSE(labelLengthsReceived);
    THOR_THROW_IF_FALSE(inputLengthsReceived);
    featureInputReceived = false;
    labelsReceived = false;
    labelLengthsReceived = false;
    inputLengthsReceived = false;

    infer(featureInput, featureOutput, stream);

    if (nextLayer.has_value())
        nextLayer.value()->forward(featureOutput, validationPass, currentBatchSize);

    if (isInferenceOnly() || validationPass)
        return;

    THOR_THROW_IF_FALSE(previousLayer.has_value());
    backward(nullopt, currentBatchSize);
}

void CtcLoss::advanceDataIfReady(bool validationPass) {
    if (featureInputReceived && labelsReceived && labelLengthsReceived && inputLengthsReceived) {
        stream.waitEvent(labelsStream.putEvent());
        stream.waitEvent(labelLengthsStream.putEvent());
        stream.waitEvent(inputLengthsStream.putEvent());
        forward(nullopt, validationPass);
    }
}

void CtcLoss::ensureNoDeviceCrossing() {
    Loss::ensureNoDeviceCrossing();
    if (featureInput.has_value()) {
        if (labelLengthsInput.has_value())
            THOR_THROW_IF_FALSE(labelLengthsInput.value().getPlacement() == featureInput.value().getPlacement());
        if (inputLengthsInput.has_value())
            THOR_THROW_IF_FALSE(inputLengthsInput.value().getPlacement() == featureInput.value().getPlacement());
        if (workspace.has_value())
            THOR_THROW_IF_FALSE(workspace.value().getPlacement() == featureInput.value().getPlacement());
        if (inferenceGradientScratch.has_value())
            THOR_THROW_IF_FALSE(inferenceGradientScratch.value().getPlacement() == featureInput.value().getPlacement());
        if (packedLabels.has_value())
            THOR_THROW_IF_FALSE(packedLabels.value().getPlacement() == featureInput.value().getPlacement());
    }
}

string CtcLoss::getType() { return "CtcLoss"; }

}  // namespace ThorImplementation
