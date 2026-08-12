#include "DeepLearning/Implementation/Layers/Loss/CtcLoss.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

#include <algorithm>
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

}  // namespace

CtcLoss::CtcLoss(CtcLossOobGradientMode oobGradientMode, optional<float> lossWeight)
    : Loss(DataType::FP32), oobGradientMode(oobGradientMode), lossWeight(normalizeLossWeight(lossWeight)) {}

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
    } else if (connectionType == LABEL_OFFSETS_CONNECTION_TYPE) {
        return connectToLabelOffsetsInputLayer(previousLayer, featureInput, stream);
    } else if (connectionType == INPUT_LENGTHS_CONNECTION_TYPE) {
        return connectToInputLengthsInputLayer(previousLayer, featureInput, stream);
    }
    THOR_UNREACHABLE();
}

optional<Tensor> CtcLoss::connectToLabelOffsetsInputLayer(Layer* labelOffsetsLayer,
                                                          optional<Tensor> labelOffsets,
                                                          Stream labelOffsetsStream) {
    (void)labelOffsetsLayer;
    THOR_THROW_IF_FALSE(!this->labelOffsetsInput.has_value());
    THOR_THROW_IF_FALSE(labelOffsets.has_value());

    if (featureInput.has_value()) {
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == labelOffsets.value().getPlacement());
    }
    if (labelsInput.has_value()) {
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == labelOffsets.value().getPlacement());
    }
    if (inputLengthsInput.has_value()) {
        THOR_THROW_IF_FALSE(inputLengthsInput.value().getPlacement() == labelOffsets.value().getPlacement());
    }

    this->labelOffsetsInput = labelOffsets;
    this->labelOffsetsStream = labelOffsetsStream;
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
    if (labelOffsetsInput.has_value()) {
        THOR_THROW_IF_FALSE(labelOffsetsInput.value().getPlacement() == inputLengths.value().getPlacement());
    }

    this->inputLengthsInput = inputLengths;
    this->inputLengthsStream = inputLengthsStream;
    return nullopt;
}

void CtcLoss::initialize() {
    Loss::initialize();
    labelOffsetsReceived = false;
    inputLengthsReceived = false;
}

void CtcLoss::cleanup() {
    ctcPlan.reset();
    workspace.reset();
    inferenceGradientScratch.reset();
    generatedLabelLengths.reset();
    labelOffsetsValidationErrorBits.reset();
    maxTimeSteps = 0;
    ctcBatchSize = 0;
    numClasses = 0;
    maxTotalLabelValues = 0;
    backendMaxLabelLength = 0;
    Layer::cleanup();
}

void CtcLoss::validateConnectedDescriptors() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(labelOffsetsInput.has_value());
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
    THOR_THROW_IF_FALSE(labelOffsetsInput.value().isInitialized());
    THOR_THROW_IF_FALSE(inputLengthsInput.value().isInitialized());
    THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == featureInput.value().getPlacement());
    THOR_THROW_IF_FALSE(labelOffsetsInput.value().getPlacement() == featureInput.value().getPlacement());
    THOR_THROW_IF_FALSE(inputLengthsInput.value().getPlacement() == featureInput.value().getPlacement());

    THOR_THROW_IF_FALSE(labelsInput.value().getDescriptor().getDataType() == DataType::INT32);
    THOR_THROW_IF_FALSE(isRowPartitionOffsetDTypeSupported(labelOffsetsInput.value().getDescriptor().getDataType()));
    THOR_THROW_IF_FALSE(isCudnnCtcLengthDataType(inputLengthsInput.value().getDescriptor().getDataType()));

    const vector<uint64_t> labelsDimensions = labelsInput.value().getDescriptor().getDimensions();
    const vector<uint64_t> offsetsDimensions = labelOffsetsInput.value().getDescriptor().getDimensions();
    THOR_THROW_IF_FALSE(labelsDimensions.size() == 1);
    THOR_THROW_IF_FALSE(labelsDimensions[0] > 0);
    THOR_THROW_IF_FALSE(offsetsDimensions == vector<uint64_t>{static_cast<uint64_t>(ctcBatchSize) + 1});
    THOR_THROW_IF_FALSE(isBatchLengthVector(inputLengthsInput.value(), ctcBatchSize));
    maxTotalLabelValues = labelsDimensions[0];

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
    backendMaxLabelLength = std::min<uint32_t>(maxTimeSteps, 255U);
    THOR_THROW_IF_FALSE(backendMaxLabelLength > 0);
    config.maxLabelLength = backendMaxLabelLength;
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

    generatedLabelLengths = Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::INT32, {ctcBatchSize}));
    labelOffsetsValidationErrorBits = Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::UINT32, {1}));

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
    THOR_THROW_IF_FALSE(labelOffsetsInput.has_value());
    THOR_THROW_IF_FALSE(inputLengthsInput.has_value());
    THOR_THROW_IF_FALSE(generatedLabelLengths.has_value());
    THOR_THROW_IF_FALSE(labelOffsetsValidationErrorBits.has_value());

    Tensor& gradientTensor = errorOutput.has_value() ? errorOutput.value() : inferenceGradientScratch.value();
    THOR_THROW_IF_FALSE(gradientTensor.isInitialized());

    rowPartitionOffsetsToInt32LengthsChecked(labelOffsetsInput.value(),
                                             generatedLabelLengths.value(),
                                             labelOffsetsValidationErrorBits.value(),
                                             ctcBatchSize,
                                             maxTotalLabelValues,
                                             backendMaxLabelLength,
                                             stream);

    const size_t workspaceSizeBytes = ctcPlan->getWorkspaceSizeInBytes();
    void* workspacePtr = workspace.has_value() ? workspace.value().getMemPtr() : nullptr;

    ctcPlan->run(featureInput.value().getMemPtr(),
                 labelsInput.value().getMemPtr<int>(),
                 generatedLabelLengths.value().getMemPtr<int>(),
                 inputLengthsInput.value().getMemPtr<int>(),
                 featureOutput.value().getMemPtr(),
                 gradientTensor.getMemPtr(),
                 workspacePtr,
                 workspaceSizeBytes,
                 stream);

    // cuDNN currently reports zero cost for active rows whose target length is
    // zero. CTC itself has well-defined empty-target semantics: the only valid
    // alignment is blank at every valid time step. Repair those rows on-device
    // before Thor applies loss/gradient scaling.
    launchCorrectCtcEmptyTargetRows(featureInput.value().getMemPtr<float>(),
                                    generatedLabelLengths.value().getMemPtr<int>(),
                                    inputLengthsInput.value().getMemPtr<int>(),
                                    featureOutput.value().getMemPtr<float>(),
                                    gradientTensor.getMemPtr<float>(),
                                    ctcBatchSize,
                                    maxTimeSteps,
                                    numClasses,
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
    stream.waitEvent(labelOffsetsStream.putEvent());
    stream.waitEvent(inputLengthsStream.putEvent());

    runCudnn(stream);

    labelsStream.waitEvent(stream.putEvent());
    labelOffsetsStream.waitEvent(stream.putEvent());
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

void CtcLoss::forward(optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount) {
    THOR_THROW_IF_FALSE(running);
    THOR_THROW_IF_FALSE(labelsStream.isInitialized());
    THOR_THROW_IF_FALSE(labelOffsetsStream.isInitialized());
    THOR_THROW_IF_FALSE(inputLengthsStream.isInitialized());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(labelOffsetsInput.has_value());
    THOR_THROW_IF_FALSE(inputLengthsInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(featureInput.has_value());
    if (!isInferenceOnly()) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
    }

    if (inputTensor.has_value()) {
        recordBatchCardinality(validExampleCount);
        if (inputTensor.value() == featureInput.value()) {
            forwardFeatures(inputTensor.value(), validationPass);
            return;
        }
        if (inputTensor.value() == labelsInput.value()) {
            forwardLabels(inputTensor.value(), validationPass);
            return;
        }
        if (inputTensor.value() == labelOffsetsInput.value()) {
            THOR_THROW_IF_FALSE(labelOffsetsReceived == false);
            labelOffsetsReceived = true;
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
    THOR_THROW_IF_FALSE(labelOffsetsReceived);
    THOR_THROW_IF_FALSE(inputLengthsReceived);
    featureInputReceived = false;
    labelsReceived = false;
    labelOffsetsReceived = false;
    inputLengthsReceived = false;
    finishBatchCardinality();

    infer(featureInput, featureOutput, stream);
    maskInvalidLossTail();

    if (nextLayer.has_value())
        nextLayer.value()->forward(featureOutput, validationPass, currentValidExampleCount);

    if (isInferenceOnly() || validationPass)
        return;

    THOR_THROW_IF_FALSE(previousLayer.has_value());
    backward(nullopt, currentValidExampleCount);
}

void CtcLoss::advanceDataIfReady(bool validationPass) {
    if (featureInputReceived && labelsReceived && labelOffsetsReceived && inputLengthsReceived) {
        stream.waitEvent(labelsStream.putEvent());
        stream.waitEvent(labelOffsetsStream.putEvent());
        stream.waitEvent(inputLengthsStream.putEvent());
        forward(nullopt, validationPass);
    }
}

void CtcLoss::ensureNoDeviceCrossing() {
    Loss::ensureNoDeviceCrossing();
    if (featureInput.has_value()) {
        if (labelOffsetsInput.has_value())
            THOR_THROW_IF_FALSE(labelOffsetsInput.value().getPlacement() == featureInput.value().getPlacement());
        if (inputLengthsInput.has_value())
            THOR_THROW_IF_FALSE(inputLengthsInput.value().getPlacement() == featureInput.value().getPlacement());
        if (generatedLabelLengths.has_value())
            THOR_THROW_IF_FALSE(generatedLabelLengths.value().getPlacement() == featureInput.value().getPlacement());
        if (labelOffsetsValidationErrorBits.has_value())
            THOR_THROW_IF_FALSE(labelOffsetsValidationErrorBits.value().getPlacement() == featureInput.value().getPlacement());
        if (workspace.has_value())
            THOR_THROW_IF_FALSE(workspace.value().getPlacement() == featureInput.value().getPlacement());
        if (inferenceGradientScratch.has_value())
            THOR_THROW_IF_FALSE(inferenceGradientScratch.value().getPlacement() == featureInput.value().getPlacement());
    }
}

string CtcLoss::getType() { return "CtcLoss"; }

vector<Stream> CtcLoss::getProcessingStreams() {
    vector<Stream> processingStreams = Loss::getProcessingStreams();
    if (labelOffsetsStream.isInitialized())
        processingStreams.push_back(labelOffsetsStream);
    if (inputLengthsStream.isInitialized())
        processingStreams.push_back(inputLengthsStream);
    return processingStreams;
}

vector<Event> CtcLoss::getSynchronizeEvents() {
    vector<Event> events;
    set<uint64_t> synchronizedStreamIds;
    appendSynchronizeEvent(events, synchronizedStreamIds, stream);
    appendSynchronizeEvent(events, synchronizedStreamIds, labelsStream);
    appendSynchronizeEvent(events, synchronizedStreamIds, labelOffsetsStream);
    appendSynchronizeEvent(events, synchronizedStreamIds, inputLengthsStream);
    return events;
}

}  // namespace ThorImplementation
