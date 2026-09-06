#include "DeepLearning/Implementation/Layers/Loss/SparseCategoricalCrossEntropyWithLogits.h"

#include <limits>

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

using namespace ThorImplementation;
using namespace std;

namespace {

uint64_t productDimensions(const vector<uint64_t> &dims, size_t begin, size_t end) {
    uint64_t result = 1;
    THOR_THROW_IF_FALSE(begin <= end && end <= dims.size());
    for (size_t i = begin; i < end; ++i) {
        THOR_THROW_IF_FALSE(dims[i] > 0);
        result *= dims[i];
    }
    return result;
}

bool isSupportedMaskType(DataType dataType) {
    return dataType == DataType::BOOLEAN || dataType == DataType::UINT8 || dataType == DataType::FP16 || dataType == DataType::FP32;
}

}  // namespace

SparseCategoricalCrossEntropyWithLogits::SparseCategoricalCrossEntropyWithLogits(DataType lossDataType,
                                                                                 optional<float> lossWeight,
                                                                                 optional<uint32_t> ignoreIndex,
                                                                                 optional<uint32_t> raggedBatchSize)
    : Loss(lossDataType),
      ignoreIndex(ignoreIndex),
      lossWeight(normalizeLossWeight(lossWeight)),
      raggedBatchSize(raggedBatchSize) {
    if (raggedBatchSize.has_value())
        THOR_THROW_IF_FALSE(raggedBatchSize.value() > 0);
}

vector<uint64_t> SparseCategoricalCrossEntropyWithLogits::rawLossDimensionsForFeatureInput(const vector<uint64_t> &featureInputDimensions) {
    THOR_THROW_IF_FALSE(featureInputDimensions.size() >= 1);
    THOR_THROW_IF_FALSE(featureInputDimensions.back() > 1);
    if (featureInputDimensions.size() == 1)
        return {1};
    return vector<uint64_t>(featureInputDimensions.begin(), featureInputDimensions.end() - 1);
}

bool SparseCategoricalCrossEntropyWithLogits::sparseLabelOrMaskDimensionsMatchFeaturePrefix(
    const vector<uint64_t> &candidateDimensions, const vector<uint64_t> &featureInputDimensions) {
    THOR_THROW_IF_FALSE(featureInputDimensions.size() >= 1);
    const size_t prefixRank = featureInputDimensions.size() - 1;
    if (prefixRank == 0) {
        return candidateDimensions.size() == 1 && candidateDimensions[0] == 1;
    }
    if (candidateDimensions.size() == prefixRank) {
        for (size_t i = 0; i < prefixRank; ++i) {
            if (candidateDimensions[i] != featureInputDimensions[i])
                return false;
        }
        return true;
    }
    if (candidateDimensions.size() == prefixRank + 1 && candidateDimensions.back() == 1) {
        for (size_t i = 0; i < prefixRank; ++i) {
            if (candidateDimensions[i] != featureInputDimensions[i])
                return false;
        }
        return true;
    }
    return false;
}

optional<Tensor> SparseCategoricalCrossEntropyWithLogits::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    return Tensor(featureInput.value().getPlacement(),
                  TensorDescriptor(lossDataType, rawLossDimensionsForFeatureInput(featureInput.value().getDescriptor().getDimensions())));
}

optional<Tensor> SparseCategoricalCrossEntropyWithLogits::createErrorOutputTensor(bool backPropagateError) {
    if (backPropagateError && !isInferenceOnly()) {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone();
    }
    return nullopt;
}

optional<Tensor> SparseCategoricalCrossEntropyWithLogits::connectToPreviousLayer(Layer *previousLayer,
                                                                                 optional<Tensor> featureInput,
                                                                                 Stream stream,
                                                                                 bool backPropagateError,
                                                                                 int connectionType) {
    if (connectionType == static_cast<int>(ConnectionType::FORWARD_BACKWARD)) {
        return connectToPredictionsInputLayer(previousLayer, featureInput, stream, backPropagateError);
    } else if (connectionType == static_cast<int>(ConnectionType::LABELS)) {
        return connectToLabelsInputLayer(previousLayer, featureInput, stream);
    } else if (connectionType == MASK_CONNECTION_TYPE) {
        return connectToMaskInputLayer(previousLayer, featureInput, stream);
    } else if (connectionType == ACTIVE_COUNT_CONNECTION_TYPE) {
        return connectToActiveCountInputLayer(previousLayer, featureInput, stream);
    }
    THOR_UNREACHABLE();
}

optional<Tensor> SparseCategoricalCrossEntropyWithLogits::connectToMaskInputLayer(Layer *maskLayer, optional<Tensor> mask, Stream maskStream) {
    (void)maskLayer;
    THOR_THROW_IF_FALSE(!maskInput.has_value());
    THOR_THROW_IF_FALSE(mask.has_value());

    if (featureInput.has_value()) {
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == mask.value().getPlacement());
    }
    if (labelsInput.has_value()) {
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == mask.value().getPlacement());
    }

    maskInput = mask;
    this->maskStream = maskStream;
    return nullopt;
}

optional<Tensor> SparseCategoricalCrossEntropyWithLogits::connectToActiveCountInputLayer(
    Layer *activeCountLayer, optional<Tensor> activeCount, Stream activeCountStream) {
    (void)activeCountLayer;
    THOR_THROW_IF_FALSE(usesRaggedActiveCount());
    THOR_THROW_IF_FALSE(!activeCountInput.has_value());
    THOR_THROW_IF_FALSE(activeCount.has_value());
    THOR_THROW_IF_FALSE(activeCount.value().getDimensions() == vector<uint64_t>{1});
    THOR_THROW_IF_FALSE(activeCount.value().getDataType() == DataType::UINT32 ||
                        activeCount.value().getDataType() == DataType::UINT64);

    if (featureInput.has_value()) {
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == activeCount.value().getPlacement());
    }
    if (labelsInput.has_value())
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == activeCount.value().getPlacement());
    if (maskInput.has_value())
        THOR_THROW_IF_FALSE(maskInput.value().getPlacement() == activeCount.value().getPlacement());

    activeCountInput = activeCount;
    this->activeCountStream = activeCountStream;
    return nullopt;
}

void SparseCategoricalCrossEntropyWithLogits::initialize() {
    Loss::initialize();
    maskReceived = false;
    activeCountReceived = false;
}

void SparseCategoricalCrossEntropyWithLogits::cleanup() {
    maskReadyEvent = Event();
    maskReusableEvent = Event();
    activeCountReadyEvent = Event();
    activeCountReusableEvent = Event();
    Loss::cleanup();
}

void SparseCategoricalCrossEntropyWithLogits::compileImpl() {
    Layer::compileImpl();
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDataType() == DataType::FP16 ||
                        featureInput.value().getDescriptor().getDataType() == DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == DataType::FP16 ||
                        featureOutput.value().getDescriptor().getDataType() == DataType::FP32);

    const DataType labelsDataType = labelsInput.value().getDescriptor().getDataType();
    THOR_THROW_IF_FALSE(labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 || labelsDataType == DataType::UINT32);

    const vector<uint64_t> featureInputDimensions = featureInput.value().getDescriptor().getDimensions();
    THOR_THROW_IF_FALSE(featureInputDimensions.size() >= 1);
    THOR_THROW_IF_FALSE(featureInputDimensions.back() > 1);
    THOR_THROW_IF_FALSE(sparseLabelOrMaskDimensionsMatchFeaturePrefix(labelsInput.value().getDescriptor().getDimensions(), featureInputDimensions));
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDimensions() == rawLossDimensionsForFeatureInput(featureInputDimensions));

    if (usesRaggedActiveCount()) {
        THOR_THROW_IF_FALSE(activeCountInput.has_value());
        THOR_THROW_IF_FALSE(activeCountInput.value().isInitialized());
        THOR_THROW_IF_FALSE(activeCountInput.value().getPlacement() == featureInput.value().getPlacement());
        THOR_THROW_IF_FALSE(activeCountInput.value().getDimensions() == vector<uint64_t>{1});
        THOR_THROW_IF_FALSE(activeCountInput.value().getDataType() == DataType::UINT32 ||
                            activeCountInput.value().getDataType() == DataType::UINT64);
        // Ragged sparse CE is tokenwise over packed [capacity, C] values. The
        // runtime active-count scalar, not a product of logical dimensions,
        // selects the valid prefix within this capacity.
        THOR_THROW_IF_FALSE(featureInputDimensions.size() == 2);
    } else {
        THOR_THROW_IF_FALSE(!activeCountInput.has_value());
    }

    const uint64_t effectiveRows = usesRaggedActiveCount()
                                       ? featureInputDimensions.front()
                                       : (featureInputDimensions.size() == 1
                                              ? 1
                                              : productDimensions(featureInputDimensions, 0, featureInputDimensions.size() - 1));
    THOR_THROW_IF_FALSE(effectiveRows <= numeric_limits<uint32_t>::max());
    numRows = static_cast<uint32_t>(effectiveRows);
    THOR_THROW_IF_FALSE(featureInputDimensions.back() <= numeric_limits<uint32_t>::max());
    numClasses = static_cast<uint32_t>(featureInputDimensions.back());

    THOR_THROW_IF_FALSE(errorOutput.has_value() || isInferenceOnly());
    if (errorOutput.has_value()) {
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor() == featureInput.value().getDescriptor());
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement() == featureInput.value().getPlacement());
    }

    THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
    THOR_THROW_IF_FALSE(labelsInput.value().getPlacement() == featureInput.value().getPlacement());

    if (maskInput.has_value()) {
        THOR_THROW_IF_FALSE(maskInput.value().isInitialized());
        THOR_THROW_IF_FALSE(maskInput.value().getPlacement() == featureInput.value().getPlacement());
        THOR_THROW_IF_FALSE(sparseLabelOrMaskDimensionsMatchFeaturePrefix(maskInput.value().getDescriptor().getDimensions(), featureInputDimensions));
        THOR_THROW_IF_FALSE(isSupportedMaskType(maskInput.value().getDescriptor().getDataType()));
    }
}

uint32_t SparseCategoricalCrossEntropyWithLogits::resolveRaggedValidExampleCount(uint32_t validExampleCount) const {
    THOR_THROW_IF_FALSE(raggedBatchSize.has_value());
    const uint32_t resolved = validExampleCount == 0 ? raggedBatchSize.value() : validExampleCount;
    THOR_THROW_IF_FALSE(resolved > 0);
    THOR_THROW_IF_FALSE(resolved <= raggedBatchSize.value());
    return resolved;
}

void SparseCategoricalCrossEntropyWithLogits::recordCurrentBatchCardinality(uint32_t validExampleCount) {
    if (!usesRaggedActiveCount()) {
        recordBatchCardinality(validExampleCount);
        return;
    }

    const uint32_t resolved = resolveRaggedValidExampleCount(validExampleCount);
    if (batchCardinalitySet) {
        THOR_THROW_IF_FALSE(currentValidExampleCount == resolved);
        return;
    }
    currentValidExampleCount = resolved;
    batchCardinalitySet = true;
}

void SparseCategoricalCrossEntropyWithLogits::finishCurrentBatchCardinality() {
    if (!usesRaggedActiveCount()) {
        finishBatchCardinality();
        return;
    }
    THOR_THROW_IF_FALSE(batchCardinalitySet);
    batchCardinalitySet = false;
}

void SparseCategoricalCrossEntropyWithLogits::infer(optional<Tensor> logits, optional<Tensor> loss, Stream stream) {
    THOR_THROW_IF_FALSE(logits.has_value());
    THOR_THROW_IF_FALSE(loss.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(compiled);

    ScopedGpu scopedGpu(logits.value().getPlacement().getDeviceNum());
    launchForCurrentTypes();
}

void SparseCategoricalCrossEntropyWithLogits::backProp(optional<Tensor> labels, optional<Tensor> logits, optional<Tensor> lossGradient, Stream stream) {
    (void)labels;
    (void)logits;
    (void)stream;
    THOR_THROW_IF_FALSE(lossGradient.has_value());
    THOR_THROW_IF_FALSE(lossGradient.value().getDataType() == DataType::FP32 || lossGradient.value().getDataType() == DataType::FP16);
}

void SparseCategoricalCrossEntropyWithLogits::forward(optional<Tensor> inputTensor, bool validationPass, uint32_t validExampleCount) {
    THOR_THROW_IF_FALSE(running);
    THOR_THROW_IF_FALSE(labelsStream.isInitialized());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(featureInput.has_value());
    if (!isInferenceOnly()) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
    }

    if (inputTensor.has_value()) {
        recordCurrentBatchCardinality(validExampleCount);
        if (inputTensor.value() == featureInput.value()) {
            forwardFeatures(inputTensor.value(), validationPass);
            return;
        }
        if (inputTensor.value() == labelsInput.value()) {
            forwardLabels(inputTensor.value(), validationPass);
            return;
        }
        if (maskInput.has_value() && inputTensor.value() == maskInput.value()) {
            THOR_THROW_IF_FALSE(maskReceived == false);
            maskReceived = true;
            advanceDataIfReady(validationPass);
            return;
        }
        if (activeCountInput.has_value() && inputTensor.value() == activeCountInput.value()) {
            THOR_THROW_IF_FALSE(activeCountReceived == false);
            activeCountReceived = true;
            advanceDataIfReady(validationPass);
            return;
        }
        THOR_UNREACHABLE();
    }

    THOR_THROW_IF_FALSE(!inputTensor.has_value());
    THOR_THROW_IF_FALSE(featureInputReceived);
    THOR_THROW_IF_FALSE(labelsReceived);
    THOR_THROW_IF_FALSE(!maskInput.has_value() || maskReceived);
    THOR_THROW_IF_FALSE(!usesRaggedActiveCount() || activeCountReceived);
    featureInputReceived = false;
    labelsReceived = false;
    maskReceived = false;
    activeCountReceived = false;
    finishCurrentBatchCardinality();

    infer(featureInput, featureOutput, stream);
    if (!usesRaggedActiveCount())
        maskInvalidLossTail();

    if (maskInput.has_value())
        maskStream.waitFor(stream, maskReusableEvent);
    if (activeCountInput.has_value())
        activeCountStream.waitFor(stream, activeCountReusableEvent);
    if (isInferenceOnly() || validationPass)
        markLabelsReusableAfterCompute();

    if (nextLayer.has_value())
        nextLayer.value()->forward(featureOutput, validationPass, currentValidExampleCount);

    if (isInferenceOnly() || validationPass)
        return;

    THOR_THROW_IF_FALSE(previousLayer.has_value());
    backward(nullopt, currentValidExampleCount);
}

void SparseCategoricalCrossEntropyWithLogits::backward(optional<Tensor> errorInput, uint32_t validExampleCount) {
    if (!usesRaggedActiveCount()) {
        Loss::backward(errorInput, validExampleCount);
        return;
    }

    THOR_THROW_IF_FALSE(running);
    THOR_THROW_IF_FALSE(!errorInput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value() && labelsInput.value().isInitialized());
    THOR_THROW_IF_FALSE(errorOutput.has_value() && errorOutput.value().isInitialized());
    THOR_THROW_IF_FALSE(labelsStream.isInitialized());
    const uint32_t resolved = validExampleCount == 0 ? currentValidExampleCount : resolveRaggedValidExampleCount(validExampleCount);
    THOR_THROW_IF_FALSE(resolved == currentValidExampleCount);

    // The logits-native kernel materializes dL/dlogits during infer(). backProp
    // preserves the ordinary Loss contract but must not apply dense
    // valid-example tail masking: packed token extent is controlled solely by
    // DEVICE_ACTIVE_COUNT.
    backProp(labelsInput, featureInput, errorOutput, stream);
    markLabelsReusableAfterCompute();

    if (previousLayer.has_value())
        previousLayer.value()->backward(errorOutput, resolved);
}


void SparseCategoricalCrossEntropyWithLogits::advanceDataIfReady(bool validationPass) {
    if (featureInputReceived && labelsReceived && (!maskInput.has_value() || maskReceived) &&
        (!usesRaggedActiveCount() || activeCountReceived)) {
        waitForLabelsReady();
        if (maskInput.has_value())
            stream.waitFor(maskStream, maskReadyEvent);
        if (activeCountInput.has_value())
            stream.waitFor(activeCountStream, activeCountReadyEvent);
        forward(nullopt, validationPass);
    }
}

void SparseCategoricalCrossEntropyWithLogits::ensureNoDeviceCrossing() {
    Loss::ensureNoDeviceCrossing();
    if (maskInput.has_value()) {
        if (featureInput.has_value())
            THOR_THROW_IF_FALSE(maskInput.value().getPlacement() == featureInput.value().getPlacement());
        if (labelsInput.has_value())
            THOR_THROW_IF_FALSE(maskInput.value().getPlacement() == labelsInput.value().getPlacement());
    }
    if (activeCountInput.has_value()) {
        if (featureInput.has_value())
            THOR_THROW_IF_FALSE(activeCountInput.value().getPlacement() == featureInput.value().getPlacement());
        if (labelsInput.has_value())
            THOR_THROW_IF_FALSE(activeCountInput.value().getPlacement() == labelsInput.value().getPlacement());
        if (maskInput.has_value())
            THOR_THROW_IF_FALSE(activeCountInput.value().getPlacement() == maskInput.value().getPlacement());
    }
}

void SparseCategoricalCrossEntropyWithLogits::launchForCurrentTypes() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());

    const DataType logitsType = featureInput.value().getDescriptor().getDataType();
    const DataType lossType = featureOutput.value().getDescriptor().getDataType();
    const DataType labelsType = labelsInput.value().getDescriptor().getDataType();
    const DataType maskType = maskInput.has_value() ? maskInput.value().getDescriptor().getDataType() : DataType::UINT8;

#define LAUNCH_WITH_MASK(LABEL_CPP_TYPE, LOGIT_CPP_TYPE, LOSS_CPP_TYPE, MASK_CPP_TYPE) \
    do { \
        if (usesRaggedActiveCount()) { \
            THOR_THROW_IF_FALSE(activeCountInput.has_value()); \
            launchRaggedSparseCategoricalCrossEntropyWithLogits<LABEL_CPP_TYPE, LOGIT_CPP_TYPE, LOSS_CPP_TYPE, MASK_CPP_TYPE>( \
                labelsInput.value().getMemPtr(), \
                featureInput.value().getMemPtr(), \
                maskInput.has_value() ? maskInput.value().getMemPtr() : nullptr, \
                featureOutput.value().getMemPtr(), \
                isInferenceOnly() ? nullptr : errorOutput.value().getMemPtr(), \
                activeCountInput.value().getMemPtr(), \
                activeCountInput.value().getDataType(), \
                numClasses, \
                numRows, \
                !isInferenceOnly(), \
                lossScalingFactor, \
                materializeLossWeight(lossWeight), \
                ignoreIndex.has_value(), \
                ignoreIndex.value_or(0), \
                maskInput.has_value(), \
                stream); \
        } else { \
            launchSparseCategoricalCrossEntropyWithLogits<LABEL_CPP_TYPE, LOGIT_CPP_TYPE, LOSS_CPP_TYPE, MASK_CPP_TYPE>( \
                labelsInput.value().getMemPtr(), \
                featureInput.value().getMemPtr(), \
                maskInput.has_value() ? maskInput.value().getMemPtr() : nullptr, \
                featureOutput.value().getMemPtr(), \
                isInferenceOnly() ? nullptr : errorOutput.value().getMemPtr(), \
                numClasses, \
                numRows, \
                !isInferenceOnly(), \
                lossScalingFactor, \
                materializeLossWeight(lossWeight), \
                ignoreIndex.has_value(), \
                ignoreIndex.value_or(0), \
                maskInput.has_value(), \
                stream); \
        } \
    } while (false)

#define DISPATCH_MASK(LABEL_CPP_TYPE, LOGIT_CPP_TYPE, LOSS_CPP_TYPE) \
    do { \
        if (maskType == DataType::BOOLEAN) { \
            LAUNCH_WITH_MASK(LABEL_CPP_TYPE, LOGIT_CPP_TYPE, LOSS_CPP_TYPE, bool); \
        } else if (maskType == DataType::UINT8) { \
            LAUNCH_WITH_MASK(LABEL_CPP_TYPE, LOGIT_CPP_TYPE, LOSS_CPP_TYPE, uint8_t); \
        } else if (maskType == DataType::FP16) { \
            LAUNCH_WITH_MASK(LABEL_CPP_TYPE, LOGIT_CPP_TYPE, LOSS_CPP_TYPE, half); \
        } else if (maskType == DataType::FP32) { \
            LAUNCH_WITH_MASK(LABEL_CPP_TYPE, LOGIT_CPP_TYPE, LOSS_CPP_TYPE, float); \
        } else { \
            THOR_UNREACHABLE(); \
        } \
    } while (false)

#define DISPATCH_LOSS(LABEL_CPP_TYPE, LOGIT_CPP_TYPE) \
    do { \
        if (lossType == DataType::FP16) { \
            DISPATCH_MASK(LABEL_CPP_TYPE, LOGIT_CPP_TYPE, half); \
        } else if (lossType == DataType::FP32) { \
            DISPATCH_MASK(LABEL_CPP_TYPE, LOGIT_CPP_TYPE, float); \
        } else { \
            THOR_UNREACHABLE(); \
        } \
    } while (false)

#define DISPATCH_LOGITS(LABEL_CPP_TYPE) \
    do { \
        if (logitsType == DataType::FP16) { \
            DISPATCH_LOSS(LABEL_CPP_TYPE, half); \
        } else if (logitsType == DataType::FP32) { \
            DISPATCH_LOSS(LABEL_CPP_TYPE, float); \
        } else { \
            THOR_UNREACHABLE(); \
        } \
    } while (false)

    if (labelsType == DataType::UINT8) {
        DISPATCH_LOGITS(uint8_t);
    } else if (labelsType == DataType::UINT16) {
        DISPATCH_LOGITS(uint16_t);
    } else if (labelsType == DataType::UINT32) {
        DISPATCH_LOGITS(uint32_t);
    } else {
        THOR_UNREACHABLE();
    }

#undef DISPATCH_LOGITS
#undef DISPATCH_LOSS
#undef DISPATCH_MASK
#undef LAUNCH_WITH_MASK
}

string SparseCategoricalCrossEntropyWithLogits::getType() { return "SparseCategoricalCrossEntropyWithLogits"; }

vector<Stream> SparseCategoricalCrossEntropyWithLogits::getProcessingStreams() {
    vector<Stream> processingStreams = Loss::getProcessingStreams();
    if (maskStream.isInitialized())
        processingStreams.push_back(maskStream);
    if (activeCountStream.isInitialized())
        processingStreams.push_back(activeCountStream);
    return processingStreams;
}

vector<Event> SparseCategoricalCrossEntropyWithLogits::getSynchronizeEvents() {
    vector<Event> events;
    set<uint64_t> synchronizedStreamIds;
    appendSynchronizeEvent(events, synchronizedStreamIds, stream);
    appendSynchronizeEvent(events, synchronizedStreamIds, labelsStream);
    appendSynchronizeEvent(events, synchronizedStreamIds, maskStream);
    appendSynchronizeEvent(events, synchronizedStreamIds, activeCountStream);
    return events;
}
