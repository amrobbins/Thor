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
                                                                                 optional<uint32_t> ignoreIndex)
    : Loss(lossDataType), ignoreIndex(ignoreIndex), lossWeight(normalizeLossWeight(lossWeight)) {}

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

void SparseCategoricalCrossEntropyWithLogits::initialize() {
    Loss::initialize();
    maskReceived = false;
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

    const uint64_t effectiveRows = featureInputDimensions.size() == 1 ? 1 : productDimensions(featureInputDimensions, 0, featureInputDimensions.size() - 1);
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

void SparseCategoricalCrossEntropyWithLogits::infer(optional<Tensor> logits, optional<Tensor> loss, Stream stream) {
    THOR_THROW_IF_FALSE(logits.has_value());
    THOR_THROW_IF_FALSE(loss.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(compiled);

    ScopedGpu scopedGpu(logits.value().getPlacement().getDeviceNum());
    stream.waitEvent(labelsStream.putEvent());
    if (maskInput.has_value())
        stream.waitEvent(maskStream.putEvent());

    launchForCurrentTypes();

    labelsStream.waitEvent(stream.putEvent());
    if (maskInput.has_value())
        maskStream.waitEvent(stream.putEvent());
}

void SparseCategoricalCrossEntropyWithLogits::backProp(optional<Tensor> labels, optional<Tensor> logits, optional<Tensor> lossGradient, Stream stream) {
    (void)labels;
    (void)logits;
    (void)stream;
    THOR_THROW_IF_FALSE(lossGradient.has_value());
    THOR_THROW_IF_FALSE(lossGradient.value().getDataType() == DataType::FP32 || lossGradient.value().getDataType() == DataType::FP16);
}

void SparseCategoricalCrossEntropyWithLogits::forward(optional<Tensor> inputTensor, bool validationPass, uint32_t batchSize) {
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
        if (maskInput.has_value() && inputTensor.value() == maskInput.value()) {
            THOR_THROW_IF_FALSE(maskReceived == false);
            maskReceived = true;
            if (featureInputReceived && labelsReceived) {
                stream.waitEvent(labelsStream.putEvent());
                stream.waitEvent(maskStream.putEvent());
                forward(nullopt, validationPass);
            }
            return;
        }
        THOR_UNREACHABLE();
    }

    THOR_THROW_IF_FALSE(!inputTensor.has_value());
    THOR_THROW_IF_FALSE(featureInputReceived);
    THOR_THROW_IF_FALSE(labelsReceived);
    THOR_THROW_IF_FALSE(!maskInput.has_value() || maskReceived);
    featureInputReceived = false;
    labelsReceived = false;
    maskReceived = false;

    infer(featureInput, featureOutput, stream);

    if (nextLayer.has_value())
        nextLayer.value()->forward(featureOutput, validationPass, currentBatchSize);

    if (isInferenceOnly() || validationPass)
        return;

    THOR_THROW_IF_FALSE(previousLayer.has_value());
    backward(nullopt, currentBatchSize);
}


void SparseCategoricalCrossEntropyWithLogits::advanceDataIfReady(bool validationPass) {
    if (featureInputReceived && labelsReceived && (!maskInput.has_value() || maskReceived)) {
        stream.waitEvent(labelsStream.putEvent());
        if (maskInput.has_value())
            stream.waitEvent(maskStream.putEvent());
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
        stream)

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

vector<Event> SparseCategoricalCrossEntropyWithLogits::getSynchronizeEvents() {
    vector<Event> events;
    set<uint64_t> synchronizedStreamIds;
    appendSynchronizeEvent(events, synchronizedStreamIds, stream);
    appendSynchronizeEvent(events, synchronizedStreamIds, labelsStream);
    appendSynchronizeEvent(events, synchronizedStreamIds, maskStream);
    return events;
}
