#include "DeepLearning/Implementation/Layers/Loss/BoxIouLoss.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Loss/BoxIouLoss.h"

#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

void validateBoxesDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported BoxIouLoss ") + tensorName + " dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePhysicalBoxDims(const vector<uint64_t>& dims, const string& tensorName) {
    if (!((dims.size() == 2 && dims[1] == 4) || (dims.size() == 3 && dims[2] == 4))) {
        throw runtime_error("BoxIouLoss " + tensorName + " must have physical dimensions [batch, 4] or [batch, boxes, 4].");
    }
}

uint32_t boxesPerBatchElementFromDims(const vector<uint64_t>& dims) {
    validatePhysicalBoxDims(dims, "input");
    return dims.size() == 2 ? 1u : static_cast<uint32_t>(dims[1]);
}

BoxIouLossKind toKernelKind(BoxIouLoss::Kind kind) {
    return static_cast<BoxIouLossKind>(static_cast<uint32_t>(kind));
}

template <typename LabelT, typename PredictionT, typename LossT>
void dispatchBoxIouLoss(void* labels,
                        void* predictions,
                        void* loss,
                        void* gradient,
                        uint32_t numBoxes,
                        BoxIouLoss::Kind kind,
                        float eps,
                        bool computeGradient,
                        float lossScalingFactor,
                        Stream stream) {
    launchBoxIouLoss<LabelT, PredictionT, LossT>(
        labels, predictions, loss, gradient, numBoxes, toKernelKind(kind), eps, computeGradient, lossScalingFactor, stream);
}

}  // namespace

BoxIouLoss::BoxIouLoss(Kind kind, DataType lossDataType, float eps) : Loss(lossDataType), kind(kind), eps(eps) {}

optional<Tensor> BoxIouLoss::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    const vector<uint64_t>& inputDims = featureInput.value().getDescriptor().getDimensions();
    validatePhysicalBoxDims(inputDims, "predictions");

    vector<uint64_t> outputDims;
    if (inputDims.size() == 2) {
        outputDims = {inputDims[0], 1};
    } else {
        outputDims = {inputDims[0], inputDims[1]};
    }
    return Tensor(featureInput.value().getPlacement(), TensorDescriptor(lossDataType, outputDims));
}

void BoxIouLoss::compileImpl() {
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

    const TensorDescriptor& predictionsDescriptor = featureInput.value().getDescriptor();
    const TensorDescriptor& labelsDescriptor = labelsInput.value().getDescriptor();
    validateBoxesDType("predictions", predictionsDescriptor.getDataType());
    validateBoxesDType("labels", labelsDescriptor.getDataType());
    validatePhysicalBoxDims(predictionsDescriptor.getDimensions(), "predictions");
    THOR_THROW_IF_FALSE(predictionsDescriptor.getDimensions() == labelsDescriptor.getDimensions());
    THOR_THROW_IF_FALSE(lossDataType == DataType::FP16 || lossDataType == DataType::FP32);
    THOR_THROW_IF_FALSE(featureOutput.value().getDescriptor().getDataType() == lossDataType);
    THOR_THROW_IF_FALSE(eps > 0.0f);

    batchSize = static_cast<uint32_t>(predictionsDescriptor.getDimensions()[0]);
    boxesPerBatchElement = boxesPerBatchElementFromDims(predictionsDescriptor.getDimensions());
    totalBoxes = batchSize * boxesPerBatchElement;

    const vector<uint64_t>& outputDims = featureOutput.value().getDescriptor().getDimensions();
    THOR_THROW_IF_FALSE(outputDims.size() == 2);
    THOR_THROW_IF_FALSE(outputDims[0] == batchSize);
    THOR_THROW_IF_FALSE(outputDims[1] == boxesPerBatchElement);

    if (!isInferenceOnly()) {
        THOR_THROW_IF_FALSE(errorOutput.has_value());
        THOR_THROW_IF_FALSE(errorOutput.value().isInitialized());
        THOR_THROW_IF_FALSE(errorOutput.value().getPlacement() == featureInput.value().getPlacement());
        THOR_THROW_IF_FALSE(errorOutput.value().getDescriptor() == featureInput.value().getDescriptor());
    }
}

void BoxIouLoss::infer(optional<Tensor> predictions, optional<Tensor> loss, Stream stream) {
    THOR_THROW_IF_FALSE(predictions.has_value());
    THOR_THROW_IF_FALSE(loss.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    THOR_THROW_IF_FALSE(predictions.value() == featureInput.value());
    THOR_THROW_IF_FALSE(loss.value() == featureOutput.value());

    stream.waitEvent(labelsStream.putEvent());
    launchKernel(!isInferenceOnly(), stream);
}

void BoxIouLoss::backProp(optional<Tensor> labels, optional<Tensor> predictions, optional<Tensor> lossGradient, Stream stream) {
    (void)labels;
    (void)predictions;
    (void)lossGradient;
    (void)stream;
    // The prediction gradient is computed by the forward kernel so that one
    // vectorized per-box pass produces both the raw loss and the scaled dL/dbox.
}

void BoxIouLoss::launchKernel(bool computeGradient, Stream stream) {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(labelsInput.has_value());
    if (computeGradient)
        THOR_THROW_IF_FALSE(errorOutput.has_value());

    ScopedGpu scopedGpu(featureInput.value().getPlacement().getDeviceNum());

    void* gradient = computeGradient ? errorOutput.value().getMemPtr() : nullptr;
    void* labels = labelsInput.value().getMemPtr();
    void* predictions = featureInput.value().getMemPtr();
    void* loss = featureOutput.value().getMemPtr();

    const DataType labelsDType = labelsInput.value().getDescriptor().getDataType();
    const DataType predictionsDType = featureInput.value().getDescriptor().getDataType();
    const DataType outputDType = featureOutput.value().getDescriptor().getDataType();

    if (predictionsDType == DataType::FP16 && labelsDType == DataType::FP16 && outputDType == DataType::FP16) {
        dispatchBoxIouLoss<half, half, half>(labels, predictions, loss, gradient, totalBoxes, kind, eps, computeGradient, getLossScalingFactor(), stream);
    } else if (predictionsDType == DataType::FP16 && labelsDType == DataType::FP16 && outputDType == DataType::FP32) {
        dispatchBoxIouLoss<half, half, float>(labels, predictions, loss, gradient, totalBoxes, kind, eps, computeGradient, getLossScalingFactor(), stream);
    } else if (predictionsDType == DataType::FP16 && labelsDType == DataType::FP32 && outputDType == DataType::FP16) {
        dispatchBoxIouLoss<float, half, half>(labels, predictions, loss, gradient, totalBoxes, kind, eps, computeGradient, getLossScalingFactor(), stream);
    } else if (predictionsDType == DataType::FP16 && labelsDType == DataType::FP32 && outputDType == DataType::FP32) {
        dispatchBoxIouLoss<float, half, float>(labels, predictions, loss, gradient, totalBoxes, kind, eps, computeGradient, getLossScalingFactor(), stream);
    } else if (predictionsDType == DataType::FP32 && labelsDType == DataType::FP16 && outputDType == DataType::FP16) {
        dispatchBoxIouLoss<half, float, half>(labels, predictions, loss, gradient, totalBoxes, kind, eps, computeGradient, getLossScalingFactor(), stream);
    } else if (predictionsDType == DataType::FP32 && labelsDType == DataType::FP16 && outputDType == DataType::FP32) {
        dispatchBoxIouLoss<half, float, float>(labels, predictions, loss, gradient, totalBoxes, kind, eps, computeGradient, getLossScalingFactor(), stream);
    } else if (predictionsDType == DataType::FP32 && labelsDType == DataType::FP32 && outputDType == DataType::FP16) {
        dispatchBoxIouLoss<float, float, half>(labels, predictions, loss, gradient, totalBoxes, kind, eps, computeGradient, getLossScalingFactor(), stream);
    } else if (predictionsDType == DataType::FP32 && labelsDType == DataType::FP32 && outputDType == DataType::FP32) {
        dispatchBoxIouLoss<float, float, float>(labels, predictions, loss, gradient, totalBoxes, kind, eps, computeGradient, getLossScalingFactor(), stream);
    } else {
        THOR_UNREACHABLE();
    }
}
