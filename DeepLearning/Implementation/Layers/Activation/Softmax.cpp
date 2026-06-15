#include <optional>
#include <vector>
#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"

#include <cudnn.h>

#include "DeepLearning/Implementation/ThorError.h"
using namespace ThorImplementation;

namespace {

std::vector<unsigned long> flattenLastDimensionForSoftmax(const std::vector<unsigned long>& dims) {
    THOR_THROW_IF_FALSE(dims.size() >= 2);
    const unsigned long classes = dims.back();
    THOR_THROW_IF_FALSE(classes > 0);
    unsigned long effectiveBatch = 1;
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        THOR_THROW_IF_FALSE(dims[i] > 0);
        effectiveBatch *= dims[i];
    }
    return {effectiveBatch, classes};
}

}  // namespace

const float Softmax::ALPHA_NO_SCALE = 1.0f;
const float Softmax::BETA_CLEAR = 0.0f;

Softmax::Softmax() { backwardComputedExternally = false; }

Softmax::Softmax(bool backwardComputedExternally) { this->backwardComputedExternally = backwardComputedExternally; }

std::optional<Tensor> Softmax::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions().size() >= 2);
    THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions().back() > 0);
    return featureInput.value().clone();
}

void Softmax::postCompile() {
    std::vector<unsigned long> softmaxDimensions =
        flattenLastDimensionForSoftmax(featureInput.value().getDescriptor().getDimensions());
    cudnnTensorDescriptor = createCudnnTensorDescriptor(softmaxDimensions, featureInput.value().getDescriptor().getDataType());

    if (backwardComputedExternally) {
        // ErrorInput to the previous layer is the errorInput coming to this layer,
        // then backProp is a no op
        if (errorInput.has_value() && errorOutput.has_value() && previousLayer.has_value()) {
            previousLayer.value()->replaceErrorInput(errorOutput, errorInput);
        }
        errorOutput = errorInput;
    }
    Layer::postCompile();
}

void Softmax::cleanup() {
    if (compiled) {
        cudnnStatus_t cudnnStatus;
        cudnnStatus = cudnnDestroyTensorDescriptor(cudnnTensorDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        compiled = false;
    }
}

void Softmax::infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) {
    THOR_THROW_IF_FALSE(inputTensor.has_value());
    THOR_THROW_IF_FALSE(outputTensor.has_value());
    TensorPlacement placement = inputTensor.value().getPlacement();
    THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

    cudnnStatus_t cudnnStatus;
    cudnnStatus = cudnnSoftmaxForward(stream.getCudnnHandle(),
                                      CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &ALPHA_NO_SCALE,
                                      cudnnTensorDescriptor,
                                      inputTensor.value().getMemPtr(),
                                      &BETA_CLEAR,
                                      cudnnTensorDescriptor,
                                      outputTensor.value().getMemPtr());
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

void Softmax::backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) {
    THOR_THROW_IF_FALSE(dataIn.has_value());
    THOR_THROW_IF_FALSE(errorIn.has_value());
    THOR_THROW_IF_FALSE(errorOut.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    TensorPlacement placement = errorOut.value().getPlacement();
    THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

    if (backwardComputedExternally)
        return;

    cudnnStatus_t cudnnStatus;
    cudnnStatus = cudnnSoftmaxBackward(stream.getCudnnHandle(),
                                       CUDNN_SOFTMAX_ACCURATE,
                                       CUDNN_SOFTMAX_MODE_CHANNEL,
                                       &ALPHA_NO_SCALE,
                                       cudnnTensorDescriptor,
                                       // cudnn softmax wants y instead of x, hopefully that is because the math works out
                                       // to where the already computed y can be used for backpropagation.
                                       featureOutput.value().getMemPtr(),
                                       cudnnTensorDescriptor,
                                       errorIn.value().getMemPtr(),
                                       &BETA_CLEAR,
                                       cudnnTensorDescriptor,
                                       errorOut.value().getMemPtr());
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

bool Softmax::isBackwardComputedExternally() { return backwardComputedExternally; }

std::string Softmax::getType() { return "Softmax"; }
