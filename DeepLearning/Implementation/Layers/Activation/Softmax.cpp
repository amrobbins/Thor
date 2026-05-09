#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"

#include <cudnn.h>

#include "DeepLearning/Implementation/ThorError.h"
using namespace ThorImplementation;

const float Softmax::ALPHA_NO_SCALE = 1.0f;
const float Softmax::BETA_CLEAR = 0.0f;

Softmax::Softmax() { backwardComputedExternally = false; }

Softmax::Softmax(bool backwardComputedExternally) { this->backwardComputedExternally = backwardComputedExternally; }

Optional<Tensor> Softmax::createFeatureOutputTensor() {
    THOR_THROW_IF_FALSE(featureInput.isPresent());
    THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDimensions().size() == 2);
    return featureInput.get().clone();
}

void Softmax::postCompile() {
    cudnnTensorDescriptor =
        createCudnnTensorDescriptor(featureInput.get().getDescriptor().getDimensions(), featureInput.get().getDescriptor().getDataType());

    if (backwardComputedExternally) {
        // ErrorInput to the previous layer is the errorInput coming to this layer,
        // then backProp is a no op
        if (errorInput.isPresent() && errorOutput.isPresent() && previousLayer.isPresent()) {
            previousLayer.get()->replaceErrorInput(errorOutput, errorInput);
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

void Softmax::infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
    THOR_THROW_IF_FALSE(inputTensor.isPresent());
    THOR_THROW_IF_FALSE(outputTensor.isPresent());
    TensorPlacement placement = inputTensor.get().getPlacement();
    THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

    cudnnStatus_t cudnnStatus;
    cudnnStatus = cudnnSoftmaxForward(stream.getCudnnHandle(),
                                      CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &ALPHA_NO_SCALE,
                                      cudnnTensorDescriptor,
                                      inputTensor.get().getMemPtr(),
                                      &BETA_CLEAR,
                                      cudnnTensorDescriptor,
                                      outputTensor.get().getMemPtr());
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

void Softmax::backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
    THOR_THROW_IF_FALSE(dataIn.isPresent());
    THOR_THROW_IF_FALSE(errorIn.isPresent());
    THOR_THROW_IF_FALSE(errorOut.isPresent());
    THOR_THROW_IF_FALSE(featureOutput.isPresent());
    TensorPlacement placement = errorOut.get().getPlacement();
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
                                       featureOutput.get().getMemPtr(),
                                       cudnnTensorDescriptor,
                                       errorIn.get().getMemPtr(),
                                       &BETA_CLEAR,
                                       cudnnTensorDescriptor,
                                       errorOut.get().getMemPtr());
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

bool Softmax::isBackwardComputedExternally() { return backwardComputedExternally; }

std::string Softmax::getType() { return "Softmax"; }
