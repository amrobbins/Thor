#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"

#include <cudnn.h>

using namespace ThorImplementation;

const float Softmax::ALPHA_NO_SCALE = 1.0f;
const float Softmax::BETA_CLEAR = 0.0f;

Softmax::Softmax() { backwardComputedExternally = false; }

Softmax::Softmax(bool backwardComputedExternally) { this->backwardComputedExternally = backwardComputedExternally; }

Optional<Tensor> Softmax::createFeatureOutputTensor() {
    assert(featureInput.isPresent());
    assert(featureInput.get().getDescriptor().getDimensions().size() == 2);
    return featureInput.get().clone();
}

void Softmax::compile() {
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
}

void Softmax::cleanup() {
    cudnnStatus_t cudnnStatus;
    cudnnStatus = cudnnDestroyTensorDescriptor(cudnnTensorDescriptor);
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

void Softmax::infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
    assert(inputTensor.isPresent());
    assert(outputTensor.isPresent());
    TensorPlacement placement = inputTensor.get().getPlacement();
    assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

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
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

void Softmax::backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
    assert(dataIn.isPresent());
    assert(errorIn.isPresent());
    assert(errorOut.isPresent());
    assert(featureOutput.isPresent());
    TensorPlacement placement = errorOut.get().getPlacement();
    assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

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
    assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
}

std::string Softmax::getType() { return "Softmax"; }