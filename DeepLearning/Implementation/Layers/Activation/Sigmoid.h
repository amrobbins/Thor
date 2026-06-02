#pragma once

#include <optional>
#include <cudnn.h>

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Sigmoid : public Activation {
   public:
    Sigmoid() { this->backwardComputedExternally = false; }
    Sigmoid(bool backwardComputedExternally) { this->backwardComputedExternally = backwardComputedExternally; }

    ~Sigmoid() override {}

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone();
    }

    void postCompile() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());

        cudnnStatus_t cudnnStatus = cudnnCreateActivationDescriptor(&activationDescriptor);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
        cudnnStatus = cudnnSetActivationDescriptor(activationDescriptor, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0);
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnTensorDescriptor = createCudnnTensorDescriptor(featureInput.value().getDescriptor().getDimensions(),
                                                            featureInput.value().getDescriptor().getDataType());

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

    void cleanup() override {
        if (compiled) {
            cudnnStatus_t cudnnStatus;
            cudnnStatus = cudnnDestroyTensorDescriptor(cudnnTensorDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            cudnnTensorDescriptor = nullptr;

            cudnnStatus = cudnnDestroyActivationDescriptor(activationDescriptor);
            THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
            activationDescriptor = nullptr;
        }
        Layer::cleanup();
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.has_value());
        THOR_THROW_IF_FALSE(outputTensor.has_value());
        TensorPlacement placement = inputTensor.value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

        cudnnStatus_t cudnnStatus = cudnnActivationForward(stream.getCudnnHandle(),
                                                           activationDescriptor,
                                                           &ALPHA_NO_SCALE,
                                                           cudnnTensorDescriptor,
                                                           inputTensor.value().getMemPtr(),
                                                           &BETA_CLEAR,
                                                           cudnnTensorDescriptor,
                                                           outputTensor.value().getMemPtr());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        THOR_THROW_IF_FALSE(dataIn.has_value());
        THOR_THROW_IF_FALSE(errorIn.has_value());
        THOR_THROW_IF_FALSE(errorOut.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        TensorPlacement placement = errorOut.value().getPlacement();
        THOR_THROW_IF_FALSE(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);

        if (backwardComputedExternally)
            return;

        cudnnStatus_t cudnnStatus = cudnnActivationBackward(stream.getCudnnHandle(),
                                                            activationDescriptor,
                                                            &ALPHA_NO_SCALE,
                                                            cudnnTensorDescriptor,
                                                            featureOutput.value().getMemPtr(),
                                                            cudnnTensorDescriptor,
                                                            errorIn.value().getMemPtr(),
                                                            cudnnTensorDescriptor,
                                                            dataIn.value().getMemPtr(),
                                                            &BETA_CLEAR,
                                                            cudnnTensorDescriptor,
                                                            errorOut.value().getMemPtr());
        THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    }

    std::string getType() override { return "Sigmoid"; }

   private:
    static constexpr float ALPHA_NO_SCALE = 1.0f;
    static constexpr float BETA_CLEAR = 0.0f;

    cudnnActivationDescriptor_t activationDescriptor = nullptr;
    cudnnTensorDescriptor_t cudnnTensorDescriptor = nullptr;
    bool backwardComputedExternally;
};

}  // namespace ThorImplementation
