#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class DeviceCrossing : public Layer {
   public:
    ~DeviceCrossing() override {}

    DeviceCrossing() { uninitialized = true; }

    DeviceCrossing(TensorPlacement inputPlacement, TensorPlacement outputPlacement) {
        THOR_THROW_IF_FALSE(inputPlacement != outputPlacement);

        uninitialized = false;
        this->inputPlacement = inputPlacement;
        this->outputPlacement = outputPlacement;
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(!uninitialized);
        THOR_THROW_IF_FALSE(featureInput.has_value());
        outputBuffer = featureInput.value().clone();
        finishedCopyEvent = stream.putEvent();
        return Tensor(outputPlacement, featureInput.value().getDescriptor());
    }

    // Crosses from source device to dest device
    // Output is buffered so the stream on the source device is not blocked during copy
    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(!uninitialized);
        THOR_THROW_IF_FALSE(inputTensor.has_value());
        THOR_THROW_IF_FALSE(outputTensor.has_value());
        THOR_THROW_IF_FALSE(outputPlacement != featureInput.value().getPlacement());
        THOR_THROW_IF_FALSE(stream.getGpuNum() == inputPlacement.getDeviceNum());
        THOR_THROW_IF_FALSE(inputTensor.value().getPlacement() == inputPlacement);
        THOR_THROW_IF_FALSE(outputTensor.value().getPlacement() == outputPlacement);

        // Ensure the previous data transfer has finished, so the buffer is available
        stream.waitEvent(finishedCopyEvent);

        // Copy to the on device buffer, then stream is unblocked
        outputBuffer.copyFromAsync(inputTensor.value(), stream);

        // output stream waits for copy to buffer to complete
        // output buffer is offloaded to the other device
        // an event is placed on the output stream to indicate when the offload copy is complete
        otherDeviceStream.waitEvent(stream.putEvent());
        outputTensor.value().copyFromAsync(outputBuffer, otherDeviceStream);
        finishedCopyEvent = otherDeviceStream.putEvent();
    }

    // Crosses from dest device to source device
    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {
        THOR_THROW_IF_FALSE(!uninitialized);
        if (errorOut.has_value()) {
            stream.waitEvent(otherDeviceStream.putEvent());
            errorOut.value().copyFromAsync(errorIn.value(), stream);
        }
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);

        THOR_THROW_IF_FALSE(!this->nextLayer.has_value());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = std::nullopt;

        errorInput = nextLayer->connectToPreviousLayer(
            this, featureOutput, otherDeviceStream, shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);

        // When the next layer says that there is no error back propagation path here, then this layer removes that path
        // from itself and informs the adjacent layer in the back propagation path to do the same.
        if (!errorInput.has_value() && errorOutput.has_value() && previousLayer.has_value()) {
            previousLayer.value()->replaceErrorInput(errorOutput, errorInput);
            errorOutput.reset();
        }

        if (errorInput.has_value() && featureOutput.has_value()) {
            THOR_THROW_IF_FALSE(errorInput.value().getDescriptor() == featureOutput.value().getDescriptor());
            THOR_THROW_IF_FALSE(errorInput.value().getPlacement() == featureOutput.value().getPlacement());
        }

        ensureNoDeviceCrossing();
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!uninitialized);
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == inputPlacement);

        if (outputPlacement.getMemDevice() == TensorPlacement::MemDevices::CPU)
            otherDeviceStream = stream;
        else
            otherDeviceStream = Stream(outputPlacement.getDeviceNum());

        return Layer::connectToPreviousLayer(previousLayer, featureInput, stream, backPropagateError);
    }

    void ensureNoDeviceCrossing() override {
        // device crossing allowed here.
    }

   private:
    bool uninitialized;
    TensorPlacement inputPlacement;
    TensorPlacement outputPlacement;

    Stream otherDeviceStream;
    Tensor outputBuffer;
    Event finishedCopyEvent;
};

}  // namespace ThorImplementation
