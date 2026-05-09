#pragma once

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

    Optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(!uninitialized);
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        outputBuffer = featureInput.get().clone();
        finishedCopyEvent = stream.putEvent();
        return Tensor(outputPlacement, featureInput.get().getDescriptor());
    }

    // Crosses from source device to dest device
    // Output is buffered so the stream on the source device is not blocked during copy
    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(!uninitialized);
        THOR_THROW_IF_FALSE(inputTensor.isPresent());
        THOR_THROW_IF_FALSE(outputTensor.isPresent());
        THOR_THROW_IF_FALSE(outputPlacement != featureInput.get().getPlacement());
        THOR_THROW_IF_FALSE(stream.getGpuNum() == inputPlacement.getDeviceNum());
        THOR_THROW_IF_FALSE(inputTensor.get().getPlacement() == inputPlacement);
        THOR_THROW_IF_FALSE(outputTensor.get().getPlacement() == outputPlacement);

        // Ensure the previous data transfer has finished, so the buffer is available
        stream.waitEvent(finishedCopyEvent);

        // Copy to the on device buffer, then stream is unblocked
        outputBuffer.copyFromAsync(inputTensor, stream);

        // output stream waits for copy to buffer to complete
        // output buffer is offloaded to the other device
        // an event is placed on the output stream to indicate when the offload copy is complete
        otherDeviceStream.waitEvent(stream.putEvent());
        outputTensor.get().copyFromAsync(outputBuffer, otherDeviceStream);
        finishedCopyEvent = otherDeviceStream.putEvent();
    }

    // Crosses from dest device to source device
    void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) override {
        THOR_THROW_IF_FALSE(!uninitialized);
        if (errorOut.isPresent()) {
            stream.waitEvent(otherDeviceStream.putEvent());
            errorOut.get().copyFromAsync(errorIn, stream);
        }
    }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!compiled);

        THOR_THROW_IF_FALSE(this->nextLayer.isEmpty());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = Optional<Tensor>::empty();

        errorInput = nextLayer->connectToPreviousLayer(
            this, featureOutput, otherDeviceStream, shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);

        // When the next layer says that there is no error back propagation path here, then this layer removes that path
        // from itself and informs the adjacent layer in the back propagation path to do the same.
        if (errorInput.isEmpty() && errorOutput.isPresent() && previousLayer.isPresent()) {
            previousLayer.get()->replaceErrorInput(errorOutput, errorInput);
            errorOutput.clear();
        }

        if (errorInput.isPresent() && featureOutput.isPresent()) {
            THOR_THROW_IF_FALSE(errorInput.get().getDescriptor() == featureOutput.get().getDescriptor());
            THOR_THROW_IF_FALSE(errorInput.get().getPlacement() == featureOutput.get().getPlacement());
        }

        ensureNoDeviceCrossing();
    }

    Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!uninitialized);
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        THOR_THROW_IF_FALSE(featureInput.get().getPlacement() == inputPlacement);

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
