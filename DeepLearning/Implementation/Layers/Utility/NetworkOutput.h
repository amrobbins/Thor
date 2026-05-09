#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class NetworkOutput : public Layer {
   public:
    ~NetworkOutput() override {}

    NetworkOutput(Optional<TensorPlacement> outputPlacement) : outputPlacement(outputPlacement) {}

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override { THOR_UNREACHABLE(); }

    Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        THOR_THROW_IF_FALSE(this->previousLayer.isEmpty());
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        THOR_THROW_IF_FALSE(this->featureInput.isEmpty());

        this->featureInput = featureInput;
        this->previousLayer = previousLayer;
        this->stream = stream;

        if (featureInput.isPresent())
            featureOutput = createFeatureOutputTensor();

        // No backward error tensor:
        return Optional<Tensor>::empty();
    }

    virtual Event getOutputReadyEvent() { return outputReadyEvent; }

    Optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.isEmpty() == outputPlacement.isEmpty());

        if (outputPlacement.isEmpty()) {
            return Optional<Tensor>::empty();
        } else if (outputPlacement.get() != featureInput.get().getPlacement()) {
            // Create an on device output buffer so that the main stream is not blocked
            // during offloading of the output across devices
            outputBuffer = featureInput.get().clone();
            outputStream = Stream::getNextDownloadStream(featureInput.get().getPlacement().getDeviceNum());
            outputReadyEvent = outputStream.get().putEvent(false, true);
            return featureInput.get().clone(outputPlacement.get());
        } else {
            return featureInput.get().clone(outputPlacement.get());
        }
    }

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.isPresent() == outputTensor.isPresent());

        if (inputTensor.isPresent()) {
            if (outputPlacement.get() == featureInput.get().getPlacement()) {
                outputTensor.get().copyFromAsync(inputTensor, stream);
                outputReadyEvent = stream.putEvent(false, true);
            } else {
                THOR_THROW_IF_FALSE(outputBuffer.isPresent());
                THOR_THROW_IF_FALSE(outputStream.isPresent());

                // Ensure that the previous offload has completed:
                stream.waitEvent(outputReadyEvent);

                // Copy to the on device buffer, then stream is unblocked
                outputBuffer.get().copyFromAsync(inputTensor, stream);

                // output stream waits for copy to buffer to complete
                // output buffer is offloaded to the other device
                // an event is placed on the output stream to indicate when the offload copy is complete
                outputStream.get().waitEvent(stream.putEvent());
                outputTensor.get().copyFromAsync(outputBuffer, outputStream);
                outputReadyEvent = outputStream.get().putEvent(false, true);
            }
        }
    }

    void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) override {}

    void backward(Optional<Tensor> errorInput, uint32_t batchSize = 0) override {}

   protected:
    Event outputReadyEvent;

    Optional<TensorPlacement> outputPlacement;

    Optional<Tensor> outputBuffer;
    Optional<Stream> outputStream;
};

}  // namespace ThorImplementation
