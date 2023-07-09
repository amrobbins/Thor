#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class NetworkOutput : public Layer {
   public:
    virtual ~NetworkOutput() {}

    NetworkOutput(Optional<TensorPlacement> outputPlacement) : outputPlacement(outputPlacement) {}

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) { assert(false); }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        assert(this->previousLayer.isEmpty());
        assert(featureInput.isPresent());
        assert(this->featureInput.isEmpty());

        this->featureInput = featureInput;
        this->previousLayer = previousLayer;
        this->stream = stream;

        if (featureInput.isPresent())
            featureOutput = createFeatureOutputTensor();

        // No backward error tensor:
        return Optional<Tensor>::empty();
    }

    virtual Event getOutputReadyEvent() { return outputReadyEvent; }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isEmpty() == outputPlacement.isEmpty());

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

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent() == outputTensor.isPresent());

        if (inputTensor.isPresent()) {
            if (outputPlacement.get() == featureInput.get().getPlacement()) {
                outputTensor.get().copyFromAsync(inputTensor, stream);
                outputReadyEvent = stream.putEvent(false, true);
            } else {
                assert(outputBuffer.isPresent());
                assert(outputStream.isPresent());

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

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {}

    virtual void backward(Optional<Tensor> errorInput) {}

   protected:
    Event outputReadyEvent;

    Optional<TensorPlacement> outputPlacement;

    Optional<Tensor> outputBuffer;
    Optional<Stream> outputStream;
};

}  // namespace ThorImplementation
