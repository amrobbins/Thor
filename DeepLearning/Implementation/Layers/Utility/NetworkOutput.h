#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class NetworkOutput : public Layer {
   public:
    ~NetworkOutput() override {}

    NetworkOutput(std::optional<TensorPlacement> outputPlacement) : outputPlacement(outputPlacement) {}

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override { THOR_UNREACHABLE(); }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        THOR_THROW_IF_FALSE(!this->previousLayer.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->featureInput.has_value());

        this->featureInput = featureInput;
        this->previousLayer = previousLayer;
        this->stream = stream;

        if (featureInput.has_value())
            featureOutput = createFeatureOutputTensor();

        // No backward error tensor:
        return std::nullopt;
    }

    virtual Event getOutputReadyEvent() { return outputReadyEvent; }

    virtual void extendOutputWritableEvent(Event event) { outputWritableEvent = event; }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(!featureInput.has_value() == !outputPlacement.has_value());

        if (!outputPlacement.has_value()) {
            return std::nullopt;
        } else if (outputPlacement.value() != featureInput.value().getPlacement()) {
            // Create an on device output buffer so that the main stream is not blocked
            // during offloading of the output across devices
            outputBuffer = featureInput.value().clone();
            outputStream = Stream::getNextDownloadStream(featureInput.value().getPlacement().getDeviceNum());
            outputReadyEvent = outputStream.value().putEvent(false, true);
            outputWritableEvent = outputReadyEvent;
            return featureInput.value().clone(outputPlacement.value());
        } else {
            return featureInput.value().clone(outputPlacement.value());
        }
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.has_value() == outputTensor.has_value());

        if (inputTensor.has_value()) {
            if (outputPlacement.value() == featureInput.value().getPlacement()) {
                if (outputWritableEvent.isInitialized()) {
                    stream.waitEvent(outputWritableEvent);
                }
                outputTensor.value().copyFromAsync(inputTensor.value(), stream);
                stream.putEvent(outputReadyEvent, false, true);
                outputWritableEvent = outputReadyEvent;
            } else {
                THOR_THROW_IF_FALSE(outputBuffer.has_value());
                THOR_THROW_IF_FALSE(outputStream.has_value());

                // Ensure that the previous offload has completed:
                stream.waitEvent(outputReadyEvent);

                // Copy to the on device buffer, then stream is unblocked
                outputBuffer.value().copyFromAsync(inputTensor.value(), stream);

                // output stream waits for copy to buffer to complete.  The public output tensor
                // may still be owned by a queued host stats callback, so only the offload stream
                // waits for the output-writable event.  The main input/compute stream only waits
                // for outputReadyEvent above, which means the GPU scratch outputBuffer is reusable.
                stream.putEvent(outputBufferReadyEvent);
                outputStream.value().waitEvent(outputBufferReadyEvent);
                if (outputWritableEvent.isInitialized()) {
                    outputStream.value().waitEvent(outputWritableEvent);
                }
                outputTensor.value().copyFromAsync(outputBuffer.value(), outputStream.value());
                outputStream.value().putEvent(outputReadyEvent, false, true);
                outputWritableEvent = outputReadyEvent;
            }
        }
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {}

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {}

   protected:
    Event outputReadyEvent;
    Event outputWritableEvent;
    Event outputBufferReadyEvent;

    std::optional<TensorPlacement> outputPlacement;

    std::optional<Tensor> outputBuffer;
    std::optional<Stream> outputStream;
};

}  // namespace ThorImplementation
