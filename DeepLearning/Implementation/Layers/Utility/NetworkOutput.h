#pragma once

#include <optional>
#include <vector>
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

        if (featureInput.has_value()) {
            featureOutput = createFeatureOutputTensor();
            initializeOutputSlotZero();
        }

        // No backward error tensor:
        return std::nullopt;
    }

    virtual Event getOutputReadyEvent() { return getOutputReadyEventForSlot(0); }

    virtual Event getOutputReadyEventForSlot(uint32_t slotIndex) {
        requireOutputSlot(slotIndex);
        return outputSlots[slotIndex].outputReadyEvent;
    }

    virtual void extendOutputWritableEvent(Event event) { extendOutputWritableEventForSlot(0, event); }

    virtual void extendOutputWritableEventForSlot(uint32_t slotIndex, Event event) {
        requireOutputSlot(slotIndex);
        outputSlots[slotIndex].outputWritableEvent = event;
    }

    virtual void setActiveOutputSlot(uint32_t slotIndex) {
        requireOutputSlot(slotIndex);
        activeOutputSlot = slotIndex;
    }

    virtual std::optional<Tensor> getFeatureOutputForSlot(uint32_t slotIndex) {
        requireOutputSlot(slotIndex);
        return outputSlots[slotIndex].outputTensor;
    }

    virtual void preallocateOutputSlots(uint32_t numSlots) {
        THOR_THROW_IF_FALSE(numSlots >= 1);
        if (!outputPlacement.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        allocateOutputSlots(numSlots);
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(!featureInput.has_value() == !outputPlacement.has_value());

        if (!outputPlacement.has_value()) {
            return std::nullopt;
        } else {
            return featureInput.value().clone(outputPlacement.value());
        }
    }

    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        (void)validationPass;
        (void)batchSize;
        THOR_THROW_IF_FALSE(running);
        THOR_THROW_IF_FALSE(outputSlots.size() > activeOutputSlot);
        infer(featureInput, outputSlots[activeOutputSlot].outputTensor, stream);
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        THOR_THROW_IF_FALSE(inputTensor.has_value() == outputTensor.has_value());

        if (inputTensor.has_value()) {
            THOR_THROW_IF_FALSE(outputPlacement.has_value());
            OutputSlot& slot = outputSlots[activeOutputSlot];
            THOR_THROW_IF_FALSE(slot.outputTensor.has_value());
            THOR_THROW_IF_FALSE(slot.outputTensor.value() == outputTensor.value());

            if (outputPlacement.value() == featureInput.value().getPlacement()) {
                if (slot.outputWritableEvent.isInitialized()) {
                    stream.waitEvent(slot.outputWritableEvent);
                }
                outputTensor.value().copyFromAsync(inputTensor.value(), stream);
                stream.putEvent(slot.outputReadyEvent, false, true);
                slot.outputWritableEvent = slot.outputReadyEvent;
            } else {
                THOR_THROW_IF_FALSE(slot.outputBuffer.has_value());
                THOR_THROW_IF_FALSE(outputStream.has_value());

                // Ensure that this slot's previous offload has completed before reusing
                // the slot-local GPU scratch buffer.  Other slots may continue to offload
                // independently, so queued training only waits when the same slot is reused.
                if (slot.outputReadyEvent.isInitialized()) {
                    stream.waitEvent(slot.outputReadyEvent);
                }

                // Copy to the on-device buffer, then the main stream is unblocked.
                slot.outputBuffer.value().copyFromAsync(inputTensor.value(), stream);

                // The output stream waits for the slot-local GPU copy to complete.  The
                // public output tensor for this slot may still be owned by a queued host
                // stats callback, so only the offload stream waits for the slot-local
                // output-writable event.  The main input/compute stream only waits for
                // this slot's previous outputReadyEvent above, which means unrelated slots
                // do not serialize on CPU output consumption.
                stream.putEvent(slot.outputBufferReadyEvent);
                outputStream.value().waitEvent(slot.outputBufferReadyEvent);
                if (slot.outputWritableEvent.isInitialized()) {
                    outputStream.value().waitEvent(slot.outputWritableEvent);
                }
                outputTensor.value().copyFromAsync(slot.outputBuffer.value(), outputStream.value());
                outputStream.value().putEvent(slot.outputReadyEvent, false, true);
                slot.outputWritableEvent = slot.outputReadyEvent;
            }
        }
    }

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {}

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {}

   protected:
    struct OutputSlot {
        std::optional<Tensor> outputTensor;
        std::optional<Tensor> outputBuffer;
        Event outputReadyEvent;
        Event outputWritableEvent;
        Event outputBufferReadyEvent;
    };

    void initializeOutputSlotZero() {
        THOR_THROW_IF_FALSE(featureOutput.has_value() == outputPlacement.has_value());
        if (!outputPlacement.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(outputSlots.empty());
        OutputSlot slot;
        slot.outputTensor = featureOutput;
        if (outputPlacement.value() != featureInput.value().getPlacement()) {
            slot.outputBuffer = featureInput.value().clone();
            outputStream = Stream::getNextDownloadStream(featureInput.value().getPlacement().getDeviceNum());
        }
        outputSlots.push_back(slot);
    }

    void requireOutputSlot(uint32_t slotIndex) const {
        THOR_THROW_IF_FALSE(outputPlacement.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(slotIndex < outputSlots.size());
    }

    void allocateOutputSlots(uint32_t numSlots) {
        THOR_THROW_IF_FALSE(outputPlacement.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(numSlots >= 1);

        if (outputSlots.empty()) {
            initializeOutputSlotZero();
        }

        while (outputSlots.size() < numSlots) {
            OutputSlot slot;
            slot.outputTensor = featureInput.value().clone(outputPlacement.value());
            if (outputPlacement.value() != featureInput.value().getPlacement()) {
                slot.outputBuffer = featureInput.value().clone();
                if (!outputStream.has_value()) {
                    outputStream = Stream::getNextDownloadStream(featureInput.value().getPlacement().getDeviceNum());
                }
            }
            outputSlots.push_back(slot);
        }
    }

    std::optional<TensorPlacement> outputPlacement;

    uint32_t activeOutputSlot = 0;
    std::vector<OutputSlot> outputSlots;
    std::optional<Stream> outputStream;
};

}  // namespace ThorImplementation
