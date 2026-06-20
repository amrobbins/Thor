#pragma once

#include <optional>
#include <utility>
#include <vector>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/LayerSubmitDiagnostics.h"

namespace ThorImplementation {

// To run the network forward, for every input NetworkInput, load the tensor and call forward(myInputTensor)
// Data type will be converted if necessary, memory will be copied across devices if necessary.
class NetworkInput : public Layer {
   public:
    ~NetworkInput() override {}

    NetworkInput(TensorPlacement networkPlacement,
                 std::optional<DataType> contentDataType,
                 std::optional<std::vector<unsigned long>> contentDimensions) {
        construct(networkPlacement, contentDataType, contentDimensions);
    }

    NetworkInput(Tensor exampleTensor) {
        THOR_THROW_IF_FALSE(exampleTensor.isInitialized());
        construct(exampleTensor.getPlacement(), exampleTensor.getDescriptor().getDataType(), exampleTensor.getDescriptor().getDimensions());
    }

    void construct(TensorPlacement networkPlacement,
                   std::optional<DataType> contentDataType,
                   std::optional<std::vector<unsigned long>> contentDimensions) {
        THOR_THROW_IF_FALSE(contentDimensions.has_value() == contentDataType.has_value());
        this->networkPlacement = networkPlacement;
        this->contentDataType = contentDataType;
        this->contentDimensions = contentDimensions;
        int gpuNum = 0;
        if (networkPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU)
            gpuNum = networkPlacement.getDeviceNum();
        this->stream = Stream(gpuNum);
        this->loadStream = Stream::getNextUploadStream(gpuNum);
    }

    virtual bool isInput() { return true; }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!this->nextLayer.has_value());

        this->nextLayer = nextLayer;

        featureOutput = createFeatureOutputTensor();
        initializeInputSlotZero();

        nextLayer->connectToPreviousLayer(this, featureOutput, stream, false, loaderConnectionType);
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        THOR_UNREACHABLE();
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        if (contentDimensions.has_value())
            return Tensor(networkPlacement, TensorDescriptor(contentDataType.value(), contentDimensions.value()));
        return std::nullopt;
    }

    virtual void setActiveInputSlot(uint32_t slotIndex) {
        if (!contentDimensions.has_value()) {
            return;
        }
        requireInputSlot(slotIndex);
        activeInputSlot = slotIndex;
    }

    virtual void preallocateInputSlots(uint32_t numSlots) {
        THOR_THROW_IF_FALSE(numSlots >= 1);
        if (!contentDimensions.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        allocateInputSlots(numSlots);
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {}
    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {}

    // Only called for input endpoints
    // When the source tensor that will be sent to the input is loaded via a stream,
    // then this version of forward is used which causes the loader stream to wait till copy is finished.
    virtual void forward(std::optional<Tensor> featureInput,
                         bool validationPass,
                         Event copyToSourceTensorFinished,
                         uint32_t batchSize = 0) {
        loadStream.waitEvent(copyToSourceTensorFinished);
        forward(featureInput, validationPass, batchSize);
    }

    // Only called for input endpoints
    // This version of forward expects that the memory in featureInput has already been populated before forward is called.
    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        const bool emitDiagnostics = layerSubmitDiagnosticsActive();
        const auto totalStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        THOR_THROW_IF_FALSE(contentDimensions.has_value() == featureInput.has_value());

        if (contentDimensions.has_value())
            THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == contentDimensions.value());

        const auto copyStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        uint64_t copyMicros = 0;
        if (contentDimensions.has_value()) {
            THOR_THROW_IF_FALSE(featureOutput.has_value());
            THOR_THROW_IF_FALSE(activeInputSlot < inputSlots.size());
            InputSlot& slot = inputSlots[activeInputSlot];
            THOR_THROW_IF_FALSE(slot.outputBuffer.has_value());

            // Only wait when reusing the same slot-local prefetch buffer.  Other
            // native-queued slots may upload into their own buffers while this
            // slot's previous buffer -> featureOutput copy is still pending on
            // the processing stream.
            loadStream.waitEvent(slot.outputBufferWritableEvent);

            // Copy into the slot-local prefetch buffer using the upload stream.
            slot.outputBuffer.value().copyFromAsync(featureInput.value(), loadStream);
            loadStream.putEvent(slot.outputBufferLoadedEvent);
            stream.waitEvent(slot.outputBufferLoadedEvent);

            // Copy from the slot-local prefetch buffer into the connected public
            // NetworkInput feature tensor.  The public feature tensor remains
            // single-address/statically connected to downstream layers, so this
            // ring only removes false dependencies on the staging buffer without
            // changing layer-graph tensor identities.
            featureOutput.value().copyFromAsync(slot.outputBuffer.value(), stream);
            stream.putEvent(slot.outputBufferWritableEvent);
        }
        if (emitDiagnostics) {
            copyMicros = layerSubmitDiagnosticElapsedMicros(copyStart, layerSubmitDiagnosticNow());
        }

        if (!nextLayer.has_value()) {
            if (emitDiagnostics) {
                emitLayerSubmitDiagnostic("network_input_forward",
                                          layerSubmitDiagnosticLabel("NetworkInput", getId(), getName()),
                                          getId(),
                                          layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                                          {{"copy_us", copyMicros}, {"downstream_us", 0}, {"has_content", contentDimensions.has_value() ? 1UL : 0UL}});
            }
            return;
        }

        const auto downstreamStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        nextLayer.value()->forward(featureOutput, validationPass, batchSize);
        const uint64_t downstreamMicros =
            emitDiagnostics ? layerSubmitDiagnosticElapsedMicros(downstreamStart, layerSubmitDiagnosticNow()) : 0;

        if (emitDiagnostics) {
            emitLayerSubmitDiagnostic("network_input_forward",
                                      layerSubmitDiagnosticLabel("NetworkInput", getId(), getName()),
                                      getId(),
                                      layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                                      {{"copy_us", copyMicros}, {"downstream_us", downstreamMicros}, {"has_content", contentDimensions.has_value() ? 1UL : 0UL}});
        }
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {}

   protected:
    struct InputSlot {
        std::optional<Tensor> outputBuffer;
        Event outputBufferLoadedEvent;
        Event outputBufferWritableEvent;
    };

    void initializeInputSlotZero() {
        THOR_THROW_IF_FALSE(featureOutput.has_value() == contentDimensions.has_value());
        if (!contentDimensions.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(inputSlots.empty());
        InputSlot slot;
        slot.outputBuffer = createFeatureOutputTensor();
        // Pre-initialize and record the slot-local events so the hot path never
        // allocates Events and first use can wait on already-complete events.
        loadStream.putEvent(slot.outputBufferLoadedEvent);
        loadStream.putEvent(slot.outputBufferWritableEvent);
        inputSlots.push_back(slot);
    }

    void requireInputSlot(uint32_t slotIndex) const {
        if (!contentDimensions.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(slotIndex < inputSlots.size());
    }

    void allocateInputSlots(uint32_t numSlots) {
        THOR_THROW_IF_FALSE(contentDimensions.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(numSlots >= 1);

        if (inputSlots.empty()) {
            initializeInputSlotZero();
        }

        while (inputSlots.size() < numSlots) {
            InputSlot slot;
            slot.outputBuffer = createFeatureOutputTensor();
            // Eagerly allocate/reuse-initialize the events as well as the tensor.
            loadStream.putEvent(slot.outputBufferLoadedEvent);
            loadStream.putEvent(slot.outputBufferWritableEvent);
            inputSlots.push_back(slot);
        }
    }

    std::optional<std::vector<unsigned long>> contentDimensions;
    TensorPlacement networkPlacement;
    std::optional<DataType> contentDataType;

    uint32_t activeInputSlot = 0;
    std::vector<InputSlot> inputSlots;
    Stream loadStream;
};

}  // namespace ThorImplementation
