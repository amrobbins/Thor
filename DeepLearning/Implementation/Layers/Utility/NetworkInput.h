#pragma once

#include <optional>
#include <utility>
#include <vector>
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Data/BatchFieldSource.h"
#include "DeepLearning/Api/Data/DeviceBatchReference.h"
#include "DeepLearning/Api/Data/BatchSourceResource.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/LayerSubmitDiagnostics.h"

namespace ThorImplementation {

// To run the network forward, for every input NetworkInput, load the tensor and call forward(myInputTensor)
// Data type will be converted if necessary, memory will be copied across devices if necessary.
class NetworkInput : public Layer {
   public:
    ~NetworkInput() override {}

    enum class Mode { ExternalLoad, DeviceLoad, DeviceReference, PassThrough };

    NetworkInput(TensorPlacement networkPlacement,
                 std::optional<DataType> contentDataType,
                 std::optional<std::vector<unsigned long>> contentDimensions,
                 bool aliasSamePlacementInputs = false) {
        construct(networkPlacement, contentDataType, contentDimensions, Mode::ExternalLoad, aliasSamePlacementInputs, std::nullopt);
    }

    NetworkInput(TensorPlacement networkPlacement,
                 std::optional<DataType> contentDataType,
                 std::optional<std::vector<unsigned long>> contentDimensions,
                 Mode mode,
                 std::optional<Tensor> passThroughSourceTensor = std::nullopt) {
        construct(networkPlacement, contentDataType, contentDimensions, mode, false, passThroughSourceTensor);
    }

    NetworkInput(Tensor exampleTensor, bool aliasSamePlacementInputs = false) {
        THOR_THROW_IF_FALSE(exampleTensor.isInitialized());
        construct(exampleTensor.getPlacement(),
                  exampleTensor.getDescriptor().getDataType(),
                  exampleTensor.getDescriptor().getDimensions(),
                  Mode::ExternalLoad,
                  aliasSamePlacementInputs,
                  std::nullopt);
    }

    static NetworkInput passThrough(TensorPlacement networkPlacement,
                                    std::optional<DataType> contentDataType,
                                    std::optional<std::vector<unsigned long>> contentDimensions,
                                    Tensor sourceTensor) {
        return NetworkInput(networkPlacement, contentDataType, contentDimensions, Mode::PassThrough, sourceTensor);
    }

    void construct(TensorPlacement networkPlacement,
                   std::optional<DataType> contentDataType,
                   std::optional<std::vector<unsigned long>> contentDimensions,
                   Mode mode = Mode::ExternalLoad,
                   bool aliasSamePlacementInputs = false,
                   std::optional<Tensor> passThroughSourceTensor = std::nullopt) {
        THOR_THROW_IF_FALSE(contentDimensions.has_value() == contentDataType.has_value());
        THOR_THROW_IF_FALSE(mode == Mode::ExternalLoad || !aliasSamePlacementInputs);
        THOR_THROW_IF_FALSE(mode == Mode::PassThrough || !passThroughSourceTensor.has_value());
        THOR_THROW_IF_FALSE((mode != Mode::DeviceLoad && mode != Mode::DeviceReference) ||
                            networkPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU);
        this->networkPlacement = networkPlacement;
        this->contentDataType = contentDataType;
        this->contentDimensions = contentDimensions;
        this->mode = mode;
        this->aliasSamePlacementInputs = aliasSamePlacementInputs;
        int gpuNum = 0;
        if (networkPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU)
            gpuNum = networkPlacement.getDeviceNum();
        this->stream = Stream(gpuNum);
        this->loadStream = Stream::getNextUploadStream(gpuNum);

        if (passThroughSourceTensor.has_value()) {
            configurePassThroughSource(passThroughSourceTensor.value());
        }
    }

    std::vector<Event> getSynchronizeEvents() override {
        std::vector<Event> events;
        std::set<uint64_t> synchronizedStreamIds;
        appendSynchronizeEvent(events, synchronizedStreamIds, stream);
        if (!isDeviceLoad() && !isDeviceReferenceLoad()) {
            appendSynchronizeEvent(events, synchronizedStreamIds, loadStream);
        }
        return events;
    }

    virtual bool isInput() { return mode != Mode::PassThrough; }
    virtual bool isPassThrough() const { return mode == Mode::PassThrough; }
    virtual bool isDeviceLoad() const { return mode == Mode::DeviceLoad; }
    virtual bool isDeviceReferenceLoad() const { return mode == Mode::DeviceReference; }
    virtual bool requiresBatchInput() const { return mode != Mode::PassThrough; }

    /**
     * Configures how runtime batch values are loaded. Same-GPU materialized
     * tensors use DeviceLoad and copy directly into featureOutput. Device
     * references use a ring of small reference descriptors whose materializers
     * write directly into featureOutput. Other or unknown tensor placements use
     * the regular full-tensor staging ring. Initial configuration must precede
     * preallocateInputSlots().
     */
    virtual void configureBatchInputSource(const Thor::BatchFieldSourceDescription& source) {
        THOR_THROW_IF_FALSE(!isPassThrough());

        Mode requestedMode = Mode::ExternalLoad;
        if (source.kind == Thor::BatchFieldSourceKind::DEVICE_REFERENCE) {
            THOR_THROW_IF_FALSE(source.placement.has_value());
            THOR_THROW_IF_FALSE(source.placement.value().getMemDevice() == TensorPlacement::MemDevices::GPU);
            THOR_THROW_IF_FALSE(networkPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU);
            THOR_THROW_IF_FALSE(source.placement.value() == networkPlacement);
            requestedMode = Mode::DeviceReference;
        } else {
            const bool directDeviceLoad = source.placement.has_value() &&
                                          source.placement.value().getMemDevice() == TensorPlacement::MemDevices::GPU &&
                                          networkPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU &&
                                          source.placement.value() == networkPlacement;
            requestedMode = directDeviceLoad ? Mode::DeviceLoad : Mode::ExternalLoad;
        }

        // Setup may be repeated with the same mode after slots have been
        // allocated. A real mode change after slot allocation is forbidden
        // because queued users may still own those slot payloads.
        if (!inputSlots.empty()) {
            THOR_THROW_IF_FALSE(mode == requestedMode);
            return;
        }

        mode = requestedMode;
        activeInputSlot = 0;
    }

    virtual void configureBatchInputPlacement(std::optional<TensorPlacement> batchTensorPlacement) {
        configureBatchInputSource(Thor::BatchFieldSourceDescription::materialized(batchTensorPlacement));
    }

    [[nodiscard]] virtual uint32_t getNumInputSlots() const { return static_cast<uint32_t>(inputSlots.size()); }

    // Deprecated compatibility flag from the old runtime-alias attempt.  It is intentionally
    // not used to swap tensors in forward(); correct NetworkInput composition is pass-through
    // mode, where the upstream tensor is bound before downstream layers are connected.
    virtual bool getAliasSamePlacementInputs() const { return aliasSamePlacementInputs; }

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        THOR_THROW_IF_FALSE(!this->nextLayer.has_value());

        this->nextLayer = nextLayer;

        if (isPassThrough()) {
            THOR_THROW_IF_FALSE(featureOutput.has_value());
        } else {
            featureOutput = createFeatureOutputTensor();
        }

        const bool backPropagateThroughInput =
            isPassThrough() && previousLayer.has_value() && shouldConnectToBackPropErrorIn() && !isInferenceOnly();
        errorInput = nextLayer->connectToPreviousLayer(this, featureOutput, stream, backPropagateThroughInput, loaderConnectionType);

        // A pass-through NetworkInput is an identity edge.  If the upstream side
        // was connected first, it returned a provisional errorOutput tensor to
        // the producer.  Once the downstream layer exposes its real error input,
        // fuse the identity by making the producer listen to that downstream
        // tensor directly.  This is the same zero-copy gradient forwarding
        // pattern used by other identity/fanout layers.
        if (isPassThrough() && errorInput.has_value() && errorOutput.has_value() && previousLayer.has_value() &&
            errorOutput.value() != errorInput.value()) {
            previousLayer.value()->replaceErrorInput(errorOutput, errorInput);
            errorOutput = errorInput;
        }
    }

    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) override {
        (void)connectionType;
        THOR_THROW_IF_FALSE(isPassThrough());
        THOR_THROW_IF_FALSE(!this->previousLayer.has_value());
        THOR_THROW_IF_FALSE(!this->featureInput.has_value());
        if (backPropagateError) {
            THOR_THROW_IF_FALSE(featureInput.has_value());
        }

        this->previousLayer = previousLayer;
        this->stream = stream;
        configurePassThroughSource(featureInput);

        if (backPropagateError && !isInferenceOnly()) {
            // If the downstream side is already connected, return that gradient
            // tensor directly.  Otherwise create a temporary identity-gradient
            // edge; connectToNextLayer() will replace it with the downstream
            // tensor as soon as that side is connected.
            if (errorInput.has_value()) {
                errorOutput = errorInput;
            } else {
                errorOutput = featureInput.value().clone();
            }
        } else {
            errorOutput = std::nullopt;
        }

        return errorOutput;
    }

    std::optional<Tensor> createFeatureOutputTensor() override {
        if (isPassThrough()) {
            THOR_THROW_IF_FALSE(featureOutput.has_value());
            return featureOutput;
        }
        if (contentDimensions.has_value())
            return Tensor(networkPlacement, TensorDescriptor(contentDataType.value(), contentDimensions.value()));
        return std::nullopt;
    }

    virtual void setActiveInputSlot(uint32_t slotIndex) {
        if (isPassThrough() || isDeviceLoad() || !contentDimensions.has_value()) {
            return;
        }
        if (inputSlots.empty()) {
            // Synchronous inference does not preallocate a ring; its default
            // slot zero is allocated lazily by forward(). Queued callers must
            // preallocate before selecting any nonzero slot.
            THOR_THROW_IF_FALSE(slotIndex == 0);
            activeInputSlot = 0;
            return;
        }
        requireInputSlot(slotIndex);
        activeInputSlot = slotIndex;
    }

    virtual void preallocateInputSlots(uint32_t numSlots) {
        THOR_THROW_IF_FALSE(numSlots >= 1);
        if (isPassThrough() || isDeviceLoad() || !contentDimensions.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        allocateInputSlots(numSlots);
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {}
    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {}

    // Only called for input endpoints
    // When the source tensor that will be sent to the input is loaded via a stream,
    // then this version of forward is used which causes the batch-source stream to wait till copy is finished.
    virtual void forward(std::optional<Tensor> featureInput,
                         bool validationPass,
                         Event copyToSourceTensorFinished,
                         uint32_t batchSize = 0) {
        forward(
            featureInput,
            validationPass,
            copyToSourceTensorFinished,
            batchSize,
            std::nullopt);
    }

    virtual void forward(
        std::optional<Tensor> featureInput,
        bool validationPass,
        Event copyToSourceTensorFinished,
        uint32_t batchSize,
        std::optional<Thor::BatchSourceReference> sourceReference) {
        if (isPassThrough() || isDeviceLoad()) {
            stream.waitEvent(copyToSourceTensorFinished);
        } else {
            loadStream.waitEvent(copyToSourceTensorFinished);
        }
        forward(
            featureInput,
            validationPass,
            batchSize,
            std::move(sourceReference));
    }

    // Only called for input endpoints
    // This version of forward expects that the memory in featureInput has already been populated before forward is called.
    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        forward(featureInput, validationPass, batchSize, std::nullopt);
    }

    virtual void forward(
        std::optional<Tensor> featureInput,
        bool validationPass,
        uint32_t batchSize,
        std::optional<Thor::BatchSourceReference> sourceReference) {
        const bool emitDiagnostics = layerSubmitDiagnosticsActive();
        const auto totalStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        if (isPassThrough()) {
            validateOptionalForwardTensorMatchesPassThrough(featureInput);
        } else {
            THOR_THROW_IF_FALSE(!isDeviceReferenceLoad());
            THOR_THROW_IF_FALSE(contentDimensions.has_value() == featureInput.has_value());

            if (contentDimensions.has_value())
                THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == contentDimensions.value());
        }

        const auto copyStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        uint64_t copyMicros = 0;
        std::optional<Tensor> downstreamFeatureOutput = featureOutput;
        if (contentDimensions.has_value() && !isPassThrough()) {
            THOR_THROW_IF_FALSE(featureOutput.has_value());
            if (isDeviceLoad()) {
                // Same-GPU materialized inputs copy directly into the statically
                // connected feature tensor. The source-resource event recorded
                // immediately after this copy lets its session slot recycle
                // independently of downstream network completion.
                THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == networkPlacement);
                THOR_THROW_IF_FALSE(networkPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU);
                featureOutput.value().copyFromAsync(featureInput.value(), stream);
                if (sourceReference.has_value()) {
                    sourceReference->recordConsumption(stream);
                }
            } else {
                if (inputSlots.empty()) {
                    // Non-queued inference callers do not preallocate slots.
                    // Preserve that API while keeping queued-training
                    // allocations out of graph placement.
                    allocateInputSlots(1);
                }
                THOR_THROW_IF_FALSE(activeInputSlot < inputSlots.size());
                InputSlot &slot = inputSlots[activeInputSlot];
                THOR_THROW_IF_FALSE(slot.outputBuffer.has_value());

                // Only wait when reusing the same slot-local prefetch buffer.  Other
                // native-queued slots may upload into their own buffers while this
                // slot's previous buffer -> featureOutput copy is still pending on
                // the processing stream.
                loadStream.waitEvent(slot.outputBufferWritableEvent);

                // Copy into the slot-local prefetch buffer using the upload stream.
                slot.outputBuffer.value().copyFromAsync(featureInput.value(), loadStream);
                if (sourceReference.has_value()) {
                    sourceReference->recordConsumption(loadStream);
                }
                loadStream.putEvent(slot.outputBufferLoadedEvent);
                stream.waitEvent(slot.outputBufferLoadedEvent);

                // Copy from the slot-local prefetch buffer into the connected public
                // NetworkInput feature tensor.  The public feature tensor remains
                // single-address/statically connected to downstream layers, so this
                // ring only removes false dependencies on the staging buffer.
                featureOutput.value().copyFromAsync(slot.outputBuffer.value(), stream);
                stream.putEvent(slot.outputBufferWritableEvent);
            }
        }
        if (!contentDimensions.has_value() && sourceReference.has_value()) {
            sourceReference->recordConsumption(stream);
        }
        if (emitDiagnostics) {
            copyMicros = layerSubmitDiagnosticElapsedMicros(copyStart, layerSubmitDiagnosticNow());
        }

        if (!nextLayer.has_value()) {
            if (emitDiagnostics) {
                emitLayerSubmitDiagnostic(
                    "network_input_forward",
                    layerSubmitDiagnosticLabel("NetworkInput", getId(), getName()),
                    getId(),
                    layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                    {{"copy_us", copyMicros}, {"downstream_us", 0}, {"has_content", contentDimensions.has_value() ? 1UL : 0UL}});
            }
            return;
        }

        const auto downstreamStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        nextLayer.value()->forward(downstreamFeatureOutput, validationPass, batchSize);
        const uint64_t downstreamMicros =
            emitDiagnostics ? layerSubmitDiagnosticElapsedMicros(downstreamStart, layerSubmitDiagnosticNow()) : 0;

        if (emitDiagnostics) {
            emitLayerSubmitDiagnostic(
                "network_input_forward",
                layerSubmitDiagnosticLabel("NetworkInput", getId(), getName()),
                getId(),
                layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                {{"copy_us", copyMicros}, {"downstream_us", downstreamMicros}, {"has_content", contentDimensions.has_value() ? 1UL : 0UL}});
        }
    }


    // Only called for input endpoints configured for reference-valued batch fields.
    // The small reference value is retained in the selected NetworkInput ring slot;
    // its materializer writes directly into the statically connected featureOutput.
    virtual void forward(const Thor::DeviceBatchReference& deviceBatchReference,
                         bool validationPass,
                         uint32_t batchSize = 0) {
        forward(
            deviceBatchReference,
            validationPass,
            batchSize,
            std::nullopt);
    }

    virtual void forward(
        const Thor::DeviceBatchReference& deviceBatchReference,
        bool validationPass,
        uint32_t batchSize,
        std::optional<Thor::BatchSourceReference> sourceReference) {
        THOR_THROW_IF_FALSE(isDeviceReferenceLoad());
        THOR_THROW_IF_FALSE(deviceBatchReference.isInitialized());
        THOR_THROW_IF_FALSE(contentDimensions.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(deviceBatchReference.getOutputDescriptor() == featureOutput.value().getDescriptor());
        THOR_THROW_IF_FALSE(deviceBatchReference.getOutputPlacement() == networkPlacement);
        if (batchSize == 0) {
            batchSize = deviceBatchReference.getBatchSize();
        }
        THOR_THROW_IF_FALSE(batchSize == deviceBatchReference.getBatchSize());

        const bool emitDiagnostics = layerSubmitDiagnosticsActive();
        const auto totalStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        const auto materializeStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();

        if (inputSlots.empty()) {
            // Synchronous callers retain the existing lazy slot-zero behavior.
            allocateInputSlots(1);
        }
        THOR_THROW_IF_FALSE(activeInputSlot < inputSlots.size());
        InputSlot& slot = inputSlots[activeInputSlot];
        THOR_THROW_IF_FALSE(!slot.outputBuffer.has_value());
        slot.deviceBatchReference = deviceBatchReference;
        slot.deviceBatchReference.value().enqueueMaterialization(featureOutput.value(), stream);
        if (sourceReference.has_value()) {
            sourceReference->recordConsumption(stream);
        }

        const uint64_t materializeMicros =
            emitDiagnostics ? layerSubmitDiagnosticElapsedMicros(materializeStart, layerSubmitDiagnosticNow()) : 0;

        if (!nextLayer.has_value()) {
            if (emitDiagnostics) {
                emitLayerSubmitDiagnostic(
                    "network_input_forward_reference",
                    layerSubmitDiagnosticLabel("NetworkInput", getId(), getName()),
                    getId(),
                    layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                    {{"materialize_us", materializeMicros}, {"downstream_us", 0}, {"has_content", 1UL}});
            }
            return;
        }

        const auto downstreamStart = emitDiagnostics ? layerSubmitDiagnosticNow() : LayerSubmitDiagnosticTimePoint();
        nextLayer.value()->forward(featureOutput, validationPass, batchSize);
        const uint64_t downstreamMicros =
            emitDiagnostics ? layerSubmitDiagnosticElapsedMicros(downstreamStart, layerSubmitDiagnosticNow()) : 0;

        if (emitDiagnostics) {
            emitLayerSubmitDiagnostic(
                "network_input_forward_reference",
                layerSubmitDiagnosticLabel("NetworkInput", getId(), getName()),
                getId(),
                layerSubmitDiagnosticElapsedMicros(totalStart, layerSubmitDiagnosticNow()),
                {{"materialize_us", materializeMicros}, {"downstream_us", downstreamMicros}, {"has_content", 1UL}});
        }
    }

    void backward(std::optional<Tensor> incomingErrorInput, uint32_t batchSize = 0) override {
        if (!isPassThrough()) {
            return;
        }
        if (!incomingErrorInput.has_value()) {
            return;
        }
        if (errorInput.has_value()) {
            THOR_THROW_IF_FALSE(incomingErrorInput.value() == errorInput.value());
        }
        if (!previousLayer.has_value() || !errorOutput.has_value()) {
            return;
        }
        previousLayer.value()->backward(errorOutput, batchSize);
    }

   protected:
    struct InputSlot {
        std::optional<Tensor> outputBuffer;
        std::optional<Thor::DeviceBatchReference> deviceBatchReference;
        Event outputBufferLoadedEvent;
        Event outputBufferWritableEvent;
    };

    void initializeInputSlotZero() {
        THOR_THROW_IF_FALSE(!isPassThrough());
        THOR_THROW_IF_FALSE(!isDeviceLoad());
        THOR_THROW_IF_FALSE(featureOutput.has_value() == contentDimensions.has_value());
        if (!contentDimensions.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(inputSlots.empty());
        InputSlot slot;
        if (isDeviceReferenceLoad()) {
            // The slot retains only the small type-erased reference.  Its
            // materializer writes directly into featureOutput on stream.
            slot.outputBuffer = std::nullopt;
        } else {
            slot.outputBuffer = createFeatureOutputTensor();
            // Pre-initialize and record the slot-local events so the hot path never
            // allocates Events and first use can wait on already-complete events.
            loadStream.putEvent(slot.outputBufferLoadedEvent);
            loadStream.putEvent(slot.outputBufferWritableEvent);
        }
        inputSlots.push_back(slot);
    }

    void requireInputSlot(uint32_t slotIndex) const {
        if (!contentDimensions.has_value()) {
            return;
        }
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(slotIndex < inputSlots.size());
    }

    void configurePassThroughSource(std::optional<Tensor> sourceTensor) {
        THOR_THROW_IF_FALSE(isPassThrough());
        THOR_THROW_IF_FALSE(contentDimensions.has_value() == sourceTensor.has_value());
        if (!sourceTensor.has_value()) {
            featureInput = std::nullopt;
            featureOutput = std::nullopt;
            return;
        }

        const TensorDescriptor expectedDescriptor(contentDataType.value(), contentDimensions.value());
        THOR_THROW_IF_FALSE(sourceTensor.value().getDescriptor() == expectedDescriptor);
        THOR_THROW_IF_FALSE(sourceTensor.value().getPlacement() == networkPlacement);
        featureInput = sourceTensor;
        featureOutput = sourceTensor;
    }

    void validateOptionalForwardTensorMatchesPassThrough(std::optional<Tensor> suppliedTensor) const {
        THOR_THROW_IF_FALSE(featureOutput.has_value() == contentDimensions.has_value());
        if (!contentDimensions.has_value()) {
            THOR_THROW_IF_FALSE(!suppliedTensor.has_value());
            return;
        }
        if (suppliedTensor.has_value()) {
            THOR_THROW_IF_FALSE(suppliedTensor.value() == featureOutput.value());
        }
    }

    void allocateInputSlots(uint32_t numSlots) {
        THOR_THROW_IF_FALSE(!isPassThrough());
        THOR_THROW_IF_FALSE(!isDeviceLoad());
        THOR_THROW_IF_FALSE(contentDimensions.has_value());
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(numSlots >= 1);

        if (inputSlots.empty()) {
            initializeInputSlotZero();
        }

        while (inputSlots.size() < numSlots) {
            InputSlot slot;
            if (isDeviceReferenceLoad()) {
                slot.outputBuffer = std::nullopt;
            } else {
                slot.outputBuffer = createFeatureOutputTensor();
                // Eagerly allocate/reuse-initialize the events as well as the tensor.
                loadStream.putEvent(slot.outputBufferLoadedEvent);
                loadStream.putEvent(slot.outputBufferWritableEvent);
            }
            inputSlots.push_back(slot);
        }
    }

    std::optional<std::vector<unsigned long>> contentDimensions;
    TensorPlacement networkPlacement;
    std::optional<DataType> contentDataType;

    uint32_t activeInputSlot = 0;
    std::vector<InputSlot> inputSlots;
    Stream loadStream;
    Mode mode = Mode::ExternalLoad;
    bool aliasSamePlacementInputs = false;
};

}  // namespace ThorImplementation
