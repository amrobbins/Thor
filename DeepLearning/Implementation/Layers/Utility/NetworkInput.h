#pragma once

#include <optional>
#include <utility>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

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

        std::optional<Tensor> outputBufferTensor = createFeatureOutputTensor();
        if (outputBufferTensor.has_value())
            outputBuffer = outputBufferTensor.value();
        featureOutput = createFeatureOutputTensor();

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

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {}
    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override {}

    // Only called for input endpoints
    // When the source tensor that will be sent to the input is loaded via a stream,
    // then this version of forward is used which causes the loader stream to wait till copy is finished.
    virtual void forward(std::optional<Tensor> featureInput, bool validationPass, Event copyToSourceTensorFinished, uint32_t batchSize = 0) {
        loadStream.waitEvent(copyToSourceTensorFinished);
        forward(featureInput, validationPass, batchSize);
    }

    // Only called for input endpoints
    // This version of forward expects that the memory in featureInput has already been populated before forward is called.
    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        THOR_THROW_IF_FALSE(contentDimensions.has_value() == featureInput.has_value());

        if (contentDimensions.has_value())
            THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == contentDimensions.value());

        if (!nextLayer.has_value())
            return;

        if (contentDimensions.has_value()) {
            THOR_THROW_IF_FALSE(featureOutput.has_value());

            // Ping-pong the network input backing allocations instead of copying the prefetched payload into a
            // fixed featureOutput allocation. The logical featureOutput tensor stays stable, so downstream layers
            // that already copied/stamped it continue to see the updated backing pointer without restamping.
            //
            // outputBuffer owns the inactive backing allocation at this point. Do not overwrite it until the last
            // compute-stream consumers of that backing have completed, but do not wait on the host.
            if (outputBufferReusableEvent.has_value()) {
                loadStream.waitEvent(outputBufferReusableEvent.value());
                outputBufferReusableEvent.reset();
            }

            // The async copy captures outputBuffer's current raw allocation when it is enqueued.
            outputBuffer.copyFromAsync(featureInput.value(), loadStream);

            // Downstream work is launched on stream, so order it after the upload without synchronizing the host.
            Event uploadDone = loadStream.putEvent();
            stream.waitEvent(uploadDone);

            // Flip the logical featureOutput tensor to the newly uploaded backing before launching consumers. This
            // is a host-side pointer swap only; already enqueued GPU work captured raw pointers at enqueue time.
            featureOutput.value().swapBackingMemoryWith(outputBuffer);

            // The reusable event follows the backing allocation, not the Tensor handle. After the backing swap,
            // outputBuffer owns the previously active backing and must carry its prior reusable event.
            std::swap(featureOutputReusableEvent, outputBufferReusableEvent);
        }

        nextLayer.value()->forward(featureOutput, validationPass, batchSize);

        if (contentDimensions.has_value() && featureOutput.has_value()) {
            featureOutputReusableEvent = stream.putEvent();
        }
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {}

   protected:
    std::optional<std::vector<unsigned long>> contentDimensions;
    TensorPlacement networkPlacement;
    std::optional<DataType> contentDataType;

    Tensor outputBuffer;
    Stream loadStream;
    std::optional<Event> featureOutputReusableEvent;
    std::optional<Event> outputBufferReusableEvent;
};

}  // namespace ThorImplementation
