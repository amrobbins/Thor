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
        THOR_THROW_IF_FALSE(contentDimensions.has_value() == featureInput.has_value());

        if (contentDimensions.has_value())
            THOR_THROW_IF_FALSE(featureInput.value().getDescriptor().getDimensions() == contentDimensions.value());

        if (!nextLayer.has_value())
            return;

        if (contentDimensions.has_value()) {
            THOR_THROW_IF_FALSE(featureOutput.has_value());

            // Wait for previous featureOutput load to finish
            // Copy into buffer using the load stream
            // stream waits for all previously scheduled work to finish and for copy to finish - copy tends to finish first
            outputBuffer.copyFromAsync(featureInput.value(), loadStream);
            stream.waitEvent(loadStream.putEvent());

            // Copy from buffer to featureOutput
            // LoadStream waits for copy to finish.
            // After this happens the buffer can be loaded with the next payload.
            featureOutput.value().copyFromAsync(outputBuffer, stream);
            loadStream.waitEvent(stream.putEvent());
        }

        nextLayer.value()->forward(featureOutput, validationPass, batchSize);
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {}

   protected:
    std::optional<std::vector<unsigned long>> contentDimensions;
    TensorPlacement networkPlacement;
    std::optional<DataType> contentDataType;

    Tensor outputBuffer;
    Stream loadStream;
};

}  // namespace ThorImplementation
