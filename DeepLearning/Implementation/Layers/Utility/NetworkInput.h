#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

// To run the network forward, for every input NetworkInput, load the tensor and call forward(myInputTensor)
// Data type will be converted if necessary, memory will be copied across devices if necessary.
class NetworkInput : public Layer {
   public:
    virtual ~NetworkInput() {}

    NetworkInput(TensorPlacement networkPlacement,
                 Optional<TensorDescriptor::DataType> contentDataType,
                 Optional<vector<unsigned long>> contentDimensions) {
        construct(networkPlacement, contentDataType, contentDimensions);
    }

    NetworkInput(Tensor exampleTensor) {
        assert(exampleTensor.isInitialized());
        construct(exampleTensor.getPlacement(), exampleTensor.getDescriptor().getDataType(), exampleTensor.getDescriptor().getDimensions());
    }

    void construct(TensorPlacement networkPlacement,
                   Optional<TensorDescriptor::DataType> contentDataType,
                   Optional<vector<unsigned long>> contentDimensions) {
        assert(contentDimensions.isPresent() == contentDataType.isPresent());
        this->networkPlacement = networkPlacement;
        this->contentDataType = contentDataType;
        this->contentDimensions = contentDimensions;
        int gpuNum = 0;
        if (networkPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU)
            gpuNum = networkPlacement.getDeviceNum();
        this->stream = Stream(gpuNum);
        this->loadStream = Stream(gpuNum);
    }

    virtual bool isInput() { return true; }

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        assert(this->nextLayer.isEmpty());

        this->nextLayer = nextLayer;

        outputBuffer = createFeatureOutputTensor();
        featureOutput = createFeatureOutputTensor();

        nextLayer->connectToPreviousLayer(this, featureOutput, stream, false, loaderConnectionType);
    }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        assert(false);
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        if (contentDimensions.isPresent())
            return Tensor(networkPlacement, TensorDescriptor(contentDataType, contentDimensions));
        return Optional<Tensor>::empty();
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {}
    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {}

    // Only called for input endpoints
    // When the source tensor that will be sent to the input is loaded via a stream,
    // then this version of forward is used which causes the loader stream to wait till copy is finished.
    virtual void forward(Optional<Tensor> featureInput, bool validationPass, Event copyToSourceTensorFinished) {
        loadStream.waitEvent(copyToSourceTensorFinished);
        forward(featureInput, validationPass);
    }

    // Only called for input endpoints
    // This version of forward expects that the memory in featureInput has already been populated before forward is called.
    virtual void forward(Optional<Tensor> featureInput, bool validationPass) {
        assert(contentDimensions.isPresent() == featureInput.isPresent());

        if (contentDimensions.isPresent())
            assert(featureInput.get().getDescriptor().getDimensions() == contentDimensions.get());

        if (!nextLayer.isPresent())
            return;

        if (contentDimensions.isPresent()) {
            assert(featureOutput.isPresent());

            // Wait for previous featureOutput load to finish
            // Copy into buffer using the load stream
            // stream waits for all previously scheduled work to finish and for copy to finish - copy tends to finish first
            outputBuffer.copyFromAsync(featureInput, loadStream);
            stream.waitEvent(loadStream.putEvent());

            // Copy from buffer to featureOutput
            // LoadStream waits for copy to finish.
            // After this happens the buffer can be loaded with the next payload.
            featureOutput.get().copyFromAsync(outputBuffer, stream);
            loadStream.waitEvent(stream.putEvent());
        }

        nextLayer.get()->forward(featureOutput, validationPass);
    }

    virtual void backward(Optional<Tensor> errorInput) {}

   protected:
    Optional<vector<unsigned long>> contentDimensions;
    TensorPlacement networkPlacement;
    Optional<TensorDescriptor::DataType> contentDataType;

    Tensor outputBuffer;
    Stream loadStream;
    Event loadReadyEvent;
};

}  // namespace ThorImplementation
