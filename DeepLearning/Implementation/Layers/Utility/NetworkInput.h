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
    }

    virtual bool isInput() { return true; }

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        assert(this->nextLayer.isEmpty());

        this->nextLayer = nextLayer;

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
    virtual void forward(Optional<Tensor> featureInput) {
        assert(contentDimensions.isPresent() == featureInput.isPresent());

        if (contentDimensions.isPresent())
            assert(featureInput.get().getDescriptor().getDimensions() == contentDimensions.get());

        if (!nextLayer.isPresent())
            return;

        if (contentDimensions.isPresent()) {
            assert(featureOutput.isPresent());
            featureOutput.get().copyFromAsync(featureInput, stream);
        }

        nextLayer.get()->forward(featureOutput);
    }

    virtual void backward(Optional<Tensor> errorInput) {}

   protected:
    Optional<vector<unsigned long>> contentDimensions;
    TensorPlacement networkPlacement;
    Optional<TensorDescriptor::DataType> contentDataType;
};

}  // namespace ThorImplementation
