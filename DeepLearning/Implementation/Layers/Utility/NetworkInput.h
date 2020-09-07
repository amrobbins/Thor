#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

// To run the network forward, for every input NetworkInput, load the tensor and call forward(myInputTensor)
// Data type will be converted if necessary, memory will be copied across devices if necessary.
class NetworkInput : public Layer {
   public:
    virtual ~NetworkInput() {}

    NetworkInput(Optional<TensorPlacement> networkPlacement,
                 Optional<TensorDescriptor::DataType> contentDataType,
                 Optional<vector<unsigned long>> contentDimensions,
                 Stream stream) {
        construct(networkPlacement, contentDataType, contentDimensions, stream);
    }

    NetworkInput(Optional<Tensor> exampleTensor, Stream stream) {
        if (exampleTensor.isPresent())
            construct(exampleTensor.get().getPlacement(),
                      exampleTensor.get().getDescriptor().getDataType(),
                      exampleTensor.get().getDescriptor().getDimensions(),
                      stream);
        else
            construct(Optional<TensorPlacement>::empty(),
                      Optional<TensorDescriptor::DataType>::empty(),
                      Optional<vector<unsigned long>>::empty(),
                      stream);
    }

    void construct(Optional<TensorPlacement> networkPlacement,
                   Optional<TensorDescriptor::DataType> contentDataType,
                   Optional<vector<unsigned long>> contentDimensions,
                   Stream stream) {
        assert(contentDimensions.isPresent() == networkPlacement.isPresent());
        assert(contentDimensions.isPresent() == contentDataType.isPresent());
        this->networkPlacement = networkPlacement;
        this->contentDataType = contentDataType;
        this->contentDimensions = contentDimensions;
        this->stream = stream;
    }

    virtual bool isInput() { return true; }

    virtual void connectToNextLayer(Layer *nextLayer, int connectionType = 0) {
        assert(this->nextLayer.isEmpty());

        this->nextLayer = nextLayer;

        featureOutput = createFeatureOutputTensor();

        nextLayer->connectToPreviousLayer(this, featureOutput, stream, false, connectionType);
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
    Optional<TensorPlacement> networkPlacement;
    Optional<TensorDescriptor::DataType> contentDataType;
};
