#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

class NetworkOutput : public Layer {
   public:
    NetworkOutput(Optional<TensorPlacement> contentPlacement) : contentPlacement(contentPlacement) {}

    virtual void connectToNextLayer(Layer *nextLayer) { assert(false); }

    virtual Optional<Tensor> connectToPreviousLayer(Layer *previousLayer,
                                                    Optional<Tensor> featureInput,
                                                    Stream stream,
                                                    bool backPropagateError) {
        assert(this->previousLayer.isEmpty());
        assert(featureInput.isPresent());
        assert(this->featureInput.isEmpty());

        this->featureInput = featureInput;
        this->featureOutput = featureInput;
        this->previousLayer = previousLayer;
        this->stream = stream;

        if (featureInput.isPresent())
            featureOutput = createFeatureOutputTensor();
        return Optional<Tensor>::empty();
    }

    virtual Event getOutputReadyEvent() { return outputReadyEvent; }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isEmpty() == contentPlacement.isEmpty());
        if (contentPlacement.isEmpty())
            return Optional<Tensor>::empty();
        else
            return featureInput.get().clone(contentPlacement);
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent() == outputTensor.isPresent());

        if (inputTensor.isPresent())
            outputTensor.get().copyFromAsync(inputTensor, stream);

        outputReadyEvent = stream.putEvent();
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {}

    virtual void backward(Optional<Tensor> errorInput) {}

   protected:
    Event outputReadyEvent;

    Optional<TensorPlacement> contentPlacement;
};
