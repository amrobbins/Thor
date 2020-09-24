#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class NetworkOutput : public Layer {
   public:
    virtual ~NetworkOutput() {}

    NetworkOutput(Optional<TensorPlacement> outputPlacement) : outputPlacement(outputPlacement) {}

    virtual void connectToNextLayer(Layer *nextLayer, int connectionType = 0) { assert(false); }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        assert(this->previousLayer.isEmpty());
        assert(featureInput.isPresent());
        assert(this->featureInput.isEmpty());

        this->featureInput = featureInput;
        this->previousLayer = previousLayer;
        this->stream = stream;

        if (featureInput.isPresent())
            featureOutput = createFeatureOutputTensor();
        return Optional<Tensor>::empty();
    }

    virtual Event getOutputReadyEvent() { return outputReadyEvent; }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isEmpty() == outputPlacement.isEmpty());
        if (outputPlacement.isEmpty())
            return Optional<Tensor>::empty();
        else
            return featureInput.get().clone(outputPlacement);
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(inputTensor.isPresent() == outputTensor.isPresent());

        if (inputTensor.isPresent())
            outputTensor.get().copyFromAsync(inputTensor, stream);

        outputReadyEvent = stream.putEvent(false, true);
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {}

    virtual void backward(Optional<Tensor> errorInput) {}

   protected:
    Event outputReadyEvent;

    Optional<TensorPlacement> outputPlacement;
};

}  // namespace ThorImplementation
