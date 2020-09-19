#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class DeviceCrossing : public Layer {
   public:
    virtual ~DeviceCrossing() {}

    DeviceCrossing() { uninitialized = true; }

    DeviceCrossing(TensorPlacement inputPlacement, TensorPlacement outputPlacement) {
        assert(inputPlacement != outputPlacement);

        uninitialized = false;
        this->inputPlacement = inputPlacement;
        this->outputPlacement = outputPlacement;
    }

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(!uninitialized);
        assert(featureInput.isPresent());
        return Tensor(outputPlacement, featureInput.get().getDescriptor());
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(!uninitialized);
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());
        Event finishedCopyEvent = outputTensor.get().copyFromAsync(inputTensor, stream.putEvent());

        // Tell the stream on the dest gpu to wait for the copy from source gpu to finish
        otherDeviceStream.waitEvent(finishedCopyEvent);
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        assert(!uninitialized);
        if (errorOut.isPresent())
            errorOut.get().copyFromAsync(errorIn, stream.putEvent());
    }

    virtual void connectToNextLayer(Layer *nextLayer, int connectionType = 0) {
        assert(!uninitialized);
        assert(!running);
        this->nextLayer = nextLayer;
        featureOutput = createFeatureOutputTensor();
        otherDeviceStream = Stream(outputPlacement.getMemDevice() == TensorPlacement::MemDevices::CPU ? inputPlacement.getDeviceNum()
                                                                                                      : outputPlacement.getDeviceNum());
        errorInput =
            nextLayer->connectToPreviousLayer(this, featureOutput, otherDeviceStream, shouldConnectToBackPropErrorIn(), connectionType);

        if (errorInput.isPresent()) {
            assert(errorInput.get().getDescriptor() == featureOutput.get().getDescriptor());
            assert(errorInput.get().getPlacement() == featureOutput.get().getPlacement());
        }
    }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int connectionType = 0) {
        assert(!uninitialized);
        assert(featureInput.isPresent());
        assert(featureInput.get().getPlacement() == inputPlacement);
        return Layer::connectToPreviousLayer(previousLayer, featureInput, stream, backPropagateError);
    }

    virtual void ensureNoDeviceCrossing() {
        // device crossing allowed here.
    }

   private:
    bool uninitialized;
    TensorPlacement inputPlacement;
    TensorPlacement outputPlacement;

    Stream otherDeviceStream;
};

}  // namespace ThorImplementation
