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

    // Crosses from source device to dest device
    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        assert(!uninitialized);
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());
        outputTensor.get().copyFromAsync(inputTensor, stream);

        // Tell the stream on the dest gpu to wait for the copy from source gpu to finish
        Event finishedCopyEvent = stream.putEvent();
        otherDeviceStream.waitEvent(finishedCopyEvent);
    }

    // Crosses from dest device to source device
    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        assert(!uninitialized);
        if (errorOut.isPresent()) {
            stream.waitEvent(otherDeviceStream.putEvent());
            errorOut.get().copyFromAsync(errorIn, stream);
        }
    }

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        assert(!uninitialized);
        assert(!running);
        this->nextLayer = nextLayer;
        featureOutput = createFeatureOutputTensor();
        otherDeviceStream = Stream(outputPlacement.getMemDevice() == TensorPlacement::MemDevices::CPU ? inputPlacement.getDeviceNum()
                                                                                                      : outputPlacement.getDeviceNum());
        errorInput = nextLayer->connectToPreviousLayer(
            this, featureOutput, otherDeviceStream, shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);

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
