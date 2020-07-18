#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/DeepLearning/CrossEntropyLoss.h"

/**
 * A Loss layer may have 1 input and 1 output, the output of a loss layer is the loss tensor.
 *
 * Loss layers do not connect an errorInput from the next layer, so they are a point at which
 * back propagation will terminate if connected at the output to a back-propagable layer.
 *
 * The loss tensor give the loss per element of the batch, so it is a one dimensions array of
 * batchSize elements, the batch loss is the summation of the elements of the loss tensor.
 */
class Loss : public Layer {
   public:
    Loss(float lossScalingFactor = 1.0f) { this->lossScalingFactor = lossScalingFactor; }

    virtual Optional<Tensor> connectToPreviousLayer(Layer *previousLayer,
                                                    Optional<Tensor> inputTensor,
                                                    Stream stream,
                                                    bool backPropagateError) {
        assert(inputTensor.isPresent());
        assert(inputTensor.get().getDescriptor().getDimensions().size() >= 2);
        assert(this->labelsTensor.isEmpty());
        if (this->featureInput.isPresent()) {
            assert(this->featureInput.get().getDescriptor().getDimensions() == inputTensor.get().getDescriptor().getDimensions());
            assert(this->featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
            assert(this->featureInput.get().getPlacement() == inputTensor.get().getPlacement());
        }

        // First connection is the activations
        // Second connection is the labels
        if (this->featureInput.isEmpty()) {
            // Allocates this->featureInput and this->errorOutput
            Layer::connectToPreviousLayer(previousLayer, inputTensor, stream, backPropagateError);

            // Allocate loss tensor
            vector<unsigned long> onePerBatchDimensions;
            onePerBatchDimensions.push_back(featureInput.get().getDescriptor().getDimensions()[0]);
            lossTensor =
                Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, onePerBatchDimensions));

            // Allocate scaling factor tensor
            vector<unsigned long> scalarDimensions;
            scalarDimensions.push_back(1);
            Tensor lossScalingFactorTensorCpu =
                Tensor(TensorPlacement::MemDevices::CPU, TensorDescriptor(TensorDescriptor::DataType::FP32, scalarDimensions));
            ((float *)lossScalingFactorTensorCpu.getMemPtr())[0] = lossScalingFactor;
            lossScalingFactorTensor =
                Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, scalarDimensions));
            lossScalingFactorTensor.copyFromAsync(lossScalingFactorTensorCpu, stream);

            return errorOutput;
        } else {
            labelsTensor = inputTensor;
            labelsStream = stream;

            return Optional<Tensor>::empty();
        }
    }

    virtual void initialize() {
        activationsReceived = false;
        labelsReceived = false;
    }

    virtual void connectToNextLayer(Layer *nextLayer) {
        assert(!running);
        assert(lossTensor.isPresent());
        assert(this->nextLayer.isEmpty());

        this->nextLayer = nextLayer;

        nextLayer->connectToPreviousLayer(this, lossTensor, stream, false);

        ensureNoDeviceCrossing();
    }

    virtual void forward(Optional<Tensor> inputTensor) {
        assert(running);
        assert(labelsTensor.isPresent());
        assert(labelsTensor.get().isInitialized());
        assert(lossTensor.isPresent());
        assert(lossTensor.get().isInitialized());
        assert(labelsStream.isInitialized());
        assert(inputTensor.isPresent());
        assert(featureOutput.isEmpty());
        assert(featureInput.isPresent());

        // Receiving activations
        if (inputTensor.get() == featureInput.get()) {
            assert(activationsReceived == false);
            activationsReceived = true;
            // Receiving labels
        } else if (inputTensor.get() == labelsTensor.get()) {
            assert(labelsReceived == false);
            labelsReceived = true;
            if (labelsStream != stream)
                stream.waitEvent(labelsStream.putEvent());
        }

        if (activationsReceived && labelsReceived) {
            infer(featureInput, lossTensor, stream);

            activationsReceived = false;
            labelsReceived = false;
        } else {
            return;
        }

        if (nextLayer.isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayer.get()->forward(lossTensor);
    }

    virtual void backward(Optional<Tensor> errorInput) {
        assert(running);
        assert(labelsTensor.isPresent());
        assert(labelsTensor.get().isInitialized());
        assert(lossTensor.isPresent());
        assert(lossTensor.get().isInitialized());
        assert(labelsStream.isInitialized());
        assert(errorInput.isEmpty());

        if (errorOutput.isPresent())
            backProp(featureInput, Optional<Tensor>::empty(), errorOutput, stream);

        if (previousLayer.isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayer.get()->backward(errorOutput);
    }

    // lossTensor is populated during the call to connectToPreviousLayer(...)
    virtual Tensor getLossTensor() { return lossTensor; }

    virtual void ensureNoDeviceCrossing() {
        if (featureInput.isPresent() && errorOutput.isPresent())
            assert(featureInput.get().getPlacement() == errorOutput.get().getPlacement());
        if (featureInput.isPresent()) {
            if (labelsTensor.isPresent())
                assert(labelsTensor.get().getPlacement() == featureInput.get().getPlacement());
            if (lossTensor.isPresent())
                assert(lossTensor.get().getPlacement() == featureInput.get().getPlacement());
        }
    }

   protected:
    Optional<Tensor> labelsTensor;
    Optional<Tensor> lossTensor;

    float lossScalingFactor;
    Tensor lossScalingFactorTensor;
    Stream labelsStream;

    bool activationsReceived;
    bool labelsReceived;
};
