#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/CudnnHelper.h"
#include "Utilities/Common/Optional.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Arithmetic/Average.h"
#include "Utilities/TensorOperations/Arithmetic/ElementwiseSubtract.h"
#include "Utilities/TensorOperations/Arithmetic/Exponentiation.h"
#include "Utilities/TensorOperations/Arithmetic/MultiplyByScalar.h"
#include "Utilities/TensorOperations/Arithmetic/Sum.h"
#include "Utilities/TensorOperations/Arithmetic/SumManyToOne.h"
#include "Utilities/TensorOperations/Arithmetic/SumScale.h"

#include <cuda_fp16.h>
#include <cudnn.h>

#include <functional>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation {

class Tensor;
class TensorDescriptor;

/**
 * To implement a new layer, you must at the minimum implement:
 *    infer(...)
 *    backProp(...)
 *
 * and if your output tensor is not the same shape as the input tensor, you will also need to implement:
 *    createFeatureOutputTensor(...)
 *
 * If you don't need to do anything extra, the rest is all standard implementation that is provided by Layer.
 *
 *
 * Before infer or backprop may be called on a layer, the following sequence will occur:
 * 1. connectToNextLayer()
 * 2. compile()
 * 3. initialize()
 *
 * Before a layer will be deleted, the following will occur:
 * 1. cleanup()
 *
 * Difference between compile() and initialize():
 * compile() allocates and measures anything necessary to set up the network,
 * initialize() sets the initial state, for example the value of the weights.
 * So it is legal to call compile() once, then initialize() and run the network,
 * then initiailze() again to reset the state of the network and run it again.
 * cleanup() is the inverse of compile().
 */
class Layer {
   public:
    Layer() : id(nextId.fetch_add(1)) {
        compiled = false;
        running = false;
        inferenceOnly = false;
        connectToBackPropErrorIn = true;
    }

    // Use pointers for layers
    Layer(const Layer &) = delete;

    // allocate anything needed for execution, choose optimal kernels, etc.
    virtual void compile() {}

    virtual void parentCompile() {
        assert(!compiled);
        compiled = true;
    }

    virtual ~Layer() {}

    virtual void setName(std::string name) {
        assert(this->name.empty());
        this->name = name;
    }

    virtual std::string getName() const { return name; }

    // initialize weights using the configured initializer. In general set any initial values.
    virtual void initialize() {}

    // Initialization needed by the Layer parent object
    virtual void parentInitialize() {
        assert(compiled);
        assert(!running);
        running = true;
    }

    // release any resources that are used for execution and need to be released
    virtual void cleanup() {}

    virtual void parentCleanup() {
        compiled = false;
        running = false;
    }

    // Note: featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
    virtual Optional<Tensor> createFeatureOutputTensor() {
        // The default implementation just creates a clone of the corresponding feature input tensor,
        // this is the behavior of math layers etc that apply a function to the input tensor but do not reshape it.
        assert(featureInput.isPresent());
        return featureInput.get().clone();
    }

    virtual void forward(Optional<Tensor> featureInput, bool validationPass) {
        assert(running);

        infer(featureInput, featureOutput, stream);

        if (nextLayer.isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayer.get()->forward(featureOutput, validationPass);
    }

    virtual void backward(Optional<Tensor> errorInput) {
        assert(running);

        // Experimental - back propagation stops at empty error input
        // i.e. the errorInput data is the thing necessitates processing.
        // if all errorInputs are empty for any layer that back propagation path will stop at that layer.
        if (errorInput.isEmpty())
            return;

        backProp(featureInput, errorInput, errorOutput, stream);

        if (previousLayer.isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayer.get()->backward(errorOutput);
    }

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        assert(!compiled);

        assert(this->nextLayer.isEmpty());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = Optional<Tensor>::empty();

        errorInput = nextLayer->connectToPreviousLayer(
            this, featureOutput, stream, shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);

        // When the next layer says that there is no error back propagation path here, then this layer removes that path
        // from itself and informs the adjacent layer in the back propagation path to do the same.
        if (errorInput.isEmpty() && errorOutput.isPresent() && previousLayer.isPresent()) {
            previousLayer.get()->replaceErrorInput(errorOutput, errorInput);
            errorOutput.clear();
        }

        if (errorInput.isPresent() && featureOutput.isPresent()) {
            assert(errorInput.get().getDescriptor() == featureOutput.get().getDescriptor());
            assert(errorInput.get().getPlacement() == featureOutput.get().getPlacement());
        }

        ensureNoDeviceCrossing();
    }

    virtual bool hasFeatureInput() { return true; }

    virtual Optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, Optional<Tensor> featureInput, Stream stream, bool backPropagateError, int loaderConnectionType = 0) {
        assert(!compiled);
        assert(this->previousLayer.isEmpty());
        assert(this->featureInput.isEmpty());
        assert(this->errorOutput.isEmpty());
        if (backPropagateError)
            assert(featureInput.isPresent());

        this->previousLayer = previousLayer;
        this->stream = stream;
        this->featureInput = featureInput;
        if (backPropagateError && !isInferenceOnly())
            errorOutput = featureInput.get().clone();
        else
            errorOutput = Optional<Tensor>::empty();

        return errorOutput;
    }

    // For situations where the error input should just pass through to the error output,
    // this method is used to avoid duplicating the tensor and unnecessary data movement.
    // This is also used when there is a path that hits in a non-back prop layer,
    // in that case the errorInput is set to Optional<Tensor>.empty().
    virtual void replaceErrorInput(Optional<Tensor> oldErrorInput, Optional<Tensor> newErrorInput) {
        assert(oldErrorInput.isPresent());
        assert(oldErrorInput.get() == errorInput.get());

        if (errorOutput.isPresent()) {
            // 1. When it was populated but now should not be, then deallocate it
            // 2. When they are fused already they need to remain fused, and pass the message to check for this condition backward.
            if (newErrorInput.isEmpty() || (errorOutput.get() == errorInput.get())) {
                if (previousLayer.isPresent())
                    previousLayer.get()->replaceErrorInput(errorOutput, newErrorInput);
                errorOutput = newErrorInput;
            }
        }
        errorInput = newErrorInput;
    }

    virtual void ensureNoDeviceCrossing() {
        if (featureInput.isPresent() && featureOutput.isPresent())
            assert(featureInput.get().getPlacement() == featureOutput.get().getPlacement());
        if (featureInput.isPresent() && errorInput.isPresent())
            assert(featureInput.get().getPlacement() == errorInput.get().getPlacement());
        if (featureInput.isPresent() && errorOutput.isPresent())
            assert(featureInput.get().getPlacement() == errorOutput.get().getPlacement());

        if (featureOutput.isPresent() && errorInput.isPresent())
            assert(featureOutput.get().getPlacement() == errorInput.get().getPlacement());
        if (featureOutput.isPresent() && errorOutput.isPresent())
            assert(featureOutput.get().getPlacement() == errorOutput.get().getPlacement());

        if (errorInput.isPresent() && errorOutput.isPresent())
            assert(errorInput.get().getPlacement() == errorOutput.get().getPlacement());
    }

    virtual TensorPlacement getPlacement() {
        if (errorInput.isPresent()) {
            return errorInput.get().getPlacement();
        } else if (errorOutput.isPresent()) {
            return errorOutput.get().getPlacement();
        } else if (featureInput.isPresent()) {
            return featureInput.get().getPlacement();
        } else if (featureOutput.isPresent()) {
            return featureOutput.get().getPlacement();
        } else {
            return TensorPlacement(TensorPlacement::MemDevices::CPU);
        }
    }

    virtual bool isInferenceOnly() { return inferenceOnly; }
    virtual void setConstructForInferenceOnly(bool inferenceOnly) {
        assert(!compiled);
        this->inferenceOnly = inferenceOnly;
    }
    // When connecting to the next layer, choose if this layer will connect to the back prop error tensor.
    virtual void setConnectToBackPropErrorIn(bool connectToBackPropErrorIn) {
        assert(!compiled);
        this->connectToBackPropErrorIn = connectToBackPropErrorIn;
    }
    virtual bool shouldConnectToBackPropErrorIn() { return connectToBackPropErrorIn; }

    virtual bool isBackPropStub() { return errorOutput.isEmpty(); }

    virtual Optional<Tensor> getFeatureInput() { return featureInput; }
    virtual Optional<Tensor> getFeatureOutput() { return featureOutput; }
    virtual Optional<Tensor> getErrorInput() { return errorInput; }
    virtual Optional<Tensor> getErrorOutput() { return errorOutput; }
    virtual Optional<Layer *> getNextLayer() { return nextLayer; }
    virtual Stream getStream() { return stream; }

    static void average(half result_d[], half *sources_d[], int numElements, int numSources, Stream stream) {
        launchAverage(result_d, sources_d, numSources, numElements, stream);
    }

    static void sum(half result_d[], half *sources_d[], uint32_t numSources, uint64_t numElements, Stream stream) {
        launchSum<half>(result_d, sources_d, numSources, numElements, stream);
    }

    static void sumScale(
        float result_d[], float nonScaledSource_d[], float scaledSource_d[], float scale, uint64_t numElements, Stream stream) {
        launchSumScale(result_d, nonScaledSource_d, scaledSource_d, scale, numElements, stream);
    }

    static void sumScaleHalfSourceDest(
        half result_d[], half nonScaledSource_d[], float scaledSource_d[], float scale, uint64_t numElements, Stream stream) {
        launchSumScaleHalfSourceDest(result_d, nonScaledSource_d, scaledSource_d, scale, numElements, stream);
    }

    static void sumScaleHalfSourceDestScaleSource(
        half result_d[], half nonScaledSource_d[], half scaledSource_d[], float scale, uint64_t numElements, Stream stream) {
        launchSumScaleHalfSourceDestScaleSource(result_d, nonScaledSource_d, scaledSource_d, scale, numElements, stream);
    }

    static void sumScaleHalfAll(
        half result_d[], half nonScaledSource_d[], half scaledSource_d[], half scale, uint64_t numElements, Stream stream) {
        launchSumScaleHalfAll(result_d, nonScaledSource_d, scaledSource_d, scale, numElements, stream);
    }

    uint64_t getId() const { return id; }

    // compute the fan in for one element of a batch
    virtual uint64_t getFanIn() {
        assert(featureInput.isPresent());
        std::vector<uint64_t> inputDimensions = featureInput.get().getDescriptor().getDimensions();
        uint64_t fanIn = 1;
        for (uint32_t i = 1; i < inputDimensions.size(); ++i) {
            fanIn *= inputDimensions[i];
        }
        return fanIn;
    }

    // compute the fan out for one element of a batch
    virtual uint64_t getFanOut() {
        assert(featureOutput.isPresent());
        std::vector<uint64_t> outputDimensions = featureOutput.get().getDescriptor().getDimensions();
        uint64_t fanOut = 1;
        for (uint32_t i = 1; i < outputDimensions.size(); ++i) {
            fanOut *= outputDimensions[i];
        }
        if (nextLayer.isPresent()) {
            fanOut *= nextLayer.get()->getDownStreamFanoutMultiplier();
        }
        return fanOut;
    }

    virtual uint32_t getDownStreamFanoutMultiplier() { return 1; }

    virtual uint64_t floatingPointOperationsPerExampleForward() { return 0; }

    virtual uint64_t floatingPointOperationsPerExampleBackward() { return 0; }

    virtual std::string getType() { return "Layer"; }

    static cudnnTensorDescriptor_t createCudnnTensorDescriptor(std::vector<unsigned long> featureInputDimensions,
                                                               TensorDescriptor::DataType dataType);

    virtual bool isKerasCompatible(std::string &explanation) {
        explanation.clear();
        return true;
    }

    virtual bool isCompiled() { return compiled; }

   protected:
    Optional<Tensor> featureInput;
    Optional<Tensor> featureOutput;
    Optional<Tensor> errorInput;
    Optional<Tensor> errorOutput;
    Stream stream;
    Optional<Layer *> nextLayer;
    Optional<Layer *> previousLayer;

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) = 0;

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) = 0;

    bool running;
    bool compiled;

    std::string name;

    std::mutex mtx;

   private:
    bool inferenceOnly;
    bool connectToBackPropErrorIn;

    uint64_t id;
    static std::atomic<uint64_t> nextId;
};

}  // namespace ThorImplementation
