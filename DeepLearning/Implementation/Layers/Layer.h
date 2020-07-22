#pragma once

#include "DeepLearning/Implementation/Tensor/DistributedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/CudnnHelper.h"
#include "Utilities/Common/Optional.h"
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
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

using std::make_pair;
using std::map;
using std::mutex;
using std::pair;
using std::priority_queue;
using std::set;
using std::unique_lock;
using std::unordered_multimap;
using std::vector;

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
    Layer() {
        compiled = false;
        running = false;
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
        assert(running);

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

    virtual void forward(Optional<Tensor> featureInput) {
        assert(running);

        infer(featureInput, featureOutput, stream);

        if (nextLayer.isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayer.get()->forward(featureOutput);
    }

    virtual void backward(Optional<Tensor> errorInput) {
        assert(running);

        backProp(featureInput, errorInput, errorOutput, stream);

        if (previousLayer.isEmpty())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayer.get()->backward(errorOutput);
    }

    virtual void connectToNextLayer(Layer *nextLayer) {
        assert(!compiled);

        assert(this->nextLayer.isEmpty());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = Optional<Tensor>::empty();

        errorInput = nextLayer->connectToPreviousLayer(this, featureOutput, stream, true);

        if (errorInput.isPresent() && featureOutput.isPresent()) {
            assert(errorInput.get().getDescriptor() == featureOutput.get().getDescriptor());
            assert(errorInput.get().getPlacement() == featureOutput.get().getPlacement());
        }

        ensureNoDeviceCrossing();
    }

    virtual bool hasFeatureInput() { return true; }

    // postConnectToNextLayer() is called after each call to connectToNextLayer(...)
    virtual void postConnectToNextLayer() {}

    virtual Optional<Tensor> connectToPreviousLayer(Layer *previousLayer,
                                                    Optional<Tensor> featureInput,
                                                    Stream stream,
                                                    bool backPropagateError) {
        assert(!compiled);
        assert(this->previousLayer.isEmpty());
        assert(this->featureInput.isEmpty());
        assert(this->errorOutput.isEmpty());
        if (backPropagateError)
            assert(featureInput.isPresent());

        this->previousLayer = previousLayer;
        this->stream = stream;
        this->featureInput = featureInput;
        if (backPropagateError)
            errorOutput = featureInput.get().clone();
        else
            errorOutput = Optional<Tensor>::empty();

        return errorOutput;
    }

    virtual void ensureNoDeviceCrossing() {
        if (featureInput.isPresent() && errorOutput.isPresent())
            assert(featureInput.get().getPlacement() == errorOutput.get().getPlacement());
        if (featureOutput.isPresent() && errorInput.isPresent())
            assert(featureOutput.get().getPlacement() == errorInput.get().getPlacement());
    }

    // FIXME: implement inference only connections that use isInferenceOnly()
    virtual bool isInferenceOnly() { return inferenceOnly; }
    virtual void setInferenceOnly(bool inferenceOnly) { this->inferenceOnly = inferenceOnly; }

    virtual bool isBackPropStub() { return errorOutput.isEmpty(); }

    virtual Optional<Tensor> getFeatureInput() { return featureInput; }
    virtual Optional<Tensor> getFeatureOutput() { return featureOutput; }
    virtual Optional<Tensor> getErrorInput() { return errorInput; }
    virtual Optional<Tensor> getErrorOutput() { return errorOutput; }
    virtual Optional<Layer *> getNextLayer() { return nextLayer; }
    virtual Stream getStream() { return stream; }

    void average(half result_d[], half *sources_d[], int numElements, int numSources, Stream stream) {
        launchAverage(result_d, sources_d, numSources, numElements, stream);
    }

    void sum(half result_d[], half *sources_d[], int numElements, int numSources, Stream stream) {
        launchSum(result_d, sources_d, numSources, numElements, stream);
    }

    void sumScale(half result_d[], half source0_d[], half source1_d[], float scale, int numElements, Stream stream) {
        launchSumScale(result_d, source0_d, source1_d, scale, numElements, stream);
    }

   protected:
    bool inferenceOnly;

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

    mutex mtx;
};
