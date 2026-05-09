#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/CudnnHelper.h"
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
class Initializer;

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
    virtual void compile() final {
        preCompile();
        compileImpl();
        postCompile();
    }

    virtual ~Layer() {}

    virtual void setName(std::string name) {
        THOR_THROW_IF_FALSE(this->name.empty());
        this->name = name;
    }

    virtual std::string getName() const { return name; }

    // initialize weights using the configured initializer. In general set any initial values.
    virtual void initialize() {
        THOR_THROW_IF_FALSE(compiled);
        THOR_THROW_IF_FALSE(!running);
        running = true;
    }

    // release any resources that are used for execution and need to be released
    virtual void cleanup() {
        compiled = false;
        running = false;
    }

    // Note: featureInput is guaranteed to be connected before createFeatureOutputTensor() is called.
    virtual std::optional<Tensor> createFeatureOutputTensor() {
        // The default implementation just creates a clone of the corresponding feature input tensor,
        // this is the behavior of math layers etc that apply a function to the input tensor but do not reshape it.
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone();
    }

    virtual void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) {
        THOR_THROW_IF_FALSE(running);

        infer(featureInput, featureOutput, stream);

        if (!nextLayer.has_value())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        nextLayer.value()->forward(featureOutput, validationPass, batchSize);
    }

    virtual void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) {
        THOR_THROW_IF_FALSE(running);

        // Experimental - back propagation stops at empty error input
        // i.e. the errorInput data is the thing necessitates processing.
        // if all errorInputs are empty for any layer that back propagation path will stop at that layer.
        if (!errorInput.has_value())
            return;

        backProp(featureInput, errorInput, errorOutput, stream);

        if (!previousLayer.has_value())
            return;

        // Expecting to get tail-recursion optimization of -O3 so that stack space does not build up here.
        previousLayer.value()->backward(errorOutput, batchSize);
    }

    virtual void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) {
        THOR_THROW_IF_FALSE(!compiled);

        THOR_THROW_IF_FALSE(!this->nextLayer.has_value());
        this->nextLayer = nextLayer;
        if (nextLayer->hasFeatureInput())
            featureOutput = createFeatureOutputTensor();
        else
            featureOutput = std::nullopt;

        errorInput = nextLayer->connectToPreviousLayer(
            this, featureOutput, stream, shouldConnectToBackPropErrorIn() && !isBackPropStub(), loaderConnectionType);

        // When the next layer says that there is no error back propagation path here, then this layer removes that path
        // from itself and informs the adjacent layer in the back propagation path to do the same.
        if (!errorInput.has_value() && errorOutput.has_value() && previousLayer.has_value()) {
            previousLayer.value()->replaceErrorInput(errorOutput, errorInput);
            errorOutput.reset();
        }

        if (errorInput.has_value() && featureOutput.has_value()) {
            THOR_THROW_IF_FALSE(errorInput.value().getDescriptor() == featureOutput.value().getDescriptor());
            THOR_THROW_IF_FALSE(errorInput.value().getPlacement() == featureOutput.value().getPlacement());
        }

        ensureNoDeviceCrossing();
    }

    virtual bool hasFeatureInput() { return true; }

    virtual std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) {
        // backPropagateError allows the previous layer to specify that it does not support back propagation,
        // inferenceOnly means that even though back propagation may be supported, we are not using it since we are not training.
        if (backPropagateError && !isInferenceOnly())
            return featureInput.value().clone();
        else
            return std::nullopt;
    }

    virtual std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer, std::optional<Tensor> featureInput, Stream stream, bool backPropagateError, int loaderConnectionType = 0) {
        THOR_THROW_IF_FALSE(!compiled);
        THOR_THROW_IF_FALSE(!this->previousLayer.has_value());
        THOR_THROW_IF_FALSE(!this->featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->errorOutput.has_value());
        if (backPropagateError)
            THOR_THROW_IF_FALSE(featureInput.has_value());

        this->previousLayer = previousLayer;
        this->stream = stream;
        this->featureInput = featureInput;
        this->errorOutput = createErrorOutputTensor(backPropagateError);

        return this->errorOutput;
    }

    // For situations where the error input should just pass through to the error output,
    // this method is used to avoid duplicating the tensor and unnecessary data movement.
    // This is also used when there is a path that hits in a non-back prop layer,
    // in that case the errorInput is set to std::nullopt.
    virtual void replaceErrorInput(std::optional<Tensor> oldErrorInput, std::optional<Tensor> newErrorInput) {
        THOR_THROW_IF_FALSE(oldErrorInput.has_value());
        if (errorInput.has_value()) {
            THOR_THROW_IF_FALSE(oldErrorInput.value() == errorInput.value());
        }

        if (errorOutput.has_value()) {
            // 1. When it was populated but now should not be, then deallocate it
            // 2. When they are fused already they need to remain fused, and pass the message to check for this condition backward.
            if (!newErrorInput.has_value() || (errorOutput.value() == errorInput.value())) {
                if (previousLayer.has_value())
                    previousLayer.value()->replaceErrorInput(errorOutput, newErrorInput);
                errorOutput = newErrorInput;
            }
        }
        errorInput = newErrorInput;
    }

    virtual void ensureNoDeviceCrossing() {
        if (featureInput.has_value() && featureOutput.has_value())
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == featureOutput.value().getPlacement());
        if (featureInput.has_value() && errorInput.has_value())
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == errorInput.value().getPlacement());
        if (featureInput.has_value() && errorOutput.has_value())
            THOR_THROW_IF_FALSE(featureInput.value().getPlacement() == errorOutput.value().getPlacement());

        if (featureOutput.has_value() && errorInput.has_value())
            THOR_THROW_IF_FALSE(featureOutput.value().getPlacement() == errorInput.value().getPlacement());
        if (featureOutput.has_value() && errorOutput.has_value())
            THOR_THROW_IF_FALSE(featureOutput.value().getPlacement() == errorOutput.value().getPlacement());

        if (errorInput.has_value() && errorOutput.has_value())
            THOR_THROW_IF_FALSE(errorInput.value().getPlacement() == errorOutput.value().getPlacement());
    }

    virtual TensorPlacement getPlacement() {
        if (errorInput.has_value()) {
            return errorInput.value().getPlacement();
        } else if (errorOutput.has_value()) {
            return errorOutput.value().getPlacement();
        } else if (featureInput.has_value()) {
            return featureInput.value().getPlacement();
        } else if (featureOutput.has_value()) {
            return featureOutput.value().getPlacement();
        } else {
            return TensorPlacement(TensorPlacement::MemDevices::CPU);
        }
    }

    virtual bool isInferenceOnly() { return inferenceOnly; }
    virtual void setConstructForInferenceOnly(bool inferenceOnly) {
        THOR_THROW_IF_FALSE(!compiled);
        this->inferenceOnly = inferenceOnly;
    }
    // When connecting to the next layer, choose if this layer will connect to the back prop error tensor.
    virtual void setConnectToBackPropErrorIn(bool connectToBackPropErrorIn) {
        THOR_THROW_IF_FALSE(!compiled);
        this->connectToBackPropErrorIn = connectToBackPropErrorIn;
    }
    virtual bool shouldConnectToBackPropErrorIn() { return connectToBackPropErrorIn; }

    virtual bool isBackPropStub() { return !errorOutput.has_value(); }

    virtual std::optional<Tensor> getFeatureInput() { return featureInput; }
    virtual std::optional<Tensor> getFeatureOutput() { return featureOutput; }
    virtual std::optional<Tensor> getErrorInput() { return errorInput; }
    virtual std::optional<Tensor> getErrorOutput() { return errorOutput; }
    virtual std::optional<Layer *> getNextLayer() { return nextLayer; }
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
        THOR_THROW_IF_FALSE(featureInput.has_value());
        std::vector<uint64_t> inputDimensions = featureInput.value().getDescriptor().getDimensions();
        uint64_t fanIn = 1;
        for (uint32_t i = 1; i < inputDimensions.size(); ++i) {
            fanIn *= inputDimensions[i];
        }
        return fanIn;
    }

    // compute the fan out for one element of a batch
    virtual uint64_t getFanOut() {
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        std::vector<uint64_t> outputDimensions = featureOutput.value().getDescriptor().getDimensions();
        uint64_t fanOut = 1;
        for (uint32_t i = 1; i < outputDimensions.size(); ++i) {
            fanOut *= outputDimensions[i];
        }
        if (nextLayer.has_value()) {
            fanOut *= nextLayer.value()->getDownStreamFanoutMultiplier();
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

    virtual void setInitializer(Tensor target, std::shared_ptr<ThorImplementation::Initializer> initializer) {
        // Should not be called on a layer that does not override it.
        THOR_UNREACHABLE();
    }

    virtual bool hasInitializer(Tensor target) {
        // Should not be called on a layer that does not override it.
        THOR_UNREACHABLE();
        return false;
    }

    virtual Event initializeTensor(Tensor target) {
        // Should not be called on a layer that does not override it.
        THOR_UNREACHABLE();
        return Event();
    }

   protected:
    std::optional<Tensor> featureInput;
    std::optional<Tensor> featureOutput;
    std::optional<Tensor> errorInput;
    std::optional<Tensor> errorOutput;
    Stream stream;
    std::optional<Layer *> nextLayer;
    std::optional<Layer *> previousLayer;

    virtual void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) = 0;

    virtual void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) = 0;

    // Layers can override preCompile() and must first call <Super>::preCompile()
    // Layers can override compileImpl() and must call <Super>::compileImpl() at any point during the function, usually first.
    // Layers can override postCompile() and must call <Super>::postCompile() last
    virtual void preCompile() { THOR_THROW_IF_FALSE(!compiled); }
    virtual void compileImpl() {};
    virtual void postCompile() { compiled = true; }

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
