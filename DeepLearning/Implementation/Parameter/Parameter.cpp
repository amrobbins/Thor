#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Initializers/Initializer.h"

using namespace std;

namespace ThorImplementation {
Parameter::Parameter(string name, bool trainable) : name(name), trainable(trainable), trainingEnabled(trainable) { assert(!name.empty()); }

Parameter::Parameter(string name, bool trainable, const std::vector<uint64_t>& shape, const TensorDescriptor::DataType dtype)
    : name(std::move(name)), trainable(trainable), trainingEnabled(trainable), shape(shape), dtype(dtype) {}

void Parameter::compileStorageAndOptimizer(const StorageContext& context,
                                           const Optional<Stream>& gradientUpdateStream,
                                           bool inferenceOnly) {
    this->gradientUpdateStream = gradientUpdateStream;
    if (inferenceOnly)
        trainingEnabled = false;

    createStorage(context);

    if (isTrainable() && !inferenceOnly) {
        assert(hasOptimizer());
        assert(gradientUpdateStream.isPresent());
        optimizer->compile(storage, gradientUpdateStream.get());
    }
    storageInitialized = true;
}

void Parameter::compileStorageAndOptimizer(const Tensor& inputTensor, const Optional<Stream>& gradientUpdateStream, bool inferenceOnly) {
    compileStorageAndOptimizer(StorageContext({{"feature_input", inputTensor}}), gradientUpdateStream, inferenceOnly);
}

void Parameter::compileInitializer(uint64_t fanIn, uint64_t fanOut) {
    if (!hasInitializer()) {
        return;
    }

    Stream initStream = gradientUpdateStream.isPresent()
                            ? gradientUpdateStream.get()
                            : Stream::getMostRecentGradientUpdateStream(storage.get().getPlacement().getDeviceNum());

    initializer->compile(storage, initStream, fanIn, fanOut);
}

void Parameter::clearStorage() { storage.clear(); }

Event Parameter::initialize() {
    assert(hasInitializer());
    Event initReadyEvent = initializer->initialize();
    return initReadyEvent;
}

bool Parameter::applyGradient(uint32_t batchSize) {
    assert(!needExpressionRecompile);
    if (!isTrainingEnabled())
        return false;
    assert(optimizer != nullptr);
    optimizer->updateWeights(batchSize);
    return true;
}

bool Parameter::hasOptimizer() { return optimizer != nullptr; }
void Parameter::setOptimizer(Optional<shared_ptr<Optimizer>> newOptimizer) { this->optimizer = newOptimizer; }
shared_ptr<Optimizer> Parameter::getOptimizer() { return optimizer; }
void Parameter::clearOptimizer() { optimizer = nullptr; }

bool Parameter::hasInitializer() { return initializer != nullptr; }
void Parameter::setInitializer(Optional<shared_ptr<Initializer>> newInitializer) { this->initializer = newInitializer; }
shared_ptr<Initializer> Parameter::getInitializer() { return initializer; }
void Parameter::clearInitializer() { initializer = nullptr; }

string Parameter::getName() const { return name; }
Optional<Tensor> Parameter::getStorage() { return storage; }

bool Parameter::isTrainable() const { return trainable; }
bool Parameter::isTrainingEnabled() const { return isTrainable() && trainingEnabled; }
void Parameter::setTrainingEnabled(bool enabled) {
    assert(isTrainable());

    if (enabled != trainingEnabled) {
        trainingEnabled = enabled;
        if (expressionBased) {
            // Set of gradients changed to include or exclude the gradient of this parameter.
            // Need to recompile the expression.
            needExpressionRecompile = true;
        }
    }
}

bool Parameter::isStorageInitialized() const { return storageInitialized; }

void Parameter::createStorage(const StorageContext& context) {
    assert(shape.isPresent());
    assert(dtype.isPresent());
    std::vector<std::string> inputNames = context.getInputNames();
    assert(!inputNames.empty());
    Tensor anInput = context.getInput(inputNames[0]);

    storage = allocateStorage(anInput.getPlacement(), shape, dtype);
}

Tensor Parameter::allocateStorage(const TensorPlacement placement,
                                  const std::vector<uint64_t>& shape,
                                  const TensorDescriptor::DataType dtype) {
    TensorDescriptor storageDescriptor(dtype, shape);
    return Tensor(placement, storageDescriptor);
}

}  // namespace ThorImplementation
