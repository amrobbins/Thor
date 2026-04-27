#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Initializers/Initializer.h"

using namespace std;

namespace ThorImplementation {
PhysicalParameter::PhysicalParameter(string name, bool trainable) : name(name), trainable(trainable), trainingEnabled(trainable) {
    assert(!name.empty());
}

PhysicalParameter::PhysicalParameter(string name,
                                     bool trainable,
                                     const std::vector<uint64_t>& shape,
                                     const TensorDescriptor::DataType dtype)
    : name(std::move(name)), trainable(trainable), trainingEnabled(trainable), shape(shape), dtype(dtype) {}

void PhysicalParameter::compileStorageAndOptimizer(const StorageContext& context,
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

void PhysicalParameter::compileStorageAndOptimizer(const Tensor& inputTensor,
                                                   const Optional<Stream>& gradientUpdateStream,
                                                   bool inferenceOnly) {
    compileStorageAndOptimizer(StorageContext({{"feature_input", inputTensor}}), gradientUpdateStream, inferenceOnly);
}

void PhysicalParameter::compileInitializer(uint64_t fanIn, uint64_t fanOut) {
    if (!hasInitializer()) {
        return;
    }

    Stream initStream = gradientUpdateStream.isPresent()
                            ? gradientUpdateStream.get()
                            : Stream::getMostRecentGradientUpdateStream(storage.get().getPlacement().getDeviceNum());

    initializer->compile(storage, initStream, fanIn, fanOut);
}

void PhysicalParameter::clearStorage() { storage.clear(); }

Event PhysicalParameter::initialize() {
    assert(hasInitializer());
    Event initReadyEvent = initializer->initialize();
    return initReadyEvent;
}

bool PhysicalParameter::applyGradient(uint32_t batchSize) {
    assert(!needExpressionRecompile);
    if (!isTrainingEnabled())
        return false;
    assert(optimizer != nullptr);
    optimizer->updateWeights(batchSize);
    return true;
}

bool PhysicalParameter::hasOptimizer() { return optimizer != nullptr; }
void PhysicalParameter::setOptimizer(Optional<shared_ptr<Optimizer>> newOptimizer) { this->optimizer = newOptimizer; }
shared_ptr<Optimizer> PhysicalParameter::getOptimizer() { return optimizer; }
void PhysicalParameter::clearOptimizer() { optimizer = nullptr; }

bool PhysicalParameter::hasInitializer() { return initializer != nullptr; }
void PhysicalParameter::setInitializer(Optional<shared_ptr<Initializer>> newInitializer) { this->initializer = newInitializer; }
shared_ptr<Initializer> PhysicalParameter::getInitializer() { return initializer; }
void PhysicalParameter::clearInitializer() { initializer = nullptr; }

string PhysicalParameter::getName() const { return name; }
Optional<Tensor> PhysicalParameter::getStorage() { return storage; }

bool PhysicalParameter::isTrainable() const { return trainable; }
bool PhysicalParameter::isTrainingEnabled() const { return isTrainable() && trainingEnabled; }
void PhysicalParameter::setTrainingEnabled(bool enabled) {
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

bool PhysicalParameter::isStorageInitialized() const { return storageInitialized; }

void PhysicalParameter::createStorage(const StorageContext& context) {
    assert(shape.isPresent());
    assert(dtype.isPresent());
    std::vector<std::string> inputNames = context.getInputNames();
    assert(!inputNames.empty());
    Tensor anInput = context.getInput(inputNames[0]);

    storage = allocateStorage(anInput.getPlacement(), shape, dtype);
}

Tensor PhysicalParameter::allocateStorage(const TensorPlacement placement,
                                          const std::vector<uint64_t>& shape,
                                          const TensorDescriptor::DataType dtype) {
    TensorDescriptor storageDescriptor(dtype, shape);
    return Tensor(placement, storageDescriptor);
}

}  // namespace ThorImplementation
