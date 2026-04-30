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

void PhysicalParameter::compileStorage(const StorageContext& context) {
    createStorage(context);
    storageInitialized = true;
}

void PhysicalParameter::compileStorage(const Tensor& inputTensor) { compileStorage(StorageContext({{"feature_input", inputTensor}})); }

void PhysicalParameter::compileOptimizer(const Optional<Stream>& gradientUpdateStream, bool inferenceOnly) {
    assert(trainable == true);
    this->inferenceOnly = inferenceOnly;
    this->gradientUpdateStream = gradientUpdateStream;

    if (isTrainingEnabled()) {
        assert(hasOptimizer());
        assert(this->gradientUpdateStream.isPresent());
        assert(storage.isPresent());
        if (!optimizer->isCompiled()) {
            optimizer->compile(storage, this->gradientUpdateStream.get());
        }
    }
}

void PhysicalParameter::compileInitializer(uint64_t fanIn, uint64_t fanOut) {
    if (!hasInitializer()) {
        return;
    }

    initializer->compile(storage, fanIn, fanOut);
}

void PhysicalParameter::compileInitializer() { compileInitializer(1, 1); }

void PhysicalParameter::clearStorage() { storage.clear(); }

void PhysicalParameter::initialize(Stream initStream) {
    assert(hasInitializer());
    initializer->initialize(initStream);
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
// void PhysicalParameter::setOptimizer(const Optional<shared_ptr<Optimizer>>& newOptimizer) {
//     if (newOptimizer.isPresent() && newOptimizer.get() != nullptr) {
//         this->optimizer = newOptimizer.get()->clone();
//     } else {
//         this->optimizer = nullptr;
//     }
// }
shared_ptr<Optimizer> PhysicalParameter::getOptimizer() { return optimizer; }
void PhysicalParameter::clearOptimizer() { optimizer = nullptr; }

bool PhysicalParameter::hasInitializer() { return initializer != nullptr; }
void PhysicalParameter::setInitializer(Optional<shared_ptr<Initializer>> newInitializer) { this->initializer = newInitializer; }
// void PhysicalParameter::setInitializer(const Optional<shared_ptr<Initializer>>& newInitializer) {
//     if (newInitializer.isPresent() && newInitializer.get() != nullptr) {
//         this->initializer = newInitializer.get()->clone();
//     } else {
//         this->initializer = nullptr;
//     }
// }
shared_ptr<Initializer> PhysicalParameter::getInitializer() { return initializer; }
void PhysicalParameter::clearInitializer() { initializer = nullptr; }

string PhysicalParameter::getName() const { return name; }
Optional<Tensor> PhysicalParameter::getStorage() { return storage; }

bool PhysicalParameter::isTrainable() const { return trainable; }
bool PhysicalParameter::isTrainingEnabled() const { return isTrainable() && trainingEnabled && !inferenceOnly; }
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
