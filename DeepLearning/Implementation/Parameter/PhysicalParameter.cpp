#include <optional>
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Initializers/Initializer.h"

#include "DeepLearning/Implementation/ThorError.h"
using namespace std;

namespace ThorImplementation {
PhysicalParameter::PhysicalParameter(string name, bool trainable) : name(name), trainable(trainable), trainingEnabled(trainable) {
    THOR_THROW_IF_FALSE(!name.empty());
}

PhysicalParameter::PhysicalParameter(string name,
                                     bool trainable,
                                     const std::vector<uint64_t>& shape,
                                     const DataType dtype)
    : name(std::move(name)), trainable(trainable), trainingEnabled(trainable), shape(shape), dtype(dtype) {}

void PhysicalParameter::compileStorage(const StorageContext& context) {
    createStorage(context);
    storageInitialized = true;
}

void PhysicalParameter::compileStorage(const Tensor& inputTensor) { compileStorage(StorageContext({{"feature_input", inputTensor}})); }

void PhysicalParameter::compileOptimizer(const std::optional<Stream>& gradientUpdateStream, bool inferenceOnly) {
    THOR_THROW_IF_FALSE(trainable == true);
    this->inferenceOnly = inferenceOnly;
    this->gradientUpdateStream = gradientUpdateStream;

    if (isTrainingEnabled()) {
        THOR_THROW_IF_FALSE(hasOptimizer());
        THOR_THROW_IF_FALSE(this->gradientUpdateStream.has_value());
        THOR_THROW_IF_FALSE(storage.has_value());
        if (!optimizer->isCompiled()) {
            optimizer->compile(storage.value(), this->gradientUpdateStream.value());
        }
    }
}

void PhysicalParameter::compileInitializer(uint64_t fanIn, uint64_t fanOut) {
    if (!hasInitializer()) {
        return;
    }

    initializer->compile(storage.value(), fanIn, fanOut);
}

void PhysicalParameter::compileInitializer() { compileInitializer(1, 1); }

void PhysicalParameter::clearStorage() { storage.reset(); }

void PhysicalParameter::initialize(Stream initStream) {
    THOR_THROW_IF_FALSE(hasInitializer());
    initializer->initialize(initStream);
}

bool PhysicalParameter::applyGradient(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(!needExpressionRecompile);
    if (!isTrainingEnabled())
        return false;
    THOR_THROW_IF_FALSE(optimizer != nullptr);
    optimizer->updateWeights(batchSize);
    return true;
}

bool PhysicalParameter::hasOptimizer() { return optimizer != nullptr; }
void PhysicalParameter::setOptimizer(std::optional<shared_ptr<Optimizer>> newOptimizer) { this->optimizer = newOptimizer.value_or(nullptr); }
// void PhysicalParameter::setOptimizer(const std::optional<shared_ptr<Optimizer>>& newOptimizer) {
//     if (newOptimizer.has_value() && newOptimizer.value() != nullptr) {
//         this->optimizer = newOptimizer.value()->clone();
//     } else {
//         this->optimizer = nullptr;
//     }
// }
shared_ptr<Optimizer> PhysicalParameter::getOptimizer() { return optimizer; }
void PhysicalParameter::clearOptimizer() { optimizer = nullptr; }

bool PhysicalParameter::hasInitializer() { return initializer != nullptr; }
void PhysicalParameter::setInitializer(std::optional<shared_ptr<Initializer>> newInitializer) { this->initializer = newInitializer.value_or(nullptr); }
// void PhysicalParameter::setInitializer(const std::optional<shared_ptr<Initializer>>& newInitializer) {
//     if (newInitializer.has_value() && newInitializer.value() != nullptr) {
//         this->initializer = newInitializer.value()->clone();
//     } else {
//         this->initializer = nullptr;
//     }
// }
shared_ptr<Initializer> PhysicalParameter::getInitializer() { return initializer; }
void PhysicalParameter::clearInitializer() { initializer = nullptr; }

string PhysicalParameter::getName() const { return name; }
std::optional<Tensor> PhysicalParameter::getStorage() { return storage; }

bool PhysicalParameter::isTrainable() const { return trainable; }
bool PhysicalParameter::isTrainingEnabled() const { return isTrainable() && trainingEnabled && !inferenceOnly; }
void PhysicalParameter::setTrainingEnabled(bool enabled) {
    THOR_THROW_IF_FALSE(isTrainable());

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
    THOR_THROW_IF_FALSE(shape.has_value());
    THOR_THROW_IF_FALSE(dtype.has_value());
    std::vector<std::string> inputNames = context.getInputNames();
    THOR_THROW_IF_FALSE(!inputNames.empty());
    Tensor anInput = context.getInput(inputNames[0]);

    storage = allocateStorage(anInput.getPlacement(), shape.value(), dtype.value());
}

Tensor PhysicalParameter::allocateStorage(const TensorPlacement placement,
                                          const std::vector<uint64_t>& shape,
                                          const DataType dtype) {
    TensorDescriptor storageDescriptor(dtype, shape);
    return Tensor(placement, storageDescriptor);
}

}  // namespace ThorImplementation
