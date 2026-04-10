#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Initializers/Initializer.h"

using namespace std;

namespace ThorImplementation {

Parameter::Parameter(string name, bool trainable, bool trainingEnabled)
    : name(name), trainable(trainable), trainingEnabled(trainingEnabled) {
    assert(!name.empty());
    assert(!(trainable == false && trainingEnabled));
}

void Parameter::compileStorageAndOptimizer(const std::vector<uint64_t>& inputDims,
                                           const std::vector<uint64_t>& outputDims,
                                           const TensorDescriptor::DataType& outputDataType,
                                           const TensorPlacement& placement,
                                           const Optional<Stream>& gradientUpdateStream,
                                           bool inferenceOnly) {
    this->outputDims = outputDims;
    this->gradientUpdateStream = gradientUpdateStream;

    createStorage(inputDims, outputDims, outputDataType, placement);

    if (isTrainable() && !inferenceOnly) {
        assert(hasOptimizer());
        assert(gradientUpdateStream.isPresent());
        optimizer->compile(storage, gradientUpdateStream.get());
    }
}

void Parameter::compileInitializer() { compileInitializer(0, 0); }

void Parameter::compileInitializer(uint64_t explicitFanIn, uint64_t explicitFanOut) {
    if (!hasInitializer()) {
        return;
    }

    const uint64_t fanIn = (explicitFanIn == 0) ? 1 : explicitFanIn;

    uint64_t fanOut = explicitFanOut;
    if (fanOut == 0) {
        uint64_t outputNumelPerExample = 1;
        for (uint32_t i = 1; i < outputDims.size(); ++i) {
            outputNumelPerExample *= outputDims[i];
        }

        const uint64_t paramNumel = storage.get().getTotalNumElements();
        fanOut = (paramNumel == 0) ? 1 : std::max<uint64_t>(1, outputNumelPerExample / paramNumel);
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

void Parameter::applyGradient(uint32_t batchSize) {
    if (isTrainingEnabled())
        return;
    assert(optimizer != nullptr);
    optimizer.get()->updateWeights(batchSize);
}

bool Parameter::hasOptimizer() { return optimizer != nullptr; }
void Parameter::setOptimizer(Optional<shared_ptr<Optimizer>> newOptimizer) { this->optimizer = newOptimizer; }
shared_ptr<Optimizer> Parameter::getOptimizer() { return optimizer; }
void Parameter::clearOptimizer() { optimizer = nullptr; }

bool Parameter::hasInitializer() { return initializer != nullptr; }
void Parameter::setInitializer(Optional<shared_ptr<Initializer>> newInitializer) { this->initializer = newInitializer; }
shared_ptr<Initializer> Parameter::getInitializer() { return initializer; }
void Parameter::clearInitializer() { initializer = nullptr; }

string Parameter::getName() { return name; }
Tensor Parameter::getStorage() { return storage; }

bool Parameter::isTrainable() const { return trainable; }
bool Parameter::isTrainingEnabled() const { return isTrainable() && trainingEnabled; }
void Parameter::setTrainingEnabled(bool enabled) {
    assert(isTrainable());

    throw runtime_error("Toggling parameter trainabilty on/off is not yet supported.");
    // Will need to ensure that the gradients are not computed for this parameter when trainability is off.
    trainingEnabled = enabled;
}

}  // namespace ThorImplementation
