#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Initializers/Initializer.h"

using namespace std;

namespace ThorImplementation {

Parameter::Parameter(string name, bool trainable, bool trainingEnabled)
    : name(name), trainable(trainable), trainingEnabled(trainingEnabled) {
    assert(!name.empty());
    assert(!(trainable == false && trainingEnabled));
}

// Remember this is called by API layer so that will hand over the optimizer
// 1. Create storage given featureInput 2. Compile the optimizer
void Parameter::compile(const Tensor &featureInput,
                        const Tensor &featureOutput,
                        const Optional<Stream> &gradientUpdateStream,
                        bool inferenceOnly,
                        uint64_t explicitFanIn,
                        uint64_t explicitFanOut) {
    assert(featureInput.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    createStorage(featureInput, featureInput.getPlacement().getDeviceNum());

    if (isTrainable() && !inferenceOnly) {
        assert(hasOptimizer());
        assert(gradientUpdateStream.isPresent());
        optimizer->compile(storage, gradientUpdateStream.get());
    }

    if (hasInitializer()) {
        uint64_t fanIn;
        if (explicitFanIn == 0)
            fanIn = 1;
        else
            fanIn = explicitFanIn;

        uint64_t fanOut;
        if (explicitFanOut == 0) {
            uint64_t outputNumelPerExample = 1;
            const auto outputDims = featureOutput.getDimensions();
            for (uint32_t i = 1; i < outputDims.size(); ++i) {
                outputNumelPerExample *= outputDims[i];
            }
            fanOut = max<uint64_t>(1, outputNumelPerExample / storage.get().getTotalNumElements());
        } else {
            fanOut = explicitFanOut;
        }

        Stream initStream = gradientUpdateStream.isPresent()
                                ? gradientUpdateStream.get()
                                : Stream::getMostRecentGradientUpdateStream(storage.get().getPlacement().getDeviceNum());

        initializer->compile(storage, initStream, fanIn, fanOut);
    }
}

void Parameter::createStorage(std::unordered_map<std::string, Tensor> featureInput, uint32_t gpuId) {
    throw runtime_error("Not implemented");
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
