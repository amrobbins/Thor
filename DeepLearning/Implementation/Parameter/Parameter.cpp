#include "DeepLearning/Implementation/Parameter/Parameter.h"

using namespace std;

namespace ThorImplementation {

Parameter::Parameter(string name, Tensor storage, bool trainable, bool trainingEnabled)
    : name(name), storage(storage), trainable(trainable), trainingEnabled(trainingEnabled) {
    assert(!name.empty());
    assert(!(trainable == false && trainingEnabled));
}

void Parameter::applyGradient(Tensor gradient, Stream gradientReadyStream) {
    if (!trainable || !trainingEnabled)
        return;
    assert(optimizer != nullptr);
    optimizer.get()->computeWeightsUpdate(gradient, gradientReadyStream, false);
}

bool Parameter::hasOptimizer() { return optimizer != nullptr; }
void Parameter::setOptimizer(Optional<shared_ptr<Optimizer>> newOptimizer) { this->optimizer = newOptimizer; }
shared_ptr<Optimizer> Parameter::getOptimizer() { return optimizer; }
void Parameter::clearOptimizer() { optimizer = nullptr; }

string Parameter::getName() { return name; }
Tensor Parameter::getStorage() { return storage; }

}  // namespace ThorImplementation
