#include "DeepLearning/Api/Optimizers/CustomOptimizer.h"

#include <stdexcept>

using namespace std;
using json = nlohmann::json;

namespace Thor {

CustomOptimizer::CustomOptimizer() : Optimizer() {}

CustomOptimizer::CustomOptimizer(uint64_t originalId) : Optimizer(originalId) {}

CustomOptimizer::CustomOptimizer(std::vector<StateSpec> stateSpecs,
                                 UpdateExpressionBuilder updateExpressionBuilder,
                                 RuntimeScalarBuilder runtimeScalarBuilder,
                                 bool supportsSparseRowGradients)
    : Optimizer(),
      stateSpecs(std::move(stateSpecs)),
      updateExpressionBuilder(std::move(updateExpressionBuilder)),
      runtimeScalarBuilder(std::move(runtimeScalarBuilder)),
      supportsSparseRowGradients(supportsSparseRowGradients) {
    THOR_THROW_IF_FALSE(static_cast<bool>(this->updateExpressionBuilder));
}

shared_ptr<ThorImplementation::Optimizer> CustomOptimizer::stamp(shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) {
    (void)trainableLayer;
    return make_shared<ThorImplementation::CustomOptimizer>(
        getId(), stateSpecs, updateExpressionBuilder, runtimeScalarBuilder, supportsSparseRowGradients);
}

json CustomOptimizer::architectureJson() const {
    throw runtime_error(
        "CustomOptimizer is not serializable because arbitrary C++ update-expression and runtime-scalar callbacks cannot be saved. "
        "Create a named Optimizer subclass with an explicit architectureJson()/deserialize() implementation for serializable optimizers.");
}

shared_ptr<Optimizer> CustomOptimizer::clone() const { return make_shared<CustomOptimizer>(*this); }

}  // namespace Thor
