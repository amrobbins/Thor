#include "DeepLearning/Implementation/Layers/Optimizers/Adagrad.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <memory>
#include <utility>
#include <unordered_map>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct Adagrad::RuntimeState {
    float alpha;
    float epsilon;
};

namespace {

shared_ptr<Adagrad::RuntimeState> makeRuntimeState(float alpha, float epsilon) {
    return make_shared<Adagrad::RuntimeState>(Adagrad::RuntimeState{alpha, epsilon});
}

vector<CustomOptimizerStateSpec> adagradStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("accumulator", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<Adagrad::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // Adagrad:
        // accumulator_{t+1} = accumulator_t + g_t^2
        // w_{t+1} = w_t - alpha * g_t / (sqrt(accumulator_{t+1}) + epsilon)
        //
        // alpha is a runtime scalar so it can be adjusted without rebuilding the
        // optimizer expression. The raw gradient is normalized here so one
        // expression works for dense, dense-fused, sparse-row, and sparse-row-fused updates.
        auto alpha = context.runtimeScalar("alpha", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto accumulator = context.state("accumulator", DataType::FP32, DataType::FP32);

        Expression epsilonExpr = Expression::constantScalar(state->epsilon);

        Expression accumulatorNext = accumulator + g * g;
        Expression wNext = (w - alpha * g / (Expression::sqrt(accumulatorNext) + epsilonExpr)).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"accumulator", accumulatorNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<Adagrad::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "alpha", state->alpha},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<Adagrad::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"alpha", state->alpha}, {"epsilon", state->epsilon}};
    };
}

}  // namespace

Adagrad::Adagrad(uint64_t id, float alpha, float epsilon) : Adagrad(id, makeRuntimeState(alpha, epsilon)) {}

Adagrad::Adagrad(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      adagradStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/true,
                      /*hyperParameterUpdateBuilder=*/{},
                      makeHyperParameterSnapshotBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float Adagrad::getAlpha() const { return runtimeState->alpha; }

float Adagrad::getEpsilon() const { return runtimeState->epsilon; }

void Adagrad::setAlpha(float alpha) {
    THOR_THROW_IF_FALSE(alpha > 0.0f);
    runtimeState->alpha = alpha;
}

void Adagrad::setEpsilon(float epsilon) {
    THOR_THROW_IF_FALSE(epsilon > 0.0f);
    runtimeState->epsilon = epsilon;
}

shared_ptr<Optimizer> Adagrad::clone() const {
    return shared_ptr<Adagrad>(new Adagrad(getId(), makeRuntimeState(runtimeState->alpha, runtimeState->epsilon)));
}

}  // namespace ThorImplementation
