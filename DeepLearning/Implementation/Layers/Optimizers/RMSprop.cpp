#include "DeepLearning/Implementation/Layers/Optimizers/RMSprop.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct RMSprop::RuntimeState {
    float alpha;
    float rho;
    float epsilon;
};

namespace {

shared_ptr<RMSprop::RuntimeState> makeRuntimeState(float alpha, float rho, float epsilon) {
    return make_shared<RMSprop::RuntimeState>(RMSprop::RuntimeState{alpha, rho, epsilon});
}

vector<CustomOptimizerStateSpec> rmspropStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("square_average", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<RMSprop::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // RMSprop:
        // square_average_{t+1} = rho * square_average_t + (1 - rho) * g_t^2
        // w_{t+1} = w_t - alpha * g_t / (sqrt(square_average_{t+1}) + epsilon)
        //
        // alpha is a runtime scalar so it can be adjusted without rebuilding the
        // optimizer expression. The raw gradient is normalized here so one
        // expression works for dense, dense-fused, sparse-row, and sparse-row-fused updates.
        auto alpha = context.runtimeScalar("alpha", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto squareAverage = context.state("square_average", DataType::FP32, DataType::FP32);

        Expression rhoExpr = Expression::constantScalar(state->rho);
        Expression oneMinusRhoExpr = Expression::constantScalar(1.0f - state->rho);
        Expression epsilonExpr = Expression::constantScalar(state->epsilon);

        Expression squareAverageNext = rhoExpr * squareAverage + oneMinusRhoExpr * g * g;
        Expression wNext = (w - alpha * g / (Expression::sqrt(squareAverageNext) + epsilonExpr)).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"square_average", squareAverageNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<RMSprop::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "alpha", state->alpha},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<RMSprop::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"alpha", state->alpha}, {"rho", state->rho}, {"epsilon", state->epsilon}};
    };
}

}  // namespace

RMSprop::RMSprop(uint64_t id, float alpha, float rho, float epsilon) : RMSprop(id, makeRuntimeState(alpha, rho, epsilon)) {}

RMSprop::RMSprop(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      rmspropStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/true,
                      /*hyperParameterUpdateBuilder=*/{},
                      makeHyperParameterSnapshotBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float RMSprop::getAlpha() const { return runtimeState->alpha; }

float RMSprop::getRho() const { return runtimeState->rho; }

float RMSprop::getEpsilon() const { return runtimeState->epsilon; }

void RMSprop::setAlpha(float alpha) {
    THOR_THROW_IF_FALSE(alpha > 0.0f);
    runtimeState->alpha = alpha;
}

void RMSprop::setRho(float rho) {
    THOR_THROW_IF_FALSE(rho >= 0.0f && rho < 1.0f);
    runtimeState->rho = rho;
}

void RMSprop::setEpsilon(float epsilon) {
    THOR_THROW_IF_FALSE(epsilon > 0.0f);
    runtimeState->epsilon = epsilon;
}

shared_ptr<Optimizer> RMSprop::clone() const {
    return shared_ptr<RMSprop>(new RMSprop(getId(), makeRuntimeState(runtimeState->alpha, runtimeState->rho, runtimeState->epsilon)));
}

}  // namespace ThorImplementation
