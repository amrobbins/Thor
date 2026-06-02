#include "DeepLearning/Implementation/Layers/Optimizers/Adadelta.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct Adadelta::RuntimeState {
    float alpha;
    float rho;
    float epsilon;
};

namespace {

shared_ptr<Adadelta::RuntimeState> makeRuntimeState(float alpha, float rho, float epsilon) {
    return make_shared<Adadelta::RuntimeState>(Adadelta::RuntimeState{alpha, rho, epsilon});
}

vector<CustomOptimizerStateSpec> adadeltaStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("gradient_square_average", DataType::FP32),
        CustomOptimizerStateSpec::sameShapeAsWeights("update_square_average", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<Adadelta::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // Adadelta:
        // gradient_square_average_{t+1} = rho * gradient_square_average_t + (1 - rho) * g_t^2
        // update_t = sqrt(update_square_average_t + epsilon) / sqrt(gradient_square_average_{t+1} + epsilon) * g_t
        // update_square_average_{t+1} = rho * update_square_average_t + (1 - rho) * update_t^2
        // w_{t+1} = w_t - alpha * update_t
        //
        // alpha is a runtime scalar so it can be adjusted without rebuilding the
        // optimizer expression. The raw gradient is normalized here so one
        // expression works for dense, dense-fused, sparse-row, and sparse-row-fused updates.
        auto alpha = context.runtimeScalar("alpha", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto gradientSquareAverage = context.state("gradient_square_average", DataType::FP32, DataType::FP32);
        auto updateSquareAverage = context.state("update_square_average", DataType::FP32, DataType::FP32);

        Expression rhoExpr = Expression::constantScalar(state->rho);
        Expression oneMinusRhoExpr = Expression::constantScalar(1.0f - state->rho);
        Expression epsilonExpr = Expression::constantScalar(state->epsilon);

        Expression gradientSquareAverageNext = rhoExpr * gradientSquareAverage + oneMinusRhoExpr * g * g;
        Expression update = Expression::sqrt(updateSquareAverage + epsilonExpr) /
                            Expression::sqrt(gradientSquareAverageNext + epsilonExpr) * g;
        Expression updateSquareAverageNext = rhoExpr * updateSquareAverage + oneMinusRhoExpr * update * update;
        Expression wNext = (w - alpha * update).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"gradient_square_average", gradientSquareAverageNext},
            {"update_square_average", updateSquareAverageNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<Adadelta::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "alpha", state->alpha},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<Adadelta::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"alpha", state->alpha}, {"rho", state->rho}, {"epsilon", state->epsilon}};
    };
}

}  // namespace

Adadelta::Adadelta(uint64_t id, float alpha, float rho, float epsilon) : Adadelta(id, makeRuntimeState(alpha, rho, epsilon)) {}

Adadelta::Adadelta(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      adadeltaStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/true,
                      /*hyperParameterUpdateBuilder=*/{},
                      makeHyperParameterSnapshotBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float Adadelta::getAlpha() const { return runtimeState->alpha; }

float Adadelta::getRho() const { return runtimeState->rho; }

float Adadelta::getEpsilon() const { return runtimeState->epsilon; }

void Adadelta::setAlpha(float alpha) {
    THOR_THROW_IF_FALSE(alpha > 0.0f);
    runtimeState->alpha = alpha;
}

void Adadelta::setRho(float rho) {
    THOR_THROW_IF_FALSE(rho >= 0.0f && rho < 1.0f);
    runtimeState->rho = rho;
}

void Adadelta::setEpsilon(float epsilon) {
    THOR_THROW_IF_FALSE(epsilon > 0.0f);
    runtimeState->epsilon = epsilon;
}

shared_ptr<Optimizer> Adadelta::clone() const {
    return shared_ptr<Adadelta>(new Adadelta(getId(), makeRuntimeState(runtimeState->alpha, runtimeState->rho, runtimeState->epsilon)));
}

}  // namespace ThorImplementation
