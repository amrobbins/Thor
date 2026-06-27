#include "DeepLearning/Implementation/Layers/Optimizers/NAdam.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct NAdam::RuntimeState {
    float alpha;
    float beta1;
    float beta2;
    float epsilon;
    float t;
};

namespace {

shared_ptr<NAdam::RuntimeState> makeRuntimeState(float alpha, float beta1, float beta2, float epsilon, float t = 0.0f) {
    return make_shared<NAdam::RuntimeState>(NAdam::RuntimeState{alpha, beta1, beta2, epsilon, t});
}

vector<CustomOptimizerStateSpec> nadamStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("m", DataType::FP32),
        CustomOptimizerStateSpec::sameShapeAsWeights("v", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<NAdam::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // NAdam:
        // m_{t+1} = beta1 * m_t + (1 - beta1) * g_t
        // v_{t+1} = beta2 * v_t + (1 - beta2) * g_t^2
        // w_{t+1} = w_t - (mScale * m_{t+1} + gradientScale * g_t) /
        //                 (sqrt(v_{t+1}) + epsilon)
        //
        // mScale and gradientScale include alpha plus the Nesterov/Adam bias
        // correction terms. The raw gradient is normalized here so the same
        // expression works for dense, dense-fused, sparse-row, and sparse-row-fused updates.
        auto mScale = context.runtimeScalar("mScale", DataType::FP32, DataType::FP32);
        auto gradientScale = context.runtimeScalar("gradientScale", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto m = context.state("m", DataType::FP32, DataType::FP32);
        auto v = context.state("v", DataType::FP32, DataType::FP32);

        Expression beta1Expr = Expression::constantScalar(state->beta1);
        Expression beta2Expr = Expression::constantScalar(state->beta2);
        Expression oneMinusBeta1Expr = Expression::constantScalar(1.0f - state->beta1);
        Expression oneMinusBeta2Expr = Expression::constantScalar(1.0f - state->beta2);
        Expression epsilonExpr = Expression::constantScalar(state->epsilon);

        Expression mNext = beta1Expr * m + oneMinusBeta1Expr * g;
        Expression vNext = beta2Expr * v + oneMinusBeta2Expr * g * g;
        Expression nesterovMomentum = mScale * mNext + gradientScale * g;
        Expression wNext = (w - nesterovMomentum / (Expression::sqrt(vNext) + epsilonExpr)).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"m", mNext},
            {"v", vNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<NAdam::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        state->t += 1.0f;

        const double beta1 = static_cast<double>(state->beta1);
        const double beta2 = static_cast<double>(state->beta2);
        const double t = static_cast<double>(state->t);
        const double sqrtOneMinusBeta2PowT = std::sqrt(1.0 - std::pow(beta2, t));

        const double mScale64 = static_cast<double>(state->alpha) * beta1 * sqrtOneMinusBeta2PowT /
                                (1.0 - std::pow(beta1, t + 1.0));
        const double gradientScale64 = static_cast<double>(state->alpha) * (1.0 - beta1) * sqrtOneMinusBeta2PowT /
                                       (1.0 - std::pow(beta1, t));

        const float mScale = static_cast<float>(mScale64);
        const float gradientScale = static_cast<float>(gradientScale64);
        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "mScale", mScale},
                                            {namePrefix + "gradientScale", gradientScale},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterUpdateBuilder makeHyperParameterUpdateBuilder(shared_ptr<NAdam::RuntimeState> state) {
    return [state](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        (void)epoch;
        (void)batch;
        (void)batchesPerEpoch;

        // NAdam updates t when an optimizer update actually runs so dense,
        // sparse, and fused paths keep the same step-count semantics.
        return unordered_map<string, float>{{"t", state->t}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<NAdam::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"t", state->t},
                                            {"alpha", state->alpha},
                                            {"beta1", state->beta1},
                                            {"beta2", state->beta2},
                                            {"epsilon", state->epsilon}};
    };
}

CustomOptimizer::HyperParameterRestoreBuilder makeHyperParameterRestoreBuilder(shared_ptr<NAdam::RuntimeState> state) {
    return [state](const unordered_map<string, float>& hyperParameters) {
        if (auto it = hyperParameters.find("t"); it != hyperParameters.end()) {
            state->t = it->second;
        }
    };
}
}  // namespace

NAdam::NAdam(uint64_t id, float alpha, float beta1, float beta2, float epsilon)
    : NAdam(id, makeRuntimeState(alpha, beta1, beta2, epsilon)) {}

NAdam::NAdam(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      nadamStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/true,
                      makeHyperParameterUpdateBuilder(runtimeState),
                      makeHyperParameterSnapshotBuilder(runtimeState),
                      makeHyperParameterRestoreBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float NAdam::getT() const { return runtimeState->t; }

float NAdam::getAlpha() const { return runtimeState->alpha; }

float NAdam::getBeta1() const { return runtimeState->beta1; }

float NAdam::getBeta2() const { return runtimeState->beta2; }

float NAdam::getEpsilon() const { return runtimeState->epsilon; }

void NAdam::setT(float t) {
    THOR_THROW_IF_FALSE(t >= 0.0f);
    runtimeState->t = t;
}

void NAdam::setAlpha(float alpha) {
    THOR_THROW_IF_FALSE(alpha > 0.0f);
    runtimeState->alpha = alpha;
}

void NAdam::setBeta1(float beta1) {
    THOR_THROW_IF_FALSE(beta1 >= 0.0f && beta1 < 1.0f);
    runtimeState->beta1 = beta1;
}

void NAdam::setBeta2(float beta2) {
    THOR_THROW_IF_FALSE(beta2 >= 0.0f && beta2 < 1.0f);
    runtimeState->beta2 = beta2;
}

void NAdam::setEpsilon(float epsilon) {
    THOR_THROW_IF_FALSE(epsilon > 0.0f);
    runtimeState->epsilon = epsilon;
}

shared_ptr<Optimizer> NAdam::clone() const {
    return shared_ptr<NAdam>(new NAdam(getId(),
                                       makeRuntimeState(runtimeState->alpha,
                                                        runtimeState->beta1,
                                                        runtimeState->beta2,
                                                        runtimeState->epsilon,
                                                        runtimeState->t)));
}

}  // namespace ThorImplementation
