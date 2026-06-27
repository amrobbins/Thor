#include "DeepLearning/Implementation/Layers/Optimizers/Adamax.h"

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

struct Adamax::RuntimeState {
    float alpha;
    float beta1;
    float beta2;
    float epsilon;
    float t;
};

namespace {

shared_ptr<Adamax::RuntimeState> makeRuntimeState(float alpha, float beta1, float beta2, float epsilon, float t = 0.0f) {
    return make_shared<Adamax::RuntimeState>(Adamax::RuntimeState{alpha, beta1, beta2, epsilon, t});
}

vector<CustomOptimizerStateSpec> adamaxStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("m", DataType::FP32),
        CustomOptimizerStateSpec::sameShapeAsWeights("u", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<Adamax::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // Adamax:
        // m_{t+1} = beta1 * m_t + (1 - beta1) * g_t
        // u_{t+1} = max(beta2 * u_t, abs(g_t))
        // w_{t+1} = w_t - alphaT * m_{t+1} / (u_{t+1} + epsilon)
        //
        // alphaT is the bias-corrected learning rate computed on CPU and passed
        // as a runtime scalar. The raw gradient is normalized here so the same
        // expression works for dense, dense-fused, sparse-row, and sparse-row-fused updates.
        auto alphaT = context.runtimeScalar("alphaT", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto m = context.state("m", DataType::FP32, DataType::FP32);
        auto u = context.state("u", DataType::FP32, DataType::FP32);

        Expression beta1Expr = Expression::constantScalar(state->beta1);
        Expression beta2Expr = Expression::constantScalar(state->beta2);
        Expression oneMinusBeta1Expr = Expression::constantScalar(1.0f - state->beta1);
        Expression epsilonExpr = Expression::constantScalar(state->epsilon);

        Expression mNext = beta1Expr * m + oneMinusBeta1Expr * g;
        Expression uNext = (beta2Expr * u).max(g.abs());
        Expression wNext = (w - alphaT * mNext / (uNext + epsilonExpr)).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"m", mNext},
            {"u", uNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<Adamax::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        state->t += 1.0f;
        const double alphaT64 = static_cast<double>(state->alpha) / (1.0 - std::pow(static_cast<double>(state->beta1), state->t));
        const float alphaT = static_cast<float>(alphaT64);
        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "alphaT", alphaT},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterUpdateBuilder makeHyperParameterUpdateBuilder(shared_ptr<Adamax::RuntimeState> state) {
    return [state](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        (void)epoch;
        (void)batch;
        (void)batchesPerEpoch;

        // Adamax updates t when an optimizer update actually runs so dense,
        // sparse, and fused paths keep the same step-count semantics.
        return unordered_map<string, float>{{"t", state->t}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<Adamax::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"t", state->t},
                                            {"alpha", state->alpha},
                                            {"beta1", state->beta1},
                                            {"beta2", state->beta2},
                                            {"epsilon", state->epsilon}};
    };
}

CustomOptimizer::HyperParameterRestoreBuilder makeHyperParameterRestoreBuilder(shared_ptr<Adamax::RuntimeState> state) {
    return [state](const unordered_map<string, float>& hyperParameters) {
        if (auto it = hyperParameters.find("t"); it != hyperParameters.end()) {
            state->t = it->second;
        }
    };
}
}  // namespace

Adamax::Adamax(uint64_t id, float alpha, float beta1, float beta2, float epsilon)
    : Adamax(id, makeRuntimeState(alpha, beta1, beta2, epsilon)) {}

Adamax::Adamax(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      adamaxStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/true,
                      makeHyperParameterUpdateBuilder(runtimeState),
                      makeHyperParameterSnapshotBuilder(runtimeState),
                      makeHyperParameterRestoreBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float Adamax::getT() const { return runtimeState->t; }

float Adamax::getAlpha() const { return runtimeState->alpha; }

float Adamax::getBeta1() const { return runtimeState->beta1; }

float Adamax::getBeta2() const { return runtimeState->beta2; }

float Adamax::getEpsilon() const { return runtimeState->epsilon; }

void Adamax::setT(float t) {
    THOR_THROW_IF_FALSE(t >= 0.0f);
    runtimeState->t = t;
}

void Adamax::setAlpha(float alpha) {
    THOR_THROW_IF_FALSE(alpha > 0.0f);
    runtimeState->alpha = alpha;
}

void Adamax::setBeta1(float beta1) {
    THOR_THROW_IF_FALSE(beta1 >= 0.0f && beta1 < 1.0f);
    runtimeState->beta1 = beta1;
}

void Adamax::setBeta2(float beta2) {
    THOR_THROW_IF_FALSE(beta2 >= 0.0f && beta2 < 1.0f);
    runtimeState->beta2 = beta2;
}

void Adamax::setEpsilon(float epsilon) {
    THOR_THROW_IF_FALSE(epsilon > 0.0f);
    runtimeState->epsilon = epsilon;
}

shared_ptr<Optimizer> Adamax::clone() const {
    return shared_ptr<Adamax>(new Adamax(getId(),
                                         makeRuntimeState(runtimeState->alpha,
                                                          runtimeState->beta1,
                                                          runtimeState->beta2,
                                                          runtimeState->epsilon,
                                                          runtimeState->t)));
}

}  // namespace ThorImplementation
