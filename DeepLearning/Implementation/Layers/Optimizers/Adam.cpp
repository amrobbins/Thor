#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct Adam::RuntimeState {
    float alpha;
    float beta1;
    float beta2;
    float epsilon;
    float t;
    bool amsgrad;
};

namespace {

shared_ptr<Adam::RuntimeState> makeRuntimeState(float alpha,
                                                float beta1,
                                                float beta2,
                                                float epsilon,
                                                bool amsgrad,
                                                float t = 0.0f) {
    return make_shared<Adam::RuntimeState>(Adam::RuntimeState{alpha, beta1, beta2, epsilon, t, amsgrad});
}

vector<CustomOptimizerStateSpec> adamStateSpecs(bool amsgrad) {
    vector<CustomOptimizerStateSpec> specs{
        CustomOptimizerStateSpec::sameShapeAsWeights("m", DataType::FP32),
        CustomOptimizerStateSpec::sameShapeAsWeights("v", DataType::FP32),
    };
    if (amsgrad) {
        specs.push_back(CustomOptimizerStateSpec::sameShapeAsWeights("vhat", DataType::FP32));
    }
    return specs;
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<Adam::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // Regular Adam:
        // m_{t+1} = beta1 * m_t + (1 - beta1) * g_t
        // v_{t+1} = beta2 * v_t + (1 - beta2) * g_t^2
        // w_{t+1} = w_t - alphaT * m_{t+1} / (sqrt(v_{t+1}) + epsilon)
        //
        // AMSGrad additionally tracks vhat_{t+1} = max(vhat_t, v_{t+1}) and
        // uses vhat_{t+1} in the denominator.
        //
        // alphaT is the bias-corrected learning rate computed on CPU and passed
        // in as a runtime scalar. The raw gradient is normalized here so the same
        // expression works for dense, dense-fused, sparse-row, and sparse-row-fused updates.
        auto alphaT = context.runtimeScalar("alphaT", DataType::FP32, DataType::FP32);
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
        Expression denominatorSecondMoment = vNext;
        CustomOptimizerUpdateExpression update{{
            {"weights", Expression::constantScalar(0.0f)},
            {"m", mNext},
            {"v", vNext},
        }};

        if (state->amsgrad) {
            auto vhat = context.state("vhat", DataType::FP32, DataType::FP32);
            Expression vhatNext = vhat.max(vNext);
            denominatorSecondMoment = vhatNext;
            update.outputs.push_back({"vhat", vhatNext});
        }

        Expression wNext =
            (w - alphaT * mNext / (Expression::sqrt(denominatorSecondMoment) + epsilonExpr)).withOutputDType(weightsDType);
        update.outputs[0] = {"weights", wNext};
        return update;
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<Adam::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        state->t += 1.0f;
        const double alphaT64 = static_cast<double>(state->alpha) *
                                std::sqrt(1.0 - std::pow(static_cast<double>(state->beta2), state->t)) /
                                (1.0 - std::pow(static_cast<double>(state->beta1), state->t));
        const float alphaT = static_cast<float>(alphaT64);
        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "alphaT", alphaT},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterUpdateBuilder makeHyperParameterUpdateBuilder(shared_ptr<Adam::RuntimeState> state) {
    return [state](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        (void)epoch;
        (void)batch;
        (void)batchesPerEpoch;

        // Adam updates t when an optimizer update actually runs so dense, sparse,
        // and fused paths keep the same step-count semantics.
        return unordered_map<string, float>{{"t", state->t}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<Adam::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"t", state->t},
                                            {"alpha", state->alpha},
                                            {"beta1", state->beta1},
                                            {"beta2", state->beta2},
                                            {"epsilon", state->epsilon}};
    };
}

CustomOptimizer::HyperParameterRestoreBuilder makeHyperParameterRestoreBuilder(shared_ptr<Adam::RuntimeState> state) {
    return [state](const unordered_map<string, float>& hyperParameters) {
        if (auto it = hyperParameters.find("t"); it != hyperParameters.end()) {
            state->t = it->second;
        }
    };
}
}  // namespace

Adam::Adam(uint64_t id, float alpha, float beta1, float beta2, float epsilon, bool amsgrad)
    : Adam(id, makeRuntimeState(alpha, beta1, beta2, epsilon, amsgrad)) {}

Adam::Adam(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      adamStateSpecs(runtimeState->amsgrad),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/true,
                      makeHyperParameterUpdateBuilder(runtimeState),
                      makeHyperParameterSnapshotBuilder(runtimeState),
                      makeHyperParameterRestoreBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float Adam::getT() const { return runtimeState->t; }

float Adam::getAlpha() const { return runtimeState->alpha; }

float Adam::getBeta1() const { return runtimeState->beta1; }

float Adam::getBeta2() const { return runtimeState->beta2; }

float Adam::getEpsilon() const { return runtimeState->epsilon; }

bool Adam::getAmsgrad() const { return runtimeState->amsgrad; }

void Adam::setT(float t) { runtimeState->t = t; }

void Adam::setAlpha(float alpha) { runtimeState->alpha = alpha; }

void Adam::setBeta1(float beta1) { runtimeState->beta1 = beta1; }

void Adam::setBeta2(float beta2) { runtimeState->beta2 = beta2; }

void Adam::setEpsilon(float epsilon) { runtimeState->epsilon = epsilon; }

shared_ptr<Optimizer> Adam::clone() const {
    return shared_ptr<Adam>(new Adam(getId(),
                                     makeRuntimeState(runtimeState->alpha,
                                                      runtimeState->beta1,
                                                      runtimeState->beta2,
                                                      runtimeState->epsilon,
                                                      runtimeState->amsgrad,
                                                      runtimeState->t)));
}

}  // namespace ThorImplementation
