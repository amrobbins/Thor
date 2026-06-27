#include "DeepLearning/Implementation/Layers/Optimizers/AdamW.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct AdamW::RuntimeState {
    float alpha;
    float beta1;
    float beta2;
    float epsilon;
    float weightDecay;
    float t;
};

namespace {

shared_ptr<AdamW::RuntimeState> makeRuntimeState(float alpha,
                                                 float beta1,
                                                 float beta2,
                                                 float epsilon,
                                                 float weightDecay,
                                                 float t = 0.0f) {
    return make_shared<AdamW::RuntimeState>(AdamW::RuntimeState{alpha, beta1, beta2, epsilon, weightDecay, t});
}

vector<CustomOptimizerStateSpec> adamWStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("m", DataType::FP32),
        CustomOptimizerStateSpec::sameShapeAsWeights("v", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<AdamW::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // AdamW:
        // m_{t+1} = beta1 * m_t + (1 - beta1) * g_t
        // v_{t+1} = beta2 * v_t + (1 - beta2) * g_t^2
        // w_{t+1} = w_t - alpha * weight_decay * w_t
        //                 - alphaT * m_{t+1} / (sqrt(v_{t+1}) + epsilon)
        //
        // alphaT and alphaWeightDecay are computed on CPU and passed as runtime
        // scalars. The raw gradient is normalized here so one expression works for
        // dense, dense-fused, sparse-row, and sparse-row-fused updates.
        auto alphaT = context.runtimeScalar("alphaT", DataType::FP32, DataType::FP32);
        auto alphaWeightDecay = context.runtimeScalar("alphaWeightDecay", DataType::FP32, DataType::FP32);
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
        Expression wNext = (w - alphaWeightDecay * w - alphaT * mNext / (Expression::sqrt(vNext) + epsilonExpr)).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"m", mNext},
            {"v", vNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<AdamW::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        state->t += 1.0f;
        const double alphaT64 = static_cast<double>(state->alpha) *
                                std::sqrt(1.0 - std::pow(static_cast<double>(state->beta2), state->t)) /
                                (1.0 - std::pow(static_cast<double>(state->beta1), state->t));
        const float alphaT = static_cast<float>(alphaT64);
        const float alphaWeightDecay = state->alpha * state->weightDecay;
        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "alphaT", alphaT},
                                            {namePrefix + "alphaWeightDecay", alphaWeightDecay},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterUpdateBuilder makeHyperParameterUpdateBuilder(shared_ptr<AdamW::RuntimeState> state) {
    return [state](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        (void)epoch;
        (void)batch;
        (void)batchesPerEpoch;

        // AdamW updates t when an optimizer update actually runs so dense, sparse,
        // and fused paths keep the same step-count semantics.
        return unordered_map<string, float>{{"t", state->t}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<AdamW::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"t", state->t},
                                            {"alpha", state->alpha},
                                            {"beta1", state->beta1},
                                            {"beta2", state->beta2},
                                            {"epsilon", state->epsilon},
                                            {"weightDecay", state->weightDecay}};
    };
}

CustomOptimizer::HyperParameterRestoreBuilder makeHyperParameterRestoreBuilder(shared_ptr<AdamW::RuntimeState> state) {
    return [state](const unordered_map<string, float>& hyperParameters) {
        if (auto it = hyperParameters.find("t"); it != hyperParameters.end()) {
            state->t = it->second;
        }
    };
}
}  // namespace

AdamW::AdamW(uint64_t id, float alpha, float beta1, float beta2, float epsilon, float weightDecay)
    : AdamW(id, makeRuntimeState(alpha, beta1, beta2, epsilon, weightDecay)) {}

AdamW::AdamW(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      adamWStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/true,
                      makeHyperParameterUpdateBuilder(runtimeState),
                      makeHyperParameterSnapshotBuilder(runtimeState),
                      makeHyperParameterRestoreBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float AdamW::getT() const { return runtimeState->t; }

float AdamW::getAlpha() const { return runtimeState->alpha; }

float AdamW::getBeta1() const { return runtimeState->beta1; }

float AdamW::getBeta2() const { return runtimeState->beta2; }

float AdamW::getEpsilon() const { return runtimeState->epsilon; }

float AdamW::getWeightDecay() const { return runtimeState->weightDecay; }

void AdamW::setT(float t) { runtimeState->t = t; }

void AdamW::setAlpha(float alpha) { runtimeState->alpha = alpha; }

void AdamW::setBeta1(float beta1) { runtimeState->beta1 = beta1; }

void AdamW::setBeta2(float beta2) { runtimeState->beta2 = beta2; }

void AdamW::setEpsilon(float epsilon) { runtimeState->epsilon = epsilon; }

void AdamW::setWeightDecay(float weightDecay) { runtimeState->weightDecay = weightDecay; }

shared_ptr<Optimizer> AdamW::clone() const {
    return shared_ptr<AdamW>(new AdamW(getId(),
                                       makeRuntimeState(runtimeState->alpha,
                                                        runtimeState->beta1,
                                                        runtimeState->beta2,
                                                        runtimeState->epsilon,
                                                        runtimeState->weightDecay,
                                                        runtimeState->t)));
}

}  // namespace ThorImplementation
