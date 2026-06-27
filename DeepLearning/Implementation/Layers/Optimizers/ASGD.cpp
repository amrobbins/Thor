#include "DeepLearning/Implementation/Layers/Optimizers/ASGD.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct ASGD::RuntimeState {
    float alpha;
    float lambd;
    float power;
    float t0;
    float weightDecay;
    float t;
};

namespace {

void validateRuntimeState(const ASGD::RuntimeState& state) {
    THOR_THROW_IF_FALSE(state.alpha > 0.0f);
    THOR_THROW_IF_FALSE(state.lambd >= 0.0f);
    THOR_THROW_IF_FALSE(state.power >= 0.0f);
    THOR_THROW_IF_FALSE(state.t0 >= 1.0f);
    THOR_THROW_IF_FALSE(state.weightDecay >= 0.0f);
    THOR_THROW_IF_FALSE(state.t >= 0.0f);
}

shared_ptr<ASGD::RuntimeState> makeRuntimeState(float alpha,
                                                float lambd,
                                                float power,
                                                float t0,
                                                float weightDecay,
                                                float t = 0.0f) {
    auto state = make_shared<ASGD::RuntimeState>(ASGD::RuntimeState{alpha, lambd, power, t0, weightDecay, t});
    validateRuntimeState(*state);
    return state;
}

vector<CustomOptimizerStateSpec> asgdStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("averaged_weights", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<ASGD::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        if (!context.isDense()) {
            throw runtime_error("ASGD optimizer supports dense updates only; averaged weights are full-tensor optimizer state.");
        }

        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // ASGD:
        // g_t       = normalized raw gradient
        // eta_t     = alpha / (1 + lambd * alpha * t)^power
        // w_{t+1}   = (1 - lambd * eta_t) * w_t - eta_t * (g_t + weight_decay * w_t)
        // ax_{t+1}  = ax_t before t0, then running average of w_{t+1}
        //
        // The update step and averaging coefficient are CPU-computed runtime
        // scalars so the expression remains pointwise and works with Thor's
        // normal dense-update and dense-fusion paths.
        auto eta = context.runtimeScalar("eta", DataType::FP32, DataType::FP32);
        auto lambdEta = context.runtimeScalar("lambdEta", DataType::FP32, DataType::FP32);
        auto averagingScale = context.runtimeScalar("averagingScale", DataType::FP32, DataType::FP32);
        auto weightDecay = context.runtimeScalar("weightDecay", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto averagedWeights = context.state("averaged_weights", DataType::FP32, DataType::FP32);

        Expression one = Expression::constantScalar(1.0f);
        Expression update = g + weightDecay * w;
        Expression wNextFp32 = (one - lambdEta) * w - eta * update;
        Expression averagedWeightsNext = averagedWeights + averagingScale * (wNextFp32 - averagedWeights);
        Expression wNext = wNextFp32.withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"averaged_weights", averagedWeightsNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<ASGD::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        state->t += 1.0f;

        const double alpha = static_cast<double>(state->alpha);
        const double lambd = static_cast<double>(state->lambd);
        const double power = static_cast<double>(state->power);
        const double t = static_cast<double>(state->t);
        const double eta64 = alpha / std::pow(1.0 + lambd * alpha * t, power);
        const float eta = static_cast<float>(eta64);
        const float lambdEta = static_cast<float>(lambd * eta64);
        const float averagingScale = state->t >= state->t0 ? 1.0f / (state->t - state->t0 + 1.0f) : 0.0f;
        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "eta", eta},
                                            {namePrefix + "lambdEta", lambdEta},
                                            {namePrefix + "averagingScale", averagingScale},
                                            {namePrefix + "weightDecay", state->weightDecay},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterUpdateBuilder makeHyperParameterUpdateBuilder(shared_ptr<ASGD::RuntimeState> state) {
    return [state](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        (void)epoch;
        (void)batch;
        (void)batchesPerEpoch;

        // ASGD increments t when an optimizer update actually runs so dense and
        // fused dense paths share the same step-count semantics.
        return unordered_map<string, float>{{"t", state->t}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<ASGD::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"t", state->t},
                                            {"alpha", state->alpha},
                                            {"lambd", state->lambd},
                                            {"power", state->power},
                                            {"t0", state->t0},
                                            {"weightDecay", state->weightDecay}};
    };
}

CustomOptimizer::HyperParameterRestoreBuilder makeHyperParameterRestoreBuilder(shared_ptr<ASGD::RuntimeState> state) {
    return [state](const unordered_map<string, float>& hyperParameters) {
        if (auto it = hyperParameters.find("t"); it != hyperParameters.end()) {
            state->t = it->second;
        }
    };
}
}  // namespace

ASGD::ASGD(uint64_t id, float alpha, float lambd, float power, float t0, float weightDecay)
    : ASGD(id, makeRuntimeState(alpha, lambd, power, t0, weightDecay)) {}

ASGD::ASGD(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      asgdStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/false,
                      makeHyperParameterUpdateBuilder(runtimeState),
                      makeHyperParameterSnapshotBuilder(runtimeState),
                      makeHyperParameterRestoreBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float ASGD::getAlpha() const { return runtimeState->alpha; }

float ASGD::getLambd() const { return runtimeState->lambd; }

float ASGD::getPower() const { return runtimeState->power; }

float ASGD::getT0() const { return runtimeState->t0; }

float ASGD::getWeightDecay() const { return runtimeState->weightDecay; }

float ASGD::getT() const { return runtimeState->t; }

void ASGD::setAlpha(float alpha) {
    THOR_THROW_IF_FALSE(alpha > 0.0f);
    runtimeState->alpha = alpha;
}

void ASGD::setLambd(float lambd) {
    THOR_THROW_IF_FALSE(lambd >= 0.0f);
    runtimeState->lambd = lambd;
}

void ASGD::setPower(float power) {
    THOR_THROW_IF_FALSE(power >= 0.0f);
    runtimeState->power = power;
}

void ASGD::setT0(float t0) {
    THOR_THROW_IF_FALSE(t0 >= 1.0f);
    runtimeState->t0 = t0;
}

void ASGD::setWeightDecay(float weightDecay) {
    THOR_THROW_IF_FALSE(weightDecay >= 0.0f);
    runtimeState->weightDecay = weightDecay;
}

void ASGD::setT(float t) {
    THOR_THROW_IF_FALSE(t >= 0.0f);
    runtimeState->t = t;
}

shared_ptr<Optimizer> ASGD::clone() const {
    return shared_ptr<ASGD>(new ASGD(getId(),
                                     makeRuntimeState(runtimeState->alpha,
                                                      runtimeState->lambd,
                                                      runtimeState->power,
                                                      runtimeState->t0,
                                                      runtimeState->weightDecay,
                                                      runtimeState->t)));
}

}  // namespace ThorImplementation
