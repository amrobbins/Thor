#include "DeepLearning/Implementation/Layers/Optimizers/Lamb.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct Lamb::RuntimeState {
    float alpha;
    float beta1;
    float beta2;
    float epsilon;
    float weightDecay;
    float trustRatioEpsilon;
    float t;
};

namespace {

void validateRuntimeState(const Lamb::RuntimeState& state) {
    THOR_THROW_IF_FALSE(state.alpha > 0.0f);
    THOR_THROW_IF_FALSE(state.beta1 >= 0.0f && state.beta1 < 1.0f);
    THOR_THROW_IF_FALSE(state.beta2 >= 0.0f && state.beta2 < 1.0f);
    THOR_THROW_IF_FALSE(state.epsilon > 0.0f);
    THOR_THROW_IF_FALSE(state.weightDecay >= 0.0f);
    THOR_THROW_IF_FALSE(state.trustRatioEpsilon > 0.0f);
    THOR_THROW_IF_FALSE(state.t >= 0.0f);
}

shared_ptr<Lamb::RuntimeState> makeRuntimeState(float alpha,
                                                float beta1,
                                                float beta2,
                                                float epsilon,
                                                float weightDecay,
                                                float trustRatioEpsilon,
                                                float t = 0.0f) {
    auto state = make_shared<Lamb::RuntimeState>(Lamb::RuntimeState{alpha, beta1, beta2, epsilon, weightDecay, trustRatioEpsilon, t});
    validateRuntimeState(*state);
    return state;
}

vector<CustomOptimizerStateSpec> lambStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("m", DataType::FP32),
        CustomOptimizerStateSpec::sameShapeAsWeights("v", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<Lamb::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();
        const vector<uint64_t> weightDims = context.weightsTensor().getDimensions();
        if (weightDims.empty()) {
            throw invalid_argument("Lamb optimizer requires a non-scalar weight tensor.");
        }

        // LAMB:
        // m_{t+1} = beta1 * m_t + (1 - beta1) * g_t
        // v_{t+1} = beta2 * v_t + (1 - beta2) * g_t^2
        // u_t     = m_hat / (sqrt(v_hat) + epsilon) + weight_decay * w_t
        // ratio   = ||w_t||_2 / (||u_t||_2 + trust_ratio_epsilon)
        // w_{t+1} = w_t - alpha * ratio * u_t
        //
        // The raw gradient is normalized here so the same expression works for
        // standalone dense updates and dense optimizer fusion. 1-D tensors use
        // an AdamW-style ratio of 1.0, matching the common practice of excluding
        // bias and normalization parameters from the layer-wise trust ratio.
        auto alpha = context.runtimeScalar("alpha", DataType::FP32, DataType::FP32);
        auto weightDecay = context.runtimeScalar("weightDecay", DataType::FP32, DataType::FP32);
        auto invBiasCorrection1 = context.runtimeScalar("invBiasCorrection1", DataType::FP32, DataType::FP32);
        auto invBiasCorrection2 = context.runtimeScalar("invBiasCorrection2", DataType::FP32, DataType::FP32);
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
        Expression trustRatioEpsilonExpr = Expression::constantScalar(state->trustRatioEpsilon);

        Expression mNext = beta1Expr * m + oneMinusBeta1Expr * g;
        Expression vNext = beta2Expr * v + oneMinusBeta2Expr * g * g;
        Expression mHat = mNext * invBiasCorrection1;
        Expression vHat = vNext * invBiasCorrection2;
        Expression adamUpdate = mHat / (Expression::sqrt(vHat) + epsilonExpr);
        Expression update = adamUpdate + weightDecay * w;

        Expression trustRatio = Expression::constantScalar(1.0f);
        if (weightDims.size() > 1) {
            Expression weightNorm = (w * w).reduce_sum(/*reduction_axes=*/{}, /*squeeze_axes=*/{UINT64_MAX}, DataType::FP32).sqrt();
            Expression updateNorm = (update * update).reduce_sum(/*reduction_axes=*/{}, /*squeeze_axes=*/{UINT64_MAX}, DataType::FP32).sqrt();
            trustRatio = weightNorm / (updateNorm + trustRatioEpsilonExpr);
        }

        Expression wNext = (w - alpha * trustRatio * update).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"m", mNext},
            {"v", vNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<Lamb::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        state->t += 1.0f;
        const float invBiasCorrection1 = static_cast<float>(1.0 / (1.0 - pow(static_cast<double>(state->beta1), state->t)));
        const float invBiasCorrection2 = static_cast<float>(1.0 / (1.0 - pow(static_cast<double>(state->beta2), state->t)));
        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "alpha", state->alpha},
                                            {namePrefix + "weightDecay", state->weightDecay},
                                            {namePrefix + "invBiasCorrection1", invBiasCorrection1},
                                            {namePrefix + "invBiasCorrection2", invBiasCorrection2},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterUpdateBuilder makeHyperParameterUpdateBuilder(shared_ptr<Lamb::RuntimeState> state) {
    return [state](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        (void)epoch;
        (void)batch;
        (void)batchesPerEpoch;

        return unordered_map<string, float>{{"t", state->t}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<Lamb::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"t", state->t},
                                            {"alpha", state->alpha},
                                            {"beta1", state->beta1},
                                            {"beta2", state->beta2},
                                            {"epsilon", state->epsilon},
                                            {"weightDecay", state->weightDecay},
                                            {"trustRatioEpsilon", state->trustRatioEpsilon}};
    };
}

}  // namespace

Lamb::Lamb(uint64_t id, float alpha, float beta1, float beta2, float epsilon, float weightDecay, float trustRatioEpsilon)
    : Lamb(id, makeRuntimeState(alpha, beta1, beta2, epsilon, weightDecay, trustRatioEpsilon)) {}

Lamb::Lamb(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      lambStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/false,
                      makeHyperParameterUpdateBuilder(runtimeState),
                      makeHyperParameterSnapshotBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float Lamb::getT() const { return runtimeState->t; }

float Lamb::getAlpha() const { return runtimeState->alpha; }

float Lamb::getBeta1() const { return runtimeState->beta1; }

float Lamb::getBeta2() const { return runtimeState->beta2; }

float Lamb::getEpsilon() const { return runtimeState->epsilon; }

float Lamb::getWeightDecay() const { return runtimeState->weightDecay; }

float Lamb::getTrustRatioEpsilon() const { return runtimeState->trustRatioEpsilon; }

void Lamb::setT(float t) {
    THOR_THROW_IF_FALSE(t >= 0.0f);
    runtimeState->t = t;
}

void Lamb::setAlpha(float alpha) {
    THOR_THROW_IF_FALSE(alpha > 0.0f);
    runtimeState->alpha = alpha;
}

void Lamb::setBeta1(float beta1) {
    THOR_THROW_IF_FALSE(beta1 >= 0.0f && beta1 < 1.0f);
    runtimeState->beta1 = beta1;
}

void Lamb::setBeta2(float beta2) {
    THOR_THROW_IF_FALSE(beta2 >= 0.0f && beta2 < 1.0f);
    runtimeState->beta2 = beta2;
}

void Lamb::setEpsilon(float epsilon) {
    THOR_THROW_IF_FALSE(epsilon > 0.0f);
    runtimeState->epsilon = epsilon;
}

void Lamb::setWeightDecay(float weightDecay) {
    THOR_THROW_IF_FALSE(weightDecay >= 0.0f);
    runtimeState->weightDecay = weightDecay;
}

void Lamb::setTrustRatioEpsilon(float trustRatioEpsilon) {
    THOR_THROW_IF_FALSE(trustRatioEpsilon > 0.0f);
    runtimeState->trustRatioEpsilon = trustRatioEpsilon;
}

shared_ptr<Optimizer> Lamb::clone() const {
    return shared_ptr<Lamb>(new Lamb(getId(),
                                    makeRuntimeState(runtimeState->alpha,
                                                     runtimeState->beta1,
                                                     runtimeState->beta2,
                                                     runtimeState->epsilon,
                                                     runtimeState->weightDecay,
                                                     runtimeState->trustRatioEpsilon,
                                                     runtimeState->t)));
}

}  // namespace ThorImplementation
