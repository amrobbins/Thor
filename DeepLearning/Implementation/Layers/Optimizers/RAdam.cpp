#include "DeepLearning/Implementation/Layers/Optimizers/RAdam.h"

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

struct RAdam::RuntimeState {
    float alpha;
    float beta1;
    float beta2;
    float epsilon;
    float t;
};

namespace {

shared_ptr<RAdam::RuntimeState> makeRuntimeState(float alpha, float beta1, float beta2, float epsilon, float t = 0.0f) {
    return make_shared<RAdam::RuntimeState>(RAdam::RuntimeState{alpha, beta1, beta2, epsilon, t});
}

vector<CustomOptimizerStateSpec> radamStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("m", DataType::FP32),
        CustomOptimizerStateSpec::sameShapeAsWeights("v", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<RAdam::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // RAdam:
        // m_{t+1} = beta1 * m_t + (1 - beta1) * g_t
        // v_{t+1} = beta2 * v_t + (1 - beta2) * g_t^2
        // rho_t = rho_inf - 2 * t * beta2^t / (1 - beta2^t)
        // if rho_t >= 5:
        //     w_{t+1} = w_t - rectifiedAlphaT * m_{t+1} / (sqrt(v_{t+1}) + epsilon)
        // else:
        //     w_{t+1} = w_t - unrectifiedAlphaT * m_{t+1}
        //
        // The branch decision and bias-corrected step sizes are computed on CPU
        // and passed as runtime scalars. The expression graph blends the two paths
        // arithmetically instead of using where/select so sparse-row optimizer
        // kernels only need their existing scalar pointwise math surface. The raw
        // gradient is normalized here.
        auto rectifiedAlphaT = context.runtimeScalar("rectifiedAlphaT", DataType::FP32, DataType::FP32);
        auto unrectifiedAlphaT = context.runtimeScalar("unrectifiedAlphaT", DataType::FP32, DataType::FP32);
        auto useRectified = context.runtimeScalar("useRectified", DataType::FP32, DataType::FP32);
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
        Expression rectifiedStep = rectifiedAlphaT * mNext / (Expression::sqrt(vNext) + epsilonExpr);
        Expression unrectifiedStep = unrectifiedAlphaT * mNext;
        Expression step = useRectified * rectifiedStep + (Expression::constantScalar(1.0f) - useRectified) * unrectifiedStep;
        Expression wNext = (w - step).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"m", mNext},
            {"v", vNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<RAdam::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        state->t += 1.0f;

        const double beta1 = static_cast<double>(state->beta1);
        const double beta2 = static_cast<double>(state->beta2);
        const double t = static_cast<double>(state->t);
        const double beta1PowT = std::pow(beta1, t);
        const double beta2PowT = std::pow(beta2, t);
        const double oneMinusBeta1PowT = 1.0 - beta1PowT;
        const double oneMinusBeta2PowT = 1.0 - beta2PowT;
        THOR_THROW_IF_FALSE(oneMinusBeta1PowT > 0.0);
        THOR_THROW_IF_FALSE(oneMinusBeta2PowT > 0.0);

        const double rhoInf = 2.0 / (1.0 - beta2) - 1.0;
        const double rhoT = rhoInf - (2.0 * t * beta2PowT / oneMinusBeta2PowT);
        const bool shouldRectify = rhoT >= 5.0;

        double rectifiedAlphaT64 = 0.0;
        if (shouldRectify) {
            const double numerator = (rhoT - 4.0) * (rhoT - 2.0) * rhoInf;
            const double denominator = (rhoInf - 4.0) * (rhoInf - 2.0) * rhoT;
            THOR_THROW_IF_FALSE(numerator > 0.0);
            THOR_THROW_IF_FALSE(denominator > 0.0);
            const double rectification = std::sqrt(numerator / denominator);
            rectifiedAlphaT64 = static_cast<double>(state->alpha) * rectification * std::sqrt(oneMinusBeta2PowT) /
                                oneMinusBeta1PowT;
        }

        const double unrectifiedAlphaT64 = static_cast<double>(state->alpha) / oneMinusBeta1PowT;
        const float rectifiedAlphaT = static_cast<float>(rectifiedAlphaT64);
        const float unrectifiedAlphaT = static_cast<float>(unrectifiedAlphaT64);
        const float useRectified = shouldRectify ? 1.0f : 0.0f;
        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "rectifiedAlphaT", rectifiedAlphaT},
                                            {namePrefix + "unrectifiedAlphaT", unrectifiedAlphaT},
                                            {namePrefix + "useRectified", useRectified},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterUpdateBuilder makeHyperParameterUpdateBuilder(shared_ptr<RAdam::RuntimeState> state) {
    return [state](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        (void)epoch;
        (void)batch;
        (void)batchesPerEpoch;

        // RAdam updates t when an optimizer update actually runs so dense,
        // sparse, and fused paths keep the same step-count semantics.
        return unordered_map<string, float>{{"t", state->t}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<RAdam::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"t", state->t},
                                            {"alpha", state->alpha},
                                            {"beta1", state->beta1},
                                            {"beta2", state->beta2},
                                            {"epsilon", state->epsilon}};
    };
}

CustomOptimizer::HyperParameterRestoreBuilder makeHyperParameterRestoreBuilder(shared_ptr<RAdam::RuntimeState> state) {
    return [state](const unordered_map<string, float>& hyperParameters) {
        if (auto it = hyperParameters.find("t"); it != hyperParameters.end()) {
            state->t = it->second;
        }
    };
}
}  // namespace

RAdam::RAdam(uint64_t id, float alpha, float beta1, float beta2, float epsilon)
    : RAdam(id, makeRuntimeState(alpha, beta1, beta2, epsilon)) {}

RAdam::RAdam(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      radamStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/true,
                      makeHyperParameterUpdateBuilder(runtimeState),
                      makeHyperParameterSnapshotBuilder(runtimeState),
                      makeHyperParameterRestoreBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float RAdam::getT() const { return runtimeState->t; }

float RAdam::getAlpha() const { return runtimeState->alpha; }

float RAdam::getBeta1() const { return runtimeState->beta1; }

float RAdam::getBeta2() const { return runtimeState->beta2; }

float RAdam::getEpsilon() const { return runtimeState->epsilon; }

void RAdam::setT(float t) {
    THOR_THROW_IF_FALSE(t >= 0.0f);
    runtimeState->t = t;
}

void RAdam::setAlpha(float alpha) {
    THOR_THROW_IF_FALSE(alpha > 0.0f);
    runtimeState->alpha = alpha;
}

void RAdam::setBeta1(float beta1) {
    THOR_THROW_IF_FALSE(beta1 >= 0.0f && beta1 < 1.0f);
    runtimeState->beta1 = beta1;
}

void RAdam::setBeta2(float beta2) {
    THOR_THROW_IF_FALSE(beta2 >= 0.0f && beta2 < 1.0f);
    runtimeState->beta2 = beta2;
}

void RAdam::setEpsilon(float epsilon) {
    THOR_THROW_IF_FALSE(epsilon > 0.0f);
    runtimeState->epsilon = epsilon;
}

shared_ptr<Optimizer> RAdam::clone() const {
    return shared_ptr<RAdam>(new RAdam(getId(),
                                       makeRuntimeState(runtimeState->alpha,
                                                        runtimeState->beta1,
                                                        runtimeState->beta2,
                                                        runtimeState->epsilon,
                                                        runtimeState->t)));
}

}  // namespace ThorImplementation
