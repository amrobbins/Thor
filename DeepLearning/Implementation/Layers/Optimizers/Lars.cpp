#include "DeepLearning/Implementation/Layers/Optimizers/Lars.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct Lars::RuntimeState {
    float alpha;
    float momentum;
    float weightDecay;
    float trustCoefficient;
    float epsilon;
    bool useNesterovMomentum;
};

namespace {

void validateRuntimeState(const Lars::RuntimeState& state) {
    THOR_THROW_IF_FALSE(state.alpha > 0.0f);
    THOR_THROW_IF_FALSE(state.momentum >= 0.0f);
    THOR_THROW_IF_FALSE(state.weightDecay >= 0.0f);
    THOR_THROW_IF_FALSE(state.trustCoefficient > 0.0f);
    THOR_THROW_IF_FALSE(state.epsilon > 0.0f);
}

shared_ptr<Lars::RuntimeState> makeRuntimeState(float alpha,
                                                float momentum,
                                                float weightDecay,
                                                float trustCoefficient,
                                                float epsilon,
                                                bool useNesterovMomentum) {
    auto state = make_shared<Lars::RuntimeState>(
        Lars::RuntimeState{alpha, momentum, weightDecay, trustCoefficient, epsilon, useNesterovMomentum});
    validateRuntimeState(*state);
    return state;
}

vector<CustomOptimizerStateSpec> larsStateSpecs() {
    return {
        CustomOptimizerStateSpec::sameShapeAsWeights("velocity", DataType::FP32),
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<Lars::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();
        const vector<uint64_t> weightDims = context.weightsTensor().getDimensions();
        if (weightDims.empty()) {
            throw invalid_argument("LARS optimizer requires a non-scalar weight tensor.");
        }

        // LARS:
        // g_t       = normalized raw gradient
        // u_t       = g_t + weight_decay * w_t
        // ratio     = trust_coefficient * ||w_t||_2 / (||g_t||_2 + weight_decay * ||w_t||_2 + epsilon)
        // v_{t+1}   = momentum * v_t + alpha * ratio * u_t
        // w_{t+1}   = w_t - v_{t+1}
        //
        // For 1-D tensors, Thor uses a trust ratio of 1.0. This matches the
        // common practice of excluding bias and normalization parameters from
        // layer-wise adaptive scaling while still applying the base SGD update.
        auto alpha = context.runtimeScalar("alpha", DataType::FP32, DataType::FP32);
        auto weightDecay = context.runtimeScalar("weightDecay", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto velocity = context.state("velocity", DataType::FP32, DataType::FP32);

        Expression momentumExpr = Expression::constantScalar(state->momentum);
        Expression trustCoefficientExpr = Expression::constantScalar(state->trustCoefficient);
        Expression epsilonExpr = Expression::constantScalar(state->epsilon);

        Expression trustRatio = Expression::constantScalar(1.0f);
        if (weightDims.size() > 1) {
            Expression weightNorm = (w * w).reduce_sum(/*reduction_axes=*/{}, /*squeeze_axes=*/{UINT64_MAX}, DataType::FP32).sqrt();
            Expression gradientNorm = (g * g).reduce_sum(/*reduction_axes=*/{}, /*squeeze_axes=*/{UINT64_MAX}, DataType::FP32).sqrt();
            trustRatio = trustCoefficientExpr * weightNorm / (gradientNorm + weightDecay * weightNorm + epsilonExpr);
        }

        Expression update = g + weightDecay * w;
        Expression scaledUpdate = alpha * trustRatio * update;
        Expression velocityNext = momentumExpr * velocity + scaledUpdate;
        Expression weightsUpdate = velocityNext;
        if (state->useNesterovMomentum) {
            weightsUpdate = momentumExpr * velocityNext + scaledUpdate;
        }
        Expression wNext = (w - weightsUpdate).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"velocity", velocityNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<Lars::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

        return unordered_map<string, float>{{namePrefix + "alpha", state->alpha},
                                            {namePrefix + "weightDecay", state->weightDecay},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<Lars::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"alpha", state->alpha},
                                            {"momentum", state->momentum},
                                            {"weightDecay", state->weightDecay},
                                            {"trustCoefficient", state->trustCoefficient},
                                            {"epsilon", state->epsilon},
                                            {"useNesterovMomentum", state->useNesterovMomentum ? 1.0f : 0.0f}};
    };
}

}  // namespace

Lars::Lars(uint64_t id,
           float alpha,
           float momentum,
           float weightDecay,
           float trustCoefficient,
           float epsilon,
           bool useNesterovMomentum)
    : Lars(id, makeRuntimeState(alpha, momentum, weightDecay, trustCoefficient, epsilon, useNesterovMomentum)) {}

Lars::Lars(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      larsStateSpecs(),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/false,
                      CustomOptimizer::HyperParameterUpdateBuilder{},
                      makeHyperParameterSnapshotBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

float Lars::getAlpha() const { return runtimeState->alpha; }

float Lars::getMomentum() const { return runtimeState->momentum; }

float Lars::getWeightDecay() const { return runtimeState->weightDecay; }

float Lars::getTrustCoefficient() const { return runtimeState->trustCoefficient; }

float Lars::getEpsilon() const { return runtimeState->epsilon; }

bool Lars::getUseNesterovMomentum() const { return runtimeState->useNesterovMomentum; }

void Lars::setAlpha(float alpha) {
    THOR_THROW_IF_FALSE(alpha > 0.0f);
    runtimeState->alpha = alpha;
}

void Lars::setMomentum(float momentum) {
    THOR_THROW_IF_FALSE(momentum >= 0.0f);
    runtimeState->momentum = momentum;
}

void Lars::setWeightDecay(float weightDecay) {
    THOR_THROW_IF_FALSE(weightDecay >= 0.0f);
    runtimeState->weightDecay = weightDecay;
}

void Lars::setTrustCoefficient(float trustCoefficient) {
    THOR_THROW_IF_FALSE(trustCoefficient > 0.0f);
    runtimeState->trustCoefficient = trustCoefficient;
}

void Lars::setEpsilon(float epsilon) {
    THOR_THROW_IF_FALSE(epsilon > 0.0f);
    runtimeState->epsilon = epsilon;
}

void Lars::setUseNesterovMomentum(bool useNesterovMomentum) { runtimeState->useNesterovMomentum = useNesterovMomentum; }

shared_ptr<Optimizer> Lars::clone() const {
    return shared_ptr<Lars>(new Lars(getId(),
                                    makeRuntimeState(runtimeState->alpha,
                                                     runtimeState->momentum,
                                                     runtimeState->weightDecay,
                                                     runtimeState->trustCoefficient,
                                                     runtimeState->epsilon,
                                                     runtimeState->useNesterovMomentum)));
}

}  // namespace ThorImplementation
