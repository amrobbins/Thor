#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct Sgd::RuntimeState {
    float initialLearningRate;
    float decay;
    float momentum;
    bool useNesterovMomentum;
    uint64_t currentEpoch;
    uint64_t currentBatch = 0;
    float currentLearningRate;
};

namespace {

shared_ptr<Sgd::RuntimeState> makeRuntimeState(float initialLearningRate,
                                               float decay,
                                               float momentum,
                                               bool useNesterovMomentum,
                                               uint64_t startResumeEpoch) {
    return make_shared<Sgd::RuntimeState>(Sgd::RuntimeState{initialLearningRate,
                                                            decay,
                                                            momentum,
                                                            useNesterovMomentum,
                                                            startResumeEpoch,
                                                            0,
                                                            initialLearningRate});
}

vector<CustomOptimizerStateSpec> stateSpecsFor(const Sgd::RuntimeState& state) {
    vector<CustomOptimizerStateSpec> stateSpecs;
    if (state.momentum > 0.0f) {
        stateSpecs.push_back(CustomOptimizerStateSpec::sameShapeAndDTypeAsWeights("velocity"));
    }
    return stateSpecs;
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<Sgd::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient();
        auto step = context.runtimeScalar("step", DataType::FP32, DataType::FP32);

        if (state->momentum > 0.0f) {
            auto v = context.state("velocity", DataType::FP32, DataType::FP32);
            auto mu = Expression::constantScalar(state->momentum);
            auto vNext = (mu * v - step * g).withOutputDType(weightsDType);

            if (state->useNesterovMomentum) {
                // v_{t+1} = mu * v_t - step * g
                // w_{t+1} = w_t + mu * v_{t+1} - step * g
                auto wNext = (w + mu * vNext - step * g).withOutputDType(weightsDType);
                return CustomOptimizerUpdateExpression{{
                    {"weights", wNext},
                    {"velocity", vNext},
                }};
            }

            // Classical momentum:
            // v_{t+1} = mu * v_t - step * g
            // w_{t+1} = w_t + v_{t+1}
            auto wNext = (w + vNext).withOutputDType(weightsDType);
            return CustomOptimizerUpdateExpression{{
                {"weights", wNext},
                {"velocity", vNext},
            }};
        }

        return CustomOptimizerUpdateExpression{{
            {"weights", (w - step * g).withOutputDType(weightsDType)},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<Sgd::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        const float step = state->currentLearningRate / (static_cast<float>(batchSize) * lossScalingFactor);
        return unordered_map<string, float>{{namePrefix + "step", step}};
    };
}

CustomOptimizer::HyperParameterUpdateBuilder makeHyperParameterUpdateBuilder(shared_ptr<Sgd::RuntimeState> state) {
    return [state](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        (void)batchesPerEpoch;
        if (state->currentEpoch != epoch) {
            state->currentLearningRate = static_cast<float>(
                static_cast<double>(state->initialLearningRate) *
                std::pow(1.0 - static_cast<double>(state->decay), static_cast<double>(epoch)));
        }
        state->currentEpoch = epoch;
        state->currentBatch = batch;
        return unordered_map<string, float>{{"currentLearningRate", state->currentLearningRate}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<Sgd::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"currentLearningRate", state->currentLearningRate},
                                            {"initialLearningRate", state->initialLearningRate},
                                            {"decay", state->decay},
                                            {"momentum", state->momentum},
                                            {"useNesterovMomentum", state->useNesterovMomentum ? 1.0f : 0.0f},
                                            {"epoch", static_cast<float>(state->currentEpoch)}};
    };
}

}  // namespace

Sgd::Sgd(uint64_t id, float initialLearningRate, float decay, float momentum, bool useNesterovMomentum, uint64_t startResumeEpoch)
    : Sgd(id, makeRuntimeState(initialLearningRate, decay, momentum, useNesterovMomentum, startResumeEpoch)) {}

Sgd::Sgd(uint64_t id, shared_ptr<RuntimeState> runtimeState)
    : CustomOptimizer(id,
                      stateSpecsFor(*runtimeState),
                      makeUpdateExpressionBuilder(runtimeState),
                      makeRuntimeScalarBuilder(runtimeState),
                      /*supportsSparseRowGradients=*/true,
                      makeHyperParameterUpdateBuilder(runtimeState),
                      makeHyperParameterSnapshotBuilder(runtimeState)),
      runtimeState(std::move(runtimeState)) {}

void Sgd::setInitialLearningRate(float initialLearningRate) {
    THOR_THROW_IF_FALSE(initialLearningRate >= 0.0f);
    runtimeState->initialLearningRate = initialLearningRate;
}

void Sgd::setDecay(float decay) {
    THOR_THROW_IF_FALSE(decay >= 0.0f);
    THOR_THROW_IF_FALSE(decay <= 1.0f);
    runtimeState->decay = decay;
}

void Sgd::setMomentum(float momentum) {
    THOR_THROW_IF_FALSE(momentum >= 0.0f);
    runtimeState->momentum = momentum;
}

void Sgd::setUseNesterovMomentum(bool useNesterovMomentum) { runtimeState->useNesterovMomentum = useNesterovMomentum; }

float Sgd::getInitialLearningRate() const { return runtimeState->initialLearningRate; }

float Sgd::getDecay() const { return runtimeState->decay; }

float Sgd::getMomentum() const { return runtimeState->momentum; }

bool Sgd::getUseNesterovMomentum() const { return runtimeState->useNesterovMomentum; }

uint64_t Sgd::getEpoch() const { return runtimeState->currentEpoch; }

float Sgd::getCurrentLearningRate() const { return runtimeState->currentLearningRate; }

shared_ptr<Optimizer> Sgd::clone() const {
    return make_shared<Sgd>(getId(),
                            runtimeState->initialLearningRate,
                            runtimeState->decay,
                            runtimeState->momentum,
                            runtimeState->useNesterovMomentum,
                            runtimeState->currentEpoch);
}

}  // namespace ThorImplementation
