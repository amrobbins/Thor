#include "DeepLearning/Implementation/Layers/Optimizers/Muon.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct Muon::RuntimeState {
    float alpha;
    float beta;
    float epsilon;
    float weightDecay;
    bool nesterov;
};

namespace {

void validateRuntimeState(const Muon::RuntimeState& state) {
    THOR_THROW_IF_FALSE(state.alpha > 0.0f);
    THOR_THROW_IF_FALSE(state.beta >= 0.0f && state.beta < 1.0f);
    THOR_THROW_IF_FALSE(state.epsilon > 0.0f);
    THOR_THROW_IF_FALSE(state.weightDecay >= 0.0f);
}

shared_ptr<Muon::RuntimeState> makeRuntimeState(float alpha, float beta, float epsilon, float weightDecay, bool nesterov) {
    auto state = make_shared<Muon::RuntimeState>(Muon::RuntimeState{alpha, beta, epsilon, weightDecay, nesterov});
    validateRuntimeState(*state);
    return state;
}

vector<CustomOptimizerStateSpec> muonStateSpecs() {
    return {CustomOptimizerStateSpec::sameShapeAsWeights("momentum", DataType::FP32)};
}

CustomOptimizer::UpdateExpressionBuilder makeUpdateExpressionBuilder(shared_ptr<Muon::RuntimeState> state,
                                                                      NewtonSchulzOrthogonalizationOptions options) {
    return [state, options](const CustomOptimizerUpdateContext& context) {
        const vector<uint64_t> weightDims = context.weightsTensor().getDimensions();
        if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
            throw invalid_argument("Muon matrix update requires a rank-2 non-empty weight tensor.");
        }

        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();
        auto alpha = context.runtimeScalar("alpha", DataType::FP32, DataType::FP32);
        auto alphaWeightDecay = context.runtimeScalar("alphaWeightDecay", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto momentum = context.state("momentum", DataType::FP32, DataType::FP32);

        Expression beta = Expression::constantScalar(state->beta);
        Expression oneMinusBeta = Expression::constantScalar(1.0f - state->beta);
        Expression momentumNext = beta * momentum + oneMinusBeta * g;
        Expression updateSource = state->nesterov ? beta * momentumNext + oneMinusBeta * g : momentumNext;

        NewtonSchulzOrthogonalizationOptions localOptions = options;
        localOptions.epsilon = state->epsilon;
        localOptions.outputDType = DataType::FP32;
        Expression orthogonalUpdate = newtonSchulzOrthogonalize(updateSource, weightDims[0], weightDims[1], localOptions);

        Expression wNext = (w - alphaWeightDecay * w - alpha * orthogonalUpdate).withOutputDType(weightsDType);
        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"momentum", momentumNext},
        }};
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<Muon::RuntimeState> state) {
    return [state](uint32_t batchSize, const string& namePrefix) {
        THOR_THROW_IF_FALSE(batchSize > 0);
        const float lossScalingFactor = Loss::getLossScalingFactor();
        THOR_THROW_IF_FALSE(lossScalingFactor > 0.0f);

        const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);
        return unordered_map<string, float>{{namePrefix + "alpha", state->alpha},
                                            {namePrefix + "alphaWeightDecay", state->alpha * state->weightDecay},
                                            {namePrefix + "invBatchLossScale", invBatchLossScale}};
    };
}

CustomOptimizer::HyperParameterUpdateBuilder makeHyperParameterUpdateBuilder(shared_ptr<Muon::RuntimeState> state) {
    return [state](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        (void)epoch;
        (void)batch;
        (void)batchesPerEpoch;
        return unordered_map<string, float>{{"alpha", state->alpha},
                                            {"beta", state->beta},
                                            {"epsilon", state->epsilon},
                                            {"weightDecay", state->weightDecay},
                                            {"nesterov", state->nesterov ? 1.0f : 0.0f}};
    };
}

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<Muon::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"alpha", state->alpha},
                                            {"beta", state->beta},
                                            {"epsilon", state->epsilon},
                                            {"weightDecay", state->weightDecay},
                                            {"nesterov", state->nesterov ? 1.0f : 0.0f}};
    };
}

}  // namespace

Muon::Muon(uint64_t id,
           float alpha,
           float beta,
           float epsilon,
           float weightDecay,
           bool nesterov,
           NewtonSchulzOrthogonalizationOptions orthogonalizationOptions,
           shared_ptr<Optimizer> fallbackOptimizer)
    : Muon(id, makeRuntimeState(alpha, beta, epsilon, weightDecay, nesterov), orthogonalizationOptions, std::move(fallbackOptimizer)) {}

Muon::Muon(uint64_t id,
           shared_ptr<RuntimeState> runtimeState,
           NewtonSchulzOrthogonalizationOptions orthogonalizationOptions,
           shared_ptr<Optimizer> fallbackOptimizer)
    : Optimizer(id),
      runtimeState_(std::move(runtimeState)),
      orthogonalizationOptions_(std::move(orthogonalizationOptions)),
      fallbackOptimizer_(std::move(fallbackOptimizer)) {
    THOR_THROW_IF_FALSE(runtimeState_ != nullptr);
    validateRuntimeState(*runtimeState_);
    THOR_THROW_IF_FALSE(fallbackOptimizer_ != nullptr);
}

bool Muon::shouldUseMuonMatrixPath(const Tensor& weights) const {
    const vector<uint64_t> dims = weights.getDimensions();
    return dims.size() == 2 && dims[0] > 0 && dims[1] > 0;
}

shared_ptr<Optimizer> Muon::makeMuonMatrixOptimizer() const {
    return make_shared<CustomOptimizer>(getId(),
                                        muonStateSpecs(),
                                        makeUpdateExpressionBuilder(runtimeState_, orthogonalizationOptions_),
                                        makeRuntimeScalarBuilder(runtimeState_),
                                        /*supportsSparseRowGradients=*/false,
                                        makeHyperParameterUpdateBuilder(runtimeState_),
                                        makeHyperParameterSnapshotBuilder(runtimeState_));
}

void Muon::selectOptimizerForDenseWeights(const Tensor& weights) {
    if (selectedOptimizer_ != nullptr) {
        return;
    }

    if (shouldUseMuonMatrixPath(weights)) {
        selectedOptimizer_ = makeMuonMatrixOptimizer();
        usingMuonMatrixPath_ = true;
    } else {
        selectFallbackOptimizer();
    }
}

void Muon::selectFallbackOptimizer() {
    if (selectedOptimizer_ != nullptr) {
        return;
    }
    selectedOptimizer_ = fallbackOptimizer_->clone();
    usingMuonMatrixPath_ = false;
}

void Muon::mirrorSelectedOptimizerState() {
    if (selectedOptimizer_ == nullptr) {
        return;
    }
    weightsGradient = selectedOptimizer_->getWeightsGradient();
    sparseRowGradient = selectedOptimizer_->getSparseRowGradient();
}

void Muon::compile(const Tensor& weights, Stream& gradientUpdateStream, bool materializeDenseGradient) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());

    this->weights = weights;
    this->gradientUpdateStream = gradientUpdateStream;
    selectOptimizerForDenseWeights(weights);
    selectedOptimizer_->compile(weights, gradientUpdateStream, materializeDenseGradient);
    mirrorSelectedOptimizerState();
    compiled = true;
}

SparseRowGradient Muon::compileSparseRows(const Tensor& weights, uint64_t maxSparseRows, Stream& gradientUpdateStream) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());

    this->weights = weights;
    this->gradientUpdateStream = gradientUpdateStream;
    selectFallbackOptimizer();
    SparseRowGradient gradient = selectedOptimizer_->compileSparseRows(weights, maxSparseRows, gradientUpdateStream);
    mirrorSelectedOptimizerState();
    compiled = true;
    return gradient;
}

bool Muon::supportsSparseRowGradients() const { return fallbackOptimizer_->supportsSparseRowGradients(); }

bool Muon::supportsSparseRowUpdateFusion() const { return fallbackOptimizer_->supportsSparseRowUpdateFusion(); }

bool Muon::supportsDenseUpdateFusion() const { return fallbackOptimizer_->supportsDenseUpdateFusion(); }

DenseOptimizerExpression Muon::toDenseUpdateExpression(const Tensor& weights, const Expression& gradient, const string& namePrefix) {
    selectOptimizerForDenseWeights(weights);
    return selectedOptimizer_->toDenseUpdateExpression(weights, gradient, namePrefix);
}

SparseRowOptimizerExpression Muon::toSparseRowUpdateExpression(const Tensor& weights, SparseRowGradient& sparseRowGradient) {
    selectFallbackOptimizer();
    return selectedOptimizer_->toSparseRowUpdateExpression(weights, sparseRowGradient);
}

unordered_map<string, float> Muon::denseUpdateRuntimeScalars(uint32_t batchSize, const string& namePrefix) {
    THOR_THROW_IF_FALSE(selectedOptimizer_ != nullptr);
    return selectedOptimizer_->denseUpdateRuntimeScalars(batchSize, namePrefix);
}

unordered_map<string, float> Muon::sparseRowUpdateRuntimeScalars(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(selectedOptimizer_ != nullptr);
    return selectedOptimizer_->sparseRowUpdateRuntimeScalars(batchSize);
}

void Muon::updateWeights(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(selectedOptimizer_ != nullptr);
    selectedOptimizer_->updateWeights(batchSize);
}

void Muon::updateSparseRows(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(selectedOptimizer_ != nullptr);
    selectedOptimizer_->updateSparseRows(batchSize);
}

unordered_map<string, float> Muon::updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    if (selectedOptimizer_ != nullptr) {
        return selectedOptimizer_->updateHyperParameters(epoch, batch, batchesPerEpoch);
    }
    return makeHyperParameterUpdateBuilder(runtimeState_)(epoch, batch, batchesPerEpoch);
}

unordered_map<string, float> Muon::getAllHyperParameters() {
    if (selectedOptimizer_ != nullptr) {
        return selectedOptimizer_->getAllHyperParameters();
    }
    return makeHyperParameterSnapshotBuilder(runtimeState_)();
}

void Muon::restoreHyperParameters(const unordered_map<string, float>& hyperParameters) {
    if (selectedOptimizer_ != nullptr) {
        selectedOptimizer_->restoreHyperParameters(hyperParameters);
    }
}

float Muon::getAlpha() const { return runtimeState_->alpha; }
float Muon::getBeta() const { return runtimeState_->beta; }
float Muon::getEpsilon() const { return runtimeState_->epsilon; }
float Muon::getWeightDecay() const { return runtimeState_->weightDecay; }
bool Muon::getNesterov() const { return runtimeState_->nesterov; }
const NewtonSchulzOrthogonalizationOptions& Muon::getOrthogonalizationOptions() const { return orthogonalizationOptions_; }

void Muon::setAlpha(float alpha) {
    runtimeState_->alpha = alpha;
    validateRuntimeState(*runtimeState_);
}
void Muon::setBeta(float beta) {
    runtimeState_->beta = beta;
    validateRuntimeState(*runtimeState_);
}
void Muon::setEpsilon(float epsilon) {
    runtimeState_->epsilon = epsilon;
    validateRuntimeState(*runtimeState_);
}
void Muon::setWeightDecay(float weightDecay) {
    runtimeState_->weightDecay = weightDecay;
    validateRuntimeState(*runtimeState_);
}
void Muon::setNesterov(bool nesterov) { runtimeState_->nesterov = nesterov; }

shared_ptr<Optimizer> Muon::clone() const {
    return shared_ptr<Muon>(new Muon(getId(),
                                     makeRuntimeState(runtimeState_->alpha,
                                                      runtimeState_->beta,
                                                      runtimeState_->epsilon,
                                                      runtimeState_->weightDecay,
                                                      runtimeState_->nesterov),
                                     orthogonalizationOptions_,
                                     fallbackOptimizer_->clone()));
}

}  // namespace ThorImplementation
