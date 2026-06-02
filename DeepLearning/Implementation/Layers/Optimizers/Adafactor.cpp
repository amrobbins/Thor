#include "DeepLearning/Implementation/Layers/Optimizers/Adafactor.h"

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace ThorImplementation {

struct Adafactor::RuntimeState {
    float alpha;
    float beta2;
    float epsilon;
    float weightDecay;
    bool factorSecondMoment;
};

namespace {

void validateRuntimeState(const Adafactor::RuntimeState& state) {
    THOR_THROW_IF_FALSE(state.alpha > 0.0f);
    THOR_THROW_IF_FALSE(state.beta2 >= 0.0f && state.beta2 < 1.0f);
    THOR_THROW_IF_FALSE(state.epsilon > 0.0f);
    THOR_THROW_IF_FALSE(state.weightDecay >= 0.0f);
}

shared_ptr<Adafactor::RuntimeState> makeRuntimeState(float alpha, float beta2, float epsilon, float weightDecay, bool factorSecondMoment) {
    auto state = make_shared<Adafactor::RuntimeState>(Adafactor::RuntimeState{alpha, beta2, epsilon, weightDecay, factorSecondMoment});
    validateRuntimeState(*state);
    return state;
}

vector<CustomOptimizerStateSpec> unfactoredStateSpecs() {
    return {CustomOptimizerStateSpec::sameShapeAsWeights("second_moment", DataType::FP32)};
}

vector<CustomOptimizerStateSpec> factoredStateSpecs(const vector<uint64_t>& weightDims) {
    if (weightDims.size() < 2)
        throw invalid_argument("Factored Adafactor requires a rank-2 or higher weight tensor.");

    vector<uint64_t> rowShape = weightDims;
    vector<uint64_t> columnShape = weightDims;
    rowShape.back() = 1;
    columnShape[columnShape.size() - 2] = 1;

    return {
        CustomOptimizerStateSpec{"row_second_moment", DataType::FP32, optional<vector<uint64_t>>(rowShape), false},
        CustomOptimizerStateSpec{"column_second_moment", DataType::FP32, optional<vector<uint64_t>>(columnShape), false},
    };
}

CustomOptimizer::RuntimeScalarBuilder makeRuntimeScalarBuilder(shared_ptr<Adafactor::RuntimeState> state) {
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

CustomOptimizer::HyperParameterSnapshotBuilder makeHyperParameterSnapshotBuilder(shared_ptr<Adafactor::RuntimeState> state) {
    return [state]() {
        return unordered_map<string, float>{{"alpha", state->alpha},
                                            {"beta2", state->beta2},
                                            {"epsilon", state->epsilon},
                                            {"weightDecay", state->weightDecay},
                                            {"factorSecondMoment", state->factorSecondMoment ? 1.0f : 0.0f}};
    };
}

CustomOptimizer::UpdateExpressionBuilder makeUnfactoredUpdateExpressionBuilder(shared_ptr<Adafactor::RuntimeState> state) {
    return [state](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();

        // Unfactored Adafactor fallback:
        // second_moment_{t+1} = beta2 * second_moment_t + (1 - beta2) * g_t^2
        // update_t            = g_t / sqrt(second_moment_{t+1} + epsilon)
        // w_{t+1}             = w_t - alpha * update_t - alpha * weight_decay * w_t
        //
        // This path is also used for sparse-row gradients because sparse-row optimizer
        // fusion intentionally supports pointwise update math only.
        auto alpha = context.runtimeScalar("alpha", DataType::FP32, DataType::FP32);
        auto alphaWeightDecay = context.runtimeScalar("alphaWeightDecay", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto secondMoment = context.state("second_moment", DataType::FP32, DataType::FP32);

        Expression beta2 = Expression::constantScalar(state->beta2);
        Expression oneMinusBeta2 = Expression::constantScalar(1.0f - state->beta2);
        Expression epsilon = Expression::constantScalar(state->epsilon);

        Expression secondMomentNext = beta2 * secondMoment + oneMinusBeta2 * g * g;
        Expression update = g / Expression::sqrt(secondMomentNext + epsilon);
        Expression wNext = (w - alphaWeightDecay * w - alpha * update).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"second_moment", secondMomentNext},
        }};
    };
}

CustomOptimizer::UpdateExpressionBuilder makeFactoredUpdateExpressionBuilder(shared_ptr<Adafactor::RuntimeState> state,
                                                                             vector<uint64_t> weightDims) {
    return [state, weightDims = std::move(weightDims)](const CustomOptimizerUpdateContext& context) {
        const DataType weightsDType = context.weightsTensor().getDescriptor().getDataType();
        if (context.weightsTensor().getDimensions() != weightDims)
            throw invalid_argument("Factored Adafactor update was built for different weight dimensions.");
        if (weightDims.size() < 2)
            throw invalid_argument("Factored Adafactor requires a rank-2 or higher weight tensor.");

        // Factored Adafactor over the final two dimensions of a weight tensor with shape [..., M, N]:
        // row_second_moment_{t+1}    = beta2 * row_t + (1 - beta2) * mean(g_t^2, axis=-1)
        // column_second_moment_{t+1} = beta2 * column_t + (1 - beta2) * mean(g_t^2, axis=-2)
        // second_moment_hat          = row_next * column_next / mean(row_next, axis=-2)
        // update_t                   = g_t / sqrt(second_moment_hat + epsilon)
        // w_{t+1}                    = w_t - alpha * update_t - alpha * weight_decay * w_t
        //
        // Compute mean(row_next, axis=-2) algebraically from the old row state and gradient square instead of
        // reducing rowSecondMomentNext directly. This keeps rowSecondMomentNext as a named optimizer-state output
        // rather than letting the staged reduction planner materialize it only as an intermediate for the second reduction.
        auto alpha = context.runtimeScalar("alpha", DataType::FP32, DataType::FP32);
        auto alphaWeightDecay = context.runtimeScalar("alphaWeightDecay", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = context.runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = context.weights(DataType::FP32, DataType::FP32);
        auto g = context.gradient() * invBatchLossScale;
        auto rowSecondMoment = context.state("row_second_moment", DataType::FP32, DataType::FP32);
        auto columnSecondMoment = context.state("column_second_moment", DataType::FP32, DataType::FP32);

        Expression beta2 = Expression::constantScalar(state->beta2);
        Expression oneMinusBeta2 = Expression::constantScalar(1.0f - state->beta2);
        Expression epsilon = Expression::constantScalar(state->epsilon);

        const uint64_t rank = static_cast<uint64_t>(weightDims.size());
        const uint64_t rowAxis = rank - 2;
        const uint64_t columnAxis = rank - 1;

        Expression gradientSquared = g * g;
        Expression rowSecondMomentNext =
            beta2 * rowSecondMoment + oneMinusBeta2 * gradientSquared.reduce_mean({columnAxis}, {}, DataType::FP32);
        Expression columnSecondMomentNext =
            beta2 * columnSecondMoment + oneMinusBeta2 * gradientSquared.reduce_mean({rowAxis}, {}, DataType::FP32);
        Expression rowMean = beta2 * rowSecondMoment.reduce_mean({rowAxis}, {}, DataType::FP32) +
                             oneMinusBeta2 * gradientSquared.reduce_mean({rowAxis, columnAxis}, {}, DataType::FP32);
        Expression secondMomentEstimate = (rowSecondMomentNext / (rowMean + epsilon)) * columnSecondMomentNext;
        Expression update = g / Expression::sqrt(secondMomentEstimate + epsilon);
        Expression wNext = (w - alphaWeightDecay * w - alpha * update).withOutputDType(weightsDType);

        return CustomOptimizerUpdateExpression{{
            {"weights", wNext},
            {"row_second_moment", rowSecondMomentNext},
            {"column_second_moment", columnSecondMomentNext},
        }};
    };
}

}  // namespace

Adafactor::Adafactor(uint64_t id, float alpha, float beta2, float epsilon, float weightDecay, bool factorSecondMoment)
    : Adafactor(id, makeRuntimeState(alpha, beta2, epsilon, weightDecay, factorSecondMoment)) {}

Adafactor::Adafactor(uint64_t id, shared_ptr<RuntimeState> runtimeState) : Optimizer(id), runtimeState_(std::move(runtimeState)) {
    THOR_THROW_IF_FALSE(runtimeState_ != nullptr);
    validateRuntimeState(*runtimeState_);
}

bool Adafactor::shouldUseFactoredPath(const Tensor& weights) const {
    return runtimeState_->factorSecondMoment && weights.getDimensions().size() >= 2;
}

shared_ptr<Optimizer> Adafactor::makeFactoredOptimizer(const vector<uint64_t>& weightDims) const {
    return make_shared<CustomOptimizer>(getId(),
                                        factoredStateSpecs(weightDims),
                                        makeFactoredUpdateExpressionBuilder(runtimeState_, weightDims),
                                        makeRuntimeScalarBuilder(runtimeState_),
                                        /*supportsSparseRowGradients=*/false,
                                        /*hyperParameterUpdateBuilder=*/CustomOptimizer::HyperParameterUpdateBuilder{},
                                        makeHyperParameterSnapshotBuilder(runtimeState_));
}

shared_ptr<Optimizer> Adafactor::makeUnfactoredOptimizer() const {
    return make_shared<CustomOptimizer>(getId(),
                                        unfactoredStateSpecs(),
                                        makeUnfactoredUpdateExpressionBuilder(runtimeState_),
                                        makeRuntimeScalarBuilder(runtimeState_),
                                        /*supportsSparseRowGradients=*/true,
                                        /*hyperParameterUpdateBuilder=*/CustomOptimizer::HyperParameterUpdateBuilder{},
                                        makeHyperParameterSnapshotBuilder(runtimeState_));
}

void Adafactor::selectOptimizerForDenseWeights(const Tensor& weights) {
    if (selectedOptimizer_ != nullptr)
        return;

    if (shouldUseFactoredPath(weights)) {
        selectedOptimizer_ = makeFactoredOptimizer(weights.getDimensions());
        usingFactoredPath_ = true;
    } else {
        selectedOptimizer_ = makeUnfactoredOptimizer();
        usingFactoredPath_ = false;
    }
}

void Adafactor::selectUnfactoredOptimizer() {
    if (selectedOptimizer_ != nullptr) {
        if (usingFactoredPath_)
            throw runtime_error(
                "Adafactor physical optimizer already selected the factored dense path and cannot switch to sparse-row mode.");
        return;
    }
    selectedOptimizer_ = makeUnfactoredOptimizer();
    usingFactoredPath_ = false;
}

void Adafactor::mirrorSelectedOptimizerState() {
    if (selectedOptimizer_ == nullptr)
        return;

    weightsGradient = selectedOptimizer_->getWeightsGradient();
    sparseRowGradient = selectedOptimizer_->getSparseRowGradient();
}

void Adafactor::compile(const Tensor& weights, Stream& gradientUpdateStream, bool materializeDenseGradient) {
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

SparseRowGradient Adafactor::compileSparseRows(const Tensor& weights, uint64_t maxSparseRows, Stream& gradientUpdateStream) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());

    this->weights = weights;
    this->gradientUpdateStream = gradientUpdateStream;
    selectUnfactoredOptimizer();
    SparseRowGradient gradient = selectedOptimizer_->compileSparseRows(weights, maxSparseRows, gradientUpdateStream);
    mirrorSelectedOptimizerState();
    compiled = true;
    return gradient;
}

DenseOptimizerExpression Adafactor::toDenseUpdateExpression(const Tensor& weights, const Expression& gradient, const string& namePrefix) {
    selectOptimizerForDenseWeights(weights);
    return selectedOptimizer_->toDenseUpdateExpression(weights, gradient, namePrefix);
}

SparseRowOptimizerExpression Adafactor::toSparseRowUpdateExpression(const Tensor& weights, SparseRowGradient& sparseRowGradient) {
    selectUnfactoredOptimizer();
    return selectedOptimizer_->toSparseRowUpdateExpression(weights, sparseRowGradient);
}

unordered_map<string, float> Adafactor::denseUpdateRuntimeScalars(uint32_t batchSize, const string& namePrefix) {
    if (selectedOptimizer_ != nullptr)
        return selectedOptimizer_->denseUpdateRuntimeScalars(batchSize, namePrefix);
    return makeRuntimeScalarBuilder(runtimeState_)(batchSize, namePrefix);
}

unordered_map<string, float> Adafactor::sparseRowUpdateRuntimeScalars(uint32_t batchSize) {
    if (selectedOptimizer_ != nullptr)
        return selectedOptimizer_->sparseRowUpdateRuntimeScalars(batchSize);
    return makeRuntimeScalarBuilder(runtimeState_)(batchSize, "");
}

void Adafactor::updateWeights(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(selectedOptimizer_ != nullptr);
    selectedOptimizer_->updateWeights(batchSize);
}

void Adafactor::updateSparseRows(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(selectedOptimizer_ != nullptr);
    selectedOptimizer_->updateSparseRows(batchSize);
}

unordered_map<string, float> Adafactor::updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    if (selectedOptimizer_ != nullptr)
        return selectedOptimizer_->updateHyperParameters(epoch, batch, batchesPerEpoch);
    (void)epoch;
    (void)batch;
    (void)batchesPerEpoch;
    return {};
}

unordered_map<string, float> Adafactor::getAllHyperParameters() {
    if (selectedOptimizer_ != nullptr)
        return selectedOptimizer_->getAllHyperParameters();
    return makeHyperParameterSnapshotBuilder(runtimeState_)();
}

float Adafactor::getAlpha() const { return runtimeState_->alpha; }
float Adafactor::getBeta2() const { return runtimeState_->beta2; }
float Adafactor::getEpsilon() const { return runtimeState_->epsilon; }
float Adafactor::getWeightDecay() const { return runtimeState_->weightDecay; }
bool Adafactor::getFactorSecondMoment() const { return runtimeState_->factorSecondMoment; }

void Adafactor::setAlpha(float alpha) {
    runtimeState_->alpha = alpha;
    validateRuntimeState(*runtimeState_);
}
void Adafactor::setBeta2(float beta2) {
    runtimeState_->beta2 = beta2;
    validateRuntimeState(*runtimeState_);
}
void Adafactor::setEpsilon(float epsilon) {
    runtimeState_->epsilon = epsilon;
    validateRuntimeState(*runtimeState_);
}
void Adafactor::setWeightDecay(float weightDecay) {
    runtimeState_->weightDecay = weightDecay;
    validateRuntimeState(*runtimeState_);
}
void Adafactor::setFactorSecondMoment(bool factorSecondMoment) { runtimeState_->factorSecondMoment = factorSecondMoment; }

shared_ptr<Optimizer> Adafactor::clone() const {
    return shared_ptr<Adafactor>(new Adafactor(getId(),
                                               makeRuntimeState(runtimeState_->alpha,
                                                                runtimeState_->beta2,
                                                                runtimeState_->epsilon,
                                                                runtimeState_->weightDecay,
                                                                runtimeState_->factorSecondMoment)));
}

}  // namespace ThorImplementation
