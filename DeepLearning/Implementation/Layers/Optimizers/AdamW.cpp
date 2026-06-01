#include "DeepLearning/Implementation/Layers/Optimizers/AdamW.h"

#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Initializers/ZerosInitializer.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cmath>
#include <stdexcept>

using namespace std;

namespace ThorImplementation {

namespace {

void ensureAdamWState(Optimizer &optimizer, const Tensor &weights, Stream &gradientUpdateStream) {
    if (!optimizer.hasParameter("m")) {
        THOR_THROW_IF_FALSE(!optimizer.hasParameter("v"));
        shared_ptr<PhysicalParameter> mParameter = make_shared<PhysicalParameter>("m", false, weights.getDimensions(), DataType::FP32);
        shared_ptr<PhysicalParameter> vParameter = make_shared<PhysicalParameter>("v", false, weights.getDimensions(), DataType::FP32);
        shared_ptr<Initializer> paramInitializer = make_shared<ZerosInitializer>();

        mParameter->setInitializer(paramInitializer->clone());
        optimizer.addParameter(mParameter);
        mParameter->compileStorage(weights);
        mParameter->compileInitializer();

        vParameter->setInitializer(paramInitializer->clone());
        optimizer.addParameter(vParameter);
        vParameter->compileStorage(weights);
        vParameter->compileInitializer();

        THOR_THROW_IF_FALSE(mParameter->getStorage().has_value());
        THOR_THROW_IF_FALSE(vParameter->getStorage().has_value());
        mParameter->initialize(gradientUpdateStream);
        vParameter->initialize(gradientUpdateStream);
    }

    THOR_THROW_IF_FALSE(optimizer.hasParameter("m"));
    THOR_THROW_IF_FALSE(optimizer.hasParameter("v"));
    THOR_THROW_IF_FALSE(optimizer.getParameter("m")->getStorage().has_value());
    THOR_THROW_IF_FALSE(optimizer.getParameter("v")->getStorage().has_value());
}

struct AdamWUpdateExpressions {
    Expression weights;
    Expression m;
    Expression v;
};

AdamWUpdateExpressions adamWWeightUpdate(const Expression &w,
                                         const Expression &g,
                                         const Expression &m,
                                         const Expression &v,
                                         const Expression &alphaT,
                                         const Expression &alphaWeightDecay,
                                         float beta1,
                                         float beta2,
                                         float epsilon,
                                         DataType weightsDType) {
    Expression beta1Expr = Expression::constantScalar(beta1);
    Expression beta2Expr = Expression::constantScalar(beta2);
    Expression oneMinusBeta1Expr = Expression::constantScalar(1.0f - beta1);
    Expression oneMinusBeta2Expr = Expression::constantScalar(1.0f - beta2);
    Expression epsilonExpr = Expression::constantScalar(epsilon);

    Expression mNext = beta1Expr * m + oneMinusBeta1Expr * g;
    Expression vNext = beta2Expr * v + oneMinusBeta2Expr * g * g;

    // Decoupled AdamW weight decay:
    //   w <- w - alpha * weight_decay * w - alpha_t * m_next / (sqrt(v_next) + epsilon)
    // The weight decay term intentionally uses the raw learning rate alpha, while alpha_t is
    // the bias-corrected Adam step size.
    Expression wNext = (w - alphaWeightDecay * w - alphaT * mNext / (Expression::sqrt(vNext) + epsilonExpr)).withOutputDType(weightsDType);

    return AdamWUpdateExpressions{wNext, mNext, vNext};
}

}  // namespace

AdamW::AdamW(uint64_t id, float alpha, float beta1, float beta2, float epsilon, float weightDecay)
    : Optimizer(id), alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon), weightDecay(weightDecay), t(0) {}

void AdamW::compile(const Tensor &weights, Stream &gradientUpdateStream, bool materializeDenseGradient) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    this->gradientUpdateStream = gradientUpdateStream;
    this->weights = weights;

    const DataType weightsDType = weights.getDescriptor().getDataType();
    const int32_t gpuNum = weights.getPlacement().getDeviceNum();

    ensureAdamWState(*this, weights, gradientUpdateStream);
    shared_ptr<PhysicalParameter> mParameter = getParameter("m");
    shared_ptr<PhysicalParameter> vParameter = getParameter("v");

    if (!materializeDenseGradient) {
        // CustomLayer dense optimizer fusion provides the gradient as an expression and updates the
        // parameter/state tensors directly.  In that production path the dense gradient tensor and
        // standalone optimizer update stamp are intentionally not allocated.
        compiled = true;
        return;
    }

    this->weightsGradient = weights.clone();

    auto alphaT = Expression::runtimeScalar("alphaT", DataType::FP32, DataType::FP32);
    auto alphaWeightDecay = Expression::runtimeScalar("alphaWeightDecay", DataType::FP32, DataType::FP32);
    auto invBatchLossScale = Expression::runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32) * invBatchLossScale;
    auto m = Expression::input("m_in", DataType::FP32, DataType::FP32);
    auto v = Expression::input("v_in", DataType::FP32, DataType::FP32);

    unordered_map<string, Tensor> stampInputs;
    unordered_map<string, Tensor> stampOutputs;
    stampInputs["weights_in"] = weights;
    THOR_THROW_IF_FALSE(weightsGradient.has_value());
    stampInputs["gradient"] = weightsGradient.value();
    stampOutputs["weights"] = weights;

    stampInputs["m_in"] = mParameter->getStorage().value();
    stampInputs["v_in"] = vParameter->getStorage().value();
    stampOutputs["m"] = mParameter->getStorage().value();
    stampOutputs["v"] = vParameter->getStorage().value();

    AdamWUpdateExpressions updateExpressions =
        adamWWeightUpdate(w, g, m, v, alphaT, alphaWeightDecay, beta1, beta2, epsilon, weightsDType);

    auto outs = Expression::outputs({
        {"weights", updateExpressions.weights},
        {"m", updateExpressions.m},
        {"v", updateExpressions.v},
    });

    FusedEquation adamWUpdateEquation = FusedEquation::compile(outs.physicalOutputs(), gpuNum);
    updateEquationStamped =
        make_unique<StampedExecutionPlan>(adamWUpdateEquation.stamp(stampInputs, gradientUpdateStream, {}, stampOutputs));

    compiled = true;
}

DenseOptimizerExpression AdamW::toDenseUpdateExpression(const Tensor &weights,
                                                        const Expression &gradient,
                                                        const std::string &namePrefix) {
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());

    const DataType weightsDType = weights.getDescriptor().getDataType();

    ensureAdamWState(*this, weights, gradientUpdateStream);
    shared_ptr<PhysicalParameter> mParameter = getParameter("m");
    shared_ptr<PhysicalParameter> vParameter = getParameter("v");

    auto alphaT = Expression::runtimeScalar(namePrefix + "alphaT", DataType::FP32, DataType::FP32);
    auto alphaWeightDecay = Expression::runtimeScalar(namePrefix + "alphaWeightDecay", DataType::FP32, DataType::FP32);
    auto invBatchLossScale = Expression::runtimeScalar(namePrefix + "invBatchLossScale", DataType::FP32, DataType::FP32);

    auto w = Expression::input(namePrefix + "weights_in", DataType::FP32, DataType::FP32);
    auto g = gradient * invBatchLossScale;
    auto m = Expression::input(namePrefix + "m_in", DataType::FP32, DataType::FP32);
    auto v = Expression::input(namePrefix + "v_in", DataType::FP32, DataType::FP32);

    AdamWUpdateExpressions updateExpressions =
        adamWWeightUpdate(w, g, m, v, alphaT, alphaWeightDecay, beta1, beta2, epsilon, weightsDType);

    DenseOptimizerExpression result;
    result.inputs[namePrefix + "weights_in"] = weights;
    result.inputs[namePrefix + "m_in"] = mParameter->getStorage().value();
    result.inputs[namePrefix + "v_in"] = vParameter->getStorage().value();
    result.preallocatedOutputs["weights"] = weights;
    result.preallocatedOutputs["m"] = mParameter->getStorage().value();
    result.preallocatedOutputs["v"] = vParameter->getStorage().value();

    auto outs = Expression::outputs({
        {"weights", updateExpressions.weights},
        {"m", updateExpressions.m},
        {"v", updateExpressions.v},
    });
    result.outputs = outs.physicalOutputs();
    return result;
}

std::unordered_map<std::string, float> AdamW::denseUpdateRuntimeScalars(uint32_t batchSize, const std::string &namePrefix) {
    std::unordered_map<std::string, float> scalars = sparseRowUpdateRuntimeScalars(batchSize);
    std::unordered_map<std::string, float> prefixed;
    prefixed.reserve(scalars.size());
    for (const auto &[name, value] : scalars) {
        prefixed[namePrefix + name] = value;
    }
    return prefixed;
}

SparseRowOptimizerExpression AdamW::toSparseRowUpdateExpression(const Tensor &weights, SparseRowGradient &sparseRowGradient) {
    THOR_THROW_IF_FALSE(weights.isInitialized());
    sparseRowGradient.validate();
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());

    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Sparse AdamW weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    if (sparseRowGradient.embeddingDim != weightDims[1]) {
        throw std::invalid_argument("Sparse AdamW sparse-gradient embedding dimension does not match weights.");
    }
    if (sparseRowGradient.vocabularySize != weightDims[0]) {
        throw std::invalid_argument("Sparse AdamW sparse-gradient vocabulary size does not match weights.");
    }

    const DataType weightsDType = weights.getDescriptor().getDataType();

    ensureAdamWState(*this, weights, gradientUpdateStream);
    shared_ptr<PhysicalParameter> mParameter = getParameter("m");
    shared_ptr<PhysicalParameter> vParameter = getParameter("v");

    auto alphaT = Expression::runtimeScalar("alphaT", DataType::FP32, DataType::FP32);
    auto alphaWeightDecay = Expression::runtimeScalar("alphaWeightDecay", DataType::FP32, DataType::FP32);
    auto invBatchLossScale = Expression::runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32) * invBatchLossScale;
    auto m = Expression::input("m_in", DataType::FP32, DataType::FP32);
    auto v = Expression::input("v_in", DataType::FP32, DataType::FP32);

    AdamWUpdateExpressions updateExpressions =
        adamWWeightUpdate(w, g, m, v, alphaT, alphaWeightDecay, beta1, beta2, epsilon, weightsDType);

    SparseRowOptimizerExpression result;
    result.inputs["weights_in"] = SparseRowUpdateTensorBinding{weights, SparseRowUpdateTensorKind::IndexedRows};
    result.inputs["gradient"] = SparseRowUpdateTensorBinding{sparseRowGradient.values, SparseRowUpdateTensorKind::DenseLogicalRows};
    result.inputs["m_in"] = SparseRowUpdateTensorBinding{mParameter->getStorage().value(), SparseRowUpdateTensorKind::IndexedRows};
    result.inputs["v_in"] = SparseRowUpdateTensorBinding{vParameter->getStorage().value(), SparseRowUpdateTensorKind::IndexedRows};

    result.indexedOutputs["weights"] = weights;
    result.indexedOutputs["m"] = mParameter->getStorage().value();
    result.indexedOutputs["v"] = vParameter->getStorage().value();

    auto outs = Expression::outputs({
        {"weights", updateExpressions.weights},
        {"m", updateExpressions.m},
        {"v", updateExpressions.v},
    });
    result.outputs = outs.physicalOutputs();
    return result;
}

SparseRowGradient AdamW::compileSparseRows(const Tensor &weights, uint64_t maxSparseRows, Stream &gradientUpdateStream) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Sparse AdamW weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    if (maxSparseRows == 0) {
        throw std::invalid_argument("Sparse AdamW maxSparseRows must be non-zero.");
    }
    if (maxSparseRows > weightDims[0]) {
        throw std::invalid_argument("Sparse AdamW maxSparseRows cannot exceed the embedding vocabulary size.");
    }

    this->gradientUpdateStream = gradientUpdateStream;
    this->weights = weights;
    this->sparseRowGradient =
        SparseRowGradient::allocate(weights.getPlacement(),
                                    maxSparseRows,
                                    weightDims[0],
                                    weightDims[1],
                                    DataType::FP32,
                                    SparseRowGradient::chooseRowDataType(weightDims[0]));
    THOR_THROW_IF_FALSE(sparseRowGradient.has_value());

    const int32_t gpuNum = weights.getPlacement().getDeviceNum();
    SparseRowOptimizerExpression updateExpression = toSparseRowUpdateExpression(weights, sparseRowGradient.value());

    sparseUpdatePlan = SparseRowUpdatePlan::compile(updateExpression.outputs,
                                                   sparseRowGradient->rows,
                                                   sparseRowGradient->numRows,
                                                   updateExpression.inputs,
                                                   updateExpression.indexedOutputs,
                                                   gpuNum);

    compiled = true;
    return sparseRowGradient.value();
}

std::unordered_map<std::string, float> AdamW::sparseRowUpdateRuntimeScalars(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(batchSize > 0);
    const float lossScalingFactor = Loss::getLossScalingFactor();
    THOR_THROW_IF_FALSE(lossScalingFactor > 0);

    t += 1;
    const double alphaT64 = static_cast<double>(alpha) * std::sqrt(1.0 - std::pow(static_cast<double>(beta2), t)) /
                            (1.0 - std::pow(static_cast<double>(beta1), t));
    const float alphaT = static_cast<float>(alphaT64);
    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);
    const float alphaWeightDecay = alpha * weightDecay;
    return {{"alphaT", alphaT}, {"alphaWeightDecay", alphaWeightDecay}, {"invBatchLossScale", invBatchLossScale}};
}

void AdamW::updateSparseRows(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(sparseRowGradient.has_value());
    THOR_THROW_IF_FALSE(sparseUpdatePlan != nullptr);

    sparseRowGradient->validate();

    sparseUpdatePlan->run(sparseRowUpdateRuntimeScalars(batchSize), gradientUpdateStream);
}

void AdamW::updateWeights(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(weightsGradient.has_value());
    THOR_THROW_IF_FALSE(weightsGradient.value().isInitialized());
    THOR_THROW_IF_FALSE(weightsGradient.value().getPlacement() == weights.getPlacement());
    THOR_THROW_IF_FALSE(updateEquationStamped != nullptr);

    updateEquationStamped->run(sparseRowUpdateRuntimeScalars(batchSize));
}

float AdamW::getT() const { return t; }
float AdamW::getAlpha() const { return alpha; }
float AdamW::getBeta1() const { return beta1; }
float AdamW::getBeta2() const { return beta2; }
float AdamW::getEpsilon() const { return epsilon; }
float AdamW::getWeightDecay() const { return weightDecay; }

void AdamW::setT(float t) { this->t = t; }
void AdamW::setAlpha(float alpha) { this->alpha = alpha; }
void AdamW::setBeta1(float beta1) { this->beta1 = beta1; }
void AdamW::setBeta2(float beta2) { this->beta2 = beta2; }
void AdamW::setEpsilon(float epsilon) { this->epsilon = epsilon; }
void AdamW::setWeightDecay(float weightDecay) { this->weightDecay = weightDecay; }

unordered_map<std::string, float> AdamW::updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    (void)epoch;
    (void)batch;
    (void)batchesPerEpoch;

    // AdamW updates t when the parameter update runs so dense, sparse, and fused update paths remain aligned.
    unordered_map<string, float> hyperParameters;
    hyperParameters["t"] = t;
    return hyperParameters;
}

unordered_map<std::string, float> AdamW::getAllHyperParameters() {
    unordered_map<string, float> hyperParameters;
    hyperParameters["t"] = t;
    hyperParameters["alpha"] = alpha;
    hyperParameters["beta1"] = beta1;
    hyperParameters["beta2"] = beta2;
    hyperParameters["epsilon"] = epsilon;
    hyperParameters["weightDecay"] = weightDecay;
    return hyperParameters;
}

}  // namespace ThorImplementation
