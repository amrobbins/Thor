#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"

#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Initializers/ZerosInitializer.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "DeepLearning/Implementation/ThorError.h"
using namespace std;

namespace ThorImplementation {

Adam::Adam(uint64_t id, float alpha, float beta1, float beta2, float epsilon)
    : Optimizer(id), alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

void Adam::compile(const Tensor &weights, Stream &gradientUpdateStream, bool materializeDenseGradient) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    this->gradientUpdateStream = gradientUpdateStream;
    this->weights = weights;

    const DataType weightsDType = weights.getDescriptor().getDataType();
    const int32_t gpuNum = weights.getPlacement().getDeviceNum();

    if (!hasParameter("m")) {
        THOR_THROW_IF_FALSE(!hasParameter("v"));
        shared_ptr<PhysicalParameter> mParameter = make_shared<PhysicalParameter>("m", false, weights.getDimensions(), DataType::FP32);
        shared_ptr<PhysicalParameter> vParameter = make_shared<PhysicalParameter>("v", false, weights.getDimensions(), DataType::FP32);
        shared_ptr<Initializer> paramInitializer = make_shared<ZerosInitializer>();

        mParameter->setInitializer(paramInitializer->clone());
        addParameter(mParameter);
        mParameter->compileStorage(weights);
        mParameter->compileInitializer();

        vParameter->setInitializer(paramInitializer->clone());
        addParameter(vParameter);
        vParameter->compileStorage(weights);
        vParameter->compileInitializer();

        THOR_THROW_IF_FALSE(mParameter->getStorage().has_value());
        THOR_THROW_IF_FALSE(vParameter->getStorage().has_value());
        mParameter->initialize(gradientUpdateStream);
        vParameter->initialize(gradientUpdateStream);
    }
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

    // Regular Adam:
    // m_{t+1} = beta1 * m_t + (1 - beta1) * g_t
    // v_{t+1} = beta2 * v_t + (1 - beta2) * g_t^2
    // w_{t+1} = w_t - alphaT * m_{t+1} / (sqrt(v_{t+1}) + epsilon)
    //
    // alphaT is the bias-corrected learning rate computed on CPU and passed
    // in as a runtime scalar.
    Expression beta1Expr = Expression::constantScalar(beta1);
    Expression beta2Expr = Expression::constantScalar(beta2);
    Expression oneMinusBeta1Expr = Expression::constantScalar(1.0 - beta1);
    Expression oneMinusBeta2Expr = Expression::constantScalar(1.0 - beta2);
    Expression epsilonExpr = Expression::constantScalar(epsilon);

    Expression mNext = beta1Expr * m + oneMinusBeta1Expr * g;
    Expression vNext = beta2Expr * v + oneMinusBeta2Expr * g * g;
    Expression wNext = (w - alphaT * mNext / (Expression::sqrt(vNext) + epsilonExpr)).withOutputDType(weightsDType);

    auto outs = Expression::outputs({
        {"weights", wNext},
        {"m", mNext},
        {"v", vNext},
    });

    FusedEquation adamUpdateEquation = FusedEquation::compile(outs.physicalOutputs(), gpuNum);
    updateEquationStamped =
        make_unique<StampedExecutionPlan>(adamUpdateEquation.stamp(stampInputs, gradientUpdateStream, {}, stampOutputs));

    compiled = true;
}


DenseOptimizerExpression Adam::toDenseUpdateExpression(const Tensor& weights,
                                                       const Expression& gradient,
                                                       const std::string& namePrefix) {
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());

    const DataType weightsDType = weights.getDescriptor().getDataType();

    if (!hasParameter("m")) {
        THOR_THROW_IF_FALSE(!hasParameter("v"));
        shared_ptr<PhysicalParameter> mParameter = make_shared<PhysicalParameter>("m", false, weights.getDimensions(), DataType::FP32);
        shared_ptr<PhysicalParameter> vParameter = make_shared<PhysicalParameter>("v", false, weights.getDimensions(), DataType::FP32);
        shared_ptr<Initializer> paramInitializer = make_shared<ZerosInitializer>();

        mParameter->setInitializer(paramInitializer->clone());
        addParameter(mParameter);
        mParameter->compileStorage(weights);
        mParameter->compileInitializer();

        vParameter->setInitializer(paramInitializer->clone());
        addParameter(vParameter);
        vParameter->compileStorage(weights);
        vParameter->compileInitializer();

        THOR_THROW_IF_FALSE(mParameter->getStorage().has_value());
        THOR_THROW_IF_FALSE(vParameter->getStorage().has_value());
        mParameter->initialize(gradientUpdateStream);
        vParameter->initialize(gradientUpdateStream);
    }
    shared_ptr<PhysicalParameter> mParameter = getParameter("m");
    shared_ptr<PhysicalParameter> vParameter = getParameter("v");
    THOR_THROW_IF_FALSE(mParameter->getStorage().has_value());
    THOR_THROW_IF_FALSE(vParameter->getStorage().has_value());

    auto alphaT = Expression::runtimeScalar(namePrefix + "alphaT", DataType::FP32, DataType::FP32);
    auto invBatchLossScale = Expression::runtimeScalar(namePrefix + "invBatchLossScale", DataType::FP32, DataType::FP32);

    auto w = Expression::input(namePrefix + "weights_in", DataType::FP32, DataType::FP32);
    auto g = gradient * invBatchLossScale;
    auto m = Expression::input(namePrefix + "m_in", DataType::FP32, DataType::FP32);
    auto v = Expression::input(namePrefix + "v_in", DataType::FP32, DataType::FP32);

    Expression beta1Expr = Expression::constantScalar(beta1);
    Expression beta2Expr = Expression::constantScalar(beta2);
    Expression oneMinusBeta1Expr = Expression::constantScalar(1.0 - beta1);
    Expression oneMinusBeta2Expr = Expression::constantScalar(1.0 - beta2);
    Expression epsilonExpr = Expression::constantScalar(epsilon);

    Expression mNext = beta1Expr * m + oneMinusBeta1Expr * g;
    Expression vNext = beta2Expr * v + oneMinusBeta2Expr * g * g;
    Expression wNext = (w - alphaT * mNext / (Expression::sqrt(vNext) + epsilonExpr)).withOutputDType(weightsDType);

    DenseOptimizerExpression result;
    result.inputs[namePrefix + "weights_in"] = weights;
    result.inputs[namePrefix + "m_in"] = mParameter->getStorage().value();
    result.inputs[namePrefix + "v_in"] = vParameter->getStorage().value();
    result.preallocatedOutputs["weights"] = weights;
    result.preallocatedOutputs["m"] = mParameter->getStorage().value();
    result.preallocatedOutputs["v"] = vParameter->getStorage().value();

    auto outs = Expression::outputs({
        {"weights", wNext},
        {"m", mNext},
        {"v", vNext},
    });
    result.outputs = outs.physicalOutputs();
    return result;
}

std::unordered_map<std::string, float> Adam::denseUpdateRuntimeScalars(uint32_t batchSize, const std::string& namePrefix) {
    std::unordered_map<std::string, float> scalars = sparseRowUpdateRuntimeScalars(batchSize);
    std::unordered_map<std::string, float> prefixed;
    prefixed.reserve(scalars.size());
    for (const auto& [name, value] : scalars) {
        prefixed[namePrefix + name] = value;
    }
    return prefixed;
}

SparseRowOptimizerExpression Adam::toSparseRowUpdateExpression(const Tensor &weights, SparseRowGradient &sparseRowGradient) {
    THOR_THROW_IF_FALSE(weights.isInitialized());
    sparseRowGradient.validate();
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());

    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Sparse Adam weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    if (sparseRowGradient.embeddingDim != weightDims[1]) {
        throw std::invalid_argument("Sparse Adam sparse-gradient embedding dimension does not match weights.");
    }
    if (sparseRowGradient.vocabularySize != weightDims[0]) {
        throw std::invalid_argument("Sparse Adam sparse-gradient vocabulary size does not match weights.");
    }

    const DataType weightsDType = weights.getDescriptor().getDataType();

    if (!hasParameter("m")) {
        THOR_THROW_IF_FALSE(!hasParameter("v"));
        shared_ptr<PhysicalParameter> mParameter = make_shared<PhysicalParameter>("m", false, weights.getDimensions(), DataType::FP32);
        shared_ptr<PhysicalParameter> vParameter = make_shared<PhysicalParameter>("v", false, weights.getDimensions(), DataType::FP32);
        shared_ptr<Initializer> paramInitializer = make_shared<ZerosInitializer>();

        mParameter->setInitializer(paramInitializer->clone());
        addParameter(mParameter);
        mParameter->compileStorage(weights);
        mParameter->compileInitializer();

        vParameter->setInitializer(paramInitializer->clone());
        addParameter(vParameter);
        vParameter->compileStorage(weights);
        vParameter->compileInitializer();

        THOR_THROW_IF_FALSE(mParameter->getStorage().has_value());
        THOR_THROW_IF_FALSE(vParameter->getStorage().has_value());
        mParameter->initialize(gradientUpdateStream);
        vParameter->initialize(gradientUpdateStream);
    }
    shared_ptr<PhysicalParameter> mParameter = getParameter("m");
    shared_ptr<PhysicalParameter> vParameter = getParameter("v");
    THOR_THROW_IF_FALSE(mParameter->getStorage().has_value());
    THOR_THROW_IF_FALSE(vParameter->getStorage().has_value());

    auto alphaT = Expression::runtimeScalar("alphaT", DataType::FP32, DataType::FP32);
    auto invBatchLossScale = Expression::runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32) * invBatchLossScale;
    auto m = Expression::input("m_in", DataType::FP32, DataType::FP32);
    auto v = Expression::input("v_in", DataType::FP32, DataType::FP32);

    Expression beta1Expr = Expression::constantScalar(beta1);
    Expression beta2Expr = Expression::constantScalar(beta2);
    Expression oneMinusBeta1Expr = Expression::constantScalar(1.0 - beta1);
    Expression oneMinusBeta2Expr = Expression::constantScalar(1.0 - beta2);
    Expression epsilonExpr = Expression::constantScalar(epsilon);

    Expression mNext = beta1Expr * m + oneMinusBeta1Expr * g;
    Expression vNext = beta2Expr * v + oneMinusBeta2Expr * g * g;
    Expression wNext = (w - alphaT * mNext / (Expression::sqrt(vNext) + epsilonExpr)).withOutputDType(weightsDType);

    SparseRowOptimizerExpression result;
    result.inputs["weights_in"] = SparseRowUpdateTensorBinding{weights, SparseRowUpdateTensorKind::IndexedRows};
    result.inputs["gradient"] = SparseRowUpdateTensorBinding{sparseRowGradient.values, SparseRowUpdateTensorKind::DenseLogicalRows};
    result.inputs["m_in"] = SparseRowUpdateTensorBinding{mParameter->getStorage().value(), SparseRowUpdateTensorKind::IndexedRows};
    result.inputs["v_in"] = SparseRowUpdateTensorBinding{vParameter->getStorage().value(), SparseRowUpdateTensorKind::IndexedRows};

    result.indexedOutputs["weights"] = weights;
    result.indexedOutputs["m"] = mParameter->getStorage().value();
    result.indexedOutputs["v"] = vParameter->getStorage().value();

    auto outs = Expression::outputs({
        {"weights", wNext},
        {"m", mNext},
        {"v", vNext},
    });
    result.outputs = outs.physicalOutputs();
    return result;
}

SparseRowGradient Adam::compileSparseRows(const Tensor &weights, uint64_t maxSparseRows, Stream &gradientUpdateStream) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Sparse Adam weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    if (maxSparseRows == 0) {
        throw std::invalid_argument("Sparse Adam maxSparseRows must be non-zero.");
    }
    if (maxSparseRows > weightDims[0]) {
        throw std::invalid_argument("Sparse Adam maxSparseRows cannot exceed the embedding vocabulary size.");
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

std::unordered_map<std::string, float> Adam::sparseRowUpdateRuntimeScalars(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(batchSize > 0);
    const float lossScalingFactor = Loss::getLossScalingFactor();
    THOR_THROW_IF_FALSE(lossScalingFactor > 0);

    t += 1;
    const double alphaT64 = static_cast<double>(alpha) * std::sqrt(1.0 - std::pow(static_cast<double>(beta2), t)) /
                            (1.0 - std::pow(static_cast<double>(beta1), t));
    const float alphaT = static_cast<float>(alphaT64);
    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);
    return {{"alphaT", alphaT}, {"invBatchLossScale", invBatchLossScale}};
}

void Adam::updateSparseRows(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(sparseRowGradient.has_value());
    THOR_THROW_IF_FALSE(sparseUpdatePlan != nullptr);

    sparseRowGradient->validate();

    sparseUpdatePlan->run(sparseRowUpdateRuntimeScalars(batchSize), gradientUpdateStream);
}

void Adam::updateWeights(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(weightsGradient.has_value());
    THOR_THROW_IF_FALSE(weightsGradient.value().isInitialized());
    THOR_THROW_IF_FALSE(weightsGradient.value().getPlacement() == weights.getPlacement());
    THOR_THROW_IF_FALSE(updateEquationStamped != nullptr);

    THOR_THROW_IF_FALSE(batchSize > 0);
    const float lossScalingFactor = Loss::getLossScalingFactor();
    THOR_THROW_IF_FALSE(lossScalingFactor > 0);

    t += 1;
    double alphaT64 = static_cast<double>(alpha) * std::sqrt(1.0 - std::pow(static_cast<double>(beta2), t)) /
                      (1.0 - std::pow(static_cast<double>(beta1), t));
    auto alphaT = static_cast<float>(alphaT64);
    float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

    updateEquationStamped->run({
        {"alphaT", alphaT},
        {"invBatchLossScale", invBatchLossScale},
    });
}

float Adam::getT() const { return t; }
float Adam::getAlpha() const { return alpha; }
float Adam::getBeta1() const { return beta1; }
float Adam::getBeta2() const { return beta2; }
float Adam::getEpsilon() const { return epsilon; }

void Adam::setT(float t) { this->t = t; }
void Adam::setAlpha(float alpha) { this->alpha = alpha; }
void Adam::setBeta1(float beta1) { this->beta1 = beta1; }
void Adam::setBeta2(float beta2) { this->beta2 = beta2; }
void Adam::setEpsilon(float epsilon) { this->epsilon = epsilon; }

unordered_map<std::string, float> Adam::updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    // Adam automatically updates its parameters every mini-batch
    // FIXME: That will not work for multiple stamps, together which form a minibatch
    unordered_map<string, float> hyperParameters;
    hyperParameters["t"] = t;
    return hyperParameters;
}

unordered_map<std::string, float> Adam::getAllHyperParameters() {
    unordered_map<string, float> hyperParameters;
    hyperParameters["t"] = t;
    hyperParameters["alpha"] = alpha;
    hyperParameters["beta1"] = beta1;
    hyperParameters["beta2"] = beta2;
    hyperParameters["epsilon"] = epsilon;
    return hyperParameters;
}

}  // namespace ThorImplementation
