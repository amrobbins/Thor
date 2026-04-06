#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"

using namespace std;

namespace ThorImplementation {

Adam::Adam(uint64_t id, float alpha, float beta1, float beta2, float epsilon)
    : Optimizer(id), alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

DynamicExpression Adam::buildExpression() {
    return DynamicExpression([this](const DynamicExpression::TensorMap &inputs, Stream &stream) -> StampedExecutionPlan {
        const Tensor &wTensor = inputs.at("w");
        const Tensor &gTensor = inputs.at("g");

        const DataType weightsDType = wTensor.getDescriptor().getDataType();
        const int32_t gpuNum = wTensor.getPlacement().getDeviceNum();

        auto alphaT = Expression::runtimeScalar("alphaT", DataType::FP32, DataType::FP32);
        auto invBatchLossScale = Expression::runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

        auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
        auto g = Expression::input("gradient", DataType::FP32, DataType::FP32) * invBatchLossScale;
        auto m = Expression::input("m_in", DataType::FP32, DataType::FP32);
        auto v = Expression::input("v_in", DataType::FP32, DataType::FP32);

        unordered_map<string, Tensor> stampInputs;
        unordered_map<string, Tensor> stampOutputs;
        stampInputs["weights_in"] = wTensor;
        stampInputs["gradient"] = gTensor;
        stampOutputs["weights"] = wTensor;

        // Allocate Adam state once, same shape/device as weights, moments always fp32.
        if (mBuffer.isEmpty()) {
            mBuffer = wTensor.clone(DataType::FP32);
            mBuffer.get().memsetAsync(stream, 0);
        }
        if (vBuffer.isEmpty()) {
            vBuffer = wTensor.clone(DataType::FP32);
            vBuffer.get().memsetAsync(stream, 0);
        }

        stampInputs["m_in"] = mBuffer;
        stampInputs["v_in"] = vBuffer;
        stampOutputs["m"] = mBuffer;
        stampOutputs["v"] = vBuffer;

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
        return adamUpdateEquation.stamp(stampInputs, stream, {}, stampOutputs);
    });
}

void Adam::compile(const Tensor &weights, Stream &gradientUpdateStream) {
    assert(!compiled);
    assert(gradientUpdateStream.isInitialized());
    assert(weights.isInitialized());
    assert(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    this->gradientUpdateStream = gradientUpdateStream;
    this->weights = weights;
    this->weightsGradient = weights.clone();

    // DynamicExpression::TensorMap inputTensors;
    // inputTensors["w"] = weights;
    // inputTensors["g"] = weightsGradient;
    // DynamicExpression weightsUpdateExpression = buildExpression();
    // updateEquationStamped = make_unique<StampedExecutionPlan>(weightsUpdateExpression.stamp(inputTensors, gradientUpdateStream));

    const DataType weightsDType = weights.getDescriptor().getDataType();
    const int32_t gpuNum = weights.getPlacement().getDeviceNum();

    auto alphaT = Expression::runtimeScalar("alphaT", DataType::FP32, DataType::FP32);
    auto invBatchLossScale = Expression::runtimeScalar("invBatchLossScale", DataType::FP32, DataType::FP32);

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32) * invBatchLossScale;
    auto m = Expression::input("m_in", DataType::FP32, DataType::FP32);
    auto v = Expression::input("v_in", DataType::FP32, DataType::FP32);

    unordered_map<string, Tensor> stampInputs;
    unordered_map<string, Tensor> stampOutputs;
    stampInputs["weights_in"] = weights;
    stampInputs["gradient"] = weightsGradient;
    stampOutputs["weights"] = weights;

    // Allocate Adam state once, same shape/device as weights, moments always fp32.
    if (mBuffer.isEmpty()) {
        mBuffer = weights.clone(DataType::FP32);
        mBuffer.get().memsetAsync(gradientUpdateStream, 0);
    }
    if (vBuffer.isEmpty()) {
        vBuffer = weights.clone(DataType::FP32);
        vBuffer.get().memsetAsync(gradientUpdateStream, 0);
    }

    stampInputs["m_in"] = mBuffer;
    stampInputs["v_in"] = vBuffer;
    stampOutputs["m"] = mBuffer;
    stampOutputs["v"] = vBuffer;

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

void Adam::updateWeights(uint32_t batchSize) {
    assert(compiled);
    assert(weightsGradient.isPresent());
    assert(weightsGradient.get().isInitialized());
    assert(weightsGradient.get().getPlacement() == weights.getPlacement());
    assert(updateEquationStamped != nullptr);

    assert(batchSize > 0);
    const float lossScalingFactor = Loss::getLossScalingFactor();
    assert(lossScalingFactor > 0);

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
