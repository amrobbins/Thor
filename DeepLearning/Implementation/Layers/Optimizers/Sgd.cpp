#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include <optional>

#include "DeepLearning/Implementation/Initializers/ZerosInitializer.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"

#include "DeepLearning/Implementation/ThorError.h"
using namespace ThorImplementation;
using namespace std;

Sgd::Sgd(uint64_t id, float initialLearningRate, float decay, float momentum, bool useNesterovMomentum, uint64_t startResumeEpoch)
    : Optimizer(id),
      initialLearningRate(initialLearningRate),
      decay(decay),
      momentum(momentum),
      useNesterovMomentum(useNesterovMomentum),
      currentEpoch(startResumeEpoch),
      currentBatch(0),
      currentLearningRate(initialLearningRate) {}

void Sgd::compile(const Tensor& weights, Stream& gradientUpdateStream) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    this->gradientUpdateStream = gradientUpdateStream;
    this->weights = weights;
    this->weightsGradient = weights.clone();

    const DataType weightsDType = weights.getDescriptor().getDataType();
    const int32_t gpuNum = weights.getPlacement().getDeviceNum();

    auto w = Expression::input("weights_in", DataType::FP32, DataType::FP32);
    auto g = Expression::input("gradient", DataType::FP32, DataType::FP32);
    auto step = Expression::runtimeScalar("step", DataType::FP32, DataType::FP32);

    unordered_map<string, Tensor> stampInputs;
    unordered_map<string, TensorScalarBinding> tensorScalarInputs;
    unordered_map<string, Tensor> preallocatedOutputs;

    stampInputs["weights_in"] = weights;
    THOR_THROW_IF_FALSE(weightsGradient.has_value());
    stampInputs["gradient"] = weightsGradient.value();
    preallocatedOutputs["weights"] = weights;

    std::optional<Outputs> expressionOutputs;
    if (momentum > 0.0f) {
        if (!hasParameter("momentum")) {
            shared_ptr<PhysicalParameter> momentumParameter =
                make_shared<PhysicalParameter>("momentum", false, weights.getDimensions(), weightsDType);
            shared_ptr<Initializer> paramInitializer = make_shared<ZerosInitializer>();
            momentumParameter->setInitializer(paramInitializer->clone());
            addParameter(momentumParameter);
            momentumParameter->compileStorage(weights);
            momentumParameter->compileInitializer();
            momentumParameter->initialize(gradientUpdateStream);
            THOR_THROW_IF_FALSE(momentumParameter->getStorage().has_value());
        }
        shared_ptr<PhysicalParameter> momentumParameter = getParameter("momentum");
        Tensor momentumTensor = momentumParameter->getStorage().value();

        std::optional<Expression> vNext;
        std::optional<Expression> wNext;
        if (useNesterovMomentum) {
            // Nesterov:
            // v_{t+1} = mu * v_t - step * g
            // w_{t+1} = w_t + mu * v_{t+1} - step * g
            Expression v = Expression::input("velocity_in", DataType::FP32, DataType::FP32);
            vNext.emplace(Expression::constantScalar(momentum) * v - step * g);
            wNext.emplace(w + Expression::constantScalar(momentum) * (*vNext) - step * g);
        } else {
            // Classical momentum:
            // v_{t+1} = mu * v_t - step * g
            // w_{t+1} = w_t + v_{t+1}
            Expression v = Expression::input("velocity_in", DataType::FP32, DataType::FP32);
            vNext.emplace(Expression::constantScalar(momentum) * v - step * g);
            wNext.emplace(w + (*vNext));
        }

        expressionOutputs.emplace(Expression::outputs({
            {"weights", *wNext},
            {"velocity", *vNext},
        }));

        stampInputs["velocity_in"] = momentumTensor;
        preallocatedOutputs["velocity"] = momentumTensor;
    } else {
        if (hasParameter("momentum"))
            dropParameter("momentum");

        // Plain SGD:
        // w_{t+1} = w_t - step * g
        auto wNext = (w - step * g).withOutputDType(weightsDType);
        expressionOutputs.emplace(Expression::outputs({
            {"weights", wNext},
        }));
    }

    FusedEquation sgdUpdateEquation = FusedEquation::compile(expressionOutputs->physicalOutputs(), gpuNum);
    updateEquationStamped = make_unique<StampedExecutionPlan>(
        sgdUpdateEquation.stamp(stampInputs, gradientUpdateStream, tensorScalarInputs, preallocatedOutputs));

    compiled = true;
}

SparseRowGradient Sgd::compileSparseRows(const Tensor& weights, uint64_t maxSparseRows, Stream& gradientUpdateStream) {
    THOR_THROW_IF_FALSE(!compiled);
    THOR_THROW_IF_FALSE(gradientUpdateStream.isInitialized());
    THOR_THROW_IF_FALSE(weights.isInitialized());
    THOR_THROW_IF_FALSE(weights.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Sparse SGD weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    if (maxSparseRows == 0) {
        throw std::invalid_argument("Sparse SGD maxSparseRows must be non-zero.");
    }
    if (maxSparseRows > weightDims[0]) {
        throw std::invalid_argument("Sparse SGD maxSparseRows cannot exceed the embedding vocabulary size.");
    }

    this->gradientUpdateStream = gradientUpdateStream;
    this->weights = weights;
    this->sparseRowGradient =
        SparseRowGradient::allocate(weights.getPlacement(), maxSparseRows, weightDims[0], weightDims[1], DataType::FP32, DataType::UINT64);

    // Step 1 only installs the optimizer-owned reduced sparse-gradient sink. The sparse expression update plan is the
    // next slice, so sparse update calls fail explicitly rather than silently doing nothing.
    compiled = true;
    return sparseRowGradient.value();
}

void Sgd::updateWeights(uint32_t batchSize) {
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(weightsGradient.has_value());
    THOR_THROW_IF_FALSE(weightsGradient.value().isInitialized());
    THOR_THROW_IF_FALSE(weightsGradient.value().getPlacement() == weights.getPlacement());
    THOR_THROW_IF_FALSE(updateEquationStamped != nullptr);

    THOR_THROW_IF_FALSE(batchSize > 0);
    const float lossScalingFactor = Loss::getLossScalingFactor();
    THOR_THROW_IF_FALSE(lossScalingFactor > 0);

    const float step = currentLearningRate / (static_cast<float>(batchSize) * lossScalingFactor);
    updateEquationStamped->run({
        {"step", step},
    });
}

void Sgd::updateSparseRows(uint32_t batchSize) {
    (void)batchSize;
    THOR_THROW_IF_FALSE(compiled);
    THOR_THROW_IF_FALSE(sparseRowGradient.has_value());
    throw std::runtime_error(
        "Sparse SGD update has an optimizer-owned reduced sparse-gradient sink, but the sparse row expression update stage is not "
        "implemented yet.");
}

unordered_map<string, float> Sgd::updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    unordered_map<string, float> updatedParameters;

    if (currentEpoch != epoch)
        currentLearningRate = static_cast<float>(initialLearningRate * pow(1.0 - static_cast<double>(decay), static_cast<double>(epoch)));

    currentEpoch = epoch;

    updatedParameters["currentLearningRate"] = currentLearningRate;

    return updatedParameters;
}

unordered_map<string, float> Sgd::getAllHyperParameters() {
    unordered_map<string, float> parameters;
    parameters["currentLearningRate"] = currentLearningRate;
    parameters["initialLearningRate"] = initialLearningRate;
    parameters["decay"] = decay;
    parameters["momentum"] = momentum;
    parameters["useNesterovMomentum"] = useNesterovMomentum ? 1.0f : 0.0f;

    return parameters;
}

void Sgd::setInitialLearningRate(float initialLearningRate) { this->initialLearningRate = initialLearningRate; }

void Sgd::setDecay(float decay) { this->decay = decay; }

void Sgd::setMomentum(float momentum) {
    THOR_THROW_IF_FALSE(momentum >= 0.0f);
    this->momentum = momentum;
}

void Sgd::setUseNesterovMomentum(bool useNesterovMomentum) { this->useNesterovMomentum = useNesterovMomentum; }

float Sgd::getInitialLearningRate() const { return initialLearningRate; }

float Sgd::getDecay() const { return decay; }

float Sgd::getMomentum() const { return momentum; }

bool Sgd::getUseNesterovMomentum() const { return useNesterovMomentum; }

uint64_t Sgd::getEpoch() const { return currentEpoch; }

float Sgd::getCurrentLearningRate() const { return currentLearningRate; }
