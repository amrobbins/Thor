#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"

using namespace ThorImplementation;
using namespace std;

Adam::Adam(std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer,
           float alpha,
           float beta1,
           float beta2,
           float epsilon,
           Optional<Tensor> errorInput,
           Optional<Tensor> errorOutput) {
    if (errorInput.isEmpty())
        return;

    this->alpha = alpha;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
    this->t = 0.0f;

    this->trainableLayerShared = trainableLayer;
    this->trainableLayer = trainableLayer.get();

    TensorPlacement layerPlacement = trainableLayer->getPlacement();
    assert(layerPlacement == TensorPlacement::MemDevices::GPU);
    gpuNum = layerPlacement.getDeviceNum();
    gradientUpdateStream = Stream::getNextGradientUpdateStream(gpuNum);

    Tensor weights = trainableLayer->getWeights();
    weightsGradient = weights.clone();

    Optional<Tensor> biases = trainableLayer->getBiases();
    if (biases.isPresent())
        biasesGradient = biases.get().clone();

    assert(errorInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    if (errorOutput.isPresent()) {
        assert(errorInput.get().getPlacement().getMemDevice() == errorOutput.get().getPlacement().getMemDevice());
        assert(errorInput.get().getPlacement().getDeviceNum() == errorOutput.get().getPlacement().getDeviceNum());
    }
    assert(errorInput.get().getDimensions().size() > 0);
}

void Adam::initialize() {
    assert(weightsGradient.getDataType() == TensorDescriptor::DataType::FP16 ||
           weightsGradient.getDataType() == TensorDescriptor::DataType::FP32);

    // The minimum strictly positive (subnormal) value of fp16 is 2^−24 ≈ 5.96 × 10^−8
    // So the default value of epsilon (which prevents divide by zero) is set to no less than this when epsilon is FP16.
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    if (weightsGradient.getDataType() == TensorDescriptor::DataType::FP16 && epsilon < 5.96046448e-8)
        epsilon = __float2half_ru(5.9604644e-8f);

    weightsUpdate = weightsGradient.clone();
    if (biasesGradient.isPresent())
        biasesUpdate = biasesGradient.get().clone();
    else
        biasesUpdate = Optional<Tensor>::empty();

    TensorPlacement cpuPlacement = TensorPlacement(TensorPlacement::MemDevices::CPU);
    TensorDescriptor scalarDescriptor = TensorDescriptor(weightsGradient.getDataType(), {1});

    m = weightsGradient.clone();
    v = weightsGradient.clone();
    if (biasesGradient.isPresent()) {
        mBias = biasesGradient.get().clone();
        vBias = biasesGradient.get().clone();
    }

    Tensor cpuWeightsVectorBufferTensor = Tensor(cpuPlacement, weightsGradient.getDescriptor());
    Tensor cpuBiasesVectorBufferTensor;
    if (biasesGradient.isPresent())
        cpuBiasesVectorBufferTensor = Tensor(cpuPlacement, biasesGradient.get().getDescriptor());
    if (weightsGradient.getDataType() == TensorDescriptor::DataType::FP16) {
        void *cpuVectorBufferMem = cpuWeightsVectorBufferTensor.getMemPtr();
        memset(cpuVectorBufferMem, 0, cpuWeightsVectorBufferTensor.getTotalNumElements() * sizeof(half));
        m.copyFromAsync(cpuWeightsVectorBufferTensor, gradientUpdateStream);
        v.copyFromAsync(cpuWeightsVectorBufferTensor, gradientUpdateStream);
        if (biasesGradient.isPresent()) {
            cpuVectorBufferMem = cpuBiasesVectorBufferTensor.getMemPtr();
            memset(cpuVectorBufferMem, 0, cpuBiasesVectorBufferTensor.getTotalNumElements() * sizeof(half));
            mBias.get().copyFromAsync(cpuBiasesVectorBufferTensor, gradientUpdateStream);
            vBias.get().copyFromAsync(cpuBiasesVectorBufferTensor, gradientUpdateStream);
        }
    } else {
        void *cpuVectorBufferMem = cpuWeightsVectorBufferTensor.getMemPtr();
        memset(cpuVectorBufferMem, 0, cpuWeightsVectorBufferTensor.getTotalNumElements() * sizeof(float));
        m.copyFromAsync(cpuWeightsVectorBufferTensor, gradientUpdateStream);
        v.copyFromAsync(cpuWeightsVectorBufferTensor, gradientUpdateStream);
        if (biasesGradient.isPresent()) {
            cpuVectorBufferMem = cpuBiasesVectorBufferTensor.getMemPtr();
            memset(cpuVectorBufferMem, 0, cpuBiasesVectorBufferTensor.getTotalNumElements() * sizeof(float));
            mBias.get().copyFromAsync(cpuBiasesVectorBufferTensor, gradientUpdateStream);
            vBias.get().copyFromAsync(cpuBiasesVectorBufferTensor, gradientUpdateStream);
        }
    }

    t = 0.0f;
}

void Adam::computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues) {
    trainableLayer->computeWeightsGradient(weightsGradient, biasesGradient, featureIn, errorIn, gradientUpdateStream, accumulateValues);

    t += 1;
    if (featureIn.get().getDataType() == TensorDescriptor::DataType::FP16) {
        assert(errorIn.get().getDataType() == TensorDescriptor::DataType::FP16);
        launchAdamStep<half>((half *)weightsUpdate.getMemPtr(),
                             (half *)weightsGradient.getMemPtr(),
                             (half *)m.getMemPtr(),
                             (half *)v.getMemPtr(),
                             t,
                             alpha,
                             beta1,
                             beta2,
                             epsilon,
                             weightsGradient.getTotalNumElements(),
                             gradientUpdateStream);
        if (biasesGradient.isPresent()) {
            assert(biasesUpdate.isPresent());
            launchAdamStep<half>((half *)biasesUpdate.get().getMemPtr(),
                                 (half *)biasesGradient.get().getMemPtr(),
                                 (half *)mBias.get().getMemPtr(),
                                 (half *)vBias.get().getMemPtr(),
                                 t,
                                 alpha,
                                 beta1,
                                 beta2,
                                 epsilon,
                                 biasesGradient.get().getTotalNumElements(),
                                 gradientUpdateStream);
        }
    } else if (featureIn.get().getDataType() == TensorDescriptor::DataType::FP32) {
        launchAdamStep<float>((float *)weightsUpdate.getMemPtr(),
                              (float *)weightsGradient.getMemPtr(),
                              (float *)m.getMemPtr(),
                              (float *)v.getMemPtr(),
                              t,
                              alpha,
                              beta1,
                              beta2,
                              epsilon,
                              weightsGradient.getTotalNumElements(),
                              gradientUpdateStream);
        if (biasesGradient.isPresent())
            assert(biasesUpdate.isPresent());
        launchAdamStep<float>((float *)biasesUpdate.get().getMemPtr(),
                              (float *)biasesGradient.get().getMemPtr(),
                              (float *)mBias.get().getMemPtr(),
                              (float *)vBias.get().getMemPtr(),
                              t,
                              alpha,
                              beta1,
                              beta2,
                              epsilon,
                              biasesGradient.get().getTotalNumElements(),
                              gradientUpdateStream);
    } else {
        assert(false);
    }
}

void Adam::updateWeights(Tensor weights, Optional<Tensor> biases, uint32_t batchSize) {
    // Note: the algorithm describes the update for a single example, since this computation is done for
    // each example in a minibatch and summed, the update must be divided by the batch size to get the average update.
    // Note also that this summation is part of the matrix multiply of weightsGradient = featureInput * errorInput in the case of FC layer.
    const float ALPHA = 1.0f / (Loss::getLossScalingFactor() * batchSize);
    const float BETA = 1.0f;

    accumulateScale(weights, weightsUpdate, &ALPHA, &BETA, gradientUpdateStream);
    if (biases.isPresent()) {
        assert(biasesUpdate.isPresent());
        accumulateScale(biases, biasesUpdate, &ALPHA, &BETA, gradientUpdateStream);
    }
}

float Adam::getT() { return t; }
float Adam::getAlpha() { return alpha; }
float Adam::getBeta1() { return beta1; }
float Adam::getBeta2() { return beta2; }
float Adam::getEpsilon() { return epsilon; }

void Adam::setT(float t) { this->t = t; }

void Adam::setAlpha(float alpha) { this->alpha = alpha; }

void Adam::setBeta1(float beta1) { this->beta1 = beta1; }

void Adam::setBeta2(float beta2) { this->beta2 = beta2; }

void Adam::setEpsilon(float epsilon) { this->epsilon = epsilon; }

unordered_map<std::string, float> Adam::updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    // Adam automatically updates its parameters every minibatch
    unordered_map<string, float> hyperParameters;
    hyperParameters["t"] = t;
    return hyperParameters;
}

unordered_map<std::string, float> Adam::getAllHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
    unordered_map<string, float> hyperParameters;
    hyperParameters["t"] = t;
    hyperParameters["alpha"] = alpha;
    hyperParameters["beta1"] = beta1;
    hyperParameters["beta2"] = beta2;
    hyperParameters["epsilon"] = epsilon;
    return hyperParameters;
}
