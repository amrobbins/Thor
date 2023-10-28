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

    m = weightsGradient.clone();
    v = weightsGradient.clone();
    if (biasesGradient.isPresent()) {
        mBias = biasesGradient.get().clone();
        vBias = biasesGradient.get().clone();
    }

    m.memsetAsync(gradientUpdateStream, 0);
    v.memsetAsync(gradientUpdateStream, 0);
    if (biasesGradient.isPresent()) {
        mBias.get().memsetAsync(gradientUpdateStream, 0);
        vBias.get().memsetAsync(gradientUpdateStream, 0);
    }

    t = 0.0f;
}

void Adam::computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues) {
    trainableLayer->computeWeightsGradient(weightsGradient, biasesGradient, featureIn, errorIn, gradientUpdateStream, accumulateValues);

    // Only increment t when receiving the first errorInput, because when there are multiple stamps of a layer there will be
    // multiple error inputs
    if (!accumulateValues)
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
    // Adam automatically updates its parameters every mini-batch
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

template <typename T>
void Adam::getM(vector<T> &m) {
    if (is_same<T, half>::value)
        assert(this->m.getDataType() == TensorDescriptor::DataType::FP16);
    else if (is_same<T, float>::value)
        assert(this->m.getDataType() == TensorDescriptor::DataType::FP32);
    else
        assert(false);

    m.clear();
    this->m.loadValuesIntoVector(m, gradientUpdateStream);
    gradientUpdateStream.synchronize();
}

template <typename T>
void Adam::getV(vector<T> &v) {
    if (is_same<T, half>::value)
        assert(this->v.getDataType() == TensorDescriptor::DataType::FP16);
    else if (is_same<T, float>::value)
        assert(this->v.getDataType() == TensorDescriptor::DataType::FP32);
    else
        assert(false);

    v.clear();
    this->v.loadValuesIntoVector(v, gradientUpdateStream);
    gradientUpdateStream.synchronize();
}

template <typename T>
void Adam::getMBias(vector<T> &mBias) {
    mBias.clear();
    if (this->mBias.isEmpty())
        return;

    if (is_same<T, half>::value)
        assert(this->mBias.get().getDataType() == TensorDescriptor::DataType::FP16);
    else if (is_same<T, float>::value)
        assert(this->mBias.get().getDataType() == TensorDescriptor::DataType::FP32);
    else
        assert(false);

    this->mBias.get().loadValuesIntoVector(mBias, gradientUpdateStream);
    gradientUpdateStream.synchronize();
}

template <typename T>
void Adam::getVBias(vector<T> &vBias) {
    vBias.clear();
    if (this->vBias.isEmpty())
        return;

    if (is_same<T, half>::value)
        assert(this->vBias.get().getDataType() == TensorDescriptor::DataType::FP16);
    else if (is_same<T, float>::value)
        assert(this->vBias.get().getDataType() == TensorDescriptor::DataType::FP32);
    else
        assert(false);

    this->vBias.get().loadValuesIntoVector(vBias, gradientUpdateStream);
    gradientUpdateStream.synchronize();
}

template <typename T>
void Adam::getWeightsUpdate(std::vector<T> &weightsUpdate) {
    if (is_same<T, half>::value)
        assert(this->weightsUpdate.getDataType() == TensorDescriptor::DataType::FP16);
    else if (is_same<T, float>::value)
        assert(this->weightsUpdate.getDataType() == TensorDescriptor::DataType::FP32);
    else
        assert(false);

    weightsUpdate.clear();
    this->weightsUpdate.loadValuesIntoVector(weightsUpdate, gradientUpdateStream);
    gradientUpdateStream.synchronize();
}

template <typename T>
void Adam::getBiasesUpdate(std::vector<T> &biasesUpdate) {
    vBias.clear();
    if (this->biasesUpdate.isEmpty())
        return;

    if (is_same<T, half>::value)
        assert(this->biasesUpdate.get().getDataType() == TensorDescriptor::DataType::FP16);
    else if (is_same<T, float>::value)
        assert(this->biasesUpdate.get().getDataType() == TensorDescriptor::DataType::FP32);
    else
        assert(false);

    this->biasesUpdate.get().loadValuesIntoVector(biasesUpdate, gradientUpdateStream);
    gradientUpdateStream.synchronize();
}

template void Adam::getM<half>(vector<half> &m);
template void Adam::getM<float>(vector<float> &m);
template void Adam::getV<half>(vector<half> &v);
template void Adam::getV<float>(vector<float> &v);
template void Adam::getMBias<half>(vector<half> &mBias);
template void Adam::getMBias<float>(vector<float> &mBias);
template void Adam::getVBias<half>(vector<half> &vBias);
template void Adam::getVBias<float>(vector<float> &vBias);
template void Adam::getWeightsUpdate<half>(vector<half> &weightsUpdate);
template void Adam::getWeightsUpdate<float>(vector<float> &weightsUpdate);
template void Adam::getBiasesUpdate<half>(vector<half> &biasesUpdate);
template void Adam::getBiasesUpdate<float>(vector<float> &biasesUpdate);
