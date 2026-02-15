#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"

using namespace ThorImplementation;
using namespace std;

Adam::Adam(std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer, float alpha, float beta1, float beta2, float epsilon) {
    this->alpha = alpha;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
    this->t = 0.0f;

    this->trainableLayerShared = trainableLayer;
    this->trainableLayer = trainableLayer.get();
}

void Adam::compile() {
    TensorPlacement layerPlacement = trainableLayer->getPlacement();
    assert(layerPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    gpuNum = layerPlacement.getDeviceNum();
    gradientUpdateStream = Stream::getNextGradientUpdateStream(gpuNum);

    // Allocate all params
    Tensor weights = trainableLayer->getWeights();
    weightsGradient = weights.clone();
    Optional<Tensor> biases = trainableLayer->getBiases();
    if (biases.isPresent())
        biasesGradient = biases.get().clone();

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

    // The minimum strictly positive (subnormal) value of fp16 is 2^−24 ≈ 5.96 × 10^−8
    // So the default value of epsilon (which prevents divide by zero) is set to no less than this when epsilon is FP16.
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    if (weightsGradient.getDataType() == TensorDescriptor::DataType::FP16 && epsilon < 5.96046448e-8)
        epsilon = __float2half_ru(5.9604644e-8f);

    assert(weightsGradient.getDataType() == TensorDescriptor::DataType::FP16 ||
           weightsGradient.getDataType() == TensorDescriptor::DataType::FP32);

    compiled = true;
}

void Adam::computeWeightsUpdate(Optional<Tensor> featureIn, Optional<Tensor> errorIn, bool accumulateValues) {
    trainableLayer->computeWeightsGradient(weightsGradient, biasesGradient, featureIn, errorIn, gradientUpdateStream, accumulateValues);

    if (featureDataType.isEmpty()) {
        featureDataType = featureIn.get().getDataType();
        assert(errorIn.get().getDataType() == featureDataType);
    }

    stepFromPrecomputedGradient(accumulateValues);
}

void Adam::stepFromPrecomputedGradient(bool accumulateValues) {
    // Only increment t when receiving the first errorInput, because when there are multiple stamps of a layer there will be
    // multiple error inputs
    if (!accumulateValues)
        t += 1;
    if (featureDataType.get() == TensorDescriptor::DataType::FP16) {
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
    } else if (featureDataType.get() == TensorDescriptor::DataType::FP32) {
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
    // FIXME: That will not work for multiple stamps, together which form a minibatch
    unordered_map<string, float> hyperParameters;
    hyperParameters["t"] = t;
    return hyperParameters;
}

void Adam::dumpMToFile(std::string filename, Optional<Stream> stream) {
    if (stream.isEmpty())
        stream = getGradientUpdateStream();
    if (m.getAttachedFilename() != filename)
        m.attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, true);
    m.dumpToFile(stream);
}

void Adam::dumpVToFile(std::string filename, Optional<Stream> stream) {
    if (stream.isEmpty())
        stream = getGradientUpdateStream();
    if (v.getAttachedFilename() != filename)
        v.attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, true);
    v.dumpToFile(stream);
}

void Adam::dumpMBiasToFile(std::string filename, Optional<Stream> stream) {
    assert(mBias.isPresent());
    if (stream.isEmpty())
        stream = getGradientUpdateStream();
    if (mBias.get().getAttachedFilename() != filename)
        mBias.get().attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, true);
    mBias.get().dumpToFile(stream);
}

void Adam::dumpVBiasToFile(std::string filename, Optional<Stream> stream) {
    assert(vBias.isPresent());
    if (stream.isEmpty())
        stream = getGradientUpdateStream();
    if (vBias.get().getAttachedFilename() != filename)
        vBias.get().attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, true);
    vBias.get().dumpToFile(stream);
}

void Adam::loadMFromFile(std::string filename, Optional<Stream> stream) {
    if (stream.isEmpty())
        stream = getGradientUpdateStream();
    if (m.getAttachedFilename() != filename)
        m.attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, false);
    m.loadFromFile(stream);
}

void Adam::loadVFromFile(std::string filename, Optional<Stream> stream) {
    if (stream.isEmpty())
        stream = getGradientUpdateStream();
    if (v.getAttachedFilename() != filename)
        v.attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, false);
    v.loadFromFile(stream);
}

void Adam::loadMBiasFromFile(std::string filename, Optional<Stream> stream) {
    assert(mBias.isPresent());
    if (stream.isEmpty())
        stream = getGradientUpdateStream();
    if (mBias.get().getAttachedFilename() != filename)
        mBias.get().attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, false);
    mBias.get().loadFromFile(stream);
}

void Adam::loadVBiasFromFile(std::string filename, Optional<Stream> stream) {
    assert(vBias.isPresent());
    if (stream.isEmpty())
        stream = getGradientUpdateStream();
    if (vBias.get().getAttachedFilename() != filename)
        vBias.get().attachFile(filename, 0, Tensor::FileAccess::READ_WRITE, false);
    vBias.get().loadFromFile(stream);
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

Tensor Adam::getM() { return m; }
Tensor Adam::getV() { return v; }
Optional<Tensor> Adam::getMBias() { return mBias; }
Optional<Tensor> Adam::getVBias() { return vBias; }
