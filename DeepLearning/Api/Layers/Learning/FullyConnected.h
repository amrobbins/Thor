#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/LayerBase.h"

#include <assert.h>

namespace Thor {

class FullyConnected : LayerBase {
   public:
    class Builder;

    FullyConnected() { initialized = false; }

    virtual Tensor getFeatureOutput();  // this is not implementation Tensor

   private:
    bool initialized;

    Tensor featureInput;
    uint32_t numOutputFeatures;
    bool hasBias;
    Initializer weightsInitializer;
    Initializer biasInitializer;
    Activation activation;

    float dropProportion;

    bool useBatchNormalization;
    Optional<double> batchNormExponentialRunningAverageFactor;
    Optional<double> batchNormEpsilon;
};

// featureInput and numOutputFeatures are required, all other parameters are optional.
class FullyConnected::Builder {
   public:
    Builder();

    virtual Layer build() {
        assert(_featureInput.isPresent());
        assert(_numOutputFeatures.isPresent());
        if (_hasBias.isEmpty())
            _hasBias = false;
        if (_weightsInitializer.isEmpty())
            _weightsInitializer = Initializer(new XavierInitializer());
        if (_biasInitializer.isEmpty())
            _biasInitializer = Initializer(new UniformRandomInitializer());
        if (_activation.isEmpty())
            _activation = Activation(new Relu());
        if (_dropProportion.isEmpty())
            _dropProportion = 0.0f;
        if (_useBatchNormalization.isEmpty()) {
            _useBatchNormalization = false;
        }

        FullyConnected *fullyConnected = new FullyConnected();

        fullyConnected->featureInput = _featureInput;  // featureInput should be immutable
        fullyConnected->numOutputFeatures = _numOutputFeatures;
        fullyConnected->hasBias = _hasBias;
        fullyConnected->weightsInitializer = _weightsInitializer;
        fullyConnected->biasInitializer = _biasInitializer;
        fullyConnected->activation = _activation;
        fullyConnected->dropProportion = _dropProportion;
        fullyConnected->useBatchNormalization = _useBatchNormalization;
        fullyConnected->batchNormExponentialRunningAverageFactor = _batchNormExponentialRunningAverageFactor;
        fullyConnected->batchNormEpsilon = _batchNormEpsilon;
        fullyConnected->initialized = true;

        return Layer(fullyConnected);
    }

    FullyConnected::Builder featureInput(Tensor _featureInput) {
        assert(_featureInput.isInitialized());
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    FullyConnected::Builder numOutputFeatures(uint32_t _numOutputFeatures) {
        assert(!this->_numOutputFeatures.isPresent());
        this->_numOutputFeatures = _numOutputFeatures;
        return *this;
    }

    FullyConnected::Builder hasBias(bool _hasBias) {
        assert(!this->_hasBias.isPresent());
        this->_hasBias = _hasBias;
        return *this;
    }

    FullyConnected::Builder weightsInitializer(Initializer _weightsInitializer) {
        assert(!this->_weightsInitializer.isPresent());
        this->_weightsInitializer = _weightsInitializer;
        return *this;
    }

    FullyConnected::Builder biasInitializer(Initializer _biasInitializer) {
        assert(!this->_biasInitializer.isPresent());
        this->_biasInitializer = _biasInitializer;
        return *this;
    }

    // Adds an activation layer after this FullyConnected layer
    FullyConnected::Builder activation(Activation _activation) {
        assert(!this->_activation.isPresent());
        this->_activation = _activation;
        return *this;
    }

    // Adds a DropOut layer before this FullyConnected layer, but after the BatchNormalization layer when that is also present.
    FullyConnected::Builder dropOut(float _dropProportion) {
        assert(!this->_dropProportion.isPresent());
        this->_dropProportion = _dropProportion;
        return *this;
    }

    // Adds a BatchNormalization layer before this FullyConnected layer and before the DropOut layer when that is also present
    // exponentialRunningAverageFactor and epsilon will be set to good default values when not specified.
    FullyConnected::Builder batchNormalization(Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                                               Optional<double> epsilon = Optional<double>::empty()) {
        assert(!_useBatchNormalization.isPresent());
        this->_useBatchNormalization = true;
        this->_batchNormExponentialRunningAverageFactor = exponentialRunningAverageFactor;
        this->_batchNormEpsilon = epsilon;
        return *this;
    }

   private:
    Optional<Tensor> _featureInput;
    Optional<uint32_t> _numOutputFeatures;
    Optional<bool> _hasBias;
    Optional<Initializer> _weightsInitializer;
    Optional<Initializer> _biasInitializer;
    Optional<Activation> _activation;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;
};

}  // namespace Thor
