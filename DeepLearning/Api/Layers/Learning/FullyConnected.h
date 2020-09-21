#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/LayerBase.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"

#include <assert.h>

namespace Thor {

class FullyConnected : TrainableWeightsBiasesLayerBase {
   public:
    class Builder;

    FullyConnected() { initialized = false; }

    virtual bool isMultiLayer() { return useBatchNormalization || dropProportion > 0.0f || activation.isPresent(); }
    virtual void toSingleLayers(vector<LayerBase *> &singleLayers) {
        if (isMultiLayer()) {
            singleLayers.push_back(this);
        } else {
            if (useBatchNormalization) {
                BatchNormalization::Builder builder;
                if (batchNormExponentialRunningAverageFactor.isPresent())
                    builder.exponentialRunningAverageFactor(batchNormExponentialRunningAverageFactor);
                if (batchNormEpsilon.isPresent())
                    builder.epsilon(batchNormEpsilon);
                singleLayers.push_back(builder.build());
            }
            if (dropProportion > 0.0f) {
                singleLayers.push_back(DropOut::Builder().dropProportion(dropProportion).build());
            }

            singleLayers.push_back(this);

            if (activation.isPresent()) {
                singleLayers.push_back(activation.get());
            }
        }
    }

   private:
    bool initialized;

    uint32_t numOutputFeatures;
    bool hasBias;
    Initializer weightsInitializer;
    Initializer biasInitializer;
    Optional<Activation> activation;

    float dropProportion;

    bool useBatchNormalization;
    Optional<double> batchNormExponentialRunningAverageFactor;
    Optional<double> batchNormEpsilon;
};

// featureInput and numOutputFeatures are required, all other parameters are optional.
class FullyConnected::Builder {
   public:
    Builder() { _activationExplicitlyRemoved = false; }

    virtual Layer build() {
        assert(_featureInput.isPresent());
        assert(_numOutputFeatures.isPresent());
        if (_hasBias.isEmpty())
            _hasBias = false;
        if (_weightsInitializer.isEmpty())
            _weightsInitializer = Initializer(new XavierInitializer());
        if (_biasInitializer.isEmpty())
            _biasInitializer = Initializer(new UniformRandomInitializer());
        if (_activation.isEmpty() && !_activationExplicitlyRemoved)
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
        fullyConnected->featureOutput = Tensor();
        fullyConnected->initialized = true;

        return Layer(fullyConnected);
    }

    FullyConnected::Builder featureInput(Tensor _featureInput) {
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
    FullyConnected::Builder activation(Optional<Activation> _activation) {
        assert(!this->_activation.isPresent());
        assert(!_activationExplicitlyRemoved);

        if (_activation.isEmpty()) {
            _activationExplicitlyRemoved = true;
        } else {
            this->_activation = _activation;
        }
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

    // Adds a DropOut layer before this FullyConnected layer, but after the BatchNormalization layer when that is also present.
    FullyConnected::Builder dropOut(float _dropProportion) {
        assert(!this->_dropProportion.isPresent());
        this->_dropProportion = _dropProportion;
        return *this;
    }

   private:
    Optional<Tensor> _featureInput;
    Optional<uint32_t> _numOutputFeatures;
    Optional<bool> _hasBias;
    Optional<Initializer> _weightsInitializer;
    Optional<Initializer> _biasInitializer;
    Optional<Activation> _activation;
    bool _activationExplicitlyRemoved;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;
};

}  // namespace Thor
