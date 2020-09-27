#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandomInitializer.h"
#include "DeepLearning/Api/Initializers/XavierInitializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"

#include <assert.h>

namespace Thor {

class FullyConnected : public TrainableWeightsBiasesLayer {
   public:
    class Builder;

    FullyConnected() {}

    virtual ~FullyConnected() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<FullyConnected>(*this); }

   protected:
    virtual bool isMultiLayer() const { return useBatchNormalization || dropProportion > 0.0f || activationBuilder; }

    virtual void convertToSingleLayersAndAddToNetwork();

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
    }

   private:
    uint32_t numOutputFeatures;
    bool hasBias;
    Initializer weightsInitializer;
    Initializer biasInitializer;
    shared_ptr<Activation::Builder> activationBuilder;

    float dropProportion;

    Network *network;
    bool useBatchNormalization;
    Optional<double> batchNormExponentialRunningAverageFactor;
    Optional<double> batchNormEpsilon;
};

// featureInput and numOutputFeatures are required, all other parameters are optional.
class FullyConnected::Builder {
   public:
    Builder() { _activationExplicitlyRemoved = false; }

    virtual FullyConnected build() {
        assert(_network.isPresent());
        assert(!_featureInputs.empty());
        assert(_numOutputFeatures.isPresent());
        if (_hasBias.isEmpty())
            _hasBias = false;
        if (_weightsInitializer.isEmpty())
            _weightsInitializer = XavierInitializer();
        if (_biasInitializer.isEmpty())
            _biasInitializer = UniformRandomInitializer();
        if (!_activationBuilder && !_activationExplicitlyRemoved)
            _activationBuilder = make_shared<Relu::Builder>(Relu::Builder());
        if (_dropProportion.isEmpty())
            _dropProportion = 0.0f;
        if (_useBatchNormalization.isEmpty()) {
            _useBatchNormalization = false;
        }

        FullyConnected fullyConnected;

        fullyConnected.network = _network;
        fullyConnected.featureInputs = _featureInputs;
        fullyConnected.numOutputFeatures = _numOutputFeatures;
        for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i)
            fullyConnected.featureOutputs.push_back(
                Tensor(fullyConnected.featureInputs[0].getDataType(), {fullyConnected.numOutputFeatures}));
        fullyConnected.hasBias = _hasBias;
        fullyConnected.weightsInitializer = _weightsInitializer;
        fullyConnected.biasInitializer = _biasInitializer;
        fullyConnected.activationBuilder = _activationBuilder;
        fullyConnected.dropProportion = _dropProportion;
        fullyConnected.useBatchNormalization = _useBatchNormalization;
        fullyConnected.batchNormExponentialRunningAverageFactor = _batchNormExponentialRunningAverageFactor;
        fullyConnected.batchNormEpsilon = _batchNormEpsilon;
        fullyConnected.initialized = true;
        fullyConnected.addToNetwork(fullyConnected.network);

        return fullyConnected;
    }

    virtual FullyConnected::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual FullyConnected::Builder &featureInput(Tensor _featureInput) {
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            assert(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            assert(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual FullyConnected::Builder &numOutputFeatures(uint32_t _numOutputFeatures) {
        assert(!this->_numOutputFeatures.isPresent());
        this->_numOutputFeatures = _numOutputFeatures;
        return *this;
    }

    virtual FullyConnected::Builder &hasBias(bool _hasBias) {
        assert(!this->_hasBias.isPresent());
        this->_hasBias = _hasBias;
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializer(Initializer _weightsInitializer) {
        assert(!this->_weightsInitializer.isPresent());
        this->_weightsInitializer = _weightsInitializer;
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializer(Initializer _biasInitializer) {
        assert(!this->_biasInitializer.isPresent());
        this->_biasInitializer = _biasInitializer;
        return *this;
    }

    // Adds an activation layer after this FullyConnected layer
    virtual FullyConnected::Builder &activationBuilder(Activation::Builder &_activationBuilder) {
        assert(!this->_activationBuilder);
        assert(!_activationExplicitlyRemoved);

        // FIXME: is there a better solution so that I don't have to know in advance every type of activation?
        // maybe not?
        // https://stackoverflow.com/questions/55084135/is-it-possible-to-automatically-cast-to-most-derived-type#:~:text=C%2B%2B%20Is%20it%20possible%20to,need%20is%20a%20virtual%20function.
        // The issue is that I want to own the memory of the object and so I need to instantiate the object, therefore I need to know what
        // type of object to instantiate. so what I need is:
        //  this->_activationBuilder =
        //  make_shared<derivedTypeOf(_activationBuilder)>(*dynamic_cast<derivedTypeOf(_activationBuilder)*>(&_activationBuilder));

        if (dynamic_cast<Relu::Builder *>(&_activationBuilder)) {
            this->_activationBuilder = make_shared<Relu::Builder>(*dynamic_cast<Relu::Builder *>(&_activationBuilder));
        } else if (dynamic_cast<Tanh::Builder *>(&_activationBuilder)) {
            this->_activationBuilder = make_shared<Tanh::Builder>(*dynamic_cast<Tanh::Builder *>(&_activationBuilder));
        } else {
            assert(false);
        }

        return *this;
    }

    virtual FullyConnected::Builder &noActivation() {
        assert(!this->_activationBuilder);

        _activationExplicitlyRemoved = true;
        return *this;
    }

    // Adds a BatchNormalization layer before this FullyConnected layer and before the DropOut layer when that is also present
    // exponentialRunningAverageFactor and epsilon will be set to good default values when not specified.
    virtual FullyConnected::Builder &batchNormalization(Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                                                        Optional<double> epsilon = Optional<double>::empty()) {
        assert(!_useBatchNormalization.isPresent());
        this->_useBatchNormalization = true;
        this->_batchNormExponentialRunningAverageFactor = exponentialRunningAverageFactor;
        this->_batchNormEpsilon = epsilon;
        return *this;
    }

    // Adds a DropOut layer before this FullyConnected layer, but after the BatchNormalization layer when that is also present.
    virtual FullyConnected::Builder &dropOut(float _dropProportion) {
        assert(!this->_dropProportion.isPresent());
        this->_dropProportion = _dropProportion;
        return *this;
    }

   private:
    Optional<Network *> _network;
    vector<Tensor> _featureInputs;
    Optional<uint32_t> _numOutputFeatures;
    Optional<bool> _hasBias;
    Optional<Initializer> _weightsInitializer;
    Optional<Initializer> _biasInitializer;
    shared_ptr<Activation::Builder> _activationBuilder;
    bool _activationExplicitlyRemoved;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;
};

}  // namespace Thor
