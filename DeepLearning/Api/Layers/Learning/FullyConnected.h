#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandomInitializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"

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

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer = nullptr,
                                             Thor::Tensor connectingApiTensor = Thor::Tensor()) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        ThorImplementation::FullyConnected *fullyConnected = new ThorImplementation::FullyConnected(numOutputFeatures, hasBias);
        Thor::Layer::connectTwoLayers(drivingLayer, fullyConnected, drivingApiLayer, this, connectingApiTensor);

        weightsInitializerBuilder->network(*network);
        weightsInitializerBuilder->tensorToInitialize(fullyConnected->getWeights());
        weightsInitializerBuilder->build();

        if (fullyConnected->getBiases().isPresent()) {
            biasInitializerBuilder->network(*network);
            biasInitializerBuilder->tensorToInitialize(fullyConnected->getBiases().get());
            biasInitializerBuilder->build();
        }

        return fullyConnected;
    }

    // mem requirements are the weights
    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize) const {
        // FIXME: workspace? Or do I assume no workspace at first and can add one later if have extra mem?
        uint64_t numInputFeatures = featureInputs[0].getDimensions()[0];
        uint64_t numWeights = numInputFeatures * numOutputFeatures;
        uint64_t numBiases = numOutputFeatures;
        // have weights and gradient accumulators, as FP16 elements
        uint64_t fixedMem = 2 * (numWeights + numBiases) * 2;
        uint64_t batchSizeDependentMem =
            featureInputs.size() * (featureInputs[0].getTotalSizeInBytes() + featureOutputs[0].getTotalSizeInBytes()) * batchSize;

        return fixedMem + batchSizeDependentMem;
    }

    virtual uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize) const {
        uint64_t batchSizeDependentMem =
            featureInputs.size() * (featureInputs[0].getTotalSizeInBytes() + featureOutputs[0].getTotalSizeInBytes()) * batchSize;
        return batchSizeDependentMem;
    }

   private:
    uint32_t numOutputFeatures;
    bool hasBias;
    shared_ptr<Initializer::Builder> weightsInitializerBuilder;
    shared_ptr<Initializer::Builder> biasInitializerBuilder;
    shared_ptr<Activation::Builder> activationBuilder;

    float dropProportion;

    Network *network;
    bool useBatchNormalization;
    Optional<double> batchNormExponentialRunningAverageFactor;
    Optional<double> batchNormEpsilon;

    friend class Network;
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
        if (_weightsInitializerBuilder == nullptr)
            _weightsInitializerBuilder =
                make_shared<UniformRandomInitializer::Builder>(UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1));
        if (_biasInitializerBuilder == nullptr)
            _biasInitializerBuilder =
                make_shared<UniformRandomInitializer::Builder>(UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1));
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
        for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
            fullyConnected.featureOutputs.push_back(
                Tensor(fullyConnected.featureInputs[0].getDataType(), {fullyConnected.numOutputFeatures}));
            fullyConnected.outputTensorFromInputTensor[fullyConnected.featureInputs[i]] = fullyConnected.featureOutputs.back();
            fullyConnected.inputTensorFromOutputTensor[fullyConnected.featureOutputs.back()] = fullyConnected.featureInputs[i];
        }

        fullyConnected.hasBias = _hasBias;
        fullyConnected.weightsInitializerBuilder = _weightsInitializerBuilder->clone();
        fullyConnected.biasInitializerBuilder = _biasInitializerBuilder->clone();
        if (_activationBuilder != nullptr)
            fullyConnected.activationBuilder = _activationBuilder->clone();
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
        assert(!_featureInput.getDimensions().empty());
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

    virtual FullyConnected::Builder &weightsInitializerBuilder(Initializer::Builder &_weightsInitializerBuilder) {
        assert(this->_weightsInitializerBuilder == nullptr);
        this->_weightsInitializerBuilder = _weightsInitializerBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializerBuilder(Initializer::Builder &&_weightsInitializerBuilder) {
        assert(this->_weightsInitializerBuilder == nullptr);
        this->_weightsInitializerBuilder = _weightsInitializerBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializerBuilder(Initializer::Builder &_biasInitializerBuilder) {
        assert(this->_biasInitializerBuilder == nullptr);
        this->_biasInitializerBuilder = _biasInitializerBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializerBuilder(Initializer::Builder &&_biasInitializerBuilder) {
        assert(this->_biasInitializerBuilder == nullptr);
        this->_biasInitializerBuilder = _biasInitializerBuilder.clone();
        return *this;
    }

    // Adds an activation layer after this FullyConnected layer
    virtual FullyConnected::Builder &activationBuilder(Activation::Builder &_activationBuilder) {
        assert(this->_activationBuilder == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activationBuilder = _activationBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &activationBuilder(Activation::Builder &&_activationBuilder) {
        assert(this->_activationBuilder == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activationBuilder = _activationBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &noActivation() {
        assert(!this->_activationBuilder);

        _activationExplicitlyRemoved = true;
        return *this;
    }

    // FIXME: batchNormalization and dropOut should be passed as builders. To support this Layer::Builder will need to be created with
    // virtual shared_ptr<Layer::Builder> clone.

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
    shared_ptr<Initializer::Builder> _weightsInitializerBuilder;
    shared_ptr<Initializer::Builder> _biasInitializerBuilder;
    shared_ptr<Activation::Builder> _activationBuilder;
    bool _activationExplicitlyRemoved;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;
};

}  // namespace Thor
