#pragma once

#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Layers/Utility/Pooling.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <string>

namespace Thor {

class Inception : public TrainableWeightsBiasesLayer {
   public:
    class Builder;
    Inception() {}

    virtual ~Inception() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Inception>(*this); }

    virtual std::string getLayerType() const { return "Inception"; }

   protected:
    virtual bool isMultiLayer() const { return true; }
    virtual void convertToSingleLayersAndAddToNetwork();

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             std::vector<std::shared_ptr<Initializer>> &initializers) const {
        // Inception is purely a compound layer consisting of other layers, each which stamp themselves.
        // So when an inception layer converts itself to single layers, none of those layers is an inception layer
        // so it will never be added to the network as a stampable layer, and so stamp will never be called for inception.
        assert(false);
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // Inception is purely a compound layer consisting of other layers, each which stamp themselves.
        // So when an inception layer converts itself to single layers, none of those layers is an inception layer
        // so it will never be added to the network as a stampable layer, and so getFirstInstanceMemRequirementInBytes
        // will never be called for inception.
        return 0;
    }

   private:
    Network *network;
    uint32_t outputChannels1x1;
    uint32_t inputChannels3x3;
    uint32_t outputChannels3x3;
    uint32_t inputChannels5x5;
    uint32_t outputChannels5x5;
    uint32_t outputChannelsPooling;

    std::shared_ptr<Initializer::Builder> weightsInitializerBuilder;
    std::shared_ptr<Initializer::Builder> biasInitializerBuilder;
};

class Inception::Builder {
   public:
    virtual Inception build() {
        assert(_network.isPresent());
        assert(!_featureInputs.empty());
        assert(_outputChannels1x1.isPresent());
        assert(_inputChannels3x3.isPresent());
        assert(_outputChannels3x3.isPresent());
        assert(_inputChannels5x5.isPresent());
        assert(_outputChannels5x5.isPresent());
        assert(_outputChannelsPooling.isPresent());

        if (_weightsInitializerBuilder == nullptr)
            _weightsInitializerBuilder = std::make_shared<Glorot::Builder>(Glorot::Builder());
        if (_biasInitializerBuilder == nullptr)
            _biasInitializerBuilder = std::make_shared<Glorot::Builder>(Glorot::Builder());

        Inception inception;
        inception.network = _network;
        inception.featureInputs = _featureInputs;
        inception.outputChannels1x1 = _outputChannels1x1;
        inception.inputChannels3x3 = _inputChannels3x3;
        inception.outputChannels3x3 = _outputChannels3x3;
        inception.inputChannels5x5 = _inputChannels5x5;
        inception.outputChannels5x5 = _outputChannels5x5;
        inception.outputChannelsPooling = _outputChannelsPooling;

        inception.weightsInitializerBuilder = _weightsInitializerBuilder->clone();
        inception.biasInitializerBuilder = _biasInitializerBuilder->clone();

        inception.initialized = true;
        inception.addToNetwork(_network.get());
        return inception;
    }

    virtual Inception::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Inception::Builder &featureInput(Tensor _featureInput) {
        assert(_featureInput.getDimensions().size() == 3);
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            assert(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            assert(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual Inception::Builder &outputChannels1x1(uint32_t _outputChannels1x1) {
        assert(!this->_outputChannels1x1.isPresent());
        assert(_outputChannels1x1 > 0);
        this->_outputChannels1x1 = _outputChannels1x1;
        return *this;
    }

    virtual Inception::Builder &inputChannels3x3(uint32_t _inputChannels3x3) {
        assert(!this->_inputChannels3x3.isPresent());
        assert(_inputChannels3x3 > 0);
        this->_inputChannels3x3 = _inputChannels3x3;
        return *this;
    }

    virtual Inception::Builder &outputChannels3x3(uint32_t _outputChannels3x3) {
        assert(!this->_outputChannels3x3.isPresent());
        assert(_outputChannels3x3 > 0);
        this->_outputChannels3x3 = _outputChannels3x3;
        return *this;
    }

    virtual Inception::Builder &inputChannels5x5(uint32_t _inputChannels5x5) {
        assert(!this->_inputChannels5x5.isPresent());
        assert(_inputChannels5x5 > 0);
        this->_inputChannels5x5 = _inputChannels5x5;
        return *this;
    }

    virtual Inception::Builder &outputChannels5x5(uint32_t _outputChannels5x5) {
        assert(!this->_outputChannels5x5.isPresent());
        assert(_outputChannels5x5 > 0);
        this->_outputChannels5x5 = _outputChannels5x5;
        return *this;
    }

    virtual Inception::Builder &outputChannelsPooling(uint32_t _outputChannelsPooling) {
        assert(!this->_outputChannelsPooling.isPresent());
        assert(_outputChannelsPooling > 0);
        this->_outputChannelsPooling = _outputChannelsPooling;
        return *this;
    }

    virtual Inception::Builder &weightsInitializerBuilder(Initializer::Builder &_weightsInitializerBuilder) {
        assert(this->_weightsInitializerBuilder == nullptr);
        this->_weightsInitializerBuilder = _weightsInitializerBuilder.clone();
        return *this;
    }

    virtual Inception::Builder &weightsInitializerBuilder(Initializer::Builder &&_weightsInitializerBuilder) {
        assert(this->_weightsInitializerBuilder == nullptr);
        this->_weightsInitializerBuilder = _weightsInitializerBuilder.clone();
        return *this;
    }

    virtual Inception::Builder &biasInitializerBuilder(Initializer::Builder &_biasInitializerBuilder) {
        assert(this->_biasInitializerBuilder == nullptr);
        this->_biasInitializerBuilder = _biasInitializerBuilder.clone();
        return *this;
    }

    virtual Inception::Builder &biasInitializerBuilder(Initializer::Builder &&_biasInitializerBuilder) {
        assert(this->_biasInitializerBuilder == nullptr);
        this->_biasInitializerBuilder = _biasInitializerBuilder.clone();
        return *this;
    }

   private:
    Optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    Optional<uint32_t> _outputChannels1x1;
    Optional<uint32_t> _inputChannels3x3;
    Optional<uint32_t> _outputChannels3x3;
    Optional<uint32_t> _inputChannels5x5;
    Optional<uint32_t> _outputChannels5x5;
    Optional<uint32_t> _outputChannelsPooling;

    std::shared_ptr<Initializer::Builder> _weightsInitializerBuilder;
    std::shared_ptr<Initializer::Builder> _biasInitializerBuilder;
};

}  // namespace Thor
