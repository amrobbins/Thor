#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

namespace Thor {

class NetworkOutput : public Layer {
   public:
    class Builder;

    NetworkOutput() {}

    virtual ~NetworkOutput() {}

    virtual string getName() const { return name; }

    virtual shared_ptr<Layer> clone() const { return make_shared<NetworkOutput>(*this); }

    Tensor::DataType getDataType() const { return dataType; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::NetworkOutput *networkOutput = new ThorImplementation::NetworkOutput(placement);
        networkOutput->setName(name);
        Thor::Layer::connectTwoLayers(drivingLayer, networkOutput, drivingApiLayer, this, connectingApiTensor);
        return networkOutput;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize) const {
        uint64_t totalBytes = 0;
        if (featureInput.get().getDataType() != dataType)
            totalBytes += featureInput.get().clone(dataType).getTotalSizeInBytes();
        return totalBytes;
    }

   private:
    string name;
    Tensor::DataType dataType;
};

class NetworkOutput::Builder {
   public:
    virtual NetworkOutput build() {
        assert(_network.isPresent());
        assert(!_inputTensor.isEmpty());
        if (_dataType.isEmpty())
            _dataType = Tensor::DataType::FP32;

        NetworkOutput networkOutput;
        if (_name.isPresent())
            networkOutput.name = _name;
        else
            networkOutput.name = string("NetworkOutput") + to_string(networkOutput.getId());
        networkOutput.dataType = _dataType;
        networkOutput.featureInput = _inputTensor;
        networkOutput.featureOutput = Tensor(_dataType, _inputTensor.get().getDimensions());
        networkOutput.initialized = true;
        networkOutput.addToNetwork(_network.get());
        return networkOutput;
    }

    virtual NetworkOutput::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual NetworkOutput::Builder &name(string _name) {
        assert(!_name.empty());
        assert(this->_name.isEmpty());
        this->_name = _name;
        return *this;
    }

    virtual NetworkOutput::Builder &inputTensor(Tensor _inputTensor) {
        assert(_inputTensor.isInitialized());
        this->_inputTensor = _inputTensor;
        return *this;
    }

    virtual NetworkOutput::Builder &dataType(Tensor::DataType _dataType) {
        assert(Tensor::dataTypeValid(_dataType));
        this->_dataType = _dataType;
        return *this;
    }

   private:
    Optional<string> _name;
    Optional<Network *> _network;
    Optional<Tensor> _inputTensor;
    Optional<Tensor::DataType> _dataType;
};

}  // namespace Thor
