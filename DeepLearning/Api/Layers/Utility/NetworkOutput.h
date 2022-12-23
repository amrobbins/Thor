#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

namespace Thor {

class NetworkOutput : public Layer {
   public:
    class Builder;

    NetworkOutput() {}

    virtual ~NetworkOutput() {}

    virtual std::string getName() const { return name; }

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<NetworkOutput>(*this); }

    Tensor::DataType getDataType() const { return dataType; }

    virtual std::string getLayerType() const { return "NetworkOutput"; }

    virtual bool isMultiLayer() const {
        assert(featureInput.isPresent());
        return featureInput.get().getDataType() != dataType;
    }

    virtual void buildSupportLayersAndAddToNetwork();

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             std::vector<std::shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == featureInput.get());

        ThorImplementation::NetworkOutput *networkOutput = new ThorImplementation::NetworkOutput(placement);
        networkOutput->setName(name);
        Thor::Layer::connectTwoLayers(drivingLayer, networkOutput, drivingApiLayer, this, connectingApiTensor);
        return networkOutput;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // If there is a type conversion required, the memory requirement is reported by the TypeConverter layer
        return 0;
    }

   private:
    std::string name;
    Tensor::DataType dataType;
    Network *network;
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
            networkOutput.name = std::string("NetworkOutput") + std::to_string(networkOutput.getId());
        networkOutput.dataType = _dataType;
        networkOutput.featureInput = _inputTensor;
        networkOutput.initialized = true;
        networkOutput.network = _network.get();

        if (networkOutput.isMultiLayer()) {
            // A type converter will be stamped where the new data type will take effect, when it is needed.
            networkOutput.buildSupportLayersAndAddToNetwork();
        } else {
            networkOutput.featureOutput = Tensor(_inputTensor.get().getDataType(), _inputTensor.get().getDimensions());
            networkOutput.addToNetwork(_network.get());
        }

        return networkOutput;
    }

    virtual NetworkOutput::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual NetworkOutput::Builder &name(std::string _name) {
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
    Optional<std::string> _name;
    Optional<Network *> _network;
    Optional<Tensor> _inputTensor;
    Optional<Tensor::DataType> _dataType;
};

}  // namespace Thor
