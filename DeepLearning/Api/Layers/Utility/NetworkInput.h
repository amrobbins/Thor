#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"

#include <string>

namespace Thor {

class NetworkInput : public Layer {
   public:
    class Builder;

    NetworkInput() {}

    virtual ~NetworkInput() {}

    virtual std::string getName() const { return name; }
    std::vector<uint64_t> getDimensions() const { return dimensions; }
    Tensor::DataType getDataType() const { return dataType; }

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<NetworkInput>(*this); }

    virtual std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) { return {featureOutput}; }

    virtual std::string getLayerType() const { return "NetworkInput"; }

   protected:
    virtual std::shared_ptr<ThorImplementation::NetworkInput> stamp(ThorImplementation::TensorPlacement placement,
                                                                    uint32_t batchSize) const {
        assert(initialized);

        std::vector<uint64_t> batchDimensions;
        batchDimensions.push_back(batchSize);
        for (uint32_t i = 0; i < dimensions.size(); ++i)
            batchDimensions.push_back(dimensions[i]);

        std::shared_ptr<ThorImplementation::NetworkInput> networkInput = std::make_shared<ThorImplementation::NetworkInput>(
            placement, Tensor::convertToImplementationDataType(dataType), batchDimensions);
        networkInput->setName(name);

        return networkInput;
    }

    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(false);
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // Input has a prefetch buffer in addition to storing the output tensor
        return 2 * featureOutput.get().getTotalSizeInBytes();
    }

   private:
    std::string name;
    std::vector<uint64_t> dimensions;
    Tensor::DataType dataType;

    friend class Network;
};

class NetworkInput::Builder {
   public:
    virtual NetworkInput build() {
        assert(_network.isPresent());
        assert(_dimensions.isPresent());
        assert(_dataType.isPresent());

        NetworkInput networkInput;
        if (_name.isPresent())
            networkInput.name = _name;
        else
            networkInput.name = std::string("NetworkInput") + std::to_string(networkInput.getId());
        networkInput.dimensions = _dimensions;
        networkInput.dataType = _dataType;
        networkInput.featureInput = Tensor(_dataType, _dimensions);
        networkInput.featureOutput = Tensor(_dataType, _dimensions);
        networkInput.initialized = true;
        networkInput.addToNetwork(_network.get());
        return networkInput;
    }

    virtual NetworkInput::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual NetworkInput::Builder &name(std::string _name) {
        assert(!_name.empty());
        assert(this->_name.isEmpty());
        this->_name = _name;
        return *this;
    }

    // Note: dimensions do not include batch size
    virtual NetworkInput::Builder &dimensions(std::vector<uint64_t> _dimensions) {
        assert(!_dimensions.empty());
        this->_dimensions = _dimensions;
        return *this;
    }

    virtual NetworkInput::Builder &dataType(Tensor::DataType _dataType) {
        assert(Tensor::dataTypeValid(_dataType));
        this->_dataType = _dataType;
        return *this;
    }

   private:
    Optional<std::string> _name;
    Optional<Network *> _network;
    Optional<std::vector<uint64_t>> _dimensions;
    Optional<Tensor::DataType> _dataType;
};

}  // namespace Thor
