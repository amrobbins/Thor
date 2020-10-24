#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"

#include <string>

using std::string;
using std::to_string;

namespace Thor {

class NetworkInput : public Layer {
   public:
    class Builder;

    NetworkInput() {}

    virtual ~NetworkInput() {}

    virtual string getName() const { return name; }
    vector<uint64_t> getDimensions() const { return dimensions; }
    Tensor::DataType getDataType() const { return dataType; }

    virtual shared_ptr<Layer> clone() const { return make_shared<NetworkInput>(*this); }

    virtual vector<Tensor> getOutputsFromInput(Tensor inputTensor) { return {featureOutput}; }

   protected:
    virtual ThorImplementation::NetworkInput *stamp(ThorImplementation::TensorPlacement placement,
                                                    uint32_t batchSize,
                                                    vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);

        ThorImplementation::TensorDescriptor::DataType implementationDataType;
        if (dataType == Tensor::DataType::FP32)
            implementationDataType = ThorImplementation::TensorDescriptor::DataType::FP32;
        else if (dataType == Tensor::DataType::FP16)
            implementationDataType = ThorImplementation::TensorDescriptor::DataType::FP16;
        else if (dataType == Tensor::DataType::UINT8)
            implementationDataType = ThorImplementation::TensorDescriptor::DataType::UINT8;
        else
            assert(false);

        vector<uint64_t> batchDimensions;
        batchDimensions.push_back(batchSize);
        for (uint32_t i = 0; i < dimensions.size(); ++i)
            batchDimensions.push_back(dimensions[i]);

        ThorImplementation::NetworkInput *networkInput =
            new ThorImplementation::NetworkInput(placement, implementationDataType, batchDimensions);
        networkInput->setName(name);

        return networkInput;
    }

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers,
                                             StreamPackage gradientUpdateStreamPackage = StreamPackage()) const {
        assert(false);
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        uint64_t elements = featureInput.get().getTotalNumElements();

        uint32_t bytesPerElement;
        if (dataType == Tensor::DataType::FP32)
            bytesPerElement = 6;  // original + converted
        else if (dataType == Tensor::DataType::FP16)
            bytesPerElement = 2;
        else if (dataType == Tensor::DataType::UINT8)
            bytesPerElement = 3;  // original + converted
        else
            assert(false);

        return batchSize * elements * bytesPerElement;
    }

   private:
    string name;
    vector<uint64_t> dimensions;
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
            networkInput.name = string("NetworkInput") + to_string(networkInput.getId());
        networkInput.dimensions = _dimensions;
        networkInput.dataType = _dataType;
        networkInput.featureInput = Tensor(_dataType, _dimensions);
        networkInput.featureOutput = Tensor(Tensor::DataType::FP16, _dimensions);
        networkInput.initialized = true;
        networkInput.addToNetwork(_network.get());
        return networkInput;
    }

    virtual NetworkInput::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual NetworkInput::Builder &name(string _name) {
        assert(!_name.empty());
        assert(this->_name.isEmpty());
        this->_name = _name;
        return *this;
    }

    // Note: dimensions do not include batch size
    virtual NetworkInput::Builder &dimensions(vector<uint64_t> _dimensions) {
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
    Optional<string> _name;
    Optional<Network *> _network;
    Optional<vector<uint64_t>> _dimensions;
    Optional<Tensor::DataType> _dataType;
};

}  // namespace Thor
