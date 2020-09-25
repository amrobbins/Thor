#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"

namespace Thor {

class NetworkInput : public Layer {
   public:
    class Builder;

    NetworkInput() {}

    virtual ~NetworkInput() {}

    vector<uint64_t> getDimensions() const { return dimensions; }
    Tensor::DataType getDataType() const { return dataType; }

    virtual shared_ptr<Layer> clone() const { return make_shared<NetworkInput>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement, uint32_t batchSize) const {
        assert(initialized);

        ThorImplementation::TensorDescriptor::DataType implementationDataType;
        if (dataType == Tensor::DataType::FP32)
            implementationDataType = ThorImplementation::TensorDescriptor::DataType::FP32;
        else
            implementationDataType = ThorImplementation::TensorDescriptor::DataType::FP16;

        vector<uint64_t> batchDimensions;
        batchDimensions.push_back(batchSize);
        for (uint32_t i = 0; i < dimensions.size(); ++i)
            batchDimensions.push_back(dimensions[i]);

        return new ThorImplementation::NetworkInput(placement, implementationDataType, batchDimensions);
    }

   private:
    vector<uint64_t> dimensions;
    Tensor::DataType dataType;
};

class NetworkInput::Builder {
   public:
    virtual NetworkInput build() {
        assert(_network.isPresent());
        assert(_dimensions.isPresent());
        assert(_dataType.isPresent());

        NetworkInput networkInput;
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

    // Note: dimensions do not include batch size
    virtual NetworkInput::Builder &dimensions(vector<uint64_t> _dimensions) {
        assert(!_dimensions.empty());
        this->_dimensions = _dimensions;
        return *this;
    }

    virtual NetworkInput::Builder &dataType(Tensor::DataType _dataType) {
        assert(Tensor::dataTypeValid(_dataType));
        this->_dimensions = _dimensions;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<vector<uint64_t>> _dimensions;
    Optional<Tensor::DataType> _dataType;
};

}  // namespace Thor
