#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

namespace Thor {

class NetworkOutput : public Layer {
   public:
    NetworkOutput() { initialized = false; }

    virtual ~NetworkOutput() {}

    class Builder;

    virtual shared_ptr<Layer> clone() const { return make_shared<NetworkOutput>(*this); }

    Tensor::DataType getDataType() const { return dataType; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement, uint32_t batchSize) const {
        assert(initialized);

        vector<uint64_t> batchDimensions;
        batchDimensions.push_back(batchSize);
        vector<uint64_t> dimensions = featureInput.get().getDimensions();
        for (uint32_t i = 0; i < dimensions.size(); ++i)
            batchDimensions.push_back(dimensions[i]);

        return new ThorImplementation::NetworkOutput(placement);
    }

   private:
    bool initialized;
    Tensor::DataType dataType;
};

class NetworkOutput::Builder {
   public:
    virtual NetworkOutput build() {
        assert(!_inputTensor.isEmpty());
        if (_dataType.isEmpty())
            _dataType = Tensor::DataType::FP32;

        NetworkOutput networkOutput;
        networkOutput.dataType = _dataType;
        networkOutput.featureInput = _inputTensor;
        networkOutput.featureOutput = Tensor(_dataType, _inputTensor.get().getDimensions());
        networkOutput.initialized = true;
        return networkOutput;
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
    Optional<Tensor> _inputTensor;
    Optional<Tensor::DataType> _dataType;
};

}  // namespace Thor
