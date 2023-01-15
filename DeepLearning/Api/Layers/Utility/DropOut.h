#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/CudnnHelper.h"

namespace Thor {

class DropOut : public Layer {
   public:
    class Builder;
    virtual ~DropOut() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<DropOut>(*this); }

    virtual float getDropProportion() { return dropProportion; }

    virtual std::string getLayerType() const { return "DropOut"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        ThorImplementation::DropOut *dropOut = new ThorImplementation::DropOut(dropProportion, true);
        return dropOut;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        assert(tensorPlacement.getMemDevice() == ThorImplementation::TensorPlacement::MemDevices::GPU);
        uint32_t gpuNum = tensorPlacement.getDeviceNum();
        cudnnHandle_t cudnnHandle = ThorImplementation::CudnnHelper::getCudnnHandle(gpuNum);
        uint64_t randomStateSize = ThorImplementation::DropOut::getRandomStateSizeInBytes(cudnnHandle);

        uint64_t featureOutputSize = featureOutput.get().getTotalSizeInBytes();
        uint64_t errorOutputSize = featureInput.get().getTotalSizeInBytes();

        return randomStateSize + getReservedStateSizeInBytes(batchSize) + batchSize * (featureOutputSize + errorOutputSize);
    }

   protected:
    virtual uint64_t getReservedStateSizeInBytes(uint32_t batchSize) const {
        ThorImplementation::TensorDescriptor::DataType dataType = ThorImplementation::TensorDescriptor::DataType::FP16;
        std::vector<uint64_t> featureInputDimensionsWithBatchSize;
        featureInputDimensionsWithBatchSize.push_back(batchSize);
        for (uint32_t i = 0; i < featureInput.get().getDimensions().size(); ++i)
            featureInputDimensionsWithBatchSize.push_back(featureInput.get().getDimensions()[i]);
        return ThorImplementation::DropOut::getReservedSpaceSizeInBytes(featureInputDimensionsWithBatchSize, dataType);
    }

   private:
    float dropProportion;
};

class DropOut::Builder {
   public:
    virtual DropOut build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_dropProportion.isPresent());

        DropOut dropOut;
        dropOut.featureInput = _featureInput;
        dropOut.featureOutput = _featureInput.get().clone();
        dropOut.dropProportion = _dropProportion;
        dropOut.initialized = true;
        dropOut.addToNetwork(_network.get());
        return dropOut;
    }

    virtual DropOut::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual DropOut::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual DropOut::Builder &dropProportion(float _dropProportion) {
        assert(!this->_dropProportion.isPresent());
        assert(_dropProportion > 0.0);
        assert(_dropProportion <= 1.0);
        this->_dropProportion = _dropProportion;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<float> _dropProportion;
};

}  // namespace Thor
