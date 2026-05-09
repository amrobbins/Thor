#pragma once
#include "DeepLearning/Implementation/ThorError.h"

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

    virtual nlohmann::json architectureJson() const;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput());

        std::shared_ptr<ThorImplementation::DropOut> dropOut = std::make_shared<ThorImplementation::DropOut>(dropProportion, true);
        return dropOut;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        THOR_THROW_IF_FALSE(tensorPlacement.getMemDevice() == ThorImplementation::TensorPlacement::MemDevices::GPU);
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
        THOR_THROW_IF_FALSE(_network.isPresent());
        THOR_THROW_IF_FALSE(_featureInput.isPresent());
        THOR_THROW_IF_FALSE(_dropProportion.isPresent());

        DropOut dropOut;
        dropOut.featureInput = _featureInput;
        dropOut.featureOutput = _featureInput.get().clone();
        dropOut.dropProportion = _dropProportion;
        dropOut.initialized = true;
        dropOut.addToNetwork(_network.get());
        return dropOut;
    }

    virtual DropOut::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual DropOut::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual DropOut::Builder &dropProportion(float _dropProportion) {
        THOR_THROW_IF_FALSE(!this->_dropProportion.isPresent());
        THOR_THROW_IF_FALSE(_dropProportion >= 0.0);
        THOR_THROW_IF_FALSE(_dropProportion <= 1.0);
        this->_dropProportion = _dropProportion;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<float> _dropProportion;
};

}  // namespace Thor
