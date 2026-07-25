#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Common/CudnnHelper.h"
#include <optional>

namespace Thor {

class DropOut : public Layer {
   public:
    class Builder;
    ~DropOut() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<DropOut>(*this); }

    virtual float getDropProportion() { return dropProportion; }

    std::string getLayerType() const override { return "DropOut"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());

        // An inference-only placement never applies dropout.  A zero-rate layer is
        // also stamped as a metadata-only identity so it can remain in the API
        // graph without allocating tensors or launching kernels.
        return std::make_shared<ThorImplementation::DropOut>(dropProportion, !inferenceOnly);
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        if (dropProportion == 0.0f)
            return 0;
        THOR_THROW_IF_FALSE(tensorPlacement.getMemDevice() == ThorImplementation::TensorPlacement::MemDevices::GPU);
        const ThorImplementation::DataType dataType = featureInput.value().getDataType();
        uint64_t randomStateSize = 0;
        if (!ThorImplementation::DropOut::usesNativeKernel(dataType)) {
            uint32_t gpuNum = tensorPlacement.getDeviceNum();
            cudnnHandle_t cudnnHandle = ThorImplementation::CudnnHelper::getCudnnHandle(gpuNum);
            randomStateSize = ThorImplementation::DropOut::getRandomStateSizeInBytes(cudnnHandle);
        }

        uint64_t featureOutputSize = featureOutput.value().getTotalSizeInBytes();
        uint64_t errorOutputSize = featureInput.value().getTotalSizeInBytes();

        return randomStateSize + getReservedStateSizeInBytes(batchSize) + batchSize * (featureOutputSize + errorOutputSize);
    }

   protected:
    virtual uint64_t getReservedStateSizeInBytes(uint32_t batchSize) const {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        ThorImplementation::DataType dataType = featureInput.value().getDataType();
        std::vector<uint64_t> featureInputDimensionsWithBatchSize;
        featureInputDimensionsWithBatchSize.push_back(batchSize);
        for (uint32_t i = 0; i < featureInput.value().getDimensions().size(); ++i)
            featureInputDimensionsWithBatchSize.push_back(featureInput.value().getDimensions()[i]);
        if (ThorImplementation::DropOut::usesNativeKernel(dataType))
            return ThorImplementation::DropOut::getNativeReserveSpaceSizeInBytes(featureInputDimensionsWithBatchSize);
        return ThorImplementation::DropOut::getReservedSpaceSizeInBytes(featureInputDimensionsWithBatchSize, dataType);
    }

   private:
    float dropProportion;
};

class DropOut::Builder {
   public:
    virtual DropOut build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());
        THOR_THROW_IF_FALSE(_dropProportion.has_value());

        DropOut dropOut;
        dropOut.featureInput = _featureInput;
        dropOut.featureOutput = _featureInput.value().clone();
        dropOut.dropProportion = _dropProportion.value();
        dropOut.initialized = true;
        dropOut.addToNetwork(_network.value());
        return dropOut;
    }

    virtual DropOut::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual DropOut::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual DropOut::Builder &dropProportion(float _dropProportion) {
        THOR_THROW_IF_FALSE(!this->_dropProportion.has_value());
        THOR_THROW_IF_FALSE(_dropProportion >= 0.0f);
        THOR_THROW_IF_FALSE(_dropProportion <= 1.0f);
        this->_dropProportion = _dropProportion;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<float> _dropProportion;
};

}  // namespace Thor
