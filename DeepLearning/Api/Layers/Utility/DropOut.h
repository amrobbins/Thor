#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/TrainingDropoutControllable.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

namespace Thor {

class DropOut : public Layer, public TrainingDropoutControllable {
   public:
    class Builder;
    ~DropOut() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<DropOut>(*this); }

    virtual float getDropProportion() { return dropProportion; }

    std::string getLayerType() const override { return "DropOut"; }

    [[nodiscard]] bool getUseRagged() const { return raggedFeatureInput.has_value(); }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    std::vector<Tensor> getAllInputTensors() const override {
        if (!raggedFeatureInput.has_value()) return Layer::getAllInputTensors();
        return {raggedFeatureInput->getValues(), raggedFeatureInput->getOffsets()};
    }
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() const override { return raggedFeatureInput.has_value(); }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    int getConnectionType(Tensor connectingTensor) const override;

    [[nodiscard]] uint64_t getOutputTensorBytes(uint32_t batchSize) const override {
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        if (raggedFeatureInput.has_value()) return featureOutput->getTotalSizeInBytes();
        return featureOutput->getTotalSizeInBytes() * batchSize;
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)drivingLayer;
        (void)drivingApiLayer;
        THOR_THROW_IF_FALSE(initialized);
        bool knownInput = connectingApiTensor == getFeatureInput().value();
        if (raggedFeatureInput.has_value() && connectingApiTensor == raggedFeatureInput->getOffsets()) knownInput = true;
        THOR_THROW_IF_FALSE(knownInput);

        std::optional<ThorImplementation::DropOut::RaggedConfiguration> raggedConfiguration;
        if (raggedFeatureInput.has_value()) {
            uint64_t elementsPerValue = 1;
            for (uint64_t dim : raggedFeatureInput->getTrailingDimensions()) {
                THOR_THROW_IF_FALSE(dim > 0);
                if (elementsPerValue > std::numeric_limits<uint64_t>::max() / dim)
                    throw std::overflow_error("Ragged DropOut elements-per-value overflow.");
                elementsPerValue *= dim;
            }
            raggedConfiguration = ThorImplementation::DropOut::RaggedConfiguration{
                raggedFeatureInput->getMaxTotalValues(), elementsPerValue};
        }

        // An inference-only placement never applies dropout. A zero-rate layer is
        // also stamped as a metadata-only identity so it can remain in the API
        // graph without allocating tensors or launching kernels.
        return std::make_shared<ThorImplementation::DropOut>(
            dropProportion, !inferenceOnly, isTrainingDropoutEnabled(), raggedConfiguration);
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        if (dropProportion == 0.0f && !raggedFeatureInput.has_value())
            return 0;
        THOR_THROW_IF_FALSE(tensorPlacement.getMemDevice() == ThorImplementation::TensorPlacement::MemDevices::GPU);
        const ThorImplementation::DataType dataType = featureInput.value().getDataType();
        THOR_THROW_IF_FALSE(ThorImplementation::DropOut::nativeKernelSupportsDataType(dataType));
        const uint64_t randomStateSize = 0;

        uint64_t featureOutputSize = featureOutput.value().getTotalSizeInBytes();
        uint64_t errorOutputSize = featureInput.value().getTotalSizeInBytes();
        const uint64_t tensorMultiplier = raggedFeatureInput.has_value() ? 1 : batchSize;

        const uint64_t reserveStateSize = dropProportion == 0.0f ? 0 : getReservedStateSizeInBytes(batchSize);
        return randomStateSize + reserveStateSize + tensorMultiplier * (featureOutputSize + errorOutputSize);
    }

   protected:
    virtual uint64_t getReservedStateSizeInBytes(uint32_t batchSize) const {
        (void)batchSize;
        THOR_THROW_IF_FALSE(featureInput.has_value());
        const ThorImplementation::DataType dataType = featureInput.value().getDataType();
        THOR_THROW_IF_FALSE(ThorImplementation::DropOut::nativeKernelSupportsDataType(dataType));
        // Native Philox dropout regenerates the forward mask during backward.
        return 0;
    }

   private:
    float dropProportion;
    std::optional<RaggedTensor> raggedFeatureInput;
    std::optional<RaggedTensor> raggedFeatureOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;
};

class DropOut::Builder {
   public:
    virtual DropOut build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());
        THOR_THROW_IF_FALSE(_dropProportion.has_value());

        if (!ThorImplementation::DropOut::nativeKernelSupportsDataType(_featureInput->getDataType())) {
            throw std::invalid_argument("DropOut supports FP16, FP32, FP64, and BF16 values.");
        }

        DropOut dropOut;
        dropOut.featureInput = _featureInput;
        dropOut.featureOutput = _featureInput.value().clone();
        if (_raggedFeatureInput.has_value()) {
            dropOut.raggedFeatureInput = _raggedFeatureInput.value();
            dropOut.raggedFeatureOutput = RaggedTensor(dropOut.featureOutput.value(), _raggedFeatureInput->getOffsets());
        }
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
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual DropOut::Builder &featureInput(RaggedTensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        this->_raggedFeatureInput = _featureInput;
        this->_featureInput = _featureInput.getValues();
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
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::optional<float> _dropProportion;
};

}  // namespace Thor
