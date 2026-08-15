#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include <optional>
#include <set>

namespace Thor {

class TypeConverter : public MultiConnectionLayer {
   public:
    class Builder;
    TypeConverter();

    ~TypeConverter() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<TypeConverter>(*this); }

    std::string getLayerType() const override { return "TypeConverter"; }

    std::optional<Tensor> getFeatureInput() const override {
        if (featureInputs.empty()) return std::nullopt;
        return featureInputs.front();
    }
    std::optional<Tensor> getFeatureOutput() const override {
        if (featureOutputs.empty()) return std::nullopt;
        return featureOutputs.front();
    }

    [[nodiscard]] bool getUseRagged() const { return raggedFeatureInput.has_value(); }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    [[nodiscard]] bool outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const override {
        THOR_THROW_IF_FALSE(!featureOutputs.empty());
        THOR_THROW_IF_FALSE(outputTensor == featureOutputs.front());
        return dimensionsIncludeBatch_.value_or(false);
    }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() const override { return raggedFeatureInput.has_value(); }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    int getConnectionType(Tensor connectingTensor) const override;

    [[nodiscard]] uint64_t getOutputTensorBytes(uint32_t batchSize) const override;

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                   ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    std::optional<RaggedTensor> raggedFeatureInput;
    std::optional<RaggedTensor> raggedFeatureOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;
    // Derived during placement from the semantic API producer. This is a placement
    // contract rather than persisted model state, so every stamp recomputes it.
    mutable std::optional<bool> dimensionsIncludeBatch_;

    friend class Builder;
};

class TypeConverter::Builder {
   public:
    virtual TypeConverter build();

    virtual TypeConverter::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual TypeConverter::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual TypeConverter::Builder &featureInput(RaggedTensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        this->_raggedFeatureInput = _featureInput;
        this->_featureInput = _featureInput.getValues();
        return *this;
    }

    virtual TypeConverter::Builder &newDataType(DataType _newDataType) {
        THOR_THROW_IF_FALSE(!this->_newDataType.has_value());
        THOR_THROW_IF_FALSE(Tensor::dataTypeValid(_newDataType));
        this->_newDataType = _newDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::optional<DataType> _newDataType;
};

}  // namespace Thor
