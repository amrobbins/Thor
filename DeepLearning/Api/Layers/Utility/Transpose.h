#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <algorithm>
#include <optional>
#include <vector>

namespace Thor {

class Transpose : public Layer {
   public:
    class Builder;
    Transpose();
    ~Transpose() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Transpose>(*this); }

    std::string getLayerType() const override { return "Transpose"; }

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
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());

        using ThorImplementation::DynamicExpression;
        using ThorImplementation::Expression;
        using ThorImplementation::ExpressionDefinition;

        Expression featureInput = Expression::input("feature_input");
        Expression featureOutput = featureInput.transpose();
        ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", featureOutput}}));

        std::shared_ptr<ThorImplementation::CustomLayer> physicalTranspose = std::make_shared<ThorImplementation::CustomLayer>(
            DynamicExpression::fromExpressionDefinition(definition),
            std::vector<std::string>{"feature_input"},
            std::vector<std::string>{"feature_output"},
            placement,
            std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{},
            inferenceOnly);
        physicalTranspose->setLayerName("Transpose");
        return physicalTranspose;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)batchSize;
        (void)tensorPlacement;
        THOR_THROW_IF_FALSE(getFeatureOutput().has_value());
        return getFeatureOutput().value().getTotalSizeInBytes();
    }
};

class Transpose::Builder {
   public:
    virtual Transpose build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());

        std::vector<uint64_t> outputDimensions = _featureInput.value().getDimensions();
        THOR_THROW_IF_FALSE(outputDimensions.size() >= 2);
        std::swap(outputDimensions[outputDimensions.size() - 2], outputDimensions[outputDimensions.size() - 1]);

        Transpose transpose;
        transpose.featureInput = _featureInput.value();
        transpose.featureOutput = Tensor(_featureInput.value().getDataType(), outputDimensions);
        transpose.initialized = true;
        transpose.addToNetwork(_network.value());
        return transpose;
    }

    virtual Transpose::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Transpose::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        this->_featureInput = _featureInput;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
};

}  // namespace Thor
