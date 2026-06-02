#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"

namespace Thor {

class HardTanh : public Activation {
   public:
    class Builder;
    HardTanh(double minValue = -1.0, double maxValue = 1.0) : minValue(minValue), maxValue(maxValue) {
        THOR_THROW_IF_FALSE(minValue <= maxValue);
    }

    ~HardTanh() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<HardTanh> myClone = std::make_shared<HardTanh>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.hardTanh(minValue, maxValue);

    }

    std::string getLayerType() const override { return "HardTanh"; }

    nlohmann::json architectureJson() const override {
        nlohmann::json j = Activation::architectureJson();
        j["min_value"] = minValue;
        j["max_value"] = maxValue;
        return j;
    }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in HardTanh::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "hard_tanh")
            throw std::runtime_error("Layer type mismatch in HardTanh::deserialize: " + j.at("layer_type").get<std::string>());
        double minValue = j.value("min_value", -1.0);
        double maxValue = j.value("max_value", 1.0);
        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        HardTanh activation(minValue, maxValue);
        activation.featureInput = featureInput;
        activation.featureOutput = featureOutput;
        activation.initialized = true;
        activation.addToNetwork(network);
    }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)drivingLayer;
        (void)drivingApiLayer;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());

        using ThorImplementation::DynamicExpression;
        using ThorImplementation::Expression;
        using ThorImplementation::ExpressionDefinition;

        Expression featureInputExpr = Expression::input("feature_input");
        Expression featureOutputExpr = toExpression(featureInputExpr);
        ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"feature_output", featureOutputExpr}}));

        std::shared_ptr<ThorImplementation::CustomLayer> physicalActivation = std::make_shared<ThorImplementation::CustomLayer>(
            DynamicExpression::fromExpressionDefinition(definition),
            std::vector<std::string>{"feature_input"},
            std::vector<std::string>{"feature_output"},
            placement,
            std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{},
            inferenceOnly);
        physicalActivation->setLayerName("HardTanh");
        return physicalActivation;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)tensorPlacement;
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }

    const double minValue;
    const double maxValue;
};

class HardTanh::Builder : public Activation::Builder {
   public:
    virtual HardTanh::Builder& minValue(double minValue) {
        THOR_THROW_IF_FALSE(!this->_minValue.has_value());
        this->_minValue = minValue;
        return *this;
    }

    virtual HardTanh::Builder& maxValue(double maxValue) {
        THOR_THROW_IF_FALSE(!this->_maxValue.has_value());
        this->_maxValue = maxValue;
        return *this;
    }

    std::shared_ptr<Activation> build() override {
        std::shared_ptr<HardTanh> activation = std::make_shared<HardTanh>(_minValue.value_or(-1.0), _maxValue.value_or(1.0));
        if (_featureInput.has_value()) {
            THOR_THROW_IF_FALSE(_network.has_value());
            activation->featureInput = _featureInput;
            activation->featureOutput = _featureInput.value().clone();
            activation->initialized = true;
            activation->addToNetwork(_network.value());
        } else {
            activation->initialized = true;
        }

        return activation;
    }

    HardTanh::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    HardTanh::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

   private:
    std::optional<double> _minValue;
    std::optional<double> _maxValue;
};

}  // namespace Thor
