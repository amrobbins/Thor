#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"

namespace Thor {

class Threshold : public Activation {
   public:
    class Builder;
    Threshold(double thresholdValue = 0.0, double value = 0.0) : thresholdValue(thresholdValue), value(value) {}

    ~Threshold() override {}

    std::shared_ptr<Layer> clone() const override {
        std::shared_ptr<Threshold> myClone = std::make_shared<Threshold>(*this);
        myClone->id = getUnusedId();
        return myClone;
    }

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        return input.threshold(thresholdValue, value);

    }

    std::string getLayerType() const override { return "Threshold"; }

    nlohmann::json architectureJson() const override {
        nlohmann::json j = Activation::architectureJson();
        j["threshold"] = thresholdValue;
        j["value"] = value;
        return j;
    }

    static void deserialize(const nlohmann::json &j, Network *network) {
        if (j.at("version").get<std::string>() != "1.0.0")
            throw std::runtime_error("Unsupported version in Threshold::deserialize: " + j["version"].get<std::string>());
        if (j.at("layer_type").get<std::string>() != "threshold")
            throw std::runtime_error("Layer type mismatch in Threshold::deserialize: " + j.at("layer_type").get<std::string>());
        double thresholdValue = j.value("threshold", 0.0);
        double value = j.value("value", 0.0);
        nlohmann::json input = j["feature_input"].get<nlohmann::json>();
        uint64_t originalTensorId = input.at("id").get<uint64_t>();
        Tensor featureInput = network->getApiTensorByOriginalId(originalTensorId);

        Tensor featureOutput = Tensor::deserialize(j.at("feature_output").get<nlohmann::json>());

        Threshold activation(thresholdValue, value);
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
        physicalActivation->setLayerName("Threshold");
        return physicalActivation;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)tensorPlacement;
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }

    const double thresholdValue;
    const double value;
};

class Threshold::Builder : public Activation::Builder {
   public:
    virtual Threshold::Builder& threshold(double thresholdValue) {
        THOR_THROW_IF_FALSE(!this->_threshold.has_value());
        this->_threshold = thresholdValue;
        return *this;
    }

    virtual Threshold::Builder& value(double value) {
        THOR_THROW_IF_FALSE(!this->_value.has_value());
        this->_value = value;
        return *this;
    }

    std::shared_ptr<Activation> build() override {
        std::shared_ptr<Threshold> activation = std::make_shared<Threshold>(_threshold.value_or(0.0), _value.value_or(0.0));
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

    Threshold::Builder &network(Network &_network) override {
        Activation::Builder::network(_network);
        return *this;
    }

    Threshold::Builder &featureInput(Tensor _featureInput) override {
        Activation::Builder::featureInput(_featureInput);
        return *this;
    }

   private:
    std::optional<double> _threshold;
    std::optional<double> _value;
};

}  // namespace Thor
