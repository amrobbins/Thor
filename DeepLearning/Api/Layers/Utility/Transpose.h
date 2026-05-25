#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Thor {

class Transpose : public Layer {
   public:
    class Builder;
    Transpose();
    explicit Transpose(std::optional<ThorImplementation::Expression> epilogue);
    ~Transpose() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Transpose>(*this); }

    std::string getLayerType() const override { return "Transpose"; }
    Tensor::DataType getOutputDataType() const { return outputDataType; }

    static const char *epilogueInputName() { return "__transpose_epilogue_input"; }
    static const char *epilogueOutputName() { return "__transpose_epilogue_output"; }

    [[nodiscard]] static ThorImplementation::Expression epilogueInput(
        std::optional<ThorImplementation::DataType> computeDType = std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType = std::nullopt) {
        return LayerEpilogue::input(epilogueInputName(), computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(
        const ThorImplementation::Expression &expression) {
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), epilogueOutputName(), "Transpose");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression &expression) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), epilogueOutputName(), "Transpose");
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), epilogueOutputName(), "Transpose");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition &definition) {
        return LayerEpilogue::expressionFromDefinition(definition, epilogueInputName(), epilogueOutputName(), "Transpose");
    }

    [[nodiscard]] static ThorImplementation::Expression applyEpilogue(const ThorImplementation::Expression &input,
                                                                      const ThorImplementation::Expression &epilogue) {
        return LayerEpilogue::apply(input, epilogue, epilogueInputName());
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
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());

        using ThorImplementation::DynamicExpression;
        using ThorImplementation::Expression;
        using ThorImplementation::ExpressionDefinition;

        Expression featureInput = Expression::input("feature_input");
        Expression featureOutput = featureInput.transpose().withOutputDType(outputDataType);
        if (epilogue.has_value()) {
            featureOutput = Transpose::applyEpilogue(featureOutput, epilogue.value()).withOutputDType(outputDataType);
        }
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

   private:
    Tensor::DataType outputDataType = Tensor::DataType::FP32;
    std::optional<ThorImplementation::Expression> epilogue;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;

    friend class Builder;
};

class Transpose::Builder {
   public:
    virtual Transpose build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());
        if (_epilogue.has_value()) {
            Transpose::validateEpilogueExpression(_epilogue.value());
        }

        std::vector<uint64_t> outputDimensions = _featureInput.value().getDimensions();
        THOR_THROW_IF_FALSE(outputDimensions.size() >= 2);
        std::swap(outputDimensions[outputDimensions.size() - 2], outputDimensions[outputDimensions.size() - 1]);

        Transpose transpose(_epilogue);
        transpose.outputDataType = _outputDataType.value_or(_featureInput.value().getDataType());
        transpose.featureInput = _featureInput.value();
        transpose.featureOutput = Tensor(transpose.outputDataType, outputDimensions);
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

    virtual Transpose::Builder &outputDataType(Tensor::DataType dataType) {
        THOR_THROW_IF_FALSE(!this->_outputDataType.has_value());
        if (!Tensor::dataTypeValid(dataType)) {
            throw std::invalid_argument("Transpose outputDataType is invalid.");
        }
        this->_outputDataType = dataType;
        return *this;
    }

    virtual Transpose::Builder &epilogue(const ThorImplementation::Expression &expression) {
        THOR_THROW_IF_FALSE(!this->_epilogue.has_value());
        Transpose::validateEpilogueExpression(expression);
        this->_epilogue = expression;
        return *this;
    }

    virtual Transpose::Builder &epilogue(const Activation &activation) { return epilogue(activation.toExpression(Transpose::epilogueInput())); }

    virtual Transpose::Builder &epilogue(const std::shared_ptr<Activation> &activation) {
        if (activation == nullptr) {
            throw std::invalid_argument("Transpose epilogue activation must be non-null.");
        }
        return epilogue(*activation);
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<Tensor::DataType> _outputDataType;
    std::optional<ThorImplementation::Expression> _epilogue;
};

}  // namespace Thor
