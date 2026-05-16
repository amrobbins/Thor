#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Thor {

class RMSNorm : public TrainableLayer {
   public:
    class Builder;

    RMSNorm() = default;
    explicit RMSNorm(std::optional<ThorImplementation::Expression> epilogue) : epilogue(std::move(epilogue)) {}
    ~RMSNorm() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<RMSNorm>(*this); }

    std::string getLayerType() const override { return "RMSNorm"; }

    const std::vector<uint64_t>& getNormalizedShape() const { return normalizedShape; }
    double getEpsilon() const { return epsilon; }
    Tensor::DataType getParameterDataType() const { return parameterDataType; }
    static const char* epilogueInputName() { return "__rms_norm_epilogue_input"; }
    static const char* epilogueOutputName() { return "__rms_norm_epilogue_output"; }

    [[nodiscard]] static ThorImplementation::Expression epilogueInput(
        std::optional<ThorImplementation::TensorDescriptor::DataType> computeDType = std::nullopt,
        std::optional<ThorImplementation::TensorDescriptor::DataType> outputDType = std::nullopt) {
        return LayerEpilogue::input(epilogueInputName(), computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(const ThorImplementation::Expression& expression) {
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), epilogueOutputName(), "RMSNorm");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression& expression) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), epilogueOutputName(), "RMSNorm");
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition& definition) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), epilogueOutputName(), "RMSNorm");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition& definition) {
        return LayerEpilogue::expressionFromDefinition(definition, epilogueInputName(), epilogueOutputName(), "RMSNorm");
    }

    [[nodiscard]] static ThorImplementation::Expression applyEpilogue(const ThorImplementation::Expression& input,
                                                                      const ThorImplementation::Expression& epilogue) {
        return LayerEpilogue::apply(input, epilogue, epilogueInputName());
    }

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const nlohmann::json& j, Network* network);
    nlohmann::json architectureJson() const override;

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

   private:
    static bool isRMSNormInputDataType(Tensor::DataType dataType);
    static uint64_t checkedFeatureCount(const std::vector<uint64_t>& shape, const std::string& what);
    static void validateNormalizedShapeForInput(const std::vector<uint64_t>& inputDims, const std::vector<uint64_t>& normalizedShape);

    std::vector<uint64_t> normalizedShape;
    double epsilon = 1.0e-5;
    Tensor::DataType parameterDataType = Tensor::DataType::FP32;
    std::optional<ThorImplementation::Expression> epilogue;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;

    friend class Network;
    friend class Builder;
};

class RMSNorm::Builder {
   public:
    virtual ~Builder() = default;

    virtual RMSNorm build();

    virtual RMSNorm::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual RMSNorm::Builder& featureInput(Tensor featureInput) {
        THOR_THROW_IF_FALSE(featureInput.isInitialized());
        this->_featureInputs.push_back(featureInput);
        if (_featureInputs.size() > 1) {
            THOR_THROW_IF_FALSE(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            THOR_THROW_IF_FALSE(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual RMSNorm::Builder& normalizedShape(const std::vector<uint64_t>& shape) {
        if (!this->_normalizedShape.empty()) {
            throw std::invalid_argument("RMSNorm normalizedShape may only be set once.");
        }
        RMSNorm::checkedFeatureCount(shape, "normalizedShape");
        this->_normalizedShape = shape;
        return *this;
    }

    virtual RMSNorm::Builder& epsilon(double epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = epsilon;
        return *this;
    }

    virtual RMSNorm::Builder& parameterDataType(Tensor::DataType dtype) {
        THOR_THROW_IF_FALSE(!this->_parameterDataType.has_value());
        this->_parameterDataType = dtype;
        return *this;
    }

    virtual RMSNorm::Builder& weightsInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = initializer;
        return *this;
    }

    virtual RMSNorm::Builder& weightsOptimizer(std::shared_ptr<Optimizer> optimizer) {
        THOR_THROW_IF_FALSE(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = optimizer;
        return *this;
    }

    virtual RMSNorm::Builder& epilogue(const ThorImplementation::Expression& expression) {
        THOR_THROW_IF_FALSE(!this->_epilogue.has_value());
        RMSNorm::validateEpilogueExpression(expression);
        this->_epilogue = expression;
        return *this;
    }

    virtual RMSNorm::Builder& epilogue(const Activation& activation) { return epilogue(activation.toExpression(RMSNorm::epilogueInput())); }

    virtual RMSNorm::Builder& epilogue(const std::shared_ptr<Activation>& activation) {
        if (activation == nullptr) {
            throw std::invalid_argument("RMSNorm epilogue activation must be non-null.");
        }
        return epilogue(*activation);
    }

   private:
    void verifyConfig() const;

    std::optional<Network*> _network;
    std::vector<Tensor> _featureInputs;
    std::vector<uint64_t> _normalizedShape;
    std::optional<double> _epsilon;
    std::optional<Tensor::DataType> _parameterDataType;
    std::shared_ptr<Initializer> _weightsInitializer = nullptr;
    std::shared_ptr<Optimizer> _weightsOptimizer = nullptr;
    std::optional<ThorImplementation::Expression> _epilogue;
};

}  // namespace Thor
