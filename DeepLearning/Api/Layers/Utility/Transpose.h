#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
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
    DataType getOutputDataType() const { return outputDataType; }
    [[nodiscard]] bool getUseRagged() const { return raggedFeatureInput.has_value(); }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureOutput() const { return raggedFeatureOutput; }

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

    [[nodiscard]] bool outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const override {
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(outputTensor == featureOutput.value());
        return raggedFeatureInput.has_value();
    }

    bool mustConnectAllInputsToDriveOutput() const override { return raggedFeatureInput.has_value(); }
    std::vector<Tensor> getAllInputTensors() const override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    int getConnectionType(Tensor connectingTensor) const override;

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)batchSize;
        (void)tensorPlacement;
        THOR_THROW_IF_FALSE(getFeatureOutput().has_value());
        return getFeatureOutput().value().getTotalSizeInBytes();
    }

   private:
    DataType outputDataType = DataType::FP32;
    std::optional<ThorImplementation::Expression> epilogue;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;
    std::optional<RaggedTensor> raggedFeatureInput;
    std::optional<RaggedTensor> raggedFeatureOutput;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;

    friend class Builder;
};

class Transpose::Builder {
   public:
    virtual Transpose build();

    virtual Transpose::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Transpose::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Transpose::Builder &featureInput(RaggedTensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        this->_raggedFeatureInput = _featureInput;
        this->_featureInput = _featureInput.getValues();
        return *this;
    }

    virtual Transpose::Builder &outputDataType(DataType dataType) {
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

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::optional<DataType> _outputDataType;
    std::optional<ThorImplementation::Expression> _epilogue;
};

}  // namespace Thor
