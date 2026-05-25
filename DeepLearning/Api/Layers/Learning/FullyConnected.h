#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/Expression.h"


#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <optional>

namespace Thor {

class FullyConnected : public TrainableLayer {
   public:
    class Builder;

    FullyConnected(const std::optional<ThorImplementation::Expression> epilogue) : epilogue(epilogue) {}

    ~FullyConnected() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<FullyConnected>(*this); }

    nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork &stampedNetwork) const override;

    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);

    nlohmann::json architectureJson() const override;

    static const char *epilogueInputName() { return "__fully_connected_epilogue_input"; }
    static const char *epilogueOutputName() { return "__fully_connected_epilogue_output"; }

    [[nodiscard]] static ThorImplementation::Expression epilogueInput(
        std::optional<ThorImplementation::DataType> computeDType =
            std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType =
            std::nullopt) {
        return LayerEpilogue::input(epilogueInputName(), computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(const ThorImplementation::Expression &expression) {
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), epilogueOutputName(), "FullyConnected");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression &expression) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), epilogueOutputName(), "FullyConnected");
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), epilogueOutputName(), "FullyConnected");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition &definition) {
        return LayerEpilogue::expressionFromDefinition(definition, epilogueInputName(), epilogueOutputName(), "FullyConnected");
    }

    [[nodiscard]] static ThorImplementation::Expression applyEpilogue(const ThorImplementation::Expression &input,
                                                                      const ThorImplementation::Expression &epilogue) {
        return LayerEpilogue::apply(input, epilogue, epilogueInputName());
    }

    int getConnectionType(Tensor connectingTensor) const override {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (connectingTensor == featureInputs[i])
                return static_cast<int>(i);
        }

        for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
            if (connectingTensor == featureOutputs[i])
                return static_cast<int>(i);
        }

        throw std::runtime_error("Tensor is not connected to this FullyConnected layer.");
    }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;


    std::string getLayerType() const override { return "FullyConnected"; }

   private:
    uint32_t numOutputFeatures;
    bool hasBias;
    std::shared_ptr<Activation> activation;
    DataType weightsDataType;
    DataType computeDataType;
    DataType outputDataType;

    const std::optional<ThorImplementation::Expression> epilogue;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;

    static bool isFullyConnectedFloatingDataType(DataType dataType);
    static std::string dataTypeName(DataType dataType);
    static uint64_t checkedFeatureCount(const std::vector<uint64_t> &dimensions, const std::string &what);
    static void verifyFullyConnectedDataType(DataType dataType, const std::string &what);
    static void verifyFullyConnectedComputeDataType(DataType dataType);

    friend class Network;
    friend class Builder;
};

// featureInput and numOutputFeatures are required, all other parameters are optional.
class FullyConnected::Builder {
   public:
    virtual ~Builder() = default;

    Builder() { _activationExplicitlyRemoved = false; }

    virtual FullyConnected build();

    virtual FullyConnected::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual FullyConnected::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!_featureInput.getDimensions().empty());
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            THOR_THROW_IF_FALSE(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            THOR_THROW_IF_FALSE(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual FullyConnected::Builder &numOutputFeatures(uint32_t _numOutputFeatures) {
        THOR_THROW_IF_FALSE(!this->_numOutputFeatures.has_value());
        this->_numOutputFeatures = _numOutputFeatures;
        return *this;
    }

    virtual FullyConnected::Builder &hasBias(bool _hasBias) {
        THOR_THROW_IF_FALSE(!this->_hasBias.has_value());
        this->_hasBias = _hasBias;
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializer(std::shared_ptr<Initializer> &_weightsInitializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializer(std::shared_ptr<Initializer> &&_weightsInitializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializer(std::shared_ptr<Initializer> &_biasInitializer) {
        THOR_THROW_IF_FALSE(this->_biasInitializer == nullptr);
        this->_biasInitializer = _biasInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializer(std::shared_ptr<Initializer> &&_biasInitializer) {
        THOR_THROW_IF_FALSE(this->_biasInitializer == nullptr);
        this->_biasInitializer = _biasInitializer->clone();
        return *this;
    }

    // Adds an activation layer after this FullyConnected layer
    virtual FullyConnected::Builder &activation(std::shared_ptr<Activation> &_activation) {
        THOR_THROW_IF_FALSE(this->_activation == nullptr);
        THOR_THROW_IF_FALSE(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual FullyConnected::Builder &activation(std::shared_ptr<Activation> &&_activation) {
        THOR_THROW_IF_FALSE(this->_activation == nullptr);
        THOR_THROW_IF_FALSE(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual FullyConnected::Builder &weightsDataType(DataType _weightsDataType) {
        THOR_THROW_IF_FALSE(!this->_weightsDataType.has_value());
        this->_weightsDataType = _weightsDataType;
        return *this;
    }

    virtual FullyConnected::Builder &computeDataType(DataType _computeDataType) {
        THOR_THROW_IF_FALSE(!this->_computeDataType.has_value());
        this->_computeDataType = _computeDataType;
        return *this;
    }

    virtual FullyConnected::Builder &outputDataType(DataType _outputDataType) {
        THOR_THROW_IF_FALSE(!this->_outputDataType.has_value());
        this->_outputDataType = _outputDataType;
        return *this;
    }

    virtual FullyConnected::Builder &noActivation() {
        THOR_THROW_IF_FALSE(!this->_activation);

        _activationExplicitlyRemoved = true;
        return *this;
    }

    virtual FullyConnected::Builder &epilogue(const ThorImplementation::Expression &expression) {
        THOR_THROW_IF_FALSE(!this->_epilogue.has_value());
        FullyConnected::validateEpilogueExpression(expression);
        _epilogue = expression;
        return *this;
    }

    virtual FullyConnected::Builder &weightsOptimizer(std::shared_ptr<Optimizer> _weightsOptimizer) {
        THOR_THROW_IF_FALSE(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = _weightsOptimizer;
        return *this;
    }

    virtual FullyConnected::Builder &biasesOptimizer(std::shared_ptr<Optimizer> _biasesOptimizer) {
        THOR_THROW_IF_FALSE(this->_biasesOptimizer == nullptr);
        this->_biasesOptimizer = _biasesOptimizer;
        return *this;
    }

   private:
    void verifyConfig() const;

    std::optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    std::optional<uint32_t> _numOutputFeatures;
    std::optional<bool> _hasBias;
    std::shared_ptr<Activation> _activation;
    std::optional<DataType> _weightsDataType;
    std::optional<DataType> _computeDataType;
    std::optional<DataType> _outputDataType;

    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasInitializer;
    std::shared_ptr<Optimizer> _weightsOptimizer;
    std::shared_ptr<Optimizer> _biasesOptimizer;
    bool _activationExplicitlyRemoved;

    // FIXME: Future optimization, automatically fuse adjacent expressions from adjacent layers.
    //        For now epilogue gives access to post layer fusion, if that optimization goes in, it can remain as a convenience feature.
    std::optional<ThorImplementation::Expression> _epilogue;
};

}  // namespace Thor
