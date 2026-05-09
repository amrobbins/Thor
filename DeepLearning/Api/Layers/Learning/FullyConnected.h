#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/Expression.h"

#include <assert.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

class FullyConnected : public TrainableLayer {
   public:
    class Builder;

    FullyConnected(const Optional<ThorImplementation::Expression> epilogue) : epilogue(epilogue) {}

    virtual ~FullyConnected() = default;

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
        Optional<ThorImplementation::TensorDescriptor::DataType> computeDType =
            Optional<ThorImplementation::TensorDescriptor::DataType>::empty(),
        Optional<ThorImplementation::TensorDescriptor::DataType> outputDType =
            Optional<ThorImplementation::TensorDescriptor::DataType>::empty()) {
        return ThorImplementation::Expression::input(epilogueInputName(), computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(const ThorImplementation::Expression &expression) {
        ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
            ThorImplementation::Expression::outputs({{epilogueOutputName(), expression}}));
        validateEpilogueDefinition(definition);
        return definition;
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression &expression) { (void)makeEpilogueDefinition(expression); }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition) {
        definition.validate();
        if (definition.outputs.outputs.size() != 1 || definition.outputs.outputs.front().name != epilogueOutputName()) {
            throw std::invalid_argument("FullyConnected epilogue expression must have exactly one output named " +
                                        std::string(epilogueOutputName()) + ".");
        }
        if (definition.outputs.expr == nullptr || definition.outputs.expr->inputs.size() != 1 ||
            definition.outputs.expr->inputs.front().name != epilogueInputName() ||
            definition.outputs.expr->inputs.front().kind != ThorImplementation::NamedInput::Kind::Tensor) {
            throw std::invalid_argument("FullyConnected epilogue expression must have exactly one tensor input named " +
                                        std::string(epilogueInputName()) + ".");
        }
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
    Tensor::DataType weightsDataType;
    Tensor::DataType computeDataType;
    Tensor::DataType outputDataType;

    const Optional<ThorImplementation::Expression> epilogue;
    mutable Optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;

    static bool isFullyConnectedFloatingDataType(Tensor::DataType dataType);
    static std::string dataTypeName(Tensor::DataType dataType);
    static uint64_t checkedFeatureCount(const std::vector<uint64_t> &dimensions, const std::string &what);
    static void verifyFullyConnectedDataType(Tensor::DataType dataType, const std::string &what);
    static void verifyFullyConnectedComputeDataType(Tensor::DataType dataType);

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
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual FullyConnected::Builder &featureInput(Tensor _featureInput) {
        assert(!_featureInput.getDimensions().empty());
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            assert(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            assert(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual FullyConnected::Builder &numOutputFeatures(uint32_t _numOutputFeatures) {
        assert(!this->_numOutputFeatures.isPresent());
        this->_numOutputFeatures = _numOutputFeatures;
        return *this;
    }

    virtual FullyConnected::Builder &hasBias(bool _hasBias) {
        assert(!this->_hasBias.isPresent());
        this->_hasBias = _hasBias;
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializer(std::shared_ptr<Initializer> &_weightsInitializer) {
        assert(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializer(std::shared_ptr<Initializer> &&_weightsInitializer) {
        assert(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializer(std::shared_ptr<Initializer> &_biasInitializer) {
        assert(this->_biasInitializer == nullptr);
        this->_biasInitializer = _biasInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializer(std::shared_ptr<Initializer> &&_biasInitializer) {
        assert(this->_biasInitializer == nullptr);
        this->_biasInitializer = _biasInitializer->clone();
        return *this;
    }

    // Adds an activation layer after this FullyConnected layer
    virtual FullyConnected::Builder &activation(std::shared_ptr<Activation> &_activation) {
        assert(this->_activation == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual FullyConnected::Builder &activation(std::shared_ptr<Activation> &&_activation) {
        assert(this->_activation == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual FullyConnected::Builder &weightsDataType(Tensor::DataType _weightsDataType) {
        assert(this->_weightsDataType.isEmpty());
        this->_weightsDataType = _weightsDataType;
        return *this;
    }

    virtual FullyConnected::Builder &computeDataType(Tensor::DataType _computeDataType) {
        assert(this->_computeDataType.isEmpty());
        this->_computeDataType = _computeDataType;
        return *this;
    }

    virtual FullyConnected::Builder &outputDataType(Tensor::DataType _outputDataType) {
        assert(this->_outputDataType.isEmpty());
        this->_outputDataType = _outputDataType;
        return *this;
    }

    virtual FullyConnected::Builder &noActivation() {
        assert(!this->_activation);

        _activationExplicitlyRemoved = true;
        return *this;
    }

    virtual FullyConnected::Builder &epilogue(const ThorImplementation::Expression &expression) {
        assert(this->_epilogue.isEmpty());
        FullyConnected::validateEpilogueExpression(expression);
        _epilogue = expression;
        return *this;
    }

    virtual FullyConnected::Builder &weightsOptimizer(std::shared_ptr<Optimizer> _weightsOptimizer) {
        assert(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = _weightsOptimizer;
        return *this;
    }

    virtual FullyConnected::Builder &biasesOptimizer(std::shared_ptr<Optimizer> _biasesOptimizer) {
        assert(this->_biasesOptimizer == nullptr);
        this->_biasesOptimizer = _biasesOptimizer;
        return *this;
    }

   private:
    void verifyConfig() const;

    Optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    Optional<uint32_t> _numOutputFeatures;
    Optional<bool> _hasBias;
    std::shared_ptr<Activation> _activation;
    Optional<Tensor::DataType> _weightsDataType;
    Optional<Tensor::DataType> _computeDataType;
    Optional<Tensor::DataType> _outputDataType;

    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasInitializer;
    std::shared_ptr<Optimizer> _weightsOptimizer;
    std::shared_ptr<Optimizer> _biasesOptimizer;
    bool _activationExplicitlyRemoved;

    // FIXME: Future optimization, automatically fuse adjacent expressions from adjacent layers.
    //        For now epilogue gives access to post layer fusion, if that optimization goes in, it can remain as a convenience feature.
    Optional<ThorImplementation::Expression> _epilogue;
};

}  // namespace Thor
