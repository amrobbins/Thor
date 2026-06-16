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
#include <set>
#include <unordered_map>
#include <utility>

namespace Thor {

class FullyConnected : public TrainableLayer {
   public:
    class Builder;

    explicit FullyConnected(const std::optional<ThorImplementation::Expression> epilogue,
                           std::vector<std::pair<std::string, Tensor>> epilogueInputBindings = {})
        : epilogue(epilogue), epilogueInputBindings(std::move(epilogueInputBindings)) {}

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

    [[nodiscard]] static ThorImplementation::Expression epilogueAuxInput(
        const std::string &inputName,
        std::optional<ThorImplementation::DataType> computeDType = std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType = std::nullopt) {
        validateEpilogueAuxInputName(inputName);
        return LayerEpilogue::input(inputName, computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(
        const ThorImplementation::Expression &expression,
        const std::vector<std::string> &auxiliaryInputNames = {}) {
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "FullyConnected");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression &expression,
                                           const std::vector<std::string> &auxiliaryInputNames = {}) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "FullyConnected");
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition,
                                           const std::vector<std::string> &auxiliaryInputNames = {}) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "FullyConnected");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition &definition,
        const std::vector<std::string> &auxiliaryInputNames = {}) {
        return LayerEpilogue::expressionFromDefinition(definition,
                                                       epilogueInputName(),
                                                       auxiliaryInputNames,
                                                       epilogueOutputName(),
                                                       "FullyConnected");
    }

    [[nodiscard]] static ThorImplementation::Expression applyEpilogue(const ThorImplementation::Expression &input,
                                                                      const ThorImplementation::Expression &epilogue) {
        return LayerEpilogue::apply(input, epilogue, epilogueInputName());
    }

    static void validateEpilogueAuxInputName(const std::string &inputName);

    int getConnectionType(Tensor connectingTensor) const override;
    std::vector<Tensor> getFeatureInputs() const override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    void informThatInputConnectionMade(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() const override { return !epilogueInputBindings.empty(); }

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
    bool preserveInputPrefixDimensions = false;
    std::shared_ptr<Activation> activation;
    DataType weightsDataType;
    DataType computeDataType;
    DataType outputDataType;

    const std::optional<ThorImplementation::Expression> epilogue;
    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;

    std::vector<std::string> epilogueAuxInputNames() const;
    std::vector<uint32_t> inputPortIndicesForTensor(Tensor tensor) const;

    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;
    mutable std::unordered_map<uint64_t, uint32_t> nextInputConnectionCursorByTensorOriginalId;

    static bool isFullyConnectedFloatingDataType(DataType dataType);
    static std::string dataTypeName(DataType dataType);
    static uint64_t checkedFeatureCount(const std::vector<uint64_t> &dimensions, const std::string &what);
    static uint64_t checkedInputFeatureCount(const std::vector<uint64_t> &dimensions, bool preservePrefixDimensions, const std::string &what);
    static std::vector<uint64_t> fullyConnectedOutputDimensions(const std::vector<uint64_t>& inputDimensions,
                                                                uint32_t numOutputFeatures,
                                                                bool preservePrefixDimensions);
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

    virtual FullyConnected::Builder &preserveInputPrefixDimensions(bool _preserveInputPrefixDimensions) {
        THOR_THROW_IF_FALSE(!this->_preserveInputPrefixDimensions.has_value());
        this->_preserveInputPrefixDimensions = _preserveInputPrefixDimensions;
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
        FullyConnected::validateEpilogueExpression(expression, epilogueAuxInputNames());
        _epilogue = expression;
        return *this;
    }

    virtual FullyConnected::Builder &epilogueInput(const std::string &inputName, Tensor tensor) {
        FullyConnected::validateEpilogueAuxInputName(inputName);
        THOR_THROW_IF_FALSE(tensor.isInitialized());
        for (const auto &[existingName, existingTensor] : _epilogueInputBindings) {
            (void)existingTensor;
            if (existingName == inputName) {
                throw std::invalid_argument("FullyConnected epilogue input name is duplicated: " + inputName + ".");
            }
        }
        _epilogueInputBindings.emplace_back(inputName, tensor);
        if (_epilogue.has_value()) {
            FullyConnected::validateEpilogueExpression(_epilogue.value(), epilogueAuxInputNames());
        }
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
    std::optional<bool> _preserveInputPrefixDimensions;
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
    std::vector<std::pair<std::string, Tensor>> _epilogueInputBindings;

    std::vector<std::string> epilogueAuxInputNames() const {
        std::vector<std::string> names;
        names.reserve(_epilogueInputBindings.size());
        for (const auto &[name, tensor] : _epilogueInputBindings) {
            (void)tensor;
            names.push_back(name);
        }
        return names;
    }
};

}  // namespace Thor
