#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/Expression.h"

#include <atomic>
#include <utility>
#include <optional>
#include <set>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace Thor {

class Activation : public Layer {
   public:
    class Builder;

    Activation() {}
    ~Activation() override {}

    // Layer::addToNetwork is used during deserialization when an activation is an actual attached layer - or when the activation
    // is used as a standalone layer.
    // Activation::addToNetwork is used when an attached layer is added to the network as templated by that particular activation.
    using Layer::addToNetwork;
    // Activation template version.
    virtual Tensor addToNetwork(Tensor inputTensor, Network* network);
    virtual Tensor addToNetwork(Tensor inputTensor,
                                Network* network,
                                std::optional<ThorImplementation::Expression> epilogue,
                                std::vector<std::pair<std::string, Tensor>> epilogueInputBindings = {});

    // Returns an expression equivalent to applying this activation to the input expression.
    // This is used by expression-backed learning layers to fuse the activation into the layer equation.
    virtual ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const = 0;

    std::string getLayerType() const override = 0;

    static const char* epilogueInputName() { return "__activation_epilogue_input"; }
    static const char* epilogueOutputName() { return "__activation_epilogue_output"; }

    [[nodiscard]] static ThorImplementation::Expression epilogueInput(
        std::optional<ThorImplementation::DataType> computeDType = std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType = std::nullopt) {
        return LayerEpilogue::input(epilogueInputName(), computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueAuxInput(
        const std::string& inputName,
        std::optional<ThorImplementation::DataType> computeDType = std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType = std::nullopt) {
        validateEpilogueAuxInputName(inputName);
        return LayerEpilogue::input(inputName, computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(
        const ThorImplementation::Expression& expression,
        const std::vector<std::string>& auxiliaryInputNames = {}) {
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Activation");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression& expression,
                                           const std::vector<std::string>& auxiliaryInputNames = {}) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Activation");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition& definition,
        const std::vector<std::string>& auxiliaryInputNames = {}) {
        return LayerEpilogue::expressionFromDefinition(definition,
                                                       epilogueInputName(),
                                                       auxiliaryInputNames,
                                                       epilogueOutputName(),
                                                       "Activation");
    }

    [[nodiscard]] static ThorImplementation::Expression applyEpilogue(const ThorImplementation::Expression& input,
                                                                      const ThorImplementation::Expression& epilogue) {
        return LayerEpilogue::apply(input, epilogue, epilogueInputName());
    }

    static void validateEpilogueAuxInputName(const std::string& inputName);

    std::vector<Tensor> getFeatureInputs() const;
    std::vector<Tensor> getFeatureOutputs() const;
    std::vector<Tensor> getAllOutputTensors() const override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() const override { return !epilogueInputBindings.empty(); }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    int getConnectionType(Tensor connectingTensor) const override;

    nlohmann::json architectureJson() const override;
    nlohmann::json serialize(thor_file::TarWriter& archiveWriter, Stream stream) const override { return architectureJson(); }

    static void deserialize(const nlohmann::json& j, Network* network);
    static std::shared_ptr<Activation> deserializeTemplate(const nlohmann::json& j);
    using Deserializer = std::function<void(const nlohmann::json&, Network*)>;
    static std::unordered_map<std::string, Deserializer>& get_registry();
    static void register_layer(std::string name, Deserializer fn);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stampExpressionBackedActivation(ThorImplementation::TensorPlacement placement,
                                                                                Thor::Tensor connectingApiTensor,
                                                                                bool inferenceOnly) const;

    uint64_t getExpressionBackedActivationMemRequirementInBytes(uint32_t batchSize) const;
    void initializeStandaloneActivation(Tensor inputTensor,
                                        std::optional<ThorImplementation::Expression> epilogue = std::nullopt,
                                        std::vector<std::pair<std::string, Tensor>> epilogueInputBindings = {});
    void deserializeStandaloneFields(const nlohmann::json& j, Network* network);

    std::optional<ThorImplementation::Expression> epilogue;
    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;
    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;
    mutable std::unordered_map<uint64_t, uint32_t> nextInputConnectionCursorByTensorOriginalId;

   private:
    using Layer::serialize;
};

class Activation::Builder {
   public:
    virtual ~Builder() {}
    virtual Activation::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }
    virtual Activation::Builder& featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Activation::Builder& epilogue(const ThorImplementation::Expression& expression) {
        THOR_THROW_IF_FALSE(!this->_epilogue.has_value());
        Activation::validateEpilogueExpression(expression, epilogueAuxInputNames());
        this->_epilogue = expression;
        return *this;
    }

    virtual Activation::Builder& epilogueInput(const std::string& inputName, Tensor tensor) {
        Activation::validateEpilogueAuxInputName(inputName);
        for (const auto& [existingName, existingTensor] : _epilogueInputBindings) {
            (void)existingTensor;
            if (existingName == inputName) {
                throw std::invalid_argument("Activation epilogue input name is duplicated: " + inputName + ".");
            }
        }
        _epilogueInputBindings.emplace_back(inputName, tensor);
        if (_epilogue.has_value()) {
            Activation::validateEpilogueExpression(_epilogue.value(), epilogueAuxInputNames());
        }
        return *this;
    }

    virtual std::shared_ptr<Activation> build() = 0;

   protected:
    std::vector<std::string> epilogueAuxInputNames() const {
        std::vector<std::string> names;
        names.reserve(_epilogueInputBindings.size());
        for (const auto& [name, tensor] : _epilogueInputBindings) {
            (void)tensor;
            names.push_back(name);
        }
        return names;
    }

    void applyStandaloneConfiguration(Activation& activation) const {
        THOR_THROW_IF_FALSE(_featureInput.has_value());
        activation.initializeStandaloneActivation(_featureInput.value(), _epilogue, _epilogueInputBindings);
    }

    std::optional<Network*> _network;
    std::optional<Tensor> _featureInput;
    std::optional<ThorImplementation::Expression> _epilogue;
    std::vector<std::pair<std::string, Tensor>> _epilogueInputBindings;
};

}  // namespace Thor
