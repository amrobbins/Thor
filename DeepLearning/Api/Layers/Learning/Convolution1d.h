#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "Utilities/Exceptions.h"
#include "Utilities/Expression/ConvolutionSpatial.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {

enum class Convolution1dPaddingMode {
    VALID,
    SAME_UPPER,
    CAUSAL,
    EXPLICIT,
};

class Convolution1d : public TrainableLayer {
   public:
    class Builder;

    Convolution1d() = default;
    explicit Convolution1d(const std::optional<ThorImplementation::Expression> epilogue,
                           std::vector<std::pair<std::string, Tensor>> epilogueInputBindings = {})
        : epilogue(epilogue), epilogueInputBindings(std::move(epilogueInputBindings)) {}
    ~Convolution1d() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Convolution1d>(*this); }

    uint32_t getNumOutputChannels() const { return numOutputChannels; }
    uint32_t getFilterWidth() const { return filterWidth; }
    uint32_t getGroups() const { return groups; }
    uint32_t getStride() const { return static_cast<uint32_t>(spatial.stride); }
    uint32_t getDilation() const { return static_cast<uint32_t>(spatial.dilation); }
    uint32_t getPaddingLeft() const { return static_cast<uint32_t>(spatial.pre_padding); }
    uint32_t getPaddingRight() const { return static_cast<uint32_t>(spatial.post_padding); }
    Convolution1dPaddingMode getPaddingMode() const { return paddingMode; }
    bool getHasBias() const { return hasBias; }
    DataType getComputeDataType() const { return computeDataType; }
    bool getUseRagged() const { return raggedFeatureInput.has_value(); }
    std::optional<RaggedTensor> getRaggedFeatureInput() const { return raggedFeatureInput; }
    std::optional<RaggedTensor> getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    std::string getLayerType() const override { return "Convolution1d"; }
    std::string getLayerVersion() const override { return "1.0.0"; }

    nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork &stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);
    nlohmann::json architectureJson() const override;

    static const char *epilogueInputName() { return "__convolution_1d_epilogue_input"; }
    static const char *epilogueOutputName() { return "__convolution_1d_epilogue_output"; }

    [[nodiscard]] static ThorImplementation::Expression epilogueInput(
        std::optional<ThorImplementation::DataType> computeDType = std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType = std::nullopt) {
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
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Convolution1d");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression &expression,
                                           const std::vector<std::string> &auxiliaryInputNames = {}) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Convolution1d");
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition,
                                           const std::vector<std::string> &auxiliaryInputNames = {}) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Convolution1d");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition &definition,
        const std::vector<std::string> &auxiliaryInputNames = {}) {
        return LayerEpilogue::expressionFromDefinition(definition,
                                                       epilogueInputName(),
                                                       auxiliaryInputNames,
                                                       epilogueOutputName(),
                                                       "Convolution1d");
    }

    [[nodiscard]] static ThorImplementation::Expression applyEpilogue(const ThorImplementation::Expression &input,
                                                                      const ThorImplementation::Expression &epilogue) {
        return LayerEpilogue::apply(input, epilogue, epilogueInputName());
    }

    static void validateEpilogueAuxInputName(const std::string &inputName);

    using MultiConnectionLayer::getFeatureOutput;
    int getConnectionType(Tensor connectingTensor) const override;
    std::vector<Tensor> getFeatureInputs() const override;
    Tensor getFeatureOutput(Tensor inputTensor) const override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    bool mustConnectAllInputsToDriveOutput() const override { return raggedFeatureInput.has_value() || !epilogueInputBindings.empty(); }

   protected:
    void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) override {
        (void)inputTensor;
        (void)batchSize;
        (void)stream;
    }

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                  std::optional<Event> sisterLayerLoadedEvent) {
        return TrainableLayer::initialize(layer, isFirstStamp, sisterLayer, sisterLayerLoadedEvent);
    }

    std::vector<Tensor> standaloneLayerFeatureInputs;
    std::vector<Tensor> standaloneLayerFeatureOutputs;

   private:
    uint32_t numOutputChannels = 0;
    uint32_t filterWidth = 0;
    uint32_t groups = 1;
    ThorImplementation::ConvolutionSpatial1d spatial;
    Convolution1dPaddingMode paddingMode = Convolution1dPaddingMode::VALID;
    bool hasBias = false;
    DataType computeDataType = DataType::FP32;
    std::shared_ptr<Initializer> weightsInitializer;
    std::shared_ptr<Initializer> biasInitializer;
    std::shared_ptr<Activation> activation;
    std::shared_ptr<Optimizer> weightsOptimizer;
    std::shared_ptr<Optimizer> biasesOptimizer;

    const std::optional<ThorImplementation::Expression> epilogue;
    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;
    std::optional<RaggedTensor> raggedFeatureInput;
    std::optional<RaggedTensor> raggedFeatureOutput;

    std::vector<std::string> epilogueAuxInputNames() const;
    std::vector<uint32_t> inputPortIndicesForTensor(Tensor tensor) const;

    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;
    mutable std::unordered_map<uint64_t, uint32_t> nextInputConnectionCursorByTensorOriginalId;
    std::unordered_map<uint64_t, uint32_t> nextTraversalInputCursorByTensorOriginalId;
};

class Convolution1d::Builder {
   public:
    Builder() = default;
    virtual ~Builder() = default;

    Convolution1d build();

    Builder &network(Network &network) {
        THOR_THROW_IF_FALSE(!_network.has_value());
        _network = &network;
        return *this;
    }

    Builder &featureInput(Tensor featureInput) {
        THOR_THROW_IF_FALSE(featureInput.getDimensions().size() == 2);
        THOR_THROW_IF_FALSE(_featureInputs.empty());
        THOR_THROW_IF_FALSE(!_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.getDimensions()[0] > 0);
        THOR_THROW_IF_FALSE(featureInput.getDimensions()[1] > 0);
        _featureInputs.push_back(featureInput);
        return *this;
    }

    Builder &featureInput(RaggedTensor featureInput) {
        THOR_THROW_IF_FALSE(_featureInputs.empty());
        THOR_THROW_IF_FALSE(!_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.getValuesDimensions().size() == 2);
        THOR_THROW_IF_FALSE(featureInput.getTrailingDimensions().size() == 1);
        THOR_THROW_IF_FALSE(featureInput.getTrailingDimensions().front() > 0);
        _raggedFeatureInput = featureInput;
        _featureInputs.push_back(featureInput.getValues());
        return *this;
    }

    Builder &numOutputChannels(uint32_t value) {
        THOR_THROW_IF_FALSE(value > 0);
        THOR_THROW_IF_FALSE(!_numOutputChannels.has_value());
        _numOutputChannels = value;
        return *this;
    }

    Builder &filterWidth(uint32_t value) {
        THOR_THROW_IF_FALSE(value > 0);
        THOR_THROW_IF_FALSE(!_filterWidth.has_value());
        _filterWidth = value;
        return *this;
    }

    Builder &groups(uint32_t value) {
        THOR_THROW_IF_FALSE(value > 0);
        THOR_THROW_IF_FALSE(!_groups.has_value());
        _groups = value;
        return *this;
    }

    Builder &stride(uint32_t value) {
        THOR_THROW_IF_FALSE(value > 0);
        THOR_THROW_IF_FALSE(!_stride.has_value());
        _stride = value;
        return *this;
    }

    Builder &dilation(uint32_t value) {
        THOR_THROW_IF_FALSE(value > 0);
        THOR_THROW_IF_FALSE(!_dilation.has_value());
        _dilation = value;
        return *this;
    }

    Builder &validPadding() {
        THOR_THROW_IF_FALSE(!_paddingMode.has_value());
        _paddingMode = Convolution1dPaddingMode::VALID;
        return *this;
    }

    Builder &samePadding() {
        THOR_THROW_IF_FALSE(!_paddingMode.has_value());
        _paddingMode = Convolution1dPaddingMode::SAME_UPPER;
        return *this;
    }

    Builder &causalPadding() {
        THOR_THROW_IF_FALSE(!_paddingMode.has_value());
        _paddingMode = Convolution1dPaddingMode::CAUSAL;
        return *this;
    }

    Builder &padding(uint32_t left, uint32_t right) {
        THOR_THROW_IF_FALSE(!_paddingMode.has_value());
        _paddingLeft = left;
        _paddingRight = right;
        _paddingMode = Convolution1dPaddingMode::EXPLICIT;
        return *this;
    }

    Builder &hasBias(bool value) {
        THOR_THROW_IF_FALSE(!_hasBias.has_value());
        _hasBias = value;
        return *this;
    }

    Builder &computeDataType(DataType value) {
        THOR_THROW_IF_FALSE(!_computeDataType.has_value());
        if (value != DataType::FP32 && value != DataType::TF32)
            throw std::invalid_argument("Convolution1d computeDataType must be fp32 or tf32.");
        _computeDataType = value;
        return *this;
    }

    Builder &weightsInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(_weightsInitializer == nullptr);
        _weightsInitializer = initializer->clone();
        return *this;
    }

    Builder &biasInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(_biasesInitializer == nullptr);
        _biasesInitializer = initializer->clone();
        return *this;
    }

    Builder &activation(std::shared_ptr<Activation> value) {
        THOR_THROW_IF_FALSE(_activation == nullptr);
        THOR_THROW_IF_FALSE(!_activationExplicitlyRemoved);
        _activation = std::move(value);
        return *this;
    }

    Builder &noActivation() {
        THOR_THROW_IF_FALSE(_activation == nullptr);
        _activationExplicitlyRemoved = true;
        return *this;
    }

    Builder &epilogue(const ThorImplementation::Expression &expression) {
        THOR_THROW_IF_FALSE(!_epilogue.has_value());
        Convolution1d::validateEpilogueExpression(expression, epilogueAuxInputNames());
        _epilogue = expression;
        return *this;
    }

    Builder &epilogueInput(const std::string &inputName, Tensor tensor) {
        Convolution1d::validateEpilogueAuxInputName(inputName);
        THOR_THROW_IF_FALSE(tensor.isInitialized());
        for (const auto &[existingName, existingTensor] : _epilogueInputBindings) {
            (void)existingTensor;
            if (existingName == inputName)
                throw std::invalid_argument("Convolution1d epilogue input name is duplicated: " + inputName + ".");
        }
        _epilogueInputBindings.emplace_back(inputName, tensor);
        if (_epilogue.has_value())
            Convolution1d::validateEpilogueExpression(_epilogue.value(), epilogueAuxInputNames());
        return *this;
    }

    Builder &weightsOptimizer(std::shared_ptr<Optimizer> value) {
        THOR_THROW_IF_FALSE(_weightsOptimizer == nullptr);
        _weightsOptimizer = std::move(value);
        return *this;
    }

    Builder &biasesOptimizer(std::shared_ptr<Optimizer> value) {
        THOR_THROW_IF_FALSE(_biasesOptimizer == nullptr);
        _biasesOptimizer = std::move(value);
        return *this;
    }

    static uint32_t computeOutputWidth(uint32_t inputWidth,
                                       uint32_t filterWidth,
                                       const ThorImplementation::ConvolutionSpatial1d &spatial) {
        THOR_THROW_IF_FALSE(spatial.stride > 0);
        THOR_THROW_IF_FALSE(spatial.dilation > 0);
        THOR_THROW_IF_FALSE(spatial.pre_padding >= 0);
        THOR_THROW_IF_FALSE(spatial.post_padding >= 0);
        const uint64_t effectiveFilter =
            static_cast<uint64_t>(spatial.dilation) * (static_cast<uint64_t>(filterWidth) - 1ULL) + 1ULL;
        const uint64_t paddedInput = static_cast<uint64_t>(inputWidth) + static_cast<uint64_t>(spatial.pre_padding) +
                                     static_cast<uint64_t>(spatial.post_padding);
        THOR_THROW_IF_FALSE(effectiveFilter <= paddedInput);
        return static_cast<uint32_t>(1ULL + (paddedInput - effectiveFilter) / static_cast<uint64_t>(spatial.stride));
    }

   private:
    std::optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::optional<uint32_t> _numOutputChannels;
    std::optional<uint32_t> _filterWidth;
    std::optional<uint32_t> _groups;
    std::optional<uint32_t> _stride;
    std::optional<uint32_t> _dilation;
    std::optional<Convolution1dPaddingMode> _paddingMode;
    std::optional<uint32_t> _paddingLeft;
    std::optional<uint32_t> _paddingRight;
    std::optional<bool> _hasBias;
    std::optional<DataType> _computeDataType;
    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasesInitializer;
    std::shared_ptr<Activation> _activation;
    bool _activationExplicitlyRemoved = false;
    std::shared_ptr<Optimizer> _weightsOptimizer;
    std::shared_ptr<Optimizer> _biasesOptimizer;
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
