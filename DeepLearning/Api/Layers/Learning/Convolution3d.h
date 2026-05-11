#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution3d.h"
#include "Utilities/Exceptions.h"
#include <optional>

namespace Thor {

class Convolution3d : public TrainableLayer {
   public:
    class Builder;

    Convolution3d() {}
    explicit Convolution3d(const std::optional<ThorImplementation::Expression> epilogue) : epilogue(epilogue) {}
    ~Convolution3d() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Convolution3d>(*this); }

    virtual uint32_t getFilterDepth() { return filterDepth; }
    virtual uint32_t getFilterHeight() { return filterHeight; }
    virtual uint32_t getFilterWidth() { return filterWidth; }
    virtual uint32_t getDepthStride() { return depthStride; }
    virtual uint32_t getVerticalStride() { return verticalStride; }
    virtual uint32_t getHorizontalStride() { return horizontalStride; }
    virtual uint32_t getDepthPadding() { return depthPadding; }
    virtual uint32_t getVerticalPadding() { return verticalPadding; }
    virtual uint32_t getHorizontalPadding() { return horizontalPadding; }

    std::string getLayerType() const override { return "Convolution3d"; }

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const nlohmann::json& j, Network* network);
    nlohmann::json architectureJson() const override;

    static const char *epilogueInputName() { return "__convolution_3d_epilogue_input"; }
    static const char *epilogueOutputName() { return "__convolution_3d_epilogue_output"; }

    [[nodiscard]] static ThorImplementation::Expression epilogueInput(
        std::optional<ThorImplementation::TensorDescriptor::DataType> computeDType =
            std::nullopt,
        std::optional<ThorImplementation::TensorDescriptor::DataType> outputDType =
            std::nullopt) {
        return LayerEpilogue::input(epilogueInputName(), computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(const ThorImplementation::Expression &expression) {
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), epilogueOutputName(), "Convolution3d");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression &expression) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), epilogueOutputName(), "Convolution3d");
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), epilogueOutputName(), "Convolution3d");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition &definition) {
        return LayerEpilogue::expressionFromDefinition(definition, epilogueInputName(), epilogueOutputName(), "Convolution3d");
    }

    [[nodiscard]] static ThorImplementation::Expression applyEpilogue(const ThorImplementation::Expression &input,
                                                                      const ThorImplementation::Expression &epilogue) {
        return LayerEpilogue::apply(input, epilogue, epilogueInputName());
    }

   protected:
    virtual bool isMultiLayer() const { return false; }
    virtual void buildSupportLayersAndAddToNetwork(Network* network);

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
    uint32_t numOutputChannels;
    uint32_t filterDepth;
    uint32_t filterHeight;
    uint32_t filterWidth;
    uint32_t depthStride;
    uint32_t verticalStride;
    uint32_t horizontalStride;
    uint32_t depthPadding;
    uint32_t verticalPadding;
    uint32_t horizontalPadding;
    bool hasBias;
    std::shared_ptr<Initializer> weightsInitializer;
    std::shared_ptr<Initializer> biasInitializer;
    std::shared_ptr<Activation> activation;
    std::shared_ptr<Optimizer> weightsOptimizer;
    std::shared_ptr<Optimizer> biasesOptimizer;

    const std::optional<ThorImplementation::Expression> epilogue;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;
};

class Convolution3d::Builder {
   public:
    virtual ~Builder() = default;

    Builder() { _activationExplicitlyRemoved = false; }

    virtual Convolution3d build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(!_featureInputs.empty());
        THOR_THROW_IF_FALSE(_numOutputChannels.has_value());
        THOR_THROW_IF_FALSE(_filterDepth.has_value());
        THOR_THROW_IF_FALSE(_filterHeight.has_value());
        THOR_THROW_IF_FALSE(_filterWidth.has_value());

        if (!_depthStride.has_value())
            _depthStride = 1;
        if (!_verticalStride.has_value())
            _verticalStride = 1;
        if (!_horizontalStride.has_value())
            _horizontalStride = 1;
        if (!_depthPadding.has_value())
            _depthPadding = 0;
        if (!_verticalPadding.has_value())
            _verticalPadding = 0;
        if (!_horizontalPadding.has_value())
            _horizontalPadding = 0;
        if (!_hasBias.has_value())
            _hasBias = false;
        if (_weightsInitializer == nullptr)
            _weightsInitializer = Glorot::Builder().build();
        if (_biasesInitializer == nullptr)
            _biasesInitializer = Glorot::Builder().build();
        if (!_activation && !_activationExplicitlyRemoved)
            _activation = Gelu::Builder().build();

        if (_epilogue.has_value()) {
            Convolution3d::validateEpilogueExpression(_epilogue.value());
        }

        Convolution3d convolution3d(_epilogue);
        convolution3d.featureInputs = _featureInputs;
        convolution3d.numOutputChannels = _numOutputChannels.value();
        convolution3d.filterDepth = _filterDepth.value();
        convolution3d.filterHeight = _filterHeight.value();
        convolution3d.filterWidth = _filterWidth.value();
        convolution3d.depthStride = _depthStride.value();
        convolution3d.verticalStride = _verticalStride.value();
        convolution3d.horizontalStride = _horizontalStride.value();
        convolution3d.depthPadding = _depthPadding.value();
        convolution3d.verticalPadding = _verticalPadding.value();
        convolution3d.horizontalPadding = _horizontalPadding.value();

        THOR_THROW_IF_FALSE(convolution3d.depthPadding < convolution3d.filterDepth);
        THOR_THROW_IF_FALSE(convolution3d.verticalPadding < convolution3d.filterHeight);
        THOR_THROW_IF_FALSE(convolution3d.horizontalPadding < convolution3d.filterWidth);

        uint32_t outputDepth = computeOutputDimension(convolution3d.featureInputs[0].getDimensions()[1],
                                                      convolution3d.depthStride,
                                                      convolution3d.filterDepth,
                                                      convolution3d.depthPadding);
        uint32_t outputHeight = computeOutputDimension(convolution3d.featureInputs[0].getDimensions()[2],
                                                       convolution3d.verticalStride,
                                                       convolution3d.filterHeight,
                                                       convolution3d.verticalPadding);
        uint32_t outputWidth = computeOutputDimension(convolution3d.featureInputs[0].getDimensions()[3],
                                                      convolution3d.horizontalStride,
                                                      convolution3d.filterWidth,
                                                      convolution3d.horizontalPadding);

        convolution3d.hasBias = _hasBias.value();
        convolution3d.weightsInitializer = _weightsInitializer->clone();
        convolution3d.biasInitializer = _biasesInitializer->clone();
        if (_activation != nullptr)
            convolution3d.activation = _activation;
        convolution3d.weightsOptimizer = _weightsOptimizer;
        convolution3d.biasesOptimizer = _biasesOptimizer;

        const Tensor::DataType convolutionDataType = convolution3d.featureInputs.front().getDataType();
        const Tensor::DataType weightsDataType = convolutionDataType;
        const uint64_t inputChannels = convolution3d.featureInputs.front().getDimensions()[0];

        ParameterSpecification::Builder weightsParameterBuilder;
        weightsParameterBuilder.name("weights")
            .shape({convolution3d.numOutputChannels, inputChannels, convolution3d.filterDepth, convolution3d.filterHeight, convolution3d.filterWidth})
            .dtype(weightsDataType)
            .initializer(convolution3d.weightsInitializer)
            .trainable(true);
        if (convolution3d.weightsOptimizer != nullptr)
            weightsParameterBuilder.optimizer(convolution3d.weightsOptimizer);
        convolution3d.addParameter(std::make_shared<ParameterSpecification>(weightsParameterBuilder.build()));

        if (convolution3d.hasBias) {
            ParameterSpecification::Builder biasesParameterBuilder;
            biasesParameterBuilder.name("biases")
                .shape({convolution3d.numOutputChannels})
                .dtype(weightsDataType)
                .initializer(convolution3d.biasInitializer)
                .trainable(true);
            if (convolution3d.biasesOptimizer != nullptr)
                biasesParameterBuilder.optimizer(convolution3d.biasesOptimizer);
            convolution3d.addParameter(std::make_shared<ParameterSpecification>(biasesParameterBuilder.build()));
        }

        convolution3d.initialized = true;

        if (convolution3d.isMultiLayer()) {
            convolution3d.buildSupportLayersAndAddToNetwork(_network.value());
        } else {
            for (uint32_t i = 0; i < convolution3d.featureInputs.size(); ++i) {
                convolution3d.featureOutputs.push_back(
                    Tensor(convolutionDataType, {_numOutputChannels.value(), outputDepth, outputHeight, outputWidth}));
                convolution3d.outputTensorFromInputTensor[convolution3d.featureInputs[i]] = convolution3d.featureOutputs[i];
                convolution3d.inputTensorFromOutputTensor[convolution3d.featureOutputs[i]] = convolution3d.featureInputs[i];
            }

            convolution3d.standaloneLayerFeatureInputs = convolution3d.getFeatureInputs();
            convolution3d.standaloneLayerFeatureOutputs = convolution3d.getFeatureOutputs();
            convolution3d.addToNetwork(_network.value());
        }

        return convolution3d;
    }

    virtual Convolution3d::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Convolution3d::Builder& featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(_featureInput.getDimensions().size() == 4);
        this->_featureInputs.push_back(_featureInput);
        return *this;
    }

    virtual Convolution3d::Builder& numOutputChannels(uint32_t _numOutputChannels) {
        THOR_THROW_IF_FALSE(!this->_numOutputChannels.has_value());
        this->_numOutputChannels = _numOutputChannels;
        return *this;
    }

    virtual Convolution3d::Builder& filterDepth(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_filterDepth.has_value());
        this->_filterDepth = value;
        return *this;
    }
    virtual Convolution3d::Builder& filterHeight(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_filterHeight.has_value());
        this->_filterHeight = value;
        return *this;
    }
    virtual Convolution3d::Builder& filterWidth(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_filterWidth.has_value());
        this->_filterWidth = value;
        return *this;
    }
    virtual Convolution3d::Builder& depthStride(uint32_t value) {
        THOR_THROW_IF_FALSE(value != 0);
        THOR_THROW_IF_FALSE(!this->_depthStride.has_value());
        this->_depthStride = value;
        return *this;
    }
    virtual Convolution3d::Builder& verticalStride(uint32_t value) {
        THOR_THROW_IF_FALSE(value != 0);
        THOR_THROW_IF_FALSE(!this->_verticalStride.has_value());
        this->_verticalStride = value;
        return *this;
    }
    virtual Convolution3d::Builder& horizontalStride(uint32_t value) {
        THOR_THROW_IF_FALSE(value != 0);
        THOR_THROW_IF_FALSE(!this->_horizontalStride.has_value());
        this->_horizontalStride = value;
        return *this;
    }
    virtual Convolution3d::Builder& depthPadding(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_depthPadding.has_value());
        this->_depthPadding = value;
        return *this;
    }
    virtual Convolution3d::Builder& verticalPadding(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.has_value());
        this->_verticalPadding = value;
        return *this;
    }
    virtual Convolution3d::Builder& horizontalPadding(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.has_value());
        this->_horizontalPadding = value;
        return *this;
    }
    virtual Convolution3d::Builder& hasBias(bool value) {
        THOR_THROW_IF_FALSE(!this->_hasBias.has_value());
        this->_hasBias = value;
        return *this;
    }
    virtual Convolution3d::Builder& weightsInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = initializer->clone();
        return *this;
    }
    virtual Convolution3d::Builder& biasInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(this->_biasesInitializer == nullptr);
        this->_biasesInitializer = initializer->clone();
        return *this;
    }
    virtual Convolution3d::Builder& activation(std::shared_ptr<Activation> value) {
        THOR_THROW_IF_FALSE(this->_activation == nullptr);
        THOR_THROW_IF_FALSE(!_activationExplicitlyRemoved);
        this->_activation = value;
        return *this;
    }
    virtual Convolution3d::Builder& noActivation() {
        THOR_THROW_IF_FALSE(!this->_activation);
        _activationExplicitlyRemoved = true;
        return *this;
    }
    virtual Convolution3d::Builder& epilogue(const ThorImplementation::Expression &expression) {
        THOR_THROW_IF_FALSE(!this->_epilogue.has_value());
        Convolution3d::validateEpilogueExpression(expression);
        _epilogue = expression;
        return *this;
    }

    virtual Convolution3d::Builder& weightsOptimizer(std::shared_ptr<Optimizer> value) {
        THOR_THROW_IF_FALSE(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = value;
        return *this;
    }
    virtual Convolution3d::Builder& biasesOptimizer(std::shared_ptr<Optimizer> value) {
        THOR_THROW_IF_FALSE(this->_biasesOptimizer == nullptr);
        this->_biasesOptimizer = value;
        return *this;
    }

    static uint32_t computeOutputDimension(uint32_t inputSize, uint32_t stride, uint32_t filterSize, uint32_t padding) {
        THOR_THROW_IF_FALSE(filterSize <= inputSize + 2 * padding);
        THOR_THROW_IF_FALSE(stride > 0);
        return 1 + (((inputSize + 2 * padding) - filterSize) / stride);
    }

   private:
    std::optional<Network*> _network;
    std::vector<Tensor> _featureInputs;
    std::optional<uint32_t> _numOutputChannels;
    std::optional<uint32_t> _filterDepth;
    std::optional<uint32_t> _filterHeight;
    std::optional<uint32_t> _filterWidth;
    std::optional<uint32_t> _depthStride;
    std::optional<uint32_t> _verticalStride;
    std::optional<uint32_t> _horizontalStride;
    std::optional<uint32_t> _depthPadding;
    std::optional<uint32_t> _verticalPadding;
    std::optional<uint32_t> _horizontalPadding;
    std::optional<bool> _hasBias;
    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasesInitializer;
    std::shared_ptr<Activation> _activation;
    bool _activationExplicitlyRemoved;
    std::shared_ptr<Optimizer> _weightsOptimizer;
    std::shared_ptr<Optimizer> _biasesOptimizer;
    std::optional<ThorImplementation::Expression> _epilogue;
};

}  // namespace Thor
