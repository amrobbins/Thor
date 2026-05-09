#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution3d.h"
#include "Utilities/Exceptions.h"

namespace Thor {

class Convolution3d : public TrainableLayer {
   public:
    class Builder;

    Convolution3d() {}
    explicit Convolution3d(const Optional<ThorImplementation::Expression> epilogue) : epilogue(epilogue) {}
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
        Optional<ThorImplementation::TensorDescriptor::DataType> computeDType =
            Optional<ThorImplementation::TensorDescriptor::DataType>::empty(),
        Optional<ThorImplementation::TensorDescriptor::DataType> outputDType =
            Optional<ThorImplementation::TensorDescriptor::DataType>::empty()) {
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
    virtual bool isMultiLayer() const { return featureInputs.front().getDataType() != Tensor::DataType::FP16; }
    virtual void buildSupportLayersAndAddToNetwork(Network* network);

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                  Optional<Event> sisterLayerLoadedEvent) {
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

    const Optional<ThorImplementation::Expression> epilogue;
    mutable Optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;
};

class Convolution3d::Builder {
   public:
    virtual ~Builder() = default;

    Builder() { _activationExplicitlyRemoved = false; }

    virtual Convolution3d build() {
        THOR_THROW_IF_FALSE(_network.isPresent());
        THOR_THROW_IF_FALSE(!_featureInputs.empty());
        THOR_THROW_IF_FALSE(_numOutputChannels.isPresent());
        THOR_THROW_IF_FALSE(_filterDepth.isPresent());
        THOR_THROW_IF_FALSE(_filterHeight.isPresent());
        THOR_THROW_IF_FALSE(_filterWidth.isPresent());

        if (_depthStride.isEmpty())
            _depthStride = 1;
        if (_verticalStride.isEmpty())
            _verticalStride = 1;
        if (_horizontalStride.isEmpty())
            _horizontalStride = 1;
        if (_depthPadding.isEmpty())
            _depthPadding = 0;
        if (_verticalPadding.isEmpty())
            _verticalPadding = 0;
        if (_horizontalPadding.isEmpty())
            _horizontalPadding = 0;
        if (_hasBias.isEmpty())
            _hasBias = false;
        if (_weightsInitializer == nullptr)
            _weightsInitializer = Glorot::Builder().build();
        if (_biasesInitializer == nullptr)
            _biasesInitializer = Glorot::Builder().build();
        if (!_activation && !_activationExplicitlyRemoved)
            _activation = SoftPlus::Builder().build();

        if (_epilogue.isPresent()) {
            Convolution3d::validateEpilogueExpression(_epilogue.get());
        }

        Convolution3d convolution3d(_epilogue);
        convolution3d.featureInputs = _featureInputs;
        convolution3d.numOutputChannels = _numOutputChannels;
        convolution3d.filterDepth = _filterDepth;
        convolution3d.filterHeight = _filterHeight;
        convolution3d.filterWidth = _filterWidth;
        convolution3d.depthStride = _depthStride;
        convolution3d.verticalStride = _verticalStride;
        convolution3d.horizontalStride = _horizontalStride;
        convolution3d.depthPadding = _depthPadding;
        convolution3d.verticalPadding = _verticalPadding;
        convolution3d.horizontalPadding = _horizontalPadding;

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

        convolution3d.hasBias = _hasBias;
        convolution3d.weightsInitializer = _weightsInitializer->clone();
        convolution3d.biasInitializer = _biasesInitializer->clone();
        if (_activation != nullptr)
            convolution3d.activation = _activation;
        convolution3d.weightsOptimizer = _weightsOptimizer;
        convolution3d.biasesOptimizer = _biasesOptimizer;
        convolution3d.initialized = true;

        if (convolution3d.isMultiLayer()) {
            convolution3d.buildSupportLayersAndAddToNetwork(_network);
        } else {
            for (uint32_t i = 0; i < convolution3d.featureInputs.size(); ++i) {
                convolution3d.featureOutputs.push_back(
                    Tensor(Tensor::DataType::FP16, {_numOutputChannels, outputDepth, outputHeight, outputWidth}));
                convolution3d.outputTensorFromInputTensor[convolution3d.featureInputs[i]] = convolution3d.featureOutputs[i];
                convolution3d.inputTensorFromOutputTensor[convolution3d.featureOutputs[i]] = convolution3d.featureInputs[i];
            }

            convolution3d.standaloneLayerFeatureInputs = convolution3d.getFeatureInputs();
            convolution3d.standaloneLayerFeatureOutputs = convolution3d.getFeatureOutputs();
            convolution3d.addToNetwork(_network.get());
        }

        return convolution3d;
    }

    virtual Convolution3d::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Convolution3d::Builder& featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(_featureInput.getDimensions().size() == 4);
        this->_featureInputs.push_back(_featureInput);
        return *this;
    }

    virtual Convolution3d::Builder& numOutputChannels(uint32_t _numOutputChannels) {
        THOR_THROW_IF_FALSE(!this->_numOutputChannels.isPresent());
        this->_numOutputChannels = _numOutputChannels;
        return *this;
    }

    virtual Convolution3d::Builder& filterDepth(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_filterDepth.isPresent());
        this->_filterDepth = value;
        return *this;
    }
    virtual Convolution3d::Builder& filterHeight(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_filterHeight.isPresent());
        this->_filterHeight = value;
        return *this;
    }
    virtual Convolution3d::Builder& filterWidth(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_filterWidth.isPresent());
        this->_filterWidth = value;
        return *this;
    }
    virtual Convolution3d::Builder& depthStride(uint32_t value) {
        THOR_THROW_IF_FALSE(value != 0);
        THOR_THROW_IF_FALSE(!this->_depthStride.isPresent());
        this->_depthStride = value;
        return *this;
    }
    virtual Convolution3d::Builder& verticalStride(uint32_t value) {
        THOR_THROW_IF_FALSE(value != 0);
        THOR_THROW_IF_FALSE(!this->_verticalStride.isPresent());
        this->_verticalStride = value;
        return *this;
    }
    virtual Convolution3d::Builder& horizontalStride(uint32_t value) {
        THOR_THROW_IF_FALSE(value != 0);
        THOR_THROW_IF_FALSE(!this->_horizontalStride.isPresent());
        this->_horizontalStride = value;
        return *this;
    }
    virtual Convolution3d::Builder& depthPadding(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_depthPadding.isPresent());
        this->_depthPadding = value;
        return *this;
    }
    virtual Convolution3d::Builder& verticalPadding(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.isPresent());
        this->_verticalPadding = value;
        return *this;
    }
    virtual Convolution3d::Builder& horizontalPadding(uint32_t value) {
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.isPresent());
        this->_horizontalPadding = value;
        return *this;
    }
    virtual Convolution3d::Builder& hasBias(bool value) {
        THOR_THROW_IF_FALSE(!this->_hasBias.isPresent());
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
        THOR_THROW_IF_FALSE(this->_epilogue.isEmpty());
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
    Optional<Network*> _network;
    std::vector<Tensor> _featureInputs;
    Optional<uint32_t> _numOutputChannels;
    Optional<uint32_t> _filterDepth;
    Optional<uint32_t> _filterHeight;
    Optional<uint32_t> _filterWidth;
    Optional<uint32_t> _depthStride;
    Optional<uint32_t> _verticalStride;
    Optional<uint32_t> _horizontalStride;
    Optional<uint32_t> _depthPadding;
    Optional<uint32_t> _verticalPadding;
    Optional<uint32_t> _horizontalPadding;
    Optional<bool> _hasBias;
    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasesInitializer;
    std::shared_ptr<Activation> _activation;
    bool _activationExplicitlyRemoved;
    std::shared_ptr<Optimizer> _weightsOptimizer;
    std::shared_ptr<Optimizer> _biasesOptimizer;
    Optional<ThorImplementation::Expression> _epilogue;
};

}  // namespace Thor
