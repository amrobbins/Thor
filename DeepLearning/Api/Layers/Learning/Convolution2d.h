#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2d.h"
#include "Utilities/Exceptions.h"
#include "Utilities/Expression/ConvolutionSpatial.h"
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>

namespace Thor {

enum class ConvolutionPaddingMode {
    VALID,
    SAME_UPPER,
    EXPLICIT,
};

class Convolution2d : public TrainableLayer {
   public:
    class Builder;

    Convolution2d() {}
    explicit Convolution2d(const std::optional<ThorImplementation::Expression> epilogue,
                           std::vector<std::pair<std::string, Tensor>> epilogueInputBindings = {})
        : epilogue(epilogue), epilogueInputBindings(std::move(epilogueInputBindings)) {}
    ~Convolution2d() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Convolution2d>(*this); }

    virtual uint32_t getFilterHeight() { return filterHeight; }
    virtual uint32_t getFilterWidth() { return filterWidth; }
    virtual uint32_t getGroups() const { return groups; }
    virtual uint32_t getVerticalStride() { return static_cast<uint32_t>(spatial.stride_h); }
    virtual uint32_t getHorizontalStride() { return static_cast<uint32_t>(spatial.stride_w); }
    virtual uint32_t getVerticalDilation() { return static_cast<uint32_t>(spatial.dilation_h); }
    virtual uint32_t getHorizontalDilation() { return static_cast<uint32_t>(spatial.dilation_w); }
    virtual uint32_t getPaddingTop() { return static_cast<uint32_t>(spatial.pre_padding_h); }
    virtual uint32_t getPaddingBottom() { return static_cast<uint32_t>(spatial.post_padding_h); }
    virtual uint32_t getPaddingLeft() { return static_cast<uint32_t>(spatial.pre_padding_w); }
    virtual uint32_t getPaddingRight() { return static_cast<uint32_t>(spatial.post_padding_w); }
    virtual ConvolutionPaddingMode getPaddingMode() const { return paddingMode; }
    DataType getComputeDataType() const { return computeDataType; }

    std::string getLayerType() const override { return "Convolution2d"; }
    std::string getLayerVersion() const override { return "4.0.0"; }

    nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork &stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);
    nlohmann::json architectureJson() const override;

    static const char *epilogueInputName() { return "__convolution_2d_epilogue_input"; }
    static const char *epilogueOutputName() { return "__convolution_2d_epilogue_output"; }

    [[nodiscard]] static ThorImplementation::Expression epilogueInput(
        std::optional<ThorImplementation::DataType> computeDType =
            std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType =
            std::nullopt) {
        return LayerEpilogue::input(epilogueInputName(), computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueAuxInput(
        const std::string &inputName,
        std::optional<ThorImplementation::DataType> computeDType =
            std::nullopt,
        std::optional<ThorImplementation::DataType> outputDType =
            std::nullopt) {
        validateEpilogueAuxInputName(inputName);
        return LayerEpilogue::input(inputName, computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(const ThorImplementation::Expression &expression) {
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), epilogueOutputName(), "Convolution2d");
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(
        const ThorImplementation::Expression &expression,
        const std::vector<std::string> &auxiliaryInputNames) {
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Convolution2d");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression &expression) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), epilogueOutputName(), "Convolution2d");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression &expression,
                                           const std::vector<std::string> &auxiliaryInputNames) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Convolution2d");
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), epilogueOutputName(), "Convolution2d");
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition,
                                           const std::vector<std::string> &auxiliaryInputNames) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), auxiliaryInputNames, epilogueOutputName(), "Convolution2d");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition &definition) {
        return LayerEpilogue::expressionFromDefinition(definition, epilogueInputName(), epilogueOutputName(), "Convolution2d");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition &definition,
        const std::vector<std::string> &auxiliaryInputNames) {
        return LayerEpilogue::expressionFromDefinition(definition,
                                                       epilogueInputName(),
                                                       auxiliaryInputNames,
                                                       epilogueOutputName(),
                                                       "Convolution2d");
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
    bool mustConnectAllInputsToDriveOutput() const override { return !epilogueInputBindings.empty(); }

   protected:
    virtual bool isMultiLayer() const {
        return useBatchNormalization || dropProportion > 0.0f;
    }
    virtual void buildSupportLayersAndAddToNetwork(Network *network);

    void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) override {
        (void)inputTensor;
        (void)batchSize;
        (void)stream;
        // Conv2D lowers through cuDNN Frontend expression graphs; there is no
        // separate classic-cuDNN kernel-selection pre-warm step.
    }

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                  std::optional<Event> sisterLayerLoadedEvent);

    std::vector<Tensor> standaloneLayerFeatureInputs;
    std::vector<Tensor> standaloneLayerFeatureOutputs;

   private:
    uint32_t numOutputChannels;
    uint32_t filterHeight;
    uint32_t filterWidth;
    uint32_t groups = 1;
    ThorImplementation::ConvolutionSpatial2d spatial;
    ConvolutionPaddingMode paddingMode = ConvolutionPaddingMode::VALID;
    bool hasBias;
    DataType computeDataType = DataType::FP32;
    std::shared_ptr<Initializer> weightsInitializer;
    std::shared_ptr<Initializer> biasInitializer;
    std::shared_ptr<Activation> activation;
    std::shared_ptr<Optimizer> weightsOptimizer;
    std::shared_ptr<Optimizer> biasesOptimizer;

    float dropProportion;

    bool useBatchNormalization;
    std::optional<double> batchNormExponentialRunningAverageFactor;
    std::optional<double> batchNormEpsilon;

    const std::optional<ThorImplementation::Expression> epilogue;
    std::vector<std::pair<std::string, Tensor>> epilogueInputBindings;
    mutable std::optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;

    std::vector<std::string> epilogueAuxInputNames() const;
    std::vector<uint32_t> inputPortIndicesForTensor(Tensor tensor) const;

    std::set<uint32_t> connectedInputPortIndices;
    bool emittedFeatureOutputAfterAllInputsConnected = false;
    mutable std::unordered_map<uint64_t, uint32_t> nextInputConnectionCursorByTensorOriginalId;
};

// featureInput, numOutputChannels, filterHeight and filterWidth are required, all other parameters are optional.
class Convolution2d::Builder {
   public:
    virtual ~Builder() = default;

    Builder() { _activationExplicitlyRemoved = false; }

    virtual Convolution2d build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(!_featureInputs.empty());
        THOR_THROW_IF_FALSE(_numOutputChannels.has_value());
        THOR_THROW_IF_FALSE(_filterHeight.has_value());
        THOR_THROW_IF_FALSE(_filterWidth.has_value());

        if (!_verticalStride.has_value())
            _verticalStride = 1;
        if (!_horizontalStride.has_value())
            _horizontalStride = 1;
        if (!_verticalDilation.has_value())
            _verticalDilation = 1;
        if (!_horizontalDilation.has_value())
            _horizontalDilation = 1;
        if (!_groups.has_value())
            _groups = 1;
        // Unspecified padding is VALID/no padding. SAME and EXPLICIT must be requested explicitly.
        if (!_paddingMode.has_value())
            _paddingMode = ConvolutionPaddingMode::VALID;
        if (!_hasBias.has_value())
            _hasBias = false;
        if (!_computeDataType.has_value())
            _computeDataType = DataType::FP32;
        if (_weightsInitializer == nullptr)
            _weightsInitializer = Glorot::Builder().build();
        if (_biasesInitializer == nullptr)
            _biasesInitializer = Glorot::Builder().build();
        if (!_activation && !_activationExplicitlyRemoved)
            _activation = Gelu::Builder().build();
        if (!_dropProportion.has_value())
            _dropProportion = 0.0f;
        if (!_useBatchNormalization.has_value()) {
            _useBatchNormalization = false;
        }

        if (!_epilogueInputBindings.empty() && _featureInputs.size() != 1) {
            throw std::invalid_argument("Convolution2d epilogue auxiliary inputs currently require exactly one feature input.");
        }

        if (_epilogue.has_value()) {
            Convolution2d::validateEpilogueExpression(_epilogue.value(), epilogueAuxInputNames());
        } else if (!_epilogueInputBindings.empty()) {
            throw std::invalid_argument("Convolution2d epilogue_inputs were provided without an epilogue expression.");
        }

        Convolution2d convolution2d(_epilogue, _epilogueInputBindings);

        convolution2d.featureInputs = _featureInputs;
        convolution2d.numOutputChannels = _numOutputChannels.value();
        convolution2d.filterHeight = _filterHeight.value();
        convolution2d.filterWidth = _filterWidth.value();
        convolution2d.groups = _groups.value();
        convolution2d.spatial.stride_h = static_cast<int32_t>(_verticalStride.value());
        convolution2d.spatial.stride_w = static_cast<int32_t>(_horizontalStride.value());
        convolution2d.spatial.dilation_h = static_cast<int32_t>(_verticalDilation.value());
        convolution2d.spatial.dilation_w = static_cast<int32_t>(_horizontalDilation.value());
        convolution2d.paddingMode = _paddingMode.value();
        switch (convolution2d.paddingMode) {
            case ConvolutionPaddingMode::VALID:
                convolution2d.spatial.pre_padding_h = 0;
                convolution2d.spatial.post_padding_h = 0;
                convolution2d.spatial.pre_padding_w = 0;
                convolution2d.spatial.post_padding_w = 0;
                break;
            case ConvolutionPaddingMode::SAME_UPPER: {
                const auto [paddingTop, paddingBottom] =
                    computeSamePadding(convolution2d.featureInputs[0].getDimensions()[1],
                                       static_cast<uint32_t>(convolution2d.spatial.stride_h),
                                       convolution2d.filterHeight,
                                       static_cast<uint32_t>(convolution2d.spatial.dilation_h));
                const auto [paddingLeft, paddingRight] =
                    computeSamePadding(convolution2d.featureInputs[0].getDimensions()[2],
                                       static_cast<uint32_t>(convolution2d.spatial.stride_w),
                                       convolution2d.filterWidth,
                                       static_cast<uint32_t>(convolution2d.spatial.dilation_w));
                convolution2d.spatial.pre_padding_h = static_cast<int32_t>(paddingTop);
                convolution2d.spatial.post_padding_h = static_cast<int32_t>(paddingBottom);
                convolution2d.spatial.pre_padding_w = static_cast<int32_t>(paddingLeft);
                convolution2d.spatial.post_padding_w = static_cast<int32_t>(paddingRight);
                break;
            }
            case ConvolutionPaddingMode::EXPLICIT:
                convolution2d.spatial.pre_padding_h = static_cast<int32_t>(_paddingTop.value());
                convolution2d.spatial.post_padding_h = static_cast<int32_t>(_paddingBottom.value());
                convolution2d.spatial.pre_padding_w = static_cast<int32_t>(_paddingLeft.value());
                convolution2d.spatial.post_padding_w = static_cast<int32_t>(_paddingRight.value());
                break;
        }

        uint32_t outputHeight = computeOutputDimension(convolution2d.featureInputs[0].getDimensions()[1],
                                                       static_cast<uint32_t>(convolution2d.spatial.stride_h),
                                                       convolution2d.filterHeight,
                                                       static_cast<uint32_t>(convolution2d.spatial.pre_padding_h),
                                                       static_cast<uint32_t>(convolution2d.spatial.post_padding_h),
                                                       static_cast<uint32_t>(convolution2d.spatial.dilation_h));
        uint32_t outputWidth = computeOutputDimension(convolution2d.featureInputs[0].getDimensions()[2],
                                                      static_cast<uint32_t>(convolution2d.spatial.stride_w),
                                                      convolution2d.filterWidth,
                                                      static_cast<uint32_t>(convolution2d.spatial.pre_padding_w),
                                                      static_cast<uint32_t>(convolution2d.spatial.post_padding_w),
                                                      static_cast<uint32_t>(convolution2d.spatial.dilation_w));

        convolution2d.hasBias = _hasBias.value();
        convolution2d.computeDataType = _computeDataType.value();
        convolution2d.weightsInitializer = _weightsInitializer->clone();
        convolution2d.biasInitializer = _biasesInitializer->clone();
        if (_activation != nullptr)
            convolution2d.activation = _activation;
        convolution2d.dropProportion = _dropProportion.value();
        convolution2d.useBatchNormalization = _useBatchNormalization.value();
        convolution2d.batchNormExponentialRunningAverageFactor = _batchNormExponentialRunningAverageFactor;
        convolution2d.batchNormEpsilon = _batchNormEpsilon;

        // When this layer gets a specific optimizer, set it now, otherwise network will attach the network default optimizer to it.
        convolution2d.weightsOptimizer = _weightsOptimizer;
        convolution2d.biasesOptimizer = _biasesOptimizer;

        const DataType convolutionDataType = convolution2d.featureInputs.front().getDataType();
        if (convolution2d.computeDataType == DataType::TF32 && convolutionDataType != DataType::FP32)
            throw std::invalid_argument("Convolution2d TF32 compute requires FP32 input/weights/output storage.");
        const DataType weightsDataType = convolutionDataType;
        const uint64_t inputChannels = convolution2d.featureInputs.front().getDimensions()[0];
        if (inputChannels % convolution2d.groups != 0 || convolution2d.numOutputChannels % convolution2d.groups != 0)
            throw std::invalid_argument("Convolution2d requires input and output channels divisible by groups.");

        ParameterSpecification::Builder weightsParameterBuilder;
        weightsParameterBuilder.name("weights")
            .shape({convolution2d.numOutputChannels,
                    inputChannels / convolution2d.groups,
                    convolution2d.filterHeight,
                    convolution2d.filterWidth})
            .dtype(weightsDataType)
            .initializer(convolution2d.weightsInitializer)
            .trainable(true);
        if (convolution2d.weightsOptimizer != nullptr)
            weightsParameterBuilder.optimizer(convolution2d.weightsOptimizer);
        convolution2d.addParameter(std::make_shared<ParameterSpecification>(weightsParameterBuilder.build()));

        if (convolution2d.hasBias) {
            ParameterSpecification::Builder biasesParameterBuilder;
            biasesParameterBuilder.name("biases")
                .shape({convolution2d.numOutputChannels})
                .dtype(weightsDataType)
                .initializer(convolution2d.biasInitializer)
                .trainable(true);
            if (convolution2d.biasesOptimizer != nullptr)
                biasesParameterBuilder.optimizer(convolution2d.biasesOptimizer);
            convolution2d.addParameter(std::make_shared<ParameterSpecification>(biasesParameterBuilder.build()));
        }

        convolution2d.initialized = true;

        if (convolution2d.isMultiLayer()) {
            convolution2d.buildSupportLayersAndAddToNetwork(_network.value());
        } else {
            for (uint32_t i = 0; i < convolution2d.featureInputs.size(); ++i) {
                convolution2d.featureOutputs.push_back(Tensor(convolutionDataType, {_numOutputChannels.value(), outputHeight, outputWidth}));
                convolution2d.outputTensorFromInputTensor[convolution2d.featureInputs[i]] = convolution2d.featureOutputs[i];
                convolution2d.inputTensorFromOutputTensor[convolution2d.featureOutputs[i]] = convolution2d.featureInputs[i];
            }
            for (const auto &[name, tensor] : convolution2d.epilogueInputBindings) {
                (void)name;
                THOR_THROW_IF_FALSE(tensor.getDataType() == convolutionDataType);
                THOR_THROW_IF_FALSE(tensor.getDimensions() == convolution2d.featureOutputs[0].getDimensions());
                convolution2d.outputTensorFromInputTensor[tensor] = convolution2d.featureOutputs[0];
            }

            convolution2d.standaloneLayerFeatureInputs = convolution2d.featureInputs;
            convolution2d.standaloneLayerFeatureOutputs = convolution2d.getFeatureOutputs();

            convolution2d.addToNetwork(_network.value());
        }

        return convolution2d;
    }

    virtual Convolution2d::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Convolution2d::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(_featureInput.getDimensions().size() == 3);
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            THOR_THROW_IF_FALSE(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            THOR_THROW_IF_FALSE(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual Convolution2d::Builder &numOutputChannels(uint32_t _numOutputChannels) {
        THOR_THROW_IF_FALSE(!this->_numOutputChannels.has_value());
        this->_numOutputChannels = _numOutputChannels;
        return *this;
    }

    virtual Convolution2d::Builder &filterHeight(uint32_t _filterHeight) {
        THOR_THROW_IF_FALSE(!this->_filterHeight.has_value());
        this->_filterHeight = _filterHeight;
        return *this;
    }

    virtual Convolution2d::Builder &filterWidth(uint32_t _filterWidth) {
        THOR_THROW_IF_FALSE(!this->_filterWidth.has_value());
        this->_filterWidth = _filterWidth;
        return *this;
    }

    virtual Convolution2d::Builder &groups(uint32_t value) {
        THOR_THROW_IF_FALSE(value > 0);
        THOR_THROW_IF_FALSE(!this->_groups.has_value());
        this->_groups = value;
        return *this;
    }

    virtual Convolution2d::Builder &verticalStride(uint32_t _verticalStride) {
        THOR_THROW_IF_FALSE(_verticalStride != 0);
        THOR_THROW_IF_FALSE(!this->_verticalStride.has_value());
        this->_verticalStride = _verticalStride;
        return *this;
    }

    virtual Convolution2d::Builder &horizontalStride(uint32_t _horizontalStride) {
        THOR_THROW_IF_FALSE(_horizontalStride != 0);
        THOR_THROW_IF_FALSE(!this->_horizontalStride.has_value());
        this->_horizontalStride = _horizontalStride;
        return *this;
    }

    virtual Convolution2d::Builder &verticalDilation(uint32_t _verticalDilation) {
        THOR_THROW_IF_FALSE(_verticalDilation != 0);
        THOR_THROW_IF_FALSE(!this->_verticalDilation.has_value());
        this->_verticalDilation = _verticalDilation;
        return *this;
    }

    virtual Convolution2d::Builder &horizontalDilation(uint32_t _horizontalDilation) {
        THOR_THROW_IF_FALSE(_horizontalDilation != 0);
        THOR_THROW_IF_FALSE(!this->_horizontalDilation.has_value());
        this->_horizontalDilation = _horizontalDilation;
        return *this;
    }

    virtual Convolution2d::Builder &dilation(uint32_t dilation) {
        THOR_THROW_IF_FALSE(dilation != 0);
        THOR_THROW_IF_FALSE(!this->_verticalDilation.has_value());
        THOR_THROW_IF_FALSE(!this->_horizontalDilation.has_value());
        this->_verticalDilation = dilation;
        this->_horizontalDilation = dilation;
        return *this;
    }

    virtual Convolution2d::Builder &validPadding() {
        THOR_THROW_IF_FALSE(!_paddingMode.has_value());
        _paddingMode = ConvolutionPaddingMode::VALID;
        return *this;
    }

    virtual Convolution2d::Builder &samePadding() {
        THOR_THROW_IF_FALSE(!_paddingMode.has_value());
        _paddingMode = ConvolutionPaddingMode::SAME_UPPER;
        return *this;
    }

    virtual Convolution2d::Builder &padding(uint32_t top, uint32_t bottom, uint32_t left, uint32_t right) {
        THOR_THROW_IF_FALSE(!_paddingMode.has_value());
        _paddingTop = top;
        _paddingBottom = bottom;
        _paddingLeft = left;
        _paddingRight = right;
        _paddingMode = ConvolutionPaddingMode::EXPLICIT;
        return *this;
    }

    virtual Convolution2d::Builder &hasBias(bool _hasBias) {
        THOR_THROW_IF_FALSE(!this->_hasBias.has_value());
        this->_hasBias = _hasBias;
        return *this;
    }

    virtual Convolution2d::Builder &computeDataType(DataType value) {
        THOR_THROW_IF_FALSE(!_computeDataType.has_value());
        if (value != DataType::FP32 && value != DataType::TF32)
            throw std::invalid_argument("Convolution2d computeDataType must be fp32 or tf32.");
        _computeDataType = value;
        return *this;
    }

    virtual Convolution2d::Builder &weightsInitializer(std::shared_ptr<Initializer> &_weightsInitializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual Convolution2d::Builder &weightsInitializer(std::shared_ptr<Initializer> &&_weightsInitializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual Convolution2d::Builder &biasInitializer(std::shared_ptr<Initializer> &_biasInitializer) {
        THOR_THROW_IF_FALSE(this->_biasesInitializer == nullptr);
        this->_biasesInitializer = _biasInitializer->clone();
        return *this;
    }

    virtual Convolution2d::Builder &biasInitializer(std::shared_ptr<Initializer> &&_biasInitializer) {
        THOR_THROW_IF_FALSE(this->_biasesInitializer == nullptr);
        this->_biasesInitializer = _biasInitializer->clone();
        return *this;
    }

    // Adds an activation layer after this Convolution2d layer
    virtual Convolution2d::Builder &activation(std::shared_ptr<Activation> &_activation) {
        THOR_THROW_IF_FALSE(this->_activation == nullptr);
        THOR_THROW_IF_FALSE(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual Convolution2d::Builder &activation(std::shared_ptr<Activation> &&_activation) {
        THOR_THROW_IF_FALSE(this->_activation == nullptr);
        THOR_THROW_IF_FALSE(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual Convolution2d::Builder &noActivation() {
        THOR_THROW_IF_FALSE(!this->_activation);

        _activationExplicitlyRemoved = true;
        return *this;
    }

    virtual Convolution2d::Builder &epilogue(const ThorImplementation::Expression &expression) {
        THOR_THROW_IF_FALSE(!this->_epilogue.has_value());
        Convolution2d::validateEpilogueExpression(expression, epilogueAuxInputNames());
        _epilogue = expression;
        return *this;
    }

    virtual Convolution2d::Builder &epilogueInput(const std::string &inputName, Tensor tensor) {
        Convolution2d::validateEpilogueAuxInputName(inputName);
        THOR_THROW_IF_FALSE(tensor.isInitialized());
        for (const auto &[existingName, existingTensor] : _epilogueInputBindings) {
            (void)existingTensor;
            if (existingName == inputName) {
                throw std::invalid_argument("Convolution2d epilogue input name is duplicated: " + inputName + ".");
            }
        }
        _epilogueInputBindings.emplace_back(inputName, tensor);
        if (_epilogue.has_value()) {
            Convolution2d::validateEpilogueExpression(_epilogue.value(), epilogueAuxInputNames());
        }
        return *this;
    }

    virtual Convolution2d::Builder &weightsOptimizer(std::shared_ptr<Optimizer> _weightsOptimizer) {
        THOR_THROW_IF_FALSE(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = _weightsOptimizer;
        return *this;
    }

    virtual Convolution2d::Builder &biasesOptimizer(std::shared_ptr<Optimizer> _biasesOptimizer) {
        THOR_THROW_IF_FALSE(this->_biasesOptimizer == nullptr);
        this->_biasesOptimizer = _biasesOptimizer;
        return *this;
    }

    // Adds a BatchNormalization layer before this Convolution2d layer and before the DropOut layer when that is also present
    // exponentialRunningAverageFactor and epsilon will be set to good default values when not specified.
    virtual Convolution2d::Builder &batchNormalization(
        std::optional<double> exponentialRunningAverageFactor = std::nullopt,
        std::optional<double> epsilon = std::nullopt) {
        THOR_THROW_IF_FALSE(!_useBatchNormalization.has_value());
        this->_useBatchNormalization = true;
        this->_batchNormExponentialRunningAverageFactor = exponentialRunningAverageFactor;
        this->_batchNormEpsilon = epsilon;
        return *this;
    }

    // Adds a DropOut layer before this Convolution2d layer, but after the BatchNormalization layer when that is also present.
    virtual Convolution2d::Builder &dropOut(float _dropProportion) {
        THOR_THROW_IF_FALSE(!this->_dropProportion.has_value());
        this->_dropProportion = _dropProportion;
        return *this;
    }

    static uint32_t computeOutputDimension(uint32_t inputSize,
                                           uint32_t stride,
                                           uint32_t filterSize,
                                           uint32_t prePadding,
                                           uint32_t postPadding,
                                           uint32_t dilation = 1) {
        THOR_THROW_IF_FALSE(stride > 0);
        THOR_THROW_IF_FALSE(dilation > 0);
        const uint64_t effectiveFilter = static_cast<uint64_t>(dilation) * (filterSize - 1ULL) + 1ULL;
        const uint64_t paddedInput = static_cast<uint64_t>(inputSize) + prePadding + postPadding;
        THOR_THROW_IF_FALSE(effectiveFilter <= paddedInput);
        return static_cast<uint32_t>(1ULL + (paddedInput - effectiveFilter) / stride);
    }

    static std::pair<uint32_t, uint32_t> computeSamePadding(
        uint32_t inputSize, uint32_t stride, uint32_t filterSize, uint32_t dilation = 1) {
        THOR_THROW_IF_FALSE(inputSize > 0);
        THOR_THROW_IF_FALSE(stride > 0);
        THOR_THROW_IF_FALSE(filterSize > 0);
        THOR_THROW_IF_FALSE(dilation > 0);

        const uint64_t outputSize =
            (static_cast<uint64_t>(inputSize) + static_cast<uint64_t>(stride) - 1ULL) / static_cast<uint64_t>(stride);
        const uint64_t effectiveFilter =
            static_cast<uint64_t>(dilation) * (static_cast<uint64_t>(filterSize) - 1ULL) + 1ULL;
        const uint64_t coveredInput =
            outputSize > 0 ? (outputSize - 1ULL) * static_cast<uint64_t>(stride) + effectiveFilter : 0ULL;
        const uint64_t totalPadding =
            coveredInput > static_cast<uint64_t>(inputSize) ? coveredInput - static_cast<uint64_t>(inputSize) : 0ULL;

        const uint32_t prePadding = static_cast<uint32_t>(totalPadding / 2ULL);
        const uint32_t postPadding = static_cast<uint32_t>(totalPadding - prePadding);
        return {prePadding, postPadding};
    }

   private:
    std::optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    std::optional<uint32_t> _numOutputChannels;
    std::optional<uint32_t> _filterHeight;
    std::optional<uint32_t> _filterWidth;
    std::optional<uint32_t> _groups;
    std::optional<uint32_t> _verticalStride;
    std::optional<uint32_t> _horizontalStride;
    std::optional<uint32_t> _verticalDilation;
    std::optional<uint32_t> _horizontalDilation;
    std::optional<ConvolutionPaddingMode> _paddingMode;
    std::optional<uint32_t> _paddingTop;
    std::optional<uint32_t> _paddingBottom;
    std::optional<uint32_t> _paddingLeft;
    std::optional<uint32_t> _paddingRight;
    std::optional<bool> _hasBias;
    std::optional<DataType> _computeDataType;
    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasesInitializer;
    std::shared_ptr<Activation> _activation;
    bool _activationExplicitlyRemoved;
    std::shared_ptr<Optimizer> _weightsOptimizer;
    std::shared_ptr<Optimizer> _biasesOptimizer;

    std::optional<float> _dropProportion;

    std::optional<bool> _useBatchNormalization;
    std::optional<double> _batchNormExponentialRunningAverageFactor;
    std::optional<double> _batchNormEpsilon;
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
