#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/LayerEpilogue.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2d.h"
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/GpuConvolution/ConvolutionKernelRequirement.h"
#include "Utilities/TensorOperations/GpuConvolution/GpuConvolution.h"

namespace Thor {

class Convolution2d : public TrainableLayer {
   public:
    class Builder;

    Convolution2d() {}
    explicit Convolution2d(const Optional<ThorImplementation::Expression> epilogue) : epilogue(epilogue) {}
    ~Convolution2d() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Convolution2d>(*this); }

    virtual uint32_t getFilterHeight() { return filterHeight; }
    virtual uint32_t getFilterWidth() { return filterWidth; }
    virtual uint32_t getVerticalStride() { return verticalStride; }
    virtual uint32_t getHorizontalStride() { return horizontalStride; }
    virtual uint32_t getVerticalPadding() { return verticalPadding; }
    virtual uint32_t getHoriztonalPadding() { return horizontalPadding; }

    std::string getLayerType() const override { return "Convolution2d"; }

    nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork &stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);
    nlohmann::json architectureJson() const override;

    static const char *epilogueInputName() { return "__convolution_2d_epilogue_input"; }
    static const char *epilogueOutputName() { return "__convolution_2d_epilogue_output"; }

    [[nodiscard]] static ThorImplementation::Expression epilogueInput(
        Optional<ThorImplementation::TensorDescriptor::DataType> computeDType =
            Optional<ThorImplementation::TensorDescriptor::DataType>::empty(),
        Optional<ThorImplementation::TensorDescriptor::DataType> outputDType =
            Optional<ThorImplementation::TensorDescriptor::DataType>::empty()) {
        return LayerEpilogue::input(epilogueInputName(), computeDType, outputDType);
    }

    [[nodiscard]] static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(const ThorImplementation::Expression &expression) {
        return LayerEpilogue::makeDefinition(expression, epilogueInputName(), epilogueOutputName(), "Convolution2d");
    }

    static void validateEpilogueExpression(const ThorImplementation::Expression &expression) {
        LayerEpilogue::validateExpression(expression, epilogueInputName(), epilogueOutputName(), "Convolution2d");
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition) {
        LayerEpilogue::validateDefinition(definition, epilogueInputName(), epilogueOutputName(), "Convolution2d");
    }

    [[nodiscard]] static ThorImplementation::Expression epilogueExpressionFromDefinition(
        const ThorImplementation::ExpressionDefinition &definition) {
        return LayerEpilogue::expressionFromDefinition(definition, epilogueInputName(), epilogueOutputName(), "Convolution2d");
    }

    [[nodiscard]] static ThorImplementation::Expression applyEpilogue(const ThorImplementation::Expression &input,
                                                                      const ThorImplementation::Expression &epilogue) {
        return LayerEpilogue::apply(input, epilogue, epilogueInputName());
    }

   protected:
    virtual bool isMultiLayer() const {
        return useBatchNormalization || dropProportion > 0.0f || featureInputs.front().getDataType() != Tensor::DataType::FP16;
    }
    virtual void buildSupportLayersAndAddToNetwork(Network *network);

    void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) override {
        std::vector<uint64_t> inputDimensions = inputTensor.getDimensions();
        THOR_THROW_IF_FALSE(inputDimensions.size() == 3);

        uint32_t numInputChannels = inputDimensions[0];
        uint32_t numInputRows = inputDimensions[1];
        uint32_t numInputColumns = inputDimensions[2];
        std::string gpuType = MachineEvaluator::instance().getGpuType(stream.getGpuNum());
        ConvolutionKernelRequirement convolutionKernelRequirement(gpuType,
                                                                  filterWidth,
                                                                  filterHeight,
                                                                  horizontalStride,
                                                                  verticalStride,
                                                                  horizontalPadding,
                                                                  verticalPadding,
                                                                  numInputChannels,
                                                                  numOutputChannels,
                                                                  batchSize,
                                                                  numInputColumns,
                                                                  numInputRows);

        ThorImplementation::GpuConvolution::instance().chooseOptimalKernelForward(convolutionKernelRequirement, stream);
        ThorImplementation::GpuConvolution::instance().chooseOptimalKernelBackward(convolutionKernelRequirement, stream);
    }

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                  Optional<Event> sisterLayerLoadedEvent);

    std::vector<Tensor> standaloneLayerFeatureInputs;
    std::vector<Tensor> standaloneLayerFeatureOutputs;

   private:
    uint32_t numOutputChannels;
    uint32_t filterHeight;
    uint32_t filterWidth;
    uint32_t verticalStride;
    uint32_t horizontalStride;
    uint32_t verticalPadding;
    uint32_t horizontalPadding;
    bool hasBias;
    std::shared_ptr<Initializer> weightsInitializer;
    std::shared_ptr<Initializer> biasInitializer;
    std::shared_ptr<Activation> activation;
    std::shared_ptr<Optimizer> weightsOptimizer;
    std::shared_ptr<Optimizer> biasesOptimizer;

    float dropProportion;

    bool useBatchNormalization;
    Optional<double> batchNormExponentialRunningAverageFactor;
    Optional<double> batchNormEpsilon;

    const Optional<ThorImplementation::Expression> epilogue;
    mutable Optional<ThorImplementation::ExpressionDefinition> serializableEpilogue;
};

// featureInput, numOutputChannels, filterHeight and filterWidth are required, all other parameters are optional.
class Convolution2d::Builder {
   public:
    virtual ~Builder() = default;

    Builder() { _activationExplicitlyRemoved = false; }

    virtual Convolution2d build() {
        THOR_THROW_IF_FALSE(_network.isPresent());
        THOR_THROW_IF_FALSE(!_featureInputs.empty());
        THOR_THROW_IF_FALSE(_numOutputChannels.isPresent());
        THOR_THROW_IF_FALSE(_filterHeight.isPresent());
        THOR_THROW_IF_FALSE(_filterWidth.isPresent());

        if (_verticalStride.isEmpty())
            _verticalStride = 1;
        if (_horizontalStride.isEmpty())
            _horizontalStride = 1;
        if (_verticalPadding.isEmpty())
            _computeVerticalSamePadding = true;
        else if (_computeVerticalSamePadding.isEmpty())
            _computeVerticalSamePadding = false;
        if (_horizontalPadding.isEmpty())
            _computeHorizontalSamePadding = true;
        else if (_computeHorizontalSamePadding.isEmpty())
            _computeHorizontalSamePadding = false;
        if (_hasBias.isEmpty())
            _hasBias = false;
        if (_weightsInitializer == nullptr)
            _weightsInitializer = Glorot::Builder().build();
        if (_biasesInitializer == nullptr)
            _biasesInitializer = Glorot::Builder().build();
        if (!_activation && !_activationExplicitlyRemoved)
            _activation = SoftPlus::Builder().build();
        if (_dropProportion.isEmpty())
            _dropProportion = 0.0f;
        if (_useBatchNormalization.isEmpty()) {
            _useBatchNormalization = false;
        }

        if (_epilogue.isPresent()) {
            Convolution2d::validateEpilogueExpression(_epilogue.get());
        }

        Convolution2d convolution2d(_epilogue);

        convolution2d.featureInputs = _featureInputs;
        convolution2d.numOutputChannels = _numOutputChannels;
        convolution2d.filterHeight = _filterHeight;
        convolution2d.filterWidth = _filterWidth;
        convolution2d.verticalStride = _verticalStride;
        convolution2d.horizontalStride = _horizontalStride;
        if (_computeVerticalSamePadding) {
            THOR_THROW_IF_FALSE(convolution2d.verticalStride == 1);
            convolution2d.verticalPadding = computeSamePadding(
                convolution2d.featureInputs[0].getDimensions()[1], convolution2d.verticalStride, convolution2d.filterHeight);
        } else {
            convolution2d.verticalPadding = _verticalPadding;
        }
        if (_computeHorizontalSamePadding) {
            THOR_THROW_IF_FALSE(convolution2d.horizontalStride == 1);
            convolution2d.horizontalPadding = computeSamePadding(
                convolution2d.featureInputs[0].getDimensions()[2], convolution2d.horizontalStride, convolution2d.filterWidth);
        } else {
            convolution2d.horizontalPadding = _horizontalPadding;
        }

        THOR_THROW_IF_FALSE(convolution2d.verticalPadding < convolution2d.filterHeight);
        THOR_THROW_IF_FALSE(convolution2d.horizontalPadding < convolution2d.filterWidth);

        uint32_t outputHeight = computeOutputDimension(convolution2d.featureInputs[0].getDimensions()[1],
                                                       convolution2d.verticalStride,
                                                       convolution2d.filterHeight,
                                                       convolution2d.verticalPadding);
        uint32_t outputWidth = computeOutputDimension(convolution2d.featureInputs[0].getDimensions()[2],
                                                      convolution2d.horizontalStride,
                                                      convolution2d.filterWidth,
                                                      convolution2d.horizontalPadding);

        convolution2d.hasBias = _hasBias;
        convolution2d.weightsInitializer = _weightsInitializer->clone();
        convolution2d.biasInitializer = _biasesInitializer->clone();
        if (_activation != nullptr)
            convolution2d.activation = _activation;
        convolution2d.dropProportion = _dropProportion;
        convolution2d.useBatchNormalization = _useBatchNormalization;
        convolution2d.batchNormExponentialRunningAverageFactor = _batchNormExponentialRunningAverageFactor;
        convolution2d.batchNormEpsilon = _batchNormEpsilon;

        // When this layer gets a specific optimizer, set it now, otherwise network will attach the network default optimizer to it.
        convolution2d.weightsOptimizer = _weightsOptimizer;
        convolution2d.biasesOptimizer = _biasesOptimizer;

        convolution2d.initialized = true;

        if (convolution2d.isMultiLayer()) {
            convolution2d.buildSupportLayersAndAddToNetwork(_network);
        } else {
            for (uint32_t i = 0; i < convolution2d.featureInputs.size(); ++i) {
                convolution2d.featureOutputs.push_back(Tensor(Tensor::DataType::FP16, {_numOutputChannels, outputHeight, outputWidth}));
                convolution2d.outputTensorFromInputTensor[convolution2d.featureInputs[i]] = convolution2d.featureOutputs[i];
                convolution2d.inputTensorFromOutputTensor[convolution2d.featureOutputs[i]] = convolution2d.featureInputs[i];
            }

            convolution2d.standaloneLayerFeatureInputs = convolution2d.getFeatureInputs();
            convolution2d.standaloneLayerFeatureOutputs = convolution2d.getFeatureOutputs();

            convolution2d.addToNetwork(_network.get());
        }

        return convolution2d;
    }

    virtual Convolution2d::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.isPresent());
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
        THOR_THROW_IF_FALSE(!this->_numOutputChannels.isPresent());
        this->_numOutputChannels = _numOutputChannels;
        return *this;
    }

    virtual Convolution2d::Builder &filterHeight(uint32_t _filterHeight) {
        THOR_THROW_IF_FALSE(!this->_filterHeight.isPresent());
        this->_filterHeight = _filterHeight;
        return *this;
    }

    virtual Convolution2d::Builder &filterWidth(uint32_t _filterWidth) {
        THOR_THROW_IF_FALSE(!this->_filterWidth.isPresent());
        this->_filterWidth = _filterWidth;
        return *this;
    }

    virtual Convolution2d::Builder &verticalStride(uint32_t _verticalStride) {
        THOR_THROW_IF_FALSE(_verticalStride != 0);
        THOR_THROW_IF_FALSE(!this->_verticalStride.isPresent());
        this->_verticalStride = _verticalStride;
        return *this;
    }

    virtual Convolution2d::Builder &horizontalStride(uint32_t _horizontalStride) {
        THOR_THROW_IF_FALSE(_horizontalStride != 0);
        THOR_THROW_IF_FALSE(!this->_horizontalStride.isPresent());
        this->_horizontalStride = _horizontalStride;
        return *this;
    }

    virtual Convolution2d::Builder &verticalPadding(uint32_t _verticalPadding) {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = _verticalPadding;
        return *this;
    }

    virtual Convolution2d::Builder &horizontalPadding(uint32_t _horizontalPadding) {
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = _horizontalPadding;
        return *this;
    }

    virtual Convolution2d::Builder &samePadding() {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_computeVerticalSamePadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_computeHorizontalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
        this->_computeHorizontalSamePadding = true;
        this->_computeVerticalSamePadding = true;
        return *this;
    }

    virtual Convolution2d::Builder &verticalSamePadding() {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_computeVerticalSamePadding = true;
        return *this;
    }

    virtual Convolution2d::Builder &horizontalSamePadding() {
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = 0;
        this->_computeHorizontalSamePadding = true;
        return *this;
    }

    virtual Convolution2d::Builder &noPadding() {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_computeVerticalSamePadding.isPresent());
        THOR_THROW_IF_FALSE(!this->_computeHorizontalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
        return *this;
    }

    virtual Convolution2d::Builder &hasBias(bool _hasBias) {
        THOR_THROW_IF_FALSE(!this->_hasBias.isPresent());
        this->_hasBias = _hasBias;
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
        THOR_THROW_IF_FALSE(this->_epilogue.isEmpty());
        Convolution2d::validateEpilogueExpression(expression);
        _epilogue = expression;
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
    virtual Convolution2d::Builder &batchNormalization(Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                                                       Optional<double> epsilon = Optional<double>::empty()) {
        THOR_THROW_IF_FALSE(!_useBatchNormalization.isPresent());
        this->_useBatchNormalization = true;
        this->_batchNormExponentialRunningAverageFactor = exponentialRunningAverageFactor;
        this->_batchNormEpsilon = epsilon;
        return *this;
    }

    // Adds a DropOut layer before this Convolution2d layer, but after the BatchNormalization layer when that is also present.
    virtual Convolution2d::Builder &dropOut(float _dropProportion) {
        THOR_THROW_IF_FALSE(!this->_dropProportion.isPresent());
        this->_dropProportion = _dropProportion;
        return *this;
    }

    static uint32_t computeOutputDimension(uint32_t inputSize, uint32_t stride, uint32_t filterSize, uint32_t padding) {
        THOR_THROW_IF_FALSE(filterSize <= inputSize + 2 * padding);
        THOR_THROW_IF_FALSE(stride > 0);
        return 1 + (((inputSize + 2 * padding) - filterSize) / stride);
    }

    // outputSize = 1 + (((inputSize+ 2*padding) - filterSize) / filterStride);
    // padding = ((outputSize - 1) * filterStride + filterSize - inputSize) / 2
    // where outputSize == inputSize, so
    // padding = ((inputSize - 1) * filterStride + filterSize - inputSize) / 2
    // = ((filterStride-1)*inputSize - filterStride + filterSize) / 2
    static uint32_t computeSamePadding(uint32_t inputSize, uint32_t stride, uint32_t filterSize) {
        if (((stride - 1) * inputSize - stride + filterSize) % 2 == 1)
            throw std::invalid_argument(
                "Can't compute SAME padding: required total padding is odd, but this implementation requires equal padding on both sides.");

        return ((stride - 1) * inputSize - stride + filterSize) / 2;
    }

   private:
    Optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    Optional<uint32_t> _numOutputChannels;
    Optional<uint32_t> _filterHeight;
    Optional<uint32_t> _filterWidth;
    Optional<uint32_t> _verticalStride;
    Optional<uint32_t> _horizontalStride;
    Optional<bool> _computeVerticalSamePadding;
    Optional<bool> _computeHorizontalSamePadding;
    Optional<uint32_t> _verticalPadding;
    Optional<uint32_t> _horizontalPadding;
    Optional<bool> _hasBias;
    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasesInitializer;
    std::shared_ptr<Activation> _activation;
    bool _activationExplicitlyRemoved;
    std::shared_ptr<Optimizer> _weightsOptimizer;
    std::shared_ptr<Optimizer> _biasesOptimizer;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;
    Optional<ThorImplementation::Expression> _epilogue;
};

}  // namespace Thor
