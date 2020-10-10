#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandomInitializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2d.h"
#include "Utilities/TensorOperations/GpuConvolution/ConvolutionKernelRequirement.h"
#include "Utilities/TensorOperations/GpuConvolution/GpuConvolution.h"

namespace Thor {

class Convolution2d : public TrainableWeightsBiasesLayer {
   public:
    class Builder;

    Convolution2d() {}
    virtual ~Convolution2d() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Convolution2d>(*this); }

    virtual uint32_t getFilterHeight() { return filterHeight; }
    virtual uint32_t getFilterWidth() { return filterWidth; }
    virtual uint32_t getVerticalStride() { return verticalStride; }
    virtual uint32_t getHorizontalStride() { return horizontalStride; }
    virtual uint32_t getVerticalPadding() { return verticalPadding; }
    virtual uint32_t getHoriztonalPadding() { return horizontalPadding; }

   protected:
    virtual bool isMultiLayer() const { return useBatchNormalization || dropProportion > 0.0f || activationBuilder; }
    virtual void convertToSingleLayersAndAddToNetwork();

    virtual void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) {
        vector<uint64_t> inputDimensions = inputTensor.getDimensions();
        assert(inputDimensions.size() == 3);

        uint32_t numInputChannels = inputDimensions[0];
        uint32_t numInputRows = inputDimensions[1];
        uint32_t numInputColumns = inputDimensions[2];
        string gpuType = MachineEvaluator::instance().getGpuType(stream.getGpuNum());
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

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

        ThorImplementation::Convolution2d *convolution2d = new ThorImplementation::Convolution2d(
            filterWidth, filterHeight, horizontalStride, verticalStride, horizontalPadding, verticalPadding, numOutputChannels, hasBias);
        Thor::Layer::connectTwoLayers(drivingLayer, convolution2d, drivingApiLayer, this, connectingApiTensor);

        shared_ptr<Initializer::Builder> weightsInitializerBuilderClone = weightsInitializerBuilder->clone();
        weightsInitializerBuilderClone->tensorToInitialize(convolution2d->getWeights());
        weightsInitializerBuilderClone->layerThatOwnsTensor(convolution2d);
        initializers.push_back(weightsInitializerBuilderClone->build());

        if (convolution2d->getBiases().isPresent()) {
            shared_ptr<Initializer::Builder> biasInitializerBuilderClone = biasInitializerBuilder->clone();
            biasInitializerBuilderClone->tensorToInitialize(convolution2d->getBiases().get());
            biasInitializerBuilderClone->layerThatOwnsTensor(convolution2d);
            initializers.push_back(biasInitializerBuilderClone->build());
        }

        return convolution2d;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize) const {
        // FIXME: workspace size?
        uint64_t numInputChannels = featureInputs[0].getDimensions()[0];
        uint64_t numWeights = filterHeight * filterWidth * numInputChannels * numOutputChannels;
        uint64_t numBiases = numOutputChannels;
        // have weights and gradient accumulators, as FP16 elements
        uint64_t fixedMem = 2 * (numWeights + numBiases) * 2;
        uint64_t batchSizeDependentMem =
            2 * featureInputs.size() * (featureInputs[0].getTotalSizeInBytes() + featureOutputs[0].getTotalSizeInBytes()) * batchSize;

        return fixedMem + batchSizeDependentMem;
    }

    virtual uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize) const {
        uint64_t batchSizeDependentMem =
            2 * featureInputs.size() * (featureInputs[0].getTotalSizeInBytes() + featureOutputs[0].getTotalSizeInBytes()) * batchSize;
        return batchSizeDependentMem;
    }

   private:
    Network *network;
    uint32_t numOutputChannels;
    uint32_t filterHeight;
    uint32_t filterWidth;
    uint32_t verticalStride;
    uint32_t horizontalStride;
    uint32_t verticalPadding;
    uint32_t horizontalPadding;
    bool hasBias;
    shared_ptr<Initializer::Builder> weightsInitializerBuilder;
    shared_ptr<Initializer::Builder> biasInitializerBuilder;
    shared_ptr<Activation::Builder> activationBuilder;

    float dropProportion;

    bool useBatchNormalization;
    Optional<double> batchNormExponentialRunningAverageFactor;
    Optional<double> batchNormEpsilon;
};

// featureInput, numOutputChannels, filterHeight and filterWidth are required, all other parameters are optional.
class Convolution2d::Builder {
   public:
    Builder() { _activationExplicitlyRemoved = false; }

    virtual Convolution2d build() {
        assert(_network.isPresent());
        assert(!_featureInputs.empty());
        assert(_numOutputChannels.isPresent());
        assert(_filterHeight.isPresent());
        assert(_filterWidth.isPresent());

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
        if (_weightsInitializerBuilder == nullptr)
            _weightsInitializerBuilder =
                make_shared<UniformRandomInitializer::Builder>(UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1));
        if (_biasInitializerBuilder == nullptr)
            _biasInitializerBuilder =
                make_shared<UniformRandomInitializer::Builder>(UniformRandomInitializer::Builder().minValue(-0.1).maxValue(0.1));
        if (!_activationBuilder && !_activationExplicitlyRemoved)
            _activationBuilder = make_shared<Relu::Builder>(Relu::Builder());
        if (_dropProportion.isEmpty())
            _dropProportion = 0.0f;
        if (_useBatchNormalization.isEmpty()) {
            _useBatchNormalization = false;
        }

        Convolution2d convolution2d;

        convolution2d.network = _network;
        convolution2d.featureInputs = _featureInputs;
        convolution2d.numOutputChannels = _numOutputChannels;
        convolution2d.filterHeight = _filterHeight;
        convolution2d.filterWidth = _filterWidth;
        convolution2d.verticalStride = _verticalStride;
        convolution2d.horizontalStride = _horizontalStride;
        if (_computeVerticalSamePadding) {
            assert(convolution2d.verticalStride == 1);
            convolution2d.verticalPadding = computeSamePadding(
                convolution2d.featureInputs[0].getDimensions()[1], convolution2d.verticalStride, convolution2d.filterHeight);
        } else {
            convolution2d.verticalPadding = _verticalPadding;
        }
        if (_computeHorizontalSamePadding) {
            assert(convolution2d.horizontalStride == 1);
            convolution2d.horizontalPadding = computeSamePadding(
                convolution2d.featureInputs[0].getDimensions()[2], convolution2d.horizontalStride, convolution2d.filterWidth);
        } else {
            convolution2d.horizontalPadding = _horizontalPadding;
        }

        assert(convolution2d.verticalPadding < convolution2d.filterHeight);
        assert(convolution2d.horizontalPadding < convolution2d.filterWidth);

        uint32_t outputHeight = computeOutputDimension(convolution2d.featureInputs[0].getDimensions()[1],
                                                       convolution2d.verticalStride,
                                                       convolution2d.filterHeight,
                                                       convolution2d.verticalPadding);
        uint32_t outputWidth = computeOutputDimension(convolution2d.featureInputs[0].getDimensions()[2],
                                                      convolution2d.horizontalStride,
                                                      convolution2d.filterWidth,
                                                      convolution2d.horizontalPadding);

        for (uint32_t i = 0; i < convolution2d.featureInputs.size(); ++i) {
            convolution2d.featureOutputs.push_back(
                Tensor(convolution2d.featureInputs[0].getDataType(), {_numOutputChannels, outputHeight, outputWidth}));
            convolution2d.outputTensorFromInputTensor[convolution2d.featureInputs[i]] = convolution2d.featureOutputs[i];
            convolution2d.inputTensorFromOutputTensor[convolution2d.featureOutputs[i]] = convolution2d.featureInputs[i];
        }

        convolution2d.hasBias = _hasBias;
        convolution2d.weightsInitializerBuilder = _weightsInitializerBuilder->clone();
        convolution2d.biasInitializerBuilder = _biasInitializerBuilder->clone();
        if (_activationBuilder != nullptr)
            convolution2d.activationBuilder = _activationBuilder->clone();
        convolution2d.dropProportion = _dropProportion;
        convolution2d.useBatchNormalization = _useBatchNormalization;
        convolution2d.batchNormExponentialRunningAverageFactor = _batchNormExponentialRunningAverageFactor;
        convolution2d.batchNormEpsilon = _batchNormEpsilon;
        convolution2d.initialized = true;
        convolution2d.addToNetwork(_network.get());

        return convolution2d;
    }

    virtual Convolution2d::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Convolution2d::Builder &featureInput(Tensor _featureInput) {
        assert(_featureInput.getDimensions().size() == 3);
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            assert(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            assert(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual Convolution2d::Builder &numOutputChannels(uint32_t _numOutputChannels) {
        assert(!this->_numOutputChannels.isPresent());
        this->_numOutputChannels = _numOutputChannels;
        return *this;
    }

    virtual Convolution2d::Builder &filterHeight(uint32_t _filterHeight) {
        assert(!this->_filterHeight.isPresent());
        this->_filterHeight = _filterHeight;
        return *this;
    }

    virtual Convolution2d::Builder &filterWidth(uint32_t _filterWidth) {
        assert(!this->_filterWidth.isPresent());
        this->_filterWidth = _filterWidth;
        return *this;
    }

    virtual Convolution2d::Builder &verticalStride(uint32_t _verticalStride) {
        assert(_verticalStride != 0);
        assert(!this->_verticalStride.isPresent());
        this->_verticalStride = _verticalStride;
        return *this;
    }

    virtual Convolution2d::Builder &horizontalStride(uint32_t _horizontalStride) {
        assert(_horizontalStride != 0);
        assert(!this->_horizontalStride.isPresent());
        this->_horizontalStride = _horizontalStride;
        return *this;
    }

    virtual Convolution2d::Builder &verticalPadding(uint32_t _verticalPadding) {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = _verticalPadding;
        return *this;
    }

    virtual Convolution2d::Builder &horizontalPadding(uint32_t _horizontalPadding) {
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = _horizontalPadding;
        return *this;
    }

    virtual Convolution2d::Builder &samePadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
        this->_computeHorizontalSamePadding = true;
        this->_computeVerticalSamePadding = true;
        return *this;
    }

    virtual Convolution2d::Builder &verticalSamePadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_computeVerticalSamePadding = true;
        return *this;
    }

    virtual Convolution2d::Builder &horizontalSamePadding() {
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = 0;
        this->_computeHorizontalSamePadding = true;
        return *this;
    }

    virtual Convolution2d::Builder &noPadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
        return *this;
    }

    virtual Convolution2d::Builder &hasBias(bool _hasBias) {
        assert(!this->_hasBias.isPresent());
        this->_hasBias = _hasBias;
        return *this;
    }

    virtual Convolution2d::Builder &weightsInitializerBuilder(Initializer::Builder &_weightsInitializerBuilder) {
        assert(this->_weightsInitializerBuilder == nullptr);
        this->_weightsInitializerBuilder = _weightsInitializerBuilder.clone();
        return *this;
    }

    virtual Convolution2d::Builder &weightsInitializerBuilder(Initializer::Builder &&_weightsInitializerBuilder) {
        assert(this->_weightsInitializerBuilder == nullptr);
        this->_weightsInitializerBuilder = _weightsInitializerBuilder.clone();
        return *this;
    }

    virtual Convolution2d::Builder &biasInitializerBuilder(Initializer::Builder &_biasInitializerBuilder) {
        assert(this->_biasInitializerBuilder == nullptr);
        this->_biasInitializerBuilder = _biasInitializerBuilder.clone();
        return *this;
    }

    virtual Convolution2d::Builder &biasInitializerBuilder(Initializer::Builder &&_biasInitializerBuilder) {
        assert(this->_biasInitializerBuilder == nullptr);
        this->_biasInitializerBuilder = _biasInitializerBuilder.clone();
        return *this;
    }

    // Adds an activation layer after this Convolution2d layer
    virtual Convolution2d::Builder &activationBuilder(Activation::Builder &_activationBuilder) {
        assert(this->_activationBuilder == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activationBuilder = _activationBuilder.clone();
        return *this;
    }

    virtual Convolution2d::Builder &activationBuilder(Activation::Builder &&_activationBuilder) {
        assert(this->_activationBuilder == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activationBuilder = _activationBuilder.clone();
        return *this;
    }

    virtual Convolution2d::Builder &noActivation() {
        assert(!this->_activationBuilder);

        _activationExplicitlyRemoved = true;
        return *this;
    }

    // Adds a BatchNormalization layer before this Convolution2d layer and before the DropOut layer when that is also present
    // exponentialRunningAverageFactor and epsilon will be set to good default values when not specified.
    virtual Convolution2d::Builder &batchNormalization(Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                                                       Optional<double> epsilon = Optional<double>::empty()) {
        assert(!_useBatchNormalization.isPresent());
        this->_useBatchNormalization = true;
        this->_batchNormExponentialRunningAverageFactor = exponentialRunningAverageFactor;
        this->_batchNormEpsilon = epsilon;
        return *this;
    }

    // Adds a DropOut layer before this Convolution2d layer, but after the BatchNormalization layer when that is also present.
    virtual Convolution2d::Builder &dropOut(float _dropProportion) {
        assert(!this->_dropProportion.isPresent());
        this->_dropProportion = _dropProportion;
        return *this;
    }

    static uint32_t computeOutputDimension(uint32_t inputSize, uint32_t stride, uint32_t filterSize, uint32_t padding) {
        assert(filterSize <= inputSize + 2 * padding);
        assert(stride > 0);
        return 1 + (((inputSize + 2 * padding) - filterSize) / stride);
    }

    // outputSize = 1 + (((inputSize+ 2*padding) - filterSize) / filterStride);
    // padding = ((outputSize - 1) * filterStride + filterSize - inputSize) / 2
    // where outputSize == inputSize, so
    // padding = ((inputSize - 1) * filterStride + filterSize - inputSize) / 2
    // = ((filterStride-1)*inputSize - filterStride + filterSize) / 2
    static uint32_t computeSamePadding(uint32_t inputSize, uint32_t stride, uint32_t filterSize) {
        // And round up.
        return (1 + (stride - 1) * inputSize - stride + filterSize) / 2;
    }

   private:
    Optional<Network *> _network;
    vector<Tensor> _featureInputs;
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
    shared_ptr<Initializer::Builder> _weightsInitializerBuilder;
    shared_ptr<Initializer::Builder> _biasInitializerBuilder;
    shared_ptr<Activation::Builder> _activationBuilder;
    bool _activationExplicitlyRemoved;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;
};

}  // namespace Thor
