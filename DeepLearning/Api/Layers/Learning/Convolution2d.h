#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandomInitializer.h"
#include "DeepLearning/Api/Initializers/XavierInitializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"

namespace Thor {

class Convolution2d : public TrainableWeightsBiasesLayer {
   public:
    class Builder;

    Convolution2d() {}
    virtual ~Convolution2d() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Convolution2d>(*this); }

   protected:
    virtual bool isMultiLayer() const { return useBatchNormalization || dropProportion > 0.0f || activationBuilder.isPresent(); }
    virtual void convertToSingleLayersAndAddToNetwork();

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
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
    Initializer weightsInitializer;
    Initializer biasInitializer;
    Optional<Activation::Builder> activationBuilder;

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
        if (_weightsInitializer.isEmpty())
            _weightsInitializer = XavierInitializer();
        if (_biasInitializer.isEmpty())
            _biasInitializer = UniformRandomInitializer();
        if (_activationBuilder.isEmpty() && !_activationExplicitlyRemoved)
            _activationBuilder = Relu::Builder();
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
        if (_computeVerticalSamePadding)
            convolution2d.verticalPadding = computeSamePadding(
                convolution2d.featureInputs[0].getDimensions()[1], convolution2d.verticalStride, convolution2d.filterHeight);
        else
            convolution2d.verticalPadding = _verticalPadding;
        if (_computeHorizontalSamePadding)
            convolution2d.horizontalPadding = computeSamePadding(
                convolution2d.featureInputs[0].getDimensions()[2], convolution2d.horizontalStride, convolution2d.filterWidth);
        else
            convolution2d.horizontalPadding = _horizontalPadding;

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
        convolution2d.weightsInitializer = _weightsInitializer;
        convolution2d.biasInitializer = _biasInitializer;
        convolution2d.activationBuilder = _activationBuilder;
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
        assert(!this->_verticalStride.isPresent());
        this->_verticalStride = _verticalStride;
        return *this;
    }

    virtual Convolution2d::Builder &horizontalStride(uint32_t _horizontalStride) {
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

    virtual Convolution2d::Builder &weightsInitializer(Initializer _weightsInitializer) {
        assert(!this->_weightsInitializer.isPresent());
        this->_weightsInitializer = _weightsInitializer;
        return *this;
    }

    virtual Convolution2d::Builder &biasInitializer(Initializer _biasInitializer) {
        assert(!this->_biasInitializer.isPresent());
        this->_biasInitializer = _biasInitializer;
        return *this;
    }

    // Adds an activation layer after this Convolution2d layer
    virtual Convolution2d::Builder &activationBuilder(Optional<Activation::Builder> _activationBuilder) {
        assert(!this->_activationBuilder.isPresent());
        assert(!_activationExplicitlyRemoved);

        if (_activationBuilder.isEmpty()) {
            _activationExplicitlyRemoved = true;
        } else {
            this->_activationBuilder = _activationBuilder.get();
        }
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
    Optional<Initializer> _weightsInitializer;
    Optional<Initializer> _biasInitializer;
    Optional<Activation::Builder> _activationBuilder;
    bool _activationExplicitlyRemoved;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;
};

}  // namespace Thor
