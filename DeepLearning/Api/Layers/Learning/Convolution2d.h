#pragma once

#include "DeepLearning/Api/Layers/LayerBase.h"

namespace Thor {

class Convolution2d : public LayerBase {
   public:
    class Builder;

    Convolution2d() { initialized = false; }

    ~Convolution2d();

    virtual Tensor getFeatureOutput();  // this is not implementation Tensor

   private:
    bool initialized;

    Tensor featureInput;
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
    Activation activation;

    float dropProportion;

    bool useBatchNormalization;
    Optional<double> batchNormExponentialRunningAverageFactor;
    Optional<double> batchNormEpsilon;
};

// featureInput, numOutputChannels, filterHeight and filterWidth are required, all other parameters are optional.
class Convolution2d::Builder {
   public:
    Builder();

    virtual Layer build() {
        assert(_featureInput.isPresent());
        assert(_numOutputChannels.isPresent());
        assert(_filterHeight.isPresent());
        assert(_filterWidth.isPresent());

        if (_verticalStride.isEmpty())
            _verticalStride = 1;
        if (_horizontalStride.isEmpty())
            _horizontalStride = 1;
        if (_verticalPadding.isEmpty())
            _computeVerticalSamePadding = true;
        if (_horizontalPadding.isEmpty())
            _computeHorizontalSamePadding = true;
        if (_hasBias.isEmpty())
            _hasBias = false;
        if (_weightsInitializer.isEmpty())
            _weightsInitializer = Initializer(new XavierInitializer());
        if (_biasInitializer.isEmpty())
            _biasInitializer = Initializer(new UniformRandomInitializer());
        if (_activation.isEmpty())
            _activation = Activation(new Relu());
        if (_dropProportion.isEmpty())
            _dropProportion = 0.0f;
        if (_useBatchNormalization.isEmpty()) {
            _useBatchNormalization = false;
        }

        Convolution2d *convolution2d = new Convolution2d();

        convolution2d->featureInput = _featureInput;  // featureInput should be immutable
        convolution2d->numOutputChannels = _numOutputChannels;
        convolution2d->filterHeight = _filterHeight;
        convolution2d->filterWidth = _filterWidth;
        convolution2d->verticalStride = _verticalStride;
        convolution2d->horizontalStride = _horizontalStride;
        //        if(_computeVerticalSamePadding)
        //            convolution2d->verticalPadding = computeSamePadding(featureInput.getDimensions[2], verticalStride, uint32_t
        //            filterHeight);
        //        else
        convolution2d->verticalPadding = _verticalPadding;
        //        if(_computeHorizontalSamePadding)
        //            convolution2d->horizontalPadding = computeSamePadding(featureInput.getDimensions[3], horizontalStride, uint32_t
        //            filterWidth);
        //        else
        convolution2d->horizontalPadding = _horizontalPadding;
        convolution2d->hasBias = _hasBias;
        convolution2d->weightsInitializer = _weightsInitializer;
        convolution2d->biasInitializer = _biasInitializer;
        convolution2d->activation = _activation;
        convolution2d->dropProportion = _dropProportion;
        convolution2d->useBatchNormalization = _useBatchNormalization;
        convolution2d->batchNormExponentialRunningAverageFactor = _batchNormExponentialRunningAverageFactor;
        convolution2d->batchNormEpsilon = _batchNormEpsilon;
        convolution2d->initialized = true;

        return Layer(convolution2d);
    }

    Convolution2d::Builder featureInput(Tensor _featureInput) {
        assert(_featureInput.isInitialized());
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    Convolution2d::Builder numOutputChannels(uint32_t _numOutputChannels) {
        assert(!this->_numOutputChannels.isPresent());
        this->_numOutputChannels = _numOutputChannels;
        return *this;
    }

    Convolution2d::Builder filterHeight(uint32_t _filterHeight) {
        assert(!this->_filterHeight.isPresent());
        this->_filterHeight = _filterHeight;
        return *this;
    }

    Convolution2d::Builder filterWidth(uint32_t _filterWidth) {
        assert(!this->_filterWidth.isPresent());
        this->_filterWidth = _filterWidth;
        return *this;
    }

    Convolution2d::Builder verticalStride(uint32_t _verticalStride) {
        assert(!this->_verticalStride.isPresent());
        this->_verticalStride = _verticalStride;
        return *this;
    }

    Convolution2d::Builder horizontalStride(uint32_t _horizontalStride) {
        assert(!this->_horizontalStride.isPresent());
        this->_horizontalStride = _horizontalStride;
        return *this;
    }

    Convolution2d::Builder verticalPadding(uint32_t _verticalPadding) {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_verticalPadding = _verticalPadding;
        return *this;
    }

    Convolution2d::Builder horizontalPadding(uint32_t _horizontalPadding) {
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = _horizontalPadding;
        return *this;
    }

    Convolution2d::Builder samePadding() {
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

    Convolution2d::Builder noPadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
        return *this;
    }

    Convolution2d::Builder hasBias(bool _hasBias) {
        assert(!this->_hasBias.isPresent());
        this->_hasBias = _hasBias;
        return *this;
    }

    Convolution2d::Builder weightsInitializer(Initializer _weightsInitializer) {
        assert(!this->_weightsInitializer.isPresent());
        this->_weightsInitializer = _weightsInitializer;
        return *this;
    }

    Convolution2d::Builder biasInitializer(Initializer _biasInitializer) {
        assert(!this->_biasInitializer.isPresent());
        this->_biasInitializer = _biasInitializer;
        return *this;
    }

    // Adds an activation layer after this Convolution2d layer
    Convolution2d::Builder activation(Activation _activation) {
        assert(!this->_activation.isPresent());
        this->_activation = _activation;
        return *this;
    }

    // Adds a DropOut layer before this Convolution2d layer, but after the BatchNormalization layer when that is also present.
    Convolution2d::Builder dropOut(float _dropProportion) {
        assert(!this->_dropProportion.isPresent());
        this->_dropProportion = _dropProportion;
        return *this;
    }

    // Adds a BatchNormalization layer before this Convolution2d layer and before the DropOut layer when that is also present
    // exponentialRunningAverageFactor and epsilon will be set to good default values when not specified.
    Convolution2d::Builder batchNormalization(Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                                              Optional<double> epsilon = Optional<double>::empty()) {
        assert(!_useBatchNormalization.isPresent());
        this->_useBatchNormalization = true;
        this->_batchNormExponentialRunningAverageFactor = exponentialRunningAverageFactor;
        this->_batchNormEpsilon = epsilon;
        return *this;
    }

   private:
    Optional<Tensor> _featureInput;
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
    Optional<Activation> _activation;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;

    // outputSize = 1 + (((inputSize+ 2*padding) - filterSize) / filterStride);
    // padding = ((outputSize - 1) * filterStride + filterSize - inputSize) / 2
    // where outputSize == inputSize, so
    // padding = ((inputSize - 1) * filterStride + filterSize - inputSize) / 2
    // = ((filterStride-1)*inputSize - filterStride + filterSize) / 2
    uint32_t computeSamePadding(uint32_t inputSize, uint32_t stride, uint32_t filterSize) {
        // And round up.
        return (1 + (stride - 1) * inputSize - stride + filterSize) / 2;
    }
};

}  // namespace Thor
