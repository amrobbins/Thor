#pragma once

#include "DeepLearning/Api/Layers/LayerBase.h"

namespace Thor {

class Pooling : public LayerBase {
   public:
    class Builder;

    Pooling() { initialized = false; }

    ~Pooling();

   private:
    bool initialized;

    Tensor featureInput;
    uint32_t windowHeight;
    uint32_t windowWidth;
    uint32_t verticalStride;
    uint32_t horizontalStride;
    uint32_t verticalPadding;
    uint32_t horizontalPadding;

    float dropProportion;

    bool useBatchNormalization;
    Optional<double> batchNormExponentialRunningAverageFactor;
    Optional<double> batchNormEpsilon;
};

// featureInput, windowHeight and windowWidth are required, all other parameters are optional.
class Pooling::Builder {
   public:
    Builder();

    virtual Layer build() {
        assert(_featureInput.isPresent());
        assert(_windowHeight.isPresent());
        assert(_windowWidth.isPresent());

        if (_verticalStride.isEmpty())
            _verticalStride = 1;
        if (_horizontalStride.isEmpty())
            _horizontalStride = 1;
        if (_verticalPadding.isEmpty())
            _computeVerticalSamePadding = true;
        if (_horizontalPadding.isEmpty())
            _computeHorizontalSamePadding = true;
        if (_dropProportion.isEmpty())
            _dropProportion = 0.0f;
        if (_useBatchNormalization.isEmpty()) {
            _useBatchNormalization = false;
        }

        Pooling *pooling = new Pooling();

        pooling->featureInput = _featureInput;  // featureInput should be immutable
        pooling->windowHeight = _windowHeight;
        pooling->windowWidth = _windowWidth;
        pooling->verticalStride = _verticalStride;
        pooling->horizontalStride = _horizontalStride;
        // FIXME: need API tensor
        //        if(_computeVerticalSamePadding)
        //            pooling->verticalPadding = computeSamePadding(featureInput.getDimensions[2], verticalStride, uint32_t
        //            windowHeight);
        //        else
        pooling->verticalPadding = _verticalPadding;
        //        if(_computeHorizontalSamePadding)
        //            pooling->horizontalPadding = computeSamePadding(featureInput.getDimensions[3], horizontalStride, uint32_t
        //            windowWidth);
        //        else
        pooling->horizontalPadding = _horizontalPadding;
        pooling->dropProportion = _dropProportion;
        pooling->useBatchNormalization = _useBatchNormalization;
        pooling->batchNormExponentialRunningAverageFactor = _batchNormExponentialRunningAverageFactor;
        pooling->batchNormEpsilon = _batchNormEpsilon;
        pooling->initialized = true;

        return Layer(pooling);
    }

    Pooling::Builder featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    Pooling::Builder windowHeight(uint32_t _windowHeight) {
        assert(!this->_windowHeight.isPresent());
        this->_windowHeight = _windowHeight;
        return *this;
    }

    Pooling::Builder windowWidth(uint32_t _windowWidth) {
        assert(!this->_windowWidth.isPresent());
        this->_windowWidth = _windowWidth;
        return *this;
    }

    Pooling::Builder verticalStride(uint32_t _verticalStride) {
        assert(!this->_verticalStride.isPresent());
        this->_verticalStride = _verticalStride;
        return *this;
    }

    Pooling::Builder horizontalStride(uint32_t _horizontalStride) {
        assert(!this->_horizontalStride.isPresent());
        this->_horizontalStride = _horizontalStride;
        return *this;
    }

    Pooling::Builder verticalPadding(uint32_t _verticalPadding) {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = _verticalPadding;
        return *this;
    }

    Pooling::Builder horizontalPadding(uint32_t _horizontalPadding) {
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = _horizontalPadding;
        return *this;
    }

    Pooling::Builder samePadding() {
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

    Pooling::Builder verticalSamePadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_computeVerticalSamePadding = true;
        return *this;
    }

    Pooling::Builder horizontalSamePadding() {
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = 0;
        this->_computeHorizontalSamePadding = true;
        return *this;
    }

    Pooling::Builder noPadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
        return *this;
    }

    // Adds a BatchNormalization layer before this Pooling layer and before the DropOut layer when that is also present
    // exponentialRunningAverageFactor and epsilon will be set to good default values when not specified.
    Pooling::Builder batchNormalization(Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                                        Optional<double> epsilon = Optional<double>::empty()) {
        assert(!_useBatchNormalization.isPresent());
        this->_useBatchNormalization = true;
        this->_batchNormExponentialRunningAverageFactor = exponentialRunningAverageFactor;
        this->_batchNormEpsilon = epsilon;
        return *this;
    }

    // Adds a DropOut layer before this Pooling layer, but after the BatchNormalization layer when that is also present.
    Pooling::Builder dropOut(float _dropProportion) {
        assert(!this->_dropProportion.isPresent());
        this->_dropProportion = _dropProportion;
        return *this;
    }

   private:
    Optional<Tensor> _featureInput;
    Optional<uint32_t> _windowHeight;
    Optional<uint32_t> _windowWidth;
    Optional<uint32_t> _verticalStride;
    Optional<uint32_t> _horizontalStride;
    Optional<bool> _computeVerticalSamePadding;
    Optional<bool> _computeHorizontalSamePadding;
    Optional<uint32_t> _verticalPadding;
    Optional<uint32_t> _horizontalPadding;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;

    // outputSize = 1 + (((inputSize+ 2*padding) - windowSize) / windowStride);
    // padding = ((outputSize - 1) * windowStride + windowSize - inputSize) / 2
    // where outputSize == inputSize, so
    // padding = ((inputSize - 1) * windowStride + windowSize - inputSize) / 2
    //         = ((windowStride-1)*inputSize - windowStride + windowSize) / 2
    uint32_t computeSamePadding(uint32_t inputSize, uint32_t stride, uint32_t windowSize) {
        // And round up.
        return (1 + (stride - 1) * inputSize - stride + windowSize) / 2;
    }
};

}  // namespace Thor
