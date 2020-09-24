#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

namespace Thor {

class Pooling : public Layer {
   public:
    class Builder;

    Pooling() { initialized = false; }

    ~Pooling() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Pooling>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
    }

   private:
    bool initialized;

    Tensor featureInput;
    Tensor featureOutput;
    uint32_t windowHeight;
    uint32_t windowWidth;
    uint32_t verticalStride;
    uint32_t horizontalStride;
    uint32_t verticalPadding;
    uint32_t horizontalPadding;
};

// featureInput, windowHeight and windowWidth are required, all other parameters are optional.
class Pooling::Builder {
   public:
    Builder();

    virtual Pooling build() {
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

        Pooling pooling;

        pooling.featureInput = _featureInput;
        pooling.windowHeight = _windowHeight;
        pooling.windowWidth = _windowWidth;
        pooling.verticalStride = _verticalStride;
        pooling.horizontalStride = _horizontalStride;
        if (_computeVerticalSamePadding)
            pooling.verticalPadding =
                computeSamePadding(pooling.featureInput.getDimensions()[1], pooling.verticalStride, pooling.windowHeight);
        else
            pooling.verticalPadding = _verticalPadding;
        if (_computeHorizontalSamePadding)
            pooling.horizontalPadding =
                computeSamePadding(pooling.featureInput.getDimensions()[2], pooling.horizontalStride, pooling.windowWidth);
        else
            pooling.horizontalPadding = _horizontalPadding;

        uint32_t outputHeight = computeOutputDimension(
            pooling.featureInput.getDimensions()[1], pooling.verticalStride, pooling.windowHeight, pooling.verticalPadding);
        uint32_t outputWidth = computeOutputDimension(
            pooling.featureInput.getDimensions()[2], pooling.horizontalStride, pooling.windowWidth, pooling.horizontalPadding);
        pooling.featureOutput =
            Tensor(pooling.featureInput.getDataType(), {pooling.featureInput.getDimensions()[0], outputHeight, outputWidth});

        pooling.initialized = true;

        return pooling;
    }

    Pooling::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    Pooling::Builder &windowHeight(uint32_t _windowHeight) {
        assert(!this->_windowHeight.isPresent());
        this->_windowHeight = _windowHeight;
        return *this;
    }

    Pooling::Builder &windowWidth(uint32_t _windowWidth) {
        assert(!this->_windowWidth.isPresent());
        this->_windowWidth = _windowWidth;
        return *this;
    }

    Pooling::Builder &verticalStride(uint32_t _verticalStride) {
        assert(!this->_verticalStride.isPresent());
        this->_verticalStride = _verticalStride;
        return *this;
    }

    Pooling::Builder &horizontalStride(uint32_t _horizontalStride) {
        assert(!this->_horizontalStride.isPresent());
        this->_horizontalStride = _horizontalStride;
        return *this;
    }

    Pooling::Builder &verticalPadding(uint32_t _verticalPadding) {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = _verticalPadding;
        return *this;
    }

    Pooling::Builder &horizontalPadding(uint32_t _horizontalPadding) {
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = _horizontalPadding;
        return *this;
    }

    Pooling::Builder &samePadding() {
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

    Pooling::Builder &verticalSamePadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_computeVerticalSamePadding = true;
        return *this;
    }

    Pooling::Builder &horizontalSamePadding() {
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = 0;
        this->_computeHorizontalSamePadding = true;
        return *this;
    }

    Pooling::Builder &noPadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
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

    uint32_t computeOutputDimension(uint32_t inputSize, uint32_t stride, uint32_t windowSize, uint32_t padding) {
        assert(windowSize <= inputSize + 2 * padding);
        return 1 + (((inputSize + 2 * padding) - windowSize) / stride);
    }

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
