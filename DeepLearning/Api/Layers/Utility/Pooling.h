#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

namespace Thor {

class Pooling : public Layer {
   public:
    class Builder;

    Pooling() { initialized = false; }

    ~Pooling() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Pooling>(*this); }

    uint32_t getWindowHeight() { return windowHeight; }
    uint32_t getWindowWidth() { return windowWidth; }
    uint32_t getVerticalStride() { return verticalStride; }
    uint32_t getHorizontalStride() { return horizontalStride; }
    uint32_t getVerticalPadding() { return verticalPadding; }
    uint32_t getHorizontalPadding() { return horizontalPadding; }

    enum class Type { AVERAGE = 3, MAX };

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer = nullptr,
                                             Thor::Tensor connectingApiTensor = Thor::Tensor()) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        ThorImplementation::Pooling::Type implementationPoolingType;
        implementationPoolingType =
            type == Type::AVERAGE ? ThorImplementation::Pooling::Type::AVERAGE : ThorImplementation::Pooling::Type::MAX;

        ThorImplementation::Pooling *pooling = new ThorImplementation::Pooling(
            implementationPoolingType, windowHeight, windowWidth, verticalStride, horizontalStride, verticalPadding, horizontalPadding);
        Thor::Layer::connectTwoLayers(drivingLayer, pooling, drivingApiLayer, this, connectingApiTensor);
        return pooling;
    }

   private:
    Type type;
    uint32_t windowHeight;
    uint32_t windowWidth;
    uint32_t verticalStride;
    uint32_t horizontalStride;
    uint32_t verticalPadding;
    uint32_t horizontalPadding;

    // friend class Network;
};

// featureInput, windowHeight and windowWidth are required, all other parameters are optional.
class Pooling::Builder {
   public:
    virtual Pooling build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_type.isPresent());
        assert(_featureInput.get().getDimensions().size() == 3);
        assert(_windowHeight.isPresent());
        assert(_windowWidth.isPresent());

        if (_verticalStride.isEmpty())
            _verticalStride = 1;
        if (_horizontalStride.isEmpty())
            _horizontalStride = 1;
        if (_computeVerticalSamePadding.isEmpty())
            _computeVerticalSamePadding = false;
        if (_verticalPadding.isEmpty() && !_computeVerticalSamePadding.get())
            _verticalPadding = 0;
        if (_computeHorizontalSamePadding.isEmpty())
            _computeHorizontalSamePadding = false;
        if (_horizontalPadding.isEmpty() && !_computeHorizontalSamePadding.get())
            _horizontalPadding = 0;

        Pooling pooling;

        pooling.featureInput = _featureInput;
        pooling.type = _type;
        pooling.windowHeight = _windowHeight;
        pooling.windowWidth = _windowWidth;
        pooling.verticalStride = _verticalStride;
        pooling.horizontalStride = _horizontalStride;
        if (_computeVerticalSamePadding)
            pooling.verticalPadding =
                computeSamePadding(pooling.featureInput.get().getDimensions()[1], pooling.verticalStride, pooling.windowHeight);
        else
            pooling.verticalPadding = _verticalPadding;
        if (_computeHorizontalSamePadding)
            pooling.horizontalPadding =
                computeSamePadding(pooling.featureInput.get().getDimensions()[2], pooling.horizontalStride, pooling.windowWidth);
        else
            pooling.horizontalPadding = _horizontalPadding;

        uint32_t outputHeight = computeOutputDimension(
            pooling.featureInput.get().getDimensions()[1], pooling.verticalStride, pooling.windowHeight, pooling.verticalPadding);
        uint32_t outputWidth = computeOutputDimension(
            pooling.featureInput.get().getDimensions()[2], pooling.horizontalStride, pooling.windowWidth, pooling.horizontalPadding);
        pooling.featureOutput =
            Tensor(pooling.featureInput.get().getDataType(), {pooling.featureInput.get().getDimensions()[0], outputHeight, outputWidth});

        pooling.initialized = true;
        pooling.addToNetwork(_network.get());
        return pooling;
    }

    virtual Pooling::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Pooling::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Pooling::Builder &type(Pooling::Type _type) {
        assert(_type == Pooling::Type::AVERAGE || _type == Pooling::Type::MAX);
        this->_type = _type;
        return *this;
    }

    virtual Pooling::Builder &windowHeight(uint32_t _windowHeight) {
        assert(!this->_windowHeight.isPresent());
        this->_windowHeight = _windowHeight;
        return *this;
    }

    virtual Pooling::Builder &windowWidth(uint32_t _windowWidth) {
        assert(!this->_windowWidth.isPresent());
        this->_windowWidth = _windowWidth;
        return *this;
    }

    virtual Pooling::Builder &verticalStride(uint32_t _verticalStride) {
        assert(!this->_verticalStride.isPresent());
        this->_verticalStride = _verticalStride;
        return *this;
    }

    virtual Pooling::Builder &horizontalStride(uint32_t _horizontalStride) {
        assert(!this->_horizontalStride.isPresent());
        this->_horizontalStride = _horizontalStride;
        return *this;
    }

    virtual Pooling::Builder &verticalPadding(uint32_t _verticalPadding) {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = _verticalPadding;
        return *this;
    }

    virtual Pooling::Builder &horizontalPadding(uint32_t _horizontalPadding) {
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = _horizontalPadding;
        return *this;
    }

    virtual Pooling::Builder &samePadding() {
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

    virtual Pooling::Builder &verticalSamePadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_computeVerticalSamePadding = true;
        return *this;
    }

    virtual Pooling::Builder &horizontalSamePadding() {
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_horizontalPadding = 0;
        this->_computeHorizontalSamePadding = true;
        return *this;
    }

    virtual Pooling::Builder &noPadding() {
        assert(!this->_verticalPadding.isPresent());
        assert(!this->_horizontalPadding.isPresent());
        assert(!this->_computeVerticalSamePadding.isPresent());
        assert(!this->_computeHorizontalSamePadding.isPresent());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
        return *this;
    }

    static uint32_t computeOutputDimension(uint32_t inputSize, uint32_t stride, uint32_t windowSize, uint32_t padding) {
        assert(windowSize <= inputSize + 2 * padding);
        assert(stride > 0);
        return 1 + (((inputSize + 2 * padding) - windowSize) / stride);
    }

    // outputSize = 1 + (((inputSize+ 2*padding) - windowSize) / windowStride);
    // padding = ((outputSize - 1) * windowStride + windowSize - inputSize) / 2
    // where outputSize == inputSize, so
    // padding = ((inputSize - 1) * windowStride + windowSize - inputSize) / 2
    //         = ((windowStride-1)*inputSize - windowStride + windowSize) / 2
    static uint32_t computeSamePadding(uint32_t inputSize, uint32_t stride, uint32_t windowSize) {
        // And round up.
        return (1 + (stride - 1) * inputSize - stride + windowSize) / 2;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<Pooling::Type> _type;
    Optional<uint32_t> _windowHeight;
    Optional<uint32_t> _windowWidth;
    Optional<uint32_t> _verticalStride;
    Optional<uint32_t> _horizontalStride;
    Optional<bool> _computeVerticalSamePadding;
    Optional<bool> _computeHorizontalSamePadding;
    Optional<uint32_t> _verticalPadding;
    Optional<uint32_t> _horizontalPadding;
};

}  // namespace Thor
