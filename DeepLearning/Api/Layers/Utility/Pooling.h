#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Pooling.h"
#include <optional>

namespace Thor {

class Pooling : public Layer {
   public:
    class Builder;

    Pooling() { initialized = false; }

    ~Pooling() {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Pooling>(*this); }

    enum class Type { AVERAGE = 3, MAX };

    std::vector<uint64_t> getOutputDimensions() { return featureOutput.value().getDimensions(); }
    Type getPoolingType() { return type; }
    uint32_t getWindowHeight() { return windowHeight; }
    uint32_t getWindowWidth() { return windowWidth; }
    uint32_t getVerticalStride() { return verticalStride; }
    uint32_t getHorizontalStride() { return horizontalStride; }
    uint32_t getVerticalPadding() { return verticalPadding; }
    uint32_t getHorizontalPadding() { return horizontalPadding; }

    std::string getLayerType() const override { return "Pooling"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());

        ThorImplementation::Pooling::Type implementationPoolingType;
        implementationPoolingType =
            type == Type::AVERAGE ? ThorImplementation::Pooling::Type::AVERAGE : ThorImplementation::Pooling::Type::MAX;

        std::shared_ptr<ThorImplementation::Pooling> pooling = std::make_shared<ThorImplementation::Pooling>(
            implementationPoolingType, windowHeight, windowWidth, verticalStride, horizontalStride, verticalPadding, horizontalPadding);
        return pooling;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        return batchSize * featureOutput.value().getTotalSizeInBytes();
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
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());
        THOR_THROW_IF_FALSE(_type.has_value());
        THOR_THROW_IF_FALSE(_featureInput.value().getDimensions().size() == 3);
        THOR_THROW_IF_FALSE(_windowHeight.has_value());
        THOR_THROW_IF_FALSE(_windowWidth.has_value());

        if (!_verticalStride.has_value())
            _verticalStride = 1;
        if (!_horizontalStride.has_value())
            _horizontalStride = 1;
        if (!_computeVerticalSamePadding.has_value())
            _computeVerticalSamePadding = false;
        if (!_verticalPadding.has_value() && !_computeVerticalSamePadding.value())
            _verticalPadding = 0;
        if (!_computeHorizontalSamePadding.has_value())
            _computeHorizontalSamePadding = false;
        if (!_horizontalPadding.has_value() && !_computeHorizontalSamePadding.value())
            _horizontalPadding = 0;

        Pooling pooling;

        pooling.featureInput = _featureInput;
        pooling.type = _type.value();
        pooling.windowHeight = _windowHeight.value();
        pooling.windowWidth = _windowWidth.value();
        pooling.verticalStride = _verticalStride.value();
        pooling.horizontalStride = _horizontalStride.value();
        if (_computeVerticalSamePadding.value()) {
            THOR_THROW_IF_FALSE(pooling.verticalStride == 1);
            pooling.verticalPadding =
                computeSamePadding(pooling.featureInput.value().getDimensions()[1], pooling.verticalStride, pooling.windowHeight);
        } else {
            pooling.verticalPadding = _verticalPadding.value();
        }
        if (_computeHorizontalSamePadding.value()) {
            THOR_THROW_IF_FALSE(pooling.horizontalStride == 1);
            pooling.horizontalPadding =
                computeSamePadding(pooling.featureInput.value().getDimensions()[2], pooling.horizontalStride, pooling.windowWidth);
        } else {
            pooling.horizontalPadding = _horizontalPadding.value();
        }

        THOR_THROW_IF_FALSE(pooling.verticalPadding < pooling.windowHeight);
        THOR_THROW_IF_FALSE(pooling.horizontalPadding < pooling.windowWidth);

        uint32_t outputHeight = computeOutputDimension(
            pooling.featureInput.value().getDimensions()[1], pooling.verticalStride, pooling.windowHeight, pooling.verticalPadding);
        uint32_t outputWidth = computeOutputDimension(
            pooling.featureInput.value().getDimensions()[2], pooling.horizontalStride, pooling.windowWidth, pooling.horizontalPadding);
        pooling.featureOutput =
            Tensor(pooling.featureInput.value().getDataType(), {pooling.featureInput.value().getDimensions()[0], outputHeight, outputWidth});

        pooling.initialized = true;
        pooling.addToNetwork(_network.value());
        return pooling;
    }

    virtual Pooling::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Pooling::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Pooling::Builder &type(Pooling::Type _type) {
        THOR_THROW_IF_FALSE(_type == Pooling::Type::AVERAGE || _type == Pooling::Type::MAX);
        this->_type = _type;
        return *this;
    }

    virtual Pooling::Builder &windowHeight(uint32_t _windowHeight) {
        THOR_THROW_IF_FALSE(!this->_windowHeight.has_value());
        this->_windowHeight = _windowHeight;
        return *this;
    }

    virtual Pooling::Builder &windowWidth(uint32_t _windowWidth) {
        THOR_THROW_IF_FALSE(!this->_windowWidth.has_value());
        this->_windowWidth = _windowWidth;
        return *this;
    }

    virtual Pooling::Builder &verticalStride(uint32_t _verticalStride) {
        THOR_THROW_IF_FALSE(_verticalStride != 0);
        THOR_THROW_IF_FALSE(!this->_verticalStride.has_value());
        this->_verticalStride = _verticalStride;
        return *this;
    }

    virtual Pooling::Builder &horizontalStride(uint32_t _horizontalStride) {
        THOR_THROW_IF_FALSE(_horizontalStride != 0);
        THOR_THROW_IF_FALSE(!this->_horizontalStride.has_value());
        this->_horizontalStride = _horizontalStride;
        return *this;
    }

    virtual Pooling::Builder &verticalPadding(uint32_t _verticalPadding) {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.has_value());
        THOR_THROW_IF_FALSE(!this->_computeVerticalSamePadding.has_value());
        this->_verticalPadding = _verticalPadding;
        return *this;
    }

    virtual Pooling::Builder &horizontalPadding(uint32_t _horizontalPadding) {
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.has_value());
        THOR_THROW_IF_FALSE(!this->_computeHorizontalSamePadding.has_value());
        this->_horizontalPadding = _horizontalPadding;
        return *this;
    }

    virtual Pooling::Builder &samePadding() {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.has_value());
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.has_value());
        THOR_THROW_IF_FALSE(!this->_computeVerticalSamePadding.has_value());
        THOR_THROW_IF_FALSE(!this->_computeHorizontalSamePadding.has_value());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
        this->_computeHorizontalSamePadding = true;
        this->_computeVerticalSamePadding = true;
        return *this;
    }

    virtual Pooling::Builder &verticalSamePadding() {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.has_value());
        THOR_THROW_IF_FALSE(!this->_computeVerticalSamePadding.has_value());
        this->_verticalPadding = 0;
        this->_computeVerticalSamePadding = true;
        return *this;
    }

    virtual Pooling::Builder &horizontalSamePadding() {
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.has_value());
        THOR_THROW_IF_FALSE(!this->_computeHorizontalSamePadding.has_value());
        this->_horizontalPadding = 0;
        this->_computeHorizontalSamePadding = true;
        return *this;
    }

    virtual Pooling::Builder &noPadding() {
        THOR_THROW_IF_FALSE(!this->_verticalPadding.has_value());
        THOR_THROW_IF_FALSE(!this->_horizontalPadding.has_value());
        THOR_THROW_IF_FALSE(!this->_computeVerticalSamePadding.has_value());
        THOR_THROW_IF_FALSE(!this->_computeHorizontalSamePadding.has_value());
        this->_verticalPadding = 0;
        this->_horizontalPadding = 0;
        return *this;
    }

    static uint32_t computeOutputDimension(uint32_t inputSize, uint32_t stride, uint32_t windowSize, uint32_t padding) {
        THOR_THROW_IF_FALSE(windowSize <= inputSize + 2 * padding);
        THOR_THROW_IF_FALSE(stride > 0);
        return 1 + (((inputSize + 2 * padding) - windowSize) / stride);
    }

    // outputSize = 1 + (((inputSize+ 2*padding) - windowSize) / windowStride);
    // padding = ((outputSize - 1) * windowStride + windowSize - inputSize) / 2
    // where outputSize == inputSize, so
    // padding = ((inputSize - 1) * windowStride + windowSize - inputSize) / 2
    //         = ((windowStride-1)*inputSize - windowStride + windowSize) / 2
    static uint32_t computeSamePadding(uint32_t inputSize, uint32_t stride, uint32_t windowSize) {
        if (((stride - 1) * inputSize - stride + windowSize) % 2 == 1)
            throw std::invalid_argument(
                "Can't compute SAME padding: required total padding is odd, but this implementation requires equal padding on both sides.");

        return ((stride - 1) * inputSize - stride + windowSize) / 2;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<Pooling::Type> _type;
    std::optional<uint32_t> _windowHeight;
    std::optional<uint32_t> _windowWidth;
    std::optional<uint32_t> _verticalStride;
    std::optional<uint32_t> _horizontalStride;
    std::optional<bool> _computeVerticalSamePadding;
    std::optional<bool> _computeHorizontalSamePadding;
    std::optional<uint32_t> _verticalPadding;
    std::optional<uint32_t> _horizontalPadding;
};

NLOHMANN_JSON_SERIALIZE_ENUM(Pooling::Type,
                             {
                                 {Pooling::Type::AVERAGE, "average"},
                                 {Pooling::Type::MAX, "max"},
                             })

}  // namespace Thor
