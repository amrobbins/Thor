#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class TverskyLoss : public Loss {
   public:
    class Builder;
    TverskyLoss() {}

    ~TverskyLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<TverskyLoss>(*this); }

    std::string getLayerType() const override { return "TverskyLoss"; }

    float getAlpha() const { return alpha; }
    float getBeta() const { return beta; }
    float getSmooth() const { return smooth; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual bool isMultiLayer() const { return true; }

    virtual void buildSupportLayersAndAddToNetwork();

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        (void)connectingApiTensor;
        (void)inferenceOnly;
        throw std::runtime_error("TverskyLoss is a compound API loss and should not be stamped directly.");
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t lossShaperBytes = 0;
        if (isMultiLayer()) {
            lossShaperBytes = LossShaper::Builder()
                                  .lossInput(lossTensor)
                                  .reportsBatchLoss()
                                  .getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        }

        uint64_t standardLossBytes = Loss::getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        return standardLossBytes + lossShaperBytes;
    }

    float alpha = 0.5f;
    float beta = 0.5f;
    float smooth = 1.0f;
};

class TverskyLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual TverskyLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        THOR_THROW_IF_FALSE(!_predictions.value().getDimensions().empty());
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _predictions.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        float alpha = _alpha.value_or(0.5f);
        float beta = _beta.value_or(0.5f);
        float smooth = _smooth.value_or(1.0f);
        THOR_THROW_IF_FALSE(alpha >= 0.0f);
        THOR_THROW_IF_FALSE(beta >= 0.0f);
        THOR_THROW_IF_FALSE(smooth >= 0.0f);

        TverskyLoss tverskyLoss;
        tverskyLoss.predictionsTensor = _predictions.value();
        tverskyLoss.labelsTensor = _labels.value();
        tverskyLoss.lossDataType = _lossDataType.value();
        tverskyLoss.lossShape = _lossShape.value();
        tverskyLoss.alpha = alpha;
        tverskyLoss.beta = beta;
        tverskyLoss.smooth = smooth;
        tverskyLoss.network = _network.value();
        tverskyLoss.initialized = true;

        tverskyLoss.buildSupportLayersAndAddToNetwork();

        return tverskyLoss;
    }

    virtual TverskyLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual TverskyLoss::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual TverskyLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual TverskyLoss::Builder &alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        THOR_THROW_IF_FALSE(_alpha >= 0.0f);
        this->_alpha = _alpha;
        return *this;
    }

    virtual TverskyLoss::Builder &beta(float _beta) {
        THOR_THROW_IF_FALSE(!this->_beta.has_value());
        THOR_THROW_IF_FALSE(_beta >= 0.0f);
        this->_beta = _beta;
        return *this;
    }

    virtual TverskyLoss::Builder &smooth(float _smooth) {
        THOR_THROW_IF_FALSE(!this->_smooth.has_value());
        THOR_THROW_IF_FALSE(_smooth >= 0.0f);
        this->_smooth = _smooth;
        return *this;
    }

    virtual TverskyLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual TverskyLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual TverskyLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual TverskyLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual TverskyLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _alpha;
    std::optional<float> _beta;
    std::optional<float> _smooth;
};

}  // namespace Thor
