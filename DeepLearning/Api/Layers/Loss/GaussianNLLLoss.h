#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class GaussianNLLLoss : public Loss {
   public:
    class Builder;
    GaussianNLLLoss() {}

    ~GaussianNLLLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<GaussianNLLLoss>(*this); }

    std::string getLayerType() const override { return "GaussianNLLLoss"; }

    Tensor getMean() const { return predictionsTensor; }
    Tensor getTarget() const { return labelsTensor; }
    Tensor getVariance() const { return varianceTensor; }
    std::vector<Tensor> getLossInputTensors() const override { return {predictionsTensor, labelsTensor, varianceTensor}; }

    bool getFull() const { return full; }
    float getEps() const { return eps; }

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
        throw std::runtime_error("GaussianNLLLoss is a compound API loss and should not be stamped directly.");
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t lossShaperBytes = 0;
        if (isMultiLayer()) {
            lossShaperBytes = LossShaper::Builder()
                                  .lossInput(lossTensor)
                                  .reportsBatchLoss()
                                  .getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        }

        uint64_t bytes = 4;
        bytes += batchSize * predictionsTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * labelsTensor.getTotalSizeInBytes();
        bytes += batchSize * varianceTensor.getTotalSizeInBytes() * 2;
        bytes += batchSize * lossTensor.getTotalSizeInBytes();
        return bytes + lossShaperBytes;
    }

    Tensor varianceTensor;
    bool full = false;
    float eps = 1.0e-6f;
};

class GaussianNLLLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual GaussianNLLLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_predictions.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_variance.has_value());
        THOR_THROW_IF_FALSE(_predictions.value() != _labels.value());
        THOR_THROW_IF_FALSE(_predictions.value() != _variance.value());
        THOR_THROW_IF_FALSE(_labels.value() != _variance.value());
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions().size() == 1);
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _labels.value().getDimensions());
        THOR_THROW_IF_FALSE(_predictions.value().getDimensions() == _variance.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _predictions.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        float eps = _eps.value_or(1.0e-6f);
        THOR_THROW_IF_FALSE(eps > 0.0f);

        GaussianNLLLoss loss;
        loss.predictionsTensor = _predictions.value();
        loss.labelsTensor = _labels.value();
        loss.varianceTensor = _variance.value();
        loss.lossDataType = _lossDataType.value();
        loss.lossShape = _lossShape.value();
        loss.full = _full.value_or(false);
        loss.eps = eps;
        loss.network = _network.value();
        loss.initialized = true;

        loss.buildSupportLayersAndAddToNetwork();

        return loss;
    }

    virtual GaussianNLLLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.has_value());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &mean(Tensor _mean) { return predictions(_mean); }

    virtual GaussianNLLLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &target(Tensor _target) { return labels(_target); }

    virtual GaussianNLLLoss::Builder &variance(Tensor _variance) {
        THOR_THROW_IF_FALSE(!this->_variance.has_value());
        THOR_THROW_IF_FALSE(!_variance.getDimensions().empty());
        this->_variance = _variance;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &full(bool _full) {
        THOR_THROW_IF_FALSE(!this->_full.has_value());
        this->_full = _full;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &eps(float _eps) {
        THOR_THROW_IF_FALSE(!this->_eps.has_value());
        THOR_THROW_IF_FALSE(_eps > 0.0f);
        this->_eps = _eps;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual GaussianNLLLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _predictions;
    std::optional<Tensor> _labels;
    std::optional<Tensor> _variance;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<bool> _full;
    std::optional<float> _eps;
};

}  // namespace Thor
