#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"

#include <optional>
#include <stdexcept>

namespace Thor {

class NegativeBinomialNLLLoss : public Loss {
   public:
    class Builder;
    NegativeBinomialNLLLoss() {}

    ~NegativeBinomialNLLLoss() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<NegativeBinomialNLLLoss>(*this); }

    std::string getLayerType() const override { return "NegativeBinomialNLLLoss"; }

    Tensor getMean() const { return predictionsTensor; }
    Tensor getTarget() const { return labelsTensor; }
    Tensor getDispersion() const { return dispersionTensor; }
    bool getLogMean() const { return logMean; }
    bool getLogDispersion() const { return logDispersion; }
    float getEps() const { return eps; }

    std::vector<Tensor> getLossInputTensors() const override {
        std::vector<Tensor> inputs{predictionsTensor, labelsTensor, dispersionTensor};
        if (exampleWeightsTensor.has_value())
            inputs.push_back(exampleWeightsTensor.value());
        return inputs;
    }

    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == dispersionTensor)
            return (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD;
        return Loss::getConnectionType(connectingTensor);
    }

    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (dispersionTensor.isInitialized() && inputTensor == dispersionTensor)
            return "dispersion";
        return Loss::getInputPortName(inputTensor);
    }

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
        throw std::runtime_error("NegativeBinomialNLLLoss is a compound API loss and should not be stamped directly.");
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
        uint64_t dispersionBytes = batchSize * dispersionTensor.getTotalSizeInBytes() * 2;
        return standardLossBytes + dispersionBytes + lossShaperBytes;
    }

    Tensor dispersionTensor;
    bool logMean = true;
    bool logDispersion = true;
    float eps = 1.0e-8f;
};

class NegativeBinomialNLLLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual NegativeBinomialNLLLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_mean.has_value());
        THOR_THROW_IF_FALSE(_dispersion.has_value());
        THOR_THROW_IF_FALSE(_labels.has_value());
        THOR_THROW_IF_FALSE(_mean.value() != _dispersion.value());
        THOR_THROW_IF_FALSE(_mean.value() != _labels.value());
        THOR_THROW_IF_FALSE(_dispersion.value() != _labels.value());
        if (_exampleWeights.has_value()) {
            THOR_THROW_IF_FALSE(_exampleWeights.value() != _mean.value());
            THOR_THROW_IF_FALSE(_exampleWeights.value() != _dispersion.value());
            THOR_THROW_IF_FALSE(_exampleWeights.value() != _labels.value());
        }
        THOR_THROW_IF_FALSE(!_mean.value().getDimensions().empty());
        THOR_THROW_IF_FALSE(_mean.value().getDimensions() == _dispersion.value().getDimensions());
        THOR_THROW_IF_FALSE(_mean.value().getDimensions() == _labels.value().getDimensions());

        if (!_lossShape.has_value())
            _lossShape = LossShape::BATCH;
        if (!_lossDataType.has_value())
            _lossDataType = _mean.value().getDataType();
        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);

        float eps = _eps.value_or(1.0e-8f);
        THOR_THROW_IF_FALSE(eps > 0.0f);

        NegativeBinomialNLLLoss loss;
        loss.predictionsTensor = _mean.value();
        loss.dispersionTensor = _dispersion.value();
        loss.labelsTensor = _labels.value();
        loss.exampleWeightsTensor = _exampleWeights;
        loss.lossDataType = _lossDataType.value();
        loss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        loss.lossShape = _lossShape.value();
        loss.logMean = _logMean.value_or(true);
        loss.logDispersion = _logDispersion.value_or(true);
        loss.eps = eps;
        loss.network = _network.value();
        loss.initialized = true;
        loss.buildSupportLayersAndAddToNetwork();
        return loss;
    }

    virtual NegativeBinomialNLLLoss::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &mean(Tensor _mean) {
        THOR_THROW_IF_FALSE(!this->_mean.has_value());
        THOR_THROW_IF_FALSE(!_mean.getDimensions().empty());
        this->_mean = _mean;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &predictions(Tensor _mean) { return mean(_mean); }

    virtual NegativeBinomialNLLLoss::Builder &dispersion(Tensor _dispersion) {
        THOR_THROW_IF_FALSE(!this->_dispersion.has_value());
        THOR_THROW_IF_FALSE(!_dispersion.getDimensions().empty());
        this->_dispersion = _dispersion;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &target(Tensor _target) { return labels(_target); }

    virtual NegativeBinomialNLLLoss::Builder &logMean(bool _logMean) {
        THOR_THROW_IF_FALSE(!this->_logMean.has_value());
        this->_logMean = _logMean;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &logDispersion(bool _logDispersion) {
        THOR_THROW_IF_FALSE(!this->_logDispersion.has_value());
        this->_logDispersion = _logDispersion;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &exampleWeights(Tensor _exampleWeights) {
        THOR_THROW_IF_FALSE(!this->_exampleWeights.has_value());
        THOR_THROW_IF_FALSE(_exampleWeights.isInitialized());
        this->_exampleWeights = _exampleWeights;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &eps(float _eps) {
        THOR_THROW_IF_FALSE(!this->_eps.has_value());
        THOR_THROW_IF_FALSE(_eps > 0.0f);
        this->_eps = _eps;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &reportsPerExampleLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_EXAMPLE;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::PER_OUTPUT;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &reportsNoLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::NONE;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &lossDataType(DataType _lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(_lossDataType == DataType::FP16 || _lossDataType == DataType::FP32);
        this->_lossDataType = _lossDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _mean;
    std::optional<Tensor> _dispersion;
    std::optional<Tensor> _labels;
    std::optional<Tensor> _exampleWeights;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<bool> _logMean;
    std::optional<bool> _logDispersion;
    std::optional<float> _eps;
};

}  // namespace Thor
