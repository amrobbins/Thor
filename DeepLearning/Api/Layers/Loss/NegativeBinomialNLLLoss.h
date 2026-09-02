#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Network/Network.h"

#include <utility>

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
    [[nodiscard]] bool isRagged() const { return raggedPredictionsTensor.has_value(); }
    [[nodiscard]] RaggedTensor getRaggedPredictions() const {
        if (!isRagged()) throw std::runtime_error("NegativeBinomialNLLLoss mean is dense.");
        return raggedPredictionsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLabels() const {
        if (!isRagged()) throw std::runtime_error("NegativeBinomialNLLLoss target is dense.");
        return raggedLabelsTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedDispersion() const {
        if (!isRagged()) throw std::runtime_error("NegativeBinomialNLLLoss dispersion is dense.");
        return raggedDispersionTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedRawLoss() const {
        if (!raggedRawLossTensor.has_value()) throw std::runtime_error("NegativeBinomialNLLLoss raw loss is dense.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] RaggedTensor getRaggedLoss() const {
        if (!isRagged() || lossShape != LossShape::RAW || !raggedRawLossTensor.has_value())
            throw std::runtime_error("NegativeBinomialNLLLoss does not expose a ragged reported loss for this LossShape.");
        return raggedRawLossTensor.value();
    }
    [[nodiscard]] LossShape getLossShape() const { return lossShape; }

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
    std::optional<RaggedTensor> raggedPredictionsTensor;
    std::optional<RaggedTensor> raggedLabelsTensor;
    std::optional<RaggedTensor> raggedDispersionTensor;
    std::optional<RaggedTensor> raggedRawLossTensor;

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
        const bool hasDenseMean = _mean.has_value();
        const bool hasDenseDispersion = _dispersion.has_value();
        const bool hasDenseLabels = _labels.has_value();
        const bool hasRaggedMean = _raggedMean.has_value();
        const bool hasRaggedDispersion = _raggedDispersion.has_value();
        const bool hasRaggedLabels = _raggedLabels.has_value();
        THOR_THROW_IF_FALSE(hasDenseMean == hasDenseDispersion && hasDenseDispersion == hasDenseLabels);
        THOR_THROW_IF_FALSE(hasRaggedMean == hasRaggedDispersion && hasRaggedDispersion == hasRaggedLabels);
        THOR_THROW_IF_FALSE(hasDenseMean != hasRaggedMean);
        if (!_lossShape.has_value()) _lossShape = LossShape::BATCH;

        float eps = _eps.value_or(1.0e-8f);
        THOR_THROW_IF_FALSE(eps > 0.0f);

        NegativeBinomialNLLLoss loss;
        if (hasDenseMean) {
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
            if (!_lossDataType.has_value()) _lossDataType = _mean.value().getDataType();
            loss.predictionsTensor = _mean.value();
            loss.dispersionTensor = _dispersion.value();
            loss.labelsTensor = _labels.value();
            loss.exampleWeightsTensor = _exampleWeights;
        } else {
            const RaggedTensor& mean = _raggedMean.value();
            const RaggedTensor& dispersion = _raggedDispersion.value();
            const RaggedTensor& labels = _raggedLabels.value();
            THOR_THROW_IF_FALSE(mean.isInitialized() && dispersion.isInitialized() && labels.isInitialized());
            THOR_THROW_IF_FALSE(mean.getValues() != dispersion.getValues());
            THOR_THROW_IF_FALSE(mean.getValues() != labels.getValues());
            THOR_THROW_IF_FALSE(dispersion.getValues() != labels.getValues());
            if (mean.getOffsets() != dispersion.getOffsets() || mean.getOffsets() != labels.getOffsets())
                throw std::invalid_argument("NegativeBinomialNLLLoss ragged mean, dispersion, and labels must use the exact same row partition tensor.");
            if (mean.getBatchSize() != dispersion.getBatchSize() || mean.getBatchSize() != labels.getBatchSize() ||
                mean.getMaxTotalValues() != dispersion.getMaxTotalValues() || mean.getMaxTotalValues() != labels.getMaxTotalValues() ||
                mean.getTrailingDimensions() != dispersion.getTrailingDimensions() || mean.getTrailingDimensions() != labels.getTrailingDimensions())
                throw std::invalid_argument("NegativeBinomialNLLLoss ragged mean, dispersion, and labels must have identical value geometry.");
            if (_exampleWeights.has_value() && _exampleWeights->getDimensions() != std::vector<uint64_t>{1})
                throw std::invalid_argument("NegativeBinomialNLLLoss ragged example_weights must have dimensions [1] for one scalar weight per logical row.");
            if (_lossShape.value() == LossShape::PER_OUTPUT)
                throw std::invalid_argument("NegativeBinomialNLLLoss LossShape::PER_OUTPUT is undefined for ragged sequences.");
            if (!_lossDataType.has_value()) _lossDataType = mean.getValuesDataType();
            loss.predictionsTensor = mean.getValues();
            loss.dispersionTensor = dispersion.getValues();
            loss.labelsTensor = labels.getValues();
            loss.raggedPredictionsTensor = mean;
            loss.raggedDispersionTensor = dispersion;
            loss.raggedLabelsTensor = labels;
            loss.exampleWeightsTensor = _exampleWeights;
        }

        THOR_THROW_IF_FALSE(_lossDataType.value() == DataType::FP16 || _lossDataType.value() == DataType::FP32);
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

    virtual NegativeBinomialNLLLoss::Builder &mean(RaggedTensor mean) {
        THOR_THROW_IF_FALSE(!this->_raggedMean.has_value());
        THOR_THROW_IF_FALSE(mean.isInitialized());
        this->_raggedMean = std::move(mean);
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &predictions(Tensor _mean) { return mean(_mean); }
    virtual NegativeBinomialNLLLoss::Builder &predictions(RaggedTensor mean) { return this->mean(std::move(mean)); }

    virtual NegativeBinomialNLLLoss::Builder &dispersion(Tensor _dispersion) {
        THOR_THROW_IF_FALSE(!this->_dispersion.has_value());
        THOR_THROW_IF_FALSE(!_dispersion.getDimensions().empty());
        this->_dispersion = _dispersion;
        return *this;
    }
    virtual NegativeBinomialNLLLoss::Builder &dispersion(RaggedTensor dispersion) {
        THOR_THROW_IF_FALSE(!this->_raggedDispersion.has_value());
        THOR_THROW_IF_FALSE(dispersion.isInitialized());
        this->_raggedDispersion = std::move(dispersion);
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.has_value());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }
    virtual NegativeBinomialNLLLoss::Builder &labels(RaggedTensor labels) {
        THOR_THROW_IF_FALSE(!this->_raggedLabels.has_value());
        THOR_THROW_IF_FALSE(labels.isInitialized());
        this->_raggedLabels = std::move(labels);
        return *this;
    }

    virtual NegativeBinomialNLLLoss::Builder &target(Tensor _target) { return labels(_target); }
    virtual NegativeBinomialNLLLoss::Builder &target(RaggedTensor target) { return labels(std::move(target)); }

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
    std::optional<RaggedTensor> _raggedMean;
    std::optional<RaggedTensor> _raggedDispersion;
    std::optional<RaggedTensor> _raggedLabels;
    std::optional<Tensor> _exampleWeights;
    std::optional<LossShape> _lossShape;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<bool> _logMean;
    std::optional<bool> _logDispersion;
    std::optional<float> _eps;
};

}  // namespace Thor
