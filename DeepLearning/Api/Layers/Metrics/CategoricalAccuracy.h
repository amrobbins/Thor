#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Metrics/Metric.h"
#include "DeepLearning/Implementation/Layers/Metrics/CategoricalAccuracy.h"

namespace Thor {

class CategoricalAccuracy : public Metric {
   public:
    class Builder;
    CategoricalAccuracy() {}

    ~CategoricalAccuracy() override {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<CategoricalAccuracy>(*this); }

    std::string getLayerType() const override { return "CategoricalAccuracy"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

    enum class LabelType { INDEX = 5, ONE_HOT };

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput() || connectingApiTensor == labelsTensor);

        std::shared_ptr<ThorImplementation::CategoricalAccuracy> categoricalAccuracy =
            std::make_shared<ThorImplementation::CategoricalAccuracy>();
        return categoricalAccuracy;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t workspaceSize = 4 * batchSize;
        uint64_t metricOutputSize = 4;

        return workspaceSize + metricOutputSize;
    }

    LabelType labelType;
    uint32_t numClasses;
};

class CategoricalAccuracy::Builder {
   public:
    virtual CategoricalAccuracy build() {
        THOR_THROW_IF_FALSE(_network.isPresent());
        THOR_THROW_IF_FALSE(_predictions.isPresent());
        THOR_THROW_IF_FALSE(_labels.isPresent());
        THOR_THROW_IF_FALSE(_predictions.get() != _labels.get());
        THOR_THROW_IF_FALSE(_labelType.isPresent());
        THOR_THROW_IF_FALSE(_labelType == LabelType::INDEX || _labelType == LabelType::ONE_HOT);
        CategoricalAccuracy categoricalAccuracy;
        if (_labelType == LabelType::ONE_HOT) {
            std::vector<uint64_t> labelDimensions = _labels.get().getDimensions();
            THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] > 1);
            THOR_THROW_IF_FALSE(_predictions.get().getDimensions() == _labels.get().getDimensions());
            categoricalAccuracy.numClasses = _predictions.get().getDimensions()[0];
        } else {
            std::vector<uint64_t> labelDimensions = _labels.get().getDimensions();
            std::vector<uint64_t> predictionDimensions = _predictions.get().getDimensions();
            THOR_THROW_IF_FALSE(labelDimensions.size() == 1 && labelDimensions[0] == 1);
            Tensor::DataType labelsDataType = _labels.get().getDataType();
            THOR_THROW_IF_FALSE(labelsDataType == Tensor::DataType::UINT8 || labelsDataType == Tensor::DataType::UINT16 ||
                   labelsDataType == Tensor::DataType::UINT32);
            THOR_THROW_IF_FALSE(_numClasses.isPresent());
            THOR_THROW_IF_FALSE(predictionDimensions.size() == 1 && predictionDimensions[0] == _numClasses);
            categoricalAccuracy.numClasses = _numClasses;
        }

        categoricalAccuracy.featureInput = _predictions;
        categoricalAccuracy.labelsTensor = _labels;
        categoricalAccuracy.metricTensor = Tensor(Tensor::DataType::FP32, {1});
        categoricalAccuracy.labelType = _labelType;
        categoricalAccuracy.initialized = true;
        categoricalAccuracy.addToNetwork(_network.get());
        return categoricalAccuracy;
    }

    virtual CategoricalAccuracy::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual CategoricalAccuracy::Builder &predictions(Tensor _predictions) {
        THOR_THROW_IF_FALSE(!this->_predictions.isPresent());
        THOR_THROW_IF_FALSE(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual CategoricalAccuracy::Builder &labels(Tensor _labels) {
        THOR_THROW_IF_FALSE(!this->_labels.isPresent());
        THOR_THROW_IF_FALSE(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    /*
     * A numerical index is passed as the label. The value of the label is the number of the true class.
     * One number is passed per item in the batch.
     * Soft labels are not supported in this case.
     */
    virtual CategoricalAccuracy::Builder &receivesClassIndexLabels(uint32_t numClasses) {
        THOR_THROW_IF_FALSE(!_labelType.isPresent());
        THOR_THROW_IF_FALSE(numClasses > 1);
        _labelType = LabelType::INDEX;
        this->_numClasses = numClasses;
        return *this;
    }

    /**
     * A vector of labels. One label per class per example in the batch.
     * The label can be a one-hot vector, but soft labels are also supported,
     * so for example two classes may both have a label of 0.5.
     */
    virtual CategoricalAccuracy::Builder &receivesOneHotLabels() {
        THOR_THROW_IF_FALSE(!_labelType.isPresent());
        _labelType = LabelType::ONE_HOT;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _predictions;
    Optional<Tensor> _labels;
    Optional<LabelType> _labelType;
    Optional<uint32_t> _numClasses;
};

NLOHMANN_JSON_SERIALIZE_ENUM(CategoricalAccuracy::LabelType,
                             {
                                 {CategoricalAccuracy::LabelType::ONE_HOT, "one_hot"},
                                 {CategoricalAccuracy::LabelType::INDEX, "index"},
                             })

}  // namespace Thor
