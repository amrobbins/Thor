#pragma once

#include "DeepLearning/Api/Layers/Metrics/Metric.h"

namespace Thor {

class CategoricalAccuracy : public Metric {
   public:
    class Builder;
    CategoricalAccuracy() {}

    virtual ~CategoricalAccuracy() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<CategoricalAccuracy>(*this); }

    virtual std::string getLayerType() const { return "CategoricalAccuracy"; }

   private:
    enum class LabelType { INDEX = 5, ONE_HOT };

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             std::vector<std::shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput() || connectingApiTensor == labelsTensor);

        ThorImplementation::CategoricalAccuracy *categoricalAccuracy = new ThorImplementation::CategoricalAccuracy();
        Thor::Layer::connectTwoLayers(drivingLayer, categoricalAccuracy, drivingApiLayer, this, connectingApiTensor);
        return categoricalAccuracy;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        uint64_t workspaceSize = 4 * batchSize;
        uint64_t metricOutputSize = 4;

        return workspaceSize + metricOutputSize;
    }

    LabelType labelType;
};

class CategoricalAccuracy::Builder {
   public:
    virtual CategoricalAccuracy build() {
        assert(_network.isPresent());
        assert(_predictions.isPresent());
        assert(_labels.isPresent());
        assert(_predictions.get() != _labels.get());
        assert(_labelType.isPresent());
        assert(_labelType == LabelType::INDEX || _labelType == LabelType::ONE_HOT);
        if (_labelType == LabelType::ONE_HOT) {
            assert(_predictions.get().getDimensions() == _labels.get().getDimensions());
        } else {
            std::vector<uint64_t> labelDimensions = _labels.get().getDimensions();
            std::vector<uint64_t> predictionDimensions = _predictions.get().getDimensions();
            assert(labelDimensions.size() == 1 || labelDimensions.size() == 2);
            assert(predictionDimensions[0] == labelDimensions[0]);
            if (labelDimensions.size() == 2)
                assert(_labels.get().getDimensions()[1] == 1);
            else
                _labels.get().reshape({predictionDimensions[0], 1});
            Tensor::DataType labelsDataType = _labels.get().getDataType();
            assert(labelsDataType == Tensor::DataType::UINT8 || labelsDataType == Tensor::DataType::UINT16 ||
                   labelsDataType == Tensor::DataType::UINT32);
        }

        CategoricalAccuracy categoricalAccuracy;
        categoricalAccuracy.featureInput = _predictions;
        categoricalAccuracy.labelsTensor = _labels;
        categoricalAccuracy.metricTensor = Tensor(Tensor::DataType::FP32, {1});
        categoricalAccuracy.labelType = _labelType;
        categoricalAccuracy.initialized = true;
        categoricalAccuracy.addToNetwork(_network.get());
        return categoricalAccuracy;
    }

    virtual CategoricalAccuracy::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual CategoricalAccuracy::Builder &predictions(Tensor _predictions) {
        assert(!this->_predictions.isPresent());
        assert(!_predictions.getDimensions().empty());
        this->_predictions = _predictions;
        return *this;
    }

    virtual CategoricalAccuracy::Builder &labels(Tensor _labels) {
        assert(!this->_labels.isPresent());
        assert(!_labels.getDimensions().empty());
        this->_labels = _labels;
        return *this;
    }

    /*
     * A numerical index is passed as the label. The value of the label is the number of the true class.
     * One number is passed per item in the batch.
     * Soft labels are not supported in this case.
     */
    virtual CategoricalAccuracy::Builder &receivesClassIndexLabels() {
        assert(!_labelType.isPresent());
        _labelType = LabelType::INDEX;
        return *this;
    }

    /**
     * A vector of labels. One label per class per example in the batch.
     * The label can be a one-hot vector, but soft labels are also supported,
     * so for example two classes may both have a label of 0.5.
     */
    virtual CategoricalAccuracy::Builder &receivesOneHotLabels() {
        assert(!_labelType.isPresent());
        _labelType = LabelType::ONE_HOT;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _predictions;
    Optional<Tensor> _labels;
    Optional<LabelType> _labelType;
};

}  // namespace Thor