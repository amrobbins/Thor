#pragma once

#include "DeepLearning/Api/Network/Network.h"

namespace Thor {

class BatchNormalization : public TrainableWeightsBiasesLayer {
   public:
    class Builder;
    BatchNormalization() {}

    virtual ~BatchNormalization() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<BatchNormalization>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
    }

   private:
    Optional<double> exponentialRunningAverageFactor;
    Optional<double> epsilon;
};

class BatchNormalization::Builder {
   public:
    virtual BatchNormalization build() {
        assert(_network.isPresent());
        assert(!_featureInputs.empty());

        BatchNormalization batchNormalization;
        batchNormalization.featureInputs = _featureInputs;
        for (uint32_t i = 0; i < batchNormalization.featureInputs.size(); ++i)
            batchNormalization.featureOutputs.push_back(batchNormalization.featureInputs[i].clone());
        batchNormalization.exponentialRunningAverageFactor = _exponentialRunningAverageFactor;
        batchNormalization.epsilon = _epsilon;
        batchNormalization.initialized = true;
        batchNormalization.addToNetwork(_network.get());
        return batchNormalization;
    }

    virtual BatchNormalization::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual BatchNormalization::Builder featureInput(Tensor _featureInput) {
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            assert(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            assert(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual BatchNormalization::Builder &exponentialRunningAverageFactor(double exponentialRunningAverageFactor) {
        assert(!_exponentialRunningAverageFactor.isPresent());
        assert(exponentialRunningAverageFactor > 0.0);
        this->_exponentialRunningAverageFactor = exponentialRunningAverageFactor;
        return *this;
    }

    virtual BatchNormalization::Builder &epsilon(double epsilon) {
        assert(!_epsilon.isPresent());
        assert(epsilon > 0.0);
        this->_epsilon = epsilon;
        return *this;
    }

   private:
    Optional<Network *> _network;
    vector<Tensor> _featureInputs;
    Optional<double> _exponentialRunningAverageFactor;
    Optional<double> _epsilon;
};

}  // namespace Thor
