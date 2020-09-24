#pragma once

namespace Thor {

class BatchNormalization : public Layer {
   public:
    class Builder;
    BatchNormalization() : initialized(false) {}

    virtual ~BatchNormalization() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<BatchNormalization>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
    }

   private:
    Tensor featureInput;
    Optional<double> exponentialRunningAverageFactor;
    Optional<double> epsilon;
    bool initialized;
};

class BatchNormalization::Builder {
   public:
    virtual BatchNormalization build() {
        assert(_featureInput.isPresent());

        BatchNormalization batchNormalization;
        batchNormalization.exponentialRunningAverageFactor = _exponentialRunningAverageFactor;
        batchNormalization.epsilon = _epsilon;
        batchNormalization.featureInput = _featureInput;
        batchNormalization.featureOutput = _featureInput.get().clone();
        batchNormalization.initialized = true;
        return batchNormalization;
    }

    virtual BatchNormalization::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
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
    Optional<double> _exponentialRunningAverageFactor;
    Optional<double> _epsilon;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
