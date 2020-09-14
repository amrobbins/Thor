#pragma once

namespace Thor {

class BatchNormalization : public LayerBase {
   public:
    class Builder;
    BatchNormalization() : initialized(false) {}

    virtual ~BatchNormalization();

   private:
    bool initialized;
    Optional<double> exponentialRunningAverageFactor;
    Optional<double> epsilon;
};

class BatchNormalization::Builder {
   public:
    virtual Layer build() {
        BatchNormalization *batchNormalization = new BatchNormalization();
        batchNormalization->exponentialRunningAverageFactor = _exponentialRunningAverageFactor;
        batchNormalization->epsilon = _epsilon;
        batchNormalization->initialized = true;
        return Layer(batchNormalization);
    }

    BatchNormalization::Builder exponentialRunningAverageFactor(double exponentialRunningAverageFactor) {
        assert(!_exponentialRunningAverageFactor.isPresent());
        assert(exponentialRunningAverageFactor > 0.0);
        this->_exponentialRunningAverageFactor = exponentialRunningAverageFactor;
        return *this;
    }

    BatchNormalization::Builder epsilon(double epsilon) {
        assert(!_epsilon.isPresent());
        assert(epsilon > 0.0);
        this->_epsilon = epsilon;
        return *this;
    }

   private:
    Optional<double> _exponentialRunningAverageFactor;
    Optional<double> _epsilon;
};

}  // namespace Thor
