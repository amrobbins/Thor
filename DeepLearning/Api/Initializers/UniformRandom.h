#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Initializers/UniformRandom.h"

#include <assert.h>
#include <chrono>

namespace Thor {

class UniformRandom : public Initializer {
   public:
    class Builder;

    virtual ~UniformRandom() = default;

    virtual void stamp(ThorImplementation::Layer *layerThatOwnsTensor, ThorImplementation::Tensor tensorToInitialize) {
        layerThatOwnsTensor->setInitializer(tensorToInitialize, ThorImplementation::UniformRandom(maxValue, minValue).clone());
    }

    virtual std::shared_ptr<Initializer> clone() const { return std::make_shared<UniformRandom>(*this); }

    virtual nlohmann::json serialize() const;
    static std::shared_ptr<Initializer> deserialize(const nlohmann::json &j);

    double getMinValue() const { return minValue; }
    double getMaxValue() const { return maxValue; }

   protected:
    double minValue;
    double maxValue;
};

class UniformRandom::Builder : public Initializer::Builder {
   public:
    virtual ~Builder() = default;

    virtual std::shared_ptr<Initializer> build() {
        UniformRandom uniformRandomInitializer;
        uniformRandomInitializer.minValue = _minValue;
        uniformRandomInitializer.maxValue = _maxValue;
        uniformRandomInitializer.initialized = true;
        return uniformRandomInitializer.clone();
    }

    virtual UniformRandom::Builder &minValue(double _minValue) {
        assert(!this->_minValue.isPresent());
        if (_maxValue.isPresent())
            assert(_minValue <= _maxValue.get());
        this->_minValue = _minValue;
        return *this;
    }

    virtual UniformRandom::Builder &maxValue(double _maxValue) {
        assert(!this->_maxValue.isPresent());
        if (_minValue.isPresent())
            assert(_maxValue >= _minValue.get());
        this->_maxValue = _maxValue;
        return *this;
    }

    virtual std::shared_ptr<Initializer::Builder> clone() { return std::make_shared<UniformRandom::Builder>(*this); }

   protected:
    Optional<double> _minValue;
    Optional<double> _maxValue;
};

}  // namespace Thor
