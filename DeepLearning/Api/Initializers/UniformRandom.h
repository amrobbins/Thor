#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Initializers/UniformRandom.h"

#include <assert.h>

namespace Thor {

class UniformRandom : public Initializer {
   public:
    class Builder;

    virtual ~UniformRandom() {}

    virtual shared_ptr<Initializer> clone() const { return make_shared<UniformRandom>(*this); }
};

class UniformRandom::Builder : public Initializer::Builder {
   public:
    virtual ~Builder() { _layerThatOwnsTensor = nullptr; }

    virtual shared_ptr<Initializer> build() {
        assert(_tensorToInitialize.isPresent());

        UniformRandom uniformRandomInitializer;
        uniformRandomInitializer.tensorToInitialize = _tensorToInitialize;
        uniformRandomInitializer.layerThatOwnsTensor = _layerThatOwnsTensor;
        uniformRandomInitializer.implementationInitializer = ThorImplementation::UniformRandom(_maxValue.get(), _minValue.get()).clone();
        uniformRandomInitializer.initialized = true;
        return uniformRandomInitializer.clone();
    }

    virtual void tensorToInitialize(ThorImplementation::Tensor _tensorToInitialize) {
        assert(!_tensorToInitialize.getDescriptor().getDimensions().empty());
        assert(this->_tensorToInitialize.isEmpty());
        this->_tensorToInitialize = _tensorToInitialize;
    }

    virtual void layerThatOwnsTensor(ThorImplementation::Layer *_layerThatOwnsTensor) {
        assert(_layerThatOwnsTensor != nullptr);
        assert(this->_layerThatOwnsTensor == nullptr);
        this->_layerThatOwnsTensor = _layerThatOwnsTensor;
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

    virtual shared_ptr<Initializer::Builder> clone() { return make_shared<UniformRandom::Builder>(*this); }

   protected:
    Optional<ThorImplementation::Tensor> _tensorToInitialize;
    ThorImplementation::Layer *_layerThatOwnsTensor;
    Optional<double> _minValue;
    Optional<double> _maxValue;
};

}  // namespace Thor
