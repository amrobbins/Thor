#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Initializers/UniformRandomInitializer.h"

#include <assert.h>

namespace Thor {

class UniformRandomInitializer : public Initializer {
   public:
    class Builder;

    virtual ~UniformRandomInitializer() {}

    virtual shared_ptr<Initializer> clone() const { return make_shared<UniformRandomInitializer>(*this); }
};

class UniformRandomInitializer::Builder : public Initializer::Builder {
   public:
    virtual ~Builder() {}

    virtual shared_ptr<Initializer> build() {
        assert(_network.isPresent());
        assert(_tensorToInitialize.isPresent());

        UniformRandomInitializer uniformRandomInitializer;
        uniformRandomInitializer.tensorToInitialize = _tensorToInitialize;
        uniformRandomInitializer.implementationInitializer =
            ThorImplementation::UniformRandomInitializer(_maxValue.get(), _minValue.get()).clone();
        uniformRandomInitializer.initialized = true;
        uniformRandomInitializer.addToNetwork(_network.get());
        return uniformRandomInitializer.clone();
    }

    virtual void network(Network &_network) { this->_network = &_network; }

    virtual void tensorToInitialize(Tensor _tensorToInitialize) {
        assert(!_tensorToInitialize.getDimensions().empty());
        this->_tensorToInitialize = _tensorToInitialize;
    }

    virtual UniformRandomInitializer::Builder &minValue(double _minValue) {
        assert(!this->_minValue.isPresent());
        if (_maxValue.isPresent())
            assert(_minValue <= _maxValue.get());
        this->_minValue = _minValue;
        return *this;
    }

    virtual UniformRandomInitializer::Builder &maxValue(double _maxValue) {
        assert(!this->_maxValue.isPresent());
        if (_minValue.isPresent())
            assert(_maxValue >= _minValue.get());
        this->_maxValue = _maxValue;
        return *this;
    }

    virtual shared_ptr<Initializer::Builder> clone() { return make_shared<UniformRandomInitializer::Builder>(*this); }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _tensorToInitialize;
    Optional<double> _minValue;
    Optional<double> _maxValue;
};

}  // namespace Thor
