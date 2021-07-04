#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Initializers/Glorot.h"

#include <assert.h>

namespace Thor {

class Glorot : public Initializer {
   public:
    class Builder;

    virtual ~Glorot() {}

    virtual shared_ptr<Initializer> clone() const { return make_shared<Glorot>(*this); }
};

class Glorot::Builder : public Initializer::Builder {
   public:
    virtual ~Builder() { _layerThatOwnsTensor = nullptr; }

    virtual shared_ptr<Initializer> build() {
        assert(_tensorToInitialize.isPresent());

        if (_mode.isEmpty())
            _mode = ThorImplementation::Glorot::Mode::UNIFORM;

        Glorot glorotInitializer;
        glorotInitializer.tensorToInitialize = _tensorToInitialize;
        glorotInitializer.layerThatOwnsTensor = _layerThatOwnsTensor;
        glorotInitializer.implementationInitializer = ThorImplementation::Glorot(_mode.get()).clone();
        glorotInitializer.initialized = true;
        return glorotInitializer.clone();
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

    virtual Glorot::Builder &mode(ThorImplementation::Glorot::Mode _mode) {
        assert(!this->_mode.isPresent());
        this->_mode = _mode;
        return *this;
    }

    virtual shared_ptr<Initializer::Builder> clone() { return make_shared<Glorot::Builder>(*this); }

   protected:
    Optional<ThorImplementation::Tensor> _tensorToInitialize;
    ThorImplementation::Layer *_layerThatOwnsTensor;
    Optional<ThorImplementation::Glorot::Mode> _mode;
};

}  // namespace Thor
