#pragma once

#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "Utilities/Common/Stream.h"

#include <assert.h>
#include <memory>

namespace Thor {

struct StampedNetwork;

class Initializer {
   public:
    class Builder;

    Initializer() { initialized = false; }

    virtual ~Initializer() {}

    virtual std::shared_ptr<Initializer> clone() const = 0;

    // Referring to the initializer object, not the tensor that gets initialized:
    bool isInitialized() { return initialized; }

    virtual void initialize() {
        assert(tensorToInitialize.isPresent());
        assert(layerThatOwnsTensor != nullptr);
        initDoneEvent = implementationInitializer->initialize(layerThatOwnsTensor, tensorToInitialize);
    }

    Optional<Event> getInitDoneEvent() { return initDoneEvent; }

   protected:
    std::shared_ptr<ThorImplementation::Initializer> implementationInitializer;
    Optional<ThorImplementation::Tensor> tensorToInitialize;
    ThorImplementation::Layer *layerThatOwnsTensor;
    Optional<Event> initDoneEvent;

    // Referring to the initializer object, not the tensor that gets initialized:
    bool initialized;

    friend struct StampedNetwork;
};

class Initializer::Builder {
   public:
    virtual ~Builder() {}
    virtual void tensorToInitialize(ThorImplementation::Tensor _tensorToInitialize) = 0;
    virtual void layerThatOwnsTensor(ThorImplementation::Layer *_layerThatOwnsTensor) = 0;
    virtual std::shared_ptr<Initializer> build() = 0;
    virtual std::shared_ptr<Builder> clone() = 0;
};

}  // namespace Thor
