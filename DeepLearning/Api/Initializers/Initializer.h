#pragma once

#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "Utilities/Common/Stream.h"

#include <assert.h>
#include <memory>

using std::make_shared;

namespace Thor {

struct StampedNetwork;

using std::shared_ptr;

class Initializer {
   public:
    class Builder;

    Initializer() { initialized = false; }

    virtual ~Initializer() {}

    virtual shared_ptr<Initializer> clone() const = 0;

    // Referring to the initializer object, not the tensor that gets initialized:
    bool isInitialized() { return initialized; }

    virtual void initialize() {
        assert(tensorToInitialize.isPresent());
        assert(layerThatOwnsTensor != nullptr);
        implementationInitializer->initialize(layerThatOwnsTensor, tensorToInitialize);
    }

   protected:
    shared_ptr<ThorImplementation::Initializer> implementationInitializer;
    Optional<ThorImplementation::Tensor> tensorToInitialize;
    ThorImplementation::Layer *layerThatOwnsTensor;

    // If you want to initialize a tensor that is not owned by a layer, use this version:
    virtual void initializeSynchronous(Stream stream) {
        assert(tensorToInitialize.isPresent());
        implementationInitializer->initializeSynchronous(stream, tensorToInitialize);
    }

    // Referring to the initializer object, not the tensor that gets initialized:
    bool initialized;

    friend struct StampedNetwork;
};

class Initializer::Builder {
   public:
    virtual ~Builder() {}
    virtual void tensorToInitialize(ThorImplementation::Tensor _tensorToInitialize) = 0;
    virtual void layerThatOwnsTensor(ThorImplementation::Layer *_layerThatOwnsTensor) = 0;
    virtual shared_ptr<Initializer> build() = 0;
    virtual shared_ptr<Builder> clone() = 0;
};

}  // namespace Thor
