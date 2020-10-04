#pragma once

#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "Utilities/Common/Stream.h"

#include <assert.h>
#include <memory>

using std::make_shared;

namespace Thor {

using std::shared_ptr;

class Network;

class Initializer {
   public:
    class Builder;

    Initializer() { initialized = false; }

    virtual ~Initializer() {}

    virtual shared_ptr<Initializer> clone() const { assert(false); }

    virtual void addToNetwork(Network *network);

    bool isInitialized() { return initialized; }

   protected:
    shared_ptr<ThorImplementation::Initializer> implementationInitializer;
    Tensor tensorToInitialize;

    virtual void initialize(ThorImplementation::Layer *layer, ThorImplementation::Tensor implementationTensorToInitialize) {
        implementationInitializer->initialize(layer, implementationTensorToInitialize);
    }

    virtual void initializeSynchronous(Stream stream, ThorImplementation::Tensor implementationTensorToInitialize) {
        implementationInitializer->initializeSynchronous(stream, implementationTensorToInitialize);
    }

    friend class Network;

   protected:
    bool initialized;
};

class Initializer::Builder {
   public:
    virtual ~Builder() {}
    virtual void network(Network *_network) { assert(false); }
    virtual void tensorToInitialize(ThorImplementation::Tensor _tensorToInitialize) { assert(false); }
    virtual shared_ptr<Initializer> build() { assert(false); }
    virtual shared_ptr<Builder> clone() { assert(false); }
};

}  // namespace Thor
