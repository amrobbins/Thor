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

    virtual shared_ptr<Initializer> clone() const = 0;

    virtual void addToNetwork(Network *network);

    // Referring to the initializer object, not the tensor that gets initialized:
    bool isInitialized() { return initialized; }

   protected:
    shared_ptr<ThorImplementation::Initializer> implementationInitializer;
    Optional<Tensor> apiTensorToInitialize;
    Optional<ThorImplementation::Tensor> tensorToInitialize;

    virtual bool usingApiTensor() { return !tensorToInitialize.isPresent(); }

    virtual Tensor getApiTensor() {
        assert(apiTensorToInitialize.isPresent());
        return apiTensorToInitialize;
    }

    virtual void initialize(ThorImplementation::Layer *layer, Optional<ThorImplementation::Tensor> implementationTensorToInitialize) {
        if (implementationTensorToInitialize.isEmpty())
            assert(tensorToInitialize.isPresent());

        ThorImplementation::Tensor targetTensor =
            implementationTensorToInitialize.isPresent() ? implementationTensorToInitialize.get() : tensorToInitialize.get();
        implementationInitializer->initialize(layer, targetTensor);
    }

    virtual void initializeSynchronous(Stream stream, Optional<ThorImplementation::Tensor> implementationTensorToInitialize) {
        if (implementationTensorToInitialize.isEmpty())
            assert(tensorToInitialize.isPresent());

        ThorImplementation::Tensor targetTensor =
            implementationTensorToInitialize.isPresent() ? implementationTensorToInitialize.get() : tensorToInitialize.get();
        implementationInitializer->initializeSynchronous(stream, targetTensor);
    }

    friend class Network;

   protected:
    // Referring to the initializer object, not the tensor that gets initialized:
    bool initialized;
};

class Initializer::Builder {
   public:
    virtual ~Builder() {}
    virtual void network(Network &_network) = 0;
    virtual void tensorToInitialize(Tensor _tensorToInitialize) = 0;
    virtual void tensorToInitialize(ThorImplementation::Tensor _tensorToInitialize) = 0;
    virtual shared_ptr<Initializer> build() = 0;
    virtual shared_ptr<Builder> clone() = 0;
};

}  // namespace Thor
