#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

// Attach Stub to output tensors that would be dangling and are not wanted as NetworkOutputs.
namespace Thor {

class Stub : public Layer {
   public:
    class Builder;

    Stub() {}

    virtual ~Stub() {}

    virtual vector<Tensor> getOutputsFromInput(Tensor inputTensor) { return vector<Tensor>(); }

    virtual shared_ptr<Layer> clone() const { return make_shared<Stub>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement, uint32_t batchSize) const { assert(false); }

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(false);
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize) const { return 0; }

   private:
    Tensor getFeatureOutput();
};

class Stub::Builder {
   public:
    virtual Stub build() {
        assert(_network.isPresent());
        assert(!_inputTensor.isEmpty());

        Stub stub;
        stub.featureInput = _inputTensor;
        stub.initialized = true;
        stub.addToNetwork(_network.get());
        return stub;
    }

    virtual Stub::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Stub::Builder &inputTensor(Tensor _inputTensor) {
        assert(_inputTensor.isInitialized());
        this->_inputTensor = _inputTensor;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _inputTensor;
};

}  // namespace Thor
