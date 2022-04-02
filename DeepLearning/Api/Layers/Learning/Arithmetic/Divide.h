/**
 * Supports division with back propagation
 *
 * Takes a numerator and a denominator, they may be in the following forms:
 *  Constants:
 *  1. a constant scalar
 *  2. a constant vector whose size matches the last dimension of the its counterpart (numerator/denominator)
 *  3. a constant N-dimensional vector whose dimensions match the last N dimensions of its counterpart
 *
 *  Tensors:
 *  4. a scalar tensor
 *  5. a tensor whose size matches the last dimension of the its counterpart (numerator/denominator)
 *  6. an N-dimensional tensor whose dimensions match the last N dimensions of its counterpart
 *
 * Note: that's not all implemented, for now I only implemented what I need to use.
 */

#pragma once
/*
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Divide.h"

namespace Thor {

class Divide : public Layer {
   public:
    class Builder;

    Divide() { initialized = false; }

    ~Divide() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Divide>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        ThorImplementation::Divide *divide = new ThorImplementation::DivideTensorByConstantScalar();
        Thor::Layer::connectTwoLayers(drivingLayer, divide, drivingApiLayer, this, connectingApiTensor);
        return divide;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        return batchSize * featureOutput.get().getTotalSizeInBytes();
    }

   private:
   Tensor numeratorTensor;
   half denominatorScalar;
};

// featureInput, windowHeight and windowWidth are required, all other parameters are optional.
class Divide::Builder {
   public:
    virtual Divide build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_numeratorTensor.isPresent());
        assert(_divisorScalar.isPresent());

        Divide divide;

        divide.featureInput = _featureInput;
        divide.numeratorTensor = _numeratorTensor;
        divide.denominatorScalar = _denominatorScalar;

        divide.initialized = true;
        divide.addToNetwork(_network.get());
        return divide;
    }

    virtual Divide::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Divide::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Divide::Builder &numerator(Tensor _numeratorTensor) {
        assert(!this->_numeratorTensor.isPresent());
        this->_numeratorTensor = _numeratorTensor;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<Tensor> _numeratorTensor;
    Optional<half> _divisorScalar;
};

}  // namespace Thor
*/