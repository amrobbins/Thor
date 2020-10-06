#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

namespace Thor {

class DropOut : public Layer {
   public:
    class Builder;
    DropOut() { DropOut::stream.informIsStatic(); }

    virtual ~DropOut() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<DropOut>(*this); }

    virtual float getDropProportion() { return dropProportion; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());

        ThorImplementation::DropOut *dropOut = new ThorImplementation::DropOut(dropProportion, true);
        Thor::Layer::connectTwoLayers(drivingLayer, dropOut, drivingApiLayer, this, connectingApiTensor);
        return dropOut;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize) const {
        uint64_t randomStateSize = ThorImplementation::DropOut::getRandomStateSizeInBytes(stream);
        return randomStateSize + getReservedStateSizeInBytes(batchSize);
    }

   protected:
    virtual uint64_t getReservedStateSizeInBytes(uint32_t batchSize) const {
        ThorImplementation::TensorDescriptor::DataType dataType = ThorImplementation::TensorDescriptor::DataType::FP16;
        vector<uint64_t> featureInputDimensionsWithBatchSize;
        featureInputDimensionsWithBatchSize.push_back(batchSize);
        for (uint32_t i = 0; i < featureInput.get().getDimensions().size(); ++i)
            featureInputDimensionsWithBatchSize.push_back(featureInput.get().getDimensions()[i]);
        return ThorImplementation::DropOut::getReservedSpaceSizeInBytes(featureInputDimensionsWithBatchSize, dataType);
    }

   private:
    float dropProportion;

    static Stream stream;
};

class DropOut::Builder {
   public:
    virtual DropOut build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_dropProportion.isPresent());

        DropOut dropOut;
        dropOut.featureInput = _featureInput;
        dropOut.featureOutput = _featureInput.get().clone();
        dropOut.dropProportion = _dropProportion;
        dropOut.initialized = true;
        dropOut.addToNetwork(_network.get());
        return dropOut;
    }

    virtual DropOut::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual DropOut::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual DropOut::Builder &dropProportion(float dropProportion) {
        assert(!_dropProportion.isPresent());
        assert(dropProportion > 0.0);
        assert(dropProportion <= 1.0);
        this->_dropProportion = dropProportion;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<float> _dropProportion;
};

}  // namespace Thor
