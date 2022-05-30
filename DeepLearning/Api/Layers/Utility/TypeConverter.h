#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Utility/TypeConversion.h"

namespace Thor {

class TypeConverter : public Layer {
   public:
    class Builder;
    TypeConverter() {}

    virtual ~TypeConverter() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<TypeConverter>(*this); }

    virtual string getLayerType() const { return "TypeConverter"; }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());
        assert(getFeatureOutput().isPresent());

        // Implementation has 1 extra dimension due to having the batchSize dimension
        ThorImplementation::TypeConversion *typeConverter =
            new ThorImplementation::TypeConversion(Tensor::convertToImplementationDataType(getFeatureOutput().get().getDataType()));
        Thor::Layer::connectTwoLayers(drivingLayer, typeConverter, drivingApiLayer, this, connectingApiTensor);
        return typeConverter;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        assert(getFeatureOutput().isPresent());
        return getFeatureOutput().get().getTotalSizeInBytes();
    }
};

class TypeConverter::Builder {
   public:
    virtual TypeConverter build() {
        assert(_network.isPresent());
        assert(_featureInput.isPresent());
        assert(_newDataType.isPresent());

        TypeConverter TypeConverter;
        TypeConverter.featureInput = _featureInput;
        TypeConverter.featureOutput = Tensor(_newDataType.get(), _featureInput.get().getDimensions());
        TypeConverter.initialized = true;
        TypeConverter.addToNetwork(_network.get());
        return TypeConverter;
    }

    virtual TypeConverter::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual TypeConverter::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual TypeConverter::Builder &newDataType(Tensor::DataType _newDataType) {
        assert(!this->_newDataType.isPresent());
        assert(Tensor::dataTypeValid(_newDataType));
        this->_newDataType = _newDataType;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<Tensor> _featureInput;
    Optional<Tensor::DataType> _newDataType;
};

}  // namespace Thor
