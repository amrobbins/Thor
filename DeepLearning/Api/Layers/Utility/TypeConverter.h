#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Utility/TypeConversion.h"

namespace Thor {

class TypeConverter : public Layer {
   public:
    class Builder;
    TypeConverter();

    virtual ~TypeConverter();

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<TypeConverter>(*this); }

    virtual std::string getLayerType() const { return "TypeConverter"; }

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) const;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(connectingApiTensor == getFeatureInput());
        assert(getFeatureOutput().isPresent());

        // Implementation has 1 extra dimension due to having the batchSize dimension
        std::shared_ptr<ThorImplementation::TypeConversion> typeConverter = std::make_shared<ThorImplementation::TypeConversion>(
            Tensor::convertToImplementationDataType(getFeatureOutput().get().getDataType()));
        return typeConverter;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
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
