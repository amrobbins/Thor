#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Utility/TypeConversion.h"
#include <optional>

namespace Thor {

class TypeConverter : public Layer {
   public:
    class Builder;
    TypeConverter();

    ~TypeConverter() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<TypeConverter>(*this); }

    std::string getLayerType() const override { return "TypeConverter"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());
        THOR_THROW_IF_FALSE(getFeatureOutput().has_value());

        // Implementation has 1 extra dimension due to having the batchSize dimension
        std::shared_ptr<ThorImplementation::TypeConversion> typeConverter =
            std::make_shared<ThorImplementation::TypeConversion>(getFeatureOutput().value().getDataType());
        return typeConverter;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        THOR_THROW_IF_FALSE(getFeatureOutput().has_value());
        return getFeatureOutput().value().getTotalSizeInBytes();
    }
};

class TypeConverter::Builder {
   public:
    virtual TypeConverter build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());
        THOR_THROW_IF_FALSE(_newDataType.has_value());

        TypeConverter TypeConverter;
        TypeConverter.featureInput = _featureInput.value();
        TypeConverter.featureOutput = Tensor(_newDataType.value(), _featureInput.value().getDimensions());
        TypeConverter.initialized = true;
        TypeConverter.addToNetwork(_network.value());
        return TypeConverter;
    }

    virtual TypeConverter::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual TypeConverter::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual TypeConverter::Builder &newDataType(Tensor::DataType _newDataType) {
        THOR_THROW_IF_FALSE(!this->_newDataType.has_value());
        THOR_THROW_IF_FALSE(Tensor::dataTypeValid(_newDataType));
        this->_newDataType = _newDataType;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<Tensor::DataType> _newDataType;
};

}  // namespace Thor
