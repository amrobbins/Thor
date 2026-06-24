#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"

#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include <optional>

namespace Thor {

class NetworkOutput : public Layer {
   public:
    class Builder;

    NetworkOutput() {}

    ~NetworkOutput() override {}

    virtual std::string getName() const { return name; }
    bool isExternal() const { return external_; }

    std::shared_ptr<Layer> clone() const override { return std::make_shared<NetworkOutput>(*this); }

    DataType getDataType() const { return dataType; }

    std::string getLayerType() const override { return "NetworkOutput"; }

    virtual bool isMultiLayer() const {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().getDataType() != dataType;
    }

    virtual void buildSupportLayersAndAddToNetwork();

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
        THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());

        std::shared_ptr<ThorImplementation::NetworkOutput> networkOutput = std::make_shared<ThorImplementation::NetworkOutput>(placement);
        networkOutput->setName(name);
        return networkOutput;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // If there is a type conversion required, the memory requirement is reported by the TypeConverter layer
        return 0;
    }

   private:
    std::string name;
    DataType dataType;
    bool external_ = true;
    Network *network;
};

class NetworkOutput::Builder {
   public:
    virtual NetworkOutput build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_inputTensor.has_value());
        if (!_dataType.has_value())
            _dataType = _inputTensor.value().getDataType();

        NetworkOutput networkOutput;
        if (_name.has_value())
            networkOutput.name = _name.value();
        else
            networkOutput.name = std::string("NetworkOutput") + std::to_string(networkOutput.getId());
        networkOutput.dataType = _dataType.value();
        networkOutput.external_ = _external;
        networkOutput.featureInput = _inputTensor.value();
        networkOutput.initialized = true;
        networkOutput.network = _network.value();

        if (networkOutput.isMultiLayer()) {
            // A type converter will be stamped where the new data type will take effect, when it is needed.
            networkOutput.buildSupportLayersAndAddToNetwork();
        } else {
            networkOutput.featureOutput = Tensor(_inputTensor.value().getDataType(), _inputTensor.value().getDimensions());
            networkOutput.addToNetwork(_network.value());
        }

        return networkOutput;
    }

    virtual NetworkOutput::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual NetworkOutput::Builder &name(const std::string &_name) {
        THOR_THROW_IF_FALSE(!_name.empty());
        THOR_THROW_IF_FALSE(!this->_name.has_value());
        this->_name = _name;
        return *this;
    }

    virtual NetworkOutput::Builder &inputTensor(const Tensor &_inputTensor) {
        THOR_THROW_IF_FALSE(_inputTensor.isInitialized());
        this->_inputTensor = _inputTensor;
        return *this;
    }

    virtual NetworkOutput::Builder &dataType(const DataType &_dataType) {
        THOR_THROW_IF_FALSE(Tensor::dataTypeValid(_dataType));
        this->_dataType = _dataType;
        return *this;
    }

    virtual NetworkOutput::Builder &external(bool isExternal) {
        this->_external = isExternal;
        return *this;
    }

   private:
    std::optional<std::string> _name;
    std::optional<Network *> _network;
    std::optional<Tensor> _inputTensor;
    std::optional<DataType> _dataType;
    bool _external = true;
};

}  // namespace Thor
