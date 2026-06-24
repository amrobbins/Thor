#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"

#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"

#include <string>
#include <optional>

namespace Thor {

class NetworkInput : public Layer {
   public:
    class Builder;

    NetworkInput() {}

    ~NetworkInput() override {}

    virtual std::string getName() const { return name; }
    std::vector<uint64_t> getDimensions() const { return dimensions; }
    DataType getDataType() const { return dataType; }
    bool dimensionsIncludeBatch() const { return dimensionsIncludeBatch_; }
    bool isExternal() const { return external_; }
    bool hasPassThroughSource() const { return passThroughSource_.has_value(); }
    Tensor getPassThroughSource() const {
        THOR_THROW_IF_FALSE(passThroughSource_.has_value());
        return passThroughSource_.value();
    }

    std::shared_ptr<Layer> clone() const override { return std::make_shared<NetworkInput>(*this); }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override {
        (void)inputTensor;
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        return {featureOutput.value()};
    }

    std::string getLayerType() const override { return "NetworkInput"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual std::shared_ptr<ThorImplementation::NetworkInput> stamp(ThorImplementation::TensorPlacement placement,
                                                                    uint32_t batchSize) const {
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(!passThroughSource_.has_value());

        std::vector<uint64_t> physicalDimensions;
        if (dimensionsIncludeBatch_) {
            (void)batchSize;
            physicalDimensions = dimensions;
        } else {
            physicalDimensions.push_back(batchSize);
            for (uint32_t i = 0; i < dimensions.size(); ++i)
                physicalDimensions.push_back(dimensions[i]);
        }

        std::shared_ptr<ThorImplementation::NetworkInput> networkInput;
        networkInput = std::make_shared<ThorImplementation::NetworkInput>(placement, dataType, physicalDimensions);
        networkInput->setName(name);

        return networkInput;
    }

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        (void)connectingApiTensor;
        (void)inferenceOnly;
        THOR_UNREACHABLE();
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)batchSize;
        (void)tensorPlacement;
        if (passThroughSource_.has_value()) {
            return 0;
        }
        // Input has a prefetch buffer in addition to storing the output tensor
        return 2 * featureOutput.value().getTotalSizeInBytes();
    }

   private:
    std::string name;
    std::vector<uint64_t> dimensions;
    DataType dataType;
    bool dimensionsIncludeBatch_ = false;
    bool external_ = true;
    std::optional<Tensor> passThroughSource_;

    friend class Network;
};

class NetworkInput::Builder {
   public:
    virtual NetworkInput build();

    virtual NetworkInput::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual NetworkInput::Builder &name(const std::string &_name) {
        THOR_THROW_IF_FALSE(!_name.empty());
        THOR_THROW_IF_FALSE(!this->_name.has_value());
        this->_name = _name;
        return *this;
    }

    // Note: dimensions do not include batch size
    virtual NetworkInput::Builder &dimensions(const std::vector<uint64_t> &_dimensions) {
        THOR_THROW_IF_FALSE(!_dimensions.empty());
        this->_dimensions = _dimensions;
        return *this;
    }

    virtual NetworkInput::Builder &dataType(const DataType &_dataType) {
        THOR_THROW_IF_FALSE(Tensor::dataTypeValid(_dataType));
        this->_dataType = _dataType;
        return *this;
    }

    virtual NetworkInput::Builder &dimensionsIncludeBatch(bool includeBatch) {
        this->_dimensionsIncludeBatch = includeBatch;
        return *this;
    }

    virtual NetworkInput::Builder &external(bool isExternal) {
        this->_external = isExternal;
        return *this;
    }

    virtual NetworkInput::Builder &passThroughSource(const Tensor &sourceTensor) {
        THOR_THROW_IF_FALSE(sourceTensor.isInitialized());
        this->_passThroughSource = sourceTensor;
        return *this;
    }

   private:
    std::optional<std::string> _name;
    std::optional<Network *> _network;
    std::optional<std::vector<uint64_t>> _dimensions;
    std::optional<DataType> _dataType;
    bool _dimensionsIncludeBatch = false;
    bool _external = true;
    std::optional<Tensor> _passThroughSource;
};

}  // namespace Thor
