#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"

#include <assert.h>
#include <atomic>
#include <utility>

namespace Thor {

class Activation : public Layer {
   public:
    class Builder;

    Activation() {}
    virtual ~Activation() {}

    // Layer::addToNetwork is used during deserialization when an activation is an actual attached layer - or when the activation
    // is used as a standalone layer.
    // Activation::addToNetwork is used when an attached layer is added to the network as templated by that particular activation.
    using Layer::addToNetwork;
    // Activation template version
    virtual Tensor addToNetwork(Tensor inputTensor, Network* network);

    virtual std::string getLayerType() const = 0;

    // Standalone layer version
    virtual nlohmann::json serialize(thor_file::TarWriter& archiveWriter, Stream stream) const;
    // Activation template version
    virtual nlohmann::json serialize(Tensor inputTensor, Tensor outputTensor) const;

    static void deserialize(const nlohmann::json& j, Network* network);
    using Deserializer = std::function<void(const nlohmann::json&, Network*)>;
    static std::unordered_map<std::string, Deserializer>& get_registry();
    static void register_layer(std::string name, Deserializer fn);
};

class Activation::Builder {
   public:
    virtual ~Builder() {}
    virtual Activation::Builder& network(Network& _network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }
    virtual Activation::Builder& featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        assert(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual std::shared_ptr<Activation> build() = 0;

   protected:
    Optional<Network*> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
