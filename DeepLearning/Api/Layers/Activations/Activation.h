#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/Expression.h"

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

    // Returns an expression equivalent to applying this activation to the input expression.
    // This is used by expression-backed learning layers to fuse the activation into the layer equation.
    virtual ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const;

    virtual std::string getLayerType() const = 0;

    virtual nlohmann::json architectureJson() const;
    virtual nlohmann::json serialize(thor_file::TarWriter& archiveWriter, Stream stream) const { return architectureJson(); }

    static void deserialize(const nlohmann::json& j, Network* network);
    using Deserializer = std::function<void(const nlohmann::json&, Network*)>;
    static std::unordered_map<std::string, Deserializer>& get_registry();
    static void register_layer(std::string name, Deserializer fn);

   private:
    using Layer::serialize;
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
