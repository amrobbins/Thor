#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/Expression.h"

#include <atomic>
#include <utility>
#include <optional>

namespace Thor {

class Activation : public Layer {
   public:
    class Builder;

    Activation() {}
    ~Activation() override {}

    // Layer::addToNetwork is used during deserialization when an activation is an actual attached layer - or when the activation
    // is used as a standalone layer.
    // Activation::addToNetwork is used when an attached layer is added to the network as templated by that particular activation.
    using Layer::addToNetwork;
    // Activation template version
    virtual Tensor addToNetwork(Tensor inputTensor, Network* network);

    // Returns an expression equivalent to applying this activation to the input expression.
    // This is used by expression-backed learning layers to fuse the activation into the layer equation.
    virtual ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const = 0;

    std::string getLayerType() const override = 0;

    nlohmann::json architectureJson() const override;
    nlohmann::json serialize(thor_file::TarWriter& archiveWriter, Stream stream) const override { return architectureJson(); }

    static void deserialize(const nlohmann::json& j, Network* network);
    static std::shared_ptr<Activation> deserializeTemplate(const nlohmann::json& j);
    using Deserializer = std::function<void(const nlohmann::json&, Network*)>;
    static std::unordered_map<std::string, Deserializer>& get_registry();
    static void register_layer(std::string name, Deserializer fn);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stampExpressionBackedActivation(ThorImplementation::TensorPlacement placement,
                                                                                Thor::Tensor connectingApiTensor,
                                                                                bool inferenceOnly) const;

   private:
    using Layer::serialize;
};

class Activation::Builder {
   public:
    virtual ~Builder() {}
    virtual Activation::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }
    virtual Activation::Builder& featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual std::shared_ptr<Activation> build() = 0;

   protected:
    std::optional<Network*> _network;
    std::optional<Tensor> _featureInput;
};

}  // namespace Thor
