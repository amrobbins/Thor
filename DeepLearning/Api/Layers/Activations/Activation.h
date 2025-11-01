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

    virtual std::string getLayerType() const = 0;
    virtual std::string getLayerVersion() const { return "1.0.0"; }

    virtual nlohmann::json serialize(const std::string& storageDir, Stream stream) {
        assert(initialized);
        assert(featureInput.isPresent());
        assert(featureOutput.isPresent());

        nlohmann::json j;
        j["factory"] = "activation";
        j["version"] = getLayerVersion();
        j["layer_type"] = to_snake_case(getLayerType());
        j["feature_input"] = featureInput.get().serialize();
        j["feature_output"] = featureOutput.get().serialize();
        return j;
    }

   protected:
    std::string to_snake_case(const std::string& input) {
        std::string out;
        out.reserve(input.size() * 2);

        for (size_t i = 0; i < input.size(); ++i) {
            char c = input[i];
            if (std::isupper(c)) {
                if (i > 0)
                    out.push_back('_');
                out.push_back(std::tolower(c));
            } else {
                out.push_back(c);
            }
        }
        return out;
    }
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

    virtual std::shared_ptr<Layer> build() = 0;
    // You can clone a builder to instantiate multiple distinct instances because the id is only generated when build() is called.
    // So each builder that is built into an activation will have its own unique id.
    virtual std::shared_ptr<Builder> clone() = 0;

   protected:
    Optional<Network*> _network;
    Optional<Tensor> _featureInput;
};

}  // namespace Thor
