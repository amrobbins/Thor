#pragma once

#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <assert.h>
#include <memory>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace Thor {

class Initializer {
   public:
    class Builder;

    Initializer() { initialized = false; }

    virtual ~Initializer() {}

    virtual std::shared_ptr<Initializer> clone() const = 0;

    // Referring to the initializer object, not the tensor that gets initialized:
    bool isInitialized() { return initialized; }

    virtual void stamp(ThorImplementation::Layer *layerThatOwnsTensor, ThorImplementation::Tensor tensorToInitialize) = 0;

    virtual Event initialize(ThorImplementation::Tensor tensorToInitialize, ThorImplementation::Layer *layerThatOwnsTensor) {
        stamp(layerThatOwnsTensor, tensorToInitialize);
        initDoneEvent = layerThatOwnsTensor->initializeTensor(tensorToInitialize);
        return initDoneEvent;
    }

    virtual nlohmann::json serialize() const = 0;
    static std::shared_ptr<Initializer> deserialize(const nlohmann::json &j);
    using Deserializer = std::function<std::shared_ptr<Initializer>(const nlohmann::json &)>;
    static std::unordered_map<std::string, Deserializer> &getRegistry();
    static void registerLayer(std::string name, Deserializer fn);

    virtual std::string getVersion() const;

   protected:
    Optional<Event> initDoneEvent;

    // Referring to the initializer object, not the tensor that gets initialized:
    bool initialized;

    // friend struct StampedNetwork;
};

class Initializer::Builder {
   public:
    virtual ~Builder() {}
    virtual std::shared_ptr<Initializer> build() = 0;
    virtual std::shared_ptr<Builder> clone() = 0;
};

}  // namespace Thor
