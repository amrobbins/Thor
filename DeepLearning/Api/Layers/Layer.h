#pragma once

#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Common/Optional.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <nlohmann/json.hpp>

#include <assert.h>
#include <atomic>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace Thor {

class Network;
class Initializer;

class Layer {
   public:
    Layer() : initialized(false), id(getUnusedId()) {}
    virtual ~Layer() {}

    uint64_t getId() const { return id; }

    virtual Optional<Tensor> getFeatureOutput() const { return featureOutput; }
    virtual Optional<Tensor> getFeatureInput() const { return featureInput; }

    virtual std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) {
        assert(getFeatureInput().isPresent());
        assert(getFeatureOutput().isPresent());
        assert(inputTensor == getFeatureInput().get());
        return {getFeatureOutput().get()};
    }

    virtual bool mustConnectAllInputsToDriveOutput() { return false; }
    virtual void informThatInputConnectionMade(Tensor inputTensor) {}

    bool isInitialized() { return initialized; }

    virtual void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) {}

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                           ThorImplementation::TensorPlacement tensorPlacement) const = 0;
    // Layers with weights that share the weights mem with other instances of the same layer on the same gpu will have less non first
    // instance fixed mem requirements.
    virtual uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                              ThorImplementation::TensorPlacement tensorPlacement) const {
        return getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }

    bool operator==(const Layer &other) const { return id == other.id; }
    bool operator!=(const Layer &other) const { return id != other.id; }
    bool operator<(const Layer &other) const { return id < other.id; }
    bool operator>(const Layer &other) const { return id > other.id; }

    virtual int getConnectionType(Tensor connectingTensor) const {
        assert(connectingTensor == getFeatureInput() || connectingTensor == getFeatureOutput());
        return 0;
    }

    virtual std::vector<Tensor> getAllOutputTensors() const { return {getFeatureOutput().get()}; }

    virtual std::shared_ptr<Layer> clone() const = 0;

    static uint64_t getUnusedId() { return nextId.fetch_add(1); }

    virtual std::string getLayerType() const = 0;

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) const { return nlohmann::json{}; }
    static void deserialize(const nlohmann::json &j, Stream stream, Network *network);

   protected:
    Optional<Tensor> featureInput;
    Optional<Tensor> featureOutput;

    // Note: The final API typed parameters are needed to choose from multiple types of output connections and input connections for
    // physical layers that are direct replacements for API layers.
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const = 0;

    // initialize() is called for a layer after it has been stamped, the first connection that is made to the layer.
    // often layers will not need initialize() at all.
    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Layer> layer,
                                          std::vector<std::shared_ptr<Initializer>> &initializers) {
        return {};
    }

    virtual void addToNetwork(Network *network);

    static void connectTwoLayers(std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                 std::shared_ptr<ThorImplementation::Layer> loadingLayer,
                                 const std::shared_ptr<Thor::Layer> drivingApiLayer = nullptr,
                                 const std::shared_ptr<Thor::Layer> loadingApiLayer = nullptr,
                                 const Thor::Tensor connectingApiTensor = Thor::Tensor());

    bool initialized;

    uint64_t id;

   private:
    static std::atomic<int64_t> nextId;

    friend class Network;
};

}  // namespace Thor
