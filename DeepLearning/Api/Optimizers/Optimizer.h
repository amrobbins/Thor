#pragma once

#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"

#include <nlohmann/json.hpp>

#include <string>
#include <unordered_map>
#include <unordered_set>

// Note: The interface here is what the end user can use, and should include only all of the functionality that the user may want,
//       that is common to all (or the vast majority) of optimizers.

namespace Thor {

class Network;

class Optimizer {
   public:
    virtual ~Optimizer();

    /*
     * returns a map of updated parameters
     */
    static void updateHyperParameters(Network *network, uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);
    virtual std::unordered_map<std::string, float> getAllHyperParameters();

    virtual void save(std::string filename) { assert(false); /*FIXME*/ };
    virtual void load(std::string filename) { assert(false); /*FIXME*/ };

    uint64_t getId() const { return id; }
    bool operator==(const Optimizer &other) const { return id == other.id; }

    using Deserializer = std::function<void(const nlohmann::json &, Network *, uint64_t layerId)>;
    static std::unordered_map<std::string, Deserializer> &getRegistry();
    static void registerLayer(std::string name, Deserializer fn);
    static void deserialize(const nlohmann::json &j, Network *network, uint64_t layerId);

    virtual std::shared_ptr<ThorImplementation::Optimizer> stamp(
        std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer) = 0;

    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                          bool isFirstStamp,
                                          std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                          Optional<Event> sisterOptimizerLoadedEvent) = 0;

   protected:
    // Only subclasses can be instantiated
    Optimizer();

    virtual std::shared_ptr<Optimizer> clone() const = 0;

    void addToNetwork(Network *network);

    Network *network;

    // gpuNum -> stampedId -> *optimizer
    // So there is one optimizer per layer, shared by all stamps of the network
    // I think the fix is to iterate over the learning layers
    // std::unordered_map<uint32_t, std::unordered_map<int64_t, std::shared_ptr<ThorImplementation::Optimizer>>> optimizersShared;
    // std::unordered_map<uint32_t, std::unordered_map<int64_t, ThorImplementation::Optimizer *>> optimizers;

    friend class Network;

   private:
    uint64_t id;
    static std::atomic<int64_t> nextId;
};

}  // namespace Thor