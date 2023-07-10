#pragma once

#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"

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
    virtual std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);
    virtual std::unordered_map<std::string, float> getAllHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);

    virtual void save(std::string filename) { assert(false); /*FIXME*/ };
    virtual void load(std::string filename) { assert(false); /*FIXME*/ };

    uint64_t getId() const { return id; }
    bool operator==(const Optimizer &other) const { return id == other.id; }

   protected:
    // Only subclasses can be instantiated
    Optimizer();

    virtual std::shared_ptr<Optimizer> clone() const = 0;

    void addToNetwork(Network *network);
    virtual std::shared_ptr<ThorImplementation::Optimizer> stamp(
        std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer) = 0;

    // To be accessed by Thor::Network
    void attachToNetwork();
    void disconnectFromNetwork();
    void destroy();

    Network *network;

    // gpuNum -> stampedId -> *optimizer
    // So there is one optimizer per layer, shared by all stamps of the network
    std::unordered_map<uint32_t, std::unordered_map<int64_t, std::shared_ptr<ThorImplementation::Optimizer>>> optimizersShared;
    std::unordered_map<uint32_t, std::unordered_map<int64_t, ThorImplementation::Optimizer *>> optimizers;

    friend class Network;

   private:
    uint64_t id;
    static std::atomic<int64_t> nextId;
};

}  // namespace Thor