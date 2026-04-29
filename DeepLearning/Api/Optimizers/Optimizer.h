#pragma once

#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Note: The interface here is what the end user can use, and should include only all of the functionality that the user may want,
//       that is common to all (or the vast majority) of optimizers.

namespace Thor {

class Network;
class PlacedNetwork;

class Optimizer {
   public:
    virtual ~Optimizer() = default;

    virtual std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) = 0;
    // FIXME: Update
    // virtual void compile(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer) { physicalOptimizer->compile(); }
    // virtual void compile(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer) { /*FIXME*/ }
    virtual void compile(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                         ThorImplementation::Tensor weights,
                         Stream gradientStream) {
        physicalOptimizer->compile(weights, gradientStream);
    }
    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                          bool isFirstStamp,
                                          std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                          Optional<Event> sisterOptimizerLoadedEvent) {
        return {};
    };

    /*
     * returns a map of updated parameters
     */
    static void updateHyperParameters(PlacedNetwork *placedNetwork, uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);
    virtual std::unordered_map<std::string, float> getAllHyperParameters(PlacedNetwork *placedNetwork);

    uint64_t getId() const { return id; }
    bool operator==(const Optimizer &other) const { return id == other.id; }

    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                     std::string filenamePrefix,
                                     bool saveOptimizerState) const {
        return architectureJson();
    }

    // FIXME: remove network pointer from deserialize
    static std::shared_ptr<Optimizer> deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader,
                                                  const nlohmann::json &j,
                                                  Network *network);
    using Deserializer = std::function<std::shared_ptr<Optimizer>(
        std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &, Network *network)>;
    static std::unordered_map<std::string, Deserializer> &getRegistry();
    static void registerLayer(std::string name, Deserializer fn);

    virtual nlohmann::json architectureJson() const = 0;

    virtual std::string getVersion() const;

    virtual std::string getType() const = 0;

    uint64_t getOriginalId() const { return originalId; }

    virtual std::shared_ptr<Optimizer> clone() const = 0;

   protected:
    // Only subclasses can be instantiated
    Optimizer();
    Optimizer(uint64_t originalId);

    void addToNetwork(Network *network);

    uint64_t originalId;

    // gpuNum -> stampedId -> *optimizer
    // So there is one optimizer per layer, shared by all stamps of the network
    // I think the fix is to iterate over the learning layers
    // std::unordered_map<uint32_t, std::unordered_map<int64_t, std::shared_ptr<ThorImplementation::Optimizer>>> optimizersShared;
    // std::unordered_map<uint32_t, std::unordered_map<int64_t, ThorImplementation::Optimizer *>> optimizers;

    friend class Network;

   private:
    uint64_t id;
    static std::atomic<int64_t> nextId;

    static std::mutex originalIdMapLock;
    static std::unordered_map<uint64_t, uint64_t> orignalIdToId;
};

}  // namespace Thor
