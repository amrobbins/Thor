#pragma once

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "Utilities/TarFile/TarWriter.h"

namespace Thor {

class PlacedNetwork {
   public:
    // Deep-copy the network; steal stamps from caller's lvalue vector
    PlacedNetwork(const std::string& networkName, const Network& network, std::vector<ThorImplementation::StampedNetwork>& initialStamps)
        : networkName(networkName), network(network), stampedNetworks(std::exchange(initialStamps, {})) {
        // FIXME: Make a Network deep copy constructor
        assert(stampedNetworks.size() >= 1);
    }

    ~PlacedNetwork();

    // Deep-copy the network; move the rvalue
    PlacedNetwork(const Network& network, std::vector<ThorImplementation::StampedNetwork>&& initialStamps)
        : network(network), stampedNetworks(std::move(initialStamps)) {
        assert(stampedNetworks.size() >= 1);
    }

    void save(const std::string& directory, bool overwrite, bool saveOptimizerState);

    uint64_t getNumStamps() { return stampedNetworks.size(); }
    ThorImplementation::StampedNetwork& getStampedNetwork(uint64_t i) {
        assert(i < stampedNetworks.size());
        return stampedNetworks[i];
    }
    virtual std::string getNetworkName() { return networkName; }
    uint32_t getNumTrainableLayers() { return network.getNumTrainableLayers(); }

   protected:
    std::string networkName;
    Thor::Network network;
    std::vector<ThorImplementation::StampedNetwork> stampedNetworks;

    std::shared_ptr<thor_file::TarWriter> archiveWriter = nullptr;
};

}  // namespace Thor
