#pragma once

#include "DeepLearning/Api/Network/Network.h"

#include <string>
#include <unordered_map>

using std::shared_ptr;

class Optimizer {
   public:
    Optimizer() {}
    virtual ~Optimizer() {}

    virtual void setNetwork(Thor::Network *network) = 0;

    // returns a map of updated parameters
    virtual std::unordered_map<std::string, float> updateParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) = 0;
    virtual std::unordered_map<std::string, float> initializeStampedNetworkParameters(ThorImplementation::StampedNetwork &stampedNetwork,
                                                                                      uint64_t epoch,
                                                                                      uint64_t batch,
                                                                                      uint64_t batchesPerEpoch) = 0;
    virtual std::unordered_map<std::string, float> getAllParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) = 0;
};
