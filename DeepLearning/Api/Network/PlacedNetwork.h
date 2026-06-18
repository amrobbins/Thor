#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Api/Loaders/Batch.h"
#include "DeepLearning/Api/Parameter/BoundParameter.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TarFile/TarWriter.h"

#include <map>
#include <string>
#include <vector>

namespace Thor {

class PlacedNetwork {
   public:
    // Deep-copy the network; steal stamps from caller's lvalue vector
    PlacedNetwork(const std::string& networkName, const Network& network, std::vector<ThorImplementation::StampedNetwork>& initialStamps)
        : networkName(networkName), network(network), stampedNetworks(std::exchange(initialStamps, {})) {
        // FIXME: Make a Network deep copy constructor
        THOR_THROW_IF_FALSE(stampedNetworks.size() >= 1);
    }

    ~PlacedNetwork();

    // Deep-copy the network; move the rvalue
    PlacedNetwork(const Network& network, std::vector<ThorImplementation::StampedNetwork>&& initialStamps)
        : network(network), stampedNetworks(std::move(initialStamps)) {
        THOR_THROW_IF_FALSE(stampedNetworks.size() >= 1);
    }

    void save(const std::string& directory, bool overwrite, bool saveOptimizerState);

    std::map<std::string, ThorImplementation::Tensor> infer(std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                                            uint64_t stampIndex = 0);
    std::map<std::string, ThorImplementation::Tensor> infer(const Batch& batchInputs, uint64_t stampIndex = 0);
    Event submitBatch(uint64_t stampIndex,
                      std::map<std::string, ThorImplementation::Tensor> batchInputs,
                      std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                      std::map<std::string, Event>& outputReadyEvents,
                      bool isInferenceOnly,
                      Event* reusableProcessingFinishedEvent = nullptr,
                      bool waitForOutputsOnProcessingStream = true);
    Event submitBatch(uint64_t stampIndex,
                      const Batch& batchInputs,
                      std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                      std::map<std::string, Event>& outputReadyEvents,
                      bool isInferenceOnly,
                      Event* reusableProcessingFinishedEvent = nullptr,
                      bool waitForOutputsOnProcessingStream = true);

    void extendOutputWritableEvents(uint64_t stampIndex, Event event);

    uint64_t getNumStamps() { return stampedNetworks.size(); }
    ThorImplementation::StampedNetwork& getStampedNetwork(uint64_t i) {
        THOR_THROW_IF_FALSE(i < stampedNetworks.size());
        return stampedNetworks[i];
    }
    virtual std::string getNetworkName() { return networkName; }
    uint32_t getNumTrainableLayers() { return network.getNumTrainableLayers(); }

    BoundParameter resolveParameterReference(const ParameterReference& parameterReference);
    std::vector<BoundParameter> resolveParameterReferences(const std::vector<ParameterReference>& parameterReferences);

    bool hasApiTensor(const Tensor& tensor);
    Tensor resolveApiTensor(const Tensor& tensor);
    std::vector<Tensor> resolveApiTensors(const std::vector<Tensor>& tensors);

    bool hasNetworkInput(const std::string& name);
    std::vector<std::string> getNetworkInputNames(uint64_t stampIndex = 0);

   protected:
    std::string networkName;
    Thor::Network network;
    std::vector<ThorImplementation::StampedNetwork> stampedNetworks;

    std::shared_ptr<thor_file::TarWriter> archiveWriter = nullptr;
};

}  // namespace Thor
