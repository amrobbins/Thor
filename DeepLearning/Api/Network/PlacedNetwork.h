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
#include <optional>
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
                      bool waitForOutputsOnProcessingStream = true,
                      ThorImplementation::BatchSubmissionTiming* submitTiming = nullptr,
                      std::optional<uint32_t> outputSlotIndex = std::nullopt);
    Event submitBatch(uint64_t stampIndex,
                      std::map<std::string, ThorImplementation::Tensor> batchInputs,
                      const std::map<std::string, Event>& inputReadyEvents,
                      std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                      std::map<std::string, Event>& outputReadyEvents,
                      bool isInferenceOnly,
                      Event* reusableProcessingFinishedEvent = nullptr,
                      bool waitForOutputsOnProcessingStream = true,
                      ThorImplementation::BatchSubmissionTiming* submitTiming = nullptr,
                      std::optional<uint32_t> outputSlotIndex = std::nullopt);
    Event submitBatch(uint64_t stampIndex,
                      std::map<std::string, ThorImplementation::Tensor> batchInputs,
                      std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                      std::map<std::string, Event>& outputReadyEvents,
                      bool isInferenceOnly,
                      const std::vector<Tensor>& activeTrainingLossRoots,
                      Event* reusableProcessingFinishedEvent = nullptr,
                      bool waitForOutputsOnProcessingStream = true,
                      ThorImplementation::BatchSubmissionTiming* submitTiming = nullptr,
                      std::optional<uint32_t> outputSlotIndex = std::nullopt);
    Event submitBatch(uint64_t stampIndex,
                      const Batch& batchInputs,
                      std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                      std::map<std::string, Event>& outputReadyEvents,
                      bool isInferenceOnly,
                      Event* reusableProcessingFinishedEvent = nullptr,
                      bool waitForOutputsOnProcessingStream = true,
                      ThorImplementation::BatchSubmissionTiming* submitTiming = nullptr,
                      std::optional<uint32_t> outputSlotIndex = std::nullopt);
    Event submitBatch(uint64_t stampIndex,
                      const Batch& batchInputs,
                      std::map<std::string, ThorImplementation::Tensor>& batchOutputs,
                      std::map<std::string, Event>& outputReadyEvents,
                      bool isInferenceOnly,
                      const std::vector<Tensor>& activeTrainingLossRoots,
                      Event* reusableProcessingFinishedEvent = nullptr,
                      bool waitForOutputsOnProcessingStream = true,
                      ThorImplementation::BatchSubmissionTiming* submitTiming = nullptr,
                      std::optional<uint32_t> outputSlotIndex = std::nullopt);
    std::vector<uint64_t> getActiveTrainingRawLossOriginalIdsForDebug(uint64_t stampIndex = 0) const;

    void extendOutputWritableEvents(uint64_t stampIndex, Event event, std::optional<uint32_t> outputSlotIndex = std::nullopt);
    void preallocateInputSlots(uint32_t numSlots);
    void preallocateOutputSlots(uint32_t numSlots);

    // Copy parameter storage (and optimizer-owned tensor state when present)
    // from a previous placement of the same API network.  This lets Trainer.fit
    // rebuild a fresh physical graph after phase mutations without resetting
    // learned weights.
    void copyTrainingStateFrom(PlacedNetwork& source);

    // Copy parameter/optimizer state for layers that share a stable clone-source
    // identity.  This is used by composed phase graphs, where fresh placement may
    // introduce new API layer ids while preserving phase/source-layer identity.
    void copyMatchingTrainingStateFrom(PlacedNetwork& source);

    uint64_t getNumStamps() { return stampedNetworks.size(); }
    ThorImplementation::StampedNetwork& getStampedNetwork(uint64_t i) {
        THOR_THROW_IF_FALSE(i < stampedNetworks.size());
        return stampedNetworks[i];
    }
    virtual std::string getNetworkName() { return networkName; }
    uint32_t getNumTrainableLayers() { return network.getNumTrainableLayers(); }
    std::vector<ParameterReference> getTrainableParameterReferences(bool trainingEnabledOnly = true);

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
