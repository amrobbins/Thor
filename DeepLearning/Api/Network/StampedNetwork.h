#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Loaders/Batch.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "Utilities/Common/Event.h"

#include <vector>
#include <optional>
#include <cstdint>
#include <map>
#include <string>

#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"

namespace Thor {
class Network;
class PlacedNetwork;
class LocalExecutor;
}  // namespace Thor

namespace ThorImplementation {

struct BatchSubmissionTiming {
    uint64_t activeObjectiveRootsMicros = 0;
    uint64_t setActiveObjectiveRootsMicros = 0;
    uint64_t sendBatchMicros = 0;
    uint64_t batchUnwrapMicros = 0;
    uint64_t physicalTotalMicros = 0;
    uint64_t inputForwardMicros = 0;
    uint64_t outputCollectMicros = 0;
    uint64_t outputWaitOnProcessingMicros = 0;
    uint64_t processingEventMicros = 0;
    uint64_t inputFanoutMicros = 0;
    uint64_t totalMicros = 0;
    uint64_t numInputs = 0;
    uint64_t numOutputs = 0;
    uint64_t activeObjectiveRootCount = 0;
};

inline void accumulateBatchSubmissionTiming(BatchSubmissionTiming& dst, const BatchSubmissionTiming& src) {
    dst.activeObjectiveRootsMicros += src.activeObjectiveRootsMicros;
    dst.setActiveObjectiveRootsMicros += src.setActiveObjectiveRootsMicros;
    dst.sendBatchMicros += src.sendBatchMicros;
    dst.batchUnwrapMicros += src.batchUnwrapMicros;
    dst.physicalTotalMicros += src.physicalTotalMicros;
    dst.inputForwardMicros += src.inputForwardMicros;
    dst.outputCollectMicros += src.outputCollectMicros;
    dst.outputWaitOnProcessingMicros += src.outputWaitOnProcessingMicros;
    dst.processingEventMicros += src.processingEventMicros;
    dst.inputFanoutMicros += src.inputFanoutMicros;
    dst.totalMicros += src.totalMicros;
    dst.numInputs += src.numInputs;
    dst.numOutputs += src.numOutputs;
    dst.activeObjectiveRootCount += src.activeObjectiveRootCount;
}

class StampedNetwork {
   private:
    struct LayerComparatorShared {
        bool operator()(const std::shared_ptr<Layer> &lhs, const std::shared_ptr<Layer> &rhs) const { return lhs->getId() < rhs->getId(); }
    };
    struct LayerComparator {
        bool operator()(const Layer *lhs, const Layer *rhs) const { return lhs->getId() < rhs->getId(); }
    };
    template <typename T>
    int count(std::vector<T> v, T item) {
        uint32_t c = 0;
        for (uint32_t i = 0; i < v.size(); ++i) {
            if (v[i] == item)
                c += 1;
        }
        return c;
    }

   public:
    uint32_t getGpuNum() const { return gpuNum; }
    uint64_t getFloatingPointOperationsPerExampleForward() const { return floatingPointOperationsPerExampleForward; }
    uint64_t getFloatingPointOperationsPerExampleBackward() const { return floatingPointOperationsPerExampleBackward; }
    uint64_t getFloatingPointOperationsPerExampleTraining() const {
        return floatingPointOperationsPerExampleForward + floatingPointOperationsPerExampleBackward;
    }
    struct RaggedInputBinding {
        std::string valuesInputName;
        std::string offsetsInputName;
        ThorImplementation::RaggedTensorDescriptor descriptor;
    };

    std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> getInputs() { return inputsShared; }
    std::vector<std::shared_ptr<ThorImplementation::NetworkOutput>> getOutputs() { return outputsShared; }

    /**
     * Records synchronization events for every physical layer in this stamp.
     * Synchronizing every returned event guarantees that all work enqueued for
     * the stamp before this call, including parameter/optimizer initialization,
     * has completed. Callers must prevent concurrent batch submission while
     * taking this synchronization snapshot.
     */
    std::vector<Event> getSynchronizeEvents() const;

    std::vector<std::string> getNamedInputNames() const {
        std::vector<std::string> names;
        names.reserve(inputNamedShared.size());
        for (const auto& [name, _] : inputNamedShared) {
            names.push_back(name);
        }
        return names;
    }
    std::vector<std::string> getNamedOutputNames() const {
        std::vector<std::string> names;
        names.reserve(outputNamedShared.size());
        for (const auto& [name, _] : outputNamedShared) {
            names.push_back(name);
        }
        return names;
    }
    std::shared_ptr<ThorImplementation::NetworkInput> getNamedInput(const std::string& name) const {
        auto it = inputNamedShared.find(name);
        if (it == inputNamedShared.end()) {
            return nullptr;
        }
        return it->second;
    }
    std::shared_ptr<ThorImplementation::NetworkOutput> getNamedOutput(const std::string& name) const {
        auto it = outputNamedShared.find(name);
        if (it == outputNamedShared.end()) {
            return nullptr;
        }
        return it->second;
    }
    void preallocateInputSlots(uint32_t numSlots);
    void preallocateOutputSlots(uint32_t numSlots);

    uint64_t getNumTrainableLayers() { return trainableLayersShared.size(); }
    std::shared_ptr<ThorImplementation::TrainableLayer> &getTrainableLayer(uint64_t i) { return trainableLayersShared[i]; }
    std::vector<std::shared_ptr<ThorImplementation::Layer>> getOtherLayers() { return otherLayersShared; }
    void setActiveTrainingLossRoots(const std::vector<Thor::Tensor>& activeRawLossRoots);
    std::vector<uint64_t> getActiveTrainingRawLossOriginalIdsForDebug() const;

    std::shared_ptr<ThorImplementation::Layer> getPhysicalLayerFromApiLayer(uint64_t apiLayerId) {
        return apiLayerToPhysicalLayerShared[apiLayerId];
    }
    std::shared_ptr<ThorImplementation::Layer> getPhysicalLayerFromApiLayer(std::shared_ptr<Thor::Layer> apiLayer) {
        return apiLayerToPhysicalLayerShared[apiLayer->getId()];
    }
    // void recordIfParameterizable(std::shared_ptr<Thor::Layer> layer, std::shared_ptr<ThorImplementation::Layer> implementationLayer) {
    //     std::shared_ptr<Thor::Parameterizable> parameterizable = dynamic_pointer_cast<Thor::Parameterizable>(layer);
    //     if (parameterizable != nullptr) {
    //         auto implementationParameterizable = std::dynamic_pointer_cast<ThorImplementation::Parameterizable>(implementationLayer);
    //         THOR_THROW_IF_FALSE(implementationParameterizable != nullptr);
    //         apiParameterizableToPhysicalParameterizable[parameterizable->getId()] = implementationParameterizable;
    //     }
    // }
    // std::shared_ptr<ThorImplementation::Parameterizable> getPhysicalParameterizableFromApiParameterizable(uint64_t apiParameterizableId)
    // {
    //     auto it = apiParameterizableToPhysicalParameterizable.find(apiParameterizableId);
    //     THOR_THROW_IF_FALSE(it != apiParameterizableToPhysicalParameterizable.end());
    //     return it->second;
    // }
    // std::shared_ptr<ThorImplementation::Parameterizable> getPhysicalParameterizableFromApiParameterizable(
    //     std::shared_ptr<Thor::Layer> apiParameterizable) {
    //     THOR_THROW_IF_FALSE(apiParameterizable != nullptr);
    //     uint64_t apiParameterizableId = apiParameterizable->getId();
    //     return getPhysicalParameterizableFromApiParameterizable(apiParameterizableId);
    // }

#if defined(THOR_GTEST) || defined(__JETBRAINS_IDE__)
    std::map<uint64_t, std::shared_ptr<ThorImplementation::Layer>> getApiLayerToPhysicalLayer() { return apiLayerToPhysicalLayerShared; }
#endif

   protected:
    void initialize(bool initializeWeights, bool copyWeightsFromOtherStamp, StampedNetwork *otherStamp = nullptr);

    // Note that all processing is finished at the end of any input stream of the stamp.
    // Note *input* stream - this is not the case for the loader streams
    Event sendBatch(std::map<std::string, Tensor> batchInputs,
                    std::map<std::string, Tensor> &batchOutputs,
                    std::map<std::string, Event> &outputReadyEvents,
                    bool isInferenceOnly,
                    Event* reusableProcessingFinishedEvent = nullptr,
                    bool waitForOutputsOnProcessingStream = true,
                    BatchSubmissionTiming* submitTiming = nullptr,
                    std::optional<uint32_t> outputSlotIndex = std::nullopt);
    Event sendBatch(std::map<std::string, Tensor> batchInputs,
                    const std::map<std::string, Event>& inputReadyEvents,
                    std::map<std::string, Tensor> &batchOutputs,
                    std::map<std::string, Event> &outputReadyEvents,
                    bool isInferenceOnly,
                    Event* reusableProcessingFinishedEvent = nullptr,
                    bool waitForOutputsOnProcessingStream = true,
                    BatchSubmissionTiming* submitTiming = nullptr,
                    std::optional<uint32_t> outputSlotIndex = std::nullopt);

    Event sendBatch(const Batch& batchInputs,
                    std::map<std::string, Tensor> &batchOutputs,
                    std::map<std::string, Event> &outputReadyEvents,
                    bool isInferenceOnly,
                    Event* reusableProcessingFinishedEvent = nullptr,
                    bool waitForOutputsOnProcessingStream = true,
                    BatchSubmissionTiming* submitTiming = nullptr,
                    std::optional<uint32_t> outputSlotIndex = std::nullopt);

    Event sendPhysicalBatch(std::map<std::string, Tensor> batchInputs,
                            const std::map<std::string, Event>& inputReadyEvents,
                            std::map<std::string, Tensor> &batchOutputs,
                            std::map<std::string, Event> &outputReadyEvents,
                            bool isInferenceOnly,
                            uint32_t batchSize,
                            Event* reusableProcessingFinishedEvent = nullptr,
                            bool waitForOutputsOnProcessingStream = true,
                            BatchSubmissionTiming* submitTiming = nullptr,
                            std::optional<uint32_t> outputSlotIndex = std::nullopt);

    void extendOutputWritableEvents(Event event, std::optional<uint32_t> outputSlotIndex = std::nullopt);

    void clear();

    std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> inputsShared;
    std::vector<std::shared_ptr<ThorImplementation::NetworkOutput>> outputsShared;
    std::vector<std::shared_ptr<ThorImplementation::TrainableLayer>> trainableLayersShared;
    std::vector<std::shared_ptr<ThorImplementation::Layer>> otherLayersShared;
    std::vector<Event> initializationDoneEvents;
    std::map<Thor::Tensor, std::shared_ptr<ThorImplementation::Layer>> apiTensorToPhysicalDrivingLayerShared;
    std::map<uint64_t, std::shared_ptr<ThorImplementation::Layer>> apiLayerToPhysicalLayerShared;
    std::map<std::shared_ptr<ThorImplementation::Layer>, uint64_t, StampedNetwork::LayerComparatorShared> physicalLayerToApiLayerShared;
    std::map<Thor::Tensor, std::shared_ptr<Thor::Layer>> apiTensorToApiDrivingLayerShared;
    std::map<std::string, std::shared_ptr<ThorImplementation::NetworkInput>> inputNamedShared;
    std::map<std::string, RaggedInputBinding> raggedInputNamedShared;
    std::map<std::string, std::shared_ptr<ThorImplementation::NetworkOutput>> outputNamedShared;

    // std::map<uint64_t, std::shared_ptr<ThorImplementation::Parameterizable>> apiParameterizableToPhysicalParameterizable;
    // FIXME: get rid of raw pointers
    // For performance, store and use the raw pointers
    std::vector<ThorImplementation::NetworkInput *> inputs;
    std::vector<ThorImplementation::NetworkOutput *> outputs;
    std::vector<ThorImplementation::TrainableLayer *> trainableLayers;
    std::vector<ThorImplementation::Layer *> otherLayers;
    std::map<Thor::Tensor, ThorImplementation::Layer *> apiTensorToPhysicalDrivingLayer;
    std::map<uint64_t, ThorImplementation::Layer *> apiLayerToPhysicalLayer;
    std::map<ThorImplementation::Layer *, uint64_t, StampedNetwork::LayerComparator> physicalLayerToApiLayer;
    std::map<Thor::Tensor, Thor::Layer *> apiTensorToApiDrivingLayer;
    std::map<std::string, ThorImplementation::NetworkInput *> inputNamed;
    std::map<std::string, RaggedInputBinding> raggedInputNamed;
    std::map<std::string, ThorImplementation::NetworkOutput *> outputNamed;

    uint32_t gpuNum;

    uint64_t bytesRequired;
    uint64_t batchSize;

    uint64_t floatingPointOperationsPerExampleForward;
    uint64_t floatingPointOperationsPerExampleBackward;

    friend class Thor::Network;
    friend class Thor::PlacedNetwork;
    friend class Thor::LocalExecutor;
};

}  // namespace ThorImplementation
