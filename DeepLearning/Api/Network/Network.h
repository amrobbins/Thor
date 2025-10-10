#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Metrics/Metric.h"
#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Metric.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/Layers/Utility/TypeConversion.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <assert.h>
#include <deque>
#include <set>
#include <utility>
#include <vector>

#include "omp.h"

namespace Thor {
class Network;
class LocalExecutor;
}  // namespace Thor

namespace ThorImplementation {

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
    std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> getInputs() { return inputsShared; }
    std::vector<std::shared_ptr<ThorImplementation::NetworkOutput>> getOutputs() { return outputsShared; }
    std::vector<std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer>> getTrainableLayers() { return trainableLayersShared; }
    std::vector<std::shared_ptr<ThorImplementation::Layer>> getOtherLayers() { return otherLayersShared; }
    uint64_t getBytesRequired() {
        // FIXME
        assert(false);
    }

    // For testing:
    std::map<uint64_t, std::shared_ptr<ThorImplementation::Layer>> getApiLayerToPhysicalLayer() { return apiLayerToPhysicalLayerShared; }
    std::shared_ptr<ThorImplementation::Layer> getPhysicalLayerFromApiLayer(uint64_t apiLayerId) {
        return apiLayerToPhysicalLayerShared[apiLayerId];
    }

   protected:
    void initialize(bool initializeWeights, bool copyWeightsFromOtherStamp, StampedNetwork *otherStamp = nullptr) {
        for (auto it = initializersShared.begin(); it != initializersShared.end(); ++it) {
            initializers.push_back(it->get());
        }

        // First, ensure the shared pointers and raw pointers match
        for (auto it = inputsShared.begin(); it != inputsShared.end(); ++it)
            assert(count(inputs, it->get()) == 1);
        for (auto it = outputsShared.begin(); it != outputsShared.end(); ++it)
            assert(count(outputs, it->get()) == 1);
        for (auto it = trainableLayersShared.begin(); it != trainableLayersShared.end(); ++it)
            assert(count(trainableLayers, it->get()) == 1);
        for (auto it = otherLayersShared.begin(); it != otherLayersShared.end(); ++it)
            assert(count(otherLayers, it->get()) == 1);
        for (auto it = initializersShared.begin(); it != initializersShared.end(); ++it)
            assert(count(initializers, it->get()) == 1);
        for (auto it = apiTensorToPhysicalDrivingLayerShared.begin(); it != apiTensorToPhysicalDrivingLayerShared.end(); ++it) {
            assert(apiTensorToPhysicalDrivingLayer.count(it->first) == 1);
            assert(apiTensorToPhysicalDrivingLayer[it->first] == it->second.get());
        }
        for (auto it = apiLayerToPhysicalLayerShared.begin(); it != apiLayerToPhysicalLayerShared.end(); ++it) {
            assert(apiLayerToPhysicalLayer.count(it->first) == 1);
            assert(apiLayerToPhysicalLayer[it->first] == it->second.get());
        }
        for (auto it = physicalLayerToApiLayerShared.begin(); it != physicalLayerToApiLayerShared.end(); ++it) {
            assert(physicalLayerToApiLayer.count(it->first.get()) == 1);
            assert(physicalLayerToApiLayer[it->first.get()] == it->second);
        }
        for (auto it = apiTensorToApiDrivingLayerShared.begin(); it != apiTensorToApiDrivingLayerShared.end(); ++it) {
            assert(apiTensorToApiDrivingLayer.count(it->first) == 1);
            assert(apiTensorToApiDrivingLayer[it->first] == it->second.get());
        }
        for (auto it = inputNamedShared.begin(); it != inputNamedShared.end(); ++it) {
            assert(inputNamed.count(it->first) == 1);
            assert(inputNamed[it->first] == it->second.get());
        }
        for (auto it = outputNamedShared.begin(); it != outputNamedShared.end(); ++it) {
            assert(outputNamed.count(it->first) == 1);
            assert(outputNamed[it->first] == it->second.get());
        }

        // Now that checks have been run, initialize the stamp
        assert(!(initializeWeights && copyWeightsFromOtherStamp));
        if (initializeWeights) {
            // Weights are shared by all stamps so weights are only initialized once
            for (uint32_t i = 0; i < initializers.size(); ++i)
                initializers[i]->initialize();
        } else if (copyWeightsFromOtherStamp) {
            // Every GPU needs its a copy of the weights, if they have already been initialized in a weights memory, then copy that memory
            // to the target GPU.
            assert(otherStamp != nullptr);
            // FIXME use trainable layer stamped ids to copy weights and when present biases from other stamp to this stamp
            std::unordered_map<uint64_t, ThorImplementation::TrainableWeightsBiasesLayer *> trainableLayerMap;
            for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
                trainableLayerMap[trainableLayers[i]->getStampedId()] = trainableLayers[i];
            }
            std::vector<Stream> streams;
            Stream stream;
            for (uint32_t i = 0; i < otherStamp->trainableLayers.size(); ++i) {
                uint32_t stampedId = otherStamp->trainableLayers[i]->getStampedId();
                if (i == 0) {
                    streams.push_back(trainableLayerMap[stampedId]->getStreams()[0]);
                }
                Tensor uninitializedWeights = trainableLayerMap[stampedId]->getWeights();
                Optional<Tensor> uninitializedBiases = trainableLayerMap[stampedId]->getBiases();
                ThorImplementation::TrainableWeightsBiasesLayer *initializedLayer = otherStamp->trainableLayers[i];
                Tensor initializedWeights = initializedLayer->getWeights();
                Optional<Tensor> initializedBiases = initializedLayer->getBiases();
                uninitializedWeights.copyFromAsync(initializedWeights, streams.back());
                if (initializedBiases.isPresent()) {
                    assert(uninitializedBiases.isPresent());
                    uninitializedBiases.get().copyFromAsync(initializedBiases.get(), stream);
                }
            }
            for (uint32_t i = 0; i < streams.size(); ++i) {
                streams[i].synchronize();
            }
        }

        for (uint32_t i = 0; i < inputs.size(); ++i) {
            inputs[i]->parentInitialize();
            inputs[i]->initialize();
        }
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            outputs[i]->parentInitialize();
            outputs[i]->initialize();
        }
        for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
            trainableLayers[i]->parentInitialize();
            trainableLayers[i]->initialize();
        }
        for (uint32_t i = 0; i < otherLayers.size(); ++i) {
            otherLayers[i]->parentInitialize();
            otherLayers[i]->initialize();
        }
    }

    // Note that all processing is finished at the end of any input stream of the stamp.
    // Note *input* stream - this is not the case for the loader streams
    Event sendBatch(std::map<std::string, Tensor> batchInputs,
                    std::map<std::string, Tensor> &batchOutputs,
                    std::map<std::string, Event> &outputReadyEvents,
                    bool isInferenceOnly) {
        assert(batchInputs.size() == inputs.size());

        for (uint32_t i = 0; i < inputs.size(); ++i) {
            auto it = batchInputs.find(inputs[i]->getName());
            assert(it != batchInputs.end());
            Tensor inputTensor = it->second;
            inputs[i]->forward(inputTensor, isInferenceOnly);
        }

        // The stream from input 0 waits for all outputs to be ready
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            batchOutputs[outputs[i]->getName()] = outputs[i]->getFeatureOutput();
            Event outputReadyEvent = outputs[i]->getStream().putEvent();
            outputReadyEvents[outputs[i]->getName()] = outputReadyEvent;
            inputs[0]->getStream().waitEvent(outputReadyEvent);
        }

        // Processing is finished when the stream from input 0 is ready
        Event processingFinishedEvent = inputs[0]->getStream().putEvent(true, true);

        // The streams from all other inputs wait for the stream from input 0 to be ready
        for (uint i = 1; i < inputs.size(); ++i) {
            inputs[i]->getStream().waitEvent(processingFinishedEvent);
        }

        return processingFinishedEvent;
    }

    void clear() {
        for (uint32_t i = 0; i < inputs.size(); ++i) {
            inputs[i]->parentCleanup();
            inputs[i]->cleanup();
        }
        inputs.clear();

        for (uint32_t i = 0; i < outputs.size(); ++i) {
            outputs[i]->parentCleanup();
            outputs[i]->cleanup();
        }
        outputs.clear();

        for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
            trainableLayers[i]->parentCleanup();
            trainableLayers[i]->cleanup();
        }
        trainableLayers.clear();

        for (uint32_t i = 0; i < otherLayers.size(); ++i) {
            otherLayers[i]->parentCleanup();
            otherLayers[i]->cleanup();
        }
        otherLayers.clear();

        initializers.clear();
        apiTensorToPhysicalDrivingLayer.clear();
        apiLayerToPhysicalLayer.clear();
        physicalLayerToApiLayer.clear();
        apiTensorToApiDrivingLayer.clear();
        inputNamed.clear();
        outputNamed.clear();

        inputsShared.clear();
        outputsShared.clear();
        trainableLayersShared.clear();
        otherLayersShared.clear();
        initializersShared.clear();
        apiTensorToPhysicalDrivingLayerShared.clear();
        apiLayerToPhysicalLayerShared.clear();
        physicalLayerToApiLayerShared.clear();
        apiTensorToApiDrivingLayerShared.clear();
        inputNamedShared.clear();
        outputNamedShared.clear();
    }

    std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> inputsShared;
    std::vector<std::shared_ptr<ThorImplementation::NetworkOutput>> outputsShared;
    std::vector<std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer>> trainableLayersShared;
    std::vector<std::shared_ptr<ThorImplementation::Layer>> otherLayersShared;
    std::vector<std::shared_ptr<Thor::Initializer>> initializersShared;
    std::map<Thor::Tensor, std::shared_ptr<ThorImplementation::Layer>> apiTensorToPhysicalDrivingLayerShared;
    std::map<uint64_t, std::shared_ptr<ThorImplementation::Layer>> apiLayerToPhysicalLayerShared;
    std::map<std::shared_ptr<ThorImplementation::Layer>, uint64_t, StampedNetwork::LayerComparatorShared> physicalLayerToApiLayerShared;
    std::map<Thor::Tensor, std::shared_ptr<Thor::Layer>> apiTensorToApiDrivingLayerShared;
    std::map<std::string, std::shared_ptr<ThorImplementation::NetworkInput>> inputNamedShared;
    std::map<std::string, std::shared_ptr<ThorImplementation::NetworkOutput>> outputNamedShared;

    // For performance, store and use the raw pointers
    std::vector<ThorImplementation::NetworkInput *> inputs;
    std::vector<ThorImplementation::NetworkOutput *> outputs;
    std::vector<ThorImplementation::TrainableWeightsBiasesLayer *> trainableLayers;
    std::vector<ThorImplementation::Layer *> otherLayers;
    std::vector<Thor::Initializer *> initializers;
    std::map<Thor::Tensor, ThorImplementation::Layer *> apiTensorToPhysicalDrivingLayer;
    std::map<uint64_t, ThorImplementation::Layer *> apiLayerToPhysicalLayer;
    std::map<ThorImplementation::Layer *, uint64_t, StampedNetwork::LayerComparator> physicalLayerToApiLayer;
    std::map<Thor::Tensor, Thor::Layer *> apiTensorToApiDrivingLayer;
    std::map<std::string, ThorImplementation::NetworkInput *> inputNamed;
    std::map<std::string, ThorImplementation::NetworkOutput *> outputNamed;

    uint32_t gpuNum;

    uint64_t bytesRequired;
    uint64_t batchSize;

    uint64_t floatingPointOperationsPerExampleForward;
    uint64_t floatingPointOperationsPerExampleBackward;

    friend class Thor::Network;
    friend class Thor::LocalExecutor;
};

}  // namespace ThorImplementation

namespace Thor {

class Executor;

class Network {
   public:
    enum class StatusCode {
        SUCCESS = 0,
        FLOATING_INPUT,
        DANGLING_OUTPUT,
        GPU_OUT_OF_MEMORY,
        DUPLICATE_NAMED_NETWORK_INPUT,
        DUPLICATE_NAMED_NETWORK_OUTPUT,
        DEADLOCK_CYCLE
    };

    const static int32_t CPU = -1;

    Network() : frozen(false) {}
    virtual ~Network();

    virtual std::string statusCodeToString(int statusCode);

    virtual StatusCode place(uint32_t batchSize,
                             std::vector<int32_t> forcedDevices = std::vector<int32_t>(),
                             uint32_t forcedNumStampsPerGpu = 0);
    virtual std::vector<ThorImplementation::StampedNetwork> getStampedNetworks() { return stampedNetworks; }

    virtual void setNetworkName(std::string networkName) { this->networkName = networkName; }
    virtual std::string getNetworkName() { return networkName; }

    // FIXME: implement:
    virtual void save(std::string filename, bool keep_optimizer);
    virtual void load(std::string filename);
    virtual void save_as_keras(std::string filename, bool keep_optimizer);
    virtual void load_from_keras(std::string filename);

    std::shared_ptr<Optimizer> getOptimizer();

   private:
    static const bool DEBUG_STAMP = false;

    struct LayerComparator {
        bool operator()(const std::shared_ptr<Layer> &lhs, const std::shared_ptr<Layer> &rhs) const { return *lhs < *rhs; }
    };

   protected:
    std::set<std::shared_ptr<Layer>, Network::LayerComparator> network;
    std::vector<std::pair<Optional<Tensor>, std::shared_ptr<Layer>>> orderedNetwork;

    std::set<Tensor> allTensors;
    std::map<Tensor, std::vector<std::shared_ptr<Layer>>> apiTensorToApiLoadingLayers;
    std::map<Tensor, std::shared_ptr<Layer>> apiTensorToApiDrivingLayer;
    std::map<std::shared_ptr<Layer>, std::vector<Tensor>, Network::LayerComparator> apiLayerToApiOutputTensors;
    std::map<std::shared_ptr<Layer>, std::vector<Tensor>, Network::LayerComparator> apiLayerToApiInputTensors;

    std::vector<std::shared_ptr<Initializer>> initializers;
    std::shared_ptr<Optimizer> optimizer;

    std::vector<ThorImplementation::StampedNetwork> stampedNetworks;

    uint64_t computeFirstInstanceMemRequirements(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement);
    uint64_t computeNonFirstInstanceMemRequirements(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement);

    uint64_t firstInstanceBytes;
    uint64_t nonFirstInstanceBytes;

    virtual StatusCode stampNetwork(uint32_t gpuNum, uint32_t batchSize);
    virtual StatusCode preOptimize(uint32_t gpuNum, uint32_t batchSize);

    virtual StatusCode evaluateGraph();
    virtual StatusCode checkForDuplicateInOutPortNames();
    virtual StatusCode checkForFloatingInputs();
    virtual StatusCode checkForDanglingOutputs();
    virtual StatusCode checkForDeadlockCycles();
    virtual void topologicalSort();

    virtual void stampNetworkInput(const std::shared_ptr<Thor::NetworkInput> networkInput,
                                   uint32_t gpuNum,
                                   uint32_t batchSize,
                                   ThorImplementation::StampedNetwork &stampedNetwork);
    virtual void stampNetworkOutput(Tensor inputTensor,
                                    const std::shared_ptr<Thor::NetworkOutput> networkOutput,
                                    uint32_t gpuNum,
                                    uint32_t batchSize,
                                    ThorImplementation::StampedNetwork &stampedNetwork);
    virtual void stampLayer(Tensor inputTensor,
                            const std::shared_ptr<Thor::Layer> layer,
                            uint32_t gpuNum,
                            uint32_t batchSize,
                            ThorImplementation::StampedNetwork &stampedNetwork);

    void createBatchDimensions(std::vector<uint64_t> &batchDimensions, std::vector<uint64_t> tensorDimensions, uint32_t batchSize);

    void addLayerToNetwork(const Layer *layer);

    // Take a snapshot of layer and add the snapshot to the network
    void addToNetwork(Layer *layer);
    void addToNetwork(Initializer *initializer);
    void addToNetwork(Optimizer *optimizer);

    std::string networkName;

    // void reorderStampedNetworkForTestability(StampedNetwork &stampedNetwork);
    // void reorderLayers(StampedNetwork &stampedNetwork, std::vector<Layer*> &layersToReoder, std::vector<Layer*> &destinationStorage);

    bool terminatesWithoutHitting(Tensor tensor, std::shared_ptr<Layer> layer);

    bool frozen;

    class GpuOutOfMemoryError {};

    friend void Layer::addToNetwork(Network *network);
    friend void Optimizer::addToNetwork(Network *network);
    friend class Executor;
};

}  // namespace Thor
