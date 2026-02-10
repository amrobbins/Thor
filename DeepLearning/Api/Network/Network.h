#pragma once

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

#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"

#include <assert.h>
#include <deque>
#include <filesystem>
#include <set>
#include <utility>
#include <vector>

#include "omp.h"

// FIXME: at some point, considering the desire to sorta parity with TF, each input connection to say a FC layer creates a new layer
//        just they share weights. That way feedback back into an FC layer does not create a cycle and topological sort
//        can still be used to stamp the graph
//        - or perhaps -
//        update the api to provide an output tensor each time an input tensor is given very much like tensorflow, then
//        ensure the topological sort considers the flow of tensors and not layers - so the output of an FC could drive the input
//        of the same FC no problem. In fact, I am not sure if I tried that today would the network report a cycle or be fine with it.
//
//        Actually, this is the doc from network:
//         * A deadlock cycle occurs when a layer that requires all of its input to arrive before it drives its output
//         * is connected in a way where there is a path from its output to its input.
//        So I think I just remembered wrong. I think regular cycles are currently ok, so it is just a question of API:
//        1 input tensor yields 1 output tensor. Usually (always?) they will need to be the same size.

#include "DeepLearning/Api/Network/StampedNetwork.h"

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

    Network(std::string networkName) : networkName(networkName), frozen(false) {}
    virtual ~Network();

    virtual std::string statusCodeToString(int statusCode);

    virtual StatusCode connect(bool inferenceOnly);
    virtual StatusCode place(uint32_t batchSize,
                             std::vector<Event> &initDoneEvents,
                             bool inferenceOnly = false,
                             std::vector<int32_t> forcedDevices = std::vector<int32_t>(),
                             uint32_t forcedNumStampsPerGpu = 0);
    //    virtual std::vector<ThorImplementation::StampedNetwork> getStampedNetworks() { return stampedNetworks; }
    uint64_t getNumStamps() { return stampedNetworks.size(); }
    ThorImplementation::StampedNetwork &getStampedNetwork(uint64_t i) { return stampedNetworks[i]; }

    virtual std::string getNetworkName() { return networkName; }

    virtual void save(const std::string &directory, bool overwrite, bool saveOptimizerState);
    virtual void load(const std::string &directory);

    std::shared_ptr<Optimizer> getOptimizer();
    void attachOptimizerToLayers(bool replaceIfExisting);

    Tensor getApiTensorByOriginalId(uint64_t originalId);
    uint32_t getNumTrainableLayers() { return allTrainableLayersInNetwork.size(); }
    std::shared_ptr<TrainableWeightsBiasesLayer> getTrainableLayer(uint32_t i) { return allTrainableLayersInNetwork[i]; }

    uint32_t getNumLayers() { return allLayersInNetwork.size(); }
    std::shared_ptr<Layer> getLayer(uint32_t i) { return allLayersInNetworkList[i]; }

   private:
    static const bool DEBUG_STAMP = false;

    struct LayerComparator {
        bool operator()(const std::shared_ptr<Layer> &lhs, const std::shared_ptr<Layer> &rhs) const { return *lhs < *rhs; }
    };

   protected:
    std::set<std::shared_ptr<Layer>, Network::LayerComparator> allLayersInNetwork;
    std::vector<std::shared_ptr<Layer>> allLayersInNetworkList;
    std::vector<std::shared_ptr<TrainableWeightsBiasesLayer>> allTrainableLayersInNetwork;
    std::vector<std::shared_ptr<Layer>> network;
    std::vector<std::pair<Optional<Tensor>, std::shared_ptr<Layer>>> orderedNetwork;

    std::set<Tensor> allTensors;
    std::map<uint64_t, Tensor> apiTensorByOriginalId;
    std::map<Tensor, std::vector<std::shared_ptr<Layer>>> apiTensorToApiLoadingLayers;
    std::map<Tensor, std::shared_ptr<Layer>> apiTensorToApiDrivingLayer;
    std::map<std::shared_ptr<Layer>, std::vector<Tensor>, Network::LayerComparator> apiLayerToApiOutputTensors;
    std::map<std::shared_ptr<Layer>, std::vector<Tensor>, Network::LayerComparator> apiLayerToApiInputTensors;

    std::shared_ptr<Optimizer> optimizer;

    // FIXME: this should be an unordered_map of gpu -> vector of stamps
    std::vector<ThorImplementation::StampedNetwork> stampedNetworks;

    uint64_t computeFirstInstanceMemRequirements(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement);
    uint64_t computeNonFirstInstanceMemRequirements(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement);

    uint64_t firstInstanceBytes;
    uint64_t nonFirstInstanceBytes;

    virtual StatusCode createDagAndFreeze();
    virtual void preOptimize(uint32_t gpuNum, uint32_t batchSize);
    virtual StatusCode stampNetwork(uint32_t gpuNum, std::vector<Event> &initDoneEvents, uint32_t batchSize);

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

    bool terminatesWithoutHitting(Tensor tensor, std::shared_ptr<Layer> layer);

    bool frozen;

    std::shared_ptr<thor_file::TarWriter> archiveWriter = nullptr;
    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;

    class GpuOutOfMemoryError {};

    friend void Layer::addToNetwork(Network *network);
    friend void Optimizer::addToNetwork(Network *network);
    friend class Executor;
};

}  // namespace Thor
