#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

#include <assert.h>
#include <set>
#include <vector>

using std::set;
using std::vector;

namespace ThorImplementation {

class StampedNetwork {
   public:
    vector<ThorImplementation::NetworkInput *> inputs;
    vector<ThorImplementation::NetworkOutput *> outputs;
    vector<ThorImplementation::TrainableWeightsBiasesLayer *> trainableLayers;
    vector<ThorImplementation::Layer *> otherLayers;

    void clear() {
        for (uint32_t i = 0; i < inputs.size(); ++i)
            delete inputs[i];
        inputs.clear();

        for (uint32_t i = 0; i < outputs.size(); ++i)
            delete outputs[i];
        outputs.clear();

        for (uint32_t i = 0; i < trainableLayers.size(); ++i)
            delete trainableLayers[i];
        trainableLayers.clear();

        for (uint32_t i = 0; i < otherLayers.size(); ++i)
            delete otherLayers[i];
        otherLayers.clear();
    }
};

}  // namespace ThorImplementation

namespace Thor {

class ExecutorBase;

class Network {
   public:
    Network() {}
    virtual ~Network() {}

    void addToNetwork(Layer layer) {
        assert(network.count(layer) == 0);
        network.insert(layer);
    }

    enum class StatusCode { SUCCESS = 0, FLOATING_INPUT, DANGLING_OUTPUT, GPU_OUT_OF_MEMORY };

   protected:
    set<Layer> network;
    set<uint32_t> allTensors;
    map<uint32_t, vector<uint32_t>> tensorToLoadingLayers;
    map<uint32_t, uint32_t> tensorToDrivingLayer;

    uint32_t getBatchSize();
    void computeMemRequirements(uint64_t &fixedBytes, uint64_t &perBatchItemBytes);
    StatusCode stampNetwork(uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork);

    virtual StatusCode evaluateGraph();
    virtual StatusCode checkForFloatingInputs();
    virtual StatusCode checkForDanglingOutputs();

    virtual ThorImplementation::NetworkInput *stampNetworkInput(const Thor::NetworkInput *networkInput,
                                                                uint32_t gpuNum,
                                                                uint32_t batchSize);
    virtual ThorImplementation::NetworkOutput *stampNetworkOutput(const Thor::NetworkOutput *networkOutput,
                                                                  uint32_t gpuNum,
                                                                  uint32_t batchSize);
    virtual ThorImplementation::Loss *stampLoss(const Thor::LossBase *loss, uint32_t gpuNum, uint32_t batchSize);
    virtual ThorImplementation::TrainableWeightsBiasesLayer *stampTrainableWeightsBiasesLayer(
        const Thor::TrainableWeightsBiasesLayerBase *trainableWeightsBiasesLayer, uint32_t gpuNum, uint32_t batchSize);
    virtual ThorImplementation::MultiConnectionLayer *stampMultiConnectionLayer(const Thor::MultiConnectionLayerBase *multiConnectionLayer,
                                                                                uint32_t gpuNum,
                                                                                uint32_t batchSize);
    virtual ThorImplementation::Layer *stampBaseLayer(const Thor::LayerBase *layer, uint32_t gpuNum, uint32_t batchSize);

    class GpuOutOfMemoryError {};

    friend class ExecutorBase;
};

}  // namespace Thor
