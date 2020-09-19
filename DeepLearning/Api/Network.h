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
    uint32_t stampNetwork(uint32_t gpuNum,
                          uint32_t batchSize,
                          vector<NetworkInput> &inputs,
                          vector<NetworkOutput> &outputs,
                          vector<TrainableWeightsBiasesLayerBase> &trainableLayers);

    virtual StatusCode evaluateGraph();
    virtual void checkForFloatingInputs();
    virtual void checkForDanglingOUtputs();

    virtual ThorImplementation::NetworkInput *stampNetworkInput(Thor::NetworkInput *networkInput, uint32_t gpuNum, uint32_t batchSize);
    virtual ThorImplementation::NetworkOutput *stampNetworkOutput(Thor::NetworkOutput *networkOutput, uint32_t gpuNum, uint32_t batchSize);
    virtual ThorImplementation::Loss *stampLoss(Thor::Loss *loss, uint32_t gpuNum, uint32_t batchSize);
    virtual ThorImplementation::TrainableWeightsBiasesLayer *stampTrainableWeightsBiasesLayer(
        Thor::TrainableWeightsBiasesLayerBase *trainableWeightsBiasesLayer, uint32_t gpuNum, uint32_t batchSize);
    virtual ThorImplementation::MultiConnectionLayer *stampMultiConnectionLayer(Thor::MultiConnectionLayer *multiConnectionLayer,
                                                                                uint32_t gpuNum,
                                                                                uint32_t batchSize);
    virtual ThorImplementation::Layer *stampBaseLayer(Thor::Layer *layer, uint32_t gpuNum, uint32_t batchSize);

    friend class ExecutorBase;
};

}  // namespace Thor
