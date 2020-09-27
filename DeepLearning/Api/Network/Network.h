#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/Layers/Utility/TypeConversion.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

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

class Executor;

class Network {
   public:
    enum class StatusCode { SUCCESS = 0, FLOATING_INPUT, DANGLING_OUTPUT, GPU_OUT_OF_MEMORY };

    Network() : frozen(false) {}
    virtual ~Network() {}

    StatusCode create();

   protected:
    set<shared_ptr<Layer>> network;
    set<uint32_t> allTensors;
    map<uint32_t, vector<uint32_t>> tensorToLoadingLayers;
    map<uint32_t, uint32_t> tensorToDrivingLayer;
    // Api layerId to implementation layer:
    map<uint32_t, ThorImplementation::Layer *> inputLayer;
    map<uint32_t, ThorImplementation::Layer *> outputLayer;
    map<uint32_t, ThorImplementation::Layer *> outputLossLayer;

    void computeMemRequirements(uint64_t &fixedBytes, uint64_t &perBatchItemBytes);
    StatusCode stampNetwork(uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork);

    virtual StatusCode evaluateGraph();
    virtual StatusCode checkForFloatingInputs();
    virtual StatusCode checkForDanglingOutputs();

    virtual void stampNetworkInput(const Thor::NetworkInput *networkInput,
                                   uint32_t gpuNum,
                                   uint32_t batchSize,
                                   ThorImplementation::StampedNetwork &stampedNetwork);
    virtual void stampNetworkOutput(const Thor::NetworkOutput *networkOutput,
                                    uint32_t gpuNum,
                                    uint32_t batchSize,
                                    ThorImplementation::StampedNetwork &stampedNetwork);
    virtual void stampLoss(const Thor::Loss *loss, uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork);
    virtual void stampMultiConnectionLayer(const Thor::MultiConnectionLayer *multiConnectionLayer,
                                           uint32_t gpuNum,
                                           uint32_t batchSize,
                                           ThorImplementation::StampedNetwork &stampedNetwork);
    virtual void stampBaseLayer(const Thor::Layer *layer,
                                uint32_t gpuNum,
                                uint32_t batchSize,
                                ThorImplementation::StampedNetwork &stampedNetwork);

    void createBatchDimensions(vector<uint64_t> &batchDimensions, vector<uint64_t> tensorDimensions, uint32_t batchSize);

    void addSingleLayerToNetwork(const Layer *layer) {
        assert(!layer->isMultiLayer());
        network.insert(layer->clone());
    }

    // Take a snapshot of layer and add the snapshot to the network
    void addToNetwork(Layer *layer) {
        frozen = false;
        if (layer->isMultiLayer())
            layer->convertToSingleLayersAndAddToNetwork();
        else
            addSingleLayerToNetwork(layer);
    }

    class GpuOutOfMemoryError {};

    friend void Layer::addToNetwork(Network *network);
    friend class Executor;

   protected:
    bool frozen;
};

}  // namespace Thor
