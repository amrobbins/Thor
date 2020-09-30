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
#include <deque>
#include <set>
#include <utility>
#include <vector>

using std::deque;
using std::make_pair;
using std::pair;
using std::set;
using std::vector;

namespace ThorImplementation {

class StampedNetwork {
   public:
    vector<ThorImplementation::NetworkInput *> inputs;
    vector<ThorImplementation::NetworkOutput *> outputs;
    vector<ThorImplementation::TrainableWeightsBiasesLayer *> trainableLayers;
    vector<ThorImplementation::Layer *> otherLayers;

    map<Thor::Tensor, ThorImplementation::Layer *> apiTensorToPhysicalDrivingLayer;
    map<const Thor::Layer *, ThorImplementation::Layer *> apiLayerToPhysicalLayer;
    map<Thor::Tensor, Thor::Layer *> apiTensorToApiDrivingLayer;

    void clear() {
        for (uint32_t i = 0; i < inputs.size(); ++i) {
            inputs[i]->parentCleanup();
            inputs[i]->cleanup();
            delete inputs[i];
        }
        inputs.clear();

        for (uint32_t i = 0; i < outputs.size(); ++i) {
            outputs[i]->parentCleanup();
            outputs[i]->cleanup();
            delete outputs[i];
        }
        outputs.clear();

        for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
            trainableLayers[i]->parentCleanup();
            trainableLayers[i]->cleanup();
            delete trainableLayers[i];
        }
        trainableLayers.clear();

        for (uint32_t i = 0; i < otherLayers.size(); ++i) {
            otherLayers[i]->parentCleanup();
            otherLayers[i]->cleanup();
            delete otherLayers[i];
        }
        otherLayers.clear();

        apiTensorToPhysicalDrivingLayer.clear();
        apiLayerToPhysicalLayer.clear();
        apiTensorToApiDrivingLayer.clear();
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

    // Figure this out. Want to add a subnetwork into the larger network, but how to connect subnetwork inputs and outputs?
    void addSubnetwork(Network subnetwork) { assert(false); }

   protected:
    set<shared_ptr<Layer>> network;
    vector<pair<Optional<Tensor>, Layer *>> orderedNetwork;

    set<Tensor> allTensors;
    map<Tensor, vector<Layer *>> apiTensorToApiLoadingLayers;
    map<Tensor, Layer *> apiTensorToApiDrivingLayer;
    // Api layerId to implementation layer:
    // map<uint64_t, ThorImplementation::Layer *> inputLayer;
    // map<uint64_t, ThorImplementation::Layer *> outputLayer;
    // map<uint64_t, ThorImplementation::Layer *> outputLossLayer;

    void computeFirstInstanceMemRequirements(uint64_t &fixedBytes, uint64_t &perBatchItemBytes);
    void computeNonFirstInstanceMemRequirements(uint64_t &fixedBytes, uint64_t &perBatchItemBytes);
    StatusCode stampNetwork(uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork);

    uint64_t firstInstanceFixedBytes;
    uint64_t firstInstancePerBatchItemBytes;
    uint64_t nonFirstInstanceFixedBytes;
    uint64_t nonFirstInstancePerBatchItemBytes;

    virtual StatusCode evaluateGraph();
    virtual StatusCode checkForFloatingInputs();
    virtual StatusCode checkForDanglingOutputs();
    virtual void topologicalSort();

    virtual void stampNetworkInput(const Thor::NetworkInput *networkInput,
                                   uint32_t gpuNum,
                                   uint32_t batchSize,
                                   ThorImplementation::StampedNetwork &stampedNetwork);
    virtual void stampNetworkOutput(Tensor inputTensor,
                                    const Thor::NetworkOutput *networkOutput,
                                    uint32_t gpuNum,
                                    uint32_t batchSize,
                                    ThorImplementation::StampedNetwork &stampedNetwork);
    virtual void stampLayer(Tensor inputTensor,
                            const Thor::Layer *layer,
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
