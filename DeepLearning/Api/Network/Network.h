#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
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

    vector<shared_ptr<Thor::Initializer>> initializers;

    map<Thor::Tensor, ThorImplementation::Layer *> apiTensorToPhysicalDrivingLayer;
    map<uint64_t, ThorImplementation::Layer *> apiLayerToPhysicalLayer;
    map<ThorImplementation::Layer *, uint64_t> physicalLayerToApiLayer;
    map<Thor::Tensor, Thor::Layer *> apiTensorToApiDrivingLayer;

    map<string, ThorImplementation::NetworkInput *> inputNamed;
    map<string, ThorImplementation::NetworkOutput *> outputNamed;

    uint32_t gpuNum;

    uint64_t bytesRequired;
    uint64_t batchSize;

    void initialize() {
        for (uint32_t i = 0; i < initializers.size(); ++i)
            initializers[i]->initialize();

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
    Event sendBatch(map<string, Tensor> batchInputs, map<string, Tensor> &batchOutputs, map<string, Event> &outputReadyEvents) {
        assert(batchInputs.size() == inputs.size());

        for (uint32_t i = 0; i < inputs.size(); ++i) {
            auto it = batchInputs.find(inputs[i]->getName());
            assert(it != batchInputs.end());
            Tensor inputTensor = it->second;
            inputs[i]->forward(inputTensor);
        }

        // The stream from input 0 waits for all outputs to be ready
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            batchOutputs[outputs[i]->getName()] = outputs[i]->getFeatureOutput();
            Event outputReadyEvent = outputs[i]->getStream().putEvent();
            outputReadyEvents[outputs[i]->getName()] = outputReadyEvent;
            inputs[0]->getStream().waitEvent(outputReadyEvent);
        }

        // The stream from input 0 waits for all gradient updates to finish
        for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
            trainableLayers[i]->updateWeightsAndBiasesWithScaledGradient();
            assert(trainableLayers[i]->getGradientUpdateStream().isInitialized());
            Event gradientUpdateFinishedEvent = trainableLayers[i]->getGradientUpdateStream().putEvent();
            inputs[0]->getStream().waitEvent(gradientUpdateFinishedEvent);
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
    enum class StatusCode {
        SUCCESS = 0,
        FLOATING_INPUT,
        DANGLING_OUTPUT,
        GPU_OUT_OF_MEMORY,
        DUPLICATE_NAMED_NETWORK_INPUT,
        DUPLICATE_NAMED_NETWORK_OUTPUT,
        DEADLOCK_CYCLE
    };

    Network() : frozen(false) {}
    virtual ~Network() {}

    virtual StatusCode preOptimize(uint32_t gpuNum, uint32_t batchSize);
    virtual StatusCode stampNetwork(uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork);
    virtual vector<ThorImplementation::StampedNetwork> getStampedNetworks() { return stampedNetworks; }

    virtual void setNetworkName(string networkName) { this->networkName = networkName; }
    virtual string getNetworkName() { return networkName; }

   protected:
    set<shared_ptr<Layer>> network;
    vector<pair<Optional<Tensor>, Layer *>> orderedNetwork;

    set<Tensor> allTensors;
    map<Tensor, vector<Layer *>> apiTensorToApiLoadingLayers;
    map<Tensor, Layer *> apiTensorToApiDrivingLayer;
    map<Layer *, vector<Tensor>> apiLayerToApiOutputTensors;
    map<Layer *, vector<Tensor>> apiLayerToApiInputTensors;

    vector<shared_ptr<Initializer>> initializers;

    vector<ThorImplementation::StampedNetwork> stampedNetworks;

    uint64_t computeFirstInstanceMemRequirements(uint32_t batchSize, TensorPlacement tensorPlacement);
    uint64_t computeNonFirstInstanceMemRequirements(uint32_t batchSize, TensorPlacement tensorPlacement);

    uint64_t firstInstanceBytes;
    uint64_t nonFirstInstanceBytes;

    virtual StatusCode evaluateGraph();
    virtual StatusCode checkForDuplicateInOutPortNames();
    virtual StatusCode checkForFloatingInputs();
    virtual StatusCode checkForDanglingOutputs();
    virtual StatusCode checkForDeadlockCycles();
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

    void addToNetwork(Initializer *initializer) { initializers.push_back(initializer->clone()); }

    std::string networkName;

    // void reorderStampedNetworkForTestability(StampedNetwork &stampedNetwork);
    // void reorderLayers(StampedNetwork &stampedNetwork, vector<Layer*> &layersToReoder, vector<Layer*> &destinationStorage);

    bool terminatesWithoutHitting(Tensor tensor, Layer *layer);

    bool frozen;

    class GpuOutOfMemoryError {};

    friend void Layer::addToNetwork(Network *network);
    friend class Executor;
};

}  // namespace Thor
