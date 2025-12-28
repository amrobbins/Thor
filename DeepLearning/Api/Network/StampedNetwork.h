#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <assert.h>
#include <vector>

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
    // std::vector<std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer>> getTrainableLayers() { return trainableLayersShared; }
    uint64_t getNumTrainableLayers() { return trainableLayersShared.size(); }
    std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> &getTrainableLayer(uint64_t i) { return trainableLayersShared[i]; }
    std::vector<std::shared_ptr<ThorImplementation::Layer>> getOtherLayers() { return otherLayersShared; }
    uint64_t getBytesRequired() {
        // FIXME
        assert(false);
    }

    // For testing:
    // FIXME: Create a test friend?
    std::map<uint64_t, std::shared_ptr<ThorImplementation::Layer>> getApiLayerToPhysicalLayer() { return apiLayerToPhysicalLayerShared; }
    std::shared_ptr<ThorImplementation::Layer> getPhysicalLayerFromApiLayer(uint64_t apiLayerId) {
        return apiLayerToPhysicalLayerShared[apiLayerId];
    }

   protected:
    void initialize(bool initializeWeights, bool copyWeightsFromOtherStamp, StampedNetwork *otherStamp = nullptr);

    // Note that all processing is finished at the end of any input stream of the stamp.
    // Note *input* stream - this is not the case for the loader streams
    Event sendBatch(std::map<std::string, Tensor> batchInputs,
                    std::map<std::string, Tensor> &batchOutputs,
                    std::map<std::string, Event> &outputReadyEvents,
                    bool isInferenceOnly);

    void clear();

    std::vector<std::shared_ptr<ThorImplementation::NetworkInput>> inputsShared;
    std::vector<std::shared_ptr<ThorImplementation::NetworkOutput>> outputsShared;
    std::vector<std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer>> trainableLayersShared;
    std::vector<std::shared_ptr<ThorImplementation::Layer>> otherLayersShared;
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
