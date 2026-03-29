#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <assert.h>
#include <vector>

#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"

namespace Thor {
class Network;
class PlacedNetwork;
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

    uint64_t getNumTrainableLayers() { return trainableLayersShared.size(); }
    std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> &getTrainableLayer(uint64_t i) { return trainableLayersShared[i]; }
    std::vector<std::shared_ptr<ThorImplementation::Layer>> getOtherLayers() { return otherLayersShared; }

    std::shared_ptr<ThorImplementation::Layer> getPhysicalLayerFromApiLayer(uint64_t apiLayerId) {
        return apiLayerToPhysicalLayerShared[apiLayerId];
    }
    std::shared_ptr<ThorImplementation::Layer> getPhysicalLayerFromApiLayer(std::shared_ptr<Thor::Layer> apiLayer) {
        return apiLayerToPhysicalLayerShared[apiLayer->getId()];
    }
    void recordIfParameterizable(std::shared_ptr<Thor::Layer> layer, std::shared_ptr<ThorImplementation::Layer> implementationLayer) {
        std::shared_ptr<Thor::Parameterizable> parameterizable = dynamic_pointer_cast<Thor::Parameterizable>(layer);
        if (parameterizable != nullptr) {
            auto implementationParameterizable = std::dynamic_pointer_cast<ThorImplementation::Parameterizable>(implementationLayer);
            assert(implementationParameterizable != nullptr);
            apiParameterizableToPhysicalParameterizable[parameterizable->getId()] = implementationParameterizable;
        }
    }
    std::shared_ptr<ThorImplementation::Parameterizable> getPhysicalParameterizableFromApiParameterizable(uint64_t apiParameterizableId) {
        auto it = apiParameterizableToPhysicalParameterizable.find(apiParameterizableId);
        assert(it != apiParameterizableToPhysicalParameterizable.end());
        return it->second;
    }
    std::shared_ptr<ThorImplementation::Parameterizable> getPhysicalParameterizableFromApiParameterizable(
        std::shared_ptr<Thor::Layer> apiParameterizable) {
        assert(apiParameterizable != nullptr);
        uint64_t apiParameterizableId = apiParameterizable->getId();
        return getPhysicalParameterizableFromApiParameterizable(apiParameterizableId);
    }

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

    std::map<uint64_t, std::shared_ptr<ThorImplementation::Parameterizable>> apiParameterizableToPhysicalParameterizable;
 // FIXME: get rid of raw pointers
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
    friend class Thor::PlacedNetwork;
    friend class Thor::LocalExecutor;
};

}  // namespace ThorImplementation
