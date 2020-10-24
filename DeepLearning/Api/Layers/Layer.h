#pragma once

#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Common/Optional.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <assert.h>
#include <atomic>
#include <memory>
#include <utility>

using std::atomic;
using std::make_shared;
using std::shared_ptr;
using std::unique_ptr;

namespace Thor {

class Network;
class Initializer;

class Layer {
   public:
    Layer() : initialized(false), id(nextId.fetch_add(1)) {}
    virtual ~Layer() {}

    uint64_t getId() const { return id; }

    virtual Optional<Tensor> getFeatureOutput() const { return featureOutput; }
    virtual Optional<Tensor> getFeatureInput() const { return featureInput; }

    virtual vector<Tensor> getOutputsFromInput(Tensor inputTensor) {
        assert(inputTensor == featureInput.get());
        return {featureOutput};
    }

    virtual bool mustConnectAllInputsToDriveOutput() { return false; }
    virtual void informThatInputConnectionMade(Tensor inputTensor) {}

    bool isInitialized() { return initialized; }

    virtual void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) {}

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const = 0;
    // Layers with weights that share the weights mem with other instances of the same layer on the same gpu will have less non first
    // instance fixed mem requirements.
    virtual uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        return getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }

    bool operator==(const Layer &other) const { return id == other.id; }
    bool operator!=(const Layer &other) const { return id != other.id; }
    bool operator<(const Layer &other) const { return id < other.id; }
    bool operator>(const Layer &other) const { return id > other.id; }

    virtual int getConnectionType(Tensor connectingTensor) const {
        assert(connectingTensor == getFeatureInput() || connectingTensor == getFeatureOutput());
        return 0;
    }

    virtual vector<Tensor> getAllOutputTensors() const { return {getFeatureOutput().get()}; }

    virtual shared_ptr<Layer> clone() const = 0;

   protected:
    Optional<Tensor> featureInput;
    Optional<Tensor> featureOutput;

    // Note: The final API typed parameters are needed to choose from multiple types of output connections and input connections for
    // physical layers
    //       that are direct replacements for API layers.
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers) const = 0;

    virtual bool isMultiLayer() const { return false; }
    virtual void convertToSingleLayersAndAddToNetwork() { assert(false); }

    virtual void addToNetwork(Network *network);

    // Note: The final API typed parameters are needed to choose from multiple types of output connections and input connections for
    // physical layers
    //       that are direct replacements for API layers.
    static void connectTwoLayers(ThorImplementation::Layer *drivingLayer,
                                 ThorImplementation::Layer *loadingLayer,
                                 const Thor::Layer *drivingApiLayer = nullptr,
                                 const Thor::Layer *loadingApiLayer = nullptr,
                                 const Thor::Tensor connectingApiTensor = Thor::Tensor());

    bool initialized;

   private:
    uint64_t id;
    static atomic<uint64_t> nextId;

    friend class Network;
};

}  // namespace Thor
