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

class Layer {
   public:
    Layer() : id(nextId.fetch_add(1)) {}
    virtual ~Layer() {}

    uint64_t getId() const { return id; }

    virtual Optional<Tensor> getFeatureInput() const { return featureInput; }
    virtual Optional<Tensor> getFeatureOutput() const { return featureOutput; }

    bool operator==(const Layer &other) const { return id == other.id; }
    bool operator!=(const Layer &other) const { return id != other.id; }
    bool operator<(const Layer &other) const { return id < other.id; }
    bool operator>(const Layer &other) const { return id > other.id; }

    virtual shared_ptr<Layer> clone() const = 0;

   protected:
    Optional<Tensor> featureInput;
    Optional<Tensor> featureOutput;

    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const = 0;

    virtual bool isMultiLayer() const { return false; }
    virtual void toSingleLayers(vector<shared_ptr<Layer>> &singleLayers) const {
        assert(!isMultiLayer());
        singleLayers.push_back(clone());
    }

   private:
    uint64_t id;
    static atomic<uint64_t> nextId;

    friend class Network;
};

}  // namespace Thor
