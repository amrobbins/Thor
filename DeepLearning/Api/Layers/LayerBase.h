#pragma once

#include "DeepLearning/Api/Tensor.h"
#include "Utilities/Common/Optional.h"

#include <assert.h>
#include <atomic>
#include <utility>

using std::atomic;

namespace Thor {

class LayerBase {
   public:
    LayerBase() : id(nextId.fetch_add(1)) {}
    virtual ~LayerBase() {}

    uint32_t getId() const { return id; }

    virtual Optional<Tensor> getFeatureInput() const { return featureInput; }
    virtual Optional<Tensor> getFeatureOutput() const { return featureOutput; }

    bool operator==(const LayerBase &other) const { return id == other.id; }
    bool operator!=(const LayerBase &other) const { return id != other.id; }
    bool operator<(const LayerBase &other) const { return id < other.id; }
    bool operator>(const LayerBase &other) const { return id > other.id; }

   protected:
    Optional<Tensor> featureInput;
    Optional<Tensor> featureOutput;

   private:
    uint32_t id;
    static atomic<uint32_t> nextId;
};

}  // namespace Thor
