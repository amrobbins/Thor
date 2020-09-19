#pragma once

#include "DeepLearning/Api/Layers/LayerBase.h"
#include "DeepLearning/Api/Tensor.h"
// FIXME: include all the layer headers
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"

#include <assert.h>
#include <memory>

namespace Thor {

class Network;

using std::shared_ptr;

// Layer is a wrapper that wraps every supported network layer and allows an "anyLayer" typed parameter
class Layer {
   public:
    Layer() {}
    Layer(LayerBase *layerBase);

    virtual ~Layer() {}

    uint32_t getId() const { return layer->getId(); }

    virtual Optional<Tensor> getFeatureInput() const { return layer->getFeatureInput(); }
    virtual Optional<Tensor> getFeatureOutput() const { return layer->getFeatureOutput(); }

    bool operator==(const Layer &other) const;
    bool operator!=(const Layer &other) const;
    bool operator<(const Layer &other) const;
    bool operator>(const Layer &other) const;

   protected:
    shared_ptr<LayerBase> layer;

    LayerBase *getRawLayer() const;

    friend class Network;
};

}  // namespace Thor
