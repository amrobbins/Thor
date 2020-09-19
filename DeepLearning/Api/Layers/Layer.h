#pragma once

#include "DeepLearning/Api/Layers/LayerBase.h"
#include "DeepLearning/Api/Tensor.h"
// FIXME: include all the layer headers

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

    virtual Optional<Tensor> getFeatureInput() { return layer->getFeatureInput(); }
    virtual Optional<Tensor> getFeatureOutput() { return layer->getFeatureOutput(); }

    bool operator==(const Layer &other) const;
    bool operator!=(const Layer &other) const;
    bool operator<(const Layer &other) const;
    bool operator>(const Layer &other) const;

   protected:
    shared_ptr<LayerBase> layer;

    LayerBase *getRawLayer();

    friend class Network;
};

}  // namespace Thor
