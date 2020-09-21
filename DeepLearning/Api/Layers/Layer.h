#pragma once

#include "DeepLearning/Api/Layers/LayerBase.h"
// FIXME: include all the layer headers
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"

#include <assert.h>
#include <memory>
#include <vector>

using std::vector;

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

    operator LayerBase *() { return getRawLayer(); }

   protected:
    shared_ptr<LayerBase> layer;

    virtual bool isMultiLayer() { return layer->isMultiLayer(); }
    virtual void toRawSingleLayers(vector<LayerBase *> &singleLayers) { layer->toSingleLayers(singleLayers); }

    LayerBase *getRawLayer() const;

    friend class Network;
};

}  // namespace Thor
