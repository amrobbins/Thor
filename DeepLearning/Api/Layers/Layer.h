#pragma once

#include "DeepLearning/Api/Layers/LayerBase.h"

#include <assert.h>
#include <memory>

namespace Thor {

class FullyConnected;
class Convolution2d;
class Pooling;
class Relu;
class Tanh;

using std::shared_ptr;

// Layer is a wrapper that wraps every supported network layer and allows an "anyLayer" typed parameter
class Layer {
   public:
    Layer() {}
    Layer(LayerBase *layerBase);

    virtual ~Layer() {}

    Layer *getLayer();

    bool operator==(const Layer &other) const;
    bool operator!=(const Layer &other) const;
    bool operator<(const Layer &other) const;
    bool operator>(const Layer &other) const;

   private:
    shared_ptr<LayerBase> layer;
};

}  // namespace Thor
