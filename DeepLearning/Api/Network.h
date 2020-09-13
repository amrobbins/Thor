#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include <assert.h>

namespace Thor {

class Network {
   public:
    Network() {}

    void addToNetwork(Layer layer) {
        assert(network.count(layer) == 0);
        network.insert(layer);
    }

   private:
    set<Layer> network;
};

}  // namespace Thor
