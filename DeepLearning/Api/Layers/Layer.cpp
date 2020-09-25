#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace Thor;

atomic<uint64_t> Layer::nextId(2);

void Layer::addToNetwork(Network *network) {
    assert(isInitialized());
    network->addToNetwork(this);
}
