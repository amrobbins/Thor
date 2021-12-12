#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"

#include "DeepLearning/Implementation/Layers/Loss.h"

#include "DeepLearning/Implementation/Layers/Utility/Concatenate.h"

using namespace Thor;

atomic<uint64_t> Layer::nextId(2);

void Layer::addToNetwork(Network *network) {
    assert(isInitialized());
    network->addToNetwork(this);
}

void Layer::connectTwoLayers(ThorImplementation::Layer *drivingLayer,
                             ThorImplementation::Layer *loadingLayer,
                             const Thor::Layer *drivingApiLayer,
                             const Thor::Layer *loadingApiLayer,
                             const Thor::Tensor connectingApiTensor) {
    int drivingLayerConnectionType = drivingApiLayer == nullptr ? 0 : drivingApiLayer->getConnectionType(connectingApiTensor);
    int loadingLayerConnectionType = loadingApiLayer == nullptr ? 0 : loadingApiLayer->getConnectionType(connectingApiTensor);

    drivingLayer->connectToNextLayer(loadingLayer, drivingLayerConnectionType, loadingLayerConnectionType);
}
