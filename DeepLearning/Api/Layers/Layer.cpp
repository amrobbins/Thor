#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"

#include "DeepLearning/Implementation/Layers/Loss.h"

using namespace Thor;
using namespace std;

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

// FIXME: Use this version, get rid of the old one, and generally avoid any raw pointers.
void Layer::connectTwoLayers(shared_ptr<ThorImplementation::Layer> drivingLayer,
                             shared_ptr<ThorImplementation::Layer> loadingLayer,
                             const shared_ptr<Thor::Layer> drivingApiLayer,
                             const shared_ptr<Thor::Layer> loadingApiLayer,
                             const Thor::Tensor connectingApiTensor) {
    connectTwoLayers(drivingLayer.get(), loadingLayer.get(), drivingApiLayer.get(), loadingApiLayer.get(), connectingApiTensor);
}