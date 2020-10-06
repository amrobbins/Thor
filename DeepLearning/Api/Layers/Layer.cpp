#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"

#include "DeepLearning/Implementation/Layers/Loss.h"

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
    int drivingConnectionType = drivingApiLayer == nullptr ? 0 : drivingApiLayer->getConnectionType(connectingApiTensor);
    int loadingConnectionType = loadingApiLayer == nullptr ? 0 : loadingApiLayer->getConnectionType(connectingApiTensor);

    printf("connecting %d to %d\n", (int)drivingConnectionType, (int)loadingConnectionType);

    drivingLayer->connectToNextLayer(loadingLayer, drivingConnectionType, loadingConnectionType);
}

/* FIXME: delete:

    int drivingConnectionType = 0;
    const Thor::Loss *lossLoader = dynamic_cast<const Thor::Loss *>(drivingApiLayer);
    if (lossDriver) {
        if(lossDriver->getFeatureOutput().get() == connectingApiTensor) {
            drivingConnectionType = (int)ThorImplementation::Loss::ConnectionType::PREDICTIONS;
        } else {
            assert(lossDriver->getLossOutput() == connectingTensor);
            drivingConnectionType = (int)ThorImplementation::Loss::ConnectionType::LOSS;
        }
    }

    int loadingConnectionType = 0;
    const Thor::Loss *lossLoader = dynamic_cast<const Thor::Loss *>(loadingApiLayer);
    if(lossLoader) {
        if(lossLoader->getFeatureInput() == connectingApiTensor) {
            loadingConnectionType = (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD;
        } else {
            assert(lossLoader->getLabels() == connectingApiTensor);
            loadingConnectionType = (int)ThorImplementation::Loss::ConnectionType::LABELS;
        }
    }
*/
