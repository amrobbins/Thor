#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace Thor;

void Initializer::addToNetwork(Network *network) {
    assert(isInitialized());
    network->addToNetwork(this);
}
