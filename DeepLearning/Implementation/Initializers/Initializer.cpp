#include "DeepLearning/Implementation/Initializers/Initializer.h"

using namespace std;

namespace ThorImplementation {

Initializer::~Initializer() {}

Event Initializer::initialize(Layer *layer, Tensor tensorToInitialize) {
    MultiConnectionLayer *multiConnectionLayer = dynamic_cast<MultiConnectionLayer *>(layer);
    Event initDoneEvent;
    if (multiConnectionLayer != nullptr) {
        // the difference being whether there is 1 or potentially multiple input streams
        initDoneEvent = initialize(layer, tensorToInitialize, multiConnectionLayer->getStreams());
    } else {
        vector<Stream> streams;
        streams.push_back(layer->getStream());
        initDoneEvent = initialize(layer, tensorToInitialize, streams);
    }
    return initDoneEvent;
}

shared_ptr<Initializer> Initializer::clone() { assert(false); }

Event Initializer::performCopy(Tensor buffer, Tensor tensorToInitialize, vector<Stream> streams) {
    if (tensorToInitialize.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        int tensorGpuNum = tensorToInitialize.getPlacement().getDeviceNum();
        for (uint32_t i = 0; i < streams.size(); ++i) {
            assert(streams[i].getGpuNum() == tensorGpuNum);
        }
    }

    tensorToInitialize.copyFromAsync(buffer, streams[0]);

    Event tensorToInitializeUpdatedEvent = streams[0].putEvent();
    if (streams.size() > 1) {
        // The set of streams is known to represent all the timing dependencies
        for (unsigned int i = 1; i < streams.size(); ++i) {
            streams[i].waitEvent(tensorToInitializeUpdatedEvent);
        }
    }

    return tensorToInitializeUpdatedEvent;
}

}  // namespace ThorImplementation
