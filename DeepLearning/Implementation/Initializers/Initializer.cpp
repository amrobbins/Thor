#include "DeepLearning/Implementation/Initializers/Initializer.h"

using std::make_shared;
using std::shared_ptr;

namespace ThorImplementation {

Initializer::~Initializer() {}

void Initializer::initialize(Layer *layer, Tensor tensorToInitialize) {
    MultiConnectionLayer *multiConnectionLayer = dynamic_cast<MultiConnectionLayer *>(layer);
    if (multiConnectionLayer != nullptr) {
        // the difference being whether there is 1 or potentially multiple input streams
        initialize(layer, tensorToInitialize, multiConnectionLayer->getStreams());
    } else {
        vector<Stream> streams;
        streams.push_back(layer->getStream());
        initialize(layer, tensorToInitialize, streams);
    }
}

shared_ptr<Initializer> Initializer::clone() { assert(false); }

void Initializer::performCopy(Tensor buffer, Tensor tensorToInitialize, vector<Stream> streams) {
    if (tensorToInitialize.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
        int tensorGpuNum = tensorToInitialize.getPlacement().getDeviceNum();
        for (uint32_t i = 0; i < streams.size(); ++i) {
            assert(streams[i].getGpuNum() == tensorGpuNum);
        }
    }

    tensorToInitialize.copyFromAsync(buffer, streams[0]);

    if (streams.size() > 1) {
        // The set of streams is known to represent all of the timing dependencies
        Event tensorToInitializeUpdatedEvent = streams[0].putEvent();
        for (unsigned int i = 1; i < streams.size(); ++i) {
            streams[i].waitEvent(tensorToInitializeUpdatedEvent);
        }
    }
}

}  // namespace ThorImplementation
