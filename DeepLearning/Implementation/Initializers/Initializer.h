#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "omp.h"

#include <memory>

using std::make_shared;
using std::shared_ptr;

namespace ThorImplementation {

class Initializer {
   public:
    virtual ~Initializer() {}

    virtual void initialize(Layer *layer, Tensor tensorToInitialize) const {
        MultiConnectionLayer *multiConnectionLayer = dynamic_cast<MultiConnectionLayer *>(layer);
        if (multiConnectionLayer != nullptr) {
            initialize(multiConnectionLayer->getStreams(), tensorToInitialize, false);
        } else {
            vector<Stream> streams;
            streams.push_back(layer->getStream());
            initialize(streams, tensorToInitialize, false);
        }
    }

    virtual void initializeSynchronous(Stream stream, Tensor tensorToInitialize) const { initialize({stream}, tensorToInitialize, true); }

    virtual shared_ptr<Initializer> clone() { assert(false); }

   protected:
    virtual void initialize(vector<Stream> streams, Tensor tensorToInitialize, bool synchronous) const { assert(false); }

    virtual void performCopy(Tensor buffer, Tensor tensorToInitialize, vector<Stream> streams, bool synchronous) const {
        if (tensorToInitialize.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU) {
            int tensorGpuNum = tensorToInitialize.getPlacement().getDeviceNum();
            for (uint32_t i = 0; i < streams.size(); ++i) {
                assert(streams[i].getGpuNum() == tensorGpuNum);
            }
        }

        tensorToInitialize.copyFromAsync(buffer, streams[0]);

        if (synchronous) {
            // If the stream does not represent all of the timing dependencies:
            cudaError_t cudaStatus;
            cudaStatus = cudaDeviceSynchronize();
            assert(cudaStatus == cudaSuccess);
        } else if (streams.size() > 1) {
            // The set of streams is known to represent all of the timing dependencies
            Event tensorToInitializeUpdatedEvent = streams[0].putEvent();
            for (unsigned int i = 1; i < streams.size(); ++i) {
                streams[i].waitEvent(tensorToInitializeUpdatedEvent);
            }
        }
    }
};

}  // namespace ThorImplementation
