#pragma once

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Arithmetic/Relu.h"

namespace ThorImplementation {

class Linear : public Activation {
   public:
    virtual ~Linear() {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        return featureInput;
    }

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) {
        // NOP
    }

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) {
        // FIXME: it would be better to avoid duplicating the memory and copying, but that's not currently possible
        if (errorIn.isPresent()) {
            assert(errorOut.isPresent());
            errorOut.get().copyFromAsync(errorIn, stream);
        } else {
            assert(errorOut.isEmpty());
        }
    }

    virtual uint64_t floatingPointOperationsPerExampleForward() { return 0; }

    virtual uint64_t floatingPointOperationsPerExampleBackward() { return 0; }
};

}  // namespace ThorImplementation
