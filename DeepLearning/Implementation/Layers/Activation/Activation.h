#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Arithmetic/Relu.h"

namespace ThorImplementation {

class Activation : public Layer {
   public:
    virtual ~Activation() {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(featureInput.isPresent());
        return featureInput.get().clone();
    }

    // Just returns the number of floats in a single example.
    // For activation functions that require say 2 floating point operations per input float,
    // return 2 * Activation::floatingPointOperationsPerExampleForward()
    virtual uint64_t floatingPointOperationsPerExampleForward() {
        assert(featureInput.isPresent());
        std::vector<uint64_t> dimensions = featureInput.get().getDescriptor().getDimensions();
        uint64_t numElements = 1;
        for (uint32_t i = 1; i < dimensions.size(); ++i) {
            numElements *= dimensions[i];
        }
        return numElements;
    }

    virtual uint64_t floatingPointOperationsPerExampleBackward() { return floatingPointOperationsPerExampleForward(); }
};

}  // namespace ThorImplementation
