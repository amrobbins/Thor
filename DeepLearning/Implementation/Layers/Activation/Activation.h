#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Activation : public Layer {
   public:
    ~Activation() override {}

    std::optional<Tensor> createFeatureOutputTensor() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        return featureInput.value().clone();
    }

    // Just returns the number of floats in a single example.
    // For activation functions that require say 2 floating point operations per input float,
    // return 2 * Activation::floatingPointOperationsPerExampleForward()
    uint64_t floatingPointOperationsPerExampleForward() override {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        std::vector<uint64_t> dimensions = featureInput.value().getDescriptor().getDimensions();
        uint64_t numElements = 1;
        for (uint32_t i = 1; i < dimensions.size(); ++i) {
            numElements *= dimensions[i];
        }
        return numElements;
    }

    uint64_t floatingPointOperationsPerExampleBackward() override { return floatingPointOperationsPerExampleForward(); }

    uint64_t logicalByteCountForward(uint64_t validExampleCount) override {
        return Layer::logicalByteCountForward(validExampleCount);
    }

    uint64_t logicalByteCountBackward(uint64_t validExampleCount) override {
        uint64_t bytes = Layer::logicalByteCountBackward(validExampleCount);
        // Activation derivatives consume the retained forward activation.  The
        // retained tensor is a semantic backward operand even when the backend
        // aliases or caches it physically.
        if (featureOutput.has_value() && featureInput.has_value()) {
            bytes = checkedLogicalByteAdd(
                bytes,
                logicalTensorBytesForBatch(featureOutput.value(),
                                           validExampleCount,
                                           logicalBatchCapacityFromTensor(featureInput.value())),
                "Activation backward");
        }
        return bytes;
    }
};

}  // namespace ThorImplementation
