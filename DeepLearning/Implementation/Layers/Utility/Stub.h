#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Layer.h"

#include <optional>

namespace ThorImplementation {

// Terminal sink for an intentionally discarded API tensor.  Stub consumes the
// producer's physical feature output so the producer is stamped with a valid
// destination tensor, but performs no work of its own and exposes no output.
// It also terminates gradient propagation along this branch.
class Stub : public Layer {
   public:
    Stub() = default;
    ~Stub() override = default;

    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override {
        (void)nextLayer;
        (void)driverConnectionType;
        (void)loaderConnectionType;
        THOR_UNREACHABLE();
    }

    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override {
        (void)backPropagateError;
        return std::nullopt;
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        (void)stream;
        THOR_THROW_IF_FALSE(inputTensor.has_value());
        THOR_THROW_IF_FALSE(!outputTensor.has_value());
    }

    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream stream) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)stream;
    }

    std::string getType() override { return "Stub"; }
};

}  // namespace ThorImplementation
