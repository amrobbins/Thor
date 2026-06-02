#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Activations/Activation.h"

#include <optional>
#include <utility>
#include <vector>

namespace Thor {

class GatedLinearUnitActivation : public Activation {
   protected:
    enum class GateKind { Sigmoid, Relu, Gelu, Swish, Bilinear };

    explicit GatedLinearUnitActivation(GateKind gateKind) : gateKind(gateKind) {}

   public:
    ~GatedLinearUnitActivation() override = default;

    using Activation::addToNetwork;

    ThorImplementation::Expression toExpression(const ThorImplementation::Expression& input) const override {
        const auto [value, gate] = splitInputExpression(input);

        switch (gateKind) {
            case GateKind::Sigmoid:
                return value * gate.sigmoid();
            case GateKind::Relu:
                return value * gate.max(ThorImplementation::Expression(0.0));
            case GateKind::Gelu:
                return value * gate.gelu();
            case GateKind::Swish:
                return value * gate.swish();
            case GateKind::Bilinear:
                return value * gate;
        }

        throw std::runtime_error("Unsupported gated linear unit activation kind.");
    }

    Tensor addToNetwork(Tensor inputTensor, Network* network) override {
        std::optional<Tensor> maybeExistingFeatureInput = featureInput;
        std::optional<Tensor> maybeExistingFeatureOutput = featureOutput;

        featureInput = inputTensor;
        Tensor activationOutput = outputTensorForInput(featureInput.value());
        featureOutput = activationOutput;
        Layer::addToNetwork(network);

        featureInput = maybeExistingFeatureInput;
        featureOutput = maybeExistingFeatureOutput;

        return activationOutput;
    }

   public:
    static Tensor outputTensorForInput(const Tensor& input) {
        return Tensor(input.getDataType(), outputDimensionsForInput(input.getDimensions()));
    }

    static std::vector<uint64_t> outputDimensionsForInput(const std::vector<uint64_t>& inputDims) {
        THOR_THROW_IF_FALSE(!inputDims.empty());
        for (uint64_t dim : inputDims) {
            THOR_THROW_IF_FALSE(dim != 0);
        }
        THOR_THROW_IF_FALSE((inputDims.back() % 2) == 0);
        THOR_THROW_IF_FALSE(inputDims.back() >= 2);

        std::vector<uint64_t> outputDims = inputDims;
        outputDims.back() /= 2;
        return outputDims;
    }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)drivingLayer;
        (void)drivingApiLayer;
        return stampExpressionBackedActivation(placement, connectingApiTensor, inferenceOnly);
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)tensorPlacement;
        return batchSize * (featureOutput.value().getTotalSizeInBytes() + featureInput.value().getTotalSizeInBytes());
    }

   private:
    std::pair<ThorImplementation::Expression, ThorImplementation::Expression> splitInputExpression(
        const ThorImplementation::Expression& input) const {
        THOR_THROW_IF_FALSE(featureInput.has_value());

        const std::vector<uint64_t> inputDims = featureInput.value().getDimensions();
        const std::vector<uint64_t> outputDims = outputDimensionsForInput(inputDims);
        const std::vector<uint64_t> inputStrides = contiguousStrides(inputDims);
        const uint64_t halfWidth = outputDims.back();

        ThorImplementation::Expression value = input.stridedView(outputDims, inputStrides, 0);
        ThorImplementation::Expression gate = input.stridedView(outputDims, inputStrides, halfWidth);
        return {value, gate};
    }

    static std::vector<uint64_t> contiguousStrides(const std::vector<uint64_t>& dims) {
        THOR_THROW_IF_FALSE(!dims.empty());
        std::vector<uint64_t> strides(dims.size(), 1);
        uint64_t running = 1;
        for (size_t axis = dims.size(); axis-- > 0;) {
            strides[axis] = running;
            running *= dims[axis];
        }
        return strides;
    }

    GateKind gateKind;
};

}  // namespace Thor
