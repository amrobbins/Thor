#include "DeepLearning/Api/Layers/Activations/GatedLinearUnitActivation.h"

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <stdexcept>

namespace Thor {

GatedLinearUnitActivation::~GatedLinearUnitActivation() = default;

std::shared_ptr<ThorImplementation::Layer> GatedLinearUnitActivation::stamp(
    ThorImplementation::TensorPlacement placement,
    std::shared_ptr<ThorImplementation::Layer> drivingLayer,
    std::shared_ptr<Thor::Layer> drivingApiLayer,
    Thor::Tensor connectingApiTensor,
    const bool inferenceOnly) const {
    (void)drivingLayer;
    (void)drivingApiLayer;

    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(featureInput.has_value());
    THOR_THROW_IF_FALSE(featureOutput.has_value());
    THOR_THROW_IF_FALSE(connectingApiTensor == featureInput.value());

    using ThorImplementation::CustomLayer;
    using ThorImplementation::DynamicExpression;
    using ThorImplementation::DynamicExpressionBuild;
    using ThorImplementation::Expression;
    using ThorImplementation::FusedEquation;
    using ThorImplementation::Tensor;

    const GateKind stampedGateKind = gateKind;
    DynamicExpression expression(
        {"feature_input"},
        {"feature_output"},
        [stampedGateKind, placement](
            const DynamicExpression::TensorMap& inputs,
            const DynamicExpression::TensorMap& outputs,
            Stream& stream) -> DynamicExpressionBuild {
            (void)stream;

            const Tensor& inputTensor = inputs.at("feature_input");
            if (inputTensor.getPlacement() != placement) {
                throw std::runtime_error("GatedLinearUnitActivation feature input placement does not match the layer placement.");
            }

            const std::vector<uint64_t> inputDims = inputTensor.getDimensions();
            if (inputDims.size() < 2) {
                throw std::runtime_error(
                    "GatedLinearUnitActivation requires a physical feature input with batch plus at least one feature dimension.");
            }
            if (inputDims.back() < 2 || (inputDims.back() % 2) != 0) {
                throw std::runtime_error(
                    "GatedLinearUnitActivation physical final feature dimension must be non-zero and even.");
            }

            std::vector<uint64_t> outputDims = inputDims;
            outputDims.back() /= 2;
            const std::vector<uint64_t> inputStrides = GatedLinearUnitActivation::contiguousStrides(inputDims);
            const uint64_t halfWidth = outputDims.back();

            Expression input = Expression::input(
                "feature_input", inputTensor.getDataType(), inputTensor.getDataType());
            Expression value = input.stridedView(outputDims, inputStrides, 0);
            Expression gate = input.stridedView(outputDims, inputStrides, halfWidth);
            Expression output = [&]() -> Expression {
                switch (stampedGateKind) {
                    case GateKind::Sigmoid:
                        return value * gate.sigmoid();
                    case GateKind::Relu:
                        return value * gate.max(Expression(0.0));
                    case GateKind::Gelu:
                        return value * gate.gelu();
                    case GateKind::Swish:
                        return value * gate.swish();
                    case GateKind::Bilinear:
                        return value * gate;
                }
                throw std::runtime_error("Unsupported gated linear unit activation kind.");
            }();
            output = output.withOutputDType(inputTensor.getDataType());

            if (outputs.contains("feature_output")) {
                const Tensor& outputTensor = outputs.at("feature_output");
                if (outputTensor.getDimensions() != outputDims) {
                    throw std::runtime_error(
                        "GatedLinearUnitActivation feature output dimensions do not match the physical input split.");
                }
                if (outputTensor.getDataType() != inputTensor.getDataType()) {
                    throw std::runtime_error(
                        "GatedLinearUnitActivation feature output dtype does not match the feature input dtype.");
                }
                if (outputTensor.getPlacement() != placement) {
                    throw std::runtime_error(
                        "GatedLinearUnitActivation feature output placement does not match the layer placement.");
                }
            }

            auto expressionOutputs = Expression::outputs({{"feature_output", output}});
            return DynamicExpressionBuild{
                std::make_shared<FusedEquation>(
                    FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
                inputs,
                {},
                outputs,
                {},
            };
        });

    auto physicalActivation = std::make_shared<CustomLayer>(
        std::move(expression),
        std::vector<std::string>{"feature_input"},
        std::vector<std::string>{"feature_output"},
        placement,
        std::vector<std::shared_ptr<ThorImplementation::PhysicalParameter>>{},
        inferenceOnly,
        getId());
    physicalActivation->setLayerName(getLayerType() + "#" + std::to_string(getId()));
    return physicalActivation;
}

}  // namespace Thor
