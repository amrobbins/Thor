#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <optional>
#include <vector>

namespace ThorImplementation {
namespace BinaryAccuracyDetail {

inline DynamicExpression makeExpression() {
    Expression predictions = Expression::input("predictions", DataType::FP32, DataType::FP32);
    Expression labels = Expression::input("labels", DataType::FP32, DataType::FP32);

    Expression predictedLabel = Expression::where(predictions >= Expression(0.5), Expression(1.0), Expression(0.0));
    Expression correct = Expression::where(predictedLabel == labels, Expression(1.0), Expression(0.0));
    Expression metric = correct.reduce_mean({0, 1}, {0}, DataType::FP32);

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"metric", metric}}));
    return DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace BinaryAccuracyDetail

/**
 * Returns the proportion of binary predictions that match the binary labels.
 *
 * The public BinaryAccuracy metric is kept as its own layer type for API and
 * serialization stability, but its implementation is now the expression-backed
 * CustomMetric path rather than a hand-written metric kernel.
 */
class BinaryAccuracy : public CustomMetric {
   public:
    BinaryAccuracy() : CustomMetric(BinaryAccuracyDetail::makeExpression(), "predictions", "labels", "metric", "Accuracy") {}

    ~BinaryAccuracy() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override {
        if (isInferenceOnly())
            return std::nullopt;
        validateBinaryAccuracyInputs();
        return CustomMetric::createFeatureOutputTensor();
    }

    void compileImpl() override {
        if (!isInferenceOnly())
            validateBinaryAccuracyInputs();
        CustomMetric::compileImpl();
    }

    std::string getType() override { return "BinaryAccuracy"; }

   private:
    void validateBinaryAccuracyInputs() const {
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());

        const std::vector<uint64_t> featureInputDimensions = featureInput.value().getDescriptor().getDimensions();
        const std::vector<uint64_t> labelDimensions = labelsInput.value().getDescriptor().getDimensions();
        const DataType labelsDataType = labelsInput.value().getDescriptor().getDataType();
        const DataType predictionsDataType = featureInput.value().getDescriptor().getDataType();

        THOR_THROW_IF_FALSE(labelDimensions.size() == 2);
        THOR_THROW_IF_FALSE(labelDimensions[1] == 1);
        THOR_THROW_IF_FALSE(featureInputDimensions == labelDimensions);
        THOR_THROW_IF_FALSE(predictionsDataType == DataType::FP16 || predictionsDataType == DataType::FP32);
        THOR_THROW_IF_FALSE(labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 ||
                            labelsDataType == DataType::UINT32 || labelsDataType == DataType::INT8 ||
                            labelsDataType == DataType::INT16 || labelsDataType == DataType::INT32 ||
                            labelsDataType == DataType::FP16 || labelsDataType == DataType::FP32);
    }
};

}  // namespace ThorImplementation
