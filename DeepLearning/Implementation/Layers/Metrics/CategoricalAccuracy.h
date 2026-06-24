#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Layers/Metrics/CustomMetric.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation {
namespace CategoricalAccuracyDetail {

inline bool isPerClassLabelDType(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 || dtype == DataType::INT8 ||
           dtype == DataType::INT16 || dtype == DataType::INT32 || dtype == DataType::FP16 || dtype == DataType::FP32;
}

inline bool isClassIndexLabelDType(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 || dtype == DataType::INT8 ||
           dtype == DataType::INT16 || dtype == DataType::INT32;
}

inline DynamicExpression makeExpression() {
    return DynamicExpression({"predictions", "labels"},
                             {"metric"},
                             [](const DynamicExpression::TensorMap& inputs,
                                const DynamicExpression::TensorMap& outputs,
                                Stream& stream) -> DynamicExpressionBuild {
                                 auto predictionsIt = inputs.find("predictions");
                                 auto labelsIt = inputs.find("labels");
                                 if (predictionsIt == inputs.end() || labelsIt == inputs.end())
                                     throw std::invalid_argument("CategoricalAccuracy expression requires predictions and labels inputs.");

                                 const Tensor& predictionsTensor = predictionsIt->second;
                                 const Tensor& labelsTensor = labelsIt->second;
                                 const std::vector<uint64_t> predictionDims = predictionsTensor.getDescriptor().getDimensions();
                                 const std::vector<uint64_t> labelDims = labelsTensor.getDescriptor().getDimensions();
                                 const DataType predictionDType = predictionsTensor.getDescriptor().getDataType();
                                 const DataType labelDType = labelsTensor.getDescriptor().getDataType();

                                 THOR_THROW_IF_FALSE(predictionDims.size() == 2);
                                 THOR_THROW_IF_FALSE(predictionDims[0] > 0);
                                 THOR_THROW_IF_FALSE(predictionDims[1] >= 2);
                                 THOR_THROW_IF_FALSE(predictionDType == DataType::FP16 || predictionDType == DataType::FP32);

                                 const bool perClassLabels = predictionDims == labelDims && isPerClassLabelDType(labelDType);
                                 const bool classIndexLabels = labelDims.size() == 2 && labelDims[0] == predictionDims[0] &&
                                                               labelDims[1] == 1 && isClassIndexLabelDType(labelDType);
                                 THOR_THROW_IF_FALSE(perClassLabels ^ classIndexLabels);

                                 Expression predictions = Expression::input("predictions", DataType::FP32, DataType::FP32);
                                 // Keep the class dimension as size 1 so class-index labels can be compared directly
                                 // without running integer labels through a shape-only squeeze node. The current
                                 // expression dtype resolver supports integer inputs for comparisons, but generic
                                 // shape-only pointwise nodes still assume floating compute dtypes.
                                 Expression predictedClass = predictions.argmax({1}, {}, DataType::FP32);

                                 Expression trueClass = perClassLabels
                                                            ? Expression::input("labels", DataType::FP32, DataType::FP32)
                                                                  .argmax({1}, {}, DataType::FP32)
                                                            : Expression::input("labels");
                                 Expression correct = Expression::where(predictedClass == trueClass, Expression(1.0), Expression(0.0));
                                 Expression metric = correct.reduce_mean({0, 1}, {0}, DataType::FP32);

                                 ExpressionDefinition definition =
                                     ExpressionDefinition::fromOutputs(Expression::outputs({{"metric", metric}}));
                                 return DynamicExpressionBuild{
                                     std::make_shared<FusedEquation>(FusedEquation::compile(definition.outputs, stream.getGpuNum())),
                                     inputs,
                                     {},
                                     outputs,
                                     {},
                                 };
                             });
}

}  // namespace CategoricalAccuracyDetail

/**
 * Returns the proportion of predictions where the class with the highest prediction score is the true class.
 *
 * The public CategoricalAccuracy metric is kept as its own layer type for API and serialization stability,
 * but its implementation is now the expression-backed CustomMetric path rather than a hand-written metric kernel.
 */
class CategoricalAccuracy : public CustomMetric {
   public:
    CategoricalAccuracy() : CustomMetric(CategoricalAccuracyDetail::makeExpression(), "predictions", "labels", "metric", "Accuracy") {}

    ~CategoricalAccuracy() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override {
        validateCategoricalAccuracyInputs();
        return CustomMetric::createFeatureOutputTensor();
    }

    void compileImpl() override {
        validateCategoricalAccuracyInputs();
        CustomMetric::compileImpl();
    }

    enum class LABEL_FORMAT { INDICATOR_PER_CLASS_TYPE = 7, INDEX_OF_CLASS_TYPE };

    LABEL_FORMAT confirmLabelFormat() { return labelFormat; }

    std::string getType() override { return "CategoricalAccuracy"; }

   private:
    void validateCategoricalAccuracyInputs() {
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

        THOR_THROW_IF_FALSE(featureInputDimensions.size() == 2);
        THOR_THROW_IF_FALSE(featureInputDimensions[0] > 0);
        THOR_THROW_IF_FALSE(featureInputDimensions[1] >= 2);
        THOR_THROW_IF_FALSE(predictionsDataType == DataType::FP16 || predictionsDataType == DataType::FP32);

        const bool perClassLabels = featureInputDimensions == labelDimensions &&
                                    CategoricalAccuracyDetail::isPerClassLabelDType(labelsDataType);
        const bool classIndexLabels = labelDimensions.size() == 2 && featureInputDimensions[0] == labelDimensions[0] &&
                                      labelDimensions[1] == 1 &&
                                      CategoricalAccuracyDetail::isClassIndexLabelDType(labelsDataType);
        THOR_THROW_IF_FALSE(perClassLabels ^ classIndexLabels);

        labelFormat = perClassLabels ? LABEL_FORMAT::INDICATOR_PER_CLASS_TYPE : LABEL_FORMAT::INDEX_OF_CLASS_TYPE;
    }

    LABEL_FORMAT labelFormat = LABEL_FORMAT::INDICATOR_PER_CLASS_TYPE;
};

}  // namespace ThorImplementation
