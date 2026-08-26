#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Learning/Convolution1d.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Metrics/Mean.h"
#include "DeepLearning/Api/Layers/Metrics/WeightedMean.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedReduction.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/DynamicExpression.h"

#include "gtest/gtest.h"

#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;

namespace {

struct SimpleReluNetwork {
    Network network;
    NetworkInput input;
    std::shared_ptr<Activation> relu;
    NetworkOutput output;

    SimpleReluNetwork()
        : network("source_relu_network"),
          input(NetworkInput::Builder().network(network).name("features").dimensions({3}).dataType(DataType::FP32).build()),
          relu(Relu::Builder().network(network).featureInput(input.getFeatureOutput().value()).build()),
          output(NetworkOutput::Builder().network(network).name("scores").inputTensor(relu->getFeatureOutput().value()).build()) {}
};

}  // namespace

TEST(ApiSubgraphClone, ClonesSubgraphWithInputRemap) {
    SimpleReluNetwork source;

    Network destination("destination_relu_clone");
    NetworkInput destinationInput = NetworkInput::Builder()
                                        .network(destination)
                                        .name("features")
                                        .dimensions({3})
                                        .dataType(DataType::FP32)
                                        .build();

    ApiTensorRemap remap;
    remap.map(source.input.getFeatureOutput().value(), destinationInput.getFeatureOutput().value());

    ApiSubgraphCloneOptions options;
    options.namePrefix = "member_0/";
    ApiSubgraphCloneResult cloneResult = destination.cloneSubgraphInto(source.network, {"scores"}, remap, options);

    ASSERT_EQ(cloneResult.outputTensorsByName.size(), 1u);
    ASSERT_TRUE(cloneResult.outputTensorsByName.count("scores"));
    Tensor clonedScores = cloneResult.outputTensorsByName.at("scores");
    EXPECT_TRUE(clonedScores.isInitialized());
    EXPECT_EQ(clonedScores.getDimensions(), std::vector<uint64_t>({3}));
    EXPECT_EQ(clonedScores.getDataType(), DataType::FP32);
    EXPECT_NE(clonedScores.getOriginalId(), source.output.getFeatureInput().value().getOriginalId());

    NetworkOutput::Builder().network(destination).name("scores").inputTensor(clonedScores).build();

    std::vector<std::string> inputNames = destination.getInferenceNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "features");
}

TEST(ApiSubgraphClone, TwoClonesOfSameSubgraphCanCoexist) {
    SimpleReluNetwork source;

    Network destination("destination_two_relu_clones");
    NetworkInput destinationInput = NetworkInput::Builder()
                                        .network(destination)
                                        .name("features")
                                        .dimensions({3})
                                        .dataType(DataType::FP32)
                                        .build();

    ApiTensorRemap remap;
    remap.map(source.input.getFeatureOutput().value(), destinationInput.getFeatureOutput().value());

    ApiSubgraphCloneOptions options0;
    options0.namePrefix = "member_0/";
    ApiSubgraphCloneResult clone0 = destination.cloneSubgraphInto(source.network, {"scores"}, remap, options0);

    ApiSubgraphCloneOptions options1;
    options1.namePrefix = "member_1/";
    ApiSubgraphCloneResult clone1 = destination.cloneSubgraphInto(source.network, {"scores"}, remap, options1);

    Tensor scores0 = clone0.outputTensorsByName.at("scores");
    Tensor scores1 = clone1.outputTensorsByName.at("scores");
    EXPECT_TRUE(scores0.isInitialized());
    EXPECT_TRUE(scores1.isInitialized());
    EXPECT_NE(scores0, scores1);
    EXPECT_NE(scores0.getOriginalId(), scores1.getOriginalId());

    NetworkOutput::Builder().network(destination).name("scores_0").inputTensor(scores0).build();
    NetworkOutput::Builder().network(destination).name("scores_1").inputTensor(scores1).build();

    std::vector<std::string> inputNames = destination.getInferenceNetworkInputNames();
    ASSERT_EQ(inputNames.size(), 1u);
    EXPECT_EQ(inputNames[0], "features");
}

TEST(ApiSubgraphClone, ThrowsWhenRequiredInputIsNotRemapped) {
    SimpleReluNetwork source;
    Network destination("destination_missing_remap");

    ApiTensorRemap emptyRemap;
    EXPECT_THROW((destination.cloneSubgraphInto(source.network, {"scores"}, emptyRemap)), std::runtime_error);
}

TEST(ApiSubgraphClone, ThrowsWhenRequestedOutputIsMissing) {
    SimpleReluNetwork source;
    Network destination("destination_missing_output");
    NetworkInput destinationInput = NetworkInput::Builder()
                                        .network(destination)
                                        .name("features")
                                        .dimensions({3})
                                        .dataType(DataType::FP32)
                                        .build();

    ApiTensorRemap remap;
    remap.map(source.input.getFeatureOutput().value(), destinationInput.getFeatureOutput().value());

    EXPECT_THROW((destination.cloneSubgraphInto(source.network, {"missing"}, remap)), std::runtime_error);
}

TEST(ApiSubgraphClone, ThrowsWhenRemapDescriptorsDoNotMatch) {
    SimpleReluNetwork source;
    Network destination("destination_bad_remap");
    NetworkInput destinationInput = NetworkInput::Builder()
                                        .network(destination)
                                        .name("features")
                                        .dimensions({4})
                                        .dataType(DataType::FP32)
                                        .build();

    ApiTensorRemap remap;
    EXPECT_THROW(remap.map(source.input.getFeatureOutput().value(), destinationInput.getFeatureOutput().value()), std::runtime_error);
}

TEST(ApiSubgraphClone, TrainingReportBoundaryIncludesAuxiliaryMetricInputs) {
    Network source("source_auxiliary_metric_network");
    NetworkInput features =
        NetworkInput::Builder().network(source).name("features").dimensions({3}).dataType(DataType::FP32).build();
    NetworkInput actual =
        NetworkInput::Builder().network(source).name("actual").dimensions({3}).dataType(DataType::FP32).build();
    NetworkInput threshold =
        NetworkInput::Builder().network(source).name("threshold").dimensions({3}).dataType(DataType::FP32).build();
    std::shared_ptr<Activation> prediction =
        Relu::Builder().network(source).featureInput(features.getFeatureOutput().value()).build();

    ThorImplementation::Expression actualExpression =
        ThorImplementation::Expression::input("actual", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::Expression thresholdExpression =
        ThorImplementation::Expression::input("threshold", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::ExpressionDefinition maskDefinition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs(
            {{"mask", actualExpression.greaterEqual(thresholdExpression).cast(ThorImplementation::DataType::FP32)}}));
    CustomLayer mask = CustomLayer::Builder()
                           .network(source)
                           .expression(ThorImplementation::DynamicExpression::fromExpressionDefinition(maskDefinition))
                           .inputNames({"actual", "threshold"})
                           .outputNames({"mask"})
                           .inputInterface({{"actual", actual.getFeatureOutput().value()},
                                            {"threshold", threshold.getFeatureOutput().value()}})
                           .build();
    WeightedMean peakMean = WeightedMean::Builder()
                                .network(source)
                                .values(prediction->getFeatureOutput().value())
                                .weights(mask.getOutput("mask"))
                                .build();
    NetworkOutput predictionOutput = NetworkOutput::Builder()
                                         .network(source)
                                         .name("prediction")
                                         .inputTensor(prediction->getFeatureOutput().value())
                                         .build();
    NetworkOutput::Builder().network(source).name("peak_mean").inputTensor(peakMean.getMetric()).build();

    const std::vector<std::string> requiredInputs =
        source.getRequiredNetworkInputNamesForOutputs({"peak_mean"}, /*inferenceOnly=*/false);
    EXPECT_EQ(requiredInputs, (std::vector<std::string>{"features", "actual", "threshold"}));

    const std::vector<NetworkMetricReference> reportableMetrics = source.getReportableMetrics();
    ASSERT_EQ(reportableMetrics.size(), 1u);
    EXPECT_EQ(reportableMetrics.front().metricName, "peak_mean");
    EXPECT_EQ(reportableMetrics.front().aggregation, MetricAggregation::RATIO);
    EXPECT_EQ(reportableMetrics.front().predictionOutputName, "prediction");
    EXPECT_FALSE(reportableMetrics.front().targetInputName.has_value());
    EXPECT_EQ(reportableMetrics.front().requiredInputNames,
              (std::vector<std::string>{"actual", "features", "threshold"}));

    Network destination("destination_auxiliary_metric_clone");
    NetworkInput averagedPrediction = NetworkInput::Builder()
                                           .network(destination)
                                           .name("averaged_prediction")
                                           .dimensions({3})
                                           .dataType(DataType::FP32)
                                           .build();
    NetworkInput destinationActual =
        NetworkInput::Builder().network(destination).name("actual").dimensions({3}).dataType(DataType::FP32).build();
    NetworkInput destinationThreshold =
        NetworkInput::Builder().network(destination).name("threshold").dimensions({3}).dataType(DataType::FP32).build();

    ApiTensorRemap remap;
    remap.map(predictionOutput.getFeatureInput().value(), averagedPrediction.getFeatureOutput().value());
    remap.map(actual.getFeatureOutput().value(), destinationActual.getFeatureOutput().value());
    remap.map(threshold.getFeatureOutput().value(), destinationThreshold.getFeatureOutput().value());

    ApiSubgraphCloneOptions options;
    options.inferenceOnly = false;
    ApiSubgraphCloneResult cloneResult = destination.cloneSubgraphInto(source, {"peak_mean"}, remap, options);
    ASSERT_TRUE(cloneResult.outputTensorsByName.count("peak_mean"));
    EXPECT_EQ(cloneResult.outputTensorsByName.at("peak_mean").getDataType(), DataType::FP32);
    NetworkOutput::Builder()
        .network(destination)
        .name("peak_mean")
        .inputTensor(cloneResult.outputTensorsByName.at("peak_mean"))
        .build();

    const std::vector<NetworkMetricReference> clonedMetrics =
        destination.getReportableMetrics();
    ASSERT_EQ(clonedMetrics.size(), 1u);
    EXPECT_EQ(clonedMetrics.front().metricName, "peak_mean");
    EXPECT_EQ(clonedMetrics.front().aggregation, MetricAggregation::RATIO);
    EXPECT_THROW(
        (void)destination.getRequiredNetworkInputNamesForOutputs(
            {METRIC_AGGREGATION_NUMERATOR_NAME}, /*inferenceOnly=*/false),
        std::runtime_error);
    EXPECT_THROW(
        (void)destination.getRequiredNetworkInputNamesForOutputs(
            {METRIC_AGGREGATION_DENOMINATOR_NAME}, /*inferenceOnly=*/false),
        std::runtime_error);
}


TEST(ApiSubgraphClone, TrainingReportBoundaryIncludesAuxiliaryLossInputs) {
    Network source("source_auxiliary_loss_network");
    NetworkInput features =
        NetworkInput::Builder().network(source).name("features").dimensions({3}).dataType(DataType::FP32).build();
    NetworkInput actual =
        NetworkInput::Builder().network(source).name("actual").dimensions({3}).dataType(DataType::FP32).build();
    NetworkInput threshold =
        NetworkInput::Builder().network(source).name("threshold").dimensions({3}).dataType(DataType::FP32).build();
    std::shared_ptr<Activation> prediction =
        Relu::Builder().network(source).featureInput(features.getFeatureOutput().value()).build();

    ThorImplementation::Expression actualExpression =
        ThorImplementation::Expression::input("actual", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::Expression thresholdExpression =
        ThorImplementation::Expression::input("threshold", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::ExpressionDefinition maskDefinition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs(
            {{"mask", actualExpression.greaterEqual(thresholdExpression).cast(ThorImplementation::DataType::FP32)}}));
    CustomLayer mask = CustomLayer::Builder()
                           .network(source)
                           .expression(ThorImplementation::DynamicExpression::fromExpressionDefinition(maskDefinition))
                           .inputNames({"actual", "threshold"})
                           .outputNames({"mask"})
                           .inputInterface({{"actual", actual.getFeatureOutput().value()},
                                            {"threshold", threshold.getFeatureOutput().value()}})
                           .build();
    MSE loss = MSE::Builder()
                   .network(source)
                   .predictions(prediction->getFeatureOutput().value())
                   .labels(actual.getFeatureOutput().value())
                   .exampleWeights(mask.getOutput("mask"))
                   .reportsBatchLoss()
                   .build();
    NetworkOutput predictionOutput = NetworkOutput::Builder()
                                         .network(source)
                                         .name("prediction")
                                         .inputTensor(prediction->getFeatureOutput().value())
                                         .build();
    NetworkOutput::Builder().network(source).name("peak_mse").inputTensor(loss.getLoss()).build();

    const std::vector<std::string> requiredInputs =
        source.getRequiredNetworkInputNamesForOutputs({"peak_mse"}, /*inferenceOnly=*/false);
    EXPECT_EQ(requiredInputs, (std::vector<std::string>{"features", "actual", "threshold"}));

    const std::vector<NetworkLossReference> reportableLosses = source.getReportableLosses();
    ASSERT_EQ(reportableLosses.size(), 1u);
    EXPECT_EQ(reportableLosses.front().lossName, "peak_mse");
    EXPECT_EQ(reportableLosses.front().predictionOutputName, "prediction");
    EXPECT_EQ(reportableLosses.front().targetInputName, "actual");
    EXPECT_FALSE(reportableLosses.front().weightInputName.has_value());
    EXPECT_EQ(reportableLosses.front().requiredInputNames,
              (std::vector<std::string>{"actual", "features", "threshold"}));

    Network destination("destination_auxiliary_loss_clone");
    NetworkInput averagedPrediction = NetworkInput::Builder()
                                           .network(destination)
                                           .name("averaged_prediction")
                                           .dimensions({3})
                                           .dataType(DataType::FP32)
                                           .build();
    NetworkInput destinationActual =
        NetworkInput::Builder().network(destination).name("actual").dimensions({3}).dataType(DataType::FP32).build();
    NetworkInput destinationThreshold =
        NetworkInput::Builder().network(destination).name("threshold").dimensions({3}).dataType(DataType::FP32).build();

    ApiTensorRemap remap;
    remap.map(predictionOutput.getFeatureInput().value(), averagedPrediction.getFeatureOutput().value());
    remap.map(actual.getFeatureOutput().value(), destinationActual.getFeatureOutput().value());
    remap.map(threshold.getFeatureOutput().value(), destinationThreshold.getFeatureOutput().value());

    ApiSubgraphCloneOptions options;
    options.inferenceOnly = false;
    ApiSubgraphCloneResult cloneResult = destination.cloneSubgraphInto(source, {"peak_mse"}, remap, options);
    ASSERT_TRUE(cloneResult.outputTensorsByName.count("peak_mse"));
    EXPECT_EQ(cloneResult.outputTensorsByName.at("peak_mse").getDataType(), loss.getLoss().getDataType());
}


TEST(ApiSubgraphClone, SourceOnlyMetricDoesNotUseParameterizedMemberPathAsInputSource) {
    Network source("source_hidden_parameterized_metric_network");
    NetworkInput features =
        NetworkInput::Builder().network(source).name("features").dimensions({3}).dataType(DataType::FP32).build();
    FullyConnected hidden = FullyConnected::Builder()
                                .network(source)
                                .featureInput(features.getFeatureOutput().value())
                                .numOutputFeatures(3)
                                .hasBias(true)
                                .noActivation()
                                .build();
    Mean hiddenMean = Mean::Builder().network(source).values(hidden.getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(source).name("hidden_mean").inputTensor(hiddenMean.getMetric()).build();

    const std::vector<NetworkMetricReference> reportableMetrics = source.getReportableMetrics();
    ASSERT_EQ(reportableMetrics.size(), 1u);
    EXPECT_EQ(reportableMetrics.front().metricName, "hidden_mean");
    EXPECT_EQ(reportableMetrics.front().aggregation, MetricAggregation::MEAN_BY_EXAMPLE);
    EXPECT_TRUE(reportableMetrics.front().predictionOutputName.empty());
    EXPECT_FALSE(reportableMetrics.front().inputSourceName.has_value());
    EXPECT_EQ(reportableMetrics.front().requiredInputNames, (std::vector<std::string>{"features"}));
}

TEST(ApiSubgraphClone, SourceOnlyMetricDoesNotUseRaggedOffsetsToBypassParameterizedMemberPath) {
    Network source("source_hidden_parameterized_ragged_metric_network");
    RaggedTensor features = RaggedNetworkInput::Builder()
                                .network(source)
                                .name("features")
                                .valuesDataType(DataType::FP32)
                                .offsetsDataType(DataType::UINT32)
                                .trailingDimensions({3})
                                .maxTotalValues(16)
                                .maxValuesPerRow(8)
                                .batchSize(2)
                                .build();
    Convolution1d hidden = Convolution1d::Builder()
                               .network(source)
                               .featureInput(features)
                               .numOutputChannels(3)
                               .filterWidth(3)
                               .causalPadding()
                               .noActivation()
                               .build();
    SegmentedReduction pooled = SegmentedReduction::Builder()
                                    .network(source)
                                    .featureInput(hidden.getRaggedFeatureOutput().value())
                                    .reductionType(SegmentedReduction::Type::MEAN)
                                    .build();
    Mean hiddenMean = Mean::Builder().network(source).values(pooled.getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(source).name("hidden_ragged_mean").inputTensor(hiddenMean.getMetric()).build();

    const std::vector<NetworkMetricReference> reportableMetrics = source.getReportableMetrics();
    ASSERT_EQ(reportableMetrics.size(), 1u);
    EXPECT_EQ(reportableMetrics.front().metricName, "hidden_ragged_mean");
    EXPECT_EQ(reportableMetrics.front().aggregation, MetricAggregation::MEAN_BY_EXAMPLE);
    EXPECT_TRUE(reportableMetrics.front().predictionOutputName.empty());
    EXPECT_FALSE(reportableMetrics.front().inputSourceName.has_value());
    EXPECT_EQ(reportableMetrics.front().requiredInputNames, (std::vector<std::string>{"features"}));
}
