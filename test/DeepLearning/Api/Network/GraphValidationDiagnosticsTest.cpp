#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Metrics/Mean.h"
#include "DeepLearning/Api/Layers/Utility/Add.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Network/Network.h"
#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

class GraphValidationDiagnosticsTestNetwork : public Api::Network {
   public:
    using Api::Network::Network;

    Api::Network::StatusCode evaluateGraphForTest(bool inferenceOnly) { return evaluateGraph(inferenceOnly); }

    Api::Network::StatusCode injectSelfCycleAndCheckForTest(uint64_t layerId, const Api::Tensor& outputTensor) {
        std::shared_ptr<Api::Layer> target;
        for (const std::shared_ptr<Api::Layer>& layer : allLayersInNetworkList) {
            if (layer != nullptr && layer->getId() == layerId) {
                target = layer;
                break;
            }
        }
        if (target == nullptr) {
            throw std::runtime_error("Unable to find requested layer for synthetic cycle test.");
        }
        apiTensorToApiLoadingLayers[outputTensor].push_back(target);
        apiLayerToApiInputTensors[target].push_back(outputTensor);
        return checkForDeadlockCycles();
    }
};

class CountingReportValidationNetwork : public Api::Network {
   public:
    using Api::Network::Network;

    [[nodiscard]] uint32_t getTrainingValidationPasses() const { return trainingValidationPasses; }

   private:
    StatusCode checkForDuplicateInOutPortNames() override {
        ++trainingValidationPasses;
        // These tests build unique API names, so the duplicate-name check itself
        // can be elided while still counting complete evaluateGraph() passes.
        return StatusCode::SUCCESS;
    }

    uint32_t trainingValidationPasses = 0;
};

Impl::DynamicExpression makeTwoOutputExpression() {
    Impl::Expression x = Impl::Expression::input("x", Impl::DataType::FP32, Impl::DataType::FP32);
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(
        Impl::Expression::outputs({{"prediction", x + 1.0f}, {"loss_only", x * 2.0f}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

TEST(GraphValidationDiagnostics, DanglingOutputNamesDrivingLayerAndCustomOutputPort) {
    GraphValidationDiagnosticsTestNetwork network("actionable_dangling_output");

    Api::NetworkInput features = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("features")
                                     .dimensions({4})
                                     .dataType(Impl::DataType::FP32)
                                     .build();
    Api::CustomLayer multiOutput = Api::CustomLayer::Builder()
                                       .network(network)
                                       .expression(makeTwoOutputExpression())
                                       .inputInterface({{"x", features.getFeatureOutput().value()}})
                                       .build();
    (void)Api::Stub::Builder().network(network).inputTensor(multiOutput.getOutput("prediction")).build();

    EXPECT_EQ(network.evaluateGraphForTest(/*inferenceOnly=*/false), Api::Network::StatusCode::DANGLING_OUTPUT);

    const std::optional<Api::Network::GraphValidationIssue> issue = network.getLastGraphValidationIssue();
    ASSERT_TRUE(issue.has_value());
    EXPECT_EQ(issue->status, Api::Network::StatusCode::DANGLING_OUTPUT);
    EXPECT_NE(issue->summary.find("no forward consumer"), std::string::npos);

    const std::string error = network.getLastGraphValidationError();
    EXPECT_NE(error.find("Thor graph validation failed: DANGLING OUTPUT"), std::string::npos);
    EXPECT_NE(error.find("Network: \"actionable_dangling_output\""), std::string::npos);
    EXPECT_NE(error.find("Graph mode: training"), std::string::npos);
    EXPECT_NE(error.find("CustomLayer#"), std::string::npos);
    EXPECT_NE(error.find("output_name=\"loss_only\""), std::string::npos);
    EXPECT_NE(error.find("sibling_outputs"), std::string::npos);
    EXPECT_NE(error.find("Connect an intentionally discarded value to Stub"), std::string::npos);

    try {
        (void)network.getRequiredNetworkInputNamesForOutputs({"prediction"}, /*inferenceOnly=*/false);
        FAIL() << "Expected required-input discovery to propagate the graph validation failure.";
    } catch (const std::runtime_error& exception) {
        const std::string message = exception.what();
        EXPECT_NE(message.find("Unable to evaluate training graph"), std::string::npos);
        EXPECT_NE(message.find("output_name=\"loss_only\""), std::string::npos);
    }
}

TEST(GraphValidationDiagnostics, FloatingInputsListConsumersAndInputPortNames) {
    GraphValidationDiagnosticsTestNetwork network("actionable_floating_inputs");
    Api::Tensor predictions(Api::DataType::FP32, {4});
    Api::Tensor labels(Api::DataType::FP32, {4});

    (void)Api::MSE::Builder()
        .network(network)
        .predictions(predictions)
        .labels(labels)
        .reportsNoLoss()
        .build();

    EXPECT_EQ(network.evaluateGraphForTest(/*inferenceOnly=*/false), Api::Network::StatusCode::FLOATING_INPUT);

    const std::string error = network.getLastGraphValidationError();
    EXPECT_NE(error.find("Found 2 tensor(s) with no driving layer"), std::string::npos);
    EXPECT_NE(error.find("produced_by: <none>"), std::string::npos);
    EXPECT_NE(error.find("input_name=\"predictions\""), std::string::npos);
    EXPECT_NE(error.find("input_name=\"labels\""), std::string::npos);
    EXPECT_NE(error.find("Connect the tensor to a NetworkInput"), std::string::npos);
}

TEST(GraphValidationDiagnostics, DuplicateInputNamesIdentifyEveryConflictingLayer) {
    GraphValidationDiagnosticsTestNetwork network("actionable_duplicate_inputs");

    Api::NetworkInput first = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("features")
                                  .dimensions({4})
                                  .dataType(Impl::DataType::FP32)
                                  .build();
    Api::NetworkInput second = Api::NetworkInput::Builder()
                                   .network(network)
                                   .name("features")
                                   .dimensions({4})
                                   .dataType(Impl::DataType::FP32)
                                   .build();
    (void)Api::Stub::Builder().network(network).inputTensor(first.getFeatureOutput().value()).build();
    (void)Api::Stub::Builder().network(network).inputTensor(second.getFeatureOutput().value()).build();

    EXPECT_EQ(network.evaluateGraphForTest(/*inferenceOnly=*/false),
              Api::Network::StatusCode::DUPLICATE_NAMED_NETWORK_INPUT);

    const std::string error = network.getLastGraphValidationError();
    EXPECT_NE(error.find("Found 1 duplicate network input name(s)"), std::string::npos);
    EXPECT_NE(error.find("name=\"features\" is declared by"), std::string::npos);
    const std::string layerMarker = "NetworkInput#";
    const size_t firstLayer = error.find(layerMarker);
    ASSERT_NE(firstLayer, std::string::npos);
    EXPECT_NE(error.find(layerMarker, firstLayer + layerMarker.size()), std::string::npos);
    EXPECT_NE(error.find("Give every external NetworkInput"), std::string::npos);
}

TEST(GraphValidationDiagnostics, DuplicateOutputNamesIdentifyEveryConflictingLayer) {
    GraphValidationDiagnosticsTestNetwork network("actionable_duplicate_outputs");

    Api::NetworkInput features = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("features")
                                     .dimensions({4})
                                     .dataType(Impl::DataType::FP32)
                                     .build();
    (void)Api::NetworkOutput::Builder()
        .network(network)
        .name("prediction")
        .inputTensor(features.getFeatureOutput().value())
        .build();
    (void)Api::NetworkOutput::Builder()
        .network(network)
        .name("prediction")
        .inputTensor(features.getFeatureOutput().value())
        .build();

    EXPECT_EQ(network.evaluateGraphForTest(/*inferenceOnly=*/false),
              Api::Network::StatusCode::DUPLICATE_NAMED_NETWORK_OUTPUT);

    const std::string error = network.getLastGraphValidationError();
    EXPECT_NE(error.find("Found 1 duplicate network output name(s)"), std::string::npos);
    EXPECT_NE(error.find("name=\"prediction\" is declared by"), std::string::npos);
    const std::string layerMarker = "NetworkOutput#";
    const size_t firstLayer = error.find(layerMarker);
    ASSERT_NE(firstLayer, std::string::npos);
    EXPECT_NE(error.find(layerMarker, firstLayer + layerMarker.size()), std::string::npos);
    EXPECT_NE(error.find("Give every NetworkOutput a unique name"), std::string::npos);
}

TEST(GraphValidationDiagnostics, DeadlockCycleIncludesTheLayerAndTensorPath) {
    GraphValidationDiagnosticsTestNetwork network("actionable_deadlock_cycle");

    Api::NetworkInput features = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("features")
                                     .dimensions({4})
                                     .dataType(Impl::DataType::FP32)
                                     .build();
    Api::CustomLayer custom = Api::CustomLayer::Builder()
                                  .network(network)
                                  .expression(makeTwoOutputExpression())
                                  .inputInterface({{"x", features.getFeatureOutput().value()}})
                                  .build();
    const Api::Tensor prediction = custom.getOutput("prediction");
    (void)Api::Stub::Builder().network(network).inputTensor(prediction).build();
    (void)Api::Stub::Builder().network(network).inputTensor(custom.getOutput("loss_only")).build();
    ASSERT_EQ(network.evaluateGraphForTest(false), Api::Network::StatusCode::SUCCESS);

    EXPECT_EQ(network.injectSelfCycleAndCheckForTest(custom.getId(), prediction),
              Api::Network::StatusCode::DEADLOCK_CYCLE);

    const std::string error = network.getLastGraphValidationError();
    EXPECT_NE(error.find("Thor graph validation failed: DEADLOCK CYCLE"), std::string::npos);
    EXPECT_NE(error.find("Cycle anchor: CustomLayer#"), std::string::npos);
    EXPECT_NE(error.find("output_name=\"prediction\""), std::string::npos);
    EXPECT_NE(error.find("Cycle path"), std::string::npos);
    EXPECT_NE(error.find("Break the feedback path"), std::string::npos);
}

TEST(GraphValidationDiagnostics, SuccessfulValidationClearsPreviousIssue) {
    GraphValidationDiagnosticsTestNetwork invalidNetwork("invalid_then_separate_success");
    Api::NetworkInput dangling = Api::NetworkInput::Builder()
                                     .network(invalidNetwork)
                                     .name("dangling")
                                     .dimensions({1})
                                     .dataType(Impl::DataType::FP32)
                                     .build();
    (void)dangling;
    ASSERT_EQ(invalidNetwork.evaluateGraphForTest(false), Api::Network::StatusCode::DANGLING_OUTPUT);
    ASSERT_FALSE(invalidNetwork.getLastGraphValidationError().empty());

    (void)Api::Stub::Builder().network(invalidNetwork).inputTensor(dangling.getFeatureOutput().value()).build();
    EXPECT_EQ(invalidNetwork.evaluateGraphForTest(false), Api::Network::StatusCode::SUCCESS);
    EXPECT_TRUE(invalidNetwork.getLastGraphValidationError().empty());
}

TEST(GraphValidationDiagnostics, ReportableMetricDiscoveryValidatesGraphOnceForManyMetrics) {
    CountingReportValidationNetwork network("reportable_metric_validation_once");
    Api::NetworkInput values = Api::NetworkInput::Builder()
                                   .network(network)
                                   .name("values")
                                   .dimensions({4})
                                   .dataType(Impl::DataType::FP32)
                                   .build();

    for (int i = 0; i < 4; ++i) {
        Api::Mean mean = Api::Mean::Builder().network(network).values(values.getFeatureOutput().value()).build();
        Api::NetworkOutput::Builder()
            .network(network)
            .name("mean_" + std::to_string(i))
            .inputTensor(mean.getMetric())
            .build();
    }

    const std::vector<Api::NetworkMetricReference> metrics = network.getReportableMetrics();
    EXPECT_EQ(metrics.size(), 4u);
    EXPECT_EQ(network.getTrainingValidationPasses(), 1u);
}

TEST(GraphValidationDiagnostics, ReportableLossDiscoveryValidatesGraphOnceForManyLosses) {
    CountingReportValidationNetwork network("reportable_loss_validation_once");
    Api::NetworkInput predictions = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("predictions")
                                        .dimensions({4})
                                        .dataType(Impl::DataType::FP32)
                                        .build();
    Api::NetworkInput labels = Api::NetworkInput::Builder()
                                   .network(network)
                                   .name("labels")
                                   .dimensions({4})
                                   .dataType(Impl::DataType::FP32)
                                   .build();
    Api::NetworkOutput::Builder()
        .network(network)
        .name("prediction")
        .inputTensor(predictions.getFeatureOutput().value())
        .build();

    for (int i = 0; i < 3; ++i) {
        Api::MSE loss = Api::MSE::Builder()
                            .network(network)
                            .predictions(predictions.getFeatureOutput().value())
                            .labels(labels.getFeatureOutput().value())
                            .reportsBatchLoss()
                            .build();
        Api::NetworkOutput::Builder()
            .network(network)
            .name("mse_" + std::to_string(i))
            .inputTensor(loss.getLoss())
            .build();
    }

    const std::vector<Api::NetworkLossReference> losses = network.getReportableLosses();
    EXPECT_EQ(losses.size(), 3u);
    EXPECT_EQ(network.getTrainingValidationPasses(), 1u);
}

TEST(GraphValidationDiagnostics, DeadlockCycleValidationHandlesDeepReconvergentDagWithoutPathEnumeration) {
    GraphValidationDiagnosticsTestNetwork network("deadlock_validation_reconvergent_dag");
    Api::NetworkInput values = Api::NetworkInput::Builder()
                                   .network(network)
                                   .name("values")
                                   .dimensions({4})
                                   .dataType(Impl::DataType::FP32)
                                   .build();

    // Every Add is a potential deadlock-cycle anchor.  Starting at an early
    // stage, the downstream graph contains exponentially many distinct paths
    // but only linearly many tensors.  Deadlock validation must memoize tensors
    // already proven unable to reach the current anchor instead of enumerating
    // all of those paths.
    Api::Tensor current = values.getFeatureOutput().value();
    for (int stage = 0; stage < 22; ++stage) {
        std::shared_ptr<Api::Activation> left =
            Api::Relu::Builder().network(network).featureInput(current).build();
        std::shared_ptr<Api::Activation> right =
            Api::Relu::Builder().network(network).featureInput(current).build();
        Api::Add joined = Api::Add::Builder()
                              .network(network)
                              .left(left->getFeatureOutput().value())
                              .right(right->getFeatureOutput().value())
                              .build();
        current = joined.getFeatureOutput().value();
    }
    (void)Api::Stub::Builder().network(network).inputTensor(current).build();

    EXPECT_EQ(network.evaluateGraphForTest(/*inferenceOnly=*/false), Api::Network::StatusCode::SUCCESS);
}

TEST(GraphValidationDiagnostics, ReportDiscoveryHandlesDeepReconvergentDagWithoutPathEnumeration) {
    CountingReportValidationNetwork network("report_discovery_reconvergent_dag");
    Api::NetworkInput values = Api::NetworkInput::Builder()
                                   .network(network)
                                   .name("values")
                                   .dimensions({4})
                                   .dataType(Impl::DataType::FP32)
                                   .build();

    // Each stage splits and reconverges. A recursion-stack-only traversal visits
    // the prefix once per distinct path (2^N); memoized DAG traversal visits each
    // tensor once. Keep this graph large enough to catch accidental path
    // enumeration while remaining tiny after memoization.
    Api::Tensor current = values.getFeatureOutput().value();
    for (int stage = 0; stage < 22; ++stage) {
        std::shared_ptr<Api::Activation> left =
            Api::Relu::Builder().network(network).featureInput(current).build();
        std::shared_ptr<Api::Activation> right =
            Api::Relu::Builder().network(network).featureInput(current).build();
        Api::Add joined = Api::Add::Builder()
                              .network(network)
                              .left(left->getFeatureOutput().value())
                              .right(right->getFeatureOutput().value())
                              .build();
        current = joined.getFeatureOutput().value();
    }

    Api::Mean mean = Api::Mean::Builder().network(network).values(current).build();
    Api::NetworkOutput::Builder().network(network).name("reconvergent_mean").inputTensor(mean.getMetric()).build();

    const Api::NetworkTrainingValidationSnapshot snapshot = network.validateTrainingGraphAndCollectReports();
    ASSERT_EQ(snapshot.reportableMetrics.size(), 1u);
    EXPECT_EQ(snapshot.reportableMetrics[0].metricName, "reconvergent_mean");
    ASSERT_TRUE(snapshot.reportableMetrics[0].inputSourceName.has_value());
    EXPECT_EQ(snapshot.reportableMetrics[0].inputSourceName.value(), "values");
    EXPECT_EQ(snapshot.reportableMetrics[0].requiredInputNames, (std::vector<std::string>{"values"}));
    EXPECT_EQ(network.getTrainingValidationPasses(), 1u);
}
