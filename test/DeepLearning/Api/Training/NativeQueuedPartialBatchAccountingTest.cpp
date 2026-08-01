#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Implementation/Data/Sessions/BatchSessionRuntimeAccess.h"
#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Metrics/Max.h"
#include "DeepLearning/Api/Layers/Metrics/Mean.h"
#include "DeepLearning/Api/Layers/Metrics/Min.h"
#include "DeepLearning/Api/Layers/Metrics/Sum.h"
#include "DeepLearning/Api/Layers/Metrics/WeightedMean.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunner.h"
#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace Thor;

namespace {

class ExactPopulationBatchSession final : public BatchSession {
   public:
    ExactPopulationBatchSession(uint64_t trainExamples,
                                uint64_t validateExamples,
                                uint64_t capacity)
        : BatchSession("partial_batch_accounting"),
          trainExamples(trainExamples),
          validateExamples(validateExamples) {
        batchSize = capacity;
    }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        const uint64_t examples = getNumExamples(exampleType);
        return examples == 0 ? 0 : (examples + batchSize - 1) / batchSize;
    }

    uint64_t getNumExamples(ExampleType exampleType) override {
        if (exampleType == ExampleType::TRAIN) {
            return trainExamples;
        }
        if (exampleType == ExampleType::VALIDATE) {
            return validateExamples;
        }
        return 0;
    }

    uint64_t getNextBatchNum(ExampleType exampleType) override {
        if (exampleType == ExampleType::TRAIN) {
            return nextTrainBatch;
        }
        if (exampleType == ExampleType::VALIDATE) {
            return nextValidateBatch;
        }
        return 0;
    }

   private:
    Batch acquireBatch(ExampleType exampleType, uint64_t& batchNum) override {
        const uint64_t examples = getNumExamples(exampleType);
        const uint64_t batches = getNumBatchesPerEpoch(exampleType);
        if (examples == 0 || batches == 0) {
            throw std::runtime_error(
                "ExactPopulationBatchSession cannot read an empty split.");
        }

        uint64_t& nextBatch = exampleType == ExampleType::TRAIN
                                  ? nextTrainBatch
                                  : nextValidateBatch;
        if (batchNum >= batches) {
            batchNum = nextBatch;
        }

        const uint64_t first = batchNum * batchSize;
        const bool wrapTail =
            usesWrappedBatchTailForRuntime();
        const uint64_t valid =
            wrapTail ? batchSize : std::min(batchSize, examples - first);
        uint64_t& nextLogical = exampleType == ExampleType::TRAIN
                                    ? nextTrainLogical
                                    : nextValidateLogical;
        const ThorImplementation::TensorPlacement cpu(
            ThorImplementation::TensorPlacement::MemDevices::CPU);
        ThorImplementation::Tensor predictions(
            cpu,
            ThorImplementation::TensorDescriptor(
                ThorImplementation::DataType::FP32, {batchSize, 1}));
        ThorImplementation::Tensor labels(
            cpu,
            ThorImplementation::TensorDescriptor(
                ThorImplementation::DataType::FP32, {batchSize, 1}));
        ThorImplementation::Tensor weights(
            cpu,
            ThorImplementation::TensorDescriptor(
                ThorImplementation::DataType::FP32, {batchSize, 1}));
        float* predictionValues = predictions.getMemPtr<float>();
        float* labelValues = labels.getMemPtr<float>();
        float* weightValues = weights.getMemPtr<float>();

        for (uint64_t row = 0; row < valid; ++row) {
            const uint64_t logicalExample = wrapTail
                ? std::exchange(nextLogical, (nextLogical + 1) % examples)
                : first + row;
            const uint64_t tailStart = examples - 2;
            predictionValues[row] = logicalExample >= tailStart ? 3.0f : 1.0f;
            labelValues[row] = 0.0f;
            weightValues[row] = logicalExample >= tailStart ? 10.0f : 1.0f;
        }
        for (uint64_t row = valid; row < batchSize; ++row) {
            predictionValues[row] = predictionValues[valid - 1];
            labelValues[row] = labelValues[valid - 1];
            weightValues[row] = weightValues[valid - 1];
        }

        nextBatch = (batchNum + 1) % batches;

        Batch batch;
        if (valid < batchSize) {
            batch.setValidExampleCount(static_cast<uint32_t>(valid));
        }
        batch.insert("predictions", predictions);
        batch.insert("labels", labels);
        batch.insert("weights", weights);
        return batch;
    }

    void recycleBatch(ExampleType, Batch&&) override {}

    void setBatchTailModeForRuntimeImpl(
        ThorImplementation::BatchTailMode mode) override {
        (void)mode;
        nextTrainLogical = 0;
        nextValidateLogical = 0;
    }

    uint64_t trainExamples = 0;
    uint64_t validateExamples = 0;
    uint64_t nextTrainBatch = 0;
    uint64_t nextValidateBatch = 0;
    uint64_t nextTrainLogical = 0;
    uint64_t nextValidateLogical = 0;
};

class CapturingObserver final : public TrainingObserver {
   public:
    void onTrainingEvent(const TrainingEvent& event) override {
        events.push_back(event);
    }

    std::vector<TrainingStatsSnapshot> stats(TrainingEventPhase phase) const {
        std::vector<TrainingStatsSnapshot> out;
        for (const TrainingEvent& event : events) {
            if (event.type == TrainingEventType::STATS &&
                event.stats.phase == phase) {
                out.push_back(event.stats);
            }
        }
        return out;
    }

    std::vector<TrainingEvent> events;
};

std::shared_ptr<Network> makeInputLossNetwork(bool requiresFullBatch = false) {
    auto network = std::make_shared<Network>("partial_batch_accounting");
    NetworkInput predictions = NetworkInput::Builder()
                                   .network(*network)
                                   .name("predictions")
                                   .dimensions({1})
                                   .dataType(DataType::FP32)
                                   .build();
    NetworkInput labels = NetworkInput::Builder()
                              .network(*network)
                              .name("labels")
                              .dimensions({1})
                              .dataType(DataType::FP32)
                              .build();
    NetworkInput weights = NetworkInput::Builder()
                               .network(*network)
                               .name("weights")
                               .dimensions({1})
                               .dataType(DataType::FP32)
                               .build();

    // FIT requires an active trainable parameter. Feed the all-zero labels through
    // a trainable linear branch, then add that identically-zero result to the
    // predictions used by the single reported objective. The parameter is on the
    // objective path but receives an exact zero gradient because its input is zero.
    std::shared_ptr<Initializer> zeroInitializer =
        UniformRandom::Builder().minValue(0.0f).maxValue(0.0f).build();
    FullyConnected trainableAnchor = FullyConnected::Builder()
                                         .network(*network)
                                         .featureInput(labels.getFeatureOutput().value())
                                         .numOutputFeatures(1)
                                         .hasBias(false)
                                         .weightsInitializer(zeroInitializer)
                                         .noActivation()
                                         .build();

    ThorImplementation::Expression predictionExpression =
        ThorImplementation::Expression::input(
            "predictions",
            ThorImplementation::DataType::FP32,
            ThorImplementation::DataType::FP32);
    ThorImplementation::Expression anchorExpression =
        ThorImplementation::Expression::input(
            "anchor",
            ThorImplementation::DataType::FP32,
            ThorImplementation::DataType::FP32);
    ThorImplementation::ExpressionDefinition combinedDefinition =
        ThorImplementation::ExpressionDefinition::fromOutputs(
            ThorImplementation::Expression::outputs(
                {{"combined", predictionExpression + anchorExpression}}));
    CustomLayer::Builder combinedBuilder;
    combinedBuilder
        .network(*network)
        .expression(ThorImplementation::DynamicExpression::fromExpressionDefinition(
            combinedDefinition))
        .inputNames({"predictions", "anchor"})
        .outputNames({"combined"})
        .inputInterface(
            {{"predictions", predictions.getFeatureOutput().value()},
             {"anchor", trainableAnchor.getFeatureOutput().value()}});
    if (requiresFullBatch) {
        combinedBuilder.requiresFullBatch();
    }
    CustomLayer combinedPrediction = combinedBuilder.build();

    Mean predictionMean = Mean::Builder()
                              .network(*network)
                              .values(combinedPrediction.getOutput("combined"))
                              .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction_mean")
        .inputTensor(predictionMean.getMetric())
        .dataType(DataType::FP32)
        .build();

    Sum predictionSum = Sum::Builder()
                            .network(*network)
                            .values(combinedPrediction.getOutput("combined"))
                            .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction_sum")
        .inputTensor(predictionSum.getMetric())
        .dataType(DataType::FP32)
        .build();

    Min predictionMin = Min::Builder()
                            .network(*network)
                            .values(combinedPrediction.getOutput("combined"))
                            .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction_min")
        .inputTensor(predictionMin.getMetric())
        .dataType(DataType::FP32)
        .build();

    Max predictionMax = Max::Builder()
                            .network(*network)
                            .values(combinedPrediction.getOutput("combined"))
                            .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction_max")
        .inputTensor(predictionMax.getMetric())
        .dataType(DataType::FP32)
        .build();

    WeightedMean predictionWeightedMean =
        WeightedMean::Builder()
            .network(*network)
            .values(combinedPrediction.getOutput("combined"))
            .weights(weights.getFeatureOutput().value())
            .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction_weighted_mean")
        .inputTensor(predictionWeightedMean.getMetric())
        .dataType(DataType::FP32)
        .build();

    MSE loss = MSE::Builder()
                   .network(*network)
                   .predictions(combinedPrediction.getOutput("combined"))
                   .labels(labels.getFeatureOutput().value())
                   .reportsBatchLoss()
                   .lossDataType(DataType::FP32)
                   .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("loss")
        .inputTensor(loss.getLoss())
        .dataType(DataType::FP32)
        .build();
    return network;
}

std::vector<uint64_t> fieldValues(
    const std::vector<TrainingStatsSnapshot>& stats,
    uint64_t TrainingStatsSnapshot::*field) {
    std::vector<uint64_t> values;
    values.reserve(stats.size());
    for (const TrainingStatsSnapshot& snapshot : stats) {
        values.push_back(snapshot.*field);
    }
    return values;
}

}  // namespace

TEST(NativeQueuedPartialBatchAccounting,
     ExactEpochsReportValidSamplesAndPopulationWeightedLosses) {
    auto session =
        std::make_shared<ExactPopulationBatchSession>(10, 6, 4);
    auto network = makeInputLossNetwork();

    std::vector<TrainingModelSelectionContext> selectionContexts;
    TrainingRunRequest request;
    request.network = network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder().initialLearningRate(0.01f).build();
    request.datasetInputBindings = {
        TrainingInputBinding("predictions", "predictions"),
        TrainingInputBinding("labels", "labels"),
        TrainingInputBinding("weights", "weights")};
    request.runtime.scalarTensorsToReport = {
        "loss",
        "prediction_max",
        "prediction_mean",
        "prediction_min",
        "prediction_sum",
        "prediction_weighted_mean"};
    request.epochs = 2;
    request.checkBestModelEveryEpochs = 1;
    request.modelSelectionScore = TrainingModelSelectionScore(
        [&selectionContexts](const TrainingModelSelectionContext& context) {
            selectionContexts.push_back(context);
            return context.validationLoss();
        });

    CapturingObserver observer;
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{
            .maxInFlightBatches = 3,
            .synchronizeAfterEveryBatch = false});

    const std::vector<TrainingStatsSnapshot> train =
        observer.stats(TrainingEventPhase::TRAIN);
    const std::vector<TrainingStatsSnapshot> validate =
        observer.stats(TrainingEventPhase::VALIDATE);
    ASSERT_EQ(train.size(), 6u);
    ASSERT_EQ(validate.size(), 4u);

    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::validExamplesInBatch),
              (std::vector<uint64_t>{4, 4, 2, 4, 4, 2}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessedInEpoch),
              (std::vector<uint64_t>{4, 8, 10, 4, 8, 10}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessed),
              (std::vector<uint64_t>{4, 8, 10, 14, 18, 20}));

    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::validExamplesInBatch),
              (std::vector<uint64_t>{4, 2, 4, 2}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::samplesProcessedInEpoch),
              (std::vector<uint64_t>{4, 6, 4, 6}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::samplesProcessed),
              (std::vector<uint64_t>{4, 6, 10, 12}));

    ASSERT_GE(selectionContexts.size(), 2u);
    for (const TrainingModelSelectionContext& context : selectionContexts) {
        ASSERT_TRUE(context.train.loss.has_value());
        ASSERT_TRUE(context.validate.loss.has_value());
        EXPECT_NEAR(context.train.loss.value(), 2.6, 1e-5);
        EXPECT_NEAR(context.validate.loss.value(), 11.0 / 3.0, 1e-5);
        ASSERT_EQ(context.train.metrics.count("prediction_mean"), 1u);
        ASSERT_EQ(context.validate.metrics.count("prediction_mean"), 1u);
        EXPECT_NEAR(context.train.metrics.at("prediction_mean"), 1.4, 1e-5);
        EXPECT_NEAR(context.validate.metrics.at("prediction_mean"), 5.0 / 3.0, 1e-5);
        EXPECT_NEAR(context.train.metrics.at("prediction_sum"), 14.0, 1e-5);
        EXPECT_NEAR(context.validate.metrics.at("prediction_sum"), 10.0, 1e-5);
        EXPECT_NEAR(context.train.metrics.at("prediction_min"), 1.0, 1e-5);
        EXPECT_NEAR(context.validate.metrics.at("prediction_min"), 1.0, 1e-5);
        EXPECT_NEAR(context.train.metrics.at("prediction_max"), 3.0, 1e-5);
        EXPECT_NEAR(context.validate.metrics.at("prediction_max"), 3.0, 1e-5);
        EXPECT_NEAR(context.train.metrics.at("prediction_weighted_mean"), 68.0 / 28.0, 1e-5);
        EXPECT_NEAR(context.validate.metrics.at("prediction_weighted_mean"), 64.0 / 24.0, 1e-5);
    }

    ASSERT_EQ(train.front().metricBatchStats.count("prediction_sum"), 1u);
    EXPECT_EQ(train.front().metricBatchStats.at("prediction_sum").aggregation,
              MetricAggregation::SUM);
    ASSERT_EQ(train[2].metricBatchStats.count("prediction_weighted_mean"), 1u);
    const MetricBatchStat& tailWeighted =
        train[2].metricBatchStats.at("prediction_weighted_mean");
    EXPECT_EQ(tailWeighted.aggregation, MetricAggregation::RATIO);
    ASSERT_TRUE(tailWeighted.numerator.has_value());
    ASSERT_TRUE(tailWeighted.denominator.has_value());
    EXPECT_NEAR(tailWeighted.numerator.value(), 60.0, 1e-5);
    EXPECT_NEAR(tailWeighted.denominator.value(), 20.0, 1e-5);

    EXPECT_EQ(session->getNextBatchNum(ExampleType::TRAIN), 0u);
    EXPECT_EQ(session->getNextBatchNum(ExampleType::VALIDATE), 0u);
}

TEST(NativeQueuedPartialBatchAccounting,
     FullBatchOnlyLayerFallsBackToContinuousWrappedEpochs) {
    auto session =
        std::make_shared<ExactPopulationBatchSession>(10, 6, 4);

    TrainingRunRequest request;
    request.network = makeInputLossNetwork(/*requiresFullBatch=*/true);
    request.batchSession = session;
    request.optimizer = Sgd::Builder().initialLearningRate(0.01f).build();
    request.datasetInputBindings = {
        TrainingInputBinding("predictions", "predictions"),
        TrainingInputBinding("labels", "labels"),
        TrainingInputBinding("weights", "weights")};
    request.runtime.scalarTensorsToReport = {"loss", "prediction_mean"};
    request.epochs = 1;

    CapturingObserver observer;
    testing::internal::CaptureStderr();
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{
            .maxInFlightBatches = 3,
            .synchronizeAfterEveryBatch = false});
    const std::string warning = testing::internal::GetCapturedStderr();

    EXPECT_NE(warning.find("exact partial tail batches are not compatible"),
              std::string::npos);
    EXPECT_NE(warning.find("CustomLayer"), std::string::npos);
    EXPECT_NE(warning.find("legacy wrapped full-batch epochs"),
              std::string::npos);
    EXPECT_EQ(ThorImplementation::BatchSessionRuntimeAccess::getTailMode(*session),
              ThorImplementation::BatchTailMode::WRAP);
    EXPECT_EQ(ThorImplementation::BatchSessionRuntimeAccess::examplesProcessedPerEpoch(
                  *session, ExampleType::TRAIN),
              12u);
    EXPECT_EQ(ThorImplementation::BatchSessionRuntimeAccess::examplesProcessedPerEpoch(
                  *session, ExampleType::VALIDATE),
              8u);

    const std::vector<TrainingStatsSnapshot> train =
        observer.stats(TrainingEventPhase::TRAIN);
    const std::vector<TrainingStatsSnapshot> validate =
        observer.stats(TrainingEventPhase::VALIDATE);
    ASSERT_EQ(train.size(), 3u);
    ASSERT_EQ(validate.size(), 2u);
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::validExamplesInBatch),
              (std::vector<uint64_t>{4, 4, 4}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessedInEpoch),
              (std::vector<uint64_t>{4, 8, 12}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessed),
              (std::vector<uint64_t>{4, 8, 12}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::validExamplesInBatch),
              (std::vector<uint64_t>{4, 4}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::samplesProcessedInEpoch),
              (std::vector<uint64_t>{4, 8}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::samplesProcessed),
              (std::vector<uint64_t>{4, 8}));

    ASSERT_TRUE(train.back().loss.has_value());
    ASSERT_TRUE(train.back().metrics.count("prediction_mean"));
    EXPECT_NEAR(train.back().loss.value(), 5.0, 1e-5);
    EXPECT_NEAR(train.back().metrics.at("prediction_mean"), 2.0, 1e-5);
    ASSERT_TRUE(validate.back().loss.has_value());
    ASSERT_TRUE(validate.back().metrics.count("prediction_mean"));
    EXPECT_NEAR(validate.back().loss.value(), 5.0, 1e-5);
    EXPECT_NEAR(validate.back().metrics.at("prediction_mean"), 2.0, 1e-5);
}

TEST(NativeQueuedPartialBatchAccounting,
     CappedTrainingWorkQuantaContinueAcrossPopulationBoundaries) {
    auto session =
        std::make_shared<ExactPopulationBatchSession>(10, 6, 4);

    TrainingRunRequest request;
    request.network = makeInputLossNetwork();
    request.batchSession = session;
    request.optimizer = Sgd::Builder().initialLearningRate(0.01f).build();
    request.datasetInputBindings = {
        TrainingInputBinding("predictions", "predictions"),
        TrainingInputBinding("labels", "labels"),
        TrainingInputBinding("weights", "weights")};
    request.runtime.scalarTensorsToReport = {"loss", "prediction_mean"};
    request.epochs = 2;
    request.maxTrainingBatchesPerEpoch = 2;

    CapturingObserver observer;
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{
            .maxInFlightBatches = 3,
            .synchronizeAfterEveryBatch = false});

    const std::vector<TrainingStatsSnapshot> train =
        observer.stats(TrainingEventPhase::TRAIN);
    ASSERT_EQ(train.size(), 4u);
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::validExamplesInBatch),
              (std::vector<uint64_t>{4, 4, 2, 4}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessedInEpoch),
              (std::vector<uint64_t>{4, 8, 2, 6}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessed),
              (std::vector<uint64_t>{4, 8, 10, 14}));
    EXPECT_EQ(session->getNextBatchNum(ExampleType::TRAIN), 1u);
    EXPECT_EQ(session->getNextBatchNum(ExampleType::VALIDATE), 0u);
}
