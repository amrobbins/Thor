#include "DeepLearning/Api/Data/BatchSession.h"
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
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunner.h"
#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Implementation/Data/Sessions/BatchSessionRuntimeAccess.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
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
    ExactPopulationBatchSession(uint64_t trainExamples, uint64_t validateExamples, uint64_t capacity)
        : BatchSession("partial_batch_accounting"), trainExamples(trainExamples), validateExamples(validateExamples) {
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
            throw std::runtime_error("ExactPopulationBatchSession cannot read an empty split.");
        }

        uint64_t& nextBatch = exampleType == ExampleType::TRAIN ? nextTrainBatch : nextValidateBatch;
        if (batchNum >= batches) {
            batchNum = nextBatch;
        }

        const uint64_t first = batchNum * batchSize;
        const bool wrapTail = usesWrappedBatchTailForRuntime();
        const uint64_t valid = wrapTail ? batchSize : std::min(batchSize, examples - first);
        uint64_t& nextLogical = exampleType == ExampleType::TRAIN ? nextTrainLogical : nextValidateLogical;
        const ThorImplementation::TensorPlacement cpu(ThorImplementation::TensorPlacement::MemDevices::CPU);
        ThorImplementation::Tensor predictions(cpu,
                                               ThorImplementation::TensorDescriptor(ThorImplementation::DataType::FP32, {batchSize, 1}));
        ThorImplementation::Tensor labels(cpu, ThorImplementation::TensorDescriptor(ThorImplementation::DataType::FP32, {batchSize, 1}));
        ThorImplementation::Tensor weights(cpu, ThorImplementation::TensorDescriptor(ThorImplementation::DataType::FP32, {batchSize, 1}));
        float* predictionValues = predictions.getMemPtr<float>();
        float* labelValues = labels.getMemPtr<float>();
        float* weightValues = weights.getMemPtr<float>();

        for (uint64_t row = 0; row < valid; ++row) {
            const uint64_t logicalExample = wrapTail ? std::exchange(nextLogical, (nextLogical + 1) % examples) : first + row;
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

    void setBatchTailModeForRuntimeImpl(ThorImplementation::BatchTailMode mode) override {
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
    void onTrainingEvent(const TrainingEvent& event) override { events.push_back(event); }

    std::vector<TrainingStatsSnapshot> stats(TrainingEventPhase phase) const {
        std::vector<TrainingStatsSnapshot> out;
        for (const TrainingEvent& event : events) {
            if (event.type == TrainingEventType::STATS && event.stats.phase == phase) {
                out.push_back(event.stats);
            }
        }
        return out;
    }

    std::vector<TrainingEvent> events;
};

std::shared_ptr<Network> makeInputLossNetwork(bool requiresFullBatch = false) {
    auto network = std::make_shared<Network>("partial_batch_accounting");
    NetworkInput predictions =
        NetworkInput::Builder().network(*network).name("predictions").dimensions({1}).dataType(DataType::FP32).build();
    NetworkInput labels = NetworkInput::Builder().network(*network).name("labels").dimensions({1}).dataType(DataType::FP32).build();
    NetworkInput weights = NetworkInput::Builder().network(*network).name("weights").dimensions({1}).dataType(DataType::FP32).build();

    // FIT requires an active trainable parameter. Feed the all-zero labels through
    // a trainable linear branch, then add that identically-zero result to the
    // predictions used by the single reported objective. The parameter is on the
    // objective path but receives an exact zero gradient because its input is zero.
    std::shared_ptr<Initializer> zeroInitializer = UniformRandom::Builder().minValue(0.0f).maxValue(0.0f).build();
    FullyConnected trainableAnchor = FullyConnected::Builder()
                                         .network(*network)
                                         .featureInput(labels.getFeatureOutput().value())
                                         .numOutputFeatures(1)
                                         .hasBias(false)
                                         .weightsInitializer(zeroInitializer)
                                         .noActivation()
                                         .build();

    ThorImplementation::Expression predictionExpression =
        ThorImplementation::Expression::input("predictions", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::Expression anchorExpression =
        ThorImplementation::Expression::input("anchor", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::ExpressionDefinition combinedDefinition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{"combined", predictionExpression + anchorExpression}}));
    CustomLayer::Builder combinedBuilder;
    combinedBuilder.network(*network)
        .expression(ThorImplementation::DynamicExpression::fromExpressionDefinition(combinedDefinition))
        .inputNames({"predictions", "anchor"})
        .outputNames({"combined"})
        .inputInterface({{"predictions", predictions.getFeatureOutput().value()}, {"anchor", trainableAnchor.getFeatureOutput().value()}});
    if (requiresFullBatch) {
        combinedBuilder.requiresFullBatch();
    }
    CustomLayer combinedPrediction = combinedBuilder.build();

    Mean predictionMean = Mean::Builder().network(*network).values(combinedPrediction.getOutput("combined")).build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction_mean")
        .inputTensor(predictionMean.getMetric())
        .dataType(DataType::FP32)
        .build();

    Sum predictionSum = Sum::Builder().network(*network).values(combinedPrediction.getOutput("combined")).build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction_sum")
        .inputTensor(predictionSum.getMetric())
        .dataType(DataType::FP32)
        .build();

    Min predictionMin = Min::Builder().network(*network).values(combinedPrediction.getOutput("combined")).build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction_min")
        .inputTensor(predictionMin.getMetric())
        .dataType(DataType::FP32)
        .build();

    Max predictionMax = Max::Builder().network(*network).values(combinedPrediction.getOutput("combined")).build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction_max")
        .inputTensor(predictionMax.getMetric())
        .dataType(DataType::FP32)
        .build();

    WeightedMean predictionWeightedMean = WeightedMean::Builder()
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
    NetworkOutput::Builder().network(*network).name("loss").inputTensor(loss.getLoss()).dataType(DataType::FP32).build();
    return network;
}

class DeterministicTrainableBatchSession final : public BatchSession {
   public:
    DeterministicTrainableBatchSession() : BatchSession("native_queue_trainable_oracle") { batchSize = 2; }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        return exampleType == ExampleType::TRAIN ? 3 : (exampleType == ExampleType::VALIDATE ? 2 : 0);
    }

    uint64_t getNumExamples(ExampleType exampleType) override { return getNumBatchesPerEpoch(exampleType) * batchSize; }

    uint64_t getNextBatchNum(ExampleType exampleType) override {
        if (exampleType == ExampleType::TRAIN)
            return nextTrainBatch;
        if (exampleType == ExampleType::VALIDATE)
            return nextValidateBatch;
        return 0;
    }

   private:
    Batch acquireBatch(ExampleType exampleType, uint64_t& batchNum) override {
        static const std::vector<float> trainFeatures{1.0f, -2.0f, 3.0f, 0.5f, -4.0f, 2.5f};
        static const std::vector<float> trainLabels{2.0f, 1.0f, -1.0f, 3.0f, -2.0f, 0.25f};
        static const std::vector<float> validateFeatures{1.5f, -0.75f, 2.0f, -3.0f};
        static const std::vector<float> validateLabels{0.5f, -1.5f, 2.5f, 1.0f};

        const std::vector<float>* features = nullptr;
        const std::vector<float>* labels = nullptr;
        uint64_t* nextBatch = nullptr;
        if (exampleType == ExampleType::TRAIN) {
            features = &trainFeatures;
            labels = &trainLabels;
            nextBatch = &nextTrainBatch;
        } else if (exampleType == ExampleType::VALIDATE) {
            features = &validateFeatures;
            labels = &validateLabels;
            nextBatch = &nextValidateBatch;
        } else {
            throw std::runtime_error("DeterministicTrainableBatchSession supports only train/validate splits.");
        }

        const uint64_t numBatches = getNumBatchesPerEpoch(exampleType);
        if (batchNum >= numBatches)
            batchNum = *nextBatch;
        const uint64_t first = batchNum * batchSize;

        const ThorImplementation::TensorPlacement cpu(ThorImplementation::TensorPlacement::MemDevices::CPU);
        ThorImplementation::Tensor featuresTensor(cpu,
                                                  ThorImplementation::TensorDescriptor(ThorImplementation::DataType::FP32, {batchSize, 1}));
        ThorImplementation::Tensor labelsTensor(cpu,
                                                ThorImplementation::TensorDescriptor(ThorImplementation::DataType::FP32, {batchSize, 1}));
        for (uint64_t row = 0; row < batchSize; ++row) {
            featuresTensor.getMemPtr<float>()[row] = features->at(first + row);
            labelsTensor.getMemPtr<float>()[row] = labels->at(first + row);
        }

        *nextBatch = (batchNum + 1) % numBatches;
        Batch batch;
        batch.insert("features", featuresTensor);
        batch.insert("labels", labelsTensor);
        return batch;
    }

    void recycleBatch(ExampleType, Batch&&) override {}

    uint64_t nextTrainBatch = 0;
    uint64_t nextValidateBatch = 0;
};

struct TrainableOracleNetwork {
    std::shared_ptr<Network> network;
    uint64_t fullyConnectedLayerId = 0;
};

TrainableOracleNetwork makeTrainableOracleNetwork() {
    auto network = std::make_shared<Network>("native_queue_trainable_oracle");
    NetworkInput features = NetworkInput::Builder().network(*network).name("features").dimensions({1}).dataType(DataType::FP32).build();
    NetworkInput labels = NetworkInput::Builder().network(*network).name("labels").dimensions({1}).dataType(DataType::FP32).build();
    std::shared_ptr<Initializer> initializer = UniformRandom::Builder().minValue(0.25f).maxValue(0.25f).build();
    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(*network)
                                        .featureInput(features.getFeatureOutput().value())
                                        .numOutputFeatures(1)
                                        .hasBias(false)
                                        .weightsInitializer(initializer)
                                        .computeDataType(DataType::FP32)
                                        .outputDataType(DataType::FP32)
                                        .noActivation()
                                        .build();
    Thor::MSE loss = Thor::MSE::Builder()
                         .network(*network)
                         .predictions(fullyConnected.getFeatureOutput().value())
                         .labels(labels.getFeatureOutput().value())
                         .reportsBatchLoss()
                         .lossDataType(DataType::FP32)
                         .build();
    NetworkOutput::Builder().network(*network).name("loss").inputTensor(loss.getLoss()).dataType(DataType::FP32).build();
    return {network, fullyConnected.getId()};
}

struct TrainableOracleResult {
    std::vector<double> trainLosses;
    std::vector<double> validateLosses;
    float finalWeight = 0.0f;
};

TrainableOracleResult runTrainableOracle(const NativeQueuedTrainingOptions& options) {
    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto session = std::make_shared<DeterministicTrainableBatchSession>();
    std::shared_ptr<PlacedNetwork> completedPlacedNetwork;

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder().initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();
    request.datasetInputBindings = {TrainingInputBinding("features", "features"), TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 3;
    request.completedPlacedNetwork = &completedPlacedNetwork;

    CapturingObserver observer;
    runNativeQueuedTraining(request, observer, options);
    THOR_THROW_IF_FALSE(completedPlacedNetwork != nullptr);
    completedPlacedNetwork->synchronize();

    auto collectLosses = [&](TrainingEventPhase phase) {
        std::vector<double> losses;
        for (const TrainingStatsSnapshot& snapshot : observer.stats(phase)) {
            THOR_THROW_IF_FALSE(snapshot.loss.has_value());
            losses.push_back(snapshot.loss.value());
        }
        return losses;
    };

    std::shared_ptr<ThorImplementation::TrainableLayer> physicalFullyConnected =
        std::dynamic_pointer_cast<ThorImplementation::TrainableLayer>(
            completedPlacedNetwork->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fixture.fullyConnectedLayerId));
    THOR_THROW_IF_FALSE(physicalFullyConnected != nullptr);
    std::shared_ptr<ThorImplementation::PhysicalParameter> weights = physicalFullyConnected->getParameter("weights");
    THOR_THROW_IF_FALSE(weights != nullptr);
    THOR_THROW_IF_FALSE(weights->getStorage().has_value());

    const ThorImplementation::TensorPlacement cpu(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor weightCpu = weights->getStorage().value().clone(cpu);
    Stream downloadStream = Stream::getNextDownloadStream(0);
    weightCpu.copyFromAsync(weights->getStorage().value(), downloadStream);
    downloadStream.synchronize();

    return {
        collectLosses(TrainingEventPhase::TRAIN),
        collectLosses(TrainingEventPhase::VALIDATE),
        weightCpu.getMemPtr<float>()[0],
    };
}

std::vector<uint64_t> fieldValues(const std::vector<TrainingStatsSnapshot>& stats, uint64_t TrainingStatsSnapshot::* field) {
    std::vector<uint64_t> values;
    values.reserve(stats.size());
    for (const TrainingStatsSnapshot& snapshot : stats) {
        values.push_back(snapshot.*field);
    }
    return values;
}

}  // namespace

TEST(NativeQueuedPartialBatchAccounting, ExactEpochsReportValidSamplesAndPopulationWeightedLosses) {
    auto session = std::make_shared<ExactPopulationBatchSession>(10, 6, 4);
    auto network = makeInputLossNetwork();

    std::vector<TrainingModelSelectionContext> selectionContexts;
    TrainingRunRequest request;
    request.network = network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder().initialLearningRate(0.01f).build();
    request.datasetInputBindings = {TrainingInputBinding("predictions", "predictions"),
                                    TrainingInputBinding("labels", "labels"),
                                    TrainingInputBinding("weights", "weights")};
    request.runtime.scalarTensorsToReport = {
        "loss", "prediction_max", "prediction_mean", "prediction_min", "prediction_sum", "prediction_weighted_mean"};
    request.epochs = 2;
    request.checkBestModelEveryEpochs = 1;
    request.modelSelectionScore = TrainingModelSelectionScore([&selectionContexts](const TrainingModelSelectionContext& context) {
        selectionContexts.push_back(context);
        return context.validationLoss();
    });

    CapturingObserver observer;
    runNativeQueuedTraining(request, observer, NativeQueuedTrainingOptions{.maxInFlightBatches = 3, .synchronizeAfterEveryBatch = false});

    const std::vector<TrainingStatsSnapshot> train = observer.stats(TrainingEventPhase::TRAIN);
    const std::vector<TrainingStatsSnapshot> validate = observer.stats(TrainingEventPhase::VALIDATE);
    ASSERT_EQ(train.size(), 6u);
    ASSERT_EQ(validate.size(), 4u);

    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::validExamplesInBatch), (std::vector<uint64_t>{4, 4, 2, 4, 4, 2}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessedInEpoch), (std::vector<uint64_t>{4, 8, 10, 4, 8, 10}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessed), (std::vector<uint64_t>{4, 8, 10, 14, 18, 20}));

    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::validExamplesInBatch), (std::vector<uint64_t>{4, 2, 4, 2}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::samplesProcessedInEpoch), (std::vector<uint64_t>{4, 6, 4, 6}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::samplesProcessed), (std::vector<uint64_t>{4, 6, 10, 12}));

    ASSERT_GE(selectionContexts.size(), 3u);
    size_t phaseEntrySelectionContexts = 0;
    for (const TrainingModelSelectionContext& context : selectionContexts) {
        ASSERT_TRUE(context.validate.loss.has_value());
        EXPECT_NEAR(context.validate.loss.value(), 11.0 / 3.0, 1e-5);
        ASSERT_EQ(context.validate.metrics.count("prediction_mean"), 1u);
        EXPECT_NEAR(context.validate.metrics.at("prediction_mean"), 5.0 / 3.0, 1e-5);
        EXPECT_NEAR(context.validate.metrics.at("prediction_sum"), 10.0, 1e-5);
        EXPECT_NEAR(context.validate.metrics.at("prediction_min"), 1.0, 1e-5);
        EXPECT_NEAR(context.validate.metrics.at("prediction_max"), 3.0, 1e-5);
        EXPECT_NEAR(context.validate.metrics.at("prediction_weighted_mean"), 64.0 / 24.0, 1e-5);

        if (!context.train.loss.has_value()) {
            // firstModelSelectionEpoch defaults to zero, so model selection first
            // sees the phase-entry state before any optimizer update.  That
            // context is deliberately validation-only; there is no train loss or
            // train metric population yet.
            ++phaseEntrySelectionContexts;
            EXPECT_EQ(context.epoch, 0u);
            EXPECT_TRUE(context.train.metrics.empty());
            continue;
        }

        EXPECT_NEAR(context.train.loss.value(), 2.6, 1e-5);
        ASSERT_EQ(context.train.metrics.count("prediction_mean"), 1u);
        EXPECT_NEAR(context.train.metrics.at("prediction_mean"), 1.4, 1e-5);
        EXPECT_NEAR(context.train.metrics.at("prediction_sum"), 14.0, 1e-5);
        EXPECT_NEAR(context.train.metrics.at("prediction_min"), 1.0, 1e-5);
        EXPECT_NEAR(context.train.metrics.at("prediction_max"), 3.0, 1e-5);
        EXPECT_NEAR(context.train.metrics.at("prediction_weighted_mean"), 68.0 / 28.0, 1e-5);
    }
    EXPECT_EQ(phaseEntrySelectionContexts, 1u);

    ASSERT_EQ(train.front().metricBatchStats.count("prediction_sum"), 1u);
    EXPECT_EQ(train.front().metricBatchStats.at("prediction_sum").aggregation, MetricAggregation::SUM);

    auto expectWeightedRatio = [](const TrainingStatsSnapshot& snapshot, double expectedNumerator, double expectedDenominator) {
        ASSERT_EQ(snapshot.metricBatchStats.count("prediction_weighted_mean"), 1u);
        const MetricBatchStat& weighted = snapshot.metricBatchStats.at("prediction_weighted_mean");
        EXPECT_EQ(weighted.aggregation, MetricAggregation::RATIO);
        ASSERT_TRUE(weighted.numerator.has_value());
        ASSERT_TRUE(weighted.denominator.has_value());
        EXPECT_NEAR(weighted.numerator.value(), expectedNumerator, 1e-5);
        EXPECT_NEAR(weighted.denominator.value(), expectedDenominator, 1e-5);
    };

    // Assert the per-batch sufficient statistics as well as the final epoch
    // aggregate. In particular, validation batch 0 must remain 4/4 while the
    // queued tail batch is 60/20; if batch 0's shared values/weights are
    // overwritten by the tail, the final ratio collapses to exactly 3.0.
    for (size_t epochOffset : {size_t{0}, size_t{3}}) {
        expectWeightedRatio(train[epochOffset + 0], 4.0, 4.0);
        expectWeightedRatio(train[epochOffset + 1], 4.0, 4.0);
        expectWeightedRatio(train[epochOffset + 2], 60.0, 20.0);
    }
    for (size_t epochOffset : {size_t{0}, size_t{2}}) {
        expectWeightedRatio(validate[epochOffset + 0], 4.0, 4.0);
        expectWeightedRatio(validate[epochOffset + 1], 60.0, 20.0);
    }

    EXPECT_EQ(session->getNextBatchNum(ExampleType::TRAIN), 0u);
    EXPECT_EQ(session->getNextBatchNum(ExampleType::VALIDATE), 0u);
}

TEST(NativeQueuedPartialBatchAccounting, QueuedTrainingMatchesSynchronizedReferenceForParameterUpdates) {
    const TrainableOracleResult reference =
        runTrainableOracle(NativeQueuedTrainingOptions{.maxInFlightBatches = 1, .synchronizeAfterEveryBatch = true});
    const TrainableOracleResult queued =
        runTrainableOracle(NativeQueuedTrainingOptions{.maxInFlightBatches = 3, .synchronizeAfterEveryBatch = false});

    ASSERT_EQ(queued.trainLosses.size(), reference.trainLosses.size());
    ASSERT_EQ(queued.validateLosses.size(), reference.validateLosses.size());
    for (size_t i = 0; i < reference.trainLosses.size(); ++i) {
        EXPECT_NEAR(queued.trainLosses[i], reference.trainLosses[i], 1e-6) << "train batch " << i;
    }
    for (size_t i = 0; i < reference.validateLosses.size(); ++i) {
        EXPECT_NEAR(queued.validateLosses[i], reference.validateLosses[i], 1e-6) << "validate batch " << i;
    }
    EXPECT_NEAR(queued.finalWeight, reference.finalWeight, 1e-6);
}

TEST(NativeQueuedPartialBatchAccounting, FullBatchOnlyLayerFallsBackToContinuousWrappedEpochs) {
    auto session = std::make_shared<ExactPopulationBatchSession>(10, 6, 4);

    TrainingRunRequest request;
    request.network = makeInputLossNetwork(/*requiresFullBatch=*/true);
    request.batchSession = session;
    request.optimizer = Sgd::Builder().initialLearningRate(0.01f).build();
    request.datasetInputBindings = {TrainingInputBinding("predictions", "predictions"),
                                    TrainingInputBinding("labels", "labels"),
                                    TrainingInputBinding("weights", "weights")};
    request.runtime.scalarTensorsToReport = {"loss", "prediction_mean"};
    request.epochs = 1;

    CapturingObserver observer;
    testing::internal::CaptureStderr();
    runNativeQueuedTraining(request, observer, NativeQueuedTrainingOptions{.maxInFlightBatches = 3, .synchronizeAfterEveryBatch = false});
    const std::string warning = testing::internal::GetCapturedStderr();

    EXPECT_NE(warning.find("exact partial tail batches are not compatible"), std::string::npos);
    EXPECT_NE(warning.find("CustomLayer"), std::string::npos);
    EXPECT_NE(warning.find("legacy wrapped full-batch epochs"), std::string::npos);
    EXPECT_EQ(ThorImplementation::BatchSessionRuntimeAccess::getTailMode(*session), ThorImplementation::BatchTailMode::WRAP);
    EXPECT_EQ(ThorImplementation::BatchSessionRuntimeAccess::examplesProcessedPerEpoch(*session, ExampleType::TRAIN), 12u);
    EXPECT_EQ(ThorImplementation::BatchSessionRuntimeAccess::examplesProcessedPerEpoch(*session, ExampleType::VALIDATE), 8u);

    const std::vector<TrainingStatsSnapshot> train = observer.stats(TrainingEventPhase::TRAIN);
    const std::vector<TrainingStatsSnapshot> validate = observer.stats(TrainingEventPhase::VALIDATE);
    ASSERT_EQ(train.size(), 3u);
    ASSERT_EQ(validate.size(), 2u);
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::validExamplesInBatch), (std::vector<uint64_t>{4, 4, 4}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessedInEpoch), (std::vector<uint64_t>{4, 8, 12}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessed), (std::vector<uint64_t>{4, 8, 12}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::validExamplesInBatch), (std::vector<uint64_t>{4, 4}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::samplesProcessedInEpoch), (std::vector<uint64_t>{4, 8}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::samplesProcessed), (std::vector<uint64_t>{4, 8}));

    ASSERT_TRUE(train.back().loss.has_value());
    ASSERT_TRUE(train.back().metrics.count("prediction_mean"));
    EXPECT_NEAR(train.back().loss.value(), 5.0, 1e-5);
    EXPECT_NEAR(train.back().metrics.at("prediction_mean"), 2.0, 1e-5);
    ASSERT_TRUE(validate.back().loss.has_value());
    ASSERT_TRUE(validate.back().metrics.count("prediction_mean"));
    EXPECT_NEAR(validate.back().loss.value(), 5.0, 1e-5);
    EXPECT_NEAR(validate.back().metrics.at("prediction_mean"), 2.0, 1e-5);
}

TEST(NativeQueuedPartialBatchAccounting, CappedTrainingWorkQuantaContinueAcrossPopulationBoundaries) {
    auto session = std::make_shared<ExactPopulationBatchSession>(10, 6, 4);

    TrainingRunRequest request;
    request.network = makeInputLossNetwork();
    request.batchSession = session;
    request.optimizer = Sgd::Builder().initialLearningRate(0.01f).build();
    request.datasetInputBindings = {TrainingInputBinding("predictions", "predictions"),
                                    TrainingInputBinding("labels", "labels"),
                                    TrainingInputBinding("weights", "weights")};
    request.runtime.scalarTensorsToReport = {"loss", "prediction_mean"};
    request.epochs = 2;
    request.maxTrainingBatchesPerEpoch = 2;

    CapturingObserver observer;
    runNativeQueuedTraining(request, observer, NativeQueuedTrainingOptions{.maxInFlightBatches = 3, .synchronizeAfterEveryBatch = false});

    const std::vector<TrainingStatsSnapshot> train = observer.stats(TrainingEventPhase::TRAIN);
    ASSERT_EQ(train.size(), 4u);
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::validExamplesInBatch), (std::vector<uint64_t>{4, 4, 2, 4}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessedInEpoch), (std::vector<uint64_t>{4, 8, 2, 6}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::samplesProcessed), (std::vector<uint64_t>{4, 8, 10, 14}));
    EXPECT_EQ(session->getNextBatchNum(ExampleType::TRAIN), 1u);
    EXPECT_EQ(session->getNextBatchNum(ExampleType::VALIDATE), 0u);
}
