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
#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"
#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunner.h"
#include "DeepLearning/Api/Training/Executors/NativeQueuedTrainingRunnerTestHooks.h"
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
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
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
    explicit DeterministicTrainableBatchSession(
        std::function<void(ExampleType)> nextBatchObserver = {})
        : BatchSession("native_queue_trainable_oracle"),
          nextBatchObserver(std::move(nextBatchObserver)) {
        batchSize = 2;
    }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        return exampleType == ExampleType::TRAIN ? 3 : (exampleType == ExampleType::VALIDATE ? 2 : 0);
    }

    uint64_t getNumExamples(ExampleType exampleType) override { return getNumBatchesPerEpoch(exampleType) * batchSize; }

    uint64_t getNextBatchNum(ExampleType exampleType) override {
        if (nextBatchObserver) {
            nextBatchObserver(exampleType);
        }
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
    std::function<void(ExampleType)> nextBatchObserver;
};

struct HyperParameterUpdateInvocation {
    uint64_t epoch = 0;
    uint64_t batch = 0;
    uint64_t batchesPerEpoch = 0;
};

struct HyperParameterUpdateRecorder {
    void record(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
        std::lock_guard<std::mutex> lock(mutex);
        invocations.push_back(HyperParameterUpdateInvocation{epoch, batch, batchesPerEpoch});
    }

    std::vector<HyperParameterUpdateInvocation> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        return invocations;
    }

    mutable std::mutex mutex;
    std::vector<HyperParameterUpdateInvocation> invocations;
};

class CountingHyperParameterOptimizer final : public Optimizer {
   public:
    explicit CountingHyperParameterOptimizer(std::shared_ptr<HyperParameterUpdateRecorder> recorder)
        : recorder(std::move(recorder)) {
        THOR_THROW_IF_FALSE(this->recorder != nullptr);
    }

    std::shared_ptr<ThorImplementation::Optimizer> stamp(
        std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override {
        (void)trainableLayer;
        std::shared_ptr<HyperParameterUpdateRecorder> stampedRecorder = recorder;
        return std::make_shared<ThorImplementation::CustomOptimizer>(
            getId(),
            std::vector<ThorImplementation::CustomOptimizerStateSpec>{},
            [](const ThorImplementation::CustomOptimizerUpdateContext& context) {
                const ThorImplementation::DataType weightsDType = context.weightsTensor().getDataType();
                ThorImplementation::Expression weights =
                    context.weights(ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
                ThorImplementation::Expression gradient = context.gradient();
                ThorImplementation::Expression step = ThorImplementation::Expression::constantScalar(0.0001f);
                return ThorImplementation::CustomOptimizerUpdateExpression{{
                    {"weights", (weights - step * gradient).withOutputDType(weightsDType)},
                }};
            },
            ThorImplementation::CustomOptimizer::RuntimeScalarBuilder{},
            /*supportsSparseRowGradients=*/false,
            [stampedRecorder](uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) {
                stampedRecorder->record(epoch, batch, batchesPerEpoch);
                return std::unordered_map<std::string, float>{};
            });
    }

    nlohmann::json architectureJson() const override {
        return nlohmann::json{{"optimizer_type", "counting_hyper_parameter_optimizer"}, {"version", getVersion()}, {"id", getId()}};
    }

    std::string getType() const override { return "CountingHyperParameterOptimizer"; }

   protected:
    std::shared_ptr<Optimizer> clone() const override { return std::make_shared<CountingHyperParameterOptimizer>(*this); }

   private:
    std::shared_ptr<HyperParameterUpdateRecorder> recorder;
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

TEST(NativeQueuedPartialBatchAccounting, ValidationDoesNotUpdateOptimizerHyperParameters) {
    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto session = std::make_shared<DeterministicTrainableBatchSession>();
    auto recorder = std::make_shared<HyperParameterUpdateRecorder>();

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = std::make_shared<CountingHyperParameterOptimizer>(recorder);
    request.datasetInputBindings = {TrainingInputBinding("features", "features"), TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 3;

    CapturingObserver observer;
    runNativeQueuedTraining(
        request, observer, NativeQueuedTrainingOptions{.maxInFlightBatches = 3, .synchronizeAfterEveryBatch = false});

    const std::vector<HyperParameterUpdateInvocation> invocations = recorder->snapshot();
    ASSERT_EQ(invocations.size(), 9u);
    for (uint64_t epoch = 0; epoch < 3; ++epoch) {
        for (uint64_t batch = 0; batch < 3; ++batch) {
            const HyperParameterUpdateInvocation& invocation = invocations[epoch * 3 + batch];
            EXPECT_EQ(invocation.epoch, epoch);
            EXPECT_EQ(invocation.batch, batch);
            EXPECT_EQ(invocation.batchesPerEpoch, 3u);
        }
    }
}

TEST(NativeQueuedPartialBatchAccounting, SegmentStatePreservesLifecycleEventOrdering) {
    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto session = std::make_shared<DeterministicTrainableBatchSession>();

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 2;

    CapturingObserver observer;
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 3,
                                    .synchronizeAfterEveryBatch = false});

    std::vector<const TrainingEvent*> lifecycle;
    for (const TrainingEvent& event : observer.events) {
        if (event.type == TrainingEventType::EPOCH_STARTED ||
            event.type == TrainingEventType::EPOCH_FINISHED) {
            lifecycle.push_back(&event);
        }
    }

    ASSERT_EQ(lifecycle.size(), 8u);
    auto expectLifecycle = [&](size_t index,
                               TrainingEventType type,
                               TrainingEventPhase phase,
                               uint64_t epoch,
                               uint64_t stepsPerEpoch) {
        ASSERT_LT(index, lifecycle.size());
        EXPECT_EQ(lifecycle[index]->type, type) << "lifecycle event " << index;
        EXPECT_EQ(lifecycle[index]->stats.phase, phase) << "lifecycle event " << index;
        EXPECT_EQ(lifecycle[index]->stats.epoch, epoch) << "lifecycle event " << index;
        EXPECT_EQ(lifecycle[index]->stats.stepsPerEpoch, stepsPerEpoch)
            << "lifecycle event " << index;
    };

    expectLifecycle(0, TrainingEventType::EPOCH_STARTED, TrainingEventPhase::TRAIN, 1, 3);
    expectLifecycle(1, TrainingEventType::EPOCH_FINISHED, TrainingEventPhase::TRAIN, 1, 3);
    expectLifecycle(2, TrainingEventType::EPOCH_STARTED, TrainingEventPhase::VALIDATE, 1, 2);
    expectLifecycle(3, TrainingEventType::EPOCH_FINISHED, TrainingEventPhase::VALIDATE, 1, 2);
    expectLifecycle(4, TrainingEventType::EPOCH_STARTED, TrainingEventPhase::TRAIN, 2, 3);
    expectLifecycle(5, TrainingEventType::EPOCH_FINISHED, TrainingEventPhase::TRAIN, 2, 3);
    expectLifecycle(6, TrainingEventType::EPOCH_STARTED, TrainingEventPhase::VALIDATE, 2, 2);
    expectLifecycle(7, TrainingEventType::EPOCH_FINISHED, TrainingEventPhase::VALIDATE, 2, 2);

    const std::vector<TrainingStatsSnapshot> train =
        observer.stats(TrainingEventPhase::TRAIN);
    const std::vector<TrainingStatsSnapshot> validate =
        observer.stats(TrainingEventPhase::VALIDATE);
    ASSERT_EQ(train.size(), 6u);
    ASSERT_EQ(validate.size(), 4u);
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::stepsPerEpoch),
              (std::vector<uint64_t>{3, 3, 3, 3, 3, 3}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::stepsPerEpoch),
              (std::vector<uint64_t>{2, 2, 2, 2}));
    EXPECT_EQ(fieldValues(train, &TrainingStatsSnapshot::epoch),
              (std::vector<uint64_t>{1, 1, 1, 2, 2, 2}));
    EXPECT_EQ(fieldValues(validate, &TrainingStatsSnapshot::epoch),
              (std::vector<uint64_t>{1, 1, 2, 2}));
}

TEST(NativeQueuedPartialBatchAccounting, SegmentCursorIsResolvedOnSchedulerWorker) {
    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    const std::thread::id callerThread = std::this_thread::get_id();
    std::mutex observationMutex;
    std::optional<std::thread::id> firstTrainCursorThread;
    std::optional<std::thread::id> firstValidateCursorThread;
    auto session = std::make_shared<DeterministicTrainableBatchSession>(
        [&](ExampleType exampleType) {
            std::lock_guard<std::mutex> lock(observationMutex);
            std::optional<std::thread::id>* destination = nullptr;
            if (exampleType == ExampleType::TRAIN) {
                destination = &firstTrainCursorThread;
            } else if (exampleType == ExampleType::VALIDATE) {
                destination = &firstValidateCursorThread;
            }
            if (destination != nullptr && !destination->has_value()) {
                destination->emplace(std::this_thread::get_id());
            }
        });

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 1;

    CapturingObserver observer;
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 3,
                                    .synchronizeAfterEveryBatch = false});

    std::lock_guard<std::mutex> lock(observationMutex);
    ASSERT_TRUE(firstTrainCursorThread.has_value());
    ASSERT_TRUE(firstValidateCursorThread.has_value());
    EXPECT_NE(firstTrainCursorThread.value(), callerThread);
    EXPECT_NE(firstValidateCursorThread.value(), callerThread);
    EXPECT_EQ(firstTrainCursorThread.value(), firstValidateCursorThread.value());
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

TEST(NativeQueuedPartialBatchAccounting, SchedulerResourcesPersistAcrossSchedulingWindows) {
    detail::resetNativeQueuedSchedulerResourceDiagnosticsForTests();

    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto session = std::make_shared<DeterministicTrainableBatchSession>();

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 2;
    request.checkBestModelEveryEpochs = 1;
    request.firstModelSelectionEpoch = 1;
    request.modelSelectionScore = TrainingModelSelectionScore(
        [](const TrainingModelSelectionContext& context) {
            return context.validationLoss();
        });

    CapturingObserver observer;
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 3,
                                    .synchronizeAfterEveryBatch = false});

    const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
        detail::nativeQueuedSchedulerResourceDiagnosticsForTests();
    EXPECT_EQ(diagnostics.resourceConstructionCount, 1u);
    EXPECT_EQ(diagnostics.runStateConstructionCount, 1u);
    EXPECT_EQ(diagnostics.schedulingWindowCount, 2u);
    EXPECT_EQ(diagnostics.hostDecisionBarrierCount, 1u);
    EXPECT_EQ(diagnostics.workerThreadStartCount, 1u);
    EXPECT_EQ(diagnostics.distinctWorkerThreadsObserved, 1u);
    EXPECT_EQ(diagnostics.distinctResourceInstancesObserved, 1u);
    EXPECT_EQ(diagnostics.distinctRunStateInstancesObserved, 1u);
    EXPECT_TRUE(diagnostics.slotStorageStableAcrossSchedulingWindows);
    EXPECT_NE(diagnostics.firstProcessingFinishedEventId, 0u);
    EXPECT_NE(diagnostics.firstCompletionFinishedEventId, 0u);
    EXPECT_TRUE(
        diagnostics.processingFinishedEventIdStableAcrossSchedulingWindows);
    EXPECT_TRUE(
        diagnostics.completionFinishedEventIdStableAcrossSchedulingWindows);
}

TEST(NativeQueuedPartialBatchAccounting, HundredEpochRunUsesOneWindowWithoutHostDecisionBarriers) {
    detail::resetNativeQueuedSchedulerResourceDiagnosticsForTests();

    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto session = std::make_shared<DeterministicTrainableBatchSession>();

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 100;

    CapturingObserver observer;
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 32,
                                    .synchronizeAfterEveryBatch = false});

    const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
        detail::nativeQueuedSchedulerResourceDiagnosticsForTests();
    EXPECT_EQ(diagnostics.resourceConstructionCount, 1u);
    EXPECT_EQ(diagnostics.runStateConstructionCount, 1u);
    EXPECT_EQ(diagnostics.workerThreadStartCount, 1u);
    EXPECT_EQ(diagnostics.schedulingWindowCount, 1u);
    EXPECT_EQ(diagnostics.hostDecisionBarrierCount, 0u);
    EXPECT_EQ(diagnostics.distinctWorkerThreadsObserved, 1u);
    EXPECT_EQ(diagnostics.distinctResourceInstancesObserved, 1u);
    EXPECT_EQ(diagnostics.distinctRunStateInstancesObserved, 1u);
    EXPECT_TRUE(diagnostics.slotStorageStableAcrossSchedulingWindows);
    ASSERT_TRUE(diagnostics.hasSubmittedBatch);
    EXPECT_EQ(diagnostics.submittedBatchCount, 500u);
    EXPECT_EQ(diagnostics.maxOptimizerEpochSubmitted, 99u);
}

TEST(NativeQueuedPartialBatchAccounting, OrdinaryEpochBoundaryDoesNotDrainSchedulingWindow) {
    detail::resetNativeQueuedSchedulerResourceDiagnosticsForTests();

    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto session = std::make_shared<DeterministicTrainableBatchSession>();

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 2;

    class CrossEpochObserver final : public TrainingObserver {
       public:
        void onTrainingEvent(const TrainingEvent& event) override {
            if (event.type != TrainingEventType::EPOCH_FINISHED ||
                event.stats.phase != TrainingEventPhase::VALIDATE ||
                event.stats.epoch != 1) {
                return;
            }
            // Give the persistent producer a bounded opportunity to get ahead.
            // If EPOCH_FINISHED is accidentally a scheduler barrier, the producer
            // cannot satisfy this condition until this callback returns.
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(1);
            do {
                const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
                    detail::peekNativeQueuedSchedulerResourceDiagnosticsForTests();
                if (diagnostics.hasSubmittedBatch &&
                    diagnostics.maxOptimizerEpochSubmitted >= 1) {
                    epochTwoSubmittedBeforeEpochOneValidationFinished = true;
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } while (std::chrono::steady_clock::now() < deadline);
        }

        bool epochTwoSubmittedBeforeEpochOneValidationFinished = false;
    } observer;

    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 32,
                                    .synchronizeAfterEveryBatch = false});

    EXPECT_TRUE(observer.epochTwoSubmittedBeforeEpochOneValidationFinished);
    const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
        detail::nativeQueuedSchedulerResourceDiagnosticsForTests();
    EXPECT_EQ(diagnostics.schedulingWindowCount, 1u);
    EXPECT_EQ(diagnostics.hostDecisionBarrierCount, 0u);
    EXPECT_TRUE(diagnostics.hasSubmittedBatch);
    EXPECT_GE(diagnostics.maxOptimizerEpochSubmitted, 1u);
}

TEST(NativeQueuedPartialBatchAccounting, NamedValidationPopulationsShareContinuousSchedulingWindow) {
    detail::resetNativeQueuedSchedulerResourceDiagnosticsForTests();

    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto defaultSession = std::make_shared<DeterministicTrainableBatchSession>();
    auto namedSession = std::make_shared<DeterministicTrainableBatchSession>();

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = defaultSession;
    request.defaultValidationPopulation = "unseen_sku";
    request.additionalValidationSessions.push_back(
        NamedValidationSession{"seen_sku", namedSession});
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 2;

    class NamedPopulationObserver final : public TrainingObserver {
       public:
        void onTrainingEvent(const TrainingEvent& event) override {
            if (event.type == TrainingEventType::STATS &&
                event.stats.phase == TrainingEventPhase::VALIDATE &&
                event.stats.loss.has_value()) {
                losses[{event.stats.epoch, event.stats.validationPopulation}]
                    .push_back(event.stats.loss.value());
            }
            if (event.type != TrainingEventType::EPOCH_FINISHED ||
                event.stats.phase != TrainingEventPhase::VALIDATE ||
                event.stats.epoch != 1 ||
                event.stats.validationPopulation != "seen_sku") {
                return;
            }

            // The named population is the last segment in logical epoch one.
            // Lifecycle delivery is a notification only: with spare slots the
            // producer must already be allowed to submit epoch-two TRAIN work.
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(1);
            do {
                const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
                    detail::peekNativeQueuedSchedulerResourceDiagnosticsForTests();
                if (diagnostics.hasSubmittedBatch &&
                    diagnostics.maxOptimizerEpochSubmitted >= 1) {
                    epochTwoSubmittedBeforeNamedValidationFinished = true;
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } while (std::chrono::steady_clock::now() < deadline);
        }

        std::map<std::pair<uint64_t, std::string>, std::vector<double>> losses;
        bool epochTwoSubmittedBeforeNamedValidationFinished = false;
    } observer;

    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 32,
                                    .synchronizeAfterEveryBatch = false});

    for (uint64_t epoch = 1; epoch <= 2; ++epoch) {
        const auto defaultIt = observer.losses.find({epoch, "unseen_sku"});
        const auto namedIt = observer.losses.find({epoch, "seen_sku"});
        ASSERT_NE(defaultIt, observer.losses.end());
        ASSERT_NE(namedIt, observer.losses.end());
        ASSERT_EQ(defaultIt->second.size(), namedIt->second.size());
        ASSERT_FALSE(defaultIt->second.empty());
        for (size_t batch = 0; batch < defaultIt->second.size(); ++batch) {
            // Both sessions expose identical validation examples. Equal losses
            // prove both populations observed the same model checkpoint; an
            // intervening epoch-two optimizer update would change this oracle.
            EXPECT_NEAR(defaultIt->second[batch], namedIt->second[batch], 1e-6)
                << "epoch " << epoch << " validation batch " << batch;
        }
    }

    EXPECT_TRUE(observer.epochTwoSubmittedBeforeNamedValidationFinished);
    const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
        detail::nativeQueuedSchedulerResourceDiagnosticsForTests();
    EXPECT_EQ(diagnostics.schedulingWindowCount, 1u);
    EXPECT_EQ(diagnostics.hostDecisionBarrierCount, 0u);
    EXPECT_EQ(diagnostics.workerThreadStartCount, 1u);
}

TEST(NativeQueuedPartialBatchAccounting, NamedValidationPopulationFeedsModelSelectionInsideDecisionWindow) {
    detail::resetNativeQueuedSchedulerResourceDiagnosticsForTests();

    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto defaultSession = std::make_shared<DeterministicTrainableBatchSession>();
    auto namedSession = std::make_shared<DeterministicTrainableBatchSession>();
    std::vector<TrainingModelSelectionContext> selectionContexts;

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = defaultSession;
    request.defaultValidationPopulation = "unseen_sku";
    request.additionalValidationSessions.push_back(
        NamedValidationSession{"seen_sku", namedSession});
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 1;
    request.checkBestModelEveryEpochs = 1;
    request.firstModelSelectionEpoch = 1;
    request.modelSelectionScore = TrainingModelSelectionScore(
        [&selectionContexts](const TrainingModelSelectionContext& context) {
            selectionContexts.push_back(context);
            return context.validation("seen_sku").loss;
        });

    CapturingObserver observer;
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 32,
                                    .synchronizeAfterEveryBatch = false});

    ASSERT_FALSE(selectionContexts.empty());
    for (const TrainingModelSelectionContext& context : selectionContexts) {
        EXPECT_EQ(context.defaultValidationPopulation, "unseen_sku");
        ASSERT_TRUE(context.validation("unseen_sku").loss.has_value());
        ASSERT_TRUE(context.validation("seen_sku").loss.has_value());
        EXPECT_NEAR(context.validation("unseen_sku").loss.value(),
                    context.validation("seen_sku").loss.value(),
                    1e-6);
    }

    const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
        detail::nativeQueuedSchedulerResourceDiagnosticsForTests();
    // Named validation is a segment inside the one decision window, not a
    // standalone scheduler command.
    EXPECT_EQ(diagnostics.schedulingWindowCount, 1u);
    // This one-epoch fit evaluates model selection, but there is no later
    // training window for that decision to gate.
    EXPECT_EQ(diagnostics.hostDecisionBarrierCount, 0u);
}

TEST(NativeQueuedPartialBatchAccounting, ModelSelectionRetainsSchedulingWindowBarrier) {
    detail::resetNativeQueuedSchedulerResourceDiagnosticsForTests();

    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto session = std::make_shared<DeterministicTrainableBatchSession>();

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 2;
    request.checkBestModelEveryEpochs = 1;
    // Avoid a phase-entry validation command so this test observes only the
    // trained-candidate decision barrier between epochs one and two.
    request.firstModelSelectionEpoch = 1;
    request.modelSelectionScore = TrainingModelSelectionScore(
        [](const TrainingModelSelectionContext& context) {
            return context.validationLoss();
        });

    class DecisionBarrierObserver final : public TrainingObserver {
       public:
        void onTrainingEvent(const TrainingEvent& event) override {
            if (event.type != TrainingEventType::EPOCH_FINISHED ||
                event.stats.phase != TrainingEventPhase::VALIDATE ||
                event.stats.epoch != 1) {
                return;
            }
            const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
                detail::peekNativeQueuedSchedulerResourceDiagnosticsForTests();
            epochTwoWasSubmittedBeforeEpochOneDecision =
                diagnostics.hasSubmittedBatch &&
                diagnostics.maxOptimizerEpochSubmitted >= 1;
        }

        bool epochTwoWasSubmittedBeforeEpochOneDecision = false;
    } observer;

    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 32,
                                    .synchronizeAfterEveryBatch = false});

    EXPECT_FALSE(observer.epochTwoWasSubmittedBeforeEpochOneDecision);
    const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
        detail::nativeQueuedSchedulerResourceDiagnosticsForTests();
    EXPECT_EQ(diagnostics.schedulingWindowCount, 2u);
    // Epoch one gates epoch two. The final epoch's selection has no subsequent
    // optimizer work to hold back, so it is not a scheduling barrier.
    EXPECT_EQ(diagnostics.hostDecisionBarrierCount, 1u);
    EXPECT_TRUE(diagnostics.hasSubmittedBatch);
    EXPECT_GE(diagnostics.maxOptimizerEpochSubmitted, 1u);
}

TEST(NativeQueuedPartialBatchAccounting, ModelSelectionCadenceStreamsUntilDecisionEpoch) {
    detail::resetNativeQueuedSchedulerResourceDiagnosticsForTests();

    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto session = std::make_shared<DeterministicTrainableBatchSession>();
    bool epochFourSubmittedBeforeEpochThreeDecision = false;
    bool observedEpochThreeDecision = false;

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 4;
    request.checkBestModelEveryEpochs = 3;
    request.firstModelSelectionEpoch = 3;
    request.modelSelectionScore = TrainingModelSelectionScore(
        [&](const TrainingModelSelectionContext& context) {
            if (context.epoch == 3) {
                observedEpochThreeDecision = true;
                const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
                    detail::peekNativeQueuedSchedulerResourceDiagnosticsForTests();
                epochFourSubmittedBeforeEpochThreeDecision =
                    diagnostics.hasSubmittedBatch &&
                    diagnostics.maxOptimizerEpochSubmitted >= 3;
            }
            return context.validationLoss();
        });

    class CadenceObserver final : public TrainingObserver {
       public:
        void onTrainingEvent(const TrainingEvent& event) override {
            if (event.type != TrainingEventType::EPOCH_FINISHED ||
                event.stats.phase != TrainingEventPhase::VALIDATE ||
                event.stats.epoch != 1) {
                return;
            }

            // Epochs 1 and 2 are not host-decision boundaries. With enough
            // slots, the producer should be able to submit all the way through
            // epoch 3 before this lifecycle notification returns.
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(1);
            do {
                const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
                    detail::peekNativeQueuedSchedulerResourceDiagnosticsForTests();
                if (diagnostics.hasSubmittedBatch &&
                    diagnostics.maxOptimizerEpochSubmitted >= 2) {
                    epochThreeSubmittedBeforeEpochOneFinished = true;
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } while (std::chrono::steady_clock::now() < deadline);
        }

        bool epochThreeSubmittedBeforeEpochOneFinished = false;
    } observer;

    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 32,
                                    .synchronizeAfterEveryBatch = false});

    EXPECT_TRUE(observer.epochThreeSubmittedBeforeEpochOneFinished);
    EXPECT_TRUE(observedEpochThreeDecision);
    EXPECT_FALSE(epochFourSubmittedBeforeEpochThreeDecision);

    const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
        detail::nativeQueuedSchedulerResourceDiagnosticsForTests();
    // Epochs 1..3 form the first window through the decision epoch. Epoch 4 is
    // submitted only after that decision returns, in a second window.
    EXPECT_EQ(diagnostics.schedulingWindowCount, 2u);
    EXPECT_EQ(diagnostics.hostDecisionBarrierCount, 1u);
    EXPECT_TRUE(diagnostics.hasSubmittedBatch);
    EXPECT_GE(diagnostics.maxOptimizerEpochSubmitted, 3u);
}

TEST(NativeQueuedPartialBatchAccounting, EarlyCompletionDecisionDoesNotSubmitNextWindow) {
    detail::resetNativeQueuedSchedulerResourceDiagnosticsForTests();

    TrainableOracleNetwork fixture = makeTrainableOracleNetwork();
    auto session = std::make_shared<DeterministicTrainableBatchSession>();

    TrainingRunRequest request;
    request.network = fixture.network;
    request.batchSession = session;
    request.optimizer = Sgd::Builder()
                            .initialLearningRate(0.01f)
                            .decay(0.0f)
                            .momentum(0.0f)
                            .build();
    request.datasetInputBindings = {
        TrainingInputBinding("features", "features"),
        TrainingInputBinding("labels", "labels")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 5;
    request.checkBestModelEveryEpochs = 3;
    request.firstModelSelectionEpoch = 3;
    request.modelSelectionScore = TrainingModelSelectionScore(
        [](const TrainingModelSelectionContext& context) {
            return context.validationLoss();
        });
    request.earlyCompletionPolicies = {
        TrainingEarlyCompletionPolicy(
            [](double, double, uint64_t currentEpoch, uint64_t) {
                return currentEpoch == 3;
            })};

    CapturingObserver observer;
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 32,
                                    .synchronizeAfterEveryBatch = false});

    uint64_t maxTrainEpochObserved = 0;
    for (const TrainingStatsSnapshot& snapshot :
         observer.stats(TrainingEventPhase::TRAIN)) {
        maxTrainEpochObserved = std::max(maxTrainEpochObserved, snapshot.epoch);
    }
    EXPECT_EQ(maxTrainEpochObserved, 3u);

    const detail::NativeQueuedSchedulerResourceDiagnosticsForTests diagnostics =
        detail::nativeQueuedSchedulerResourceDiagnosticsForTests();
    // The first window ends at the epoch-3 decision. Early completion prevents
    // any second window, so epoch-4 optimizer work is never even submitted.
    EXPECT_EQ(diagnostics.schedulingWindowCount, 1u);
    EXPECT_EQ(diagnostics.hostDecisionBarrierCount, 1u);
    ASSERT_TRUE(diagnostics.hasSubmittedBatch);
    EXPECT_EQ(diagnostics.maxOptimizerEpochSubmitted, 2u);
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

TEST(NativeQueuedPartialBatchAccounting, LogicalWorkPairFeedsCoherentPhaseWallRates) {
    auto session = std::make_shared<ExactPopulationBatchSession>(4, 4, 4);

    TrainingRunRequest request;
    request.network = makeInputLossNetwork();
    request.batchSession = session;
    request.optimizer = Sgd::Builder().initialLearningRate(0.01f).build();
    request.datasetInputBindings = {TrainingInputBinding("predictions", "predictions"),
                                    TrainingInputBinding("labels", "labels"),
                                    TrainingInputBinding("weights", "weights")};
    request.runtime.scalarTensorsToReport = {"loss"};
    request.epochs = 1;

    CapturingObserver observer;
    runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 2, .synchronizeAfterEveryBatch = false});

    const std::vector<TrainingStatsSnapshot> train = observer.stats(TrainingEventPhase::TRAIN);
    const std::vector<TrainingStatsSnapshot> validate = observer.stats(TrainingEventPhase::VALIDATE);
    ASSERT_EQ(train.size(), 1u);
    ASSERT_EQ(validate.size(), 1u);

    auto expectCoherentLogicalRates = [](const TrainingStatsSnapshot& stats) {
        EXPECT_GT(stats.floatingPointOperationsPerBatch, 0u);
        EXPECT_GT(stats.logicalBytesPerBatch, 0u);
        EXPECT_GT(stats.floatingPointOperationsPerSecond, 0.0);
        EXPECT_GT(stats.logicalBytesPerSecond, 0.0);
        EXPECT_NEAR(stats.logicalArithmeticIntensity,
                    stats.floatingPointOperationsPerSecond / stats.logicalBytesPerSecond,
                    1e-12);
    };

    expectCoherentLogicalRates(train.front());
    expectCoherentLogicalRates(validate.front());

    // TRAIN accounts forward + backward model work; VALIDATE accounts forward
    // only. Optimizer updates remain outside both logical-work totals.
    EXPECT_GT(train.front().floatingPointOperationsPerBatch,
              validate.front().floatingPointOperationsPerBatch);
    EXPECT_GT(train.front().logicalBytesPerBatch, validate.front().logicalBytesPerBatch);
}

TEST(NativeQueuedPartialBatchAccounting, LargeInitialEpochCannotMakeThroughputFlopAccountingFatal) {
    auto session = std::make_shared<ExactPopulationBatchSession>(10, 6, 4);

    TrainingRunRequest request;
    request.network = makeInputLossNetwork();
    request.batchSession = session;
    request.optimizer = Sgd::Builder().initialLearningRate(0.01f).build();
    request.datasetInputBindings = {TrainingInputBinding("predictions", "predictions"),
                                    TrainingInputBinding("labels", "labels"),
                                    TrainingInputBinding("weights", "weights")};
    request.runtime.scalarTensorsToReport = {"loss", "prediction_mean"};
    request.epochs = 1;
    // Reproduce the production shape of the failure: training uses a local capped
    // step counter, while validation reports a large cumulative epoch-derived
    // step.  The old throughput code multiplied that cumulative step by the
    // current batch FLOP count and could throw before validation completed.
    request.initialCompletedEpochs = std::numeric_limits<uint64_t>::max() / 8;
    request.maxTrainingBatchesPerEpoch = 1;

    CapturingObserver observer;
    EXPECT_NO_THROW(runNativeQueuedTraining(
        request,
        observer,
        NativeQueuedTrainingOptions{.maxInFlightBatches = 3, .synchronizeAfterEveryBatch = false}));

    const std::vector<TrainingStatsSnapshot> train = observer.stats(TrainingEventPhase::TRAIN);
    const std::vector<TrainingStatsSnapshot> validate = observer.stats(TrainingEventPhase::VALIDATE);
    ASSERT_EQ(train.size(), 1u);
    ASSERT_EQ(validate.size(), 2u);
    EXPECT_EQ(train.front().step, 1u);
    EXPECT_GT(validate.front().step, uint64_t{1} << 60);
}
