#include "DeepLearning/Api/Training/TrainingRuns.h"

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
#include "DeepLearning/Api/Training/Events/TrainingEvent.h"
#include "DeepLearning/Api/Training/Executors/TrainingExecutor.h"
#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace Thor::Testing {
std::vector<std::string> trainingRunsComposedEvaluatorExternalInputNamesForTest(
    const std::vector<std::shared_ptr<Network>>& memberNetworks,
    const std::vector<double>& weights,
    const std::vector<std::string>& outputNames);
std::map<std::string, bool> trainingRunsComposedEvaluatorMemberInputAliasStatesForTest(
    const std::vector<std::shared_ptr<Network>>& memberNetworks,
    const std::vector<double>& weights,
    const std::vector<std::string>& outputNames);
std::map<std::string, std::vector<uint64_t>> trainingRunsComposedEvaluatorAveragedOutputDimensionsForTest(
    const std::vector<std::shared_ptr<Network>>& memberNetworks,
    const std::vector<double>& weights,
    const std::vector<std::string>& outputNames);
std::map<std::string, std::string> trainingRunsComposedEvaluatorAveragedOutputDataTypesForTest(
    const std::vector<std::shared_ptr<Network>>& memberNetworks,
    const std::vector<double>& weights,
    const std::vector<std::string>& outputNames);
std::optional<double> trainingRunsWeightedLossSumFromWeightedLossValuesForTest(
    const std::vector<std::optional<double>>& weightedLossValues);
std::vector<std::string> trainingRunsComposedLossEvaluatorExternalInputNamesForTest(
    const std::vector<std::shared_ptr<Network>>& memberNetworks,
    const std::vector<double>& weights,
    const std::vector<std::string>& requestedLossNames);
}

using namespace Thor;

namespace {

ThorImplementation::Tensor makeCpuTensor(ThorImplementation::DataType dataType, std::vector<uint64_t> dimensions) {
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    return ThorImplementation::Tensor(cpuPlacement, ThorImplementation::TensorDescriptor(dataType, std::move(dimensions)));
}

std::shared_ptr<Network> makeReluMemberNetwork(const std::string& networkName,
                                             const std::vector<uint64_t>& inputDimensions,
                                             const std::vector<std::string>& outputNames) {
    auto network = std::make_shared<Network>(networkName);
    NetworkInput input = NetworkInput::Builder()
                             .network(*network)
                             .name("features")
                             .dimensions(inputDimensions)
                             .dataType(DataType::FP32)
                             .build();
    std::shared_ptr<Activation> relu = Relu::Builder().network(*network).featureInput(input.getFeatureOutput().value()).build();
    for (const std::string& outputName : outputNames) {
        NetworkOutput::Builder().network(*network).name(outputName).inputTensor(relu->getFeatureOutput().value()).build();
    }
    return network;
}

class FakeLoader : public Loader {
   public:
    FakeLoader() { batchSize = 4; }

    Batch getBatch(ExampleType exampleType, uint64_t& batchNum) override {
        (void)exampleType;
        (void)batchNum;
        return {};
    }

    void returnBatchBuffers(ExampleType exampleType, Batch&& batch) override {
        (void)exampleType;
        (void)batch;
    }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        (void)exampleType;
        return 0;
    }

    uint64_t getNumExamples(ExampleType exampleType) override {
        (void)exampleType;
        return 0;
    }

    uint64_t getNextBatchNum(ExampleType exampleType) override {
        (void)exampleType;
        return 0;
    }
};

class Coordinator {
   public:
    explicit Coordinator(size_t expectedStarts) : expectedStarts(expectedStarts) {}

    void markStarted() {
        std::lock_guard<std::mutex> lock(mutex);
        started += 1;
        cv.notify_all();
    }

    bool waitForAllStarted(std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
        return waitForStartedCount(expectedStarts, timeout);
    }

    bool waitForStartedCount(size_t count, std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [this, count]() { return started >= count; });
    }

    void releaseAll() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }

    void waitUntilReleased(std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cv.wait_for(lock, timeout, [this]() { return released; })) {
            throw std::runtime_error("Timed out waiting for TrainingRuns test release.");
        }
    }

    void waitForCancellation(const TrainingCancellationToken& token,
                             std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::unique_lock<std::mutex> lock(mutex);
        while (!token.isCancellationRequested()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error("Timed out waiting for TrainingRuns cancellation.");
            }
            cv.wait_for(lock, std::chrono::milliseconds(1));
        }
    }

    size_t startedCount() const {
        std::lock_guard<std::mutex> lock(mutex);
        return started;
    }

   private:
    size_t expectedStarts;
    mutable std::mutex mutex;
    std::condition_variable cv;
    size_t started = 0;
    bool released = false;
};

enum class FakeExecutorBehavior { COMPLETE_AFTER_RELEASE, FAIL_AFTER_RELEASE, WAIT_FOR_CANCEL_THEN_CANCEL, OOM_AFTER_RELEASE };

TrainingStatsSnapshot makeStats(uint64_t step) {
    TrainingStatsSnapshot stats;
    stats.phase = TrainingEventPhase::TRAIN;
    stats.epoch = 1;
    stats.epochs = 1;
    stats.step = step;
    stats.stepInEpoch = step;
    stats.stepsPerEpoch = 10;
    stats.loss = 1.0 / static_cast<double>(step + 1);
    stats.elapsedSeconds = static_cast<double>(step);
    return stats;
}

class CoordinatedExecutor : public TrainingExecutor {
   public:
    CoordinatedExecutor(std::shared_ptr<Coordinator> coordinator, FakeExecutorBehavior behavior, uint64_t statsStep = 1)
        : coordinator(std::move(coordinator)), behavior(behavior), statsStep(statsStep) {}

    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        calls += 1;
        coordinator->markStarted();
        coordinator->waitUntilReleased();

        if (behavior == FakeExecutorBehavior::WAIT_FOR_CANCEL_THEN_CANCEL) {
            coordinator->waitForCancellation(request.cancellationToken);
            request.cancellationToken.throwIfCancellationRequested("cancelled by sibling failure");
            return;
        }

        request.cancellationToken.throwIfCancellationRequested();
        observer.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(statsStep)));

        if (behavior == FakeExecutorBehavior::FAIL_AFTER_RELEASE) {
            throw std::runtime_error("planned trainer failure");
        }
        if (behavior == FakeExecutorBehavior::OOM_AFTER_RELEASE) {
            throw std::runtime_error("CUDA_ERROR_OUT_OF_MEMORY during fake placement");
        }
    }

    uint32_t calls = 0;

   private:
    std::shared_ptr<Coordinator> coordinator;
    FakeExecutorBehavior behavior;
    uint64_t statsStep;
};


class RestartProgressExecutor : public TrainingExecutor {
   public:
    explicit RestartProgressExecutor(std::vector<std::vector<double>> attemptEpochLosses)
        : attemptEpochLosses(std::move(attemptEpochLosses)) {}

    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        calls += 1;
        sawLossRequested = sawLossRequested || request.runtime.scalarTensorsToReport.count("loss") != 0;
        lastEarlyCompletionPolicyCount = request.earlyCompletionPolicies.size();
        lastInitialCompletedEpochs = request.initialCompletedEpochs;
        initialCompletedEpochsByCall.push_back(request.initialCompletedEpochs);
        lastMinEarlyCompletionEpochs = request.minEarlyCompletionEpochs;
        if (!request.earlyCompletionPolicies.empty()) {
            lastEarlyCompletionDecision = request.earlyCompletionPolicies.front().shouldComplete(10.0, 9.0, 5, 4);
        }

        const size_t attemptIndex = static_cast<size_t>(calls - 1);
        const std::vector<double>& losses = attemptEpochLosses.at(std::min(attemptIndex, attemptEpochLosses.size() - 1));
        uint64_t finalEpoch = request.initialCompletedEpochs;
        for (uint32_t epoch = 1; epoch <= request.epochs; ++epoch) {
            request.cancellationToken.throwIfCancellationRequested();
            const uint64_t globalEpoch = request.initialCompletedEpochs + epoch;
            TrainingStatsSnapshot stats;
            stats.phase = TrainingEventPhase::TRAIN;
            stats.epoch = globalEpoch;
            stats.epochs = request.initialCompletedEpochs + request.epochs;
            stats.step = globalEpoch;
            stats.stepInEpoch = 1;
            stats.stepsPerEpoch = 1;
            stats.loss = losses[std::min<size_t>(epoch - 1, losses.size() - 1)];
            observer.onTrainingEvent(TrainingEvent::statsUpdated(stats));
            finalEpoch = globalEpoch;
        }
        if (request.completedTrainingEpochs != nullptr) {
            *request.completedTrainingEpochs = finalEpoch;
        }
    }

    uint32_t calls = 0;
    bool sawLossRequested = false;
    size_t lastEarlyCompletionPolicyCount = 0;
    bool lastEarlyCompletionDecision = false;
    uint64_t lastInitialCompletedEpochs = 0;
    std::vector<uint64_t> initialCompletedEpochsByCall{};
    uint64_t lastMinEarlyCompletionEpochs = 0;

   private:
    std::vector<std::vector<double>> attemptEpochLosses;
};


std::shared_ptr<Network> makeNetworkWithOutput(const std::string& name, const std::vector<uint64_t>& dimensions) {
    auto network = std::make_shared<Network>(name);
    NetworkInput::Builder().network(*network).name("features").dimensions({0, 4}).dataType(DataType::FP32).build();
    Tensor outputTensor(DataType::FP32, dimensions);
    NetworkOutput::Builder().network(*network).name("predictions").inputTensor(outputTensor).dataType(DataType::FP32).build();
    return network;
}

std::shared_ptr<Network> makeDemandSignatureNetwork(const std::string& name) {
    auto network = std::make_shared<Network>(name);
    NetworkInput::Builder().network(*network).name("features").dimensions({0, 4}).dataType(DataType::FP32).build();
    NetworkInput::Builder().network(*network).name("observed_daily").dimensions({0, 1}).dataType(DataType::FP32).build();
    NetworkInput::Builder().network(*network).name("observed_aggregate").dimensions({0, 1}).dataType(DataType::FP32).build();
    NetworkInput::Builder().network(*network).name("example_weights").dimensions({0, 1}).dataType(DataType::FP32).build();

    Tensor outputTensor(DataType::FP32, {0, 1});
    NetworkOutput::Builder().network(*network).name("daily").inputTensor(outputTensor).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("aggregate").inputTensor(outputTensor).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("forecast_p90").inputTensor(outputTensor).dataType(DataType::FP32).build();
    return network;
}

std::shared_ptr<Network> makeLossWeightedDemandNetwork(const std::string& name, float dailyLossWeight = 2.0f) {
    auto network = std::make_shared<Network>(name);
    NetworkInput features = NetworkInput::Builder().network(*network).name("features").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput observedDaily = NetworkInput::Builder().network(*network).name("observed_daily").dimensions({1}).dataType(DataType::FP32).build();
    NetworkInput observedAggregate = NetworkInput::Builder().network(*network).name("observed_aggregate").dimensions({1}).dataType(DataType::FP32).build();

    FullyConnected daily = FullyConnected::Builder()
                               .network(*network)
                               .featureInput(features.getFeatureOutput().value())
                               .numOutputFeatures(1)
                               .hasBias(true)
                               .noActivation()
                               .build();
    FullyConnected aggregate = FullyConnected::Builder()
                                   .network(*network)
                                   .featureInput(features.getFeatureOutput().value())
                                   .numOutputFeatures(1)
                                   .hasBias(true)
                                   .noActivation()
                                   .build();
    FullyConnected p90 = FullyConnected::Builder()
                             .network(*network)
                             .featureInput(features.getFeatureOutput().value())
                             .numOutputFeatures(1)
                             .hasBias(true)
                             .noActivation()
                             .build();

    MAE dailyLoss = MAE::Builder()
                        .network(*network)
                        .predictions(daily.getFeatureOutput().value())
                        .labels(observedDaily.getFeatureOutput().value())
                        .lossDataType(DataType::FP32)
                        .lossWeight(dailyLossWeight)
                        .build();
    MSE aggregateLoss = MSE::Builder()
                            .network(*network)
                            .predictions(aggregate.getFeatureOutput().value())
                            .labels(observedAggregate.getFeatureOutput().value())
                            .lossDataType(DataType::FP32)
                            .lossWeight(1.5f)
                            .build();
    QuantileLoss p90Loss = QuantileLoss::Builder()
                               .network(*network)
                               .predictions(p90.getFeatureOutput().value())
                               .labels(observedDaily.getFeatureOutput().value())
                               .quantile(0.9f)
                               .lossDataType(DataType::FP32)
                               .lossWeight(0.5f)
                               .build();

    NetworkOutput::Builder().network(*network).name("daily").inputTensor(daily.getFeatureOutput().value()).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("aggregate").inputTensor(aggregate.getFeatureOutput().value()).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("forecast_p90").inputTensor(p90.getFeatureOutput().value()).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("daily_loss").inputTensor(dailyLoss.getLoss()).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("aggregate_loss").inputTensor(aggregateLoss.getLoss()).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("p90_loss").inputTensor(p90Loss.getLoss()).dataType(DataType::FP32).build();
    return network;
}

std::shared_ptr<Network> makeDemandPredictionOnlyNetwork(const std::string& name,
                                                         std::vector<uint64_t> dailyOutputDimensions = {1},
                                                         bool includeDailyOutput = true) {
    auto network = std::make_shared<Network>(name);
    NetworkInput::Builder().network(*network).name("features").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput::Builder().network(*network).name("observed_daily").dimensions({1}).dataType(DataType::FP32).build();
    NetworkInput::Builder().network(*network).name("observed_aggregate").dimensions({1}).dataType(DataType::FP32).build();

    if (includeDailyOutput) {
        Tensor dailyTensor(DataType::FP32, std::move(dailyOutputDimensions));
        NetworkOutput::Builder().network(*network).name("daily").inputTensor(dailyTensor).dataType(DataType::FP32).build();
    }
    Tensor aggregateTensor(DataType::FP32, {1});
    Tensor p90Tensor(DataType::FP32, {1});
    NetworkOutput::Builder().network(*network).name("aggregate").inputTensor(aggregateTensor).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("forecast_p90").inputTensor(p90Tensor).dataType(DataType::FP32).build();
    return network;
}

std::shared_ptr<Network> makeAmbiguousDailyLossNetwork(const std::string& name) {
    auto network = std::make_shared<Network>(name);
    NetworkInput features = NetworkInput::Builder().network(*network).name("features").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput observedDaily = NetworkInput::Builder().network(*network).name("observed_daily").dimensions({1}).dataType(DataType::FP32).build();

    FullyConnected daily = FullyConnected::Builder()
                               .network(*network)
                               .featureInput(features.getFeatureOutput().value())
                               .numOutputFeatures(1)
                               .hasBias(true)
                               .noActivation()
                               .build();

    MAE::Builder()
        .network(*network)
        .predictions(daily.getFeatureOutput().value())
        .labels(observedDaily.getFeatureOutput().value())
        .lossDataType(DataType::FP32)
        .lossWeight(2.0f)
        .build();
    MAE::Builder()
        .network(*network)
        .predictions(daily.getFeatureOutput().value())
        .labels(observedDaily.getFeatureOutput().value())
        .lossDataType(DataType::FP32)
        .lossWeight(3.0f)
        .build();

    NetworkOutput::Builder().network(*network).name("daily").inputTensor(daily.getFeatureOutput().value()).dataType(DataType::FP32).build();
    return network;
}


std::shared_ptr<Trainer> makeTrainer(std::shared_ptr<Network> network,
                                    std::shared_ptr<TrainingExecutor> executor,
                                    std::optional<std::string> saveModelDirectory = std::nullopt,
                                    std::vector<TrainingRestartCondition> restartConditions = {},
                                    std::vector<TrainingEarlyCompletionPolicy> earlyCompletionPolicies = {}) {
    return std::make_shared<Trainer>(Trainer::Builder()
                                         .network(std::move(network))
                                         .loader(std::make_shared<FakeLoader>())
                                         .executor(std::move(executor))
                                         .observer(std::make_shared<NullTrainingObserver>())
                                         .saveModelDirectory(std::move(saveModelDirectory))
                                         .restartConditions(std::move(restartConditions))
                                         .earlyCompletionPolicies(std::move(earlyCompletionPolicies))
                                         .build());
}

void rethrowIfSet(std::exception_ptr exception) {
    if (exception != nullptr) {
        std::rethrow_exception(exception);
    }
}

std::filesystem::path uniqueTempPath(const std::string& prefix) {
    return std::filesystem::temp_directory_path() /
           (prefix + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}


}  // namespace


TEST(TrainingRuns, LossReferencesExposeLossWeightsForNamedOutputs) {
    auto network = makeLossWeightedDemandNetwork("training-runs-loss-metric-hints");

    const std::map<std::string, std::vector<NetworkLossReference>> references = network->getLossReferencesByPredictionOutputName();

    auto dailyIt = references.find("daily");
    ASSERT_NE(dailyIt, references.end());
    ASSERT_EQ(dailyIt->second.size(), 1u);
    EXPECT_EQ(dailyIt->second[0].targetInputName, "observed_daily");
    EXPECT_EQ(dailyIt->second[0].lossLayerType, "CustomLoss");
    EXPECT_DOUBLE_EQ(dailyIt->second[0].lossWeight, 2.0);

    auto aggregateIt = references.find("aggregate");
    ASSERT_NE(aggregateIt, references.end());
    ASSERT_EQ(aggregateIt->second.size(), 1u);
    EXPECT_EQ(aggregateIt->second[0].targetInputName, "observed_aggregate");
    EXPECT_EQ(aggregateIt->second[0].lossLayerType, "CustomLoss");
    EXPECT_DOUBLE_EQ(aggregateIt->second[0].lossWeight, 1.5);

    auto p90It = references.find("forecast_p90");
    ASSERT_NE(p90It, references.end());
    ASSERT_EQ(p90It->second.size(), 1u);
    EXPECT_EQ(p90It->second[0].targetInputName, "observed_daily");
    EXPECT_EQ(p90It->second[0].lossLayerType, "CustomLoss");
    EXPECT_FALSE(p90It->second[0].quantile.has_value());
    EXPECT_DOUBLE_EQ(p90It->second[0].lossWeight, 0.5);
}

TEST(TrainingRuns, RejectsInvalidRunSpecs) {
    auto network = std::make_shared<Network>("training-runs-invalid");
    auto coordinator = std::make_shared<Coordinator>(1);
    auto executor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    auto emptyRuns = []() { return TrainingRuns(std::vector<TrainingRunsSpec>{}); };
    auto emptyName = [&]() { return TrainingRuns(std::vector<TrainingRunsSpec>{TrainingRunsSpec{"", trainer}}); };
    auto nullTrainer = []() {
        return TrainingRuns(std::vector<TrainingRunsSpec>{TrainingRunsSpec{"fold_0", std::shared_ptr<Trainer>{}}});
    };
    auto duplicateNames = [&]() {
        return TrainingRuns(std::vector<TrainingRunsSpec>{TrainingRunsSpec{"fold", trainer}, TrainingRunsSpec{"fold", trainer}});
    };
    auto duplicateTrainer = [&]() {
        return TrainingRuns(std::vector<TrainingRunsSpec>{TrainingRunsSpec{"fold_0", trainer}, TrainingRunsSpec{"fold_1", trainer}});
    };
    auto invalidSummaryRate = [&]() {
        return TrainingRuns(std::vector<TrainingRunsSpec>{TrainingRunsSpec{"fold_0", trainer}},
                            TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                            -1.0);
    };
    auto invalidMaxParallelRuns = [&]() {
        return TrainingRuns(std::vector<TrainingRunsSpec>{TrainingRunsSpec{"fold_0", trainer}},
                            TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                            2.0,
                            0u);
    };

    EXPECT_THROW(emptyRuns(), std::runtime_error);
    EXPECT_THROW(emptyName(), std::runtime_error);
    EXPECT_THROW(nullTrainer(), std::runtime_error);
    EXPECT_THROW(duplicateNames(), std::runtime_error);
    EXPECT_THROW(duplicateTrainer(), std::runtime_error);
    EXPECT_THROW(invalidSummaryRate(), std::runtime_error);
    EXPECT_THROW(invalidMaxParallelRuns(), std::runtime_error);
}



TEST(TrainingRuns, AcceptsReportedLossNameFilter) {
    auto network0 = makeLossWeightedDemandNetwork("training-runs-reported-loss-policy-0");
    auto network1 = makeLossWeightedDemandNetwork("training-runs-reported-loss-policy-1");
    auto coordinator = std::make_shared<Coordinator>(2);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "demand"}, TrainingRunsSpec{"fold_1", trainer1, "demand"}},
                      TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                      2.0,
                      std::nullopt,
                      {},
                      {},
                      {},
                      {{"demand", {"daily_loss", "p90_loss"}}});

    const auto& reportedLosses = runs.getReportedLosses();
    ASSERT_EQ(reportedLosses.size(), 1u);
    const auto lossesIt = reportedLosses.find("demand");
    ASSERT_NE(lossesIt, reportedLosses.end());
    ASSERT_EQ(lossesIt->second.size(), 2u);
    EXPECT_EQ(lossesIt->second[0], "daily_loss");
    EXPECT_EQ(lossesIt->second[1], "p90_loss");
}


TEST(TrainingRuns, OmittedReportedLossesResolveAllGraphLosses) {
    auto network = makeLossWeightedDemandNetwork("training-runs-default-all-graph-losses");
    auto coordinator = std::make_shared<Coordinator>(1);
    auto executor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer, "demand"}});

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            TrainingRunsEvaluationOptions evaluationOptions;
            evaluationOptions.evaluateTrainingPopulation = false;
            result = runs.fit(TrainerFitOptions{1}, evaluationOptions);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    const TrainingEnsembleResult& ensemble = result->ensemble("demand");
    ASSERT_EQ(ensemble.namedMetrics.size(), 3u);
    EXPECT_EQ(ensemble.namedMetrics[0].name, "aggregate_loss");
    EXPECT_EQ(ensemble.namedMetrics[1].name, "daily_loss");
    EXPECT_EQ(ensemble.namedMetrics[2].name, "p90_loss");
}

TEST(TrainingRuns, ReportedLossFilterControlsNamedGraphLossesInResults) {
    auto network = makeLossWeightedDemandNetwork("training-runs-filtered-graph-losses");
    auto coordinator = std::make_shared<Coordinator>(1);
    auto executor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer, "demand"}},
                      TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                      2.0,
                      std::nullopt,
                      {},
                      {},
                      {},
                      {{"demand", {"daily_loss", "p90_loss"}}});

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            TrainingRunsEvaluationOptions evaluationOptions;
            evaluationOptions.evaluateTrainingPopulation = false;
            result = runs.fit(TrainerFitOptions{1}, evaluationOptions);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    const TrainingEnsembleResult& ensemble = result->ensemble("demand");
    ASSERT_EQ(ensemble.namedMetrics.size(), 2u);
    EXPECT_EQ(ensemble.namedMetrics[0].name, "daily_loss");
    EXPECT_EQ(ensemble.namedMetrics[1].name, "p90_loss");
}

TEST(TrainingRuns, ComposedLossEvaluatorExposesGraphLossInputs) {
    auto network0 = makeLossWeightedDemandNetwork("training-runs-composed-loss-inputs-0");
    auto network1 = makeLossWeightedDemandNetwork("training-runs-composed-loss-inputs-1");

    std::vector<std::string> inputNames = Thor::Testing::trainingRunsComposedLossEvaluatorExternalInputNamesForTest(
        {network0, network1}, {1.0, 2.0}, {});
    std::sort(inputNames.begin(), inputNames.end());

    const std::vector<std::string> expectedInputs{"features", "observed_aggregate", "observed_daily"};
    EXPECT_EQ(inputNames, expectedInputs);
}

TEST(TrainingRuns, PredictionOnlyEnsembleHasNoGraphLossMetrics) {
    auto network0 = makeDemandPredictionOnlyNetwork("training-runs-prediction-only-no-graph-loss-0");
    auto network1 = makeDemandPredictionOnlyNetwork("training-runs-prediction-only-no-graph-loss-1");
    auto coordinator = std::make_shared<Coordinator>(2);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "demand"}, TrainingRunsSpec{"fold_1", trainer1, "demand"}});

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            TrainingRunsEvaluationOptions evaluationOptions;
            evaluationOptions.evaluateTrainingPopulation = false;
            result = runs.fit(TrainerFitOptions{1}, evaluationOptions);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    const TrainingEnsembleResult& ensemble = result->ensemble("demand");
    EXPECT_TRUE(ensemble.namedMetrics.empty());
    EXPECT_FALSE(ensemble.ensembleTrainingLoss.has_value());
    EXPECT_FALSE(ensemble.ensembleTestLoss.has_value());
    EXPECT_FALSE(ensemble.ensembleTestAccuracy.has_value());
    EXPECT_FALSE(ensemble.hasEnsembleEvaluationMetrics());
}

TEST(TrainingRuns, NamedMetricResultsUseGraphLossesAndSourceLossWeight) {
    auto network = makeLossWeightedDemandNetwork("training-runs-reported-loss-weight-resolution");
    auto coordinator = std::make_shared<Coordinator>(1);
    auto executor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer, "demand"}},
                      TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                      2.0,
                      std::nullopt,
                      {},
                      {},
                      {},
                      {{"demand", {}}});

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            TrainingRunsEvaluationOptions evaluationOptions;
            evaluationOptions.evaluateTrainingPopulation = false;
            result = runs.fit(TrainerFitOptions{1}, evaluationOptions);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    const TrainingEnsembleResult& ensemble = result->ensemble("demand");
    ASSERT_EQ(ensemble.namedMetrics.size(), 3u);

    EXPECT_EQ(ensemble.namedMetrics[0].name, "aggregate_loss");
    EXPECT_EQ(ensemble.namedMetrics[0].outputName, "aggregate");
    EXPECT_EQ(ensemble.namedMetrics[0].targetInputName, "observed_aggregate");
    EXPECT_DOUBLE_EQ(ensemble.namedMetrics[0].overallWeight, 1.5);
    EXPECT_EQ(ensemble.namedMetrics[0].overallWeightSource, "loss_weight");

    EXPECT_EQ(ensemble.namedMetrics[1].name, "daily_loss");
    EXPECT_EQ(ensemble.namedMetrics[1].outputName, "daily");
    EXPECT_EQ(ensemble.namedMetrics[1].targetInputName, "observed_daily");
    EXPECT_DOUBLE_EQ(ensemble.namedMetrics[1].overallWeight, 2.0);
    EXPECT_EQ(ensemble.namedMetrics[1].overallWeightSource, "loss_weight");

    EXPECT_EQ(ensemble.namedMetrics[2].name, "p90_loss");
    EXPECT_EQ(ensemble.namedMetrics[2].outputName, "forecast_p90");
    EXPECT_EQ(ensemble.namedMetrics[2].targetInputName, "observed_daily");
    EXPECT_DOUBLE_EQ(ensemble.namedMetrics[2].overallWeight, 0.5);
    EXPECT_EQ(ensemble.namedMetrics[2].overallWeightSource, "loss_weight");
}

TEST(TrainingRuns, ReportedLossResolutionFailsFastForMissingAndAmbiguousGraphLosses) {
    auto signatureOnlyNetwork = makeDemandSignatureNetwork("training-runs-missing-reported-loss");
    auto signatureCoordinator = std::make_shared<Coordinator>(1);
    auto signatureExecutor = std::make_shared<CoordinatedExecutor>(signatureCoordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> signatureTrainer = makeTrainer(signatureOnlyNetwork, signatureExecutor);

    EXPECT_THROW((TrainingRuns({TrainingRunsSpec{"fold_0", signatureTrainer, "demand"}},
                               TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                               2.0,
                               std::nullopt,
                               {},
                               {},
                               {},
                               {{"demand", {"daily_loss"}}})),
                 std::runtime_error);
}


TEST(TrainingEnsembleResult, NamedMetricValuesContributeToEvaluationMetricPresence) {
    TrainingEnsembleResult ensemble;
    EXPECT_FALSE(ensemble.hasNamedMetricValues());
    EXPECT_FALSE(ensemble.hasEnsembleEvaluationMetrics());

    TrainingNamedMetricResult metric;
    metric.name = "daily_loss";
    metric.outputName = "daily";
    metric.targetInputName = "observed_daily";
    metric.overallWeight = 2.0;
    metric.overallWeightSource = "loss_weight";
    metric.testValue = 0.25;
    ensemble.namedMetrics.push_back(metric);

    ASSERT_EQ(ensemble.namedMetrics.size(), 1u);
    EXPECT_TRUE(ensemble.namedMetrics[0].hasValue());
    EXPECT_TRUE(ensemble.hasNamedMetricValues());
    EXPECT_TRUE(ensemble.hasEnsembleEvaluationMetrics());
    EXPECT_EQ(ensemble.namedMetrics[0].name, "daily_loss");
    EXPECT_EQ(ensemble.namedMetrics[0].outputName, "daily");
    EXPECT_EQ(ensemble.namedMetrics[0].targetInputName, "observed_daily");
    EXPECT_DOUBLE_EQ(ensemble.namedMetrics[0].overallWeight, 2.0);
    EXPECT_EQ(ensemble.namedMetrics[0].overallWeightSource, "loss_weight");
    ASSERT_TRUE(ensemble.namedMetrics[0].testValue.has_value());
    EXPECT_DOUBLE_EQ(ensemble.namedMetrics[0].testValue.value(), 0.25);
}

TEST(TrainingRuns, WeightedLossSumUsesAlreadyWeightedNamedLossValues) {
    const std::vector<std::optional<double>> weightedLossValues{0.20, 0.60, 0.15};

    std::optional<double> overall = Thor::Testing::trainingRunsWeightedLossSumFromWeightedLossValuesForTest(
        weightedLossValues);

    ASSERT_TRUE(overall.has_value());
    EXPECT_NEAR(overall.value(), 0.20 + 0.60 + 0.15, 1.0e-12);
}

TEST(TrainingRuns, WeightedLossSumRejectsMissingOrNonFiniteValues) {
    EXPECT_FALSE(Thor::Testing::trainingRunsWeightedLossSumFromWeightedLossValuesForTest({0.20, std::nullopt}).has_value());
    EXPECT_FALSE(Thor::Testing::trainingRunsWeightedLossSumFromWeightedLossValuesForTest({0.20, std::numeric_limits<double>::infinity()}).has_value());
    EXPECT_FALSE(Thor::Testing::trainingRunsWeightedLossSumFromWeightedLossValuesForTest({}).has_value());
}

TEST(TrainingRuns, RejectsIncompatibleEnsembleOutputDimensions) {
    auto network0 = makeNetworkWithOutput("training-runs-ensemble-output-0", {0, 10});
    auto network1 = makeNetworkWithOutput("training-runs-ensemble-output-1", {0, 11});
    auto coordinator = std::make_shared<Coordinator>(2);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);

    EXPECT_THROW((TrainingRuns(std::vector<TrainingRunsSpec>{TrainingRunsSpec{"fold_0", trainer0, "digits"},
                                                            TrainingRunsSpec{"fold_1", trainer1, "digits"}})),
                 std::runtime_error);
}

TEST(TrainingRunsResult, ReportsEnsembleMetadata) {
    auto network0 = makeNetworkWithOutput("training-runs-ensemble-result-0", {0, 10});
    auto network1 = makeNetworkWithOutput("training-runs-ensemble-result-1", {0, 10});
    auto coordinator = std::make_shared<Coordinator>(2);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 0);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 2);
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);
    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "digits", 1.0},
                       TrainingRunsSpec{"fold_1", trainer1, "digits", 3.0}});

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            TrainingRunsEvaluationOptions evaluationOptions;
            evaluationOptions.evaluateTrainingPopulation = false;
            result = runs.fit(TrainerFitOptions{1}, evaluationOptions);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->hasEnsembles());
    const TrainingEnsembleResult& ensemble = result->ensemble("digits");
    EXPECT_TRUE(ensemble.allCompleted());
    EXPECT_EQ(ensemble.members.size(), 2u);
    ASSERT_EQ(ensemble.outputSignature.size(), 1u);
    EXPECT_EQ(ensemble.outputSignature[0].outputName, "predictions");
    EXPECT_EQ(ensemble.outputSignature[0].dimensions, (std::vector<uint64_t>{0, 10}));
    EXPECT_DOUBLE_EQ(ensemble.totalWeight(), 4.0);
    ASSERT_EQ(ensemble.inputSignature.size(), 1u);
    EXPECT_EQ(ensemble.inputSignature[0].inputName, "features");
}


TEST(TrainingRunsResult, SaveEnsembleAllowsPartialSuccessWhenMinimumSatisfied) {
    const std::filesystem::path root = uniqueTempPath("thor-training-runs-partial-ensemble-save");
    const std::filesystem::path fold0 = root / "fold_0";
    const std::filesystem::path fold2 = root / "fold_2";
    const std::filesystem::path ensembleDir = root / "ensemble";
    std::filesystem::create_directories(fold0);
    std::filesystem::create_directories(fold2);
    {
        std::ofstream(fold0 / "training_selection_metadata.json") << "{}\n";
        std::ofstream(fold2 / "training_selection_metadata.json") << "{}\n";
    }

    TrainingRunResult result0 = TrainingRunResult::completedResult("fold_0", {}, {}, {}, TrainingRunCompletionReason::COMPLETED, 1, 1, 1.0, fold0.string());
    TrainingRunResult result1 = TrainingRunResult::fromException("fold_1", std::make_exception_ptr(std::runtime_error("planned failure")));
    TrainingRunResult result2 = TrainingRunResult::completedResult("fold_2", {}, {}, {}, TrainingRunCompletionReason::COMPLETED, 1, 1, 1.0, fold2.string());
    result0.ensembleGroup = "digits";
    result1.ensembleGroup = "digits";
    result2.ensembleGroup = "digits";

    TrainingEnsembleResult ensemble;
    ensemble.ensembleGroup = "digits";
    ensemble.minSuccessfulModels = 2;
    ensemble.members = {
        TrainingEnsembleMemberResult{"fold_0", 1.0, TrainingRunStatus::COMPLETED},
        TrainingEnsembleMemberResult{"fold_1", 1.0, TrainingRunStatus::FAILED},
        TrainingEnsembleMemberResult{"fold_2", 1.0, TrainingRunStatus::COMPLETED},
    };
    ensemble.outputSignature = {TrainingRunOutputSignature{"predictions", {0, 10}, "FP32"}};

    TrainingRunsResult results({result0, result1, result2}, {ensemble});
    const std::string manifestPath = results.saveEnsemble("digits", ensembleDir.string());

    EXPECT_EQ(manifestPath, (ensembleDir / "ensemble_manifest.json").string());
    EXPECT_TRUE(std::filesystem::exists(ensembleDir / "members" / "0000_fold_0" / "training_selection_metadata.json"));
    EXPECT_TRUE(std::filesystem::exists(ensembleDir / "members" / "0001_fold_2" / "training_selection_metadata.json"));
    EXPECT_FALSE(std::filesystem::exists(ensembleDir / "members" / "0002_fold_1"));

    std::ifstream manifest(manifestPath);
    const std::string text((std::istreambuf_iterator<char>(manifest)), std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("\"target_num_members\": 3"), std::string::npos);
    EXPECT_NE(text.find("\"actual_num_members\": 2"), std::string::npos);
    EXPECT_NE(text.find("\"min_successful_models\": 2"), std::string::npos);
    EXPECT_NE(text.find("\"name\": \"fold_0\""), std::string::npos);
    EXPECT_NE(text.find("\"name\": \"fold_2\""), std::string::npos);
    EXPECT_EQ(text.find("\"name\": \"fold_1\""), std::string::npos);

    std::filesystem::remove_all(root);
}

TEST(TrainingRuns, StartsAllTrainersConcurrentlyAndReturnsCompletedResults) {
    auto network0 = std::make_shared<Network>("training-runs-concurrent-0");
    auto network1 = std::make_shared<Network>("training-runs-concurrent-1");
    auto coordinator = std::make_shared<Coordinator>(2);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 3);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 5);
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);
    TrainingRuns runs({{"fold_0", trainer0}, {"fold_1", trainer1}});

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(1);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    EXPECT_EQ(coordinator->startedCount(), 2u);
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->allCompleted());
    EXPECT_FALSE(result->anyFailed());
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0].runName, "fold_0");
    EXPECT_EQ((*result)[1].runName, "fold_1");
    EXPECT_EQ((*result)["fold_0"].status, TrainingRunStatus::COMPLETED);
    EXPECT_EQ((*result)["fold_1"].status, TrainingRunStatus::COMPLETED);
    ASSERT_TRUE((*result)["fold_0"].finalTrainingStats.has_value());
    ASSERT_TRUE((*result)["fold_1"].finalTrainingStats.has_value());
    EXPECT_EQ((*result)["fold_0"].finalTrainingStats->step, 3u);
    EXPECT_EQ((*result)["fold_1"].finalTrainingStats->step, 5u);
    EXPECT_EQ(executor0->calls, 1u);
    EXPECT_EQ(executor1->calls, 1u);
}


TEST(TrainingRuns, MaxParallelRunsLimitsConcurrentStarts) {
    auto network0 = std::make_shared<Network>("training-runs-max-parallel-0");
    auto network1 = std::make_shared<Network>("training-runs-max-parallel-1");
    auto network2 = std::make_shared<Network>("training-runs-max-parallel-2");
    auto coordinator = std::make_shared<Coordinator>(3);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 1);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 2);
    auto executor2 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 3);
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);
    std::shared_ptr<Trainer> trainer2 = makeTrainer(network2, executor2);
    TrainingRuns runs({{"fold_0", trainer0}, {"fold_1", trainer1}, {"fold_2", trainer2}},
                      TrainingRunsFailurePolicy::CONTINUE,
                      2.0,
                      1u);

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(1);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForStartedCount(1));
    EXPECT_EQ(coordinator->startedCount(), 1u);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    EXPECT_EQ(coordinator->startedCount(), 1u);
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->allCompleted());
    EXPECT_EQ(coordinator->startedCount(), 3u);
    EXPECT_EQ(executor0->calls, 1u);
    EXPECT_EQ(executor1->calls, 1u);
    EXPECT_EQ(executor2->calls, 1u);
}

TEST(TrainingRuns, ContinuePolicyAllowsSiblingsToFinishAfterFailure) {
    auto network0 = std::make_shared<Network>("training-runs-continue-0");
    auto network1 = std::make_shared<Network>("training-runs-continue-1");
    auto coordinator = std::make_shared<Coordinator>(2);
    auto failingExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::FAIL_AFTER_RELEASE, 2);
    auto completingExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 7);
    std::shared_ptr<Trainer> failingTrainer = makeTrainer(network0, failingExecutor);
    std::shared_ptr<Trainer> completingTrainer = makeTrainer(network1, completingExecutor);
    TrainingRuns runs({{"bad_arch", failingTrainer}, {"good_arch", completingTrainer}}, TrainingRunsFailurePolicy::CONTINUE);

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(1);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->anyFailed());
    EXPECT_FALSE(result->anyCancelled());
    EXPECT_EQ((*result)["bad_arch"].status, TrainingRunStatus::FAILED);
    EXPECT_EQ((*result)["good_arch"].status, TrainingRunStatus::COMPLETED);
    EXPECT_EQ((*result)["bad_arch"].exception.message, "planned trainer failure");
}

TEST(TrainingRuns, CancelSiblingsPolicyRequestsCancellationAfterFailure) {
    auto network0 = std::make_shared<Network>("training-runs-cancel-0");
    auto network1 = std::make_shared<Network>("training-runs-cancel-1");
    auto coordinator = std::make_shared<Coordinator>(2);
    auto failingExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::FAIL_AFTER_RELEASE, 2);
    auto waitingExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::WAIT_FOR_CANCEL_THEN_CANCEL, 7);
    std::shared_ptr<Trainer> failingTrainer = makeTrainer(network0, failingExecutor);
    std::shared_ptr<Trainer> waitingTrainer = makeTrainer(network1, waitingExecutor);
    TrainingRuns runs({{"bad_fold", failingTrainer}, {"sibling_fold", waitingTrainer}}, TrainingRunsFailurePolicy::CANCEL_SIBLINGS);

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(1);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->anyFailed());
    EXPECT_TRUE(result->anyCancelled());
    EXPECT_EQ((*result)["bad_fold"].status, TrainingRunStatus::FAILED);
    EXPECT_EQ((*result)["sibling_fold"].status, TrainingRunStatus::CANCELLED);
}


TEST(TrainingRuns, MinSuccessfulModelsToleratesFailureWhileEnsembleRemainsViable) {
    auto network0 = makeNetworkWithOutput("training-runs-min-success-viable-0", {0, 10});
    auto network1 = makeNetworkWithOutput("training-runs-min-success-viable-1", {0, 10});
    auto network2 = makeNetworkWithOutput("training-runs-min-success-viable-2", {0, 10});
    auto coordinator = std::make_shared<Coordinator>(3);
    auto failingExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::FAIL_AFTER_RELEASE, 2);
    auto completingExecutor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 7);
    auto completingExecutor2 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 9);
    std::shared_ptr<Trainer> failingTrainer = makeTrainer(network0, failingExecutor);
    std::shared_ptr<Trainer> completingTrainer1 = makeTrainer(network1, completingExecutor1);
    std::shared_ptr<Trainer> completingTrainer2 = makeTrainer(network2, completingExecutor2);
    TrainingRuns runs({TrainingRunsSpec{"fold_0", failingTrainer, "digits"},
                       TrainingRunsSpec{"fold_1", completingTrainer1, "digits"},
                       TrainingRunsSpec{"fold_2", completingTrainer2, "digits"}},
                      TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                      2.0,
                      2u,
                      {},
                      {},
                      {{"digits", 2}});

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            TrainingRunsEvaluationOptions evaluationOptions;
            evaluationOptions.evaluateTrainingPopulation = false;
            result = runs.fit(TrainerFitOptions{1}, evaluationOptions);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForStartedCount(2));
    coordinator->releaseAll();
    ASSERT_TRUE(coordinator->waitForStartedCount(3));
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->anyFailed());
    EXPECT_FALSE(result->anyCancelled());
    EXPECT_EQ((*result)["fold_0"].status, TrainingRunStatus::FAILED);
    EXPECT_EQ((*result)["fold_1"].status, TrainingRunStatus::COMPLETED);
    EXPECT_EQ((*result)["fold_2"].status, TrainingRunStatus::COMPLETED);
    const TrainingEnsembleResult& ensemble = result->ensemble("digits");
    EXPECT_FALSE(ensemble.allCompleted());
    EXPECT_TRUE(ensemble.anyFailed());
    EXPECT_EQ(ensemble.successfulModels(), 2u);
    EXPECT_EQ(ensemble.requiredSuccessfulModels(), 2u);
    EXPECT_TRUE(ensemble.hasEnoughSuccessfulModels());
}

TEST(TrainingRuns, MinSuccessfulModelsCancelsWhenFailureMakesEnsembleImpossible) {
    auto network0 = makeNetworkWithOutput("training-runs-min-success-impossible-0", {0, 10});
    auto network1 = makeNetworkWithOutput("training-runs-min-success-impossible-1", {0, 10});
    auto network2 = makeNetworkWithOutput("training-runs-min-success-impossible-2", {0, 10});
    auto coordinator = std::make_shared<Coordinator>(2);
    auto failingExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::FAIL_AFTER_RELEASE, 2);
    auto waitingExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::WAIT_FOR_CANCEL_THEN_CANCEL, 7);
    auto notStartedExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE, 9);
    std::shared_ptr<Trainer> failingTrainer = makeTrainer(network0, failingExecutor);
    std::shared_ptr<Trainer> waitingTrainer = makeTrainer(network1, waitingExecutor);
    std::shared_ptr<Trainer> notStartedTrainer = makeTrainer(network2, notStartedExecutor);
    TrainingRuns runs({TrainingRunsSpec{"fold_0", failingTrainer, "digits"},
                       TrainingRunsSpec{"fold_1", waitingTrainer, "digits"},
                       TrainingRunsSpec{"fold_2", notStartedTrainer, "digits"}},
                      TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                      2.0,
                      2u,
                      {},
                      {},
                      {{"digits", 3}});

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            TrainingRunsEvaluationOptions evaluationOptions;
            evaluationOptions.evaluateTrainingPopulation = false;
            result = runs.fit(TrainerFitOptions{1}, evaluationOptions);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForStartedCount(2));
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->anyFailed());
    EXPECT_TRUE(result->anyCancelled());
    EXPECT_EQ((*result)["fold_0"].status, TrainingRunStatus::FAILED);
    EXPECT_EQ((*result)["fold_1"].status, TrainingRunStatus::CANCELLED);
    EXPECT_EQ((*result)["fold_2"].status, TrainingRunStatus::CANCELLED);
    EXPECT_EQ(notStartedExecutor->calls, 0u);
    const TrainingEnsembleResult& ensemble = result->ensemble("digits");
    EXPECT_EQ(ensemble.successfulModels(), 0u);
    EXPECT_EQ(ensemble.requiredSuccessfulModels(), 3u);
    EXPECT_FALSE(ensemble.hasEnoughSuccessfulModels());
}

TEST(TrainingRuns, ClassifiesOutOfMemoryAndCancelsSiblings) {
    auto network0 = std::make_shared<Network>("training-runs-oom-0");
    auto network1 = std::make_shared<Network>("training-runs-oom-1");
    auto coordinator = std::make_shared<Coordinator>(2);
    auto oomExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::OOM_AFTER_RELEASE, 2);
    auto waitingExecutor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::WAIT_FOR_CANCEL_THEN_CANCEL, 7);
    std::shared_ptr<Trainer> oomTrainer = makeTrainer(network0, oomExecutor);
    std::shared_ptr<Trainer> waitingTrainer = makeTrainer(network1, waitingExecutor);
    TrainingRuns runs({{"oom_fold", oomTrainer}, {"sibling_fold", waitingTrainer}});

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(1);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["oom_fold"].status, TrainingRunStatus::OUT_OF_MEMORY);
    EXPECT_EQ((*result)["sibling_fold"].status, TrainingRunStatus::CANCELLED);
}

TEST(TrainingRuns, RejectsInvalidFitOptionsBeforeLaunchingThreads) {
    auto network = std::make_shared<Network>("training-runs-fit-options");
    auto coordinator = std::make_shared<Coordinator>(1);
    auto executor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRuns runs({{"fold_0", trainer}});

    EXPECT_THROW(static_cast<void>(runs.fit(0)), std::runtime_error);
    EXPECT_EQ(coordinator->startedCount(), 0u);
    EXPECT_EQ(executor->calls, 0u);
}


TEST(Trainer, RestartConditionResetsEpochCountAndStateBeforeRetry) {
    auto network = std::make_shared<Network>("trainer-restart-global-epoch-reset");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0}, {100.0}, {100.0, 90.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(
        network,
        executor,
        std::nullopt,
        {TrainingRestartCondition{/*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/1}});

    trainer->fit(1);
    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 0u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 1u);

    TrainingRunResult result = trainer->fit(1);
    EXPECT_EQ(result.status, TrainingRunStatus::COMPLETED);
    EXPECT_EQ(executor->calls, 3u);
    ASSERT_EQ(executor->initialCompletedEpochsByCall.size(), 3u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[0], 0u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[1], 1u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[2], 0u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 1u);
}


TEST(TrainingRuns, RestartPolicyResetsEpochCountAndStateBeforeRetry) {
    auto network = std::make_shared<Network>("training-runs-restart-global-epoch-reset");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0}, {100.0}, {100.0, 90.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRunsRestartPolicy condition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/1);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {condition});
    TrainingRunsEvaluationOptions evaluationOptions;
    evaluationOptions.evaluateTrainingPopulation = false;

    TrainingRunsResult firstResult = runs.fit(TrainerFitOptions{1}, evaluationOptions);
    EXPECT_TRUE(firstResult.allCompleted());
    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 0u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 1u);

    TrainingRunsResult secondResult = runs.fit(TrainerFitOptions{1}, evaluationOptions);
    ASSERT_TRUE(secondResult.allCompleted());
    EXPECT_EQ(executor->calls, 3u);
    ASSERT_EQ(executor->initialCompletedEpochsByCall.size(), 3u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[0], 0u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[1], 1u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[2], 0u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 1u);
}

TEST(TrainingRuns, RestartPolicyDoesNotRecheckPastCumulativeEpochOnLaterFit) {
    auto network = std::make_shared<Network>("training-runs-restart-past-global-epoch");
    auto executor = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}, {100.0, 100.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRunsRestartPolicy condition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/0);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {condition});
    TrainingRunsEvaluationOptions evaluationOptions;
    evaluationOptions.evaluateTrainingPopulation = false;

    TrainingRunsResult firstResult = runs.fit(TrainerFitOptions{2}, evaluationOptions);
    EXPECT_TRUE(firstResult.allCompleted());
    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 0u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 2u);

    TrainingRunsResult secondResult = runs.fit(TrainerFitOptions{2}, evaluationOptions);
    EXPECT_TRUE(secondResult.allCompleted());
    EXPECT_EQ(executor->calls, 2u);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 2u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 4u);
}

TEST(Trainer, RestartConditionRestartsSingleTrainerUntilProgressImproves) {
    auto network = std::make_shared<Network>("trainer-restart-progress");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 98.0, 98.0}, {100.0, 90.0, 85.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(
        network, executor, std::nullopt, {TrainingRestartCondition{/*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/5}});

    trainer->fit(3);

    EXPECT_EQ(executor->calls, 2u);
    EXPECT_TRUE(executor->sawLossRequested);
}

TEST(TrainingRuns, RestartConditionRestartsRunUntilProgressImproves) {
    auto network = std::make_shared<Network>("training-runs-restart-progress");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 98.0, 98.0}, {100.0, 90.0, 85.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRunsRestartPolicy condition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/5);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {condition});

    TrainingRunsResult result = runs.fit(3);

    ASSERT_TRUE(result.allCompleted());
    EXPECT_EQ(executor->calls, 2u);
    EXPECT_TRUE(executor->sawLossRequested);
    ASSERT_TRUE(result["fold_0"].finalLossForPhase(TrainingEventPhase::TRAIN).has_value());
    EXPECT_DOUBLE_EQ(result["fold_0"].finalLossForPhase(TrainingEventPhase::TRAIN).value(), 85.0);
}

TEST(TrainingRuns, RestartConditionExhaustionFailsRunWithAttemptProgress) {
    auto network = std::make_shared<Network>("training-runs-restart-exhausted");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 98.0}, {100.0, 97.0}, {100.0, 96.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRunsRestartPolicy condition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/2);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {condition});

    TrainingRunsResult result = runs.fit(2);

    ASSERT_TRUE(result.anyFailed());
    EXPECT_EQ(executor->calls, 3u);
    const TrainingRunResult& failed = result["fold_0"];
    EXPECT_EQ(failed.status, TrainingRunStatus::FAILED);
    EXPECT_EQ(failed.exception.type, "TrainingRestartConditionExceeded");
    EXPECT_NE(failed.exception.message.find("max_restarts=2"), std::string::npos);
    EXPECT_NE(failed.exception.message.find("attempt 1"), std::string::npos);
    EXPECT_NE(failed.exception.message.find("attempt 2"), std::string::npos);
    EXPECT_NE(failed.exception.message.find("attempt 3"), std::string::npos);
    EXPECT_NE(failed.exception.message.find("observed_progress=2"), std::string::npos);
}

TEST(TrainingRuns, RestartConditionCanTargetEnsembleGroup) {
    auto network0 = makeNetworkWithOutput("training-runs-restart-group-0", {0, 10});
    auto network1 = makeNetworkWithOutput("training-runs-restart-group-1", {0, 10});
    auto executor0 = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 99.0, 99.0}, {100.0, 90.0, 80.0}});
    auto executor1 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0, 80.0}});
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);
    TrainingRunsRestartPolicy condition = TrainingRunsRestartPolicy::forEnsembleGroup(
        "digits", /*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/1);
    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "digits"}, TrainingRunsSpec{"fold_1", trainer1, "digits"}},
                      TrainingRunsFailurePolicy::CONTINUE,
                      0.0,
                      std::nullopt,
                      {condition});

    TrainingRunsEvaluationOptions evaluationOptions;
    evaluationOptions.evaluateTrainingPopulation = false;
    TrainingRunsResult result = runs.fit(TrainerFitOptions{3}, evaluationOptions);

    EXPECT_TRUE(result.allCompleted());
    EXPECT_EQ(executor0->calls, 2u);
    EXPECT_EQ(executor1->calls, 1u);
}

TEST(TrainingRuns, RestartConditionAllowsMultipleConditionsForSameEnsembleGroupWithIndependentBudgets) {
    auto network = makeNetworkWithOutput("training-runs-restart-group-multiple", {0, 10});
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 98.0, 98.0}, {100.0, 90.0, 85.0}, {100.0, 90.0, 70.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRunsRestartPolicy earlyCondition = TrainingRunsRestartPolicy::forEnsembleGroup(
        "digits", /*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/1);
    TrainingRunsRestartPolicy laterCondition = TrainingRunsRestartPolicy::forEnsembleGroup(
        "digits", /*progressCheckEpochs=*/3, /*progressImprovementMinPercentage=*/20.0, /*maxRestarts=*/1);
    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer, "digits"}},
                      TrainingRunsFailurePolicy::CONTINUE,
                      0.0,
                      std::nullopt,
                      {earlyCondition, laterCondition});

    TrainingRunsEvaluationOptions evaluationOptions;
    evaluationOptions.evaluateTrainingPopulation = false;
    TrainingRunsResult result = runs.fit(TrainerFitOptions{3}, evaluationOptions);

    EXPECT_TRUE(result.allCompleted());
    EXPECT_EQ(executor->calls, 3u);
    ASSERT_TRUE(result["fold_0"].finalLossForPhase(TrainingEventPhase::TRAIN).has_value());
    EXPECT_DOUBLE_EQ(result["fold_0"].finalLossForPhase(TrainingEventPhase::TRAIN).value(), 70.0);
}

TEST(TrainingRuns, RestartConditionAllowsMultipleConditionsForSameRunWithIndependentFailureBudgets) {
    auto network = std::make_shared<Network>("training-runs-restart-run-multiple-exhausted");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 98.0, 98.0}, {100.0, 90.0, 85.0}, {100.0, 90.0, 84.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRunsRestartPolicy earlyCondition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/1);
    TrainingRunsRestartPolicy laterCondition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/3, /*progressImprovementMinPercentage=*/20.0, /*maxRestarts=*/1);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {earlyCondition, laterCondition});

    TrainingRunsResult result = runs.fit(3);

    ASSERT_TRUE(result.anyFailed());
    EXPECT_EQ(executor->calls, 3u);
    const TrainingRunResult& failed = result["fold_0"];
    EXPECT_EQ(failed.status, TrainingRunStatus::FAILED);
    EXPECT_EQ(failed.exception.type, "TrainingRestartConditionExceeded");
    EXPECT_NE(failed.exception.message.find("progress_check_epochs=3"), std::string::npos);
    EXPECT_NE(failed.exception.message.find("progress_improvement_min_percentage=20"), std::string::npos);
    EXPECT_NE(failed.exception.message.find("max_restarts=1"), std::string::npos);
    EXPECT_EQ(failed.exception.message.find("progress_check_epochs=2"), std::string::npos);
}

TEST(TrainingRuns, EarlyCompletionRuleCanTargetEnsembleGroup) {
    auto network0 = makeNetworkWithOutput("training-runs-early-completion-group-0", {0, 10});
    auto network1 = makeNetworkWithOutput("training-runs-early-completion-group-1", {0, 10});
    auto executor0 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}});
    auto executor1 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}});
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);
    TrainingRunsEarlyCompletionRule rule = TrainingRunsEarlyCompletionRule::forEnsembleGroup(
        "digits", [](double currentScore, double bestScore, uint64_t currentEpoch, uint64_t bestEpoch) {
            return currentScore > bestScore && currentEpoch > bestEpoch;
        });
    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "digits"}, TrainingRunsSpec{"other", trainer1, "other_group"}},
                      TrainingRunsFailurePolicy::CONTINUE,
                      0.0,
                      std::nullopt,
                      {},
                      {rule});

    TrainingRunsEvaluationOptions evaluationOptions;
    evaluationOptions.evaluateTrainingPopulation = false;
    TrainingRunsResult result = runs.fit(TrainerFitOptions{2}, evaluationOptions);

    EXPECT_TRUE(result.allCompleted());
    EXPECT_EQ(executor0->lastEarlyCompletionPolicyCount, 1u);
    EXPECT_TRUE(executor0->lastEarlyCompletionDecision);
    EXPECT_EQ(executor1->lastEarlyCompletionPolicyCount, 0u);
}

TEST(TrainingRuns, EarlyCompletionRulesCombineTrainerPoliciesRunTargetsAndGroupTargets) {
    auto network0 = makeNetworkWithOutput("training-runs-early-completion-combine-0", {0, 10});
    auto network1 = makeNetworkWithOutput("training-runs-early-completion-combine-1", {0, 10});
    auto executor0 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}});
    auto executor1 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}});

    TrainingEarlyCompletionPolicy trainerPolicy([](double, double, uint64_t, uint64_t) { return false; });
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0, std::nullopt, {}, {trainerPolicy});
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);

    TrainingRunsEarlyCompletionRule runRule = TrainingRunsEarlyCompletionRule::forRun(
        "fold_0", [](double, double, uint64_t, uint64_t) { return false; });
    TrainingRunsEarlyCompletionRule groupRule = TrainingRunsEarlyCompletionRule::forEnsembleGroup(
        "digits", [](double, double, uint64_t, uint64_t) { return false; });

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "digits"}, TrainingRunsSpec{"other", trainer1, "other_group"}},
                      TrainingRunsFailurePolicy::CONTINUE,
                      0.0,
                      std::nullopt,
                      {},
                      {runRule, groupRule});

    TrainingRunsEvaluationOptions evaluationOptions;
    evaluationOptions.evaluateTrainingPopulation = false;
    TrainingRunsResult result = runs.fit(TrainerFitOptions{2}, evaluationOptions);

    EXPECT_TRUE(result.allCompleted());
    EXPECT_EQ(executor0->lastEarlyCompletionPolicyCount, 3u);
    EXPECT_EQ(executor1->lastEarlyCompletionPolicyCount, 0u);
}

TEST(TrainingRuns, RejectsInvalidEarlyCompletionRules) {
    auto network = std::make_shared<Network>("training-runs-early-completion-invalid");
    auto executor = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    TrainingRunsEarlyCompletionRule both([](double, double, uint64_t, uint64_t) { return false; });
    both.runName = "fold_0";
    both.ensembleGroup = "group";
    EXPECT_THROW((TrainingRuns({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {}, {both})),
                 std::runtime_error);

    TrainingRunsEarlyCompletionRule unknown = TrainingRunsEarlyCompletionRule::forRun(
        "missing", [](double, double, uint64_t, uint64_t) { return false; });
    EXPECT_THROW((TrainingRuns({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {}, {unknown})),
                 std::runtime_error);

    TrainingRunsEarlyCompletionRule invalid;
    invalid.runName = "fold_0";
    EXPECT_THROW((TrainingRuns({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {}, {invalid})),
                 std::runtime_error);
}

TEST(TrainingRuns, RejectsInvalidRestartConditions) {
    auto network = std::make_shared<Network>("training-runs-restart-invalid");
    auto executor = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    TrainingRunsRestartPolicy both;
    both.runName = "fold_0";
    both.ensembleGroup = "group";
    EXPECT_THROW((TrainingRuns({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {both})),
                 std::runtime_error);

    TrainingRunsRestartPolicy unknown = TrainingRunsRestartPolicy::forRun("missing");
    EXPECT_THROW((TrainingRuns({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {unknown})),
                 std::runtime_error);

    TrainingRunsRestartPolicy invalidProgress = TrainingRunsRestartPolicy::forRun("fold_0");
    invalidProgress.progressCheckEpochs = 0;
    EXPECT_THROW((TrainingRuns({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {invalidProgress})),
                 std::runtime_error);
}

TEST(TrainingRunsResult, ReportsStatusCounts) {
    TrainingRunResult completed;
    completed.runName = "fold_0";
    completed.status = TrainingRunStatus::COMPLETED;

    TrainingRunResult failed;
    failed.runName = "fold_1";
    failed.status = TrainingRunStatus::FAILED;

    TrainingRunResult cancelled;
    cancelled.runName = "fold_2";
    cancelled.status = TrainingRunStatus::CANCELLED;

    TrainingRunsResult results(std::vector<TrainingRunResult>{completed, failed, cancelled});
    const std::map<std::string, size_t> counts = results.statusCounts();

    ASSERT_EQ(counts.at("completed"), 1u);
    ASSERT_EQ(counts.at("failed"), 1u);
    ASSERT_EQ(counts.at("cancelled"), 1u);
}
