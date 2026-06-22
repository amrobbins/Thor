#include "DeepLearning/Api/Training/TrainingRuns.h"

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Training/Events/TrainingEvent.h"
#include "DeepLearning/Api/Training/Executors/TrainingExecutor.h"
#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace Thor;

namespace {

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
        sawStatsEnabled = sawStatsEnabled || request.runtime.statsEnabled;
        sawLossRequested = sawLossRequested || request.runtime.scalarTensorsToReport.count("loss") != 0;
        lastEarlyCompletionPolicyCount = request.earlyCompletionPolicies.size();
        if (!request.earlyCompletionPolicies.empty()) {
            lastEarlyCompletionDecision = request.earlyCompletionPolicies.front().shouldComplete(10.0, 9.0, 5, 4);
        }

        const size_t attemptIndex = static_cast<size_t>(calls - 1);
        const std::vector<double>& losses = attemptEpochLosses.at(std::min(attemptIndex, attemptEpochLosses.size() - 1));
        for (uint32_t epoch = 1; epoch <= request.epochs; ++epoch) {
            request.cancellationToken.throwIfCancellationRequested();
            TrainingStatsSnapshot stats;
            stats.phase = TrainingEventPhase::TRAIN;
            stats.epoch = epoch;
            stats.epochs = request.epochs;
            stats.step = epoch;
            stats.stepInEpoch = 1;
            stats.stepsPerEpoch = 1;
            stats.loss = losses[std::min<size_t>(epoch - 1, losses.size() - 1)];
            observer.onTrainingEvent(TrainingEvent::statsUpdated(stats));
        }
    }

    uint32_t calls = 0;
    bool sawStatsEnabled = false;
    bool sawLossRequested = false;
    size_t lastEarlyCompletionPolicyCount = 0;
    bool lastEarlyCompletionDecision = false;

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
                                         .statsEnabled(false)
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

}  // namespace

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

TEST(TrainingRuns, RejectsDuplicateSaveModelDirectories) {
    auto network0 = std::make_shared<Network>("training-runs-save-dir-0");
    auto network1 = std::make_shared<Network>("training-runs-save-dir-1");
    auto coordinator = std::make_shared<Coordinator>(2);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);

    const std::filesystem::path outputDir = std::filesystem::path("training-runs-output") / "shared-checkpoint";
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0, outputDir.string());
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1, (outputDir / ".." / "shared-checkpoint").string());

    EXPECT_THROW(
        (TrainingRuns(std::vector<TrainingRunsSpec>{TrainingRunsSpec{"fold_0", trainer0}, TrainingRunsSpec{"fold_1", trainer1}})),
        std::runtime_error);
}

TEST(TrainingRuns, AllowsDistinctSaveModelDirectories) {
    auto network0 = std::make_shared<Network>("training-runs-save-distinct-0");
    auto network1 = std::make_shared<Network>("training-runs-save-distinct-1");
    auto coordinator = std::make_shared<Coordinator>(2);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);

    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0, "training-runs-output/fold-0");
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1, "training-runs-output/fold-1");

    EXPECT_NO_THROW(
        (TrainingRuns(std::vector<TrainingRunsSpec>{TrainingRunsSpec{"fold_0", trainer0}, TrainingRunsSpec{"fold_1", trainer1}})));
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


TEST(Trainer, RestartConditionRestartsSingleTrainerUntilProgressImproves) {
    auto network = std::make_shared<Network>("trainer-restart-progress");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 98.0, 98.0}, {100.0, 90.0, 85.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(
        network, executor, std::nullopt, {TrainingRestartCondition{/*progressCheckEpochs=*/2, /*progressPercentage=*/5.0, /*maxRestarts=*/5}});

    trainer->fit(3);

    EXPECT_EQ(executor->calls, 2u);
    EXPECT_TRUE(executor->sawStatsEnabled);
    EXPECT_TRUE(executor->sawLossRequested);
}

TEST(TrainingRuns, RestartConditionRestartsRunUntilProgressImproves) {
    auto network = std::make_shared<Network>("training-runs-restart-progress");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 98.0, 98.0}, {100.0, 90.0, 85.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRunsRestartPolicy condition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/2, /*progressPercentage=*/5.0, /*maxRestarts=*/5);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {condition});

    TrainingRunsResult result = runs.fit(3);

    ASSERT_TRUE(result.allCompleted());
    EXPECT_EQ(executor->calls, 2u);
    EXPECT_TRUE(executor->sawStatsEnabled);
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
        "fold_0", /*progressCheckEpochs=*/2, /*progressPercentage=*/5.0, /*maxRestarts=*/2);
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
        "digits", /*progressCheckEpochs=*/2, /*progressPercentage=*/5.0, /*maxRestarts=*/1);
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
        "digits", /*progressCheckEpochs=*/2, /*progressPercentage=*/5.0, /*maxRestarts=*/1);
    TrainingRunsRestartPolicy laterCondition = TrainingRunsRestartPolicy::forEnsembleGroup(
        "digits", /*progressCheckEpochs=*/3, /*progressPercentage=*/20.0, /*maxRestarts=*/1);
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
        "fold_0", /*progressCheckEpochs=*/2, /*progressPercentage=*/5.0, /*maxRestarts=*/1);
    TrainingRunsRestartPolicy laterCondition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/3, /*progressPercentage=*/20.0, /*maxRestarts=*/1);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0, std::nullopt, {earlyCondition, laterCondition});

    TrainingRunsResult result = runs.fit(3);

    ASSERT_TRUE(result.anyFailed());
    EXPECT_EQ(executor->calls, 3u);
    const TrainingRunResult& failed = result["fold_0"];
    EXPECT_EQ(failed.status, TrainingRunStatus::FAILED);
    EXPECT_EQ(failed.exception.type, "TrainingRestartConditionExceeded");
    EXPECT_NE(failed.exception.message.find("progress_check_epochs=3"), std::string::npos);
    EXPECT_NE(failed.exception.message.find("progress_percentage=20"), std::string::npos);
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
