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


std::shared_ptr<Network> makeNetworkWithOutput(const std::string& name, const std::vector<uint64_t>& dimensions) {
    auto network = std::make_shared<Network>(name);
    NetworkInput::Builder().network(*network).name("features").dimensions({0, 4}).dataType(DataType::FP32).build();
    Tensor outputTensor(DataType::FP32, dimensions);
    NetworkOutput::Builder().network(*network).name("predictions").inputTensor(outputTensor).dataType(DataType::FP32).build();
    return network;
}

std::shared_ptr<Trainer> makeTrainer(std::shared_ptr<Network> network,
                                    std::shared_ptr<TrainingExecutor> executor,
                                    std::optional<std::string> saveModelDirectory = std::nullopt) {
    return std::make_shared<Trainer>(Trainer::Builder()
                                         .network(std::move(network))
                                         .loader(std::make_shared<FakeLoader>())
                                         .executor(std::move(executor))
                                         .observer(std::make_shared<NullTrainingObserver>())
                                         .statsEnabled(false)
                                         .saveModelDirectory(std::move(saveModelDirectory))
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

    EXPECT_THROW(runs.fit(0), std::runtime_error);
    EXPECT_EQ(coordinator->startedCount(), 0u);
    EXPECT_EQ(executor->calls, 0u);
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
