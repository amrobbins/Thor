#include "DeepLearning/Api/Training/Trainer.h"
#include "DeepLearning/Api/Training/Executors/DebugSynchronousTrainingExecutor.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingStep.h"

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

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

class CapturingExecutor : public TrainingExecutor {
   public:
    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        (void)observer;
        lastEpochs = request.epochs;
        lastNetwork = request.network.get();
        lastMaxInFlightBatches = request.runtime.maxInFlightBatches;
        lastHasTrainingProgram = request.trainingProgram != nullptr;
        lastTrainingProgramStepCount = request.trainingProgram != nullptr ? request.trainingProgram->getNumSteps() : 0;
        lastFirstStepEnabled = request.trainingProgram != nullptr && request.trainingProgram->getNumSteps() > 0
                                   ? request.trainingProgram->getStep(0).isEnabled()
                                   : false;
        lastCancellationRequested = request.cancellationToken.isCancellationRequested();
        lastSaveModelDirectory = request.saveModelDirectory;
        lastSaveModelOverwrite = request.saveModelOverwrite;
        lastSaveOptimizerState = request.saveOptimizerState;
        lastCheckBestModelEveryEpochs = request.checkBestModelEveryEpochs;
        lastMinEarlyCompletionEpochs = request.minEarlyCompletionEpochs;
        lastInitialCompletedEpochs = request.initialCompletedEpochs;
        if (request.completedTrainingEpochs != nullptr) {
            *request.completedTrainingEpochs = request.initialCompletedEpochs + request.epochs;
        }
        lastModelSelectionScoreIsCustom = request.modelSelectionScore.isCustom();
        lastModelSelectionScore = request.modelSelectionScore.evaluate(3.0, 7.0, 11);
        lastEarlyCompletionPolicyCount = request.earlyCompletionPolicies.size();
        if (!request.earlyCompletionPolicies.empty()) {
            lastEarlyCompletionDecision = request.earlyCompletionPolicies.front().shouldComplete(10.0, 9.0, 5, 4);
        }
        calls += 1;
    }

    uint32_t lastEpochs = 0;
    Network* lastNetwork = nullptr;
    uint64_t lastMaxInFlightBatches = 0;
    bool lastHasTrainingProgram = false;
    uint64_t lastTrainingProgramStepCount = 0;
    bool lastFirstStepEnabled = false;
    bool lastCancellationRequested = true;
    std::optional<std::string> lastSaveModelDirectory{};
    bool lastSaveModelOverwrite = false;
    bool lastSaveOptimizerState = true;
    uint32_t lastCheckBestModelEveryEpochs = 0;
    uint64_t lastMinEarlyCompletionEpochs = 0;
    uint64_t lastInitialCompletedEpochs = 0;
    bool lastModelSelectionScoreIsCustom = false;
    std::optional<double> lastModelSelectionScore{};
    size_t lastEarlyCompletionPolicyCount = 0;
    bool lastEarlyCompletionDecision = false;
    uint32_t calls = 0;
};


std::filesystem::path uniqueTempPath(const std::string& prefix) {
    return std::filesystem::temp_directory_path() /
           (prefix + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

}  // namespace

TEST(Trainer, BuilderRetainsSharedNetworkLifetime) {
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();
    std::weak_ptr<Network> weakNetwork;
    Network* expectedNetwork = nullptr;

    Trainer trainer;
    {
        auto network = std::make_shared<Network>("trainer-shared-network-lifetime");
        weakNetwork = network;
        expectedNetwork = network.get();
        trainer = Trainer::Builder()
                      .network(network)
                      .loader(loader)
                      .executor(executor)
                      .observer(observer)
                      .build();
    }

    EXPECT_FALSE(weakNetwork.expired());

    trainer.fit(1);

    EXPECT_EQ(executor->lastNetwork, expectedNetwork);
    EXPECT_FALSE(weakNetwork.expired());
}

TEST(Trainer, FitPassesEpochsAsRunParameter) {
    auto network = std::make_shared<Network>("trainer-test");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .loader(loader)
                          .executor(executor)
                          .observer(observer)
                          .maxInFlightBatches(64)
                          .build();

    trainer.fit(5);

    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastEpochs, 5u);
    EXPECT_EQ(executor->lastNetwork, network.get());
    EXPECT_EQ(executor->lastMaxInFlightBatches, 64u);
    EXPECT_FALSE(executor->lastCancellationRequested);
}


TEST(Trainer, FitPassesBestModelCandidateOptionsAsRunParameters) {
    auto network = std::make_shared<Network>("trainer-best-candidate-options");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .loader(loader)
                          .executor(executor)
                          .observer(observer)
                          .saveModelDirectory("/tmp/thor-best-candidate-options")
                          .saveModelOverwrite(true)
                          .saveOptimizerState(false)
                          .checkBestModelEveryEpochs(3)
                          .minEarlyCompletionEpochs(7)
                          .build();

    trainer.fit(5);

    EXPECT_EQ(executor->calls, 1u);
    ASSERT_TRUE(executor->lastSaveModelDirectory.has_value());
    EXPECT_EQ(executor->lastSaveModelDirectory.value(), "/tmp/thor-best-candidate-options");
    EXPECT_TRUE(executor->lastSaveModelOverwrite);
    EXPECT_FALSE(executor->lastSaveOptimizerState);
    EXPECT_EQ(executor->lastCheckBestModelEveryEpochs, 3u);
    EXPECT_EQ(executor->lastMinEarlyCompletionEpochs, 7u);
}

TEST(Trainer, FitPassesCumulativeCompletedEpochsAcrossFitCalls) {
    auto network = std::make_shared<Network>("trainer-cumulative-epochs");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .loader(loader)
                          .executor(executor)
                          .observer(observer)
                          .minEarlyCompletionEpochs(4)
                          .build();

    trainer.fit(2);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 0u);
    EXPECT_EQ(trainer.getCompletedTrainingEpochs(), 2u);

    trainer.fit(3);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 2u);
    EXPECT_EQ(trainer.getCompletedTrainingEpochs(), 5u);
}


TEST(Trainer, RejectsExistingSaveModelDirectoryBeforeFitWhenOverwriteIsFalse) {
    auto network = std::make_shared<Network>("trainer-existing-save-dir");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();
    const std::filesystem::path saveDir = uniqueTempPath("thor-trainer-existing-save-dir");
    std::filesystem::create_directories(saveDir);

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .loader(loader)
                          .executor(executor)
                          .observer(observer)
                          .saveModelDirectory(saveDir.string())
                          .build();

    EXPECT_THROW(trainer.fit(1), std::runtime_error);
    EXPECT_EQ(executor->calls, 0u);

    std::filesystem::remove_all(saveDir);
}

TEST(Trainer, FitPassesCustomModelSelectionScoreAsRunParameter) {
    auto network = std::make_shared<Network>("trainer-custom-model-selection-score");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    TrainingModelSelectionScore score([](std::optional<double> validationLoss, std::optional<double> trainingLoss, uint64_t epoch) {
        return validationLoss.value_or(0.0) + trainingLoss.value_or(0.0) + static_cast<double>(epoch);
    });

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .loader(loader)
                          .executor(executor)
                          .observer(observer)
                          .modelSelectionScore(score)
                          .build();

    trainer.fit(5);

    EXPECT_EQ(executor->calls, 1u);
    EXPECT_TRUE(executor->lastModelSelectionScoreIsCustom);
    ASSERT_TRUE(executor->lastModelSelectionScore.has_value());
    EXPECT_EQ(executor->lastModelSelectionScore.value(), 21.0);
}

TEST(Trainer, DefaultModelSelectionScoreUsesValidationLossWhenPresentOtherwiseTrainingLoss) {
    TrainingModelSelectionScore score;

    std::optional<double> validationScore = score.evaluate(3.0, 7.0, 11);
    ASSERT_TRUE(validationScore.has_value());
    EXPECT_EQ(validationScore.value(), 3.0);

    std::optional<double> trainingScore = score.evaluate(std::nullopt, 7.0, 11);
    ASSERT_TRUE(trainingScore.has_value());
    EXPECT_EQ(trainingScore.value(), 7.0);

    EXPECT_FALSE(score.evaluate(std::nullopt, std::nullopt, 11).has_value());
}

TEST(Trainer, RejectsZeroBestModelCandidateCheckCadence) {
    auto network = std::make_shared<Network>("trainer-best-candidate-invalid-cadence");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();

    EXPECT_THROW((static_cast<void>(Trainer::Builder()
                                        .network(network)
                                        .loader(loader)
                                        .executor(executor)
                                        .checkBestModelEveryEpochs(0)
                                        .build())),
                 std::runtime_error);
}


TEST(Trainer, FitPassesEarlyCompletionPoliciesAsRunParameters) {
    auto network = std::make_shared<Network>("trainer-early-completion-options");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    TrainingEarlyCompletionPolicy policy([](double currentScore, double bestScore, uint64_t currentEpoch, uint64_t bestEpoch) {
        return currentScore > bestScore && currentEpoch > bestEpoch;
    });

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .loader(loader)
                          .executor(executor)
                          .observer(observer)
                          .earlyCompletionPolicies({policy})
                          .build();

    trainer.fit(5);

    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastEarlyCompletionPolicyCount, 1u);
    EXPECT_TRUE(executor->lastEarlyCompletionDecision);
}

TEST(Trainer, RejectsEarlyCompletionPolicyWithoutCondition) {
    auto network = std::make_shared<Network>("trainer-early-completion-invalid");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();

    EXPECT_THROW((static_cast<void>(Trainer::Builder()
                                        .network(network)
                                        .loader(loader)
                                        .executor(executor)
                                        .earlyCompletionPolicies({TrainingEarlyCompletionPolicy{}})
                                        .build())),
                 std::runtime_error);
}

TEST(Trainer, FitPassesTrainingProgramAsRunParameter) {
    auto network = std::make_shared<Network>("trainer-test");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Tensor loss(DataType::FP32, {1});
    auto step = std::make_shared<TrainingStep>("step", std::vector<Tensor>{loss}, nullptr, std::vector<ParameterReference>{});
    auto program = std::make_shared<TrainingProgram>(std::vector<std::shared_ptr<TrainingStep>>{step});

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .loader(loader)
                          .trainingProgram(program)
                          .executor(executor)
                          .observer(observer)
                          .build();

    trainer.fit(1);

    EXPECT_EQ(executor->calls, 1u);
    EXPECT_TRUE(executor->lastHasTrainingProgram);
    EXPECT_EQ(executor->lastTrainingProgramStepCount, 1u);
}



TEST(Trainer, FitSeesTrainingProgramMutationsBetweenCalls) {
    auto network = std::make_shared<Network>("trainer-test");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Tensor loss(DataType::FP32, {1});
    auto step = std::make_shared<TrainingStep>("step", std::vector<Tensor>{loss}, nullptr, std::vector<ParameterReference>{});
    auto program = std::make_shared<TrainingProgram>(std::vector<std::shared_ptr<TrainingStep>>{step});

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .loader(loader)
                          .trainingProgram(program)
                          .executor(executor)
                          .observer(observer)
                          .build();

    trainer.fit(1);
    EXPECT_EQ(executor->calls, 1u);
    EXPECT_TRUE(executor->lastFirstStepEnabled);

    step->disable();
    trainer.fit(1);
    EXPECT_EQ(executor->calls, 2u);
    EXPECT_FALSE(executor->lastFirstStepEnabled);
}

TEST(Trainer, RejectsZeroEpochsAtFitTime) {
    auto network = std::make_shared<Network>("trainer-test");
    auto loader = std::make_shared<FakeLoader>();
    auto executor = std::make_shared<CapturingExecutor>();

    Trainer trainer = Trainer::Builder().network(network).loader(loader).executor(executor).build();

    EXPECT_THROW(trainer.fit(0), std::runtime_error);
    EXPECT_EQ(executor->calls, 0u);
}

TEST(DebugSynchronousTrainingExecutor, IsPluggableTrainingExecutorBackend) {
    static_assert(std::is_base_of_v<TrainingExecutor, DebugSynchronousTrainingExecutor>);

    std::shared_ptr<TrainingExecutor> executor = std::make_shared<DebugSynchronousTrainingExecutor>();

    EXPECT_NE(executor, nullptr);
}

TEST(Trainer, BuilderProvidesDebugSynchronousExecutorShortcut) {
    auto network = std::make_shared<Network>("trainer-test");
    auto loader = std::make_shared<FakeLoader>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .loader(loader)
                          .debugSynchronousExecutor()
                          .build();

    EXPECT_THROW(trainer.fit(1), std::runtime_error);
}
