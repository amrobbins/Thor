#include "DeepLearning/Api/Training/Trainer.h"
#include "DeepLearning/Api/Training/Executors/DebugSynchronousTrainingExecutor.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingStep.h"

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <memory>
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
        lastStatsEnabled = request.runtime.statsEnabled;
        lastMaxInFlightBatches = request.runtime.maxInFlightBatches;
        lastHasTrainingProgram = request.trainingProgram != nullptr;
        lastTrainingProgramStepCount = request.trainingProgram != nullptr ? request.trainingProgram->getNumSteps() : 0;
        lastFirstStepEnabled = request.trainingProgram != nullptr && request.trainingProgram->getNumSteps() > 0
                                   ? request.trainingProgram->getStep(0).isEnabled()
                                   : false;
        lastCancellationRequested = request.cancellationToken.isCancellationRequested();
        calls += 1;
    }

    uint32_t lastEpochs = 0;
    Network* lastNetwork = nullptr;
    bool lastStatsEnabled = true;
    uint64_t lastMaxInFlightBatches = 0;
    bool lastHasTrainingProgram = false;
    uint64_t lastTrainingProgramStepCount = 0;
    bool lastFirstStepEnabled = false;
    bool lastCancellationRequested = true;
    uint32_t calls = 0;
};

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
                      .statsEnabled(false)
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
                          .statsEnabled(false)
                          .maxInFlightBatches(64)
                          .build();

    trainer.fit(5);

    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastEpochs, 5u);
    EXPECT_EQ(executor->lastNetwork, network.get());
    EXPECT_FALSE(executor->lastStatsEnabled);
    EXPECT_EQ(executor->lastMaxInFlightBatches, 64u);
    EXPECT_FALSE(executor->lastCancellationRequested);
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
                          .statsEnabled(false)
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
                          .statsEnabled(false)
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
                          .statsEnabled(false)
                          .build();

    EXPECT_THROW(trainer.fit(1), std::runtime_error);
}
