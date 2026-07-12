#include "DeepLearning/Api/Training/Trainer.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Api/Data/FileDataset.h"
#include "DeepLearning/Api/Data/DatasetWriter.h"
#include "DeepLearning/Api/Training/Executors/DebugSynchronousTrainingExecutor.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetStorageSelection.h"
#include "DeepLearning/Api/Training/DatasetInputBindings.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingPhase.h"
#include "DeepLearning/Api/Training/TrainingStep.h"

#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using namespace Thor;

namespace {

class FakeBatchSession final : public BatchSession {
   public:
    FakeBatchSession() { batchSize = 1; }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        return exampleType == ExampleType::TRAIN ? 1 : 0;
    }
    uint64_t getNumExamples(ExampleType exampleType) override {
        return exampleType == ExampleType::TRAIN ? 1 : 0;
    }
    uint64_t getNextBatchNum(ExampleType exampleType) override {
        (void)exampleType;
        return 0;
    }

   private:
    Batch acquireBatch(ExampleType exampleType, uint64_t& batchNum) override {
        (void)exampleType;
        (void)batchNum;
        return {};
    }
    void recycleBatch(ExampleType exampleType, Batch&& batch) override {
        (void)exampleType;
        (void)batch;
    }
};

class FakeDataset final : public NamedDataset {
   public:
    FakeDataset()
        : id(DatasetId::fromStableMaterial("TrainerTest.FakeDataset")),
          schema(std::vector<DatasetField>{DatasetField{.id = 1,
                                                        .name = "features",
                                                        .dataType = ThorImplementation::DataType::FP32,
                                                        .dimensions = {1},
                                                        .kind = DatasetFieldKind::DENSE}}) {}

    const DatasetId& getId() const override { return id; }
    uint64_t getNumExamples() const override { return 1; }
    const DatasetSchema& getSchema() const override { return schema; }
    const DatasetField& getField(std::string_view name) const override { return schema.getField(name); }

   protected:
    std::shared_ptr<BatchSession> openBatchSession(const DatasetSplitManifest& splits,
                                                   const BatchPolicy& batching,
                                                   const DatasetAccessPolicy& accessPolicy,
                                                   uint64_t maxInFlightBatches,
                                                   const std::set<DatasetFieldId>& requiredFieldIds) const override {
        (void)splits;
        (void)batching;
        (void)accessPolicy;
        (void)maxInFlightBatches;
        (void)requiredFieldIds;
        return std::make_shared<FakeBatchSession>();
    }

   private:
    DatasetId id;
    DatasetSchema schema;
};

std::shared_ptr<TrainingData> makeFakeTrainingData() {
    auto dataset = std::make_shared<FakeDataset>();
    return std::make_shared<TrainingData>(dataset,
                                          DatasetSplitManifest(*dataset, {0}, {}),
                                          BatchPolicy(1, false),
                                          DatasetAccessPolicy{.deviceStorage = DeviceDatasetStorage::OFF},
                                          "fake_dataset");
}

std::shared_ptr<Network> makeFakePhaseNetwork(const std::string& networkName,
                                              const std::string& outputName) {
    auto network = std::make_shared<Network>(networkName);
    NetworkInput features = NetworkInput::Builder()
                                .network(*network)
                                .name("features")
                                .dimensions({1})
                                .dataType(ThorImplementation::DataType::FP32)
                                .build();
    NetworkOutput::Builder()
        .network(*network)
        .name(outputName)
        .inputTensor(features.getFeatureOutput().value())
        .dataType(ThorImplementation::DataType::FP32)
        .build();
    return network;
}

class CapturingExecutor : public TrainingExecutor {
   public:
    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        (void)observer;
        lastEpochs = request.epochs;
        lastNetwork = request.network.get();
        lastMaxInFlightBatches = request.runtime.maxInFlightBatches;
        lastHasTrainingProgram = request.trainingProgram != nullptr;
        lastDatasetInputBindings = request.datasetInputBindings;
        lastTrainingProgramStepCount = request.trainingProgram != nullptr ? request.trainingProgram->getNumSteps() : 0;
        lastFirstStepEnabled = request.trainingProgram != nullptr && request.trainingProgram->getNumSteps() > 0
                                   ? request.trainingProgram->getStep(0).isEnabled()
                                   : false;
        lastFirstPhaseEnabled = false;
        if (request.trainingProgram != nullptr && request.trainingProgram->getNumSteps() > 0) {
            const std::vector<std::shared_ptr<TrainingPhase>>& phases =
                request.trainingProgram->getStep(0).getPhases();
            lastFirstPhaseEnabled = !phases.empty() && phases.front() != nullptr && phases.front()->isEnabled();
        }
        lastCancellationRequested = request.cancellationToken.isCancellationRequested();
        lastSaveModelDirectory = request.saveModelDirectory;
        lastSaveModelOverwrite = request.saveModelOverwrite;
        lastCheckBestModelEveryEpochs = request.checkBestModelEveryEpochs;
        lastFirstModelSelectionEpoch = request.firstModelSelectionEpoch;
        lastMaxTrainingBatchesPerEpoch = request.maxTrainingBatchesPerEpoch;
        lastTrainingData = request.trainingData;
        lastDeviceDatasetStorageReport = request.deviceDatasetStorageReport;
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
    std::vector<TrainingInputBinding> lastDatasetInputBindings{};
    uint64_t lastTrainingProgramStepCount = 0;
    bool lastFirstStepEnabled = false;
    bool lastFirstPhaseEnabled = false;
    bool lastCancellationRequested = true;
    std::optional<std::string> lastSaveModelDirectory{};
    bool lastSaveModelOverwrite = false;
    uint32_t lastCheckBestModelEveryEpochs = 0;
    uint64_t lastFirstModelSelectionEpoch = 0;
    std::optional<uint64_t> lastMaxTrainingBatchesPerEpoch{};
    std::shared_ptr<const TrainingData> lastTrainingData = nullptr;
    DeviceDatasetStorageReport lastDeviceDatasetStorageReport{};
    uint64_t lastInitialCompletedEpochs = 0;
    bool lastModelSelectionScoreIsCustom = false;
    std::optional<double> lastModelSelectionScore{};
    size_t lastEarlyCompletionPolicyCount = 0;
    bool lastEarlyCompletionDecision = false;
    uint32_t calls = 0;
};

class SessionCapturingExecutor : public CapturingExecutor {
   public:
    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        sessions.push_back(request.batchSession);
        CapturingExecutor::fit(request, observer);
    }

    std::vector<std::shared_ptr<BatchSession>> sessions{};
};


std::filesystem::path uniqueTempPath(const std::string& prefix) {
    return std::filesystem::temp_directory_path() /
           (prefix + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

std::shared_ptr<TrainingData> makeTrainingData(
    const std::filesystem::path& path,
    DeviceDatasetStorage deviceStorage = DeviceDatasetStorage::BEST_EFFORT) {
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        std::vector<DatasetLayout::TensorShape>{
            DatasetLayout::TensorShape("features", {1}, ThorImplementation::DataType::FP32)});
    DatasetWriter writer(path, layout, 2);
    for (uint64_t i = 0; i < 4; ++i) {
        float value = static_cast<float>(i);
        DatasetWriter::TensorView view{
            .dataType = ThorImplementation::DataType::FP32,
            .dimensions = {1},
            .data = &value,
            .numBytes = sizeof(value),
        };
        writer.writeIndexedExample({{"features", view}});
    }
    writer.close();
    std::shared_ptr<FileDataset> dataset = FileDataset::open(path);
    return std::make_shared<TrainingData>(
        dataset,
        DatasetSplitManifest(*dataset, {0, 1, 2}, {3}),
        BatchPolicy(1, false),
        DatasetAccessPolicy{.deviceStorage = deviceStorage});
}

std::shared_ptr<TrainingData> makePhaseSubsetTrainingData(const std::filesystem::path& path) {
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> shapes{
        {"features", {1}},
        {"labels", {1}},
        {"future_features", {1}},
        {"unused_dataset_field", {1}},
    };
    std::vector<DatasetLayout::TensorShape> tensorShapes;
    tensorShapes.reserve(shapes.size());
    for (const auto& [name, dimensions] : shapes) {
        tensorShapes.emplace_back(name, dimensions, ThorImplementation::DataType::FP32);
    }
    DatasetLayout layout = DatasetLayout::fromTensorShapes(tensorShapes);
    DatasetWriter writer(path, layout, 2);
    for (uint64_t i = 0; i < 4; ++i) {
        float value = static_cast<float>(i);
        DatasetWriter::TensorView view{
            .dataType = ThorImplementation::DataType::FP32,
            .dimensions = {1},
            .data = &value,
            .numBytes = sizeof(value),
        };
        writer.writeIndexedExample({
            {"features", view},
            {"labels", view},
            {"future_features", view},
            {"unused_dataset_field", view},
        });
    }
    writer.close();
    std::shared_ptr<FileDataset> dataset = FileDataset::open(path);
    return std::make_shared<TrainingData>(
        dataset,
        DatasetSplitManifest(*dataset, {0, 1, 2}, {3}),
        BatchPolicy(1, false),
        DatasetAccessPolicy{.deviceStorage = DeviceDatasetStorage::OFF});
}

std::map<std::string, std::string> bindingMap(const std::vector<TrainingInputBinding>& bindings) {
    std::map<std::string, std::string> result;
    for (const TrainingInputBinding& binding : bindings) {
        result.emplace(binding.getNetworkInputName(), binding.getBatchInputName());
    }
    return result;
}

}  // namespace

TEST(Trainer, TrainingDataOpensFreshSessionForEveryFit) {
    const std::filesystem::path path = uniqueTempPath("thor-trainer-training-data");
    std::shared_ptr<TrainingData> data = makeTrainingData(path);
    auto network = std::make_shared<Network>("trainer-training-data");
    auto executor = std::make_shared<SessionCapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .observer(observer)
                          .maxInFlightBatches(3)
                          .build();
    EXPECT_EQ(trainer.getTrainingData(), data);

    trainer.fit(1);
    trainer.fit(1);

    ASSERT_EQ(executor->sessions.size(), 2u);
    EXPECT_NE(executor->sessions[0].get(), executor->sessions[1].get());
    EXPECT_NE(executor->sessions[0], nullptr);
    EXPECT_NE(executor->sessions[1], nullptr);
    executor->sessions.clear();
    std::filesystem::remove_all(path);
}

TEST(Trainer, BuilderRequiresTrainingData) {
    auto network = std::make_shared<Network>("trainer-requires-data");
    EXPECT_THROW(static_cast<void>(Trainer::Builder().network(network).build()), std::runtime_error);
}

TEST(Trainer, FitRequiresNonEmptyTrainPartition) {
    auto dataset = std::make_shared<FakeDataset>();
    auto data = std::make_shared<TrainingData>(
        dataset,
        DatasetSplitManifest(*dataset, {}, {0}),
        BatchPolicy(1, false),
        DatasetAccessPolicy{.deviceStorage = DeviceDatasetStorage::OFF},
        "evaluation_only_data");
    auto executor = std::make_shared<CapturingExecutor>();
    Trainer trainer = Trainer::Builder()
                          .network(std::make_shared<Network>("trainer-empty-train"))
                          .data(data)
                          .executor(executor)
                          .observer(std::make_shared<NullTrainingObserver>())
                          .build();

    EXPECT_THROW((void)trainer.fit(1), std::runtime_error);
    EXPECT_EQ(executor->calls, 0u);
}

TEST(Trainer, CompilesStrictDatasetInputBindingsBeforeOpeningSession) {
    const std::filesystem::path path = uniqueTempPath("thor-trainer-dataset-bindings");
    std::shared_ptr<TrainingData> data = makeTrainingData(path);
    auto network = std::make_shared<Network>("trainer-dataset-bindings");
    NetworkInput features = NetworkInput::Builder()
                                .network(*network)
                                .name("model_features")
                                .dimensions({1})
                                .dataType(ThorImplementation::DataType::FP32)
                                .build();
    DatasetInputBindings bindings;
    bindings.bind(features, data->getDataset()->getField("features"));

    auto executor = std::make_shared<SessionCapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();
    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .inputBindings(bindings)
                          .executor(executor)
                          .observer(observer)
                          .build();

    ASSERT_EQ(trainer.getDatasetInputBindings().size(), 1u);
    EXPECT_EQ(trainer.getRequiredDatasetFieldIds(),
              (std::set<DatasetFieldId>{data->getDataset()->getField("features").id}));

    trainer.fit(1);
    ASSERT_EQ(executor->lastDatasetInputBindings.size(), 1u);
    EXPECT_EQ(executor->lastDatasetInputBindings.front().getNetworkInputName(), "model_features");
    EXPECT_EQ(executor->lastDatasetInputBindings.front().getBatchInputName(), "features");
    ASSERT_EQ(executor->sessions.size(), 1u);
    std::shared_ptr<BatchSession> session = executor->sessions.front();
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->getRequiredDatasetFieldIds(),
              trainer.getRequiredDatasetFieldIds());
    std::filesystem::remove_all(path);
}

TEST(Trainer, ExactNameAutobindingIsDefaultAndValidationIsEarly) {
    const std::filesystem::path path = uniqueTempPath("thor-trainer-exact-dataset-bindings");
    std::shared_ptr<TrainingData> data = makeTrainingData(path);

    auto exactNetwork = std::make_shared<Network>("trainer-exact-bindings");
    NetworkInput::Builder()
        .network(*exactNetwork)
        .name("features")
        .dimensions({1})
        .dataType(ThorImplementation::DataType::FP32)
        .build();
    EXPECT_NO_THROW(static_cast<void>(Trainer::Builder().network(exactNetwork).data(data).build()));

    auto renamedNetwork = std::make_shared<Network>("trainer-missing-exact-binding");
    NetworkInput::Builder()
        .network(*renamedNetwork)
        .name("renamed_features")
        .dimensions({1})
        .dataType(ThorImplementation::DataType::FP32)
        .build();
    EXPECT_THROW(static_cast<void>(Trainer::Builder().network(renamedNetwork).data(data).build()), std::runtime_error);

    auto dtypeNetwork = std::make_shared<Network>("trainer-dtype-binding");
    NetworkInput dtypeInput = NetworkInput::Builder()
                                  .network(*dtypeNetwork)
                                  .name("features")
                                  .dimensions({1})
                                  .dataType(ThorImplementation::DataType::FP16)
                                  .build();
    DatasetInputBindings dtypeBindings;
    dtypeBindings.bind(dtypeInput, data->getDataset()->getField("features"));
    EXPECT_THROW(static_cast<void>(Trainer::Builder()
                                      .network(dtypeNetwork)
                                      .data(data)
                                      .inputBindings(dtypeBindings)
                                      .build()),
                 std::runtime_error);

    std::filesystem::remove_all(path);
}

TEST(Trainer, BuilderRetainsSharedNetworkLifetime) {
    auto data = makeFakeTrainingData();
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
                      .data(data)
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
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
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
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .observer(observer)
                          .saveModelDirectory("/tmp/thor-best-candidate-options")
                          .saveModelOverwrite(true)
                          .build();

    TrainerFitOptions options;
    options.epochs = 5;
    options.checkBestModelEveryEpochs = 3;
    options.firstModelSelectionEpoch = 7;
    trainer.fit(options);

    EXPECT_EQ(executor->calls, 1u);
    ASSERT_TRUE(executor->lastSaveModelDirectory.has_value());
    EXPECT_EQ(executor->lastSaveModelDirectory.value(), "/tmp/thor-best-candidate-options");
    EXPECT_TRUE(executor->lastSaveModelOverwrite);
    EXPECT_EQ(executor->lastCheckBestModelEveryEpochs, 3u);
    EXPECT_EQ(executor->lastFirstModelSelectionEpoch, 7u);
    EXPECT_FALSE(executor->lastMaxTrainingBatchesPerEpoch.has_value());
}

TEST(Trainer, FitPassesMaxTrainingBatchesPerEpochAsRunParameter) {
    auto network = std::make_shared<Network>("trainer-max-training-batches-per-epoch");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .observer(observer)
                          .build();

    TrainerFitOptions options;
    options.epochs = 3;
    options.maxTrainingBatchesPerEpoch = 500;
    trainer.fit(options);

    EXPECT_EQ(executor->calls, 1u);
    ASSERT_TRUE(executor->lastMaxTrainingBatchesPerEpoch.has_value());
    EXPECT_EQ(executor->lastMaxTrainingBatchesPerEpoch.value(), 500u);
}

TEST(Trainer, TrainingDataOwnsDefaultDeviceDatasetStoragePolicy) {
    const std::filesystem::path path = uniqueTempPath("trainer-device-dataset-storage-default");
    auto network = std::make_shared<Network>("trainer-device-dataset-storage-default");
    std::shared_ptr<TrainingData> data = makeTrainingData(path);
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    EXPECT_EQ(data->getAccessPolicy().deviceStorage, DeviceDatasetStorage::BEST_EFFORT);
    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .observer(observer)
                          .build();
    trainer.fit(1);

    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastTrainingData, data);
    EXPECT_EQ(executor->lastDeviceDatasetStorageReport.requested,
              DeviceDatasetStorage::BEST_EFFORT);
    std::filesystem::remove_all(path);
}

TEST(Trainer, TrainingDataPassesStrictDeviceAccessPolicyToRun) {
    const std::filesystem::path path = uniqueTempPath("trainer-device-dataset-storage-strict");
    auto network = std::make_shared<Network>("trainer-device-dataset-storage-strict");
    std::shared_ptr<TrainingData> data =
        makeTrainingData(path, DeviceDatasetStorage::STRICT);
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .observer(observer)
                          .build();
    trainer.fit(1);

    EXPECT_EQ(executor->calls, 1u);
    ASSERT_NE(executor->lastTrainingData, nullptr);
    EXPECT_EQ(executor->lastTrainingData->getAccessPolicy().deviceStorage,
              DeviceDatasetStorage::STRICT);
    EXPECT_EQ(executor->lastDeviceDatasetStorageReport.requested,
              DeviceDatasetStorage::STRICT);
    std::filesystem::remove_all(path);
}

TEST(Trainer, RejectsZeroMaxTrainingBatchesPerEpoch) {
    auto network = std::make_shared<Network>("trainer-zero-max-training-batches-per-epoch");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .observer(observer)
                          .build();

    TrainerFitOptions options;
    options.maxTrainingBatchesPerEpoch = 0;

    EXPECT_THROW(trainer.fit(options), std::runtime_error);
    EXPECT_EQ(executor->calls, 0u);
}

TEST(Trainer, FitPassesCumulativeCompletedEpochsAcrossFitCalls) {
    auto network = std::make_shared<Network>("trainer-cumulative-epochs");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .observer(observer)
                          .build();

    TrainerFitOptions options;
    options.epochs = 2;
    options.firstModelSelectionEpoch = 4;
    trainer.fit(options);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 0u);
    EXPECT_EQ(trainer.getCompletedTrainingEpochs(), 2u);

    options.epochs = 3;
    trainer.fit(options);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 2u);
    EXPECT_EQ(trainer.getCompletedTrainingEpochs(), 5u);
}


TEST(Trainer, RejectsExistingSaveModelDirectoryBeforeFitWhenOverwriteIsFalse) {
    auto network = std::make_shared<Network>("trainer-existing-save-dir");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();
    const std::filesystem::path saveDir = uniqueTempPath("thor-trainer-existing-save-dir");
    std::filesystem::create_directories(saveDir);

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
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
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    TrainingModelSelectionScore score([](std::optional<double> validationLoss, std::optional<double> trainingLoss, uint64_t epoch) {
        return validationLoss.value_or(0.0) + trainingLoss.value_or(0.0) + static_cast<double>(epoch);
    });

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
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


TEST(Trainer, ModelSelectionScoreCanUseContextNamedValidationLossesAndMetrics) {
    TrainingModelSelectionContext context;
    context.epoch = 11;
    context.train.loss = 100.0;
    context.validate.loss = 200.0;
    context.validate.losses["daily_mse_loss"] = 3.0;
    context.validate.losses["aggregate_mse_loss"] = 40.0;
    context.validate.metrics["daily_pred"] = 6.0;
    context.validate.metrics["daily_true"] = 5.5;

    TrainingModelSelectionScore score(TrainingModelSelectionScore::ContextScoreFunction(
        [](const TrainingModelSelectionContext& context) -> std::optional<double> {
            return context.validate.losses.at("daily_mse_loss") +
                   0.05 * context.validate.losses.at("aggregate_mse_loss") +
                   0.1 * std::abs(context.validate.metrics.at("daily_pred") - context.validate.metrics.at("daily_true"));
        }));

    std::optional<double> selectedScore = score.evaluate(context);
    ASSERT_TRUE(selectedScore.has_value());
    EXPECT_DOUBLE_EQ(selectedScore.value(), 5.05);
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

TEST(Trainer, FitDisablesBestModelCandidateChecksByDefault) {
    auto network = std::make_shared<Network>("trainer-best-candidate-default-disabled");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .observer(observer)
                          .build();

    trainer.fit(5);

    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastCheckBestModelEveryEpochs, 0u);
}


TEST(Trainer, FitPassesEarlyCompletionPoliciesAsRunParameters) {
    auto network = std::make_shared<Network>("trainer-early-completion-options");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    TrainingEarlyCompletionPolicy policy([](double currentScore, double bestScore, uint64_t currentEpoch, uint64_t bestEpoch) {
        return currentScore > bestScore && currentEpoch > bestEpoch;
    });

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .observer(observer)
                          .build();

    TrainerFitOptions options;
    options.epochs = 5;
    options.checkBestModelEveryEpochs = 1;
    options.earlyCompletionPolicies = {policy};
    trainer.fit(options);

    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastEarlyCompletionPolicyCount, 1u);
    EXPECT_TRUE(executor->lastEarlyCompletionDecision);
}

TEST(Trainer, RejectsEarlyCompletionPolicyWithoutCondition) {
    auto network = std::make_shared<Network>("trainer-early-completion-invalid");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .executor(executor)
                          .build();

    TrainerFitOptions options;
    options.epochs = 1;
    options.checkBestModelEveryEpochs = 1;
    options.earlyCompletionPolicies = {TrainingEarlyCompletionPolicy{}};
    EXPECT_THROW(static_cast<void>(trainer.fit(options)), std::runtime_error);
}

TEST(Trainer, FitPassesTrainingProgramAsRunParameter) {
    auto network = makeFakePhaseNetwork("trainer-test", "phase_output");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    auto phase = std::make_shared<TrainingPhase>("phase", network);
    auto step = std::make_shared<TrainingStep>("step", std::vector<std::shared_ptr<TrainingPhase>>{phase}, nullptr, std::vector<ParameterReference>{});
    auto program = std::make_shared<TrainingProgram>(std::vector<std::shared_ptr<TrainingStep>>{step});

    Trainer trainer = Trainer::Builder()
                          .data(data)
                          .trainingProgram(program)
                          .executor(executor)
                          .observer(observer)
                          .build();

    EXPECT_EQ(trainer.getNetwork(), nullptr);
    trainer.fit(1);

    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastNetwork, nullptr);
    EXPECT_TRUE(executor->lastHasTrainingProgram);
    EXPECT_EQ(executor->lastTrainingProgramStepCount, 1u);
}



TEST(Trainer, PhaseBackedTrainingDataUsesOnlyCurrentlyActiveDatasetFields) {
    const std::filesystem::path path = uniqueTempPath("thor-trainer-phase-subset-data");
    std::shared_ptr<TrainingData> data = makePhaseSubsetTrainingData(path);
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    auto dailyNetwork = std::make_shared<Network>("trainer-phase-subset-daily");
    NetworkInput dailyFeatures = NetworkInput::Builder()
                                     .network(*dailyNetwork)
                                     .name("features")
                                     .dimensions({1})
                                     .dataType(ThorImplementation::DataType::FP32)
                                     .build();
    NetworkInput dailyLabels = NetworkInput::Builder()
                                   .network(*dailyNetwork)
                                   .name("labels")
                                   .dimensions({1})
                                   .dataType(ThorImplementation::DataType::FP32)
                                   .build();
    NetworkOutput::Builder()
        .network(*dailyNetwork)
        .name("daily_features")
        .inputTensor(dailyFeatures.getFeatureOutput().value())
        .dataType(ThorImplementation::DataType::FP32)
        .build();
    NetworkOutput::Builder()
        .network(*dailyNetwork)
        .name("daily_labels")
        .inputTensor(dailyLabels.getFeatureOutput().value())
        .dataType(ThorImplementation::DataType::FP32)
        .build();

    auto futureNetwork = std::make_shared<Network>("trainer-phase-subset-future");
    NetworkInput futureFeatures = NetworkInput::Builder()
                                      .network(*futureNetwork)
                                      .name("future_features")
                                      .dimensions({1})
                                      .dataType(ThorImplementation::DataType::FP32)
                                      .build();
    NetworkInput futureLabels = NetworkInput::Builder()
                                    .network(*futureNetwork)
                                    .name("labels")
                                    .dimensions({1})
                                    .dataType(ThorImplementation::DataType::FP32)
                                    .build();
    NetworkOutput::Builder()
        .network(*futureNetwork)
        .name("future_output")
        .inputTensor(futureFeatures.getFeatureOutput().value())
        .dataType(ThorImplementation::DataType::FP32)
        .build();
    NetworkOutput::Builder()
        .network(*futureNetwork)
        .name("future_labels")
        .inputTensor(futureLabels.getFeatureOutput().value())
        .dataType(ThorImplementation::DataType::FP32)
        .build();

    auto dailyPhase = std::make_shared<TrainingPhase>("daily", dailyNetwork, true);
    auto futurePhase = std::make_shared<TrainingPhase>("future", futureNetwork, false);
    auto step = std::make_shared<TrainingStep>(
        "daily_then_future",
        std::vector<std::shared_ptr<TrainingPhase>>{dailyPhase, futurePhase},
        nullptr,
        std::vector<ParameterReference>{});
    auto program = std::make_shared<TrainingProgram>(
        std::vector<std::shared_ptr<TrainingStep>>{step});

    Trainer trainer = Trainer::Builder()
                          .data(data)
                          .trainingProgram(program)
                          .executor(executor)
                          .observer(observer)
                          .maxInFlightBatches(2)
                          .build();

    trainer.fit(1);
    EXPECT_EQ(bindingMap(executor->lastDatasetInputBindings),
              (std::map<std::string, std::string>{{"features", "features"}, {"labels", "labels"}}));

    futurePhase->enable();
    trainer.fit(1);
    EXPECT_EQ(bindingMap(executor->lastDatasetInputBindings),
              (std::map<std::string, std::string>{{"features", "features"},
                                                  {"future_features", "future_features"},
                                                  {"labels", "labels"}}));

    std::error_code errorCode;
    std::filesystem::remove_all(path, errorCode);
}

TEST(Trainer, RejectsStandaloneNetworkAlongsidePhaseBackedProgram) {
    auto network = std::make_shared<Network>("trainer-phase-network");
    auto phase = std::make_shared<TrainingPhase>("phase", network);
    auto step = std::make_shared<TrainingStep>(
        "step", std::vector<std::shared_ptr<TrainingPhase>>{phase}, nullptr, std::vector<ParameterReference>{});
    auto program = std::make_shared<TrainingProgram>(std::vector<std::shared_ptr<TrainingStep>>{step});

    EXPECT_THROW(static_cast<void>(Trainer::Builder()
                                      .network(network)
                                      .data(makeFakeTrainingData())
                                      .trainingProgram(program)
                                      .build()),
                 std::runtime_error);
}

TEST(Trainer, FitSeesTrainingProgramMutationsBetweenCalls) {
    auto firstNetwork = makeFakePhaseNetwork("trainer-test-first", "first_output");
    auto secondNetwork = makeFakePhaseNetwork("trainer-test-second", "second_output");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();
    auto observer = std::make_shared<NullTrainingObserver>();

    auto firstPhase = std::make_shared<TrainingPhase>("first", firstNetwork);
    auto secondPhase = std::make_shared<TrainingPhase>("second", secondNetwork);
    auto step = std::make_shared<TrainingStep>(
        "step",
        std::vector<std::shared_ptr<TrainingPhase>>{firstPhase, secondPhase},
        nullptr,
        std::vector<ParameterReference>{});
    auto program = std::make_shared<TrainingProgram>(std::vector<std::shared_ptr<TrainingStep>>{step});

    Trainer trainer = Trainer::Builder()
                          .data(data)
                          .trainingProgram(program)
                          .executor(executor)
                          .observer(observer)
                          .build();

    trainer.fit(1);
    EXPECT_EQ(executor->calls, 1u);
    EXPECT_TRUE(executor->lastFirstPhaseEnabled);

    firstPhase->disable();
    trainer.fit(1);
    EXPECT_EQ(executor->calls, 2u);
    EXPECT_FALSE(executor->lastFirstPhaseEnabled);
}

TEST(Trainer, RejectsZeroEpochsAtFitTime) {
    auto network = std::make_shared<Network>("trainer-test");
    auto data = makeFakeTrainingData();
    auto executor = std::make_shared<CapturingExecutor>();

    Trainer trainer = Trainer::Builder().network(network).data(data).executor(executor).build();

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
    auto data = makeFakeTrainingData();

    Trainer trainer = Trainer::Builder()
                          .network(network)
                          .data(data)
                          .debugSynchronousExecutor()
                          .build();

    EXPECT_THROW(trainer.fit(1), std::runtime_error);
}
