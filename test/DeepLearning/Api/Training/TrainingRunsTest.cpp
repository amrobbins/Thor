#include "DeepLearning/Api/Training/TrainingRuns.h"

#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedRowLengths.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
#include "DeepLearning/Api/Layers/Metrics/CustomMetric.h"
#include "DeepLearning/Api/Layers/Metrics/Mean.h"
#include "DeepLearning/Api/Layers/Metrics/Sum.h"
#include "DeepLearning/Api/Layers/Metrics/Min.h"
#include "DeepLearning/Api/Layers/Metrics/Max.h"
#include "DeepLearning/Api/Layers/Metrics/WeightedMean.h"
#include "DeepLearning/Api/Training/Events/TrainingEvent.h"
#include "DeepLearning/Api/Training/TrainingPhase.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingStep.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Training/Executors/TrainingExecutor.h"
#include "DeepLearning/Api/Training/Observers/TrainingObserver.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <iterator>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>


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

class CountingTrainingValidationNetwork final : public Network {
   public:
    explicit CountingTrainingValidationNetwork(const std::string& networkName) : Network(networkName) {}

    [[nodiscard]] uint32_t trainingValidationCount() const { return trainingValidationCount_; }

   protected:
    StatusCode evaluateGraph(bool inferenceOnly) override {
        if (!inferenceOnly) {
            ++trainingValidationCount_;
        }
        return Network::evaluateGraph(inferenceOnly);
    }

   private:
    uint32_t trainingValidationCount_ = 0;
};


std::shared_ptr<Network> makeRaggedRowLengthMemberNetwork(const std::string& networkName) {
    auto network = std::make_shared<Network>(networkName);
    RaggedTensor history = RaggedNetworkInput::Builder()
                               .network(*network)
                               .name("history")
                               .valuesDataType(DataType::FP32)
                               .trailingDimensions({2})
                               .maxTotalValues(8)
                               .batchSize(4)
                               .maxValuesPerRow(3)
                               .build();

    // The prediction below intentionally depends only on offsets. Keep the
    // source member itself graph-valid while exercising the ensemble builder's
    // responsibility for preserving the complete logical ragged boundary.
    Stub::Builder().network(*network).inputTensor(history.getValues()).build();

    RaggedRowLengths rowLengths =
        RaggedRowLengths::Builder().network(*network).featureInput(history).build();
    ThorImplementation::Expression lengths =
        ThorImplementation::Expression::input(
            "lengths", ThorImplementation::DataType::INT32, ThorImplementation::DataType::INT32)
            .cast(ThorImplementation::DataType::FP32);
    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(
            ThorImplementation::Expression::outputs({{"predictions", lengths}}));
    CustomLayer castLengths = CustomLayer::Builder()
                                  .network(*network)
                                  .expression(ThorImplementation::DynamicExpression::fromExpressionDefinition(definition))
                                  .inputNames({"lengths"})
                                  .outputNames({"predictions"})
                                  .inputInterface({{"lengths", rowLengths.getFeatureOutput().value()}})
                                  .build();

    NetworkOutput::Builder()
        .network(*network)
        .name("predictions")
        .inputTensor(castLengths.getOutput("predictions"))
        .dataType(DataType::FP32)
        .build();
    return network;
}

class FakeBatchSession final : public BatchSession {
   public:
    explicit FakeBatchSession(uint64_t requestedBatchSize) { batchSize = requestedBatchSize; }

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
        : id(DatasetId::fromStableMaterial("TrainingRunsTest.FakeDataset")),
          schema(std::vector<DatasetField>{
              DatasetField{.id = 1, .name = "features", .dataType = DataType::FP32, .dimensions = {4}},
              DatasetField{.id = 2, .name = "labels", .dataType = DataType::FP32, .dimensions = {1}},
              DatasetField{.id = 3, .name = "observed_daily", .dataType = DataType::FP32, .dimensions = {1}},
              DatasetField{.id = 4, .name = "observed_aggregate", .dataType = DataType::FP32, .dimensions = {1}},
              DatasetField{.id = 5, .name = "example_weights", .dataType = DataType::FP32, .dimensions = {1}},
              DatasetField{.id = 6, .name = "actual", .dataType = DataType::FP32, .dimensions = {4}},
              DatasetField{.id = 7, .name = "peak_threshold", .dataType = DataType::FP32, .dimensions = {4}},
              DatasetField{.id = 8, .name = "alternate_threshold", .dataType = DataType::FP32, .dimensions = {4}},
          }) {}

    const DatasetId& getId() const override { return id; }
    uint64_t getNumExamples() const override { return 1; }
    const DatasetSchema& getSchema() const override { return schema; }
    const DatasetField& getField(std::string_view name) const override { return schema.getField(name); }

   protected:
    std::shared_ptr<BatchSession> openBatchSession(const DatasetSplitManifest& splits,
                                                   const BatchPolicy& batching,
                                                   const DatasetAccessPolicy& accessPolicy,
                                                   uint64_t maxInFlightBatches,
                                                   const DatasetFieldMaterializationRequirements& fieldRequirements) const override {
        (void)splits;
        (void)accessPolicy;
        (void)maxInFlightBatches;
        (void)fieldRequirements;
        return std::make_shared<FakeBatchSession>(batching.getBatchSize());
    }

   private:
    DatasetId id;
    DatasetSchema schema;
};

std::shared_ptr<TrainingData> makeFakeTrainingData() {
    auto dataset = std::make_shared<FakeDataset>();
    return std::make_shared<TrainingData>(dataset,
                                          DatasetSplitManifest(*dataset, {0}, {}),
                                          BatchPolicy(4, false),
                                          DatasetAccessPolicy{.deviceStorage = DeviceDatasetStorage::OFF},
                                          "fake_dataset");
}

std::shared_ptr<TrainingData> makeFakeTestData(bool includeTestPartition) {
    auto dataset = std::make_shared<FakeDataset>();
    return std::make_shared<TrainingData>(
        dataset,
        DatasetSplitManifest(
            *dataset,
            {},
            {},
            includeTestPartition ? std::vector<uint64_t>{0} : std::vector<uint64_t>{}),
        BatchPolicy(4, false),
        DatasetAccessPolicy{.deviceStorage = DeviceDatasetStorage::OFF},
        "fake_test_dataset");
}


class ExactMetricBatchSession final : public BatchSession {
   public:
    ExactMetricBatchSession(uint64_t requestedBatchSize,
                            std::vector<float> values,
                            std::vector<float> weights,
                            std::vector<uint64_t> trainIndices,
                            std::vector<uint64_t> validateIndices,
                            std::vector<uint64_t> testIndices)
        : values(std::move(values)),
          weights(std::move(weights)),
          trainIndices(std::move(trainIndices)),
          validateIndices(std::move(validateIndices)),
          testIndices(std::move(testIndices)) {
        THOR_THROW_IF_FALSE(requestedBatchSize > 0);
        THOR_THROW_IF_FALSE(this->values.size() == this->weights.size());
        batchSize = requestedBatchSize;
    }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        const uint64_t examples = getNumExamples(exampleType);
        return examples == 0 ? 0 : (examples + batchSize - 1) / batchSize;
    }

    uint64_t getNumExamples(ExampleType exampleType) override {
        return static_cast<uint64_t>(indicesFor(exampleType).size());
    }

    uint64_t getNextBatchNum(ExampleType exampleType) override {
        (void)exampleType;
        return 0;
    }

   private:
    const std::vector<uint64_t>& indicesFor(ExampleType exampleType) const {
        switch (exampleType) {
            case ExampleType::TRAIN:
                return trainIndices;
            case ExampleType::VALIDATE:
                return validateIndices;
            case ExampleType::TEST:
                return testIndices;
            default:
                throw std::runtime_error("ExactMetricBatchSession requires a concrete example type.");
        }
    }

    Batch acquireBatch(ExampleType exampleType, uint64_t& batchNum) override {
        const std::vector<uint64_t>& indices = indicesFor(exampleType);
        const uint64_t start = batchNum * batchSize;
        if (start >= indices.size()) {
            throw std::out_of_range("ExactMetricBatchSession batch number is outside the requested partition.");
        }
        const uint64_t validRows = std::min<uint64_t>(batchSize, indices.size() - start);

        ThorImplementation::Tensor valueTensor =
            makeCpuTensor(ThorImplementation::DataType::FP32, {batchSize, 1});
        ThorImplementation::Tensor weightTensor =
            makeCpuTensor(ThorImplementation::DataType::FP32, {batchSize, 1});
        float* valueMemory = valueTensor.getMemPtr<float>();
        float* weightMemory = weightTensor.getMemPtr<float>();
        for (uint64_t row = 0; row < batchSize; ++row) {
            if (row < validRows) {
                const uint64_t sourceIndex = indices.at(start + row);
                valueMemory[row] = values.at(sourceIndex);
                weightMemory[row] = weights.at(sourceIndex);
            } else {
                // Deliberately hostile padding catches any path that forgets to
                // preserve the tail batch's valid-example count.
                valueMemory[row] = ((row - validRows) % 2 == 0) ? -1000.0f : 2000.0f;
                weightMemory[row] = 100.0f;
            }
        }

        Batch batch;
        batch.insert("values", std::move(valueTensor));
        batch.insert("weights", std::move(weightTensor));
        if (validRows != batchSize) {
            batch.setValidExampleCount(static_cast<uint32_t>(validRows));
        }
        return batch;
    }

    void recycleBatch(ExampleType exampleType, Batch&& batch) override {
        (void)exampleType;
        (void)batch;
    }

    std::vector<float> values;
    std::vector<float> weights;
    std::vector<uint64_t> trainIndices;
    std::vector<uint64_t> validateIndices;
    std::vector<uint64_t> testIndices;
};

class ExactMetricDataset final : public NamedDataset {
   public:
    ExactMetricDataset(std::string stableName,
                       std::vector<float> values,
                       std::vector<float> weights)
        : id(DatasetId::fromStableMaterial(stableName)),
          schema(std::vector<DatasetField>{
              DatasetField{.id = 1, .name = "values", .dataType = DataType::FP32, .dimensions = {1}},
              DatasetField{.id = 2, .name = "weights", .dataType = DataType::FP32, .dimensions = {1}},
          }),
          values(std::move(values)),
          weights(std::move(weights)) {
        THOR_THROW_IF_FALSE(!this->values.empty());
        THOR_THROW_IF_FALSE(this->values.size() == this->weights.size());
    }

    const DatasetId& getId() const override { return id; }
    uint64_t getNumExamples() const override { return static_cast<uint64_t>(values.size()); }
    const DatasetSchema& getSchema() const override { return schema; }
    const DatasetField& getField(std::string_view name) const override { return schema.getField(name); }

   protected:
    std::shared_ptr<BatchSession> openBatchSession(const DatasetSplitManifest& splits,
                                                   const BatchPolicy& batching,
                                                   const DatasetAccessPolicy& accessPolicy,
                                                   uint64_t maxInFlightBatches,
                                                   const DatasetFieldMaterializationRequirements& fieldRequirements) const override {
        (void)accessPolicy;
        (void)maxInFlightBatches;
        (void)fieldRequirements;
        return std::make_shared<ExactMetricBatchSession>(
            batching.getBatchSize(),
            values,
            weights,
            splits.getTrain().materialize(),
            splits.getValidate().materialize(),
            splits.getTest().materialize());
    }

   private:
    DatasetId id;
    DatasetSchema schema;
    std::vector<float> values;
    std::vector<float> weights;
};

std::shared_ptr<TrainingData> makeExactMetricTrainingData(
    const std::string& name,
    std::vector<float> validationValues,
    std::vector<float> validationWeights) {
    THOR_THROW_IF_FALSE(validationValues.size() == validationWeights.size());
    THOR_THROW_IF_FALSE(!validationValues.empty());

    std::vector<float> values{0.0f};
    values.insert(values.end(), validationValues.begin(), validationValues.end());
    std::vector<float> weights{1.0f};
    weights.insert(weights.end(), validationWeights.begin(), validationWeights.end());
    auto dataset = std::make_shared<ExactMetricDataset>(name, std::move(values), std::move(weights));

    std::vector<uint64_t> validationIndices;
    validationIndices.reserve(validationValues.size());
    for (uint64_t index = 0; index < validationValues.size(); ++index) {
        validationIndices.push_back(index + 1);
    }
    return std::make_shared<TrainingData>(
        dataset,
        DatasetSplitManifest(*dataset, {0}, std::move(validationIndices)),
        BatchPolicy(4, false),
        DatasetAccessPolicy{.deviceStorage = DeviceDatasetStorage::OFF},
        name);
}

std::shared_ptr<TrainingData> makeExactMetricTestData(
    const std::string& name,
    std::vector<float> testValues,
    std::vector<float> testWeights) {
    THOR_THROW_IF_FALSE(testValues.size() == testWeights.size());
    THOR_THROW_IF_FALSE(!testValues.empty());
    auto dataset = std::make_shared<ExactMetricDataset>(name, std::move(testValues), std::move(testWeights));

    std::vector<uint64_t> testIndices;
    testIndices.reserve(dataset->getNumExamples());
    for (uint64_t index = 0; index < dataset->getNumExamples(); ++index) {
        testIndices.push_back(index);
    }
    return std::make_shared<TrainingData>(
        dataset,
        DatasetSplitManifest(*dataset, {}, {}, std::move(testIndices)),
        BatchPolicy(4, false),
        DatasetAccessPolicy{.deviceStorage = DeviceDatasetStorage::OFF},
        name);
}

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

void setSyntheticTrainingBatchCardinality(const TrainingRunRequest& request, TrainingStatsSnapshot& stats) {
    if (request.batchSession == nullptr) {
        throw std::logic_error("TrainingRuns test executor requires a batch session.");
    }
    const uint64_t batchCapacity = request.batchSession->getBatchSize();
    const uint64_t trainingExamples = request.batchSession->getNumExamples(ExampleType::TRAIN);
    if (batchCapacity == 0 || trainingExamples == 0) {
        throw std::logic_error("TrainingRuns test executor requires a non-empty training batch.");
    }
    stats.batchSize = batchCapacity;
    stats.validExamplesInBatch = std::min(batchCapacity, trainingExamples);
}

TrainingStatsSnapshot makeStats(const TrainingRunRequest& request, uint64_t step) {
    TrainingStatsSnapshot stats;
    stats.phase = TrainingEventPhase::TRAIN;
    stats.epoch = 1;
    stats.epochs = 1;
    stats.step = step;
    stats.stepInEpoch = step;
    stats.stepsPerEpoch = 10;
    setSyntheticTrainingBatchCardinality(request, stats);
    stats.samplesProcessedInEpoch = stats.validExamplesInBatch;
    stats.samplesProcessed = stats.validExamplesInBatch;
    stats.loss = 1.0 / static_cast<double>(step + 1);
    stats.elapsedSeconds = static_cast<double>(step);
    return stats;
}

class FailedPlacementReleaseState {
   public:
    void remember(const std::shared_ptr<PlacedNetwork>& placement) {
        std::lock_guard<std::mutex> lock(mutex);
        failedPlacement = placement;
    }

    [[nodiscard]] bool failedPlacementExpired() const {
        std::lock_guard<std::mutex> lock(mutex);
        return failedPlacement.expired();
    }

   private:
    mutable std::mutex mutex;
    std::weak_ptr<PlacedNetwork> failedPlacement;
};

class PlaceThenFailExecutor : public TrainingExecutor {
   public:
    explicit PlaceThenFailExecutor(std::shared_ptr<FailedPlacementReleaseState> state)
        : state(std::move(state)) {}

    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        (void)observer;
        if (request.network == nullptr || request.completedPlacedNetwork == nullptr) {
            throw std::logic_error("TrainingRuns placement-release regression requires standalone network placement handoff.");
        }

        // A placement with an empty physical stamp is sufficient for this CPU-only
        // ownership test: TrainingRuns must still drop the Trainer's placement
        // reference after the run fails before it starts the next sequential run.
        std::vector<ThorImplementation::StampedNetwork> stamps(1);
        auto placement = std::make_shared<PlacedNetwork>(*request.network, std::move(stamps));
        state->remember(placement);
        *request.completedPlacedNetwork = std::move(placement);
        throw std::runtime_error("planned failure after placement");
    }

   private:
    std::shared_ptr<FailedPlacementReleaseState> state;
};

class RequirePriorFailedPlacementReleasedExecutor : public TrainingExecutor {
   public:
    explicit RequirePriorFailedPlacementReleasedExecutor(std::shared_ptr<FailedPlacementReleaseState> state)
        : state(std::move(state)) {}

    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        if (!state->failedPlacementExpired()) {
            throw std::runtime_error("prior failed TrainingRuns member still owns its placed network");
        }
        observedReleasedPlacement = true;
        observer.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(request, 1)));
    }

    bool observedReleasedPlacement = false;

   private:
    std::shared_ptr<FailedPlacementReleaseState> state;
};

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
        observer.onTrainingEvent(TrainingEvent::statsUpdated(makeStats(request, statsStep)));

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


class StartupOrderRecorder {
   public:
    void record(size_t runIndex) {
        std::lock_guard<std::mutex> lock(mutex);
        order.push_back(runIndex);
    }

    [[nodiscard]] std::vector<size_t> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        return order;
    }

   private:
    mutable std::mutex mutex;
    std::vector<size_t> order;
};

class OrderedStartupExecutor : public TrainingExecutor {
   public:
    OrderedStartupExecutor(size_t runIndex,
                           std::chrono::milliseconds delay,
                           std::shared_ptr<StartupOrderRecorder> recorder)
        : runIndex(runIndex), delay(delay), recorder(std::move(recorder)) {}

    void fit(const TrainingRunRequest& request,
             TrainingObserver& observer) override {
        std::this_thread::sleep_for(delay);
        if (!request.initialDeviceStartupSequencer) {
            throw std::runtime_error(
                "TrainingRuns did not provide an initial device startup "
                "sequencer.");
        }
        request.initialDeviceStartupSequencer([&]() {
            recorder->record(runIndex);
        });
        observer.onTrainingEvent(
            TrainingEvent::statsUpdated(makeStats(request, runIndex + 1)));
    }

   private:
    size_t runIndex;
    std::chrono::milliseconds delay;
    std::shared_ptr<StartupOrderRecorder> recorder;
};


class ArchitectureSavingExecutor final : public TrainingExecutor {
   public:
    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        THOR_THROW_IF_FALSE(request.network != nullptr);
        THOR_THROW_IF_FALSE(request.saveModelDirectory.has_value());
        THOR_THROW_IF_FALSE(!request.saveModelDirectory->empty());

        request.cancellationToken.throwIfCancellationRequested();
        TrainingStatsSnapshot stats = makeStats(request, 1);
        stats.epoch = request.initialCompletedEpochs + request.epochs;
        stats.epochs = stats.epoch;
        observer.onTrainingEvent(TrainingEvent::statsUpdated(stats));

        const std::filesystem::path artifactRoot(*request.saveModelDirectory);
        std::filesystem::remove_all(artifactRoot);
        request.network->save((artifactRoot / "latest").string(), /*overwrite=*/true);
        if (request.completedTrainingEpochs != nullptr) {
            *request.completedTrainingEpochs = stats.epoch;
        }
        calls += 1;
    }

    uint32_t calls = 0;
};


class RestartProgressExecutor : public TrainingExecutor {
   public:
    explicit RestartProgressExecutor(std::vector<std::vector<double>> attemptEpochLosses, bool writeLatestArtifact = false)
        : attemptEpochLosses(std::move(attemptEpochLosses)), writeLatestArtifact(writeLatestArtifact) {}

    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        calls += 1;
        sawLossRequested = sawLossRequested || request.runtime.scalarTensorsToReport.count("loss") != 0;
        lastEarlyCompletionPolicyCount = request.earlyCompletionPolicies.size();
        lastInitialCompletedEpochs = request.initialCompletedEpochs;
        initialCompletedEpochsByCall.push_back(request.initialCompletedEpochs);
        previousModelArtifactDirectoriesByCall.push_back(request.previousModelArtifactDirectory);
        lastFirstModelSelectionEpoch = request.firstModelSelectionEpoch;
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
            setSyntheticTrainingBatchCardinality(request, stats);
            const uint64_t trainingExamples = request.batchSession->getNumExamples(ExampleType::TRAIN);
            stats.samplesProcessedInEpoch = trainingExamples;
            stats.samplesProcessed = globalEpoch * trainingExamples;
            stats.loss = losses[std::min<size_t>(epoch - 1, losses.size() - 1)];
            observer.onTrainingEvent(TrainingEvent::statsUpdated(stats));
            finalEpoch = globalEpoch;
        }
        if (writeLatestArtifact && request.saveModelDirectory.has_value()) {
            std::filesystem::path root(request.saveModelDirectory.value());
            std::filesystem::remove_all(root);
            std::filesystem::create_directories(root / "latest");
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
    std::vector<std::optional<std::string>> previousModelArtifactDirectoriesByCall{};
    uint64_t lastFirstModelSelectionEpoch = 0;

   private:
    std::vector<std::vector<double>> attemptEpochLosses;
    bool writeLatestArtifact = false;
};


void populateLifecycleMetricStats(TrainingStatsSnapshot& stats,
                                  bool secondBatch,
                                  bool discardedAttempt) {
    const double scale = discardedAttempt ? 100.0 : 1.0;
    const uint64_t validExamples = secondBatch ? 2u : 4u;
    stats.batchSize = 4;
    stats.validExamplesInBatch = validExamples;

    auto addMetric = [&](const std::string& name,
                         MetricAggregation aggregation,
                         double value,
                         std::optional<double> numerator = std::nullopt,
                         std::optional<double> denominator = std::nullopt) {
        stats.metrics[name] = value;
        stats.metricBatchStats[name] = MetricBatchStat{
            .aggregation = aggregation,
            .value = value,
            .validExamples = validExamples,
            .numerator = numerator,
            .denominator = denominator,
        };
    };

    if (!secondBatch) {
        addMetric("mean", MetricAggregation::MEAN_BY_EXAMPLE, 2.5 * scale);
        addMetric("sum", MetricAggregation::SUM, 10.0 * scale);
        addMetric("min", MetricAggregation::MIN, 1.0 * scale);
        addMetric("max", MetricAggregation::MAX, 4.0 * scale);
        addMetric("ratio",
                  MetricAggregation::RATIO,
                  5.0 * scale,
                  10.0 * scale,
                  2.0);
    } else {
        addMetric("mean", MetricAggregation::MEAN_BY_EXAMPLE, 10.0 * scale);
        addMetric("sum", MetricAggregation::SUM, 20.0 * scale);
        addMetric("min", MetricAggregation::MIN, -3.0 * scale);
        addMetric("max", MetricAggregation::MAX, 11.0 * scale);
        addMetric("ratio",
                  MetricAggregation::RATIO,
                  10.0 * scale,
                  90.0 * scale,
                  9.0);
    }
}

class MetricAggregationLifecycleExecutor final : public TrainingExecutor {
   public:
    enum class Behavior { RESTART_ONCE_THEN_COMPLETE, INTERRUPT_AFTER_EPOCH };

    explicit MetricAggregationLifecycleExecutor(Behavior behavior)
        : behavior(behavior) {}

    void fit(const TrainingRunRequest& request, TrainingObserver& observer) override {
        calls += 1;
        const bool discardedAttempt =
            behavior == Behavior::RESTART_ONCE_THEN_COMPLETE && calls == 1;
        const uint32_t epochsToEmit =
            behavior == Behavior::INTERRUPT_AFTER_EPOCH ? 1u : request.epochs;

        uint64_t globalStep = 0;
        for (uint32_t epoch = 1; epoch <= epochsToEmit; ++epoch) {
            const uint64_t globalEpoch = request.initialCompletedEpochs + epoch;
            for (uint64_t batch = 0; batch < 2; ++batch) {
                TrainingStatsSnapshot stats;
                stats.phase = TrainingEventPhase::TRAIN;
                stats.epoch = globalEpoch;
                stats.epochs = request.initialCompletedEpochs + request.epochs;
                stats.step = ++globalStep;
                stats.stepInEpoch = batch + 1;
                stats.stepsPerEpoch = 2;
                stats.samplesProcessedInEpoch = batch == 0 ? 4 : 6;
                stats.samplesProcessed =
                    (globalEpoch - 1) * 6 + stats.samplesProcessedInEpoch;
                const bool successfulFinalEpoch =
                    !discardedAttempt && epoch == request.epochs;
                stats.loss = successfulFinalEpoch ? 8.0 : 10.0;
                populateLifecycleMetricStats(stats, batch == 1, discardedAttempt);
                observer.onTrainingEvent(TrainingEvent::statsUpdated(stats));
            }
        }

        if (behavior == Behavior::INTERRUPT_AFTER_EPOCH) {
            throw TrainingInterrupted("synthetic interruption after exact metric epoch");
        }
        if (request.completedTrainingEpochs != nullptr) {
            *request.completedTrainingEpochs =
                request.initialCompletedEpochs + request.epochs;
        }
    }

    uint32_t calls = 0;

   private:
    Behavior behavior;
};



std::shared_ptr<Network> makeExactMetricAggregationNetwork(const std::string& name) {
    auto network = std::make_shared<Network>(name);
    NetworkInput values = NetworkInput::Builder()
                              .network(*network)
                              .name("values")
                              .dimensions({1})
                              .dataType(DataType::FP32)
                              .build();
    NetworkInput weights = NetworkInput::Builder()
                               .network(*network)
                               .name("weights")
                               .dimensions({1})
                               .dataType(DataType::FP32)
                               .build();

    Mean mean = Mean::Builder().network(*network).values(values.getFeatureOutput().value()).build();
    Sum sum = Sum::Builder().network(*network).values(values.getFeatureOutput().value()).build();
    Min min = Min::Builder().network(*network).values(values.getFeatureOutput().value()).build();
    Max max = Max::Builder().network(*network).values(values.getFeatureOutput().value()).build();
    WeightedMean weightedMean = WeightedMean::Builder()
                                    .network(*network)
                                    .values(values.getFeatureOutput().value())
                                    .weights(weights.getFeatureOutput().value())
                                    .build();

    NetworkOutput::Builder().network(*network).name("value_mean").inputTensor(mean.getMetric()).build();
    NetworkOutput::Builder().network(*network).name("value_sum").inputTensor(sum.getMetric()).build();
    NetworkOutput::Builder().network(*network).name("value_min").inputTensor(min.getMetric()).build();
    NetworkOutput::Builder().network(*network).name("value_max").inputTensor(max.getMetric()).build();
    NetworkOutput::Builder()
        .network(*network)
        .name("value_weighted_mean")
        .inputTensor(weightedMean.getMetric())
        .build();
    return network;
}

std::shared_ptr<Network> makeNetworkWithOutput(const std::string& name, const std::vector<uint64_t>& dimensions) {
    // These helpers are used with fake executors, but the Network itself must still be a
    // valid graph: TrainingRuns intentionally validates graph structure before any executor
    // is launched.  Historical callers pass {0, output_features}, where 0 represented the
    // runtime batch dimension of a fabricated Tensor.  Build the equivalent API graph and
    // let placement supply the real batch dimension instead of manufacturing a floating,
    // zero-sized output tensor.
    THOR_THROW_IF_FALSE(!dimensions.empty());
    const uint64_t outputFeatures = dimensions.back();
    THOR_THROW_IF_FALSE(outputFeatures > 0);

    auto network = std::make_shared<Network>(name);
    NetworkInput features = NetworkInput::Builder()
                                .network(*network)
                                .name("features")
                                .dimensions({4})
                                .dataType(DataType::FP32)
                                .build();
    FullyConnected predictions = FullyConnected::Builder()
                                     .network(*network)
                                     .featureInput(features.getFeatureOutput().value())
                                     .numOutputFeatures(outputFeatures)
                                     .hasBias(true)
                                     .noActivation()
                                     .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("predictions")
        .inputTensor(predictions.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();
    return network;
}

std::shared_ptr<Network> makeDemandSignatureNetwork(const std::string& name) {
    auto network = std::make_shared<Network>(name);
    NetworkInput features = NetworkInput::Builder()
                                .network(*network)
                                .name("features")
                                .dimensions({4})
                                .dataType(DataType::FP32)
                                .build();
    FullyConnected prediction = FullyConnected::Builder()
                                    .network(*network)
                                    .featureInput(features.getFeatureOutput().value())
                                    .numOutputFeatures(1)
                                    .hasBias(true)
                                    .noActivation()
                                    .build();

    const Tensor predictionTensor = prediction.getFeatureOutput().value();
    NetworkOutput::Builder().network(*network).name("daily").inputTensor(predictionTensor).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("aggregate").inputTensor(predictionTensor).dataType(DataType::FP32).build();
    NetworkOutput::Builder().network(*network).name("forecast_p90").inputTensor(predictionTensor).dataType(DataType::FP32).build();
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
    THOR_THROW_IF_FALSE(!dailyOutputDimensions.empty());
    const uint64_t dailyOutputFeatures = dailyOutputDimensions.back();
    THOR_THROW_IF_FALSE(dailyOutputFeatures > 0);

    auto network = std::make_shared<Network>(name);
    NetworkInput features = NetworkInput::Builder()
                                .network(*network)
                                .name("features")
                                .dimensions({4})
                                .dataType(DataType::FP32)
                                .build();
    FullyConnected daily = FullyConnected::Builder()
                               .network(*network)
                               .featureInput(features.getFeatureOutput().value())
                               .numOutputFeatures(dailyOutputFeatures)
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

    if (includeDailyOutput) {
        NetworkOutput::Builder()
            .network(*network)
            .name("daily")
            .inputTensor(daily.getFeatureOutput().value())
            .dataType(DataType::FP32)
            .build();
    } else {
        Stub::Builder().network(*network).inputTensor(daily.getFeatureOutput().value()).build();
    }
    NetworkOutput::Builder()
        .network(*network)
        .name("aggregate")
        .inputTensor(aggregate.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("forecast_p90")
        .inputTensor(p90.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();
    return network;
}

std::shared_ptr<Network> makeAuxiliaryMetricNetwork(const std::string& name, const std::string& selectedThresholdInputName) {
    auto network = std::make_shared<Network>(name);
    NetworkInput features =
        NetworkInput::Builder().network(*network).name("features").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput actual =
        NetworkInput::Builder().network(*network).name("actual").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput peakThreshold =
        NetworkInput::Builder().network(*network).name("peak_threshold").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput alternateThreshold = NetworkInput::Builder()
                                          .network(*network)
                                          .name("alternate_threshold")
                                          .dimensions({4})
                                          .dataType(DataType::FP32)
                                          .build();

    std::shared_ptr<Activation> prediction =
        Relu::Builder().network(*network).featureInput(features.getFeatureOutput().value()).build();
    const Tensor selectedThreshold = selectedThresholdInputName == "peak_threshold"
                                         ? peakThreshold.getFeatureOutput().value()
                                         : alternateThreshold.getFeatureOutput().value();

    ThorImplementation::Expression actualExpression =
        ThorImplementation::Expression::input("actual", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::Expression thresholdExpression =
        ThorImplementation::Expression::input("threshold", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::ExpressionDefinition maskDefinition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs(
            {{"mask", actualExpression.greaterEqual(thresholdExpression).cast(ThorImplementation::DataType::FP32)}}));
    CustomLayer mask = CustomLayer::Builder()
                           .network(*network)
                           .expression(ThorImplementation::DynamicExpression::fromExpressionDefinition(maskDefinition))
                           .inputNames({"actual", "threshold"})
                           .outputNames({"mask"})
                           .inputInterface({{"actual", actual.getFeatureOutput().value()}, {"threshold", selectedThreshold}})
                           .build();
    WeightedMean peakMean = WeightedMean::Builder()
                                .network(*network)
                                .values(prediction->getFeatureOutput().value())
                                .weights(mask.getOutput("mask"))
                                .build();

    NetworkOutput::Builder().network(*network).name("prediction").inputTensor(prediction->getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(*network).name("peak_mean").inputTensor(peakMean.getMetric()).build();
    // Keep both candidate auxiliary inputs in the public signature so the test
    // isolates report-subgraph wiring rather than whole-network input names.
    NetworkOutput::Builder()
        .network(*network)
        .name("peak_threshold_debug")
        .inputTensor(peakThreshold.getFeatureOutput().value())
        .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("alternate_threshold_debug")
        .inputTensor(alternateThreshold.getFeatureOutput().value())
        .build();
    return network;
}

std::shared_ptr<Network> makeCustomMetricAggregationNetwork(const std::string& name, MetricAggregation aggregation) {
    auto network = std::make_shared<Network>(name);
    NetworkInput features =
        NetworkInput::Builder().network(*network).name("features").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput actual =
        NetworkInput::Builder().network(*network).name("actual").dimensions({4}).dataType(DataType::FP32).build();

    std::shared_ptr<Activation> prediction =
        Relu::Builder().network(*network).featureInput(features.getFeatureOutput().value()).build();

    ThorImplementation::Expression predictionsExpression =
        ThorImplementation::Expression::input("predictions", DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labelsExpression =
        ThorImplementation::Expression::input("labels", DataType::FP32, DataType::FP32);
    ThorImplementation::Expression difference = predictionsExpression - labelsExpression;
    ThorImplementation::Expression metricExpression =
        (difference * difference).reduce_mean({0, 1}, {0}, DataType::FP32);
    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{"metric", metricExpression}}));

    CustomMetric metric = CustomMetric::Builder()
                              .network(*network)
                              .expression(ThorImplementation::DynamicExpression::fromExpressionDefinition(definition))
                              .predictions(prediction->getFeatureOutput().value())
                              .labels(actual.getFeatureOutput().value())
                              .aggregation(aggregation)
                              .build();

    NetworkOutput::Builder().network(*network).name("prediction").inputTensor(prediction->getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(*network).name("custom_metric").inputTensor(metric.getMetric()).build();
    return network;
}

std::shared_ptr<Network> makeAuxiliaryLossNetwork(const std::string& name, const std::string& selectedThresholdInputName) {
    auto network = std::make_shared<Network>(name);
    NetworkInput features =
        NetworkInput::Builder().network(*network).name("features").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput actual =
        NetworkInput::Builder().network(*network).name("actual").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput peakThreshold =
        NetworkInput::Builder().network(*network).name("peak_threshold").dimensions({4}).dataType(DataType::FP32).build();
    NetworkInput alternateThreshold = NetworkInput::Builder()
                                          .network(*network)
                                          .name("alternate_threshold")
                                          .dimensions({4})
                                          .dataType(DataType::FP32)
                                          .build();

    std::shared_ptr<Activation> prediction =
        Relu::Builder().network(*network).featureInput(features.getFeatureOutput().value()).build();
    const Tensor selectedThreshold = selectedThresholdInputName == "peak_threshold"
                                         ? peakThreshold.getFeatureOutput().value()
                                         : alternateThreshold.getFeatureOutput().value();

    ThorImplementation::Expression actualExpression =
        ThorImplementation::Expression::input("actual", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::Expression thresholdExpression =
        ThorImplementation::Expression::input("threshold", ThorImplementation::DataType::FP32, ThorImplementation::DataType::FP32);
    ThorImplementation::ExpressionDefinition maskDefinition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs(
            {{"mask", actualExpression.greaterEqual(thresholdExpression).cast(ThorImplementation::DataType::FP32)}}));
    CustomLayer mask = CustomLayer::Builder()
                           .network(*network)
                           .expression(ThorImplementation::DynamicExpression::fromExpressionDefinition(maskDefinition))
                           .inputNames({"actual", "threshold"})
                           .outputNames({"mask"})
                           .inputInterface({{"actual", actual.getFeatureOutput().value()}, {"threshold", selectedThreshold}})
                           .build();
    MSE peakLoss = MSE::Builder()
                       .network(*network)
                       .predictions(prediction->getFeatureOutput().value())
                       .labels(actual.getFeatureOutput().value())
                       .exampleWeights(mask.getOutput("mask"))
                       .reportsBatchLoss()
                       .build();

    NetworkOutput::Builder().network(*network).name("prediction").inputTensor(prediction->getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(*network).name("peak_mse").inputTensor(peakLoss.getLoss()).build();
    NetworkOutput::Builder()
        .network(*network)
        .name("peak_threshold_debug")
        .inputTensor(peakThreshold.getFeatureOutput().value())
        .build();
    NetworkOutput::Builder()
        .network(*network)
        .name("alternate_threshold_debug")
        .inputTensor(alternateThreshold.getFeatureOutput().value())
        .build();
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


void ensureMinimalValidTrainingRunsTestNetwork(const std::shared_ptr<Network>& network) {
    if (network == nullptr || network->getNumLayers() != 0) {
        return;
    }

    // Scheduler/restart tests use fake executors, but TrainingRuns deliberately
    // validates every supplied graph before launching those executors. Keep the
    // execution fake while making graph validity real.
    NetworkInput features = NetworkInput::Builder()
                                .network(*network)
                                .name("features")
                                .dimensions({4})
                                .dataType(DataType::FP32)
                                .build();
    std::shared_ptr<Activation> relu =
        Relu::Builder().network(*network).featureInput(features.getFeatureOutput().value()).build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction")
        .inputTensor(relu->getFeatureOutput().value())
        .build();
}

std::shared_ptr<Trainer> makeTrainer(std::shared_ptr<Network> network,
                                    std::shared_ptr<TrainingExecutor> executor,
                                    std::optional<std::string> saveModelDirectory = std::nullopt,
                                    bool saveModelOverwrite = false) {
    ensureMinimalValidTrainingRunsTestNetwork(network);
    return std::make_shared<Trainer>(Trainer::Builder()
                                         .network(std::move(network))
                                         .data(makeFakeTrainingData())
                                         .executor(std::move(executor))
                                         .observer(std::make_shared<NullTrainingObserver>())
                                         .saveModelDirectory(std::move(saveModelDirectory))
                                         .saveModelOverwrite(saveModelOverwrite)
                                         .build());
}

std::shared_ptr<Trainer> makeTrainerWithData(std::shared_ptr<Network> network,
                                            std::shared_ptr<TrainingExecutor> executor,
                                            std::shared_ptr<const TrainingData> data,
                                            std::string saveModelDirectory) {
    ensureMinimalValidTrainingRunsTestNetwork(network);
    return std::make_shared<Trainer>(Trainer::Builder()
                                         .network(std::move(network))
                                         .data(std::move(data))
                                         .executor(std::move(executor))
                                         .observer(std::make_shared<NullTrainingObserver>())
                                         .saveModelDirectory(std::move(saveModelDirectory))
                                         .saveModelOverwrite(true)
                                         .build());
}

std::shared_ptr<Trainer> makePhaseTrainerForValidation(const std::string& name,
                                                                    std::shared_ptr<TrainingExecutor> executor) {
    auto phaseNetwork = std::make_shared<Network>(name + "_phase");

    NetworkInput features = NetworkInput::Builder()
                                .network(*phaseNetwork)
                                .name("features")
                                .dimensions({4})
                                .dataType(DataType::FP32)
                                .build();
    NetworkInput labels = NetworkInput::Builder()
                              .network(*phaseNetwork)
                              .name("labels")
                              .dimensions({1})
                              .dataType(DataType::FP32)
                              .build();

    FullyConnected prediction = FullyConnected::Builder()
                                    .network(*phaseNetwork)
                                    .featureInput(features.getFeatureOutput().value())
                                    .numOutputFeatures(1)
                                    .hasBias(true)
                                    .noActivation()
                                    .build();
    MSE loss = MSE::Builder()
                   .network(*phaseNetwork)
                   .predictions(prediction.getFeatureOutput().value())
                   .labels(labels.getFeatureOutput().value())
                   .lossDataType(DataType::FP32)
                   .build();

    NetworkOutput::Builder().network(*phaseNetwork).name("prediction").inputTensor(prediction.getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(*phaseNetwork).name("mse_loss").inputTensor(loss.getLoss()).build();

    auto phase = std::make_shared<TrainingPhase>("phase", phaseNetwork, true);
    auto step = std::make_shared<TrainingStep>("step",
                                               std::vector<std::shared_ptr<TrainingPhase>>{phase},
                                               Sgd::Builder().initialLearningRate(0.01f).build(),
                                               std::vector<ParameterReference>{});
    auto program = std::make_shared<TrainingProgram>(std::vector<std::shared_ptr<TrainingStep>>{step});

    return std::make_shared<Trainer>(Trainer::Builder()
                                         .data(makeFakeTrainingData())
                                         .executor(std::move(executor))
                                         .observer(std::make_shared<NullTrainingObserver>())
                                         .trainingProgram(std::move(program))
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


TEST(TrainingRuns, ReportableLossesExposeLossWeightsForNamedGraphLosses) {
    auto network = makeLossWeightedDemandNetwork("training-runs-loss-metric-hints");

    const std::vector<NetworkLossReference> reportableLosses = network->getReportableLosses();

    auto findLoss = [&](const std::string& lossName) -> const NetworkLossReference* {
        auto it = std::find_if(reportableLosses.begin(), reportableLosses.end(), [&](const NetworkLossReference& reference) {
            return reference.lossName == lossName;
        });
        return it == reportableLosses.end() ? nullptr : &*it;
    };

    const NetworkLossReference* daily = findLoss("daily_loss");
    ASSERT_NE(daily, nullptr);
    EXPECT_EQ(daily->predictionOutputName, "daily");
    EXPECT_EQ(daily->targetInputName, "observed_daily");
    EXPECT_EQ(daily->lossLayerType, "CustomLoss");
    EXPECT_DOUBLE_EQ(daily->lossWeight, 2.0);

    const NetworkLossReference* aggregate = findLoss("aggregate_loss");
    ASSERT_NE(aggregate, nullptr);
    EXPECT_EQ(aggregate->predictionOutputName, "aggregate");
    EXPECT_EQ(aggregate->targetInputName, "observed_aggregate");
    EXPECT_EQ(aggregate->lossLayerType, "CustomLoss");
    EXPECT_DOUBLE_EQ(aggregate->lossWeight, 1.5);

    const NetworkLossReference* p90 = findLoss("p90_loss");
    ASSERT_NE(p90, nullptr);
    EXPECT_EQ(p90->predictionOutputName, "forecast_p90");
    EXPECT_EQ(p90->targetInputName, "observed_daily");
    EXPECT_EQ(p90->lossLayerType, "CustomLoss");
    EXPECT_FALSE(p90->quantile.has_value());
    EXPECT_DOUBLE_EQ(p90->lossWeight, 0.5);
}


TEST(TrainingRuns, InternalNetworkOutputsAreNotPredictionSourcesForReportableLosses) {
    auto network = std::make_shared<Network>("training-runs-internal-output-loss-source");
    NetworkInput features = NetworkInput::Builder()
                                .network(*network)
                                .name("features")
                                .dimensions({4})
                                .dataType(DataType::FP32)
                                .build();
    NetworkInput labels = NetworkInput::Builder()
                              .network(*network)
                              .name("labels")
                              .dimensions({1})
                              .dataType(DataType::FP32)
                              .build();
    FullyConnected prediction = FullyConnected::Builder()
                                    .network(*network)
                                    .featureInput(features.getFeatureOutput().value())
                                    .numOutputFeatures(1)
                                    .hasBias(true)
                                    .noActivation()
                                    .build();
    MSE loss = MSE::Builder()
                   .network(*network)
                   .predictions(prediction.getFeatureOutput().value())
                   .labels(labels.getFeatureOutput().value())
                   .lossDataType(DataType::FP32)
                   .build();

    NetworkOutput::Builder()
        .network(*network)
        .name("private_prediction_handoff")
        .inputTensor(prediction.getFeatureOutput().value())
        .external(false)
        .build();
    NetworkOutput::Builder().network(*network).name("mse_loss").inputTensor(loss.getLoss()).build();

    const std::vector<NetworkLossReference> reportableLosses = network->getReportableLosses();
    ASSERT_EQ(reportableLosses.size(), 1u);
    EXPECT_EQ(reportableLosses[0].lossName, "mse_loss");
    EXPECT_TRUE(reportableLosses[0].predictionOutputName.empty());
    EXPECT_EQ(reportableLosses[0].targetInputName, "labels");
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



TEST(TrainingRuns, ValidatesEachNetworkOnceAndReusesStartupSnapshotDuringFit) {
    auto network = std::make_shared<CountingTrainingValidationNetwork>("training-runs-validation-snapshot");
    NetworkInput features = NetworkInput::Builder()
                                .network(*network)
                                .name("features")
                                .dimensions({4})
                                .dataType(DataType::FP32)
                                .build();
    NetworkInput actual = NetworkInput::Builder()
                               .network(*network)
                               .name("actual")
                               .dimensions({4})
                               .dataType(DataType::FP32)
                               .build();
    std::shared_ptr<Activation> prediction =
        Relu::Builder().network(*network).featureInput(features.getFeatureOutput().value()).build();
    MAE loss = MAE::Builder()
                   .network(*network)
                   .predictions(prediction->getFeatureOutput().value())
                   .labels(actual.getFeatureOutput().value())
                   .lossDataType(DataType::FP32)
                   .build();
    Mean labelMean = Mean::Builder().network(*network).values(actual.getFeatureOutput().value()).build();
    NetworkOutput::Builder()
        .network(*network)
        .name("prediction")
        .inputTensor(prediction->getFeatureOutput().value())
        .build();
    NetworkOutput::Builder().network(*network).name("prediction_mae").inputTensor(loss.getLoss()).build();
    NetworkOutput::Builder().network(*network).name("label_mean").inputTensor(labelMean.getMetric()).build();

    auto executor = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{1.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer}});
    EXPECT_EQ(network->trainingValidationCount(), 1U);

    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports["fold_0"] = {"prediction_mae", "label_mean"};
    TrainingRunsResult result = runs.fit(TrainerFitOptions{.epochs = 1}, sessionOptions);
    EXPECT_TRUE(result.allCompleted());
    EXPECT_EQ(network->trainingValidationCount(), 1U);
}

TEST(TrainingRuns, PhaseBackedStartupValidatesEveryPhaseNetworkOnceAndReusesSnapshotDuringFit) {
    auto activeNetwork = std::make_shared<CountingTrainingValidationNetwork>("training-runs-active-phase-validation-snapshot");
    NetworkInput activeFeatures = NetworkInput::Builder()
                                      .network(*activeNetwork)
                                      .name("features")
                                      .dimensions({4})
                                      .dataType(DataType::FP32)
                                      .build();
    std::shared_ptr<Activation> activePrediction =
        Relu::Builder().network(*activeNetwork).featureInput(activeFeatures.getFeatureOutput().value()).build();
    NetworkOutput::Builder()
        .network(*activeNetwork)
        .name("prediction")
        .inputTensor(activePrediction->getFeatureOutput().value())
        .build();

    auto futureNetwork = std::make_shared<CountingTrainingValidationNetwork>("training-runs-future-phase-validation-snapshot");
    NetworkInput futureFeatures = NetworkInput::Builder()
                                      .network(*futureNetwork)
                                      .name("features")
                                      .dimensions({4})
                                      .dataType(DataType::FP32)
                                      .build();
    std::shared_ptr<Activation> futurePrediction =
        Relu::Builder().network(*futureNetwork).featureInput(futureFeatures.getFeatureOutput().value()).build();
    NetworkOutput::Builder()
        .network(*futureNetwork)
        .name("prediction")
        .inputTensor(futurePrediction->getFeatureOutput().value())
        .build();

    auto activePhase = std::make_shared<TrainingPhase>("active", activeNetwork, true);
    auto futurePhase = std::make_shared<TrainingPhase>("future", futureNetwork, false);
    auto step = std::make_shared<TrainingStep>("step",
                                               std::vector<std::shared_ptr<TrainingPhase>>{activePhase, futurePhase},
                                               Sgd::Builder().initialLearningRate(0.01f).build(),
                                               std::vector<ParameterReference>{});
    auto program = std::make_shared<TrainingProgram>(std::vector<std::shared_ptr<TrainingStep>>{step});
    auto executor = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{1.0}});
    auto trainer = std::make_shared<Trainer>(Trainer::Builder()
                                                 .data(makeFakeTrainingData())
                                                 .executor(executor)
                                                 .observer(std::make_shared<NullTrainingObserver>())
                                                 .trainingProgram(program)
                                                 .build());

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer}});
    EXPECT_EQ(activeNetwork->trainingValidationCount(), 1U);
    EXPECT_EQ(futureNetwork->trainingValidationCount(), 1U);

    TrainingRunsResult result = runs.fit(TrainerFitOptions{.epochs = 1});
    EXPECT_TRUE(result.allCompleted());
    EXPECT_EQ(activeNetwork->trainingValidationCount(), 1U);
    EXPECT_EQ(futureNetwork->trainingValidationCount(), 1U);
}

TEST(TrainingRuns, UsesTrainingPhaseSignaturesForEnsembleValidation) {
    auto coordinator = std::make_shared<Coordinator>(2);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);

    std::shared_ptr<Trainer> trainer0 = makePhaseTrainerForValidation("phase_member_0", executor0);
    std::shared_ptr<Trainer> trainer1 = makePhaseTrainerForValidation("phase_member_1", executor1);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "phase_group"},
                       TrainingRunsSpec{"fold_1", trainer1, "phase_group"}},
                      TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                      2.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"phase_group", {"mse_loss"}}};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(TrainerFitOptions{1}, sessionOptions);
        } catch (...) {
            exception = std::current_exception();
        }
    });

    ASSERT_TRUE(coordinator->waitForAllStarted());
    coordinator->releaseAll();
    fitThread.join();
    rethrowIfSet(exception);
    EXPECT_TRUE(result.has_value());
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
                      2.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"demand", {"daily_loss", "p90_loss"}}};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(TrainerFitOptions{1}, sessionOptions);
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
                      2.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"demand", {"daily_loss", "p90_loss"}}};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(TrainerFitOptions{1}, sessionOptions);
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

TEST(TrainingRuns, ComposesGraphMetricsWithAuxiliaryInputs) {
    auto network0 = makeAuxiliaryMetricNetwork("training-runs-aux-metric-compose-0", "peak_threshold");
    auto network1 = makeAuxiliaryMetricNetwork("training-runs-aux-metric-compose-1", "peak_threshold");
    auto coordinator = std::make_shared<Coordinator>(2);
    auto executor0 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    auto executor1 = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "demand"}, TrainingRunsSpec{"fold_1", trainer1, "demand"}});
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"demand", {"peak_mean"}}};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(TrainerFitOptions{1}, sessionOptions);
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
    ASSERT_EQ(ensemble.namedGraphMetrics.size(), 1u);
    EXPECT_EQ(ensemble.namedGraphMetrics.front().name, "peak_mean");
}

TEST(TrainingRuns, RejectsGraphMetricsWithDifferentAuxiliaryInputBoundaries) {
    auto network0 = makeAuxiliaryMetricNetwork("training-runs-aux-metric-0", "peak_threshold");
    auto network1 = makeAuxiliaryMetricNetwork("training-runs-aux-metric-1", "alternate_threshold");
    auto executor0 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{1.0}});
    auto executor1 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{1.0}});
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "demand"},
                       TrainingRunsSpec{"fold_1", trainer1, "demand"}});
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"demand", {"peak_mean"}}};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    EXPECT_THROW((void)runs.fit(TrainerFitOptions{1}, sessionOptions), std::runtime_error);
    EXPECT_EQ(executor0->calls, 0u);
    EXPECT_EQ(executor1->calls, 0u);
}

TEST(TrainingRuns, AggregatesGraphMetricsExactlyAcrossTailBatchesAndSourcePopulations) {
    const std::filesystem::path root = uniqueTempPath("training-runs-exact-metric-aggregation");
    const std::filesystem::path member0Artifact = root / "member_0";
    const std::filesystem::path member1Artifact = root / "member_1";

    auto source0 = makeExactMetricTrainingData(
        "exact_metric_source_0",
        {1.0f, 2.0f, 3.0f, 4.0f},
        {1.0f, 1.0f, 1.0f, 1.0f});
    // The second population has a zero weight sum but a non-zero weighted
    // numerator. Exact ratio aggregation must retain that numerator instead of
    // reconstructing it from the displayed per-population value of zero.
    auto source1 = makeExactMetricTrainingData(
        "exact_metric_source_1",
        {10.0f, 20.0f},
        {-1.0f, 1.0f});
    auto testData = makeExactMetricTestData(
        "exact_metric_test",
        {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f});

    auto executor0 = std::make_shared<ArchitectureSavingExecutor>();
    auto executor1 = std::make_shared<ArchitectureSavingExecutor>();
    std::shared_ptr<Trainer> trainer0 = makeTrainerWithData(
        makeExactMetricAggregationNetwork("exact_metric_member_0"),
        executor0,
        source0,
        member0Artifact.string());
    std::shared_ptr<Trainer> trainer1 = makeTrainerWithData(
        makeExactMetricAggregationNetwork("exact_metric_member_1"),
        executor1,
        source1,
        member1Artifact.string());

    TrainingRuns runs(
        {TrainingRunsSpec{"fold_0", trainer0, "metrics"},
         TrainingRunsSpec{"fold_1", trainer1, "metrics"}},
        TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
        2.0,
        1u);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"metrics",
                               {"value_mean",
                                "value_sum",
                                "value_min",
                                "value_max",
                                "value_weighted_mean"}}};
    sessionOptions.evaluation.testData = testData;

    TrainingRunsResult result = runs.fit(TrainerFitOptions{.epochs = 1}, sessionOptions);
    ASSERT_TRUE(result.allCompleted());
    EXPECT_EQ(executor0->calls, 1u);
    EXPECT_EQ(executor1->calls, 1u);

    const std::map<std::string, double> expected{
        {"value_mean", 40.0 / 6.0},
        {"value_sum", 40.0},
        {"value_min", 1.0},
        {"value_max", 20.0},
        {"value_weighted_mean", 5.0},
    };
    const std::map<std::string, MetricAggregation> expectedAggregations{
        {"value_mean", MetricAggregation::MEAN_BY_EXAMPLE},
        {"value_sum", MetricAggregation::SUM},
        {"value_min", MetricAggregation::MIN},
        {"value_max", MetricAggregation::MAX},
        {"value_weighted_mean", MetricAggregation::RATIO},
    };

    const TrainingEnsembleResult& ensemble = result.ensemble("metrics");
    ASSERT_EQ(ensemble.namedGraphMetrics.size(), expected.size());
    for (const auto& [name, expectedValue] : expected) {
        const auto metricIt = std::find_if(
            ensemble.namedGraphMetrics.begin(),
            ensemble.namedGraphMetrics.end(),
            [&](const TrainingNamedMetricResult& metric) { return metric.name == name; });
        ASSERT_NE(metricIt, ensemble.namedGraphMetrics.end()) << name;
        ASSERT_TRUE(metricIt->trainValue.has_value()) << name;
        ASSERT_TRUE(metricIt->testValue.has_value()) << name;
        EXPECT_NEAR(metricIt->trainValue.value(), expectedValue, 1.0e-5) << name;
        EXPECT_NEAR(metricIt->testValue.value(), expectedValue, 1.0e-5) << name;
    }

    ASSERT_EQ(ensemble.members.size(), 2u);
    for (const TrainingEnsembleMemberResult& member : ensemble.members) {
        ASSERT_EQ(member.finalTestMetrics.size(), expected.size());
        for (const auto& [name, expectedValue] : expected) {
            ASSERT_TRUE(member.finalTestMetrics.count(name) != 0) << name;
            EXPECT_NEAR(member.finalTestMetrics.at(name), expectedValue, 1.0e-5) << name;
        }
    }

    for (const TrainingRunResult& run : result.runs()) {
        ASSERT_TRUE(run.finalTestStats.has_value());
        const TrainingStatsSnapshot& stats = run.finalTestStats.value();
        EXPECT_EQ(stats.samplesProcessed, 6u);
        EXPECT_EQ(stats.validExamplesInBatch, 2u);
        for (const auto& [name, expectedValue] : expected) {
            ASSERT_TRUE(stats.metrics.count(name) != 0) << name;
            EXPECT_NEAR(stats.metrics.at(name), expectedValue, 1.0e-5) << name;
            const auto statisticIt = stats.metricBatchStats.find(name);
            ASSERT_NE(statisticIt, stats.metricBatchStats.end()) << name;
            EXPECT_EQ(statisticIt->second.aggregation, expectedAggregations.at(name)) << name;
            EXPECT_EQ(statisticIt->second.validExamples, 6u) << name;
            EXPECT_NEAR(statisticIt->second.value, expectedValue, 1.0e-5) << name;
        }
        const MetricBatchStat& weightedStat = stats.metricBatchStats.at("value_weighted_mean");
        ASSERT_TRUE(weightedStat.numerator.has_value());
        ASSERT_TRUE(weightedStat.denominator.has_value());
        EXPECT_NEAR(weightedStat.numerator.value(), 20.0, 1.0e-5);
        EXPECT_NEAR(weightedStat.denominator.value(), 4.0, 1.0e-5);
    }

    std::filesystem::remove_all(root);
}

TEST(TrainingRuns, RejectsGraphMetricsWithDifferentAggregationContracts) {
    auto network0 = makeCustomMetricAggregationNetwork(
        "training-runs-custom-metric-mean", MetricAggregation::MEAN_BY_EXAMPLE);
    auto network1 = makeCustomMetricAggregationNetwork(
        "training-runs-custom-metric-sum", MetricAggregation::SUM);
    auto executor0 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{1.0}});
    auto executor1 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{1.0}});
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "demand"},
                       TrainingRunsSpec{"fold_1", trainer1, "demand"}});
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"demand", {"custom_metric"}}};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    EXPECT_THROW((void)runs.fit(TrainerFitOptions{1}, sessionOptions), std::runtime_error);
    EXPECT_EQ(executor0->calls, 0u);
    EXPECT_EQ(executor1->calls, 0u);
}

TEST(TrainingRuns, RejectsGraphLossesWithDifferentAuxiliaryInputBoundaries) {
    auto network0 = makeAuxiliaryLossNetwork("training-runs-aux-loss-0", "peak_threshold");
    auto network1 = makeAuxiliaryLossNetwork("training-runs-aux-loss-1", "alternate_threshold");
    auto executor0 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{1.0}});
    auto executor1 = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{1.0}});
    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "demand"},
                       TrainingRunsSpec{"fold_1", trainer1, "demand"}});
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"demand", {"peak_mse"}}};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    EXPECT_THROW((void)runs.fit(TrainerFitOptions{1}, sessionOptions), std::runtime_error);
    EXPECT_EQ(executor0->calls, 0u);
    EXPECT_EQ(executor1->calls, 0u);
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
    EXPECT_FALSE(ensemble.hasEnsembleEvaluationMetrics());
}

TEST(TrainingRuns, NamedMetricResultsUseGraphLossesAndSourceLossWeight) {
    auto network = makeLossWeightedDemandNetwork("training-runs-reported-loss-weight-resolution");
    auto coordinator = std::make_shared<Coordinator>(1);
    auto executor = std::make_shared<CoordinatedExecutor>(coordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer, "demand"}},
                      TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                      2.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"demand", {}}};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    std::optional<TrainingRunsResult> result;
    std::exception_ptr exception;
    std::thread fitThread([&]() {
        try {
            result = runs.fit(TrainerFitOptions{1}, sessionOptions);
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

TEST(TrainingRuns, ReportedLossResolutionFailsFastForMissingAndAmbiguousGraphLosses) {
    auto signatureOnlyNetwork = makeDemandSignatureNetwork("training-runs-missing-reported-loss");
    auto signatureCoordinator = std::make_shared<Coordinator>(1);
    auto signatureExecutor = std::make_shared<CoordinatedExecutor>(signatureCoordinator, FakeExecutorBehavior::COMPLETE_AFTER_RELEASE);
    std::shared_ptr<Trainer> signatureTrainer = makeTrainer(signatureOnlyNetwork, signatureExecutor);

    TrainingRuns runs({TrainingRunsSpec{"fold_0", signatureTrainer, "demand"}},
                      TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                      2.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.reports = {{"demand", {"daily_loss"}}};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;
    EXPECT_THROW(static_cast<void>(runs.fit(TrainerFitOptions{1}, sessionOptions)), std::runtime_error);
}


TEST(TrainingEnsembleResult, NamedMetricValuesContributeToEvaluationMetricPresence) {
    TrainingEnsembleResult ensemble;
    EXPECT_FALSE(ensemble.hasNamedMetricValues());
    EXPECT_FALSE(ensemble.hasEnsembleEvaluationMetrics());

    TrainingNamedMetricResult metric;
    metric.name = "daily_loss";
    metric.testValue = 0.25;
    ensemble.namedMetrics.push_back(metric);

    ASSERT_EQ(ensemble.namedMetrics.size(), 1u);
    EXPECT_TRUE(ensemble.namedMetrics[0].hasValue());
    EXPECT_TRUE(ensemble.hasNamedMetricValues());
    EXPECT_TRUE(ensemble.hasEnsembleEvaluationMetrics());
    EXPECT_EQ(ensemble.namedMetrics[0].name, "daily_loss");
    ASSERT_TRUE(ensemble.namedMetrics[0].testValue.has_value());
    EXPECT_DOUBLE_EQ(ensemble.namedMetrics[0].testValue.value(), 0.25);
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
    EXPECT_EQ(ensemble.outputSignature[0].dimensions, (std::vector<uint64_t>{10}));
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
    makeReluMemberNetwork("training_runs_save_ensemble_member_source_0", {10}, {"predictions"})->save(fold0.string(), true);
    makeReluMemberNetwork("training_runs_save_ensemble_member_source_2", {10}, {"predictions"})->save(fold2.string(), true);
    {
        std::ofstream(fold0 / "training_selection_metadata.json") << "{}\n";
        std::ofstream(fold2 / "training_selection_metadata.json") << "{}\n";
    }

    TrainingRunResult result0 = TrainingRunResult::completedResult("fold_0", {}, {}, {}, TrainingRunCompletionReason::COMPLETED, 1, 1, 1.0, fold0.string());
    TrainingRunResult result1 = TrainingRunResult::fromException("fold_1", std::make_exception_ptr(std::runtime_error("planned failure")));
    TrainingRunResult result2 = TrainingRunResult::completedResult("fold_2", {}, {}, {}, TrainingRunCompletionReason::COMPLETED, 1, 1, 1.0, fold2.string());
    result0.ensembleGroup = "digits";
    result0.savedModelNetworkName = "training_runs_save_ensemble_member_source_0";
    result1.ensembleGroup = "digits";
    result2.ensembleGroup = "digits";
    result2.savedModelNetworkName = "training_runs_save_ensemble_member_source_2";

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
    const std::string artifactPath = results.saveEnsemble("digits", ensembleDir.string());

    EXPECT_EQ(artifactPath, ensembleDir.string());
    EXPECT_FALSE(std::filesystem::exists(ensembleDir / "ensemble_manifest.json"));
    EXPECT_FALSE(std::filesystem::exists(ensembleDir / "members"));

    Network loadedEnsemble("ensemble_digits");
    EXPECT_NO_THROW(loadedEnsemble.load(ensembleDir.string()));

    std::filesystem::remove_all(root);
}


TEST(TrainingRunsResult, SaveSingleMemberEnsemblePreservesLogicalRaggedInputBoundary) {
    const std::filesystem::path root = uniqueTempPath("thor-training-runs-ragged-single-save");
    const std::filesystem::path memberDir = root / "member";
    const std::filesystem::path ensembleDir = root / "ensemble";
    std::filesystem::create_directories(memberDir);

    makeRaggedRowLengthMemberNetwork("training_runs_ragged_single_source")->save(memberDir.string(), true);

    TrainingRunResult result = TrainingRunResult::completedResult(
        "fold_0", {}, {}, {}, TrainingRunCompletionReason::COMPLETED, 1, 1, 1.0, memberDir.string());
    result.ensembleGroup = "ragged_single";
    result.savedModelNetworkName = "training_runs_ragged_single_source";

    TrainingEnsembleResult ensemble;
    ensemble.ensembleGroup = "ragged_single";
    ensemble.minSuccessfulModels = 1;
    ensemble.members = {
        TrainingEnsembleMemberResult{"fold_0", 1.0, TrainingRunStatus::COMPLETED},
    };
    ensemble.outputSignature = {TrainingRunOutputSignature{"predictions", {1}, "FP32"}};

    TrainingRunsResult results({result}, {ensemble});
    std::string savedEnsemblePath;
    EXPECT_NO_THROW(savedEnsemblePath = results.saveEnsemble("ragged_single", ensembleDir.string()));
    EXPECT_EQ(savedEnsemblePath, ensembleDir.string());

    Network loaded("ensemble_ragged_single");
    ASSERT_NO_THROW(loaded.load(ensembleDir.string()));
    EXPECT_EQ(loaded.getInferenceNetworkInputNames(), (std::vector<std::string>{"history"}));
    const std::vector<RaggedNetworkInputReference> raggedInputs = loaded.getExternalRaggedNetworkInputs();
    ASSERT_EQ(raggedInputs.size(), 1u);
    EXPECT_EQ(raggedInputs.front().name, "history");
    EXPECT_EQ(raggedInputs.front().raggedTensor.getBatchSize(), 4u);
    EXPECT_EQ(raggedInputs.front().raggedTensor.getMaxTotalValues(), 8u);
    ASSERT_TRUE(raggedInputs.front().raggedTensor.hasMaxValuesPerRow());
    EXPECT_EQ(raggedInputs.front().raggedTensor.getMaxValuesPerRow(), 3u);

    std::filesystem::remove_all(root);
}

TEST(TrainingRunsResult, SaveMultiMemberEnsemblePreservesLogicalRaggedInputBoundary) {
    const std::filesystem::path root = uniqueTempPath("thor-training-runs-ragged-multi-save");
    const std::filesystem::path member0Dir = root / "member_0";
    const std::filesystem::path member1Dir = root / "member_1";
    const std::filesystem::path ensembleDir = root / "ensemble";
    std::filesystem::create_directories(member0Dir);
    std::filesystem::create_directories(member1Dir);

    makeRaggedRowLengthMemberNetwork("training_runs_ragged_multi_source_0")->save(member0Dir.string(), true);
    makeRaggedRowLengthMemberNetwork("training_runs_ragged_multi_source_1")->save(member1Dir.string(), true);

    TrainingRunResult result0 = TrainingRunResult::completedResult(
        "fold_0", {}, {}, {}, TrainingRunCompletionReason::COMPLETED, 1, 1, 1.0, member0Dir.string());
    TrainingRunResult result1 = TrainingRunResult::completedResult(
        "fold_1", {}, {}, {}, TrainingRunCompletionReason::COMPLETED, 1, 1, 1.0, member1Dir.string());
    result0.ensembleGroup = "ragged_multi";
    result1.ensembleGroup = "ragged_multi";
    result0.savedModelNetworkName = "training_runs_ragged_multi_source_0";
    result1.savedModelNetworkName = "training_runs_ragged_multi_source_1";

    TrainingEnsembleResult ensemble;
    ensemble.ensembleGroup = "ragged_multi";
    ensemble.minSuccessfulModels = 2;
    ensemble.members = {
        TrainingEnsembleMemberResult{"fold_0", 1.0, TrainingRunStatus::COMPLETED},
        TrainingEnsembleMemberResult{"fold_1", 1.0, TrainingRunStatus::COMPLETED},
    };
    ensemble.outputSignature = {TrainingRunOutputSignature{"predictions", {1}, "FP32"}};

    TrainingRunsResult results({result0, result1}, {ensemble});
    std::string savedEnsemblePath;
    EXPECT_NO_THROW(savedEnsemblePath = results.saveEnsemble("ragged_multi", ensembleDir.string()));
    EXPECT_EQ(savedEnsemblePath, ensembleDir.string());

    Network loaded("ensemble_ragged_multi");
    ASSERT_NO_THROW(loaded.load(ensembleDir.string()));
    EXPECT_EQ(loaded.getInferenceNetworkInputNames(), (std::vector<std::string>{"history"}));
    const std::vector<RaggedNetworkInputReference> raggedInputs = loaded.getExternalRaggedNetworkInputs();
    ASSERT_EQ(raggedInputs.size(), 1u);
    EXPECT_EQ(raggedInputs.front().name, "history");
    EXPECT_EQ(raggedInputs.front().raggedTensor.getBatchSize(), 4u);
    EXPECT_EQ(raggedInputs.front().raggedTensor.getMaxTotalValues(), 8u);
    ASSERT_TRUE(raggedInputs.front().raggedTensor.hasMaxValuesPerRow());
    EXPECT_EQ(raggedInputs.front().raggedTensor.getMaxValuesPerRow(), 3u);

    std::filesystem::remove_all(root);
}

TEST(TrainingRuns, ExternalTestDataRequiresTestButNotTrainPartition) {
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{1.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(
        makeNetworkWithOutput("training-runs-test-data-partitions", {0, 10}), executor);
    TrainingRuns runs({{"fold_0", trainer}});

    TrainingRunsEvaluationOptions evaluation;
    evaluation.evaluateTrainingPopulation = false;
    evaluation.testData = makeFakeTestData(true);
    EXPECT_TRUE(runs.fit(TrainerFitOptions{.epochs = 1}, evaluation).allCompleted());
    EXPECT_EQ(executor->calls, 1u);
}

TEST(TrainingRuns, RejectsExternalTestDataWithEmptyTestPartitionBeforeFit) {
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{1.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(
        makeNetworkWithOutput("training-runs-empty-test-data", {0, 10}), executor);
    TrainingRuns runs({{"fold_0", trainer}});

    TrainingRunsEvaluationOptions evaluation;
    evaluation.evaluateTrainingPopulation = false;
    evaluation.testData = makeFakeTestData(false);
    EXPECT_THROW((void)runs.fit(TrainerFitOptions{.epochs = 1}, evaluation), std::runtime_error);
    EXPECT_EQ(executor->calls, 0u);
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
    EXPECT_EQ((*result)["fold_0"].finalTrainingStats->batchSize, 1u);
    EXPECT_EQ((*result)["fold_0"].finalTrainingStats->validExamplesInBatch, 1u);
    EXPECT_EQ((*result)["fold_0"].finalTrainingStats->samplesProcessedInEpoch, 1u);
    EXPECT_EQ(executor0->calls, 1u);
    EXPECT_EQ(executor1->calls, 1u);
}


TEST(TrainingRuns, InitialDeviceStartupTurnsFollowRunDeclarationOrder) {
    constexpr size_t numRuns = 5;
    auto recorder = std::make_shared<StartupOrderRecorder>();
    std::vector<TrainingRunsSpec> specs;
    specs.reserve(numRuns);

    for (size_t i = 0; i < numRuns; ++i) {
        // Later runs reach the sequencing hook first. The recorded acquisition
        // order must still follow the TrainingRuns declaration order.
        const auto delay =
            std::chrono::milliseconds((numRuns - i - 1) * 10);
        auto executor = std::make_shared<OrderedStartupExecutor>(
            i, delay, recorder);
        auto network = std::make_shared<Network>(
            "training-runs-startup-order-" + std::to_string(i));
        specs.emplace_back(
            "fold_" + std::to_string(i),
            makeTrainer(network, executor));
    }

    TrainingRuns runs(
        std::move(specs),
        TrainingRunsFailurePolicy::CONTINUE,
        2.0,
        numRuns);
    TrainingRunsEvaluationOptions evaluation;
    evaluation.evaluateTrainingPopulation = false;

    TrainingRunsResult result =
        runs.fit(TrainerFitOptions{.epochs = 1}, evaluation);

    EXPECT_TRUE(result.allCompleted());
    EXPECT_EQ(
        recorder->snapshot(),
        (std::vector<size_t>{0, 1, 2, 3, 4}));
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

TEST(TrainingRuns, FailedMemberReleasesPlacedNetworkBeforeNextSequentialRunStarts) {
    auto state = std::make_shared<FailedPlacementReleaseState>();
    auto failingExecutor = std::make_shared<PlaceThenFailExecutor>(state);
    auto nextExecutor = std::make_shared<RequirePriorFailedPlacementReleasedExecutor>(state);
    std::shared_ptr<Trainer> failingTrainer =
        makeTrainer(std::make_shared<Network>("training-runs-release-failed-placement-0"), failingExecutor);
    std::shared_ptr<Trainer> nextTrainer =
        makeTrainer(std::make_shared<Network>("training-runs-release-failed-placement-1"), nextExecutor);

    TrainingRuns runs({{"fold_0", failingTrainer}, {"fold_1", nextTrainer}},
                      TrainingRunsFailurePolicy::CONTINUE,
                      /*maxSummaryLogsPerSecond=*/0.0,
                      /*maxParallelRuns=*/1);
    TrainingRunsResult result = runs.fit(1);

    EXPECT_EQ(result["fold_0"].status, TrainingRunStatus::FAILED);
    EXPECT_EQ(result["fold_1"].status, TrainingRunStatus::COMPLETED);
    EXPECT_TRUE(nextExecutor->observedReleasedPlacement);
    EXPECT_TRUE(state->failedPlacementExpired());
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



TEST(Trainer, RestartedAttemptFinalStatsUseExactMetricAggregationWithoutPriorAttemptLeakage) {
    auto executor = std::make_shared<MetricAggregationLifecycleExecutor>(
        MetricAggregationLifecycleExecutor::Behavior::RESTART_ONCE_THEN_COMPLETE);
    std::shared_ptr<Trainer> trainer = makeTrainer(
        makeNetworkWithOutput("trainer-restart-exact-metrics", {0, 1}), executor);

    TrainerFitOptions options;
    options.epochs = 2;
    options.restartConditions = {
        TrainingRestartCondition{/*progressCheckEpochs=*/2,
                                 /*progressImprovementMinPercentage=*/5.0,
                                 /*maxRestarts=*/1}};

    TrainingRunResult result = trainer->fit(options);

    ASSERT_EQ(result.status, TrainingRunStatus::COMPLETED);
    ASSERT_EQ(executor->calls, 2u);
    ASSERT_TRUE(result.finalTrainingStats.has_value());
    const TrainingStatsSnapshot& stats = result.finalTrainingStats.value();
    EXPECT_NEAR(stats.metrics.at("mean"), 5.0, 1.0e-12);
    EXPECT_NEAR(stats.metrics.at("sum"), 30.0, 1.0e-12);
    EXPECT_NEAR(stats.metrics.at("min"), -3.0, 1.0e-12);
    EXPECT_NEAR(stats.metrics.at("max"), 11.0, 1.0e-12);
    EXPECT_NEAR(stats.metrics.at("ratio"), 100.0 / 11.0, 1.0e-12);
    const MetricBatchStat& ratio = stats.metricBatchStats.at("ratio");
    ASSERT_TRUE(ratio.numerator.has_value());
    ASSERT_TRUE(ratio.denominator.has_value());
    EXPECT_DOUBLE_EQ(ratio.numerator.value(), 100.0);
    EXPECT_DOUBLE_EQ(ratio.denominator.value(), 11.0);
    EXPECT_EQ(ratio.validExamples, 6u);
}

TEST(TrainingRuns, InterruptedRunRetainsExactMetricAggregationForCompletedBatches) {
    auto executor = std::make_shared<MetricAggregationLifecycleExecutor>(
        MetricAggregationLifecycleExecutor::Behavior::INTERRUPT_AFTER_EPOCH);
    std::shared_ptr<Trainer> trainer = makeTrainer(
        makeNetworkWithOutput("training-runs-interrupted-exact-metrics", {0, 1}), executor);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0);
    TrainingRunsEvaluationOptions evaluation;
    evaluation.evaluateTrainingPopulation = false;

    TrainingRunsResult result = runs.fit(TrainerFitOptions{.epochs = 1}, evaluation);

    ASSERT_EQ(result.size(), 1u);
    const TrainingRunResult& run = result[0];
    EXPECT_EQ(run.status, TrainingRunStatus::INTERRUPTED);
    ASSERT_TRUE(run.finalTrainingStats.has_value());
    const TrainingStatsSnapshot& stats = run.finalTrainingStats.value();
    EXPECT_NEAR(stats.metrics.at("mean"), 5.0, 1.0e-12);
    EXPECT_NEAR(stats.metrics.at("sum"), 30.0, 1.0e-12);
    EXPECT_NEAR(stats.metrics.at("min"), -3.0, 1.0e-12);
    EXPECT_NEAR(stats.metrics.at("max"), 11.0, 1.0e-12);
    EXPECT_NEAR(stats.metrics.at("ratio"), 100.0 / 11.0, 1.0e-12);
    const MetricBatchStat& ratio = stats.metricBatchStats.at("ratio");
    ASSERT_TRUE(ratio.numerator.has_value());
    ASSERT_TRUE(ratio.denominator.has_value());
    EXPECT_DOUBLE_EQ(ratio.numerator.value(), 100.0);
    EXPECT_DOUBLE_EQ(ratio.denominator.value(), 11.0);
    EXPECT_EQ(ratio.validExamples, 6u);
}

TEST(Trainer, RestartConditionUsesPhaseLocalEpochsAndKeepsCumulativeEpochBoundary) {
    auto network = std::make_shared<Network>("trainer-restart-phase-local-epoch");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0}, {100.0, 100.0}, {100.0, 90.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    trainer->fit(TrainerFitOptions{1});
    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 0u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 1u);

    TrainerFitOptions restartOptions;
    restartOptions.epochs = 2;
    restartOptions.restartConditions = {
        TrainingRestartCondition{/*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/1}};
    TrainingRunResult result = trainer->fit(restartOptions);
    EXPECT_EQ(result.status, TrainingRunStatus::COMPLETED);
    EXPECT_EQ(executor->calls, 3u);
    ASSERT_EQ(executor->initialCompletedEpochsByCall.size(), 3u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[0], 0u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[1], 1u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[2], 1u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 3u);
}


TEST(TrainingRuns, RestartPolicyUsesPhaseLocalEpochsAndKeepsCumulativeEpochBoundary) {
    auto network = std::make_shared<Network>("training-runs-restart-phase-local-epoch");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0}, {100.0, 100.0}, {100.0, 90.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRunsRestartPolicy condition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/1);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.restartConditions = {condition};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    TrainingRunsResult firstResult = runs.fit(TrainerFitOptions{1}, sessionOptions);
    EXPECT_TRUE(firstResult.allCompleted());
    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 0u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 1u);

    TrainingRunsResult secondResult = runs.fit(TrainerFitOptions{2}, sessionOptions);
    ASSERT_TRUE(secondResult.allCompleted());
    EXPECT_EQ(executor->calls, 3u);
    ASSERT_EQ(executor->initialCompletedEpochsByCall.size(), 3u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[0], 0u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[1], 1u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[2], 1u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 3u);
}

TEST(Trainer, LaterPhaseRestartAttemptsReuseSamePhaseInitialArtifact) {
    const std::filesystem::path saveDir = std::filesystem::temp_directory_path() / "thor_phase_initial_restart_test";
    std::filesystem::remove_all(saveDir);

    auto network = std::make_shared<Network>("trainer-restart-phase-initial-artifact");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 90.0}, {100.0, 100.0}, {100.0, 90.0}},
        /*writeLatestArtifact=*/true);
    std::shared_ptr<Trainer> trainer = makeTrainer(
        network, executor, saveDir.string(), /*saveModelOverwrite=*/true);

    trainer->fit(TrainerFitOptions{2});
    ASSERT_EQ(executor->calls, 1u);
    ASSERT_FALSE(executor->previousModelArtifactDirectoriesByCall[0].has_value());
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 2u);

    const std::filesystem::path phaseInitialArtifact = saveDir / "latest";
    ASSERT_TRUE(std::filesystem::exists(phaseInitialArtifact));

    TrainerFitOptions restartOptions;
    restartOptions.epochs = 2;
    restartOptions.restartConditions = {
        TrainingRestartCondition{/*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/1}};

    TrainingRunResult result = trainer->fit(restartOptions);
    EXPECT_EQ(result.status, TrainingRunStatus::COMPLETED);
    EXPECT_EQ(executor->calls, 3u);
    ASSERT_EQ(executor->initialCompletedEpochsByCall.size(), 3u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[0], 0u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[1], 2u);
    EXPECT_EQ(executor->initialCompletedEpochsByCall[2], 2u);

    ASSERT_EQ(executor->previousModelArtifactDirectoriesByCall.size(), 3u);
    ASSERT_FALSE(executor->previousModelArtifactDirectoriesByCall[0].has_value());
    ASSERT_TRUE(executor->previousModelArtifactDirectoriesByCall[1].has_value());
    ASSERT_TRUE(executor->previousModelArtifactDirectoriesByCall[2].has_value());
    EXPECT_EQ(executor->previousModelArtifactDirectoriesByCall[1].value(), phaseInitialArtifact.string());
    EXPECT_EQ(executor->previousModelArtifactDirectoriesByCall[2].value(), phaseInitialArtifact.string());
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 4u);

    std::filesystem::remove_all(saveDir);
}

TEST(TrainingRuns, RestartPolicyChecksCurrentPhaseEpochOnLaterFit) {
    auto network = std::make_shared<Network>("training-runs-restart-current-phase-epoch");
    auto executor = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}, {100.0, 100.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainingRunsRestartPolicy condition = TrainingRunsRestartPolicy::forRun(
        "fold_0", /*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/0);
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.restartConditions = {condition};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;

    TrainingRunsResult firstResult = runs.fit(TrainerFitOptions{2}, sessionOptions);
    EXPECT_TRUE(firstResult.allCompleted());
    EXPECT_EQ(executor->calls, 1u);
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 0u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 2u);

    TrainingRunsResult secondResult = runs.fit(TrainerFitOptions{2}, sessionOptions);
    ASSERT_EQ(executor->calls, 2u);
    EXPECT_FALSE(secondResult.allCompleted());
    ASSERT_EQ(secondResult.runs().size(), 1u);
    EXPECT_EQ(secondResult.runs()[0].status, TrainingRunStatus::FAILED);
    EXPECT_EQ(secondResult.runs()[0].exception.type, "TrainingRestartConditionExceeded");
    EXPECT_EQ(executor->lastInitialCompletedEpochs, 2u);
    EXPECT_EQ(trainer->getCompletedTrainingEpochs(), 2u);
}

TEST(Trainer, RestartConditionRestartsSingleTrainerUntilProgressImproves) {
    auto network = std::make_shared<Network>("trainer-restart-progress");
    auto executor = std::make_shared<RestartProgressExecutor>(
        std::vector<std::vector<double>>{{100.0, 98.0, 98.0}, {100.0, 90.0, 85.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);
    TrainerFitOptions restartOptions;
    restartOptions.epochs = 3;
    restartOptions.restartConditions = {
        TrainingRestartCondition{/*progressCheckEpochs=*/2, /*progressImprovementMinPercentage=*/5.0, /*maxRestarts=*/5}};

    trainer->fit(restartOptions);

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
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.restartConditions = {condition};

    TrainingRunsResult result = runs.fit(TrainerFitOptions{3}, sessionOptions);

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
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.restartConditions = {condition};

    TrainingRunsResult result = runs.fit(TrainerFitOptions{2}, sessionOptions);

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
                      0.0);

    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.restartConditions = {condition};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;
    TrainingRunsResult result = runs.fit(TrainerFitOptions{3}, sessionOptions);

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
                      0.0);

    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.restartConditions = {earlyCondition, laterCondition};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;
    TrainingRunsResult result = runs.fit(TrainerFitOptions{3}, sessionOptions);

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
    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0);
    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.restartConditions = {earlyCondition, laterCondition};

    TrainingRunsResult result = runs.fit(TrainerFitOptions{3}, sessionOptions);

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
                      0.0);

    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.earlyCompletionRules = {rule};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;
    TrainingRunsResult result = runs.fit(TrainerFitOptions{2, 1}, sessionOptions);

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

    std::shared_ptr<Trainer> trainer0 = makeTrainer(network0, executor0);
    std::shared_ptr<Trainer> trainer1 = makeTrainer(network1, executor1);

    TrainingRunsEarlyCompletionRule phaseRule = TrainingRunsEarlyCompletionRule::forRun(
        "fold_0", [](double, double, uint64_t, uint64_t) { return false; });
    TrainingRunsEarlyCompletionRule runRule = TrainingRunsEarlyCompletionRule::forRun(
        "fold_0", [](double, double, uint64_t, uint64_t) { return false; });
    TrainingRunsEarlyCompletionRule groupRule = TrainingRunsEarlyCompletionRule::forEnsembleGroup(
        "digits", [](double, double, uint64_t, uint64_t) { return false; });

    TrainingRuns runs({TrainingRunsSpec{"fold_0", trainer0, "digits"}, TrainingRunsSpec{"other", trainer1, "other_group"}},
                      TrainingRunsFailurePolicy::CONTINUE,
                      0.0);

    TrainingRunsSessionOptions sessionOptions;
    sessionOptions.earlyCompletionRules = {phaseRule, runRule, groupRule};
    sessionOptions.evaluation.evaluateTrainingPopulation = false;
    TrainingRunsResult result = runs.fit(TrainerFitOptions{2, 1}, sessionOptions);

    EXPECT_TRUE(result.allCompleted());
    EXPECT_EQ(executor0->lastEarlyCompletionPolicyCount, 3u);
    EXPECT_EQ(executor1->lastEarlyCompletionPolicyCount, 0u);
}

TEST(TrainingRuns, RejectsInvalidEarlyCompletionRules) {
    auto network = std::make_shared<Network>("training-runs-early-completion-invalid");
    auto executor = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0);

    TrainingRunsEarlyCompletionRule both([](double, double, uint64_t, uint64_t) { return false; });
    both.runName = "fold_0";
    both.ensembleGroup = "group";
    TrainingRunsSessionOptions bothOptions;
    bothOptions.earlyCompletionRules = {both};
    EXPECT_THROW(static_cast<void>(runs.fit(TrainerFitOptions{1, 1}, bothOptions)), std::runtime_error);

    TrainingRunsEarlyCompletionRule unknown = TrainingRunsEarlyCompletionRule::forRun(
        "missing", [](double, double, uint64_t, uint64_t) { return false; });
    TrainingRunsSessionOptions unknownOptions;
    unknownOptions.earlyCompletionRules = {unknown};
    EXPECT_THROW(static_cast<void>(runs.fit(TrainerFitOptions{1, 1}, unknownOptions)), std::runtime_error);

    TrainingRunsEarlyCompletionRule invalid;
    invalid.runName = "fold_0";
    TrainingRunsSessionOptions invalidOptions;
    invalidOptions.earlyCompletionRules = {invalid};
    EXPECT_THROW(static_cast<void>(runs.fit(TrainerFitOptions{1, 1}, invalidOptions)), std::runtime_error);
}

TEST(TrainingRuns, RejectsInvalidRestartConditions) {
    auto network = std::make_shared<Network>("training-runs-restart-invalid");
    auto executor = std::make_shared<RestartProgressExecutor>(std::vector<std::vector<double>>{{100.0, 90.0}});
    std::shared_ptr<Trainer> trainer = makeTrainer(network, executor);

    TrainingRuns runs({{"fold_0", trainer}}, TrainingRunsFailurePolicy::CONTINUE, 0.0);

    TrainingRunsRestartPolicy both;
    both.runName = "fold_0";
    both.ensembleGroup = "group";
    TrainingRunsSessionOptions bothOptions;
    bothOptions.restartConditions = {both};
    EXPECT_THROW(static_cast<void>(runs.fit(TrainerFitOptions{1}, bothOptions)), std::runtime_error);

    TrainingRunsRestartPolicy unknown = TrainingRunsRestartPolicy::forRun("missing");
    TrainingRunsSessionOptions unknownOptions;
    unknownOptions.restartConditions = {unknown};
    EXPECT_THROW(static_cast<void>(runs.fit(TrainerFitOptions{1}, unknownOptions)), std::runtime_error);

    TrainingRunsRestartPolicy invalidProgress = TrainingRunsRestartPolicy::forRun("fold_0");
    invalidProgress.progressCheckEpochs = 0;
    TrainingRunsSessionOptions invalidOptions;
    invalidOptions.restartConditions = {invalidProgress};
    EXPECT_THROW(static_cast<void>(runs.fit(TrainerFitOptions{1}, invalidOptions)), std::runtime_error);
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
