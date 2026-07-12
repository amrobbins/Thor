#include "DeepLearning/Implementation/Data/FileDatasetRuntimeAccess.h"
#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentNamedBatchSession.h"
#include "DeepLearning/Api/Data/FileDataset.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Implementation/Data/Sessions/DeviceResidentWindowedNamedBatchSession.h"
#include "DeepLearning/Implementation/Data/Sessions/IndexedNamedBatchSession.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetResidency.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetStorageSelection.h"
#include "DeepLearning/Implementation/Data/Residency/NamedDatasetRuntimeAccess.h"
#include "DeepLearning/Implementation/Training/DeviceStartupCoordinator.h"
#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"
#include "DeepLearning/Api/Data/DatasetWriter.h"
#include "DeepLearning/Implementation/Data/Materialization/NamedDatasetMaterializer.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using std::map;
using std::string;
using std::vector;
using Thor::BatchLease;
using Thor::BatchSession;
using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

static_assert(std::is_constructible_v<
              IndexedNamedBatchSession,
              std::shared_ptr<const Thor::FileDataset>,
              Thor::DatasetSplitManifest,
              Thor::BatchPolicy,
              uint64_t,
              std::set<Thor::DatasetFieldId>>);
static_assert(!std::is_constructible_v<
              IndexedNamedBatchSession,
              std::filesystem::path,
              DatasetLayout,
              std::vector<uint64_t>,
              std::vector<uint64_t>,
              std::optional<std::vector<uint64_t>>,
              uint64_t,
              uint64_t,
              bool,
              std::optional<uint64_t>>);
static_assert(!std::is_constructible_v<
              DeviceResidentNamedBatchSession,
              std::shared_ptr<DeviceResidentNamedDataset>,
              Thor::DatasetSplitManifest,
              Thor::BatchPolicy,
              uint64_t>);
static_assert(!std::is_constructible_v<
              DeviceResidentWindowedNamedBatchSession,
              Thor::DatasetMaterializationDescription,
              Thor::DeviceDatasetSessionDescription,
              std::shared_ptr<DeviceResidentNamedDataset>,
              uint64_t,
              uint64_t>);
static_assert(!std::is_constructible_v<
              Thor::TrainingData,
              std::shared_ptr<const Thor::NamedDataset>,
              Thor::DatasetSplitManifest,
              Thor::BatchPolicy,
              std::string>);

namespace {

nlohmann::json readJson(const std::filesystem::path &path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + path.string());
    }
    return nlohmann::json::parse(in);
}

std::filesystem::path makeTempDatasetPath(const std::string &name) {
    static uint64_t counter = 0;
    std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("thor_indexed_named_batch_session_" + name + "_" + std::to_string(counter++));
    std::filesystem::remove_all(path);
    return path;
}

DatasetLayout testLayout() {
    return DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{
            DatasetLayout::TensorShape("seasonality_inputs", {2}, DataType::FP32),
            DatasetLayout::TensorShape("monotone_inputs", {3}, DataType::FP32),
            DatasetLayout::TensorShape("daily_weight", {1}, DataType::FP32)});
}

DatasetLayout reorderedEquivalentLayout() {
    DatasetLayout layout = testLayout();
    return DatasetLayout(
        layout.recordSizeBytes(),
        vector<DatasetLayout::TensorSpec>{
            layout.tensor("daily_weight"), layout.tensor("seasonality_inputs"), layout.tensor("monotone_inputs")});
}

DatasetWriter::TensorView tensorView(vector<float> &values, vector<uint64_t> dimensions) {
    return DatasetWriter::TensorView{
        .dataType = DataType::FP32, .dimensions = std::move(dimensions), .data = values.data(), .numBytes = values.size() * sizeof(float)};
}

map<string, DatasetWriter::TensorView> exampleViews(vector<float> &seasonality,
                                                                     vector<float> &monotone,
                                                                     vector<float> &weight) {
    return {{"seasonality_inputs", tensorView(seasonality, {2})},
            {"monotone_inputs", tensorView(monotone, {3})},
            {"daily_weight", tensorView(weight, {1})}};
}

void writeIndexedExample(DatasetWriter &writer, float base) {
    vector<float> seasonality{base, base + 1.0f};
    vector<float> monotone{base + 10.0f, base + 11.0f, base + 12.0f};
    vector<float> weight{base + 100.0f};
    writer.writeIndexedExample(exampleViews(seasonality, monotone, weight));
}

void writeCanonicalDataset(const std::filesystem::path &datasetPath, const DatasetLayout &layout) {
    DatasetWriter writer(datasetPath, layout, 3);
    writeIndexedExample(writer, 0.0f);
    writeIndexedExample(writer, 20.0f);
    writeIndexedExample(writer, 40.0f);
    writeIndexedExample(writer, 60.0f);
    writeIndexedExample(writer, 80.0f);
    writer.close();
}


struct IndexedSessionArguments {
    std::shared_ptr<const Thor::FileDataset> dataset;
    Thor::DatasetSplitManifest splits;
    Thor::BatchPolicy batching;
};

IndexedSessionArguments indexedSessionArguments(
    std::shared_ptr<const Thor::FileDataset> dataset,
    std::vector<uint64_t> trainIndices,
    std::vector<uint64_t> validateIndices,
    std::optional<std::vector<uint64_t>> testIndices,
    uint64_t batchSize,
    bool randomizeTrain,
    std::optional<uint64_t> seed) {
    if (dataset == nullptr) {
        throw std::runtime_error("Test indexed session dataset must not be null.");
    }
    Thor::DatasetSplitManifest splits(
        *dataset,
        std::move(trainIndices),
        std::move(validateIndices),
        std::move(testIndices));
    return IndexedSessionArguments{
        std::move(dataset),
        std::move(splits),
        Thor::BatchPolicy(batchSize, randomizeTrain, seed)};
}

IndexedSessionArguments indexedSessionArguments(
    const std::filesystem::path &datasetPath,
    const DatasetLayout &expectedLayout,
    std::vector<uint64_t> trainIndices,
    std::vector<uint64_t> validateIndices,
    std::optional<std::vector<uint64_t>> testIndices,
    uint64_t batchSize,
    bool randomizeTrain,
    std::optional<uint64_t> seed) {
    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    ThorImplementation::FileDatasetRuntimeAccess::layout(*dataset)
        .validateRequestedLayoutExact(expectedLayout);
    return indexedSessionArguments(
        std::move(dataset),
        std::move(trainIndices),
        std::move(validateIndices),
        std::move(testIndices),
        batchSize,
        randomizeTrain,
        seed);
}

class TestIndexedNamedBatchSession : public IndexedNamedBatchSession {
   public:
    using IndexedNamedBatchSession::IndexedNamedBatchSession;

    TestIndexedNamedBatchSession(
        std::shared_ptr<const Thor::FileDataset> dataset,
        std::vector<uint64_t> trainIndices,
        std::vector<uint64_t> validateIndices,
        std::optional<std::vector<uint64_t>> testIndices,
        uint64_t batchSize,
        uint64_t batchQueueDepth = 32,
        bool randomizeTrain = true,
        std::optional<uint64_t> seed = std::nullopt)
        : TestIndexedNamedBatchSession(
              indexedSessionArguments(
                  std::move(dataset),
                  std::move(trainIndices),
                  std::move(validateIndices),
                  std::move(testIndices),
                  batchSize,
                  randomizeTrain,
                  seed),
              batchQueueDepth) {}

    TestIndexedNamedBatchSession(
        const std::filesystem::path &datasetPath,
        const DatasetLayout &expectedLayout,
        std::vector<uint64_t> trainIndices,
        std::vector<uint64_t> validateIndices,
        std::optional<std::vector<uint64_t>> testIndices,
        uint64_t batchSize,
        uint64_t batchQueueDepth = 32,
        bool randomizeTrain = true,
        std::optional<uint64_t> seed = std::nullopt)
        : TestIndexedNamedBatchSession(
              indexedSessionArguments(
                  datasetPath,
                  expectedLayout,
                  std::move(trainIndices),
                  std::move(validateIndices),
                  std::move(testIndices),
                  batchSize,
                  randomizeTrain,
                  seed),
              batchQueueDepth) {}

    BatchLease leaseBatch(ExampleType exampleType, uint64_t& batchNum) {
        attachSharedOwnershipForTest();
        return BatchSession::leaseBatch(exampleType, batchNum);
    }

   private:
    void attachSharedOwnershipForTest() {
        if (testSharedOwnership == nullptr) {
            testSharedOwnership = std::shared_ptr<BatchSession>(this, [](BatchSession*) {});
        }
    }

    TestIndexedNamedBatchSession(IndexedSessionArguments arguments, uint64_t batchQueueDepth)
        : IndexedNamedBatchSession(
              std::move(arguments.dataset),
              std::move(arguments.splits),
              std::move(arguments.batching),
              batchQueueDepth) {}

    std::shared_ptr<BatchSession> testSharedOwnership;
};

class TestDeviceResidentNamedBatchSession : public DeviceResidentNamedBatchSession {
   public:
    using DeviceResidentNamedBatchSession::DeviceResidentNamedBatchSession;

    TestDeviceResidentNamedBatchSession(
        std::shared_ptr<DeviceResidentNamedDataset> dataset,
        Thor::DatasetSplitManifest splits,
        Thor::BatchPolicy batching,
        uint64_t batchQueueDepth = 2)
        : DeviceResidentNamedBatchSession(
              Thor::DeviceDatasetLease(std::move(dataset)),
              std::move(splits),
              std::move(batching),
              batchQueueDepth) {}

    TestDeviceResidentNamedBatchSession(
        std::shared_ptr<DeviceResidentNamedDataset> dataset,
        Thor::DeviceDatasetSessionDescription session,
        uint64_t batchQueueDepth = 2)
        : DeviceResidentNamedBatchSession(
              Thor::DeviceDatasetLease(std::move(dataset)),
              std::move(session),
              batchQueueDepth) {}

    BatchLease leaseBatch(ExampleType exampleType, uint64_t& batchNum) {
        attachSharedOwnershipForTest();
        return BatchSession::leaseBatch(exampleType, batchNum);
    }

   private:
    void attachSharedOwnershipForTest() {
        if (testSharedOwnership == nullptr) {
            testSharedOwnership = std::shared_ptr<BatchSession>(this, [](BatchSession*) {});
        }
    }

    std::shared_ptr<BatchSession> testSharedOwnership;
};

class TestDeviceResidentWindowedNamedBatchSession : public DeviceResidentWindowedNamedBatchSession {
   public:
    using DeviceResidentWindowedNamedBatchSession::DeviceResidentWindowedNamedBatchSession;

    TestDeviceResidentWindowedNamedBatchSession(
        Thor::DatasetMaterializationDescription datasetDescription,
        Thor::DeviceDatasetSessionDescription sessionDescription,
        std::shared_ptr<DeviceResidentNamedDataset> windowedDataset,
        uint64_t batchQueueDepth = 2,
        uint64_t readerQueueDepth = 32)
        : DeviceResidentWindowedNamedBatchSession(
              std::move(datasetDescription),
              std::move(sessionDescription),
              Thor::DeviceDatasetLease(std::move(windowedDataset)),
              batchQueueDepth,
              readerQueueDepth) {}

    BatchLease leaseBatch(ExampleType exampleType, uint64_t& batchNum) {
        attachSharedOwnershipForTest();
        return BatchSession::leaseBatch(exampleType, batchNum);
    }

   private:
    void attachSharedOwnershipForTest() {
        if (testSharedOwnership == nullptr) {
            testSharedOwnership = std::shared_ptr<BatchSession>(this, [](BatchSession*) {});
        }
    }

    std::shared_ptr<BatchSession> testSharedOwnership;
};

class MetadataOnlyBatchSession final : public BatchSession {
   public:
    MetadataOnlyBatchSession(
        Thor::DatasetSplitManifest splits,
        Thor::BatchPolicy batching,
        std::set<Thor::DatasetFieldId> requiredFieldIds)
        : splits(std::move(splits)),
          requiredFieldIds(std::move(requiredFieldIds)) {
        batchSize = batching.getBatchSize();
    }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        const uint64_t examples = getNumExamples(exampleType);
        return examples == 0 ? 0 : (examples + batchSize - 1) / batchSize;
    }

    uint64_t getNumExamples(ExampleType exampleType) override {
        if (exampleType == ExampleType::TRAIN) {
            return splits.getTrain().size();
        }
        if (exampleType == ExampleType::VALIDATE) {
            return splits.getValidate().size();
        }
        if (exampleType == ExampleType::TEST) {
            return splits.getTest().size();
        }
        throw std::runtime_error("Unsupported ExampleType");
    }

    uint64_t getNextBatchNum(ExampleType) override { return 0; }

    const std::set<Thor::DatasetFieldId> &getRequiredDatasetFieldIds() const override {
        return requiredFieldIds;
    }

   private:
    Batch acquireBatch(ExampleType, uint64_t &) override {
        throw std::runtime_error(
            "MetadataOnlyBatchSession does not provide host batches.");
    }

    void recycleBatch(ExampleType, Batch &&) override {}

    Thor::DatasetSplitManifest splits;
    std::set<Thor::DatasetFieldId> requiredFieldIds;
};

class DenseMemoryDatasetForTest final : public Thor::NamedDataset {
   public:
    DenseMemoryDatasetForTest()
        : id(Thor::DatasetId::generate()),
          schema(std::vector<Thor::DatasetField>{
              Thor::DatasetField{
                  .id = 1,
                  .name = "features",
                  .dataType = DataType::FP32,
                  .dimensions = {2},
                  .kind = Thor::DatasetFieldKind::DENSE},
              Thor::DatasetField{
                  .id = 2,
                  .name = "labels",
                  .dataType = DataType::FP32,
                  .dimensions = {1},
                  .kind = Thor::DatasetFieldKind::DENSE}}),
          layout(DatasetLayout::fromTensorShapes(
              std::vector<DatasetLayout::TensorShape>{
                  DatasetLayout::TensorShape("features", {2}, DataType::FP32),
                  DatasetLayout::TensorShape("labels", {1}, DataType::FP32)})),
          features{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
          labels{10.0f, 20.0f, 30.0f} {}

    const Thor::DatasetId &getId() const override { return id; }
    uint64_t getNumExamples() const override { return 3; }
    const Thor::DatasetSchema &getSchema() const override { return schema; }
    const Thor::DatasetField &getField(std::string_view name) const override {
        return schema.getField(name);
    }

   private:
    std::shared_ptr<BatchSession> openBatchSession(
        const Thor::DatasetSplitManifest &splits,
        const Thor::BatchPolicy &batching,
        const Thor::DatasetAccessPolicy &,
        uint64_t,
        const std::set<Thor::DatasetFieldId> &requiredFieldIds) const override {
        return std::make_shared<MetadataOnlyBatchSession>(
            splits,
            batching,
            requiredFieldIds);
    }

    std::unique_ptr<Thor::DatasetMaterializationDescription>
    describeMaterializationForRuntime() const override {
        return std::make_unique<Thor::DatasetMaterializationDescription>(
            std::filesystem::path{},
            id,
            schema,
            layout,
            getNumExamples(),
            Thor::DatasetMaterializationSource::MEMORY);
    }

    MaterializedNamedDatasetSnapshot materializeSnapshotForRuntime(
        uint64_t readerQueueDepth) const override {
        if (readerQueueDepth == 0) {
            throw std::runtime_error("reader queue depth must be positive");
        }
        MaterializedNamedDatasetSnapshot snapshot(
            id,
            schema,
            layout,
            getNumExamples());
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor featureTensor(
            cpuPlacement,
            ThorImplementation::TensorDescriptor(DataType::FP32, {3, 2}));
        Tensor labelTensor(
            cpuPlacement,
            ThorImplementation::TensorDescriptor(DataType::FP32, {3, 1}));
        std::memcpy(
            featureTensor.getMemPtr<void>(),
            features.data(),
            features.size() * sizeof(float));
        std::memcpy(
            labelTensor.getMemPtr<void>(),
            labels.data(),
            labels.size() * sizeof(float));
        snapshot.fields.emplace(1, std::move(featureTensor));
        snapshot.fields.emplace(2, std::move(labelTensor));
        return snapshot;
    }

    Thor::DatasetId id;
    Thor::DatasetSchema schema;
    DatasetLayout layout;
    std::vector<float> features;
    std::vector<float> labels;
};

void expectTensorValues(const Tensor &tensor, const vector<float> &expected) {
    ASSERT_EQ(tensor.getDescriptor().getArraySizeInBytes(), expected.size() * sizeof(float));
    const float *actual = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "i=" << i;
    }
}

vector<float> tensorValues(const Tensor &tensor) {
    const uint64_t count = tensor.getDescriptor().getArraySizeInBytes() / sizeof(float);
    const float *actual = tensor.getMemPtr<float>();
    return vector<float>(actual, actual + count);
}

void requireCudaDevice(const char *reason) {
    int deviceCount = 0;
    const cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount <= 0) {
        GTEST_SKIP() << reason;
    }
}

std::set<Thor::DatasetFieldId> datasetFieldIds(const Thor::DatasetSchema &schema) {
    std::set<Thor::DatasetFieldId> ids;
    for (const Thor::DatasetField &field : schema.getFields()) {
        ids.insert(field.id);
    }
    return ids;
}

Thor::DatasetMaterializationDescription materializationDescription(
    const TestIndexedNamedBatchSession& session) {
    return Thor::describeDatasetMaterialization(*session.getDataset());
}

Thor::DeviceDatasetSessionDescription deviceSessionDescription(
    const TestIndexedNamedBatchSession& session) {
    return Thor::describeDeviceDatasetSession(
        session.getSplitManifest(),
        Thor::BatchPolicy(
            session.getBatchSize(),
            session.getRandomizeTrain(),
            session.getRandomSeed()),
        session.getRequiredDatasetFieldIds());
}

std::shared_ptr<Thor::TrainingData> trainingDataFor(
    const TestIndexedNamedBatchSession& session,
    Thor::DeviceDatasetStorage storage) {
    const std::string datasetName = session.getDatasetName().empty()
                                        ? "dataset"
                                        : session.getDatasetName();
    return std::make_shared<Thor::TrainingData>(
        session.getDataset(),
        session.getSplitManifest(),
        Thor::BatchPolicy(
            session.getBatchSize(),
            session.getRandomizeTrain(),
            session.getRandomSeed()),
        Thor::DatasetAccessPolicy{.deviceStorage = storage},
        datasetName);
}

Thor::DeviceDatasetStorageSelection selectDeviceStorage(
    const std::shared_ptr<TestIndexedNamedBatchSession>& sourceSession,
    Thor::DeviceDatasetStorage storage,
    TensorPlacement placement,
    uint64_t batchQueueDepth,
    std::optional<uint64_t> availableBytesOverride = std::nullopt) {
    std::shared_ptr<Thor::TrainingData> data =
        trainingDataFor(*sourceSession, storage);
    return Thor::selectDeviceDatasetStorageSession(
        sourceSession,
        *data,
        placement,
        batchQueueDepth,
        availableBytesOverride);
}

bool waitForCondition(const std::function<bool()> &condition,
                      std::chrono::milliseconds timeout = std::chrono::seconds(2)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return condition();
}

vector<float> tensorValuesOnHost(const Tensor &tensor) {
    if (tensor.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
        return tensorValues(tensor);
    }
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), tensor.getDescriptor());
    Stream stream(tensor.getPlacement());
    host.copyFromAsync(tensor, stream);
    stream.synchronize();
    return tensorValues(host);
}

void expectTensorValuesOnHost(const Tensor &tensor, const vector<float> &expected) {
    ASSERT_EQ(tensor.getDescriptor().getArraySizeInBytes(), expected.size() * sizeof(float));
    const vector<float> actual = tensorValuesOnHost(tensor);
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual.at(i), expected.at(i)) << "i=" << i;
    }
}

vector<uint8_t> uint8TensorValuesOnHost(const Tensor &tensor) {
    Tensor host = tensor;
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU) {
        host = Tensor(TensorPlacement(TensorPlacement::MemDevices::CPU), tensor.getDescriptor());
        Stream stream(tensor.getPlacement());
        host.copyFromAsync(tensor, stream);
        stream.synchronize();
    }
    const uint64_t count = host.getDescriptor().getArraySizeInBytes();
    const uint8_t *actual = host.getMemPtr<uint8_t>();
    return vector<uint8_t>(actual, actual + count);
}

void expectUint8TensorValuesOnHost(const Tensor &tensor, const vector<uint8_t> &expected) {
    ASSERT_EQ(tensor.getDescriptor().getArraySizeInBytes(), expected.size());
    const vector<uint8_t> actual = uint8TensorValuesOnHost(tensor);
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual.at(i), expected.at(i)) << "i=" << i;
    }
}

vector<float> seasonalityValues(const Batch &batch) { return tensorValues(batch.getTensor("seasonality_inputs")); }

struct IndexedBackendCase {
    const char *envValue;
    const char *displayName;
};

class ScopedEnvVar {
   public:
    ScopedEnvVar(std::string name, std::string value) : name_(std::move(name)) {
        const char *previous = std::getenv(name_.c_str());
        if (previous != nullptr) {
            previous_ = std::string(previous);
        }
        setenv(name_.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (previous_.has_value()) {
            setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnvVar(const ScopedEnvVar &) = delete;
    ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

   private:
    std::string name_;
    std::optional<std::string> previous_;
};

class ScopedUnsetEnvVar {
   public:
    explicit ScopedUnsetEnvVar(std::string name) : name_(std::move(name)) {
        const char *previous = std::getenv(name_.c_str());
        if (previous != nullptr) {
            previous_ = std::string(previous);
        }
        unsetenv(name_.c_str());
    }

    ~ScopedUnsetEnvVar() {
        if (previous_.has_value()) {
            setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedUnsetEnvVar(const ScopedUnsetEnvVar &) = delete;
    ScopedUnsetEnvVar &operator=(const ScopedUnsetEnvVar &) = delete;

   private:
    std::string name_;
    std::optional<std::string> previous_;
};

bool isReadvBackendName(const std::string &name) { return name.find("readv") != std::string::npos; }

bool isUnavailableExplicitBackend(const IndexedBackendCase &backend, const std::string &message) {
    const std::string envValue(backend.envValue);
    if (envValue == "uring_direct") {
        return message.find("io_uring_queue_init") != std::string::npos || message.find("io_uring_register_files") != std::string::npos ||
               message.find("io_uring_register_buffers") != std::string::npos ||
               message.find("uring_direct is unavailable") != std::string::npos ||
               message.find("explicit uring_direct") != std::string::npos;
    }
    if (envValue == "pread_direct") {
        return message.find("open(O_DIRECT") != std::string::npos || message.find("pread_direct") != std::string::npos;
    }
    return false;
}

std::vector<uint64_t> permutedIndices(uint64_t count, uint64_t stride) {
    if (count == 0 || stride == 0) {
        throw std::runtime_error("permutedIndices requires positive count and stride.");
    }
    std::vector<uint64_t> indices;
    indices.reserve(count);
    uint64_t value = 0;
    for (uint64_t i = 0; i < count; ++i) {
        indices.push_back(value);
        value = (value + stride) % count;
    }
    return indices;
}

float deterministicLargeValue(uint64_t exampleIndex, uint64_t elementIndex) {
    return static_cast<float>((exampleIndex * 1315423911ull + elementIndex * 2654435761ull) % 1000003ull) / 1000.0f;
}

DatasetLayout largeRecordLayout(uint64_t fp32Elements) {
    return DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("large_features", {fp32Elements}, DataType::FP32)});
}

DatasetLayout materializerWindowedLayout() {
    return DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("dense", {2}, DataType::FP32)},
        vector<DatasetLayout::WindowedTensorSourceShape>{DatasetLayout::WindowedTensorSourceShape(
            "history_source", {1}, DataType::FP32, DataType::UINT64)},
        vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
            "history", {3, 1}, "history_source", DataType::INT32, 0.0, string("history_mask"))});
}

void writeWindowedMaterializerDataset(const std::filesystem::path &datasetPath) {
    DatasetLayout layout = materializerWindowedLayout();
    DatasetWriter writer(datasetPath, layout, 10);

    uint64_t key = 7;
    vector<float> source{10.0f, 11.0f, 12.0f, 13.0f};
    writer.writeWindowSource(
        "history_source",
        DatasetWriter::WindowedTensorSourceView{.dataType = DataType::FP32,
                                                                .key = &key,
                                                                .startIndex = 10,
                                                                .dimensions = vector<uint64_t>{4, 1},
                                                                .data = source.data(),
                                                                .numBytes = source.size() * sizeof(float)});

    vector<float> dense{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vector<uint64_t> keys{7, 7, 7};
    vector<int32_t> starts{10, 8, 12};
    writer.writeIndexedExamples(
        {{"dense",
          DatasetWriter::TensorBatchView{.dataType = DataType::FP32,
                                                          .dimensions = vector<uint64_t>{3, 2},
                                                          .data = dense.data(),
                                                          .numBytes = dense.size() * sizeof(float)}}},
        {{"history",
          DatasetWriter::WindowedTensorReferenceBatchView{.keyDataType = DataType::UINT64,
                                                                          .indexDataType = DataType::INT32,
                                                                          .keys = keys.data(),
                                                                          .starts = starts.data(),
                                                                          .count = 3}}});
    writer.close();
}


DatasetLayout sharedWindowSourceLayout() {
    return DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{},
        vector<DatasetLayout::WindowedTensorSourceShape>{DatasetLayout::WindowedTensorSourceShape(
            "tokens", {}, DataType::UINT8, DataType::UINT64)},
        vector<DatasetLayout::WindowedTensorShape>{
            DatasetLayout::WindowedTensorShape("examples", {4}, "tokens", DataType::INT64),
            DatasetLayout::WindowedTensorShape("labels", {4}, "tokens", DataType::INT64)});
}

void writeSharedWindowSourceDataset(const std::filesystem::path &datasetPath) {
    DatasetLayout layout = sharedWindowSourceLayout();
    DatasetWriter writer(datasetPath, layout, 2);

    uint64_t sourceKey = 19;
    vector<uint8_t> tokens{10, 11, 12, 13, 14, 15, 16};
    writer.writeWindowSource(
        "tokens",
        DatasetWriter::WindowedTensorSourceView{.dataType = DataType::UINT8,
                                                 .key = &sourceKey,
                                                 .startIndex = 0,
                                                 .dimensions = vector<uint64_t>{tokens.size()},
                                                 .data = tokens.data(),
                                                 .numBytes = tokens.size()});

    vector<uint64_t> keys{sourceKey, sourceKey};
    vector<int64_t> exampleStarts{0, 2};
    vector<int64_t> labelStarts{1, 3};
    writer.writeIndexedExamples(
        {},
        {{"examples",
          DatasetWriter::WindowedTensorReferenceBatchView{.keyDataType = DataType::UINT64,
                                                           .indexDataType = DataType::INT64,
                                                           .keys = keys.data(),
                                                           .starts = exampleStarts.data(),
                                                           .count = 2}},
         {"labels",
          DatasetWriter::WindowedTensorReferenceBatchView{.keyDataType = DataType::UINT64,
                                                           .indexDataType = DataType::INT64,
                                                           .keys = keys.data(),
                                                           .starts = labelStarts.data(),
                                                           .count = 2}}});
    writer.close();
}

DatasetLayout affineSharedWindowSourceLayout() {
    return DatasetLayout::fromTensorShapes(
        {},
        {DatasetLayout::WindowedTensorSourceShape("tokens", {}, DataType::UINT8, DataType::UINT64)},
        {DatasetLayout::WindowedTensorShape("examples",
                                            {4},
                                            "tokens",
                                            DataType::INT64,
                                            0.0,
                                            std::nullopt,
                                            DatasetLayout::WindowedTensorReferenceMode::AFFINE),
         DatasetLayout::WindowedTensorShape("labels",
                                            {4},
                                            "tokens",
                                            DataType::INT64,
                                            0.0,
                                            std::nullopt,
                                            DatasetLayout::WindowedTensorReferenceMode::AFFINE)});
}

void writeAffineSharedWindowSourceDataset(const std::filesystem::path &datasetPath) {
    DatasetLayout layout = affineSharedWindowSourceLayout();
    DatasetWriter writer(datasetPath, layout, 2, 2, true);

    uint64_t sourceKey = 19;
    vector<uint8_t> tokens{10, 11, 12, 13, 14, 15, 16};
    writer.writeWindowSource(
        "tokens",
        DatasetWriter::WindowedTensorSourceView{.dataType = DataType::UINT8,
                                                 .key = &sourceKey,
                                                 .startIndex = 0,
                                                 .dimensions = vector<uint64_t>{tokens.size()},
                                                 .data = tokens.data(),
                                                 .numBytes = tokens.size()});
    writer.writeAffineExamples(
        1,
        {},
        {{"examples",
          DatasetWriter::AffineWindowedTensorReferenceView{.keyDataType = DataType::UINT64,
                                                            .key = &sourceKey,
                                                            .base = 0,
                                                            .stride = 2,
                                                            .fieldOffset = 0}},
         {"labels",
          DatasetWriter::AffineWindowedTensorReferenceView{.keyDataType = DataType::UINT64,
                                                            .key = &sourceKey,
                                                            .base = 0,
                                                            .stride = 2,
                                                            .fieldOffset = 1}}});
    writer.writeAffineExamples(
        1,
        {},
        {{"examples",
          DatasetWriter::AffineWindowedTensorReferenceView{.keyDataType = DataType::UINT64,
                                                            .key = &sourceKey,
                                                            .base = 2,
                                                            .stride = 2,
                                                            .fieldOffset = 0}},
         {"labels",
          DatasetWriter::AffineWindowedTensorReferenceView{.keyDataType = DataType::UINT64,
                                                            .key = &sourceKey,
                                                            .base = 2,
                                                            .stride = 2,
                                                            .fieldOffset = 1}}});
    writer.close();
}

DatasetLayout denseAffineWindowLayout() {
    return DatasetLayout::fromTensorShapes(
        {DatasetLayout::TensorShape("dense", {1}, DataType::FP32)},
        {DatasetLayout::WindowedTensorSourceShape("tokens", {}, DataType::UINT8, DataType::UINT64)},
        {DatasetLayout::WindowedTensorShape("history",
                                            {3},
                                            "tokens",
                                            DataType::INT64,
                                            0.0,
                                            std::nullopt,
                                            DatasetLayout::WindowedTensorReferenceMode::AFFINE)});
}

void writeDenseAffineWindowDataset(const std::filesystem::path &datasetPath) {
    DatasetLayout layout = denseAffineWindowLayout();
    DatasetWriter writer(datasetPath, layout, 2, 2, true);

    uint64_t sourceKey = 23;
    vector<uint8_t> tokens{20, 21, 22, 23, 24};
    writer.writeWindowSource(
        "tokens",
        DatasetWriter::WindowedTensorSourceView{.dataType = DataType::UINT8,
                                                 .key = &sourceKey,
                                                 .startIndex = 0,
                                                 .dimensions = vector<uint64_t>{tokens.size()},
                                                 .data = tokens.data(),
                                                 .numBytes = tokens.size()});

    vector<float> dense{1.5f, 2.5f};
    writer.writeAffineExamples(
        2,
        {{"dense",
          DatasetWriter::TensorBatchView{.dataType = DataType::FP32,
                                         .dimensions = {2, 1},
                                         .data = dense.data(),
                                         .numBytes = dense.size() * sizeof(float)}}},
        {{"history",
          DatasetWriter::AffineWindowedTensorReferenceView{.keyDataType = DataType::UINT64,
                                                            .key = &sourceKey,
                                                            .base = 0,
                                                            .stride = 2,
                                                            .fieldOffset = 0}}});
    writer.close();
}


void writeLargeIndexedDataset(const std::filesystem::path &datasetPath,
                              const DatasetLayout &layout,
                              uint64_t numExamples,
                              uint64_t elementsPerExample,
                              uint64_t examplesPerShard) {
    DatasetWriter writer(datasetPath, layout, examplesPerShard);
    std::vector<float> values(elementsPerExample);
    for (uint64_t exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
        for (uint64_t elementIndex = 0; elementIndex < elementsPerExample; ++elementIndex) {
            values.at(elementIndex) = deterministicLargeValue(exampleIndex, elementIndex);
        }
        writer.writeIndexedExample({{"large_features", tensorView(values, {elementsPerExample})}});
    }
    writer.close();
}

bool waitForReadyBatches(TestIndexedNamedBatchSession &loader, ExampleType exampleType, uint64_t minReady);

void exerciseIndexedLoaderWithBackendOrSkip(const IndexedBackendCase &backend) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", backend.envValue);

    const std::filesystem::path datasetPath = makeTempDatasetPath(std::string("backend_") + backend.displayName);
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    try {
        TestIndexedNamedBatchSession loader(
            datasetPath, layout, {4, 2, 0, 3, 1}, {1, 3}, vector<uint64_t>{0, 4}, 2, 3, false, std::nullopt);

        if (!waitForReadyBatches(loader, ExampleType::TRAIN, 2)) {
            throw std::runtime_error("TestIndexedNamedBatchSession did not prefill ready train batches for backend " +
                                     std::string(backend.envValue) + ".");
        }
        IndexedLocalNamedBatchAssemblerStats stats = loader.getStatsSnapshot(ExampleType::TRAIN);
        EXPECT_TRUE(isReadvBackendName(stats.resolvedIoBackend)) << stats.resolvedIoBackend;
        EXPECT_GE(stats.recordsRequested, 4);
        EXPECT_GE(stats.readBytesSubmitted, 4 * layout.recordSizeBytes());
        EXPECT_DOUBLE_EQ(stats.readAmplification(), 1.0);
        EXPECT_EQ(stats.recordsCopied, 0);
        EXPECT_EQ(stats.recordCopyThreadCount, 0);

        uint64_t batchNum = 99;
        BatchLease batchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

        const Batch& batch = batchLease.get();
        EXPECT_EQ(batchNum, 0);
        expectTensorValues(batch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
        batchLease.reset();

        BatchLease validateBatchLease = loader.leaseBatch(ExampleType::VALIDATE, batchNum);


        const Batch& validateBatch = validateBatchLease.get();
        EXPECT_EQ(batchNum, 0);
        expectTensorValues(validateBatch.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});
        validateBatchLease.reset();
    } catch (const std::exception &e) {
        std::filesystem::remove_all(datasetPath);
        if (isUnavailableExplicitBackend(backend, e.what())) {
            GTEST_SKIP() << "Explicit indexed named batch session backend " << backend.envValue
                         << " is unavailable in this runtime: " << e.what();
        }
        throw;
    }

    std::filesystem::remove_all(datasetPath);
}

bool waitForReadyBatches(TestIndexedNamedBatchSession &loader, ExampleType exampleType, uint64_t minReady) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (loader.getReadyBatchCountForTesting(exampleType) >= minReady) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return loader.getReadyBatchCountForTesting(exampleType) >= minReady;
}

}  // namespace

TEST(IndexedLocalNamedExampleReaderTest, ExposesLayoutOrdinalsAndUsesAsyncReadvIntoOrdinalDestinations) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("reader_ordinals_readv");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    std::shared_ptr<IndexedLocalNamedExampleReader> reader = IndexedLocalNamedExampleReader::openDataset(datasetPath, layout);
    EXPECT_EQ(reader->getTensorCount(), 3);
    EXPECT_EQ(reader->getLayoutTensorOrdinal("seasonality_inputs"), 0);
    EXPECT_EQ(reader->getLayoutTensorOrdinal("monotone_inputs"), 1);
    EXPECT_EQ(reader->getLayoutTensorOrdinal("daily_weight"), 2);
    EXPECT_THROW(
        {
            const uint64_t missingOrdinal = reader->getLayoutTensorOrdinal("missing_tensor");
            (void)missingOrdinal;
        },
        std::runtime_error);

    std::vector<float> seasonality(4, -1.0f);
    std::vector<float> monotone(6, -2.0f);
    std::vector<float> weight(2, -3.0f);

    std::vector<uint8_t *> destinations(reader->getTensorCount(), nullptr);
    destinations.at(reader->getLayoutTensorOrdinal("daily_weight")) = reinterpret_cast<uint8_t *>(weight.data());
    destinations.at(reader->getLayoutTensorOrdinal("seasonality_inputs")) = reinterpret_cast<uint8_t *>(seasonality.data());
    destinations.at(reader->getLayoutTensorOrdinal("monotone_inputs")) = reinterpret_cast<uint8_t *>(monotone.data());

    std::unique_ptr<IndexedLocalNamedExampleReader::Session> session = reader->createSession(4);
    session->loadExampleInto(4, 1, destinations);
    session->drain();

    EXPECT_FLOAT_EQ(seasonality.at(0), -1.0f);
    EXPECT_FLOAT_EQ(seasonality.at(1), -1.0f);
    EXPECT_FLOAT_EQ(seasonality.at(2), 80.0f);
    EXPECT_FLOAT_EQ(seasonality.at(3), 81.0f);
    EXPECT_FLOAT_EQ(monotone.at(0), -2.0f);
    EXPECT_FLOAT_EQ(monotone.at(1), -2.0f);
    EXPECT_FLOAT_EQ(monotone.at(2), -2.0f);
    EXPECT_FLOAT_EQ(monotone.at(3), 90.0f);
    EXPECT_FLOAT_EQ(monotone.at(4), 91.0f);
    EXPECT_FLOAT_EQ(monotone.at(5), 92.0f);
    EXPECT_FLOAT_EQ(weight.at(0), -3.0f);
    EXPECT_FLOAT_EQ(weight.at(1), 180.0f);

    IndexedLocalNamedExampleReaderSessionStats stats = session->takeStats();
    EXPECT_EQ(stats.readCallsSubmitted, 1);
    EXPECT_EQ(stats.readBytesSubmitted, layout.recordSizeBytes());
    EXPECT_EQ(stats.readCallsCompleted, 1);
    EXPECT_EQ(stats.readBytesCompleted, layout.recordSizeBytes());
    ASSERT_EQ(stats.resolvedIoBackends.size(), 1);
    EXPECT_TRUE(isReadvBackendName(stats.resolvedIoBackends.front())) << stats.resolvedIoBackends.front();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedExampleReaderTest, MaterializesPureAffineWindowsWithoutIndexedRecords) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("reader_affine_windows");
    DatasetLayout layout = affineSharedWindowSourceLayout();
    writeAffineSharedWindowSourceDataset(datasetPath);

    EXPECT_EQ(layout.recordSizeBytes(), 0u);
    std::shared_ptr<IndexedLocalNamedExampleReader> reader =
        IndexedLocalNamedExampleReader::openDataset(datasetPath, layout);
    EXPECT_EQ(reader->getNumExamples(), 2u);
    EXPECT_EQ(reader->getRecordSizeBytes(), 0u);

    vector<uint8_t> examples(8, 0);
    vector<uint8_t> labels(8, 0);
    vector<uint8_t *> windowDestinations(reader->getWindowedTensorCount(), nullptr);
    vector<uint8_t *> maskDestinations(reader->getWindowedTensorCount(), nullptr);
    windowDestinations.at(reader->getLayoutWindowedTensorOrdinal("examples")) = examples.data();
    windowDestinations.at(reader->getLayoutWindowedTensorOrdinal("labels")) = labels.data();

    std::unique_ptr<IndexedLocalNamedExampleReader::Session> session = reader->createSession(2);
    session->loadExampleInto(0, 0, {}, windowDestinations, maskDestinations);
    session->loadExampleInto(1, 1, {}, windowDestinations, maskDestinations);
    session->drain();

    EXPECT_EQ(examples, (vector<uint8_t>{10, 11, 12, 13, 12, 13, 14, 15}));
    EXPECT_EQ(labels, (vector<uint8_t>{11, 12, 13, 14, 13, 14, 15, 16}));
    IndexedLocalNamedExampleReaderSessionStats stats = session->takeStats();
    EXPECT_EQ(stats.readCallsSubmitted, 0u);
    EXPECT_EQ(stats.readBytesSubmitted, 0u);
    EXPECT_EQ(stats.windowedSourceReadCalls, 4u);
    EXPECT_EQ(stats.windowedSourceReadBytes, 16u);

    const nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_TRUE(manifest.at("shards").empty());
    ASSERT_EQ(manifest.at("affine_window_reference_segments").size(), 1u);
    EXPECT_EQ(manifest.at("affine_window_reference_segments").at(0).at("count").get<uint64_t>(), 2u);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedExampleReaderTest, ReadsDenseRecordsAndAffineWindowsTogether) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("reader_dense_affine_windows");
    DatasetLayout layout = denseAffineWindowLayout();
    writeDenseAffineWindowDataset(datasetPath);

    std::shared_ptr<IndexedLocalNamedExampleReader> reader =
        IndexedLocalNamedExampleReader::openDataset(datasetPath, layout);
    vector<float> dense(2, 0.0f);
    vector<uint8_t> history(6, 0);
    vector<uint8_t *> denseDestinations(reader->getTensorCount(), nullptr);
    denseDestinations.at(reader->getLayoutTensorOrdinal("dense")) =
        reinterpret_cast<uint8_t *>(dense.data());
    vector<uint8_t *> windowDestinations(reader->getWindowedTensorCount(), nullptr);
    vector<uint8_t *> maskDestinations(reader->getWindowedTensorCount(), nullptr);
    windowDestinations.at(reader->getLayoutWindowedTensorOrdinal("history")) = history.data();

    std::unique_ptr<IndexedLocalNamedExampleReader::Session> session = reader->createSession(2);
    session->loadExampleInto(0, 0, denseDestinations, windowDestinations, maskDestinations);
    session->loadExampleInto(1, 1, denseDestinations, windowDestinations, maskDestinations);
    session->drain();

    EXPECT_EQ(dense, (vector<float>{1.5f, 2.5f}));
    EXPECT_EQ(history, (vector<uint8_t>{20, 21, 22, 22, 23, 24}));
    IndexedLocalNamedExampleReaderSessionStats stats = session->takeStats();
    EXPECT_EQ(stats.readCallsSubmitted, 2u);
    EXPECT_EQ(stats.readBytesSubmitted, 2u * layout.recordSizeBytes());
    EXPECT_EQ(stats.windowedSourceReadCalls, 2u);
    EXPECT_EQ(stats.windowedSourceReadBytes, 6u);

    const nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_FALSE(manifest.at("shards").empty());
    EXPECT_EQ(manifest.at("affine_window_reference_segments").size(), 1u);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedExampleReaderTest, RejectsAffineMetadataWhoseFirstRowOverflows) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("reader_affine_first_row_overflow");
    DatasetLayout layout = affineSharedWindowSourceLayout();
    writeAffineSharedWindowSourceDataset(datasetPath);

    const std::filesystem::path manifestPath = datasetPath / DatasetWriter::MANIFEST_FILENAME;
    nlohmann::json manifest = readJson(manifestPath);
    nlohmann::json &reference = manifest.at("affine_window_reference_segments")
                                    .at(0)
                                    .at("references")
                                    .at("examples");
    reference["base"] = std::numeric_limits<int64_t>::min();
    reference["stride"] = 1;
    reference["field_offset"] = -1;
    std::ofstream out(manifestPath, std::ios::binary | std::ios::trunc);
    out << manifest.dump(2) << '\n';
    out.close();

    EXPECT_THROW(IndexedLocalNamedExampleReader::openDataset(datasetPath, layout), std::runtime_error);
    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedExampleReaderTest, AsyncReadvDrainsAndReusesIovecSlotsWhenQueueDepthIsSmall) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("reader_async_readv_queue_depth");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    std::shared_ptr<IndexedLocalNamedExampleReader> reader = IndexedLocalNamedExampleReader::openDataset(datasetPath, layout);

    std::vector<float> seasonality(10, -1.0f);
    std::vector<float> monotone(15, -2.0f);
    std::vector<float> weight(5, -3.0f);

    std::vector<uint8_t *> destinations(reader->getTensorCount(), nullptr);
    destinations.at(reader->getLayoutTensorOrdinal("seasonality_inputs")) = reinterpret_cast<uint8_t *>(seasonality.data());
    destinations.at(reader->getLayoutTensorOrdinal("monotone_inputs")) = reinterpret_cast<uint8_t *>(monotone.data());
    destinations.at(reader->getLayoutTensorOrdinal("daily_weight")) = reinterpret_cast<uint8_t *>(weight.data());

    std::unique_ptr<IndexedLocalNamedExampleReader::Session> session = reader->createSession(2);
    for (uint64_t slot = 0; slot < 5; ++slot) {
        session->loadExampleInto(slot, slot, destinations);
    }
    session->drain();

    EXPECT_EQ(seasonality, (std::vector<float>{0.0f, 1.0f, 20.0f, 21.0f, 40.0f, 41.0f, 60.0f, 61.0f, 80.0f, 81.0f}));
    EXPECT_EQ(
        monotone,
        (std::vector<float>{10.0f, 11.0f, 12.0f, 30.0f, 31.0f, 32.0f, 50.0f, 51.0f, 52.0f, 70.0f, 71.0f, 72.0f, 90.0f, 91.0f, 92.0f}));
    EXPECT_EQ(weight, (std::vector<float>{100.0f, 120.0f, 140.0f, 160.0f, 180.0f}));

    IndexedLocalNamedExampleReaderSessionStats stats = session->takeStats();
    EXPECT_EQ(stats.readCallsSubmitted, 5);
    EXPECT_EQ(stats.readBytesSubmitted, 5 * layout.recordSizeBytes());
    EXPECT_EQ(stats.readCallsCompleted, 5);
    EXPECT_EQ(stats.readBytesCompleted, 5 * layout.recordSizeBytes());
    ASSERT_EQ(stats.resolvedIoBackends.size(), 1);
    EXPECT_EQ(stats.resolvedIoBackends.front(), "pread_buffered_readv");

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, BindsBatchPointersByReaderOrdinalWhenRequestedLayoutOrderDiffers) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("requested_layout_reordered");
    DatasetLayout writerLayout = testLayout();
    writeCanonicalDataset(datasetPath, writerLayout);

    DatasetLayout requestedLayout = reorderedEquivalentLayout();
    TestIndexedNamedBatchSession loader(datasetPath, requestedLayout, {4, 2}, {1, 3}, std::nullopt, 2, 2, false, std::nullopt);

    uint64_t batchNum = 99;
    BatchLease trainBatchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    const Batch& trainBatch = trainBatchLease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(trainBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
    expectTensorValues(trainBatch.getTensor("monotone_inputs"), {90.0f, 91.0f, 92.0f, 50.0f, 51.0f, 52.0f});
    expectTensorValues(trainBatch.getTensor("daily_weight"), {180.0f, 140.0f});
    trainBatchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, ReadsFoldIndicesFromOneSharedDataset) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_read");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {4, 2, 0}, {1, 3}, std::nullopt, 2, 2, false, std::nullopt);
    EXPECT_EQ(loader.getNumDatasetExamples(), 5);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TRAIN), 3);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::TRAIN), 2);
    EXPECT_EQ(loader.getNumExamples(ExampleType::VALIDATE), 2);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TEST), 2);
    EXPECT_FALSE(loader.hasExplicitTestSplit());

    uint64_t batchNum = 99;
    BatchLease trainBatchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    const Batch& trainBatch = trainBatchLease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(trainBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
    expectTensorValues(trainBatch.getTensor("monotone_inputs"), {90.0f, 91.0f, 92.0f, 50.0f, 51.0f, 52.0f});
    expectTensorValues(trainBatch.getTensor("daily_weight"), {180.0f, 140.0f});
    trainBatchLease.reset();

    BatchLease validateBatchLease = loader.leaseBatch(ExampleType::VALIDATE, batchNum);


    const Batch& validateBatch = validateBatchLease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(validateBatch.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});
    validateBatchLease.reset();

    BatchLease testBatchLease = loader.leaseBatch(ExampleType::TEST, batchNum);


    const Batch& testBatch = testBatchLease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(testBatch.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});
    testBatchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, RangeBackedSplitDrivesBatchesWithoutMaterializingIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("range_backed_split_batches");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);
    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    Thor::DatasetSplitManifest splits(
        *dataset,
        Thor::ExampleIndexSet::strided(0, 3, 2),
        Thor::ExampleIndexSet::contiguous(1, 2));
    TestIndexedNamedBatchSession session(
        dataset,
        std::move(splits),
        Thor::BatchPolicy(2, false),
        1);

    EXPECT_TRUE(session.getSplitIndices(ExampleType::TRAIN).isRangeBacked());
    EXPECT_EQ(session.getNumExamples(ExampleType::TRAIN), 3u);
    uint64_t batchNum = 99;
    BatchLease firstLease = session.leaseBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0u);
    expectTensorValues(firstLease.get().getTensor("seasonality_inputs"),
                       {0.0f, 1.0f, 40.0f, 41.0f});
    firstLease.reset();

    BatchLease secondLease = session.leaseBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 1u);
    expectTensorValues(secondLease.get().getTensor("seasonality_inputs"),
                       {80.0f, 81.0f, 0.0f, 1.0f});
    secondLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, SupportsEmptyValidateAndTestIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("empty_validate_test");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1, 2}, {}, vector<uint64_t>{}, 2, 2, false, std::nullopt);
    EXPECT_TRUE(loader.hasExplicitTestSplit());
    EXPECT_EQ(loader.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TEST), 0);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::TEST), 0);

    uint64_t batchNum = 0;
    EXPECT_THROW(loader.leaseBatch(ExampleType::VALIDATE, batchNum), std::runtime_error);
    EXPECT_THROW(loader.leaseBatch(ExampleType::TEST, batchNum), std::runtime_error);

    BatchLease trainBatchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);


    const Batch& trainBatch = trainBatchLease.get();
    expectTensorValues(trainBatch.getTensor("seasonality_inputs"), {0.0f, 1.0f, 20.0f, 21.0f});
    trainBatchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, EmptyValidateWithoutExplicitTestYieldsEmptyTest) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("empty_validate_implicit_test");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1, 2}, {}, std::nullopt, 2, 2, false, std::nullopt);
    EXPECT_FALSE(loader.hasExplicitTestSplit());
    EXPECT_EQ(loader.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TEST), 0);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::TEST), 0);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, SupportsEmptyTrainForEvaluationOnlyRecipes) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("empty_train");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession session(
        datasetPath, layout, vector<uint64_t>{}, vector<uint64_t>{1, 3}, vector<uint64_t>{0, 4}, 2, 2, false, std::nullopt);
    EXPECT_EQ(session.getNumExamples(ExampleType::TRAIN), 0);
    EXPECT_EQ(session.getNumBatchesPerEpoch(ExampleType::TRAIN), 0);
    EXPECT_EQ(session.getNumExamples(ExampleType::VALIDATE), 2);
    EXPECT_EQ(session.getNumExamples(ExampleType::TEST), 2);

    uint64_t batchNum = 0;
    EXPECT_THROW(session.leaseBatch(ExampleType::TRAIN, batchNum), std::runtime_error);
    BatchLease testLease = session.leaseBatch(ExampleType::TEST, batchNum);
    expectTensorValues(testLease.get().getTensor("seasonality_inputs"), {0.0f, 1.0f, 80.0f, 81.0f});

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, SupportsExplicitTestIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("explicit_test");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1}, {2}, vector<uint64_t>{4, 3}, 2, 2, false, std::nullopt);
    EXPECT_TRUE(loader.hasExplicitTestSplit());
    EXPECT_EQ(loader.getNumExamples(ExampleType::TEST), 2);

    uint64_t batchNum = 0;
    BatchLease testBatchLease = loader.leaseBatch(ExampleType::TEST, batchNum);

    const Batch& testBatch = testBatchLease.get();
    expectTensorValues(testBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 60.0f, 61.0f});
    testBatchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, RejectsOutOfRangeIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("out_of_range");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    EXPECT_THROW((TestIndexedNamedBatchSession(datasetPath, layout, {0, 5}, {1}, std::nullopt, 2, 2, false, std::nullopt)),
                 std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, DeterministicRandomizedTrainOrderForFixedSeed) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("deterministic_randomized_train");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession first(datasetPath, layout, {0, 1, 2, 3, 4}, {0}, std::nullopt, 2, 2, true, 12345);
    TestIndexedNamedBatchSession second(datasetPath, layout, {0, 1, 2, 3, 4}, {0}, std::nullopt, 2, 2, true, 12345);

    uint64_t firstBatchNum = 0;
    uint64_t secondBatchNum = 0;
    for (uint64_t i = 0; i < 4; ++i) {
        BatchLease firstBatchLease = first.leaseBatch(ExampleType::TRAIN, firstBatchNum);

        const Batch& firstBatch = firstBatchLease.get();
        BatchLease secondBatchLease = second.leaseBatch(ExampleType::TRAIN, secondBatchNum);

        const Batch& secondBatch = secondBatchLease.get();
        EXPECT_EQ(firstBatchNum, secondBatchNum);
        EXPECT_EQ(seasonalityValues(firstBatch), seasonalityValues(secondBatch));
        firstBatchLease.reset();
        secondBatchLease.reset();
    }

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, ValidateAndTestSplitsAreSequentialAndWrap) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("sequential_validate_test");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {4, 3}, {1, 3, 4}, vector<uint64_t>{2, 0, 1}, 2, 2, true, 9876);

    uint64_t batchNum = 99;
    BatchLease validate0Lease = loader.leaseBatch(ExampleType::VALIDATE, batchNum);

    const Batch& validate0 = validate0Lease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(validate0.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});
    validate0Lease.reset();

    BatchLease validate1Lease = loader.leaseBatch(ExampleType::VALIDATE, batchNum);


    const Batch& validate1 = validate1Lease.get();
    EXPECT_EQ(batchNum, 1);
    expectTensorValues(validate1.getTensor("seasonality_inputs"), {80.0f, 81.0f, 20.0f, 21.0f});
    validate1Lease.reset();

    BatchLease test0Lease = loader.leaseBatch(ExampleType::TEST, batchNum);


    const Batch& test0 = test0Lease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(test0.getTensor("seasonality_inputs"), {40.0f, 41.0f, 0.0f, 1.0f});
    test0Lease.reset();

    BatchLease test1Lease = loader.leaseBatch(ExampleType::TEST, batchNum);


    const Batch& test1 = test1Lease.get();
    EXPECT_EQ(batchNum, 1);
    expectTensorValues(test1.getTensor("seasonality_inputs"), {20.0f, 21.0f, 40.0f, 41.0f});
    test1Lease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, RejectsRequestedLayoutMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("layout_mismatch");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    DatasetLayout wrongLayout = DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{
            DatasetLayout::TensorShape("seasonality_inputs", {2}, DataType::FP32),
            DatasetLayout::TensorShape("monotone_inputs", {4}, DataType::FP32),
            DatasetLayout::TensorShape("daily_weight", {1}, DataType::FP32)});

    EXPECT_THROW((TestIndexedNamedBatchSession(datasetPath, wrongLayout, {0, 1}, {2}, std::nullopt, 2, 2, false, std::nullopt)),
                 std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, FileDatasetOwnsManifestSchemaAndStableIdentity) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("immutable_dataset_schema");
    const DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    std::shared_ptr<Thor::FileDataset> reopened = Thor::FileDataset::open(datasetPath);

    EXPECT_EQ(dataset->getId(), reopened->getId());
    EXPECT_EQ(dataset->getNumExamples(), 5);
    EXPECT_EQ(dataset->getSchema(), reopened->getSchema());

    const std::vector<Thor::DatasetField> &fields = dataset->getSchema().getFields();
    ASSERT_EQ(fields.size(), 3);
    EXPECT_EQ(fields.at(0).id, 0);
    EXPECT_EQ(fields.at(0).name, "seasonality_inputs");
    EXPECT_EQ(fields.at(0).dataType, DataType::FP32);
    EXPECT_EQ(fields.at(0).dimensions, vector<uint64_t>({2}));
    EXPECT_EQ(fields.at(0).kind, Thor::DatasetFieldKind::DENSE);
    EXPECT_EQ(dataset->getField("monotone_inputs").dimensions, vector<uint64_t>({3}));
    EXPECT_EQ(dataset->getField("daily_weight").dimensions, vector<uint64_t>({1}));

    std::vector<Thor::DatasetField> incompatibleFields = fields;
    incompatibleFields.at(0).dimensions = {3};
    Thor::DatasetSchema incompatibleSchema(std::move(incompatibleFields));
    EXPECT_THROW(dataset->assertSchema(incompatibleSchema), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, TwoFoldSessionsCanShareOneDatasetWithDifferentIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("two_fold_shared_dataset");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    TestIndexedNamedBatchSession foldA(dataset, {0, 2, 4}, {1}, std::nullopt, 2, 2, false, std::nullopt);
    TestIndexedNamedBatchSession foldB(dataset, {1, 3}, {0, 4}, std::nullopt, 2, 2, false, std::nullopt);

    EXPECT_EQ(foldA.getDataset().get(), dataset.get());
    EXPECT_EQ(foldB.getDataset().get(), dataset.get());
    EXPECT_EQ(
        ThorImplementation::FileDatasetRuntimeAccess::reader(*foldA.getDataset()).get(),
        ThorImplementation::FileDatasetRuntimeAccess::reader(*foldB.getDataset()).get());

    uint64_t batchNumA = 0;
    uint64_t batchNumB = 0;
    BatchLease batchALease = foldA.leaseBatch(ExampleType::TRAIN, batchNumA);

    const Batch& batchA = batchALease.get();
    BatchLease batchBLease = foldB.leaseBatch(ExampleType::TRAIN, batchNumB);

    const Batch& batchB = batchBLease.get();

    EXPECT_EQ(batchNumA, 0);
    EXPECT_EQ(batchNumB, 0);
    expectTensorValues(batchA.getTensor("seasonality_inputs"), {0.0f, 1.0f, 40.0f, 41.0f});
    expectTensorValues(batchB.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});

    batchALease.reset();
    batchBLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, AcceptsDatasetSplitManifestAndSeparateBatchPolicy) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("split_manifest_constructor");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    Thor::DatasetSplitManifest splits(*dataset, {4, 2, 0}, {1, 3});
    Thor::BatchPolicy batching(2, false, std::nullopt);
    TestIndexedNamedBatchSession loader(dataset, splits, batching, 2);

    EXPECT_EQ(loader.getSplitManifest(), splits);
    EXPECT_FALSE(loader.hasExplicitTestSplit());
    EXPECT_EQ(loader.getSplitIndices(ExampleType::TEST), loader.getSplitIndices(ExampleType::VALIDATE));

    uint64_t batchNum = 99;
    BatchLease batchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    const Batch& batch = batchLease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(batch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
    batchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, RejectsSplitManifestForDifferentDataset) {
    const std::filesystem::path datasetPathA = makeTempDatasetPath("split_manifest_dataset_a");
    const std::filesystem::path datasetPathB = makeTempDatasetPath("split_manifest_dataset_b");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPathA, layout);
    writeCanonicalDataset(datasetPathB, layout);

    std::shared_ptr<Thor::FileDataset> datasetA = Thor::FileDataset::open(datasetPathA);
    std::shared_ptr<Thor::FileDataset> datasetB = Thor::FileDataset::open(datasetPathB);
    Thor::DatasetSplitManifest splits(*datasetA, {0, 1, 2}, {3, 4});

    EXPECT_THROW((TestIndexedNamedBatchSession(datasetB, splits, Thor::BatchPolicy(2, false), 2)), std::runtime_error);

    std::filesystem::remove_all(datasetPathA);
    std::filesystem::remove_all(datasetPathB);
}

TEST(IndexedNamedBatchSessionTest, PrefillsMultipleReadyBatchesAndRecyclesReturnedBuffers) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("prefill_recycle");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1, 2, 3, 4}, {0}, std::nullopt, 1, 3, false, std::nullopt);

    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 2));
    EXPECT_LE(loader.getReadyBatchCountForTesting(ExampleType::TRAIN), 3);
    EXPECT_EQ(loader.getNextBatchNum(ExampleType::TRAIN), 0);

    uint64_t batchNum = 99;
    BatchLease batchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    const Batch& batch = batchLease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(batch.getTensor("seasonality_inputs"), {0.0f, 1.0f});

    const uint64_t readyAfterGet = loader.getReadyBatchCountForTesting(ExampleType::TRAIN);
    EXPECT_LE(readyAfterGet, 2);

    batchLease.reset();
    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 2));

    BatchLease recycledLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);


    const Batch& recycled = recycledLease.get();
    EXPECT_EQ(batchNum, 1);
    expectTensorValues(recycled.getTensor("seasonality_inputs"), {20.0f, 21.0f});
    recycledLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, EmptyValidateAndTestSplitsHaveNoReadyBatches) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("empty_validate_test_no_ready");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1, 2}, {}, vector<uint64_t>{}, 2, 2, false, std::nullopt);

    EXPECT_EQ(loader.getReadyBatchCountForTesting(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getReadyBatchCountForTesting(ExampleType::TEST), 0);
    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 1));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchAssemblerStatsTest, ReadAmplificationUsesSubmittedLogicalBytes) {
    IndexedLocalNamedBatchAssemblerStats stats;
    stats.recordSizeBytes = 24;
    stats.recordsRequested = 3;
    stats.logicalRecordBytesRequested = 72;
    stats.readCallsSubmitted = 2;
    stats.readBytesSubmitted = 48;

    EXPECT_DOUBLE_EQ(stats.readAmplification(), 1.0);
}

TEST(IndexedNamedBatchSessionTest, DefaultShardReadQueueDepthCoversFullBatchForLargeRecords) {
    ScopedUnsetEnvVar unsetPrimary("THOR_INDEXED_LOCAL_NAMED_LOADER_SHARD_READ_QUEUE_DEPTH");
    ScopedUnsetEnvVar unsetAlias("THOR_INDEXED_LOCAL_NAMED_READER_SHARD_READ_QUEUE_DEPTH");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    constexpr uint64_t elementsPerExample = 64 * 1024;  // 256 KiB records, old byte-target depth would be 32.
    constexpr uint64_t numExamples = 33;
    constexpr uint64_t batchSize = 33;

    const std::filesystem::path datasetPath = makeTempDatasetPath("default_shard_read_depth_full_batch");
    DatasetLayout layout = largeRecordLayout(elementsPerExample);
    writeLargeIndexedDataset(datasetPath, layout, numExamples, elementsPerExample, numExamples);

    std::vector<uint64_t> trainIndices;
    trainIndices.reserve(numExamples);
    for (uint64_t i = 0; i < numExamples; ++i) {
        trainIndices.push_back(i);
    }

    TestIndexedNamedBatchSession loader(datasetPath, layout, trainIndices, {}, std::nullopt, batchSize, 1, false, std::nullopt);

    IndexedLocalNamedBatchAssemblerStats stats = loader.getStatsSnapshot(ExampleType::TRAIN);
    EXPECT_EQ(stats.shardReadQueueDepth, batchSize + 10);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, ShardReadQueueDepthCanBeOverriddenByEnvironment) {
    ScopedEnvVar scopedDepth("THOR_INDEXED_LOCAL_NAMED_LOADER_SHARD_READ_QUEUE_DEPTH", "7");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("env_shard_read_depth");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1, 2, 3}, {}, std::nullopt, 4, 1, false, std::nullopt);

    IndexedLocalNamedBatchAssemblerStats stats = loader.getStatsSnapshot(ExampleType::TRAIN);
    EXPECT_EQ(stats.shardReadQueueDepth, 7);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, StatsExposeReadAndBatchCounters) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("stats_counters");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1, 2, 3, 4}, {}, vector<uint64_t>{}, 1, 3, false, std::nullopt);

    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 2));

    IndexedLocalNamedBatchAssemblerStats stats = loader.getStatsSnapshot(ExampleType::TRAIN);
    EXPECT_EQ(stats.splitName, "train");
    EXPECT_GE(stats.recordsRequested, 2);
    EXPECT_GE(stats.logicalRecordBytesRequested, 2 * layout.recordSizeBytes());
    EXPECT_EQ(stats.logicalRecordBytesRequested % layout.recordSizeBytes(), 0);
    EXPECT_GE(stats.readCallsSubmitted, 2);
    EXPECT_GE(stats.readBytesSubmitted, 2 * layout.recordSizeBytes());
    EXPECT_GE(stats.readCallsCompleted, 2);
    EXPECT_GE(stats.readBytesCompleted, 2 * layout.recordSizeBytes());
    EXPECT_EQ(stats.recordsCopied, 0);
    EXPECT_EQ(stats.recordCopyBytes, 0);
    EXPECT_EQ(stats.recordCopyMemcpyCalls, 0);
    EXPECT_GE(stats.recordCopyActiveNanoseconds, 0);
    EXPECT_GE(stats.recordCopyPopWaitNanoseconds, 0);
    EXPECT_GE(stats.completedRecordQueuePushWaitNanoseconds, 0);
    EXPECT_GE(stats.copiedRecordQueuePushWaitNanoseconds, 0);
    EXPECT_EQ(stats.recordBufferPoolCapacity, 0);
    EXPECT_EQ(stats.currentRecordBufferPoolDepth, 0);
    EXPECT_DOUBLE_EQ(stats.averageCopyBytesPerRecord(), 0.0);
    EXPECT_DOUBLE_EQ(stats.averageCopyMemcpyCallsPerRecord(), 0.0);
    EXPECT_GE(stats.batchesAssembled, 2);
    EXPECT_EQ(stats.batchesDelivered, 0);
    EXPECT_EQ(stats.batchBuffersReturned, 0);
    EXPECT_GE(stats.currentReadyBatches, 2);
    EXPECT_EQ(stats.targetBatchQueueDepth, 3);
    EXPECT_GE(stats.loadWorkPopCalls, 0);
    EXPECT_GE(stats.loadWorkerBatches, 0);
    EXPECT_GE(stats.loadWorkerActiveNanoseconds, 0);
    EXPECT_GE(stats.loadWorkerReadSubmitNanoseconds, 0);
    EXPECT_GE(stats.loadWorkerReadDrainNanoseconds, 0);
    EXPECT_GE(stats.readvSubmitBackpressureCount, 0);
    EXPECT_GE(stats.readvCompletionWaitCalls, 0);
    EXPECT_GE(stats.readerDrainCalls, 0);
    EXPECT_GE(stats.readerDrainContextVisits, 0);
    EXPECT_GE(stats.readerDrainSubmitCalls, 0);
    EXPECT_GE(stats.readerDrainSubmitNanoseconds, 0);
    EXPECT_GE(stats.readerDrainWaitLoopNanoseconds, 0);
    EXPECT_GE(stats.readerDrainCompletionProcessNanoseconds, 0);
    EXPECT_GE(stats.readerDrainCompletions, 0);
    EXPECT_GE(stats.readerDrainMaxInflightReads, 0);
    EXPECT_GE(stats.readerShardContextOpenCount, 0);
    EXPECT_GE(stats.readerLoadExampleCalls, 0);
    EXPECT_GE(stats.readerLoadExampleNanoseconds, 0);
    EXPECT_GE(stats.readerResolveShardNanoseconds, 0);
    EXPECT_GE(stats.readerShardContextLookupCalls, 0);
    EXPECT_GE(stats.readerShardContextCacheHits, 0);
    EXPECT_GE(stats.readerShardContextCacheMisses, 0);
    EXPECT_GE(stats.readerShardContextLookupNanoseconds, 0);
    EXPECT_GE(stats.readerShardReadRequestNanoseconds, 0);
    EXPECT_GE(stats.readerIovecSlotAcquireNanoseconds, 0);
    EXPECT_GE(stats.readerIovecFillNanoseconds, 0);
    EXPECT_GE(stats.readerReadvSubmitCallNanoseconds, 0);
    EXPECT_GE(stats.startBatchCalls, 0);
    EXPECT_GE(stats.oldestPendingBatchAgeNanoseconds, 0);
    EXPECT_GE(stats.averagePendingBatchAgeNanoseconds, 0);
    EXPECT_TRUE(isReadvBackendName(stats.resolvedIoBackend)) << stats.resolvedIoBackend;

    uint64_t batchNum = 99;
    BatchLease batchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    const Batch& batch = batchLease.get();
    batchLease.reset();

    stats = loader.getStatsSnapshot(ExampleType::TRAIN);
    EXPECT_GE(stats.batchesDelivered, 1);
    EXPECT_GE(stats.batchBuffersReturned, 1);
    EXPECT_GE(stats.getBatchCalls, 1);
    EXPECT_GE(stats.returnBufferCalls, 1);
    EXPECT_GE(stats.getBatchReadyQueueEmptyCount + stats.getBatchImmediateCount, stats.getBatchCalls);

    IndexedLocalNamedBatchAssemblerStats validateStats = loader.getStatsSnapshot(ExampleType::VALIDATE);
    EXPECT_EQ(validateStats.splitName, "validate");
    EXPECT_EQ(validateStats.recordsRequested, 0);
    EXPECT_EQ(validateStats.targetBatchQueueDepth, 3);
    EXPECT_EQ(validateStats.recordBufferPoolCapacity, 0);
    EXPECT_EQ(validateStats.resolvedIoBackend, "empty");

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, RejectsReturnedBatchMissingTensor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("missing_returned_tensor");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1}, {2}, std::nullopt, 2, 2, false, std::nullopt);
    uint64_t batchNum = 0;
    BatchLease batchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    Batch malformedBatch = batchLease.get();
    malformedBatch.values().erase("daily_weight");
    EXPECT_THROW(
        loader.recycleBatchForTesting(ExampleType::TRAIN, std::move(malformedBatch)),
        std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, RejectsReturnedBatchWithExtraTensor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("extra_returned_tensor");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1}, {2}, std::nullopt, 2, 2, false, std::nullopt);
    uint64_t batchNum = 0;
    BatchLease batchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    Batch malformedBatch = batchLease.get();
    malformedBatch.insert(
        "unexpected_tensor",
        malformedBatch.getTensor("daily_weight"));
    EXPECT_THROW(
        loader.recycleBatchForTesting(ExampleType::TRAIN, std::move(malformedBatch)),
        std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, RejectsReturnedBatchWithWrongTensorDescriptor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("wrong_returned_tensor_descriptor");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(datasetPath, layout, {0, 1}, {2}, std::nullopt, 2, 2, false, std::nullopt);
    uint64_t batchNum = 0;
    BatchLease batchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    Batch malformedBatch = batchLease.get();
    malformedBatch.values()["daily_weight"] =
        malformedBatch.getTensor("seasonality_inputs");
    EXPECT_THROW(
        loader.recycleBatchForTesting(ExampleType::TRAIN, std::move(malformedBatch)),
        std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, DescribesCanonicalDatasetAndSeparateSessionWithoutConsumingBatches) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("device_materialization_description");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(
        datasetPath,
        layout,
        {4, 2, 0},
        {1, 3},
        vector<uint64_t>{0, 4},
        2,
        2,
        false,
        std::nullopt);

    Thor::DatasetMaterializationDescription datasetDescription =
        materializationDescription(loader);
    EXPECT_EQ(datasetDescription.datasetPath, datasetPath);
    EXPECT_EQ(
        datasetDescription.source,
        Thor::DatasetMaterializationSource::FILE_DATASET);
    EXPECT_EQ(datasetDescription.datasetId, loader.getDataset()->getId());
    EXPECT_NO_THROW(datasetDescription.layout.validateRequestedLayoutExact(layout));
    EXPECT_EQ(datasetDescription.numExamples, 5);

    Thor::DeviceDatasetSessionDescription sessionDescription =
        deviceSessionDescription(loader);
    EXPECT_EQ(
        sessionDescription.getSplits().getTrain().materialize(),
        (vector<uint64_t>{4, 2, 0}));
    EXPECT_EQ(
        sessionDescription.getSplits().getValidate().materialize(),
        (vector<uint64_t>{1, 3}));
    EXPECT_EQ(
        sessionDescription.getSplits().getTest().materialize(),
        (vector<uint64_t>{0, 4}));
    EXPECT_EQ(sessionDescription.getBatching().getBatchSize(), 2);
    EXPECT_FALSE(sessionDescription.getBatching().getRandomizeTrain());
    EXPECT_FALSE(sessionDescription.getBatching().getRandomSeed().has_value());

    uint64_t batchNum = 99;
    BatchLease trainBatchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    const Batch& trainBatch = trainBatchLease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(
        trainBatch.getTensor("seasonality_inputs"),
        {80.0f, 81.0f, 40.0f, 41.0f});
    trainBatchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedNamedBatchSessionTest, DeviceDatasetSessionDescriptionPreservesRandomizationOnlyInSessionState) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("device_session_randomized_metadata");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(
        datasetPath,
        layout,
        {0, 1, 2, 3, 4},
        {},
        std::nullopt,
        2,
        2,
        true,
        12345);
    Thor::DatasetMaterializationDescription datasetDescription =
        materializationDescription(loader);
    Thor::DeviceDatasetSessionDescription sessionDescription =
        deviceSessionDescription(loader);

    EXPECT_EQ(datasetDescription.numExamples, 5);
    EXPECT_TRUE(sessionDescription.getBatching().getRandomizeTrain());
    ASSERT_TRUE(sessionDescription.getBatching().getRandomSeed().has_value());
    EXPECT_EQ(sessionDescription.getBatching().getRandomSeed().value(), 12345);
    EXPECT_EQ(
        sessionDescription.getSplits().getTrain().materialize(),
        (vector<uint64_t>{0, 1, 2, 3, 4}));
    EXPECT_TRUE(sessionDescription.getSplits().getValidate().empty());
    EXPECT_TRUE(sessionDescription.getSplits().getTest().empty());

    std::filesystem::remove_all(datasetPath);
}

TEST(NamedDatasetMaterializerTest, MaterializesEveryCanonicalRowExactlyOnceInDatasetOrder) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_canonical_snapshot");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(
        datasetPath,
        layout,
        {4, 2, 0},
        {1, 3},
        vector<uint64_t>{0, 4},
        2,
        1,
        false,
        std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(loader),
        2);

    EXPECT_EQ(snapshot.datasetId, loader.getDataset()->getId());
    EXPECT_EQ(snapshot.schema, loader.getDataset()->getSchema());
    EXPECT_NO_THROW(snapshot.layout.validateRequestedLayoutExact(layout));
    EXPECT_EQ(snapshot.numExamples, 5);
    EXPECT_EQ(snapshot.totalExamples(), 5);
    EXPECT_EQ(snapshot.totalBytes(), 5 * layout.recordSizeBytes());
    EXPECT_GE(snapshot.materializationSeconds, 0.0);
    const Thor::DatasetField &seasonalityField =
        snapshot.schema.getField("seasonality_inputs");
    EXPECT_TRUE(snapshot.hasField(seasonalityField.id));
    EXPECT_EQ(
        snapshot.field(seasonalityField.id).getDescriptor(),
        ThorImplementation::TensorDescriptor(seasonalityField.dataType, {5, 2}));
    expectTensorValues(
        snapshot.tensor("seasonality_inputs"),
        {0.0f, 1.0f, 20.0f, 21.0f, 40.0f, 41.0f, 60.0f, 61.0f, 80.0f, 81.0f});
    expectTensorValues(
        snapshot.tensor("monotone_inputs"),
        {10.0f, 11.0f, 12.0f,
         30.0f, 31.0f, 32.0f,
         50.0f, 51.0f, 52.0f,
         70.0f, 71.0f, 72.0f,
         90.0f, 91.0f, 92.0f});
    expectTensorValues(
        snapshot.tensor("daily_weight"),
        {100.0f, 120.0f, 140.0f, 160.0f, 180.0f});

    std::filesystem::remove_all(datasetPath);
}

TEST(NamedDatasetMaterializerTest, MaterializationDoesNotAdvanceLiveSessionBatchState) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_does_not_consume_loader");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(
        datasetPath,
        layout,
        {4, 2, 0},
        {1, 3},
        std::nullopt,
        2,
        1,
        false,
        std::nullopt);
    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 1));
    const uint64_t beforeNextBatchNum = loader.getNextBatchNum(ExampleType::TRAIN);

    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(loader),
        2);
    EXPECT_EQ(snapshot.numExamples, 5);
    EXPECT_EQ(loader.getNextBatchNum(ExampleType::TRAIN), beforeNextBatchNum);

    uint64_t batchNum = 99;
    BatchLease sourceBatchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    const Batch& sourceBatch = sourceBatchLease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(
        sourceBatch.getTensor("seasonality_inputs"),
        {80.0f, 81.0f, 40.0f, 41.0f});
    sourceBatchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(NamedDatasetMaterializerTest, FiveFoldManifestsDoNotChangeCanonicalSnapshotOrResidentStorageEstimate) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_independent_of_splits");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);
    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);

    const vector<float> expectedSeasonality{
        0.0f, 1.0f, 20.0f, 21.0f, 40.0f, 41.0f, 60.0f, 61.0f, 80.0f, 81.0f};
    std::optional<uint64_t> expectedStorageBytes;

    for (uint64_t fold = 0; fold < 5; ++fold) {
        vector<uint64_t> trainIndices;
        for (uint64_t row = 0; row < 5; ++row) {
            if (row != fold) {
                trainIndices.push_back(row);
            }
        }
        TestIndexedNamedBatchSession foldLoader(
            dataset,
            std::move(trainIndices),
            vector<uint64_t>{fold},
            vector<uint64_t>{fold},
            fold + 1,
            1,
            (fold % 2) != 0,
            (fold % 2) != 0 ? std::optional<uint64_t>(100 + fold) : std::nullopt);

        Thor::DatasetMaterializationDescription description =
            materializationDescription(foldLoader);
        EXPECT_EQ(description.datasetId, dataset->getId());
        EXPECT_EQ(description.schema, dataset->getSchema());
        EXPECT_EQ(description.numExamples, 5);

        const uint64_t storageBytes =
            Thor::estimateDeviceResidentNamedDatasetStorageBytes(description);
        if (!expectedStorageBytes.has_value()) {
            expectedStorageBytes = storageBytes;
        }
        EXPECT_EQ(storageBytes, expectedStorageBytes.value());
        EXPECT_EQ(storageBytes, 5 * layout.recordSizeBytes());

        MaterializedNamedDatasetSnapshot snapshot =
            materializeNamedDatasetSnapshot(description, 2);
        EXPECT_EQ(snapshot.totalExamples(), 5);
        EXPECT_EQ(snapshot.totalBytes(), storageBytes);
        EXPECT_EQ(
            tensorValues(snapshot.tensor("seasonality_inputs")),
            expectedSeasonality);
    }

    std::filesystem::remove_all(datasetPath);
}

TEST(NamedDatasetMaterializerTest, MaterializesWindowedFieldsInCanonicalRowOrder) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_windowed");
    DatasetLayout layout = materializerWindowedLayout();
    writeWindowedMaterializerDataset(datasetPath);

    TestIndexedNamedBatchSession loader(
        datasetPath,
        layout,
        {2, 0},
        {1},
        vector<uint64_t>{0, 2},
        3,
        1,
        false,
        std::nullopt);
    NamedDatasetMaterializationSupport support =
        checkNamedDatasetSnapshotMaterializationSupport(
            materializationDescription(loader));
    EXPECT_TRUE(support.supported) << support.reason;

    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(loader),
        2);
    EXPECT_EQ(snapshot.numExamples, 3);
    const Thor::DatasetField &denseField = snapshot.schema.getField("dense");
    const Thor::DatasetField &historyField = snapshot.schema.getField("history");
    const Thor::DatasetField &historyMaskField =
        snapshot.schema.getField("history_mask");
    EXPECT_EQ(
        snapshot.field(denseField.id).getDescriptor(),
        ThorImplementation::TensorDescriptor(denseField.dataType, {3, 2}));
    EXPECT_EQ(historyField.dimensions, (vector<uint64_t>{3, 1}));
    EXPECT_EQ(historyMaskField.dimensions, (vector<uint64_t>{3}));
    EXPECT_EQ(
        snapshot.field(historyField.id).getDescriptor(),
        ThorImplementation::TensorDescriptor(historyField.dataType, {3, 3, 1}));
    EXPECT_EQ(
        snapshot.field(historyMaskField.id).getDescriptor(),
        ThorImplementation::TensorDescriptor(historyMaskField.dataType, {3, 3}));
    expectTensorValues(
        snapshot.tensor("dense"),
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    expectTensorValues(
        snapshot.tensor("history"),
        {10.0f, 11.0f, 12.0f,
         0.0f, 0.0f, 10.0f,
         12.0f, 13.0f, 0.0f});
    expectUint8TensorValuesOnHost(
        snapshot.tensor("history_mask"),
        {1, 1, 1, 0, 0, 1, 1, 1, 0});

    std::filesystem::remove_all(datasetPath);
}

TEST(NamedDatasetMaterializerTest, MultipleShiftedFieldsShareOnePhysicalWindowSource) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_shared_window_source");
    DatasetLayout layout = sharedWindowSourceLayout();
    writeSharedWindowSourceDataset(datasetPath);

    TestIndexedNamedBatchSession session(
        datasetPath,
        layout,
        {1, 0},
        {},
        std::nullopt,
        2,
        1,
        false,
        std::nullopt);
    NamedDatasetMaterializationSupport support =
        checkNamedDatasetSnapshotMaterializationSupport(materializationDescription(session));
    EXPECT_TRUE(support.supported) << support.reason;

    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(session),
        2);
    EXPECT_EQ(snapshot.numExamples, 2);
    expectUint8TensorValuesOnHost(
        snapshot.tensor("examples"),
        {10, 11, 12, 13,
         12, 13, 14, 15});
    expectUint8TensorValuesOnHost(
        snapshot.tensor("labels"),
        {11, 12, 13, 14,
         13, 14, 15, 16});

    const DatasetLayout persisted = DatasetLayout::readManifest(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    ASSERT_EQ(persisted.windowedTensorSources().size(), 1);
    EXPECT_EQ(persisted.windowedTensor("examples").sourceName, "tokens");
    EXPECT_EQ(persisted.windowedTensor("labels").sourceName, "tokens");
    EXPECT_EQ(persisted.windowedTensorSource("tokens").sourceSequences.size(), 1);

    std::filesystem::remove_all(datasetPath);
}

TEST(NamedDatasetMaterializerTest, MaterializesAffineWindowedFieldsInCanonicalRowOrder) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_affine_window_source");
    DatasetLayout layout = affineSharedWindowSourceLayout();
    writeAffineSharedWindowSourceDataset(datasetPath);

    TestIndexedNamedBatchSession session(
        datasetPath,
        layout,
        {1, 0},
        {},
        std::nullopt,
        2,
        1,
        false,
        std::nullopt);
    NamedDatasetMaterializationSupport support =
        checkNamedDatasetSnapshotMaterializationSupport(materializationDescription(session));
    EXPECT_TRUE(support.supported) << support.reason;

    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(session),
        2);
    EXPECT_EQ(snapshot.numExamples, 2);
    expectUint8TensorValuesOnHost(
        snapshot.tensor("examples"),
        {10, 11, 12, 13,
         12, 13, 14, 15});
    expectUint8TensorValuesOnHost(
        snapshot.tensor("labels"),
        {11, 12, 13, 14,
         13, 14, 15, 16});

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedDatasetTest, UploadsOneCanonicalDenseDatasetToGpu) {
    requireCudaDevice("CUDA device is required for device-resident named dataset tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_dataset_upload");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession loader(
        datasetPath,
        layout,
        {4, 2, 0},
        {1, 3},
        vector<uint64_t>{0, 4},
        2,
        1,
        false,
        std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(loader),
        2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(
        snapshot,
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));

    EXPECT_EQ(resident->getDatasetId(), loader.getDataset()->getId());
    EXPECT_EQ(resident->getSchema(), loader.getDataset()->getSchema());
    EXPECT_NO_THROW(resident->getLayout().validateRequestedLayoutExact(layout));
    EXPECT_EQ(resident->getNumExamples(), 5);
    EXPECT_EQ(resident->totalExamples(), 5);
    EXPECT_EQ(resident->totalBytes(), snapshot.totalBytes());
    EXPECT_GE(resident->getUploadSeconds(), 0.0);
    EXPECT_EQ(
        resident->getPlacement(),
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    expectTensorValuesOnHost(
        resident->tensor("seasonality_inputs"),
        {0.0f, 1.0f, 20.0f, 21.0f, 40.0f, 41.0f, 60.0f, 61.0f, 80.0f, 81.0f});

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceDatasetResidencyCacheTest, ConcurrentAcquisitionsShareOneUploadAndReservation) {
    requireCudaDevice("CUDA device is required for shared device dataset residency tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");
    Thor::resetDeviceDatasetMemoryReservationsForTesting();

    const std::filesystem::path datasetPath = makeTempDatasetPath("shared_residency_single_flight");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);
    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    TestIndexedNamedBatchSession source(
        dataset, {0, 1, 2, 3, 4}, {}, std::nullopt, 2, 1, false, std::nullopt);
    Thor::DatasetMaterializationDescription description =
        materializationDescription(source);
    MaterializedNamedDatasetSnapshot snapshot =
        materializeNamedDatasetSnapshot(description, 2);
    const uint64_t residentBytes = snapshot.totalBytes();
    TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);

    std::atomic<uint64_t> uploadCount{0};
    std::promise<void> constructionStartedPromise;
    std::shared_future<void> constructionStarted =
        constructionStartedPromise.get_future().share();
    std::promise<void> releaseConstructionPromise;
    std::shared_future<void> releaseConstruction =
        releaseConstructionPromise.get_future().share();

    auto construct = [&]() -> std::shared_ptr<const DeviceResidentNamedDataset> {
        uploadCount.fetch_add(1, std::memory_order_relaxed);
        constructionStartedPromise.set_value();
        releaseConstruction.wait();
        return DeviceResidentNamedDataset::fromSnapshot(snapshot, placement);
    };
    auto acquire = [&]() {
        Thor::DeviceDatasetResidencyRequest request(
            dataset->getId(),
            dataset->getNumExamples(),
            placement,
            datasetFieldIds(dataset->getSchema()),
            residentBytes,
            residentBytes,
            Thor::DeviceDatasetStorage::STRICT,
            ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES +
                residentBytes * 4,
            construct);
        return ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset)
            .acquire(request);
    };

    std::future<Thor::DeviceDatasetResidencyAcquisition> first =
        std::async(std::launch::async, acquire);
    ASSERT_EQ(constructionStarted.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    std::future<Thor::DeviceDatasetResidencyAcquisition> second =
        std::async(std::launch::async, acquire);
    const bool followerJoined = waitForCondition([&]() {
        return ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset)
                   .getTelemetry()
                   .constructionJoins == 1;
    });
    releaseConstructionPromise.set_value();
    ASSERT_TRUE(followerJoined);
    Thor::DeviceDatasetResidencyAcquisition firstAcquisition = first.get();
    Thor::DeviceDatasetResidencyAcquisition secondAcquisition = second.get();

    EXPECT_EQ(uploadCount.load(std::memory_order_relaxed), 1u);
    EXPECT_TRUE(firstAcquisition.startedConstruction);
    EXPECT_TRUE(secondAcquisition.joinedConstruction);
    EXPECT_EQ(
        firstAcquisition.lease.getShared().get(),
        secondAcquisition.lease.getShared().get());

    const Thor::DeviceDatasetResidencyTelemetry cacheTelemetry =
        ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset).getTelemetry();
    EXPECT_EQ(cacheTelemetry.constructionStarts, 1u);
    EXPECT_EQ(cacheTelemetry.constructionJoins, 1u);
    EXPECT_EQ(cacheTelemetry.successfulConstructions, 1u);
    EXPECT_EQ(cacheTelemetry.failedConstructions, 0u);

    const Thor::DeviceDatasetMemoryReservationTelemetry reservationTelemetry =
        Thor::getDeviceDatasetMemoryReservationTelemetryForTesting(0);
    EXPECT_EQ(reservationTelemetry.reservationAttempts, 1u);
    EXPECT_EQ(reservationTelemetry.reservationFailures, 0u);
    EXPECT_EQ(reservationTelemetry.currentReservedBytes, 0u);
    EXPECT_EQ(reservationTelemetry.peakReservedBytes, residentBytes);
    EXPECT_EQ(reservationTelemetry.activeCommittedBytes, residentBytes);

    std::weak_ptr<const DeviceResidentNamedDataset> weakResident =
        firstAcquisition.lease.getShared();
    firstAcquisition.lease = Thor::DeviceDatasetLease();
    EXPECT_FALSE(weakResident.expired());
    secondAcquisition.lease = Thor::DeviceDatasetLease();
    EXPECT_TRUE(weakResident.expired());
    ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset).clearExpired();
    EXPECT_EQ(
        Thor::getDeviceDatasetMemoryReservationTelemetryForTesting(0)
            .activeCommittedBytes,
        0u);

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceDatasetResidencyCacheTest, DifferentGpuDevicesReceiveSeparateReplicas) {
    int deviceCount = 0;
    const cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount < 2) {
        GTEST_SKIP() << "Two CUDA devices are required for per-device residency cache coverage.";
    }
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");
    Thor::resetDeviceDatasetMemoryReservationsForTesting();

    const std::filesystem::path datasetPath = makeTempDatasetPath("shared_residency_per_device");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);
    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    TestIndexedNamedBatchSession source(
        dataset, {0, 1, 2, 3, 4}, {}, std::nullopt, 2, 1, false, std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(source), 2);
    const uint64_t residentBytes = snapshot.totalBytes();
    const std::set<Thor::DatasetFieldId> fields = datasetFieldIds(dataset->getSchema());

    auto acquireOn = [&](int deviceNum) {
        TensorPlacement placement(TensorPlacement::MemDevices::GPU, deviceNum);
        Thor::DeviceDatasetResidencyRequest request(
            dataset->getId(),
            dataset->getNumExamples(),
            placement,
            fields,
            residentBytes,
            residentBytes,
            Thor::DeviceDatasetStorage::STRICT,
            ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES +
                residentBytes * 4,
            [&, placement]() -> std::shared_ptr<const DeviceResidentNamedDataset> {
                return DeviceResidentNamedDataset::fromSnapshot(snapshot, placement);
            });
        return ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset)
            .acquire(request);
    };

    Thor::DeviceDatasetResidencyAcquisition gpu0 = acquireOn(0);
    Thor::DeviceDatasetResidencyAcquisition gpu1 = acquireOn(1);
    EXPECT_NE(gpu0.lease.getShared().get(), gpu1.lease.getShared().get());
    EXPECT_EQ(gpu0.lease->getPlacement(), TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    EXPECT_EQ(gpu1.lease->getPlacement(), TensorPlacement(TensorPlacement::MemDevices::GPU, 1));
    EXPECT_EQ(
        ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset)
            .getTelemetry()
            .constructionStarts,
        2u);

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceDatasetResidencyCacheTest, FailedConstructionWakesWaitersAndDoesNotPoisonRetry) {
    requireCudaDevice("CUDA device is required for shared device dataset residency tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");
    Thor::resetDeviceDatasetMemoryReservationsForTesting();

    const std::filesystem::path datasetPath = makeTempDatasetPath("shared_residency_failure_retry");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);
    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    TestIndexedNamedBatchSession source(
        dataset, {0, 1, 2, 3, 4}, {}, std::nullopt, 2, 1, false, std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(source), 2);
    const uint64_t residentBytes = snapshot.totalBytes();
    TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);

    std::promise<void> constructionStartedPromise;
    std::shared_future<void> constructionStarted =
        constructionStartedPromise.get_future().share();
    std::promise<void> releaseFailurePromise;
    std::shared_future<void> releaseFailure =
        releaseFailurePromise.get_future().share();
    std::atomic<uint64_t> failedBuildCount{0};

    auto failingAcquire = [&]() {
        Thor::DeviceDatasetResidencyRequest request(
            dataset->getId(),
            dataset->getNumExamples(),
            placement,
            datasetFieldIds(dataset->getSchema()),
            residentBytes,
            residentBytes,
            Thor::DeviceDatasetStorage::BEST_EFFORT,
            ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES +
                residentBytes + 1,
            [&]() -> std::shared_ptr<const DeviceResidentNamedDataset> {
                failedBuildCount.fetch_add(1, std::memory_order_relaxed);
                constructionStartedPromise.set_value();
                releaseFailure.wait();
                throw std::runtime_error("intentional shared residency failure");
            });
        return ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset)
            .acquire(request);
    };

    std::future<Thor::DeviceDatasetResidencyAcquisition> first =
        std::async(std::launch::async, failingAcquire);
    ASSERT_EQ(constructionStarted.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    std::future<Thor::DeviceDatasetResidencyAcquisition> second =
        std::async(std::launch::async, failingAcquire);
    const bool followerJoined = waitForCondition([&]() {
        return ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset)
                   .getTelemetry()
                   .constructionJoins == 1;
    });
    releaseFailurePromise.set_value();
    ASSERT_TRUE(followerJoined);

    for (std::future<Thor::DeviceDatasetResidencyAcquisition> *future : {&first, &second}) {
        try {
            (void)future->get();
            FAIL() << "Expected shared construction failure.";
        } catch (const std::runtime_error &e) {
            EXPECT_STREQ(e.what(), "intentional shared residency failure");
        }
    }
    EXPECT_EQ(failedBuildCount.load(std::memory_order_relaxed), 1u);
    EXPECT_EQ(
        Thor::getDeviceDatasetMemoryReservationTelemetryForTesting(0)
            .currentReservedBytes,
        0u);

    Thor::DeviceDatasetResidencyRequest retryRequest(
        dataset->getId(),
        dataset->getNumExamples(),
        placement,
        datasetFieldIds(dataset->getSchema()),
        residentBytes,
        residentBytes,
        Thor::DeviceDatasetStorage::STRICT,
        ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES +
            residentBytes * 4,
        [&]() -> std::shared_ptr<const DeviceResidentNamedDataset> {
            return DeviceResidentNamedDataset::fromSnapshot(snapshot, placement);
        });
    Thor::DeviceDatasetResidencyAcquisition retry =
        ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset).acquire(retryRequest);
    EXPECT_TRUE(retry.startedConstruction);
    EXPECT_TRUE(retry.lease);

    const Thor::DeviceDatasetResidencyTelemetry telemetry =
        ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset).getTelemetry();
    EXPECT_EQ(telemetry.constructionStarts, 2u);
    EXPECT_EQ(telemetry.failedConstructions, 1u);
    EXPECT_EQ(telemetry.successfulConstructions, 1u);

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceDatasetResidencyCacheTest, DifferentDatasetsCannotReserveTheSameOverrideBytes) {
    requireCudaDevice("CUDA device is required for shared device dataset residency tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");
    Thor::resetDeviceDatasetMemoryReservationsForTesting();

    const std::filesystem::path firstPath = makeTempDatasetPath("reservation_first_dataset");
    const std::filesystem::path secondPath = makeTempDatasetPath("reservation_second_dataset");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(firstPath, layout);
    writeCanonicalDataset(secondPath, layout);
    std::shared_ptr<Thor::FileDataset> firstDataset =
        Thor::FileDataset::open(firstPath);
    std::shared_ptr<Thor::FileDataset> secondDataset =
        Thor::FileDataset::open(secondPath);
    TestIndexedNamedBatchSession firstSource(
        firstDataset, {0, 1, 2, 3, 4}, {}, std::nullopt, 2, 1, false, std::nullopt);
    TestIndexedNamedBatchSession secondSource(
        secondDataset, {0, 1, 2, 3, 4}, {}, std::nullopt, 2, 1, false, std::nullopt);
    MaterializedNamedDatasetSnapshot firstSnapshot = materializeNamedDatasetSnapshot(
        materializationDescription(firstSource), 2);
    MaterializedNamedDatasetSnapshot secondSnapshot = materializeNamedDatasetSnapshot(
        materializationDescription(secondSource), 2);
    const uint64_t residentBytes = firstSnapshot.totalBytes();
    ASSERT_EQ(secondSnapshot.totalBytes(), residentBytes);
    TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);

    std::promise<void> firstReservedPromise;
    std::shared_future<void> firstReserved = firstReservedPromise.get_future().share();
    std::promise<void> releaseFirstPromise;
    std::shared_future<void> releaseFirst = releaseFirstPromise.get_future().share();

    auto firstAcquire = [&]() {
        Thor::DeviceDatasetResidencyRequest request(
            firstDataset->getId(),
            firstDataset->getNumExamples(),
            placement,
            datasetFieldIds(firstDataset->getSchema()),
            residentBytes,
            residentBytes,
            Thor::DeviceDatasetStorage::STRICT,
            ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES +
                residentBytes,
            [&]() -> std::shared_ptr<const DeviceResidentNamedDataset> {
                firstReservedPromise.set_value();
                releaseFirst.wait();
                return DeviceResidentNamedDataset::fromSnapshot(firstSnapshot, placement);
            });
        return ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*firstDataset)
            .acquire(request);
    };
    std::future<Thor::DeviceDatasetResidencyAcquisition> first =
        std::async(std::launch::async, firstAcquire);
    ASSERT_EQ(firstReserved.wait_for(std::chrono::seconds(2)), std::future_status::ready);

    Thor::DeviceDatasetResidencyRequest secondRequest(
        secondDataset->getId(),
        secondDataset->getNumExamples(),
        placement,
        datasetFieldIds(secondDataset->getSchema()),
        residentBytes,
        residentBytes,
        Thor::DeviceDatasetStorage::STRICT,
        ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES +
            residentBytes,
        [&]() -> std::shared_ptr<const DeviceResidentNamedDataset> {
            return DeviceResidentNamedDataset::fromSnapshot(secondSnapshot, placement);
        });
    EXPECT_THROW(
        (void)ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*secondDataset)
            .acquire(secondRequest),
        Thor::DeviceDatasetResidencyAdmissionError);

    const Thor::DeviceDatasetMemoryReservationTelemetry whileReserved =
        Thor::getDeviceDatasetMemoryReservationTelemetryForTesting(0);
    EXPECT_EQ(whileReserved.reservationAttempts, 2u);
    EXPECT_EQ(whileReserved.reservationFailures, 1u);
    EXPECT_EQ(whileReserved.currentReservedBytes, residentBytes);
    EXPECT_EQ(whileReserved.peakReservedBytes, residentBytes);

    releaseFirstPromise.set_value();
    EXPECT_TRUE(first.get().lease);

    std::filesystem::remove_all(firstPath);
    std::filesystem::remove_all(secondPath);
}

TEST(DeviceDatasetStorageSelection, DenseMemoryDatasetUsesSharedResidencyPath) {
    requireCudaDevice("CUDA device is required for in-memory dataset residency tests.");
    Thor::resetDeviceDatasetMemoryReservationsForTesting();

    auto dataset = std::make_shared<DenseMemoryDatasetForTest>();
    Thor::TrainingData data(
        dataset,
        Thor::DatasetSplitManifest(
            *dataset,
            std::vector<uint64_t>{2, 0},
            std::vector<uint64_t>{1},
            std::nullopt),
        Thor::BatchPolicy(2, false, std::nullopt),
        Thor::DatasetAccessPolicy{
            .deviceStorage = Thor::DeviceDatasetStorage::STRICT},
        "dense_memory_dataset");
    std::shared_ptr<BatchSession> sourceSession = data.openSession(1);

    Thor::DatasetMaterializationDescription description =
        Thor::describeDatasetMaterialization(data);
    EXPECT_EQ(
        description.source,
        Thor::DatasetMaterializationSource::MEMORY);
    EXPECT_TRUE(description.datasetPath.empty());
    EXPECT_TRUE(
        checkNamedDatasetSnapshotMaterializationSupport(description).supported);

    constexpr uint64_t ampleBytes =
        ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES +
        (1ull << 20);
    Thor::DeviceDatasetStorageSelection selection =
        Thor::selectDeviceDatasetStorageSession(
            sourceSession,
            data,
            TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
            1,
            ampleBytes);

    auto residentSession =
        std::dynamic_pointer_cast<DeviceResidentNamedBatchSession>(
            selection.session);
    ASSERT_NE(residentSession, nullptr);
    EXPECT_TRUE(selection.report.attempted);
    EXPECT_TRUE(selection.report.used);
    EXPECT_TRUE(selection.report.reason.empty());
    EXPECT_EQ(selection.report.examples, 3u);
    EXPECT_EQ(selection.report.residentBytes, 3u * 3u * sizeof(float));
    EXPECT_TRUE(selection.report.residentConstructionStarted);
    EXPECT_EQ(selection.session->getDatasetName(), "dense_memory_dataset");

    uint64_t batchNum = 99;
    BatchLease batchLease = selection.session->leaseBatch(
        ExampleType::TRAIN,
        batchNum);
    EXPECT_EQ(batchNum, 0u);
    expectTensorValuesOnHost(
        batchLease.get().getTensor("features"),
        {5.0f, 6.0f, 1.0f, 2.0f});
    expectTensorValuesOnHost(
        batchLease.get().getTensor("labels"),
        {30.0f, 10.0f});

    std::shared_ptr<BatchSession> secondSourceSession = data.openSession(1);
    Thor::DeviceDatasetStorageSelection secondSelection =
        Thor::selectDeviceDatasetStorageSession(
            secondSourceSession,
            data,
            TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
            1,
            ampleBytes);
    EXPECT_TRUE(secondSelection.report.used);
    EXPECT_TRUE(secondSelection.report.residentCacheHit);
    EXPECT_FALSE(secondSelection.report.residentConstructionStarted);
    EXPECT_EQ(secondSelection.report.residentBytes, selection.report.residentBytes);
}

TEST(DeviceDatasetStorageSelection, FoldSessionsReuseSharedResidentDataset) {
    requireCudaDevice("CUDA device is required for shared device dataset residency tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");
    Thor::resetDeviceDatasetMemoryReservationsForTesting();

    const std::filesystem::path datasetPath = makeTempDatasetPath("selection_shared_across_folds");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);
    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    auto firstSource = std::make_shared<TestIndexedNamedBatchSession>(
        dataset, vector<uint64_t>{1, 2, 3, 4}, vector<uint64_t>{0}, std::nullopt,
        2, 1, false, std::nullopt);
    auto secondSource = std::make_shared<TestIndexedNamedBatchSession>(
        dataset, vector<uint64_t>{0, 2, 3, 4}, vector<uint64_t>{1}, std::nullopt,
        2, 1, false, std::nullopt);
    constexpr uint64_t ampleBytes =
        ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES +
        (1ull << 20);

    Thor::DeviceDatasetStorageSelection first =
        selectDeviceStorage(
            firstSource,
            Thor::DeviceDatasetStorage::STRICT,
            TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
            1,
            ampleBytes);
    Thor::DeviceDatasetStorageSelection second =
        selectDeviceStorage(
            secondSource,
            Thor::DeviceDatasetStorage::STRICT,
            TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
            1,
            /*availableBytesOverride=*/1);

    auto firstSession = std::dynamic_pointer_cast<DeviceResidentNamedBatchSession>(first.session);
    auto secondSession = std::dynamic_pointer_cast<DeviceResidentNamedBatchSession>(second.session);
    ASSERT_NE(firstSession, nullptr);
    ASSERT_NE(secondSession, nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<Thor::BatchSession>(first.session), nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<Thor::BatchSession>(second.session), nullptr);
    EXPECT_EQ(first.session->getDatasetName(), "dataset");
    EXPECT_EQ(second.session->getDatasetName(), "dataset");
    EXPECT_TRUE(first.report.residentConstructionStarted);
    EXPECT_FALSE(first.report.residentCacheHit);
    EXPECT_TRUE(second.report.residentCacheHit);
    EXPECT_FALSE(second.report.residentConstructionStarted);
    EXPECT_EQ(first.report.residentBytes, second.report.residentBytes);
    EXPECT_EQ(
        firstSession->getDeviceDataset().get(),
        secondSession->getDeviceDataset().get());
    EXPECT_EQ(
        ThorImplementation::NamedDatasetRuntimeAccess::residencyCache(*dataset)
            .getTelemetry()
            .constructionStarts,
        1u);

    first.session.reset();
    firstSession.reset();
    EXPECT_NE(secondSession->getDeviceDataset(), nullptr);

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchSessionTest, WindowedBatchesMatchSourceSession) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_windowed");
    DatasetLayout layout = materializerWindowedLayout();
    writeWindowedMaterializerDataset(datasetPath);

    TestIndexedNamedBatchSession sourceSession(
        datasetPath,
        layout,
        {2, 0, 1},
        {},
        vector<uint64_t>{},
        2,
        1,
        false,
        std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(sourceSession),
        2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(
        snapshot,
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    TestDeviceResidentNamedBatchSession deviceSession(
        resident,
        deviceSessionDescription(sourceSession),
        1);

    ASSERT_TRUE(deviceSession.getBatchTensorPlacement("dense").has_value());
    EXPECT_EQ(
        deviceSession.getBatchTensorPlacement("dense").value(),
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    ASSERT_TRUE(deviceSession.getBatchTensorPlacement("history").has_value());
    EXPECT_EQ(
        deviceSession.getBatchTensorPlacement("history").value(),
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    EXPECT_FALSE(deviceSession.getBatchTensorPlacement("missing").has_value());

    uint64_t sourceBatchNum = 99;
    uint64_t deviceBatchNum = 99;
    BatchLease sourceBatchLease = sourceSession.leaseBatch(ExampleType::TRAIN, sourceBatchNum);

    const Batch& sourceBatch = sourceBatchLease.get();
    BatchLease deviceBatchLease = deviceSession.leaseBatch(ExampleType::TRAIN, deviceBatchNum);

    const Batch& deviceBatch = deviceBatchLease.get();
    EXPECT_EQ(deviceBatchNum, sourceBatchNum);
    EXPECT_EQ(
        tensorValuesOnHost(deviceBatch.getTensor("dense")),
        tensorValues(sourceBatch.getTensor("dense")));
    EXPECT_EQ(
        tensorValuesOnHost(deviceBatch.getTensor("history")),
        tensorValues(sourceBatch.getTensor("history")));
    EXPECT_EQ(
        uint8TensorValuesOnHost(deviceBatch.getTensor("history_mask")),
        uint8TensorValuesOnHost(sourceBatch.getTensor("history_mask")));

    sourceBatchLease.reset();
    deviceBatchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceDatasetStorageSelection, BestEffortPrioritizesWindowedFeaturesWhenFullDatasetDoesNotFit) {
    requireCudaDevice("CUDA device is required for device dataset storage selection tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_storage_selects_windowed_hybrid");
    DatasetLayout layout = materializerWindowedLayout();
    writeWindowedMaterializerDataset(datasetPath);

    auto sourceSession = std::make_shared<TestIndexedNamedBatchSession>(
        datasetPath,
        layout,
        vector<uint64_t>{0, 1, 2},
        vector<uint64_t>{},
        vector<uint64_t>{},
        2,
        1,
        false,
        std::nullopt);

    const uint64_t availableBytes =
        ThorImplementation::DEVICE_STARTUP_SAFETY_RESERVE_BYTES + 92;
    Thor::DeviceDatasetStorageSelection selection =
        selectDeviceStorage(
            sourceSession,
            Thor::DeviceDatasetStorage::BEST_EFFORT,
            TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
            1,
            /*availableBytesOverride=*/availableBytes);

    EXPECT_TRUE(selection.report.used);
    EXPECT_EQ(selection.report.reason, "windowed_features_only");
    EXPECT_EQ(selection.report.requiredBytes, 91u);
    EXPECT_EQ(selection.report.examples, 3u);
    EXPECT_NE(selection.session, sourceSession);
    EXPECT_NE(
        std::dynamic_pointer_cast<DeviceResidentWindowedNamedBatchSession>(
            selection.session),
        nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<Thor::BatchSession>(selection.session), nullptr);
    EXPECT_EQ(selection.session->getDatasetName(), "dataset");

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentWindowedNamedBatchSessionTest, UsesCanonicalDeviceWindowsAndCpuDirectTensors) {
    requireCudaDevice("CUDA device is required for hybrid windowed device loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_windowed_hybrid");
    DatasetLayout layout = materializerWindowedLayout();
    writeWindowedMaterializerDataset(datasetPath);

    TestIndexedNamedBatchSession sourceSession(
        datasetPath,
        layout,
        {2, 0, 1},
        {},
        vector<uint64_t>{},
        2,
        1,
        false,
        std::nullopt);
    Thor::DatasetMaterializationDescription datasetDescription =
        materializationDescription(sourceSession);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        datasetDescription,
        2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(
        snapshot,
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
        std::set<string>{"history", "history_mask"});
    TestDeviceResidentWindowedNamedBatchSession deviceSession(
        datasetDescription,
        deviceSessionDescription(sourceSession),
        resident,
        1);

    ASSERT_TRUE(deviceSession.getBatchTensorPlacement("dense").has_value());
    EXPECT_EQ(
        deviceSession.getBatchTensorPlacement("dense").value().getMemDevice(),
        TensorPlacement::MemDevices::CPU);
    ASSERT_TRUE(deviceSession.getBatchTensorPlacement("history").has_value());
    EXPECT_EQ(
        deviceSession.getBatchTensorPlacement("history").value(),
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    ASSERT_TRUE(deviceSession.getBatchTensorPlacement("history_mask").has_value());
    EXPECT_EQ(
        deviceSession.getBatchTensorPlacement("history_mask").value(),
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    EXPECT_FALSE(deviceSession.getBatchTensorPlacement("missing").has_value());

    uint64_t sourceBatchNum = 99;
    uint64_t deviceBatchNum = 99;
    BatchLease sourceBatchLease = sourceSession.leaseBatch(ExampleType::TRAIN, sourceBatchNum);

    const Batch& sourceBatch = sourceBatchLease.get();
    BatchLease deviceBatchLease = deviceSession.leaseBatch(ExampleType::TRAIN, deviceBatchNum);

    const Batch& deviceBatch = deviceBatchLease.get();
    EXPECT_EQ(deviceBatchNum, sourceBatchNum);
    EXPECT_EQ(
        deviceBatch.getTensor("dense").getPlacement().getMemDevice(),
        TensorPlacement::MemDevices::CPU);
    EXPECT_EQ(
        deviceBatch.getTensor("history").getPlacement().getMemDevice(),
        TensorPlacement::MemDevices::GPU);
    EXPECT_EQ(
        tensorValuesOnHost(deviceBatch.getTensor("dense")),
        tensorValues(sourceBatch.getTensor("dense")));
    EXPECT_EQ(
        tensorValuesOnHost(deviceBatch.getTensor("history")),
        tensorValues(sourceBatch.getTensor("history")));
    EXPECT_EQ(
        uint8TensorValuesOnHost(deviceBatch.getTensor("history_mask")),
        uint8TensorValuesOnHost(sourceBatch.getTensor("history_mask")));

    sourceBatchLease.reset();
    deviceBatchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchSessionTest, SequentialBatchesGatherCanonicalRowsAndWrap) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_sequential");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession sourceSession(
        datasetPath,
        layout,
        {4, 2, 0},
        {1, 3},
        vector<uint64_t>{0, 4},
        2,
        1,
        false,
        std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(sourceSession),
        2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(
        snapshot,
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    TestDeviceResidentNamedBatchSession deviceSession(
        resident,
        deviceSessionDescription(sourceSession),
        2);

    EXPECT_EQ(deviceSession.getNumExamples(ExampleType::TRAIN), 3);
    EXPECT_EQ(deviceSession.getNumBatchesPerEpoch(ExampleType::TRAIN), 2);
    EXPECT_EQ(deviceSession.getNextBatchNum(ExampleType::TRAIN), 0);

    uint64_t batchNum = 99;
    BatchLease batch0Lease = deviceSession.leaseBatch(ExampleType::TRAIN, batchNum);

    const Batch& batch0 = batch0Lease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValuesOnHost(
        batch0.getTensor("seasonality_inputs"),
        {80.0f, 81.0f, 40.0f, 41.0f});
    EXPECT_EQ(deviceSession.getNextBatchNum(ExampleType::TRAIN), 1);
    batch0Lease.reset();

    BatchLease batch1Lease = deviceSession.leaseBatch(ExampleType::TRAIN, batchNum);


    const Batch& batch1 = batch1Lease.get();
    EXPECT_EQ(batchNum, 1);
    expectTensorValuesOnHost(
        batch1.getTensor("seasonality_inputs"),
        {0.0f, 1.0f, 80.0f, 81.0f});
    EXPECT_EQ(deviceSession.getNextBatchNum(ExampleType::TRAIN), 0);
    batch1Lease.reset();

    BatchLease batch2Lease = deviceSession.leaseBatch(ExampleType::TRAIN, batchNum);


    const Batch& batch2 = batch2Lease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValuesOnHost(
        batch2.getTensor("seasonality_inputs"),
        {40.0f, 41.0f, 0.0f, 1.0f});
    batch2Lease.reset();

    DeviceResidentNamedBatchSessionStats stats =
        deviceSession.getStatsSnapshot(ExampleType::TRAIN);
    EXPECT_EQ(stats.splitName, "train");
    EXPECT_EQ(stats.batchesGathered, 3);
    EXPECT_EQ(stats.batchesReturned, 3);
    EXPECT_EQ(stats.currentAvailableBatches, 2);
    EXPECT_EQ(stats.batchQueueDepth, 2);
    EXPECT_EQ(stats.residentExamples, 5);
    EXPECT_EQ(stats.residentBytes, resident->totalBytes());

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchSessionTest, ValidateAndTestManifestsAreSequentialAndIndependent) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_splits");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession sourceSession(
        datasetPath,
        layout,
        {4, 3},
        {1, 3, 4},
        vector<uint64_t>{2, 0, 1},
        2,
        1,
        true,
        9876);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(sourceSession),
        2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(
        snapshot,
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    TestDeviceResidentNamedBatchSession deviceSession(
        resident,
        deviceSessionDescription(sourceSession),
        1);

    uint64_t batchNum = 99;
    BatchLease validateBatch0Lease = deviceSession.leaseBatch(ExampleType::VALIDATE, batchNum);

    const Batch& validateBatch0 = validateBatch0Lease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValuesOnHost(
        validateBatch0.getTensor("seasonality_inputs"),
        {20.0f, 21.0f, 60.0f, 61.0f});
    validateBatch0Lease.reset();

    BatchLease testBatch0Lease = deviceSession.leaseBatch(ExampleType::TEST, batchNum);


    const Batch& testBatch0 = testBatch0Lease.get();
    EXPECT_EQ(batchNum, 0);
    expectTensorValuesOnHost(
        testBatch0.getTensor("seasonality_inputs"),
        {40.0f, 41.0f, 0.0f, 1.0f});
    testBatch0Lease.reset();

    BatchLease validateBatch1Lease = deviceSession.leaseBatch(ExampleType::VALIDATE, batchNum);


    const Batch& validateBatch1 = validateBatch1Lease.get();
    EXPECT_EQ(batchNum, 1);
    expectTensorValuesOnHost(
        validateBatch1.getTensor("seasonality_inputs"),
        {80.0f, 81.0f, 20.0f, 21.0f});
    validateBatch1Lease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchSessionTest, RandomizedTrainOrderMatchesSourceSessionForFixedSeed) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_randomized");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession sourceSession(
        datasetPath,
        layout,
        {0, 1, 2, 3, 4},
        {0},
        std::nullopt,
        2,
        2,
        true,
        12345);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(sourceSession),
        2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(
        snapshot,
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    TestDeviceResidentNamedBatchSession deviceSession(
        resident,
        deviceSessionDescription(sourceSession),
        2);

    for (uint64_t i = 0; i < 3; ++i) {
        uint64_t sourceBatchNum = 99;
        uint64_t deviceBatchNum = 99;
        BatchLease sourceBatchLease = sourceSession.leaseBatch(ExampleType::TRAIN, sourceBatchNum);

        const Batch& sourceBatch = sourceBatchLease.get();
        BatchLease deviceBatchLease = deviceSession.leaseBatch(ExampleType::TRAIN, deviceBatchNum);

        const Batch& deviceBatch = deviceBatchLease.get();
        EXPECT_EQ(deviceBatchNum, sourceBatchNum);
        EXPECT_EQ(
            tensorValuesOnHost(deviceBatch.getTensor("seasonality_inputs")),
            tensorValues(sourceBatch.getTensor("seasonality_inputs")))
            << "i=" << i;
        sourceBatchLease.reset();
        deviceBatchLease.reset();
    }

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchSessionTest, RejectsEmptySplitBatchRequest) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_empty_split");
    DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    TestIndexedNamedBatchSession sourceSession(
        datasetPath,
        layout,
        {0, 1, 2},
        {},
        vector<uint64_t>{},
        2,
        1,
        false,
        std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(sourceSession),
        2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(
        snapshot,
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    TestDeviceResidentNamedBatchSession deviceSession(
        resident,
        deviceSessionDescription(sourceSession),
        1);

    EXPECT_EQ(deviceSession.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(deviceSession.getNumBatchesPerEpoch(ExampleType::VALIDATE), 0);
    uint64_t batchNum = 99;
    EXPECT_THROW(
        (void)deviceSession.leaseBatch(ExampleType::VALIDATE, batchNum),
        std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}


class IndexedNamedBatchSessionBackendRegressionTest : public ::testing::TestWithParam<IndexedBackendCase> {};

TEST_P(IndexedNamedBatchSessionBackendRegressionTest, ExplicitBackendReadsIndexedDatasetAndReportsBackend) {
    exerciseIndexedLoaderWithBackendOrSkip(GetParam());
}

INSTANTIATE_TEST_SUITE_P(ExplicitBackends,
                         IndexedNamedBatchSessionBackendRegressionTest,
                         ::testing::Values(IndexedBackendCase{"pread_buffered", "pread_buffered"},
                                           IndexedBackendCase{"pread_direct", "pread_direct"},
                                           IndexedBackendCase{"uring_direct", "uring_direct"}),
                         [](const ::testing::TestParamInfo<IndexedNamedBatchSessionBackendRegressionTest::ParamType> &info) {
                             return std::string(info.param.displayName);
                         });

TEST(IndexedNamedBatchSessionPerf, LargeRandomRecordPrefetchSmoke) {
    const char *enabled = std::getenv("THOR_INDEXED_LOCAL_NAMED_LOADER_PERF_TEST");
    if (enabled == nullptr || enabled[0] == '\0' || (enabled[0] == '0' && enabled[1] == '\0')) {
        GTEST_SKIP() << "Set THOR_INDEXED_LOCAL_NAMED_LOADER_PERF_TEST=1 to run the indexed loader large-record smoke test.";
    }

    constexpr uint64_t elementsPerExample = 32 * 1024;  // 128 KiB records.
    constexpr uint64_t numExamples = 256;
    constexpr uint64_t batchSize = 16;
    constexpr uint64_t batchQueueDepth = 8;

    const std::filesystem::path datasetPath = makeTempDatasetPath("large_random_prefetch_smoke");
    DatasetLayout layout = largeRecordLayout(elementsPerExample);
    writeLargeIndexedDataset(datasetPath, layout, numExamples, elementsPerExample, 64);

    const std::vector<uint64_t> trainIndices = permutedIndices(numExamples, 37);
    const auto start = std::chrono::steady_clock::now();
    TestIndexedNamedBatchSession loader(datasetPath,
                                        layout,
                                        trainIndices,
                                        vector<uint64_t>{0, 1, 2, 3},
                                        vector<uint64_t>{4, 5, 6, 7},
                                        batchSize,
                                        batchQueueDepth,
                                        false,
                                        std::nullopt);

    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 4));
    const auto ready = std::chrono::steady_clock::now();
    IndexedLocalNamedBatchAssemblerStats stats = loader.getStatsSnapshot(ExampleType::TRAIN);

    EXPECT_GE(stats.currentReadyBatches, 4);
    EXPECT_GE(stats.batchesAssembled, 4);
    EXPECT_GE(stats.readBytesSubmitted, 4 * batchSize * layout.recordSizeBytes());
    EXPECT_DOUBLE_EQ(stats.readAmplification(), 1.0);
    EXPECT_EQ(stats.recordsCopied, 0);
    EXPECT_EQ(stats.recordCopyBytes, 0);
    EXPECT_EQ(stats.recordCopyMemcpyCalls, 0);
    EXPECT_EQ(stats.recordBufferPoolCapacity, 0);
    EXPECT_EQ(stats.recordCopyThreadCount, 0);
    EXPECT_GE(stats.readBytesCompleted, 4 * batchSize * layout.recordSizeBytes());

    const double elapsedSeconds = std::chrono::duration<double>(ready - start).count();
    const double recordsPerSecond = elapsedSeconds > 0.0 ? static_cast<double>(stats.recordsRequested) / elapsedSeconds : 0.0;
    const double logicalBytesPerSecond =
        elapsedSeconds > 0.0 ? static_cast<double>(stats.logicalRecordBytesRequested) / elapsedSeconds : 0.0;
    const double submittedBytesPerSecond = elapsedSeconds > 0.0 ? static_cast<double>(stats.readBytesSubmitted) / elapsedSeconds : 0.0;

    std::cerr << "TestIndexedNamedBatchSession perf smoke: records_per_second=" << recordsPerSecond
              << " logical_bytes_per_second=" << logicalBytesPerSecond << " submitted_bytes_per_second=" << submittedBytesPerSecond
              << " read_amplification=" << stats.readAmplification() << " planning_lead_records=" << stats.planningLeadRecords()
              << " copy_threads=" << stats.recordCopyThreadCount << " records_copied=" << stats.recordsCopied
              << " copy_bytes=" << stats.recordCopyBytes << " copy_memcpy_calls=" << stats.recordCopyMemcpyCalls
              << " avg_copy_ns_per_record=" << stats.averageCopyNanosecondsPerRecord()
              << " avg_copy_calls_per_record=" << stats.averageCopyMemcpyCallsPerRecord()
              << " record_buffer_pool=" << stats.currentRecordBufferPoolDepth << "/" << stats.recordBufferPoolCapacity
              << " ready_batches=" << stats.currentReadyBatches << " queue_depth=" << stats.targetBatchQueueDepth
              << " resolved_io_backend=" << stats.resolvedIoBackend << std::endl;

    uint64_t batchNum = 99;
    BatchLease batchLease = loader.leaseBatch(ExampleType::TRAIN, batchNum);

    const Batch& batch = batchLease.get();
    EXPECT_EQ(batchNum, 0);
    const float *values = batch.getTensor("large_features").getMemPtr<float>();
    EXPECT_FLOAT_EQ(values[0], deterministicLargeValue(0, 0));
    EXPECT_FLOAT_EQ(values[elementsPerExample], deterministicLargeValue(37, 0));
    batchLease.reset();

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchSessionTest, ThreeFoldSessionsShareResidentStorageAndOwnDistinctQueues) {
    requireCudaDevice("CUDA device is required for device batch session tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("three_device_sessions");
    const DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);
    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);

    TestIndexedNamedBatchSession source(
        dataset,
        vector<uint64_t>{0, 1, 2, 3, 4},
        vector<uint64_t>{},
        vector<uint64_t>{},
        2,
        1,
        false,
        std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(source),
        2);
    std::shared_ptr<DeviceResidentNamedDataset> resident =
        DeviceResidentNamedDataset::fromSnapshot(
            snapshot,
            TensorPlacement(TensorPlacement::MemDevices::GPU, 0));

    constexpr uint64_t maxInFlight = 3;
    auto first = std::make_shared<DeviceResidentNamedBatchSession>(
        Thor::DeviceDatasetLease(resident),
        Thor::DatasetSplitManifest(*dataset, {0, 1}, {}, vector<uint64_t>{}),
        Thor::BatchPolicy(2, false),
        maxInFlight);
    auto second = std::make_shared<DeviceResidentNamedBatchSession>(
        Thor::DeviceDatasetLease(resident),
        Thor::DatasetSplitManifest(*dataset, {2, 3}, {}, vector<uint64_t>{}),
        Thor::BatchPolicy(2, false),
        maxInFlight);
    auto third = std::make_shared<DeviceResidentNamedBatchSession>(
        Thor::DeviceDatasetLease(resident),
        Thor::DatasetSplitManifest(*dataset, {4, 0}, {}, vector<uint64_t>{}),
        Thor::BatchPolicy(2, false),
        maxInFlight);

    EXPECT_EQ(first->getDeviceDataset().get(), resident.get());
    EXPECT_EQ(second->getDeviceDataset().get(), resident.get());
    EXPECT_EQ(third->getDeviceDataset().get(), resident.get());
    EXPECT_EQ(first->getBatchQueueDepth(), maxInFlight);
    EXPECT_EQ(second->getBatchQueueDepth(), maxInFlight);
    EXPECT_EQ(third->getBatchQueueDepth(), maxInFlight);
    EXPECT_EQ(first->getStatsSnapshot(ExampleType::TRAIN).currentAvailableBatches, maxInFlight);
    EXPECT_EQ(second->getStatsSnapshot(ExampleType::TRAIN).currentAvailableBatches, maxInFlight);
    EXPECT_EQ(third->getStatsSnapshot(ExampleType::TRAIN).currentAvailableBatches, maxInFlight);

    auto takeBatch = [](const std::shared_ptr<DeviceResidentNamedBatchSession> &session) {
        uint64_t batchNum = 99;
        BatchLease batchLease = session->leaseBatch(ExampleType::TRAIN, batchNum);
        return std::make_pair(std::move(batchLease), batchNum);
    };
    auto firstFuture = std::async(std::launch::async, takeBatch, first);
    auto secondFuture = std::async(std::launch::async, takeBatch, second);
    auto thirdFuture = std::async(std::launch::async, takeBatch, third);

    auto firstResult = firstFuture.get();
    auto secondResult = secondFuture.get();
    auto thirdResult = thirdFuture.get();
    EXPECT_EQ(firstResult.second, 0u);
    EXPECT_EQ(secondResult.second, 0u);
    EXPECT_EQ(thirdResult.second, 0u);

    const Batch &firstBatch = firstResult.first.get();
    const Batch &secondBatch = secondResult.first.get();
    const Batch &thirdBatch = thirdResult.first.get();
    expectTensorValuesOnHost(firstBatch.getTensor("seasonality_inputs"), {0.0f, 1.0f, 20.0f, 21.0f});
    expectTensorValuesOnHost(secondBatch.getTensor("seasonality_inputs"), {40.0f, 41.0f, 60.0f, 61.0f});
    expectTensorValuesOnHost(thirdBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 0.0f, 1.0f});

    const uint64_t firstTensorId = firstBatch.getTensor("seasonality_inputs").getTensorId();
    const uint64_t secondTensorId = secondBatch.getTensor("seasonality_inputs").getTensorId();
    const uint64_t thirdTensorId = thirdBatch.getTensor("seasonality_inputs").getTensorId();
    EXPECT_NE(firstTensorId, secondTensorId);
    EXPECT_NE(firstTensorId, thirdTensorId);
    EXPECT_NE(secondTensorId, thirdTensorId);
    EXPECT_EQ(first->getStatsSnapshot(ExampleType::TRAIN).currentAvailableBatches, maxInFlight - 1);
    EXPECT_EQ(second->getStatsSnapshot(ExampleType::TRAIN).currentAvailableBatches, maxInFlight - 1);
    EXPECT_EQ(third->getStatsSnapshot(ExampleType::TRAIN).currentAvailableBatches, maxInFlight - 1);

    firstResult.first.reset();
    secondResult.first.reset();
    thirdResult.first.reset();
    EXPECT_EQ(first->getStatsSnapshot(ExampleType::TRAIN).currentAvailableBatches, maxInFlight);
    EXPECT_EQ(second->getStatsSnapshot(ExampleType::TRAIN).currentAvailableBatches, maxInFlight);
    EXPECT_EQ(third->getStatsSnapshot(ExampleType::TRAIN).currentAvailableBatches, maxInFlight);

    std::vector<Event> firstSessionEvents = first->getSynchronizeEvents();
    ASSERT_EQ(firstSessionEvents.size(), 1u);
    firstSessionEvents.front().synchronize();

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchSessionTest, CancellationUnblocksOnlyTheCancelledSession) {
    requireCudaDevice("CUDA device is required for device batch session cancellation tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_session_cancel");
    const DatasetLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);
    std::shared_ptr<Thor::FileDataset> dataset = Thor::FileDataset::open(datasetPath);
    TestIndexedNamedBatchSession source(
        dataset,
        vector<uint64_t>{0, 1, 2, 3},
        vector<uint64_t>{4},
        vector<uint64_t>{},
        2,
        1,
        false,
        std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(
        materializationDescription(source),
        2);
    std::shared_ptr<DeviceResidentNamedDataset> resident =
        DeviceResidentNamedDataset::fromSnapshot(
            snapshot,
            TensorPlacement(TensorPlacement::MemDevices::GPU, 0));

    auto cancelledSession = std::make_shared<DeviceResidentNamedBatchSession>(
        Thor::DeviceDatasetLease(resident),
        Thor::DatasetSplitManifest(*dataset, {0, 1, 2, 3}, {4}, vector<uint64_t>{}),
        Thor::BatchPolicy(2, false),
        1);
    auto survivingSession = std::make_shared<DeviceResidentNamedBatchSession>(
        Thor::DeviceDatasetLease(resident),
        Thor::DatasetSplitManifest(*dataset, {2, 3}, {4}, vector<uint64_t>{}),
        Thor::BatchPolicy(2, false),
        1);

    uint64_t heldBatchNum = 99;
    BatchLease heldBatchLease = cancelledSession->leaseBatch(ExampleType::TRAIN, heldBatchNum);

    const Batch& heldBatch = heldBatchLease.get();
    auto blocked = std::async(std::launch::async, [&] {
        uint64_t batchNum = 99;
        try {
            BatchLease unexpectedLease = cancelledSession->leaseBatch(ExampleType::TRAIN, batchNum);

            const Batch& unexpected = unexpectedLease.get();
            (void)unexpected;
            return false;
        } catch (const std::runtime_error &e) {
            return std::string(e.what()).find("cancelled") != std::string::npos;
        }
    });
    EXPECT_EQ(blocked.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    cancelledSession->cancel();
    EXPECT_EQ(blocked.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(blocked.get());
    EXPECT_TRUE(cancelledSession->isCancelled());

    uint64_t survivingBatchNum = 99;
    BatchLease survivingBatchLease = survivingSession->leaseBatch(ExampleType::TRAIN, survivingBatchNum);

    const Batch& survivingBatch = survivingBatchLease.get();
    EXPECT_EQ(survivingBatchNum, 0u);
    expectTensorValuesOnHost(survivingBatch.getTensor("seasonality_inputs"), {40.0f, 41.0f, 60.0f, 61.0f});
    survivingBatchLease.reset();

    // Returning an already borrowed batch after cancellation is intentionally a no-op.
    heldBatchLease.reset();
    std::filesystem::remove_all(datasetPath);
}
