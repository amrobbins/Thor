#include "DeepLearning/Api/Loaders/DeviceResidentNamedBatchLoader.h"
#include "DeepLearning/Api/Loaders/DeviceResidentWindowedNamedBatchLoader.h"
#include "DeepLearning/Api/Loaders/IndexedLocalNamedBatchLoader.h"
#include "DeepLearning/Api/Training/DeviceDatasetStorageSelection.h"
#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"
#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"
#include "Utilities/Loaders/NamedDatasetMaterializer.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using std::map;
using std::string;
using std::vector;
using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

std::filesystem::path makeTempDatasetPath(const std::string &name) {
    static uint64_t counter = 0;
    std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("thor_indexed_local_named_batch_loader_" + name + "_" + std::to_string(counter++));
    std::filesystem::remove_all(path);
    return path;
}

LocalNamedExampleLayout testLayout() {
    return LocalNamedExampleLayout::fromTensorShapes(
        vector<std::pair<string, vector<uint64_t>>>{{"seasonality_inputs", {2}}, {"monotone_inputs", {3}}, {"daily_weight", {1}}},
        DataType::FP32);
}

LocalNamedExampleLayout reorderedEquivalentLayout() {
    LocalNamedExampleLayout layout = testLayout();
    return LocalNamedExampleLayout(
        layout.dataType(),
        layout.recordSizeBytes(),
        vector<LocalNamedExampleLayout::TensorSpec>{
            layout.tensor("daily_weight"), layout.tensor("seasonality_inputs"), layout.tensor("monotone_inputs")});
}

LocalNamedExampleDatasetWriter::TensorView tensorView(vector<float> &values, vector<uint64_t> dimensions) {
    return LocalNamedExampleDatasetWriter::TensorView{
        .dataType = DataType::FP32, .dimensions = std::move(dimensions), .data = values.data(), .numBytes = values.size() * sizeof(float)};
}

map<string, LocalNamedExampleDatasetWriter::TensorView> exampleViews(vector<float> &seasonality,
                                                                     vector<float> &monotone,
                                                                     vector<float> &weight) {
    return {{"seasonality_inputs", tensorView(seasonality, {2})},
            {"monotone_inputs", tensorView(monotone, {3})},
            {"daily_weight", tensorView(weight, {1})}};
}

void writeIndexedExample(LocalNamedExampleDatasetWriter &writer, float base) {
    vector<float> seasonality{base, base + 1.0f};
    vector<float> monotone{base + 10.0f, base + 11.0f, base + 12.0f};
    vector<float> weight{base + 100.0f};
    writer.writeIndexedExample(exampleViews(seasonality, monotone, weight));
}

void writeCanonicalDataset(const std::filesystem::path &datasetPath, const LocalNamedExampleLayout &layout) {
    LocalNamedExampleDatasetWriter writer(datasetPath, layout, 3, LocalNamedExampleDatasetWriter::StorageMode::INDEXED);
    writeIndexedExample(writer, 0.0f);
    writeIndexedExample(writer, 20.0f);
    writeIndexedExample(writer, 40.0f);
    writeIndexedExample(writer, 60.0f);
    writeIndexedExample(writer, 80.0f);
    writer.close();
}

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

vector<float> seasonalityValues(Batch &batch) { return tensorValues(batch.getTensor("seasonality_inputs")); }

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

LocalNamedExampleLayout largeRecordLayout(uint64_t fp32Elements) {
    return LocalNamedExampleLayout::fromTensorShapes(vector<std::pair<string, vector<uint64_t>>>{{"large_features", {fp32Elements}}},
                                                     DataType::FP32);
}

LocalNamedExampleLayout materializerWindowedLayout() {
    return LocalNamedExampleLayout::fromTensorShapes(
        vector<std::pair<string, vector<uint64_t>>>{{"dense", {2}}},
        vector<LocalNamedExampleLayout::WindowedTensorShape>{LocalNamedExampleLayout::WindowedTensorShape(
            "history", {3, 1}, DataType::UINT64, DataType::INT32, 0.0, string("history_mask"))},
        DataType::FP32);
}

void writeWindowedMaterializerDataset(const std::filesystem::path &datasetPath) {
    LocalNamedExampleLayout layout = materializerWindowedLayout();
    LocalNamedExampleDatasetWriter writer(datasetPath, layout, 10, LocalNamedExampleDatasetWriter::StorageMode::INDEXED);

    uint64_t key = 7;
    vector<float> source{10.0f, 11.0f, 12.0f, 13.0f};
    writer.writeWindowedTensorSource(
        "history",
        LocalNamedExampleDatasetWriter::WindowedTensorSourceView{.dataType = DataType::FP32,
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
          LocalNamedExampleDatasetWriter::TensorBatchView{.dataType = DataType::FP32,
                                                          .dimensions = vector<uint64_t>{3, 2},
                                                          .data = dense.data(),
                                                          .numBytes = dense.size() * sizeof(float)}}},
        {{"history",
          LocalNamedExampleDatasetWriter::WindowedTensorReferenceBatchView{.keyDataType = DataType::UINT64,
                                                                          .indexDataType = DataType::INT32,
                                                                          .keys = keys.data(),
                                                                          .starts = starts.data(),
                                                                          .count = 3}}});
    writer.close();
}

void writeLargeIndexedDataset(const std::filesystem::path &datasetPath,
                              const LocalNamedExampleLayout &layout,
                              uint64_t numExamples,
                              uint64_t elementsPerExample,
                              uint64_t examplesPerShard) {
    LocalNamedExampleDatasetWriter writer(datasetPath, layout, examplesPerShard, LocalNamedExampleDatasetWriter::StorageMode::INDEXED);
    std::vector<float> values(elementsPerExample);
    for (uint64_t exampleIndex = 0; exampleIndex < numExamples; ++exampleIndex) {
        for (uint64_t elementIndex = 0; elementIndex < elementsPerExample; ++elementIndex) {
            values.at(elementIndex) = deterministicLargeValue(exampleIndex, elementIndex);
        }
        writer.writeIndexedExample({{"large_features", tensorView(values, {elementsPerExample})}});
    }
    writer.close();
}

bool waitForReadyBatches(IndexedLocalNamedBatchLoader &loader, ExampleType exampleType, uint64_t minReady);

void exerciseIndexedLoaderWithBackendOrSkip(const IndexedBackendCase &backend) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", backend.envValue);

    const std::filesystem::path datasetPath = makeTempDatasetPath(std::string("backend_") + backend.displayName);
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    try {
        IndexedLocalNamedBatchLoader loader(
            datasetPath, layout, {4, 2, 0, 3, 1}, {1, 3}, vector<uint64_t>{0, 4}, 2, 3, false, std::nullopt);

        if (!waitForReadyBatches(loader, ExampleType::TRAIN, 2)) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader did not prefill ready train batches for backend " +
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
        Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
        EXPECT_EQ(batchNum, 0);
        expectTensorValues(batch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
        loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch));

        Batch validateBatch = loader.getBatch(ExampleType::VALIDATE, batchNum);
        EXPECT_EQ(batchNum, 0);
        expectTensorValues(validateBatch.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});
        loader.returnBatchBuffers(ExampleType::VALIDATE, std::move(validateBatch));
    } catch (const std::exception &e) {
        std::filesystem::remove_all(datasetPath);
        if (isUnavailableExplicitBackend(backend, e.what())) {
            GTEST_SKIP() << "Explicit indexed local named loader backend " << backend.envValue
                         << " is unavailable in this runtime: " << e.what();
        }
        throw;
    }

    std::filesystem::remove_all(datasetPath);
}

bool waitForReadyBatches(IndexedLocalNamedBatchLoader &loader, ExampleType exampleType, uint64_t minReady) {
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
    LocalNamedExampleLayout layout = testLayout();
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

TEST(IndexedLocalNamedExampleReaderTest, AsyncReadvDrainsAndReusesIovecSlotsWhenQueueDepthIsSmall) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("reader_async_readv_queue_depth");
    LocalNamedExampleLayout layout = testLayout();
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

TEST(IndexedLocalNamedBatchLoaderTest, BindsBatchPointersByReaderOrdinalWhenRequestedLayoutOrderDiffers) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("requested_layout_reordered");
    LocalNamedExampleLayout writerLayout = testLayout();
    writeCanonicalDataset(datasetPath, writerLayout);

    LocalNamedExampleLayout requestedLayout = reorderedEquivalentLayout();
    IndexedLocalNamedBatchLoader loader(datasetPath, requestedLayout, {4, 2}, {1, 3}, std::nullopt, 2, 2, false, std::nullopt);

    uint64_t batchNum = 99;
    Batch trainBatch = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(trainBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
    expectTensorValues(trainBatch.getTensor("monotone_inputs"), {90.0f, 91.0f, 92.0f, 50.0f, 51.0f, 52.0f});
    expectTensorValues(trainBatch.getTensor("daily_weight"), {180.0f, 140.0f});
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(trainBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, ReadsFoldIndicesFromOneSharedDataset) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_read");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {4, 2, 0}, {1, 3}, std::nullopt, 2, 2, false, std::nullopt);
    EXPECT_EQ(loader.getNumDatasetExamples(), 5);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TRAIN), 3);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::TRAIN), 2);
    EXPECT_EQ(loader.getNumExamples(ExampleType::VALIDATE), 2);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TEST), 2);
    EXPECT_FALSE(loader.hasExplicitTestSplit());

    uint64_t batchNum = 99;
    Batch trainBatch = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(trainBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
    expectTensorValues(trainBatch.getTensor("monotone_inputs"), {90.0f, 91.0f, 92.0f, 50.0f, 51.0f, 52.0f});
    expectTensorValues(trainBatch.getTensor("daily_weight"), {180.0f, 140.0f});
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(trainBatch));

    Batch validateBatch = loader.getBatch(ExampleType::VALIDATE, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(validateBatch.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});
    loader.returnBatchBuffers(ExampleType::VALIDATE, std::move(validateBatch));

    Batch testBatch = loader.getBatch(ExampleType::TEST, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(testBatch.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});
    loader.returnBatchBuffers(ExampleType::TEST, std::move(testBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, SupportsEmptyValidateAndTestIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("empty_validate_test");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1, 2}, {}, vector<uint64_t>{}, 2, 2, false, std::nullopt);
    EXPECT_TRUE(loader.hasExplicitTestSplit());
    EXPECT_EQ(loader.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TEST), 0);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::TEST), 0);

    uint64_t batchNum = 0;
    EXPECT_THROW(loader.getBatch(ExampleType::VALIDATE, batchNum), std::runtime_error);
    EXPECT_THROW(loader.getBatch(ExampleType::TEST, batchNum), std::runtime_error);

    Batch trainBatch = loader.getBatch(ExampleType::TRAIN, batchNum);
    expectTensorValues(trainBatch.getTensor("seasonality_inputs"), {0.0f, 1.0f, 20.0f, 21.0f});
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(trainBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, EmptyValidateWithoutExplicitTestYieldsEmptyTest) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("empty_validate_implicit_test");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1, 2}, {}, std::nullopt, 2, 2, false, std::nullopt);
    EXPECT_FALSE(loader.hasExplicitTestSplit());
    EXPECT_EQ(loader.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TEST), 0);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::TEST), 0);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, RejectsEmptyTrainIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("empty_train");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    EXPECT_THROW((IndexedLocalNamedBatchLoader(
                     datasetPath, layout, vector<uint64_t>{}, vector<uint64_t>{}, std::nullopt, 2, 2, false, std::nullopt)),
                 std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, SupportsExplicitTestIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("explicit_test");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1}, {2}, vector<uint64_t>{4, 3}, 2, 2, false, std::nullopt);
    EXPECT_TRUE(loader.hasExplicitTestSplit());
    EXPECT_EQ(loader.getNumExamples(ExampleType::TEST), 2);

    uint64_t batchNum = 0;
    Batch testBatch = loader.getBatch(ExampleType::TEST, batchNum);
    expectTensorValues(testBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 60.0f, 61.0f});
    loader.returnBatchBuffers(ExampleType::TEST, std::move(testBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, RejectsOutOfRangeIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("out_of_range");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    EXPECT_THROW((IndexedLocalNamedBatchLoader(datasetPath, layout, {0, 5}, {1}, std::nullopt, 2, 2, false, std::nullopt)),
                 std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, DeterministicRandomizedTrainOrderForFixedSeed) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("deterministic_randomized_train");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader first(datasetPath, layout, {0, 1, 2, 3, 4}, {0}, std::nullopt, 2, 2, true, 12345);
    IndexedLocalNamedBatchLoader second(datasetPath, layout, {0, 1, 2, 3, 4}, {0}, std::nullopt, 2, 2, true, 12345);

    uint64_t firstBatchNum = 0;
    uint64_t secondBatchNum = 0;
    for (uint64_t i = 0; i < 4; ++i) {
        Batch firstBatch = first.getBatch(ExampleType::TRAIN, firstBatchNum);
        Batch secondBatch = second.getBatch(ExampleType::TRAIN, secondBatchNum);
        EXPECT_EQ(firstBatchNum, secondBatchNum);
        EXPECT_EQ(seasonalityValues(firstBatch), seasonalityValues(secondBatch));
        first.returnBatchBuffers(ExampleType::TRAIN, std::move(firstBatch));
        second.returnBatchBuffers(ExampleType::TRAIN, std::move(secondBatch));
    }

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, ValidateAndTestSplitsAreSequentialAndWrap) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("sequential_validate_test");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {4, 3}, {1, 3, 4}, vector<uint64_t>{2, 0, 1}, 2, 2, true, 9876);

    uint64_t batchNum = 99;
    Batch validate0 = loader.getBatch(ExampleType::VALIDATE, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(validate0.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});
    loader.returnBatchBuffers(ExampleType::VALIDATE, std::move(validate0));

    Batch validate1 = loader.getBatch(ExampleType::VALIDATE, batchNum);
    EXPECT_EQ(batchNum, 1);
    expectTensorValues(validate1.getTensor("seasonality_inputs"), {80.0f, 81.0f, 20.0f, 21.0f});
    loader.returnBatchBuffers(ExampleType::VALIDATE, std::move(validate1));

    Batch test0 = loader.getBatch(ExampleType::TEST, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(test0.getTensor("seasonality_inputs"), {40.0f, 41.0f, 0.0f, 1.0f});
    loader.returnBatchBuffers(ExampleType::TEST, std::move(test0));

    Batch test1 = loader.getBatch(ExampleType::TEST, batchNum);
    EXPECT_EQ(batchNum, 1);
    expectTensorValues(test1.getTensor("seasonality_inputs"), {20.0f, 21.0f, 40.0f, 41.0f});
    loader.returnBatchBuffers(ExampleType::TEST, std::move(test1));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, RejectsRequestedLayoutMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("layout_mismatch");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    LocalNamedExampleLayout wrongLayout = LocalNamedExampleLayout::fromTensorShapes(
        vector<std::pair<string, vector<uint64_t>>>{{"seasonality_inputs", {2}}, {"monotone_inputs", {4}}, {"daily_weight", {1}}},
        DataType::FP32);

    EXPECT_THROW((IndexedLocalNamedBatchLoader(datasetPath, wrongLayout, {0, 1}, {2}, std::nullopt, 2, 2, false, std::nullopt)),
                 std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, TwoFoldLoadersCanShareOneDatasetWithDifferentIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("two_fold_shared_dataset");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader foldA(datasetPath, layout, {0, 2, 4}, {1}, std::nullopt, 2, 2, false, std::nullopt);
    IndexedLocalNamedBatchLoader foldB(datasetPath, layout, {1, 3}, {0, 4}, std::nullopt, 2, 2, false, std::nullopt);

    uint64_t batchNumA = 0;
    uint64_t batchNumB = 0;
    Batch batchA = foldA.getBatch(ExampleType::TRAIN, batchNumA);
    Batch batchB = foldB.getBatch(ExampleType::TRAIN, batchNumB);

    EXPECT_EQ(batchNumA, 0);
    EXPECT_EQ(batchNumB, 0);
    expectTensorValues(batchA.getTensor("seasonality_inputs"), {0.0f, 1.0f, 40.0f, 41.0f});
    expectTensorValues(batchB.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});

    foldA.returnBatchBuffers(ExampleType::TRAIN, std::move(batchA));
    foldB.returnBatchBuffers(ExampleType::TRAIN, std::move(batchB));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, PrefillsMultipleReadyBatchesAndRecyclesReturnedBuffers) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("prefill_recycle");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1, 2, 3, 4}, {0}, std::nullopt, 1, 3, false, std::nullopt);

    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 2));
    EXPECT_LE(loader.getReadyBatchCountForTesting(ExampleType::TRAIN), 3);
    EXPECT_EQ(loader.getNextBatchNum(ExampleType::TRAIN), 0);

    uint64_t batchNum = 99;
    Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(batch.getTensor("seasonality_inputs"), {0.0f, 1.0f});

    const uint64_t readyAfterGet = loader.getReadyBatchCountForTesting(ExampleType::TRAIN);
    EXPECT_LE(readyAfterGet, 2);

    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch));
    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 2));

    Batch recycled = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 1);
    expectTensorValues(recycled.getTensor("seasonality_inputs"), {20.0f, 21.0f});
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(recycled));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, EmptyValidateAndTestSplitsHaveNoReadyBatches) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("empty_validate_test_no_ready");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1, 2}, {}, vector<uint64_t>{}, 2, 2, false, std::nullopt);

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

TEST(IndexedLocalNamedBatchLoaderTest, DefaultShardReadQueueDepthCoversFullBatchForLargeRecords) {
    ScopedUnsetEnvVar unsetPrimary("THOR_INDEXED_LOCAL_NAMED_LOADER_SHARD_READ_QUEUE_DEPTH");
    ScopedUnsetEnvVar unsetAlias("THOR_INDEXED_LOCAL_NAMED_READER_SHARD_READ_QUEUE_DEPTH");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    constexpr uint64_t elementsPerExample = 64 * 1024;  // 256 KiB records, old byte-target depth would be 32.
    constexpr uint64_t numExamples = 33;
    constexpr uint64_t batchSize = 33;

    const std::filesystem::path datasetPath = makeTempDatasetPath("default_shard_read_depth_full_batch");
    LocalNamedExampleLayout layout = largeRecordLayout(elementsPerExample);
    writeLargeIndexedDataset(datasetPath, layout, numExamples, elementsPerExample, numExamples);

    std::vector<uint64_t> trainIndices;
    trainIndices.reserve(numExamples);
    for (uint64_t i = 0; i < numExamples; ++i) {
        trainIndices.push_back(i);
    }

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, trainIndices, {}, std::nullopt, batchSize, 1, false, std::nullopt);

    IndexedLocalNamedBatchAssemblerStats stats = loader.getStatsSnapshot(ExampleType::TRAIN);
    EXPECT_EQ(stats.shardReadQueueDepth, batchSize + 10);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, ShardReadQueueDepthCanBeOverriddenByEnvironment) {
    ScopedEnvVar scopedDepth("THOR_INDEXED_LOCAL_NAMED_LOADER_SHARD_READ_QUEUE_DEPTH", "7");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("env_shard_read_depth");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1, 2, 3}, {}, std::nullopt, 4, 1, false, std::nullopt);

    IndexedLocalNamedBatchAssemblerStats stats = loader.getStatsSnapshot(ExampleType::TRAIN);
    EXPECT_EQ(stats.shardReadQueueDepth, 7);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, StatsExposeReadAndBatchCounters) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("stats_counters");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1, 2, 3, 4}, {}, vector<uint64_t>{}, 1, 3, false, std::nullopt);

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
    Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch));

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

TEST(IndexedLocalNamedBatchLoaderTest, RejectsReturnedBatchMissingTensor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("missing_returned_tensor");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1}, {2}, std::nullopt, 2, 2, false, std::nullopt);
    uint64_t batchNum = 0;
    Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
    batch.values().erase("daily_weight");
    EXPECT_THROW(loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch)), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, RejectsReturnedBatchWithExtraTensor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("extra_returned_tensor");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1}, {2}, std::nullopt, 2, 2, false, std::nullopt);
    uint64_t batchNum = 0;
    Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
    batch.insert("unexpected_tensor", batch.getTensor("daily_weight"));
    EXPECT_THROW(loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch)), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, RejectsReturnedBatchWithWrongTensorDescriptor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("wrong_returned_tensor_descriptor");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1}, {2}, std::nullopt, 2, 2, false, std::nullopt);
    uint64_t batchNum = 0;
    Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
    batch.values()["daily_weight"] = batch.getTensor("seasonality_inputs");
    EXPECT_THROW(loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch)), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, RejectsSplitStorageModeDataset) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("split_dataset_rejected");
    LocalNamedExampleLayout layout = testLayout();

    LocalNamedExampleDatasetWriter writer(datasetPath, layout, 3);
    vector<float> seasonality{0.0f, 1.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{100.0f};
    writer.writeExample(ExampleType::TRAIN, exampleViews(seasonality, monotone, weight));
    writer.close();

    EXPECT_THROW((IndexedLocalNamedBatchLoader(datasetPath, layout, {0}, {0}, std::nullopt, 1, 1, false, std::nullopt)),
                 std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}


TEST(IndexedLocalNamedBatchLoaderTest, DescribesDeviceDatasetMaterializationWithoutConsumingBatches) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("device_materialization_view");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {4, 2, 0}, {1, 3}, vector<uint64_t>{0, 4}, 2, 2, false, std::nullopt);
    ASSERT_TRUE(loader.supportsDeviceDatasetMaterialization());
    EXPECT_EQ(loader.getSplitIndices(ExampleType::TRAIN), (vector<uint64_t>{4, 2, 0}));
    EXPECT_EQ(loader.getSplitIndices(ExampleType::VALIDATE), (vector<uint64_t>{1, 3}));
    EXPECT_EQ(loader.getSplitIndices(ExampleType::TEST), (vector<uint64_t>{0, 4}));

    DeviceDatasetMaterializationView view = loader.describeDeviceDatasetMaterialization();
    EXPECT_EQ(view.datasetPath, datasetPath);
    EXPECT_NO_THROW(view.layout.validateRequestedLayoutExact(layout));
    EXPECT_EQ(view.numDatasetExamples, 5);
    EXPECT_EQ(view.batchSize, 2);
    ASSERT_EQ(view.splits.size(), 3);

    const DeviceDatasetMaterializationSplitView &train = view.split(ExampleType::TRAIN);
    EXPECT_EQ(train.splitName, "train");
    EXPECT_EQ(train.indices, (vector<uint64_t>{4, 2, 0}));
    EXPECT_EQ(train.numExamples(), 3);
    EXPECT_EQ(train.batchesPerEpoch, 2);
    EXPECT_FALSE(train.randomized);
    EXPECT_FALSE(train.seed.has_value());

    const DeviceDatasetMaterializationSplitView &validate = view.split(ExampleType::VALIDATE);
    EXPECT_EQ(validate.splitName, "validate");
    EXPECT_EQ(validate.indices, (vector<uint64_t>{1, 3}));
    EXPECT_EQ(validate.numExamples(), 2);
    EXPECT_EQ(validate.batchesPerEpoch, 1);
    EXPECT_FALSE(validate.randomized);
    EXPECT_FALSE(validate.seed.has_value());

    const DeviceDatasetMaterializationSplitView &test = view.split(ExampleType::TEST);
    EXPECT_EQ(test.splitName, "test");
    EXPECT_EQ(test.indices, (vector<uint64_t>{0, 4}));
    EXPECT_EQ(test.numExamples(), 2);
    EXPECT_EQ(test.batchesPerEpoch, 1);
    EXPECT_FALSE(test.randomized);
    EXPECT_FALSE(test.seed.has_value());

    uint64_t batchNum = 99;
    Batch trainBatch = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(trainBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(trainBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, DeviceDatasetMaterializationViewPreservesRandomizedTrainMetadata) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("device_materialization_randomized_metadata");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1, 2, 3, 4}, {}, std::nullopt, 2, 2, true, 12345);
    DeviceDatasetMaterializationView view = loader.describeDeviceDatasetMaterialization();

    const DeviceDatasetMaterializationSplitView &train = view.split(ExampleType::TRAIN);
    EXPECT_TRUE(train.randomized);
    ASSERT_TRUE(train.seed.has_value());
    EXPECT_EQ(train.seed.value(), 12345);
    EXPECT_EQ(train.indices, (vector<uint64_t>{0, 1, 2, 3, 4}));

    const DeviceDatasetMaterializationSplitView &validate = view.split(ExampleType::VALIDATE);
    EXPECT_FALSE(validate.randomized);
    EXPECT_FALSE(validate.seed.has_value());
    EXPECT_TRUE(validate.indices.empty());
    EXPECT_EQ(validate.batchesPerEpoch, 0);

    const DeviceDatasetMaterializationSplitView &test = view.split(ExampleType::TEST);
    EXPECT_FALSE(test.randomized);
    EXPECT_FALSE(test.seed.has_value());
    EXPECT_TRUE(test.indices.empty());
    EXPECT_EQ(test.batchesPerEpoch, 0);

    std::filesystem::remove_all(datasetPath);
}


TEST(NamedDatasetMaterializerTest, MaterializesDenseIndexedSplitsIntoContiguousCpuSnapshot) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_dense_snapshot");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {4, 2, 0}, {1, 3}, vector<uint64_t>{0, 4}, 2, 1, false, std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(loader.describeDeviceDatasetMaterialization(), 2);

    EXPECT_NO_THROW(snapshot.layout.validateRequestedLayoutExact(layout));
    EXPECT_EQ(snapshot.numDatasetExamples, 5);
    EXPECT_EQ(snapshot.batchSize, 2);
    EXPECT_EQ(snapshot.splits.size(), 3);
    EXPECT_EQ(snapshot.totalExamples(), 7);
    EXPECT_EQ(snapshot.totalBytes(), 7 * layout.recordSizeBytes());
    EXPECT_GE(snapshot.materializationSeconds, 0.0);

    const MaterializedNamedSplitSnapshot &train = snapshot.split(ExampleType::TRAIN);
    EXPECT_EQ(train.splitName, "train");
    EXPECT_EQ(train.sourceIndices, (vector<uint64_t>{4, 2, 0}));
    EXPECT_FALSE(train.randomized);
    EXPECT_FALSE(train.seed.has_value());
    EXPECT_EQ(train.batchesPerEpoch, 2);
    expectTensorValues(train.tensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f, 0.0f, 1.0f});
    expectTensorValues(train.tensor("monotone_inputs"), {90.0f, 91.0f, 92.0f, 50.0f, 51.0f, 52.0f, 10.0f, 11.0f, 12.0f});
    expectTensorValues(train.tensor("daily_weight"), {180.0f, 140.0f, 100.0f});

    const MaterializedNamedSplitSnapshot &validate = snapshot.split(ExampleType::VALIDATE);
    EXPECT_EQ(validate.sourceIndices, (vector<uint64_t>{1, 3}));
    EXPECT_EQ(validate.batchesPerEpoch, 1);
    expectTensorValues(validate.tensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});

    const MaterializedNamedSplitSnapshot &test = snapshot.split(ExampleType::TEST);
    EXPECT_EQ(test.sourceIndices, (vector<uint64_t>{0, 4}));
    EXPECT_EQ(test.batchesPerEpoch, 1);
    expectTensorValues(test.tensor("seasonality_inputs"), {0.0f, 1.0f, 80.0f, 81.0f});

    uint64_t batchNum = 99;
    Batch sourceBatch = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(sourceBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(sourceBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(NamedDatasetMaterializerTest, MaterializationDoesNotAdvanceLiveLoaderBatchState) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_does_not_consume_loader");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {4, 2, 0}, {1, 3}, std::nullopt, 2, 1, false, std::nullopt);
    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 1));
    const uint64_t beforeNextBatchNum = loader.getNextBatchNum(ExampleType::TRAIN);

    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(loader.describeDeviceDatasetMaterialization(), 2);
    EXPECT_EQ(snapshot.split(ExampleType::TRAIN).sourceIndices, (vector<uint64_t>{4, 2, 0}));
    EXPECT_EQ(loader.getNextBatchNum(ExampleType::TRAIN), beforeNextBatchNum);

    uint64_t batchNum = 99;
    Batch sourceBatch = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(sourceBatch.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(sourceBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(NamedDatasetMaterializerTest, PreservesEmptySplitsWithoutAllocatingZeroSizedTensors) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_empty_splits");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1, 2}, {}, vector<uint64_t>{}, 2, 1, false, std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(loader.describeDeviceDatasetMaterialization(), 2);

    EXPECT_EQ(snapshot.split(ExampleType::TRAIN).numExamples(), 3);
    EXPECT_FALSE(snapshot.split(ExampleType::TRAIN).tensors.empty());
    EXPECT_EQ(snapshot.split(ExampleType::VALIDATE).numExamples(), 0);
    EXPECT_TRUE(snapshot.split(ExampleType::VALIDATE).tensors.empty());
    EXPECT_EQ(snapshot.split(ExampleType::TEST).numExamples(), 0);
    EXPECT_TRUE(snapshot.split(ExampleType::TEST).tensors.empty());

    std::filesystem::remove_all(datasetPath);
}

TEST(NamedDatasetMaterializerTest, MaterializesWindowedLayouts) {
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("materialize_windowed");
    LocalNamedExampleLayout layout = materializerWindowedLayout();
    writeWindowedMaterializerDataset(datasetPath);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {0, 1, 2}, {}, vector<uint64_t>{}, 3, 1, false, std::nullopt);
    NamedDatasetMaterializationSupport support = checkNamedDatasetSnapshotMaterializationSupport(loader.describeDeviceDatasetMaterialization());
    EXPECT_TRUE(support.supported) << support.reason;

    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(loader.describeDeviceDatasetMaterialization(), 2);
    const MaterializedNamedSplitSnapshot &train = snapshot.split(ExampleType::TRAIN);
    expectTensorValues(train.tensor("dense"), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    expectTensorValues(train.tensor("history"), {10.0f, 11.0f, 12.0f, 0.0f, 0.0f, 10.0f, 12.0f, 13.0f, 0.0f});
    expectUint8TensorValuesOnHost(train.tensor("history_mask"), {1, 1, 1, 0, 0, 1, 1, 1, 0});

    uint64_t batchNum = 99;
    Batch sourceBatch = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(sourceBatch.getTensor("history"), {10.0f, 11.0f, 12.0f, 0.0f, 0.0f, 10.0f, 12.0f, 13.0f, 0.0f});
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(sourceBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedDatasetTest, UploadsDenseSnapshotToGpu) {
    requireCudaDevice("CUDA device is required for device-resident named dataset tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_dataset_upload");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath, layout, {4, 2, 0}, {1, 3}, vector<uint64_t>{0, 4}, 2, 1, false, std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(loader.describeDeviceDatasetMaterialization(), 2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(snapshot, TensorPlacement(TensorPlacement::MemDevices::GPU, 0));

    EXPECT_NO_THROW(resident->getLayout().validateRequestedLayoutExact(layout));
    EXPECT_EQ(resident->getBatchSize(), 2);
    EXPECT_EQ(resident->totalExamples(), 7);
    EXPECT_EQ(resident->totalBytes(), snapshot.totalBytes());
    EXPECT_GE(resident->getUploadSeconds(), 0.0);
    EXPECT_EQ(resident->getPlacement(), TensorPlacement(TensorPlacement::MemDevices::GPU, 0));

    const DeviceResidentNamedSplit &train = resident->split(ExampleType::TRAIN);
    EXPECT_EQ(train.sourceIndices, (vector<uint64_t>{4, 2, 0}));
    EXPECT_EQ(train.batchesPerEpoch, 2);
    expectTensorValuesOnHost(train.tensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f, 0.0f, 1.0f});

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchLoaderTest, WindowedBatchesMatchSourceLoader) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_windowed");
    LocalNamedExampleLayout layout = materializerWindowedLayout();
    writeWindowedMaterializerDataset(datasetPath);

    IndexedLocalNamedBatchLoader sourceLoader(datasetPath, layout, {0, 1, 2}, {}, vector<uint64_t>{}, 2, 1, false, std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(sourceLoader.describeDeviceDatasetMaterialization(), 2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(snapshot, TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    DeviceResidentNamedBatchLoader deviceLoader(resident, 1);

    uint64_t sourceBatchNum = 99;
    uint64_t deviceBatchNum = 99;
    Batch sourceBatch = sourceLoader.getBatch(ExampleType::TRAIN, sourceBatchNum);
    Batch deviceBatch = deviceLoader.getBatch(ExampleType::TRAIN, deviceBatchNum);
    EXPECT_EQ(deviceBatchNum, sourceBatchNum);
    EXPECT_EQ(tensorValuesOnHost(deviceBatch.getTensor("dense")), tensorValues(sourceBatch.getTensor("dense")));
    EXPECT_EQ(tensorValuesOnHost(deviceBatch.getTensor("history")), tensorValues(sourceBatch.getTensor("history")));
    EXPECT_EQ(uint8TensorValuesOnHost(deviceBatch.getTensor("history_mask")), uint8TensorValuesOnHost(sourceBatch.getTensor("history_mask")));

    sourceLoader.returnBatchBuffers(ExampleType::TRAIN, std::move(sourceBatch));
    deviceLoader.returnBatchBuffers(ExampleType::TRAIN, std::move(deviceBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceDatasetStorageSelection, BestEffortPrioritizesWindowedFeaturesWhenFullDatasetDoesNotFit) {
    requireCudaDevice("CUDA device is required for device dataset storage selection tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_storage_selects_windowed_hybrid");
    LocalNamedExampleLayout layout = materializerWindowedLayout();
    writeWindowedMaterializerDataset(datasetPath);

    auto sourceLoader = std::make_shared<IndexedLocalNamedBatchLoader>(
        datasetPath, layout, vector<uint64_t>{0, 1, 2}, vector<uint64_t>{}, vector<uint64_t>{}, 2, 1, false, std::nullopt);

    constexpr uint64_t twoGiB = 2ull * 1024ull * 1024ull * 1024ull;
    Thor::DeviceDatasetStorageSelection selection = Thor::selectDeviceDatasetStorageLoader(
        sourceLoader,
        Thor::DeviceDatasetStorage::BEST_EFFORT,
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
        1,
        /*availableBytesOverride=*/twoGiB + 92);

    EXPECT_TRUE(selection.report.used);
    EXPECT_EQ(selection.report.reason, "windowed_features_only");
    EXPECT_EQ(selection.report.requiredBytes, 91u);
    EXPECT_NE(selection.loader, sourceLoader);
    EXPECT_NE(std::dynamic_pointer_cast<DeviceResidentWindowedNamedBatchLoader>(selection.loader), nullptr);

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentWindowedNamedBatchLoaderTest, UsesDeviceWindowsAndCpuDirectTensors) {
    requireCudaDevice("CUDA device is required for hybrid windowed device loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_windowed_hybrid");
    LocalNamedExampleLayout layout = materializerWindowedLayout();
    writeWindowedMaterializerDataset(datasetPath);

    IndexedLocalNamedBatchLoader sourceLoader(datasetPath, layout, {0, 1, 2}, {}, vector<uint64_t>{}, 2, 1, false, std::nullopt);
    DeviceDatasetMaterializationView view = sourceLoader.describeDeviceDatasetMaterialization();
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(view, 2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(snapshot,
                                                            TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
                                                            std::set<string>{"history", "history_mask"});
    DeviceResidentWindowedNamedBatchLoader deviceLoader(view, resident, 1);

    uint64_t sourceBatchNum = 99;
    uint64_t deviceBatchNum = 99;
    Batch sourceBatch = sourceLoader.getBatch(ExampleType::TRAIN, sourceBatchNum);
    Batch deviceBatch = deviceLoader.getBatch(ExampleType::TRAIN, deviceBatchNum);
    EXPECT_EQ(deviceBatchNum, sourceBatchNum);
    EXPECT_EQ(deviceBatch.getTensor("dense").getPlacement().getMemDevice(), TensorPlacement::MemDevices::CPU);
    EXPECT_EQ(deviceBatch.getTensor("history").getPlacement().getMemDevice(), TensorPlacement::MemDevices::GPU);
    EXPECT_EQ(tensorValuesOnHost(deviceBatch.getTensor("dense")), tensorValues(sourceBatch.getTensor("dense")));
    EXPECT_EQ(tensorValuesOnHost(deviceBatch.getTensor("history")), tensorValues(sourceBatch.getTensor("history")));
    EXPECT_EQ(uint8TensorValuesOnHost(deviceBatch.getTensor("history_mask")), uint8TensorValuesOnHost(sourceBatch.getTensor("history_mask")));

    sourceLoader.returnBatchBuffers(ExampleType::TRAIN, std::move(sourceBatch));
    deviceLoader.returnBatchBuffers(ExampleType::TRAIN, std::move(deviceBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchLoaderTest, SequentialBatchesMatchSourceLoaderAndWrap) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_sequential");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader sourceLoader(datasetPath, layout, {4, 2, 0}, {1, 3}, vector<uint64_t>{0, 4}, 2, 1, false, std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(sourceLoader.describeDeviceDatasetMaterialization(), 2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(snapshot, TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    DeviceResidentNamedBatchLoader deviceLoader(resident, 2);

    EXPECT_EQ(deviceLoader.getNumExamples(ExampleType::TRAIN), 3);
    EXPECT_EQ(deviceLoader.getNumBatchesPerEpoch(ExampleType::TRAIN), 2);
    EXPECT_EQ(deviceLoader.getNextBatchNum(ExampleType::TRAIN), 0);

    uint64_t batchNum = 99;
    Batch batch0 = deviceLoader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValuesOnHost(batch0.getTensor("seasonality_inputs"), {80.0f, 81.0f, 40.0f, 41.0f});
    EXPECT_EQ(deviceLoader.getNextBatchNum(ExampleType::TRAIN), 1);
    deviceLoader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch0));

    Batch batch1 = deviceLoader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 1);
    expectTensorValuesOnHost(batch1.getTensor("seasonality_inputs"), {0.0f, 1.0f, 80.0f, 81.0f});
    EXPECT_EQ(deviceLoader.getNextBatchNum(ExampleType::TRAIN), 0);
    deviceLoader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch1));

    Batch batch2 = deviceLoader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValuesOnHost(batch2.getTensor("seasonality_inputs"), {40.0f, 41.0f, 0.0f, 1.0f});
    deviceLoader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch2));

    DeviceResidentNamedBatchLoaderStats stats = deviceLoader.getStatsSnapshot(ExampleType::TRAIN);
    EXPECT_EQ(stats.splitName, "train");
    EXPECT_EQ(stats.batchesGathered, 3);
    EXPECT_EQ(stats.batchesReturned, 3);
    EXPECT_EQ(stats.currentAvailableBatches, 2);
    EXPECT_EQ(stats.batchQueueDepth, 2);
    EXPECT_EQ(stats.residentExamples, 3);
    EXPECT_EQ(stats.residentBytes, resident->split(ExampleType::TRAIN).totalBytes());

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchLoaderTest, ValidateAndTestSplitsAreSequentialAndIndependent) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_splits");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader sourceLoader(datasetPath, layout, {4, 3}, {1, 3, 4}, vector<uint64_t>{2, 0, 1}, 2, 1, true, 9876);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(sourceLoader.describeDeviceDatasetMaterialization(), 2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(snapshot, TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    DeviceResidentNamedBatchLoader deviceLoader(resident, 1);

    uint64_t batchNum = 99;
    Batch validateBatch0 = deviceLoader.getBatch(ExampleType::VALIDATE, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValuesOnHost(validateBatch0.getTensor("seasonality_inputs"), {20.0f, 21.0f, 60.0f, 61.0f});
    deviceLoader.returnBatchBuffers(ExampleType::VALIDATE, std::move(validateBatch0));

    Batch testBatch0 = deviceLoader.getBatch(ExampleType::TEST, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValuesOnHost(testBatch0.getTensor("seasonality_inputs"), {40.0f, 41.0f, 0.0f, 1.0f});
    deviceLoader.returnBatchBuffers(ExampleType::TEST, std::move(testBatch0));

    Batch validateBatch1 = deviceLoader.getBatch(ExampleType::VALIDATE, batchNum);
    EXPECT_EQ(batchNum, 1);
    expectTensorValuesOnHost(validateBatch1.getTensor("seasonality_inputs"), {80.0f, 81.0f, 20.0f, 21.0f});
    deviceLoader.returnBatchBuffers(ExampleType::VALIDATE, std::move(validateBatch1));

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchLoaderTest, RandomizedTrainOrderMatchesSourceLoaderForFixedSeed) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_randomized");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader sourceLoader(datasetPath, layout, {0, 1, 2, 3, 4}, {0}, std::nullopt, 2, 2, true, 12345);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(sourceLoader.describeDeviceDatasetMaterialization(), 2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(snapshot, TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    DeviceResidentNamedBatchLoader deviceLoader(resident, 2);

    for (uint64_t i = 0; i < 3; ++i) {
        uint64_t sourceBatchNum = 99;
        uint64_t deviceBatchNum = 99;
        Batch sourceBatch = sourceLoader.getBatch(ExampleType::TRAIN, sourceBatchNum);
        Batch deviceBatch = deviceLoader.getBatch(ExampleType::TRAIN, deviceBatchNum);
        EXPECT_EQ(deviceBatchNum, sourceBatchNum);
        EXPECT_EQ(tensorValuesOnHost(deviceBatch.getTensor("seasonality_inputs")), tensorValues(sourceBatch.getTensor("seasonality_inputs")))
            << "i=" << i;
        sourceLoader.returnBatchBuffers(ExampleType::TRAIN, std::move(sourceBatch));
        deviceLoader.returnBatchBuffers(ExampleType::TRAIN, std::move(deviceBatch));
    }

    std::filesystem::remove_all(datasetPath);
}

TEST(DeviceResidentNamedBatchLoaderTest, RejectsEmptySplitBatchRequest) {
    requireCudaDevice("CUDA device is required for device-resident named loader tests.");
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", "pread_buffered");

    const std::filesystem::path datasetPath = makeTempDatasetPath("device_resident_loader_empty_split");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader sourceLoader(datasetPath, layout, {0, 1, 2}, {}, vector<uint64_t>{}, 2, 1, false, std::nullopt);
    MaterializedNamedDatasetSnapshot snapshot = materializeNamedDatasetSnapshot(sourceLoader.describeDeviceDatasetMaterialization(), 2);
    auto resident = DeviceResidentNamedDataset::fromSnapshot(snapshot, TensorPlacement(TensorPlacement::MemDevices::GPU, 0));
    DeviceResidentNamedBatchLoader deviceLoader(resident, 1);

    EXPECT_EQ(deviceLoader.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(deviceLoader.getNumBatchesPerEpoch(ExampleType::VALIDATE), 0);
    uint64_t batchNum = 99;
    EXPECT_THROW((void)deviceLoader.getBatch(ExampleType::VALIDATE, batchNum), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

class IndexedLocalNamedBatchLoaderBackendRegressionTest : public ::testing::TestWithParam<IndexedBackendCase> {};

TEST_P(IndexedLocalNamedBatchLoaderBackendRegressionTest, ExplicitBackendReadsIndexedDatasetAndReportsBackend) {
    exerciseIndexedLoaderWithBackendOrSkip(GetParam());
}

INSTANTIATE_TEST_SUITE_P(ExplicitBackends,
                         IndexedLocalNamedBatchLoaderBackendRegressionTest,
                         ::testing::Values(IndexedBackendCase{"pread_buffered", "pread_buffered"},
                                           IndexedBackendCase{"pread_direct", "pread_direct"},
                                           IndexedBackendCase{"uring_direct", "uring_direct"}),
                         [](const ::testing::TestParamInfo<IndexedLocalNamedBatchLoaderBackendRegressionTest::ParamType> &info) {
                             return std::string(info.param.displayName);
                         });

TEST(IndexedLocalNamedBatchLoaderPerf, LargeRandomRecordPrefetchSmoke) {
    const char *enabled = std::getenv("THOR_INDEXED_LOCAL_NAMED_LOADER_PERF_TEST");
    if (enabled == nullptr || enabled[0] == '\0' || (enabled[0] == '0' && enabled[1] == '\0')) {
        GTEST_SKIP() << "Set THOR_INDEXED_LOCAL_NAMED_LOADER_PERF_TEST=1 to run the indexed loader large-record smoke test.";
    }

    constexpr uint64_t elementsPerExample = 32 * 1024;  // 128 KiB records.
    constexpr uint64_t numExamples = 256;
    constexpr uint64_t batchSize = 16;
    constexpr uint64_t batchQueueDepth = 8;

    const std::filesystem::path datasetPath = makeTempDatasetPath("large_random_prefetch_smoke");
    LocalNamedExampleLayout layout = largeRecordLayout(elementsPerExample);
    writeLargeIndexedDataset(datasetPath, layout, numExamples, elementsPerExample, 64);

    const std::vector<uint64_t> trainIndices = permutedIndices(numExamples, 37);
    const auto start = std::chrono::steady_clock::now();
    IndexedLocalNamedBatchLoader loader(datasetPath,
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

    std::cerr << "IndexedLocalNamedBatchLoader perf smoke: records_per_second=" << recordsPerSecond
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
    Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    const float *values = batch.getTensor("large_features").getMemPtr<float>();
    EXPECT_FLOAT_EQ(values[0], deterministicLargeValue(0, 0));
    EXPECT_FLOAT_EQ(values[elementsPerExample], deterministicLargeValue(37, 0));
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch));

    std::filesystem::remove_all(datasetPath);
}
