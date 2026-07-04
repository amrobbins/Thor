#include "DeepLearning/Api/Loaders/IndexedLocalNamedBatchLoader.h"
#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using std::map;
using std::string;
using std::vector;

namespace {

std::filesystem::path makeTempDatasetPath(const std::string &name) {
    static uint64_t counter = 0;
    std::filesystem::path path = std::filesystem::temp_directory_path() /
                                 ("thor_indexed_local_named_batch_loader_" + name + "_" + std::to_string(counter++));
    std::filesystem::remove_all(path);
    return path;
}

LocalNamedExampleLayout testLayout() {
    return LocalNamedExampleLayout::fromTensorShapes(
        vector<std::pair<string, vector<uint64_t>>>{{"seasonality_inputs", {2}}, {"monotone_inputs", {3}}, {"daily_weight", {1}}},
        DataType::FP32);
}

LocalNamedExampleDatasetWriter::TensorView tensorView(vector<float> &values, vector<uint64_t> dimensions) {
    return LocalNamedExampleDatasetWriter::TensorView{.dataType = DataType::FP32,
                                                      .dimensions = std::move(dimensions),
                                                      .data = values.data(),
                                                      .numBytes = values.size() * sizeof(float)};
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

bool isUnavailableExplicitBackend(const IndexedBackendCase &backend, const std::string &message) {
    const std::string envValue(backend.envValue);
    if (envValue == "uring_direct") {
        return message.find("io_uring_queue_init") != std::string::npos ||
               message.find("io_uring_register_files") != std::string::npos ||
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
        IndexedLocalNamedBatchLoader loader(datasetPath,
                                            layout,
                                            {4, 2, 0, 3, 1},
                                            {1, 3},
                                            vector<uint64_t>{0, 4},
                                            2,
                                            3,
                                            false,
                                            std::nullopt);

        if (!waitForReadyBatches(loader, ExampleType::TRAIN, 2)) {
            throw std::runtime_error("IndexedLocalNamedBatchLoader did not prefill ready train batches for backend " +
                                     std::string(backend.envValue) + ".");
        }
        IndexedLocalNamedBatchAssemblerStats stats = loader.getStatsSnapshot(ExampleType::TRAIN);
        EXPECT_EQ(stats.resolvedIoBackend, backend.envValue);
        EXPECT_GE(stats.recordsRequested, 4);
        EXPECT_GE(stats.readBytesSubmitted, 4 * layout.recordSizeBytes());
        EXPECT_DOUBLE_EQ(stats.readAmplification(), 1.0);
        EXPECT_GE(stats.recordsCopied, 4);
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

TEST(IndexedLocalNamedBatchLoaderTest, ReadsFoldIndicesFromOneSharedDataset) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_read");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath,
                                        layout,
                                        {4, 2, 0},
                                        {1, 3},
                                        std::nullopt,
                                        2,
                                        2,
                                        false,
                                        std::nullopt);
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

    IndexedLocalNamedBatchLoader loader(datasetPath,
                                        layout,
                                        {0, 1, 2},
                                        {},
                                        vector<uint64_t>{},
                                        2,
                                        2,
                                        false,
                                        std::nullopt);
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

    IndexedLocalNamedBatchLoader loader(datasetPath,
                                        layout,
                                        {0, 1, 2},
                                        {},
                                        std::nullopt,
                                        2,
                                        2,
                                        false,
                                        std::nullopt);
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

    EXPECT_THROW((IndexedLocalNamedBatchLoader(datasetPath,
                                               layout,
                                               vector<uint64_t>{},
                                               vector<uint64_t>{},
                                               std::nullopt,
                                               2,
                                               2,
                                               false,
                                               std::nullopt)),
                 std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, SupportsExplicitTestIndices) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("explicit_test");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath,
                                        layout,
                                        {0, 1},
                                        {2},
                                        vector<uint64_t>{4, 3},
                                        2,
                                        2,
                                        false,
                                        std::nullopt);
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

    IndexedLocalNamedBatchLoader first(datasetPath,
                                       layout,
                                       {0, 1, 2, 3, 4},
                                       {0},
                                       std::nullopt,
                                       2,
                                       2,
                                       true,
                                       12345);
    IndexedLocalNamedBatchLoader second(datasetPath,
                                        layout,
                                        {0, 1, 2, 3, 4},
                                        {0},
                                        std::nullopt,
                                        2,
                                        2,
                                        true,
                                        12345);

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

    IndexedLocalNamedBatchLoader loader(datasetPath,
                                        layout,
                                        {4, 3},
                                        {1, 3, 4},
                                        vector<uint64_t>{2, 0, 1},
                                        2,
                                        2,
                                        true,
                                        9876);

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

    IndexedLocalNamedBatchLoader loader(datasetPath,
                                        layout,
                                        {0, 1, 2, 3, 4},
                                        {0},
                                        std::nullopt,
                                        1,
                                        3,
                                        false,
                                        std::nullopt);

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

    IndexedLocalNamedBatchLoader loader(datasetPath,
                                        layout,
                                        {0, 1, 2},
                                        {},
                                        vector<uint64_t>{},
                                        2,
                                        2,
                                        false,
                                        std::nullopt);

    EXPECT_EQ(loader.getReadyBatchCountForTesting(ExampleType::VALIDATE), 0);
    EXPECT_EQ(loader.getReadyBatchCountForTesting(ExampleType::TEST), 0);
    ASSERT_TRUE(waitForReadyBatches(loader, ExampleType::TRAIN, 1));

    std::filesystem::remove_all(datasetPath);
}

TEST(IndexedLocalNamedBatchLoaderTest, StatsExposeReadAndBatchCounters) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("stats_counters");
    LocalNamedExampleLayout layout = testLayout();
    writeCanonicalDataset(datasetPath, layout);

    IndexedLocalNamedBatchLoader loader(datasetPath,
                                        layout,
                                        {0, 1, 2, 3, 4},
                                        {},
                                        vector<uint64_t>{},
                                        1,
                                        3,
                                        false,
                                        std::nullopt);

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
    EXPECT_GE(stats.recordsCopied, 2);
    EXPECT_GE(stats.recordCopyBytes, 2 * layout.recordSizeBytes());
    EXPECT_EQ(stats.recordCopyBytes % layout.recordSizeBytes(), 0);
    EXPECT_EQ(stats.recordCopyMemcpyCalls, 0);
    EXPECT_GE(stats.recordCopyActiveNanoseconds, 0);
    EXPECT_GE(stats.recordCopyPopWaitNanoseconds, 0);
    EXPECT_GE(stats.completedRecordQueuePushWaitNanoseconds, 0);
    EXPECT_GE(stats.copiedRecordQueuePushWaitNanoseconds, 0);
    EXPECT_EQ(stats.recordBufferPoolCapacity, 0);
    EXPECT_EQ(stats.currentRecordBufferPoolDepth, 0);
    EXPECT_DOUBLE_EQ(stats.averageCopyBytesPerRecord(), static_cast<double>(layout.recordSizeBytes()));
    EXPECT_DOUBLE_EQ(stats.averageCopyMemcpyCallsPerRecord(), 0.0);
    EXPECT_GE(stats.batchesAssembled, 2);
    EXPECT_EQ(stats.batchesDelivered, 0);
    EXPECT_EQ(stats.batchBuffersReturned, 0);
    EXPECT_GE(stats.currentReadyBatches, 2);
    EXPECT_EQ(stats.targetBatchQueueDepth, 3);
    EXPECT_FALSE(stats.resolvedIoBackend.empty());

    uint64_t batchNum = 99;
    Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch));

    stats = loader.getStatsSnapshot(ExampleType::TRAIN);
    EXPECT_GE(stats.batchesDelivered, 1);
    EXPECT_GE(stats.batchBuffersReturned, 1);

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
    EXPECT_GE(stats.recordsCopied, 4 * batchSize);
    EXPECT_GE(stats.recordCopyBytes, 4 * batchSize * layout.recordSizeBytes());
    EXPECT_EQ(stats.recordCopyMemcpyCalls, 0);
    EXPECT_EQ(stats.recordBufferPoolCapacity, 0);
    EXPECT_EQ(stats.recordCopyThreadCount, 0);
    EXPECT_GE(stats.readBytesCompleted, 4 * batchSize * layout.recordSizeBytes());

    const double elapsedSeconds = std::chrono::duration<double>(ready - start).count();
    const double recordsPerSecond = elapsedSeconds > 0.0 ? static_cast<double>(stats.recordsRequested) / elapsedSeconds : 0.0;
    const double logicalBytesPerSecond = elapsedSeconds > 0.0 ? static_cast<double>(stats.logicalRecordBytesRequested) / elapsedSeconds : 0.0;
    const double submittedBytesPerSecond = elapsedSeconds > 0.0 ? static_cast<double>(stats.readBytesSubmitted) / elapsedSeconds : 0.0;

    std::cerr << "IndexedLocalNamedBatchLoader perf smoke: records_per_second=" << recordsPerSecond
              << " logical_bytes_per_second=" << logicalBytesPerSecond
              << " submitted_bytes_per_second=" << submittedBytesPerSecond
              << " read_amplification=" << stats.readAmplification()
              << " planning_lead_records=" << stats.planningLeadRecords()
              << " copy_threads=" << stats.recordCopyThreadCount
              << " records_copied=" << stats.recordsCopied
              << " copy_bytes=" << stats.recordCopyBytes
              << " copy_memcpy_calls=" << stats.recordCopyMemcpyCalls
              << " avg_copy_ns_per_record=" << stats.averageCopyNanosecondsPerRecord()
              << " avg_copy_calls_per_record=" << stats.averageCopyMemcpyCallsPerRecord()
              << " record_buffer_pool=" << stats.currentRecordBufferPoolDepth << "/" << stats.recordBufferPoolCapacity
              << " ready_batches=" << stats.currentReadyBatches
              << " queue_depth=" << stats.targetBatchQueueDepth
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
