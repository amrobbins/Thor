#include "DeepLearning/Api/Loaders/IndexedLocalNamedBatchLoader.h"
#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
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
