#include "DeepLearning/Api/Loaders/LocalNamedBatchLoader.h"
#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <filesystem>
#include <map>
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
                                 ("thor_local_named_batch_loader_" + name + "_" + std::to_string(counter++));
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

void writeExample(LocalNamedExampleDatasetWriter &writer, ExampleType exampleType, float base) {
    vector<float> seasonality{base, base + 1.0f};
    vector<float> monotone{base + 10.0f, base + 11.0f, base + 12.0f};
    vector<float> weight{base + 100.0f};
    writer.writeExample(exampleType, exampleViews(seasonality, monotone, weight));
}

void writeDataset(const std::filesystem::path &datasetPath, const LocalNamedExampleLayout &layout) {
    LocalNamedExampleDatasetWriter writer(datasetPath, layout, 3);
    writeExample(writer, ExampleType::TRAIN, 0.0f);
    writeExample(writer, ExampleType::TRAIN, 20.0f);
    writeExample(writer, ExampleType::TRAIN, 40.0f);
    writeExample(writer, ExampleType::VALIDATE, 100.0f);
    writeExample(writer, ExampleType::VALIDATE, 120.0f);
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

TEST(LocalNamedBatchLoaderTest, ReadsNamedBatchesFromManifestDataset) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("read_batches");
    LocalNamedExampleLayout layout = testLayout();
    writeDataset(datasetPath, layout);

    LocalNamedBatchLoader loader(datasetPath, layout, 2, 2, false);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TRAIN), 3);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::TRAIN), 2);
    EXPECT_EQ(loader.getNumExamples(ExampleType::VALIDATE), 2);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::VALIDATE), 1);
    EXPECT_EQ(loader.getNumExamples(ExampleType::TEST), 0);
    EXPECT_EQ(loader.getNumBatchesPerEpoch(ExampleType::TEST), 0);

    uint64_t batchNum = 99;
    Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
    EXPECT_EQ(batchNum, 0);
    ASSERT_TRUE(batch.contains("seasonality_inputs"));
    ASSERT_TRUE(batch.contains("monotone_inputs"));
    ASSERT_TRUE(batch.contains("daily_weight"));
    expectTensorValues(batch.getTensor("seasonality_inputs"), {0.0f, 1.0f, 20.0f, 21.0f});
    expectTensorValues(batch.getTensor("monotone_inputs"), {10.0f, 11.0f, 12.0f, 30.0f, 31.0f, 32.0f});
    expectTensorValues(batch.getTensor("daily_weight"), {100.0f, 120.0f});
    loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch));

    Batch validateBatch = loader.getBatch(ExampleType::VALIDATE, batchNum);
    EXPECT_EQ(batchNum, 0);
    expectTensorValues(validateBatch.getTensor("seasonality_inputs"), {100.0f, 101.0f, 120.0f, 121.0f});
    loader.returnBatchBuffers(ExampleType::VALIDATE, std::move(validateBatch));

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedBatchLoaderTest, RejectsRequestedLayoutMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("layout_mismatch");
    LocalNamedExampleLayout layout = testLayout();
    writeDataset(datasetPath, layout);

    LocalNamedExampleLayout requested = LocalNamedExampleLayout::fromTensorShapes(
        vector<std::pair<string, vector<uint64_t>>>{{"seasonality_inputs", {3}}, {"monotone_inputs", {3}}, {"daily_weight", {1}}},
        DataType::FP32);
    EXPECT_THROW(LocalNamedBatchLoader(datasetPath, requested, 2, 2, false), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedBatchLoaderTest, ThrowsWhenReadingEmptySplit) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("empty_split");
    LocalNamedExampleLayout layout = testLayout();
    writeDataset(datasetPath, layout);

    LocalNamedBatchLoader loader(datasetPath, layout, 2, 2, false);
    uint64_t batchNum = 0;
    EXPECT_THROW((void)loader.getBatch(ExampleType::TEST, batchNum), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedBatchLoaderTest, RejectsReturnedBatchMissingTensor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("missing_returned_tensor");
    LocalNamedExampleLayout layout = testLayout();
    writeDataset(datasetPath, layout);

    LocalNamedBatchLoader loader(datasetPath, layout, 2, 2, false);
    uint64_t batchNum = 0;
    Batch batch = loader.getBatch(ExampleType::TRAIN, batchNum);
    batch.values().erase("daily_weight");
    EXPECT_THROW(loader.returnBatchBuffers(ExampleType::TRAIN, std::move(batch)), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedBatchLoaderTest, RejectsIndexedStorageModeDataset) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_dataset_rejected");
    LocalNamedExampleLayout layout = testLayout();

    LocalNamedExampleDatasetWriter writer(datasetPath, layout, 3, LocalNamedExampleDatasetWriter::StorageMode::INDEXED);
    vector<float> seasonality{0.0f, 1.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{100.0f};
    writer.writeIndexedExample(exampleViews(seasonality, monotone, weight));
    writer.close();

    EXPECT_THROW(LocalNamedBatchLoader(datasetPath, layout, 1, 1, false), std::runtime_error);

    std::filesystem::remove_all(datasetPath);
}
