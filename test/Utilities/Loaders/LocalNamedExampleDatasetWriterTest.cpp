#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"
#include "Utilities/Loaders/Shard.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using std::map;
using std::string;
using std::vector;

namespace {

std::filesystem::path makeTempDatasetPath(const std::string &name) {
    static uint64_t counter = 0;
    std::filesystem::path path = std::filesystem::temp_directory_path() /
                                 ("thor_local_named_example_dataset_writer_" + name + "_" + std::to_string(counter++));
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

LocalNamedExampleDatasetWriter::TensorBatchView tensorBatchView(vector<float> &values, vector<uint64_t> dimensions) {
    return LocalNamedExampleDatasetWriter::TensorBatchView{.dataType = DataType::FP32,
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

vector<float> readFloats(const vector<uint8_t> &record, uint64_t offsetBytes, uint64_t count) {
    vector<float> values(count);
    std::memcpy(values.data(), record.data() + offsetBytes, count * sizeof(float));
    return values;
}

nlohmann::json readJson(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    nlohmann::json j;
    in >> j;
    return j;
}

}  // namespace

TEST(LocalNamedExampleDatasetWriterTest, WritesManifestAndShardCounts) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("manifest_and_counts");
    LocalNamedExampleLayout layout = testLayout();

    vector<float> s0{1.0f, 2.0f};
    vector<float> m0{10.0f, 11.0f, 12.0f};
    vector<float> w0{0.5f};
    vector<float> s1{3.0f, 4.0f};
    vector<float> m1{13.0f, 14.0f, 15.0f};
    vector<float> w1{0.6f};
    vector<float> s2{5.0f, 6.0f};
    vector<float> m2{16.0f, 17.0f, 18.0f};
    vector<float> w2{0.7f};
    vector<float> s3{7.0f, 8.0f};
    vector<float> m3{19.0f, 20.0f, 21.0f};
    vector<float> w3{0.8f};

    LocalNamedExampleDatasetWriter writer(datasetPath, layout, 2);
    writer.writeExample(ExampleType::TRAIN, exampleViews(s0, m0, w0));
    writer.writeExample(ExampleType::TRAIN, exampleViews(s1, m1, w1));
    writer.writeExample(ExampleType::TRAIN, exampleViews(s2, m2, w2));
    writer.writeExample(ExampleType::VALIDATE, exampleViews(s3, m3, w3));
    writer.close();

    ASSERT_TRUE(std::filesystem::exists(datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME));
    LocalNamedExampleLayout parsedLayout = LocalNamedExampleLayout::readManifest(datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME);
    EXPECT_NO_THROW(layout.validateRequestedLayoutExact(parsedLayout));

    nlohmann::json manifest = readJson(datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("format").get<string>(), LocalNamedExampleLayout::FORMAT);
    EXPECT_EQ(manifest.at("storage_mode").get<string>(), LocalNamedExampleDatasetWriter::STORAGE_MODE_SPLIT);
    EXPECT_EQ(manifest.at("data_type").get<string>(), "fp32");
    EXPECT_EQ(manifest.at("record_size_bytes").get<uint64_t>(), layout.recordSizeBytes());
    EXPECT_EQ(manifest.at("num_examples").get<uint64_t>(), 4);
    EXPECT_EQ(manifest.at("example_type_counts").at("train").get<uint64_t>(), 3);
    EXPECT_EQ(manifest.at("example_type_counts").at("validate").get<uint64_t>(), 1);
    EXPECT_EQ(manifest.at("example_type_counts").at("test").get<uint64_t>(), 0);
    ASSERT_EQ(manifest.at("shards").size(), 2);
    EXPECT_EQ(manifest.at("shards").at(0).at("num_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(1).at("num_examples").get<uint64_t>(), 2);

    Shard shard0;
    shard0.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());
    EXPECT_EQ(shard0.getExampleSizeInBytes(), layout.recordSizeBytes());
    EXPECT_EQ(shard0.getDataType(), DataType::FP32);
    EXPECT_EQ(shard0.getNumExamples(ExampleType::TRAIN), 2);
    EXPECT_EQ(shard0.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(shard0.getNumExamples(ExampleType::TEST), 0);

    Shard shard1;
    shard1.openShard((datasetPath / manifest.at("shards").at(1).at("file").get<string>()).string());
    EXPECT_EQ(shard1.getExampleSizeInBytes(), layout.recordSizeBytes());
    EXPECT_EQ(shard1.getDataType(), DataType::FP32);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::TRAIN), 1);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::VALIDATE), 1);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::TEST), 0);

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, PacksNamedTensorSlicesIntoContiguousRecord) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("packed_record");
    LocalNamedExampleLayout layout = testLayout();

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};

    LocalNamedExampleDatasetWriter writer(datasetPath, layout, 8);
    writer.writeExample(ExampleType::TRAIN, exampleViews(seasonality, monotone, weight));
    writer.close();

    nlohmann::json manifest = readJson(datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME);
    Shard shard;
    shard.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());

    vector<uint8_t> record(layout.recordSizeBytes());
    string label;
    string filename;
    shard.loadExample(record.data(), label, filename, ExampleType::TRAIN, 0);

    EXPECT_EQ(readFloats(record, layout.tensor("seasonality_inputs").offsetBytes, 2), seasonality);
    EXPECT_EQ(readFloats(record, layout.tensor("monotone_inputs").offsetBytes, 3), monotone);
    EXPECT_EQ(readFloats(record, layout.tensor("daily_weight").offsetBytes, 1), weight);

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsMissingTensor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("missing_tensor");
    LocalNamedExampleDatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    map<string, LocalNamedExampleDatasetWriter::TensorView> tensors = {{"seasonality_inputs", tensorView(seasonality, {2})},
                                                                       {"monotone_inputs", tensorView(monotone, {3})}};

    EXPECT_THROW(writer.writeExample(ExampleType::TRAIN, tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsExtraTensor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("extra_tensor");
    LocalNamedExampleDatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    vector<float> extra{99.0f};
    map<string, LocalNamedExampleDatasetWriter::TensorView> tensors = exampleViews(seasonality, monotone, weight);
    tensors["extra"] = tensorView(extra, {1});

    EXPECT_THROW(writer.writeExample(ExampleType::TRAIN, tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsShapeMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("shape_mismatch");
    LocalNamedExampleDatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    map<string, LocalNamedExampleDatasetWriter::TensorView> tensors = exampleViews(seasonality, monotone, weight);
    tensors["monotone_inputs"].dimensions = {1, 3};

    EXPECT_THROW(writer.writeExample(ExampleType::TRAIN, tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsDtypeMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("dtype_mismatch");
    LocalNamedExampleDatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    map<string, LocalNamedExampleDatasetWriter::TensorView> tensors = exampleViews(seasonality, monotone, weight);
    tensors["daily_weight"].dataType = DataType::FP16;

    EXPECT_THROW(writer.writeExample(ExampleType::TRAIN, tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsByteCountMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("byte_count_mismatch");
    LocalNamedExampleDatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    map<string, LocalNamedExampleDatasetWriter::TensorView> tensors = exampleViews(seasonality, monotone, weight);
    tensors["seasonality_inputs"].numBytes = sizeof(float);

    EXPECT_THROW(writer.writeExample(ExampleType::TRAIN, tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsNonEmptyDatasetDirectory) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("non_empty_directory");
    std::filesystem::create_directories(datasetPath);
    std::ofstream(datasetPath / "stale_file") << "stale";

    EXPECT_THROW(LocalNamedExampleDatasetWriter(datasetPath, testLayout(), 8), std::runtime_error);
    std::filesystem::remove_all(datasetPath);
}


TEST(LocalNamedExampleDatasetWriterTest, WritesIndexedManifestAndGlobalRanges) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_manifest");
    LocalNamedExampleLayout layout = testLayout();

    LocalNamedExampleDatasetWriter writer(datasetPath, layout, 2, LocalNamedExampleDatasetWriter::StorageMode::INDEXED);
    for (uint64_t i = 0; i < 5; ++i) {
        vector<float> seasonality{static_cast<float>(i), static_cast<float>(i + 1)};
        vector<float> monotone{static_cast<float>(i + 10), static_cast<float>(i + 11), static_cast<float>(i + 12)};
        vector<float> weight{static_cast<float>(i + 100)};
        writer.writeIndexedExample(exampleViews(seasonality, monotone, weight));
    }
    writer.close();

    nlohmann::json manifest = readJson(datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("storage_mode").get<string>(), LocalNamedExampleDatasetWriter::STORAGE_MODE_INDEXED);
    EXPECT_EQ(manifest.at("num_examples").get<uint64_t>(), 5);
    EXPECT_EQ(manifest.at("example_type_counts").at("train").get<uint64_t>(), 5);
    EXPECT_EQ(manifest.at("example_type_counts").at("validate").get<uint64_t>(), 0);
    EXPECT_EQ(manifest.at("example_type_counts").at("test").get<uint64_t>(), 0);
    ASSERT_EQ(manifest.at("shards").size(), 3);
    EXPECT_EQ(manifest.at("shards").at(0).at("global_start").get<uint64_t>(), 0);
    EXPECT_EQ(manifest.at("shards").at(0).at("num_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(1).at("global_start").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(1).at("num_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(2).at("global_start").get<uint64_t>(), 4);
    EXPECT_EQ(manifest.at("shards").at(2).at("num_examples").get<uint64_t>(), 1);

    Shard shard1;
    shard1.openShard((datasetPath / manifest.at("shards").at(1).at("file").get<string>()).string());
    EXPECT_EQ(shard1.getNumExamples(ExampleType::TRAIN), 2);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::TEST), 0);

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsSplitWriteInIndexedMode) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("split_write_in_indexed_mode");
    LocalNamedExampleDatasetWriter writer(datasetPath, testLayout(), 8, LocalNamedExampleDatasetWriter::StorageMode::INDEXED);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    EXPECT_THROW(writer.writeExample(ExampleType::TRAIN, exampleViews(seasonality, monotone, weight)), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsIndexedWriteInSplitMode) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_write_in_split_mode");
    LocalNamedExampleDatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    EXPECT_THROW(writer.writeIndexedExample(exampleViews(seasonality, monotone, weight)), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}


TEST(LocalNamedExampleDatasetWriterTest, WritesIndexedExamplesChunkWithExpectedCountAndCompactShards) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_chunked_preallocated");
    LocalNamedExampleLayout layout = testLayout();

    vector<float> seasonality{0.0f, 1.0f, 10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f, 40.0f, 41.0f};
    vector<float> monotone{100.0f, 101.0f, 102.0f,
                           110.0f, 111.0f, 112.0f,
                           120.0f, 121.0f, 122.0f,
                           130.0f, 131.0f, 132.0f,
                           140.0f, 141.0f, 142.0f};
    vector<float> weight{1000.0f, 1001.0f, 1002.0f, 1003.0f, 1004.0f};
    map<string, LocalNamedExampleDatasetWriter::TensorBatchView> batch = {
        {"seasonality_inputs", tensorBatchView(seasonality, {5, 2})},
        {"monotone_inputs", tensorBatchView(monotone, {5, 3})},
        {"daily_weight", tensorBatchView(weight, {5, 1})},
    };

    LocalNamedExampleDatasetWriter writer(datasetPath,
                                          layout,
                                          2,
                                          LocalNamedExampleDatasetWriter::StorageMode::INDEXED,
                                          5,
                                          true);
    EXPECT_EQ(writer.getExpectedNumExamples().value(), 5);
    EXPECT_TRUE(writer.getPreallocate());
    writer.writeIndexedExamples(batch);
    writer.close();

    nlohmann::json manifest = readJson(datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("storage_mode").get<string>(), LocalNamedExampleDatasetWriter::STORAGE_MODE_INDEXED);
    EXPECT_EQ(manifest.at("expected_num_examples").get<uint64_t>(), 5);
    EXPECT_TRUE(manifest.at("preallocated").get<bool>());
    ASSERT_EQ(manifest.at("shards").size(), 3);
    EXPECT_EQ(manifest.at("shards").at(0).at("global_start").get<uint64_t>(), 0);
    EXPECT_EQ(manifest.at("shards").at(0).at("capacity_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(0).at("num_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(2).at("capacity_examples").get<uint64_t>(), 1);
    EXPECT_EQ(manifest.at("shards").at(2).at("num_examples").get<uint64_t>(), 1);
    EXPECT_EQ(manifest.at("shards").at(2).at("num_bytes").get<uint64_t>(), layout.recordSizeBytes());

    Shard shard0;
    shard0.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());
    EXPECT_EQ(shard0.getNumExamples(ExampleType::TRAIN), 2);
    EXPECT_EQ(shard0.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(shard0.getNumExamples(ExampleType::TEST), 0);

    vector<uint8_t> record(layout.recordSizeBytes());
    string label;
    string filename;
    shard0.loadExample(record.data(), label, filename, ExampleType::TRAIN, 1);
    EXPECT_EQ(readFloats(record, layout.tensor("seasonality_inputs").offsetBytes, 2), (vector<float>{10.0f, 11.0f}));
    EXPECT_EQ(readFloats(record, layout.tensor("monotone_inputs").offsetBytes, 3), (vector<float>{110.0f, 111.0f, 112.0f}));
    EXPECT_EQ(readFloats(record, layout.tensor("daily_weight").offsetBytes, 1), (vector<float>{1001.0f}));

    Shard shard2;
    shard2.openShard((datasetPath / manifest.at("shards").at(2).at("file").get<string>()).string());
    shard2.loadExample(record.data(), label, filename, ExampleType::TRAIN, 0);
    EXPECT_EQ(readFloats(record, layout.tensor("seasonality_inputs").offsetBytes, 2), (vector<float>{40.0f, 41.0f}));

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, PreallocateRequiresExpectedNumExamples) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("preallocate_requires_expected");
    EXPECT_THROW(LocalNamedExampleDatasetWriter(datasetPath,
                                                testLayout(),
                                                8,
                                                LocalNamedExampleDatasetWriter::StorageMode::INDEXED,
                                                std::nullopt,
                                                true),
                 std::runtime_error);
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, ExpectedNumExamplesMustBeSatisfiedOnClose) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("expected_count_close");
    {
        LocalNamedExampleDatasetWriter writer(datasetPath,
                                              testLayout(),
                                              8,
                                              LocalNamedExampleDatasetWriter::StorageMode::INDEXED,
                                              2,
                                              false);

        vector<float> seasonality{1.0f, 2.0f};
        vector<float> monotone{10.0f, 11.0f, 12.0f};
        vector<float> weight{0.5f};
        writer.writeIndexedExample(exampleViews(seasonality, monotone, weight));

        EXPECT_THROW(writer.close(), std::runtime_error);
    }
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsIndexedExamplesBatchShapeMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_batch_shape_mismatch");
    LocalNamedExampleDatasetWriter writer(datasetPath, testLayout(), 8, LocalNamedExampleDatasetWriter::StorageMode::INDEXED);

    vector<float> seasonality{0.0f, 1.0f, 10.0f, 11.0f};
    vector<float> monotone{100.0f, 101.0f, 102.0f, 110.0f, 111.0f, 112.0f};
    vector<float> weight{1000.0f, 1001.0f};
    map<string, LocalNamedExampleDatasetWriter::TensorBatchView> batch = {
        {"seasonality_inputs", tensorBatchView(seasonality, {2, 2})},
        {"monotone_inputs", tensorBatchView(monotone, {2, 3})},
        {"daily_weight", tensorBatchView(weight, {1, 2})},
    };

    EXPECT_THROW(writer.writeIndexedExamples(batch), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

namespace {

LocalNamedExampleLayout windowedTestLayout() {
    return LocalNamedExampleLayout::fromTensorShapes(
        vector<std::pair<string, vector<uint64_t>>>{{"dense", {2}}},
        vector<LocalNamedExampleLayout::WindowedTensorShape>{LocalNamedExampleLayout::WindowedTensorShape(
            "history", {3, 1}, DataType::UINT64, DataType::INT32, 0.0, string("history_mask"))},
        DataType::FP32);
}

}  // namespace

TEST(LocalNamedExampleDatasetWriterTest, WritesWindowedTensorSourceAndReferencesIntoManifestAndRecords) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("windowed_source_and_refs");
    LocalNamedExampleLayout layout = windowedTestLayout();

    uint64_t key1 = 101;
    vector<float> source1{1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t key2 = 202;
    vector<float> source2{10.0f, 11.0f, 12.0f};

    vector<float> dense{5.0f, 6.0f, 7.0f, 8.0f};
    uint64_t keys[2] = {key1, key2};
    int32_t starts[2] = {11, 20};
    map<string, LocalNamedExampleDatasetWriter::TensorBatchView> tensors = {{"dense", tensorBatchView(dense, {2, 2})}};
    map<string, LocalNamedExampleDatasetWriter::WindowedTensorReferenceBatchView> refs = {
        {"history", LocalNamedExampleDatasetWriter::WindowedTensorReferenceBatchView{.keyDataType = DataType::UINT64,
                                                                                    .indexDataType = DataType::INT32,
                                                                                    .keys = keys,
                                                                                    .starts = starts,
                                                                                    .count = 2}}};

    LocalNamedExampleDatasetWriter writer(datasetPath, layout, 8, LocalNamedExampleDatasetWriter::StorageMode::INDEXED);
    writer.writeWindowedTensorSource("history",
                                     LocalNamedExampleDatasetWriter::WindowedTensorSourceView{.dataType = DataType::FP32,
                                                                                              .key = &key1,
                                                                                              .startIndex = 10,
                                                                                              .dimensions = {4, 1},
                                                                                              .data = source1.data(),
                                                                                              .numBytes = source1.size() * sizeof(float)});
    writer.writeWindowedTensorSource("history",
                                     LocalNamedExampleDatasetWriter::WindowedTensorSourceView{.dataType = DataType::FP32,
                                                                                              .key = &key2,
                                                                                              .startIndex = 20,
                                                                                              .dimensions = {3, 1},
                                                                                              .data = source2.data(),
                                                                                              .numBytes = source2.size() * sizeof(float)});
    writer.writeIndexedExamples(tensors, refs);
    writer.close();

    nlohmann::json manifest = readJson(datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME);
    ASSERT_TRUE(manifest.contains("windowed_tensors"));
    const nlohmann::json &history = manifest.at("windowed_tensors").at("history");
    EXPECT_EQ(history.at("shape").get<vector<uint64_t>>(), (vector<uint64_t>{3, 1}));
    EXPECT_EQ(history.at("key_data_type").get<string>(), "uint64");
    EXPECT_EQ(history.at("index_data_type").get<string>(), "int32");
    EXPECT_EQ(history.at("reference_offset_bytes").get<uint64_t>(), layout.windowedTensor("history").referenceOffsetBytes);
    ASSERT_TRUE(history.contains("source_storage"));
    EXPECT_EQ(history.at("source_storage").at("num_bytes").get<uint64_t>(), (source1.size() + source2.size()) * sizeof(float));
    ASSERT_EQ(history.at("source_storage").at("sequences").size(), 2);
    EXPECT_EQ(history.at("source_storage").at("sequences").at(0).at("start_index").get<int64_t>(), 10);
    EXPECT_EQ(history.at("source_storage").at("sequences").at(0).at("num_steps").get<uint64_t>(), 4);
    EXPECT_EQ(history.at("source_storage").at("sequences").at(1).at("offset_bytes").get<uint64_t>(), source1.size() * sizeof(float));

    const std::filesystem::path sourcePath = datasetPath / history.at("source_storage").at("file").get<string>();
    ASSERT_TRUE(std::filesystem::exists(sourcePath));
    std::ifstream sourceIn(sourcePath, std::ios::binary);
    vector<float> sourceValues(source1.size() + source2.size());
    sourceIn.read(reinterpret_cast<char *>(sourceValues.data()), static_cast<std::streamsize>(sourceValues.size() * sizeof(float)));
    EXPECT_EQ(sourceValues, (vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 11.0f, 12.0f}));

    Shard shard;
    shard.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());
    vector<uint8_t> record(layout.recordSizeBytes());
    string label;
    string filename;
    shard.loadExample(record.data(), label, filename, ExampleType::TRAIN, 1);
    EXPECT_EQ(readFloats(record, layout.tensor("dense").offsetBytes, 2), (vector<float>{7.0f, 8.0f}));
    uint64_t storedKey = 0;
    int32_t storedStart = 0;
    const LocalNamedExampleLayout::WindowedTensorSpec &historySpec = layout.windowedTensor("history");
    std::memcpy(&storedKey, record.data() + historySpec.referenceOffsetBytes, sizeof(storedKey));
    std::memcpy(&storedStart, record.data() + historySpec.referenceOffsetBytes + sizeof(storedKey), sizeof(storedStart));
    EXPECT_EQ(storedKey, key2);
    EXPECT_EQ(storedStart, 20);

    LocalNamedExampleLayout parsedLayout = LocalNamedExampleLayout::readManifest(datasetPath / LocalNamedExampleDatasetWriter::MANIFEST_FILENAME);
    layout.validateRequestedLayoutExact(parsedLayout);
    ASSERT_EQ(parsedLayout.windowedTensor("history").sourceSequences.size(), 2);

    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsWindowedLayoutInSplitStorageMode) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("windowed_split_rejected");
    EXPECT_THROW(LocalNamedExampleDatasetWriter(datasetPath, windowedTestLayout(), 8), std::runtime_error);
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsWindowedWriteWithoutReferences) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("windowed_missing_refs");
    LocalNamedExampleDatasetWriter writer(datasetPath,
                                          windowedTestLayout(),
                                          8,
                                          LocalNamedExampleDatasetWriter::StorageMode::INDEXED);
    vector<float> dense{5.0f, 6.0f};
    map<string, LocalNamedExampleDatasetWriter::TensorBatchView> tensors = {{"dense", tensorBatchView(dense, {1, 2})}};
    EXPECT_THROW(writer.writeIndexedExamples(tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(LocalNamedExampleDatasetWriterTest, RejectsDuplicateWindowedTensorSourceKey) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("windowed_duplicate_source_key");
    LocalNamedExampleDatasetWriter writer(datasetPath,
                                          windowedTestLayout(),
                                          8,
                                          LocalNamedExampleDatasetWriter::StorageMode::INDEXED);
    uint64_t key = 101;
    vector<float> source{1.0f, 2.0f, 3.0f};
    LocalNamedExampleDatasetWriter::WindowedTensorSourceView view{.dataType = DataType::FP32,
                                                                 .key = &key,
                                                                 .startIndex = 0,
                                                                 .dimensions = {3, 1},
                                                                 .data = source.data(),
                                                                 .numBytes = source.size() * sizeof(float)};
    writer.writeWindowedTensorSource("history", view);
    EXPECT_THROW(writer.writeWindowedTensorSource("history", view), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}
