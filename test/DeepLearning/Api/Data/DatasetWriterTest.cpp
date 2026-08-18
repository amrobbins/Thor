#include "DeepLearning/Api/Data/DatasetWriter.h"
#include "Utilities/Data/Storage/DatasetShard.h"
#include "Utilities/Data/Readers/IndexedDatasetReader.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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
                                 ("thor_dataset_writer_" + name + "_" + std::to_string(counter++));
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

DatasetWriter::TensorView tensorView(vector<float> &values, vector<uint64_t> dimensions) {
    return DatasetWriter::TensorView{.dataType = DataType::FP32,
                                                      .dimensions = std::move(dimensions),
                                                      .data = values.data(),
                                                      .numBytes = values.size() * sizeof(float)};
}

DatasetWriter::TensorBatchView tensorBatchView(vector<float> &values, vector<uint64_t> dimensions) {
    return DatasetWriter::TensorBatchView{.dataType = DataType::FP32,
                                                           .dimensions = std::move(dimensions),
                                                           .data = values.data(),
                                                           .numBytes = values.size() * sizeof(float)};
}

map<string, DatasetWriter::TensorView> exampleViews(vector<float> &seasonality,
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

TEST(DatasetWriterTest, WritesIndexedManifestAndShardCounts) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("manifest_and_counts");
    DatasetLayout layout = testLayout();

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

    DatasetWriter writer(datasetPath, layout, 2);
    writer.writeIndexedExample(exampleViews(s0, m0, w0));
    writer.writeIndexedExample(exampleViews(s1, m1, w1));
    writer.writeIndexedExample(exampleViews(s2, m2, w2));
    writer.writeIndexedExample(exampleViews(s3, m3, w3));
    writer.close();

    ASSERT_TRUE(std::filesystem::exists(datasetPath / DatasetWriter::MANIFEST_FILENAME));
    DatasetLayout parsedLayout = DatasetLayout::readManifest(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_NO_THROW(layout.validateRequestedLayoutExact(parsedLayout));

    nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("format").get<string>(), DatasetLayout::FORMAT);
    EXPECT_EQ(manifest.at("storage_mode").get<string>(), DatasetWriter::STORAGE_MODE_INDEXED);
    EXPECT_FALSE(manifest.contains("data_type"));
    EXPECT_EQ(manifest.at("tensors").at("seasonality_inputs").at("data_type").get<string>(), "fp32");
    EXPECT_EQ(manifest.at("record_size_bytes").get<uint64_t>(), layout.recordSizeBytes());
    EXPECT_EQ(manifest.at("num_examples").get<uint64_t>(), 4);
    EXPECT_FALSE(manifest.contains("example_type_counts"));
    ASSERT_EQ(manifest.at("shards").size(), 2);
    EXPECT_FALSE(manifest.at("shards").at(0).contains("example_type_counts"));
    EXPECT_FALSE(manifest.at("shards").at(1).contains("example_type_counts"));
    EXPECT_EQ(manifest.at("shards").at(0).at("num_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(1).at("num_examples").get<uint64_t>(), 2);

    DatasetShard shard0;
    shard0.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());
    EXPECT_EQ(shard0.getExampleSizeInBytes(), layout.recordSizeBytes());
    EXPECT_EQ(shard0.getDataType(), DataType::UINT8);
    EXPECT_EQ(shard0.getNumExamples(ExampleType::TRAIN), 2);
    EXPECT_EQ(shard0.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(shard0.getNumExamples(ExampleType::TEST), 0);

    DatasetShard shard1;
    shard1.openShard((datasetPath / manifest.at("shards").at(1).at("file").get<string>()).string());
    EXPECT_EQ(shard1.getExampleSizeInBytes(), layout.recordSizeBytes());
    EXPECT_EQ(shard1.getDataType(), DataType::UINT8);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::TRAIN), 2);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::TEST), 0);

    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, PacksNamedTensorSlicesIntoContiguousRecord) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("packed_record");
    DatasetLayout layout = testLayout();

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};

    DatasetWriter writer(datasetPath, layout, 8);
    writer.writeIndexedExample(exampleViews(seasonality, monotone, weight));
    writer.close();

    nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    DatasetShard shard;
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

TEST(DatasetWriterTest, PacksIndependentFieldDtypesIntoOneByteRecord) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("mixed_dtypes");
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{
            DatasetLayout::TensorShape("token", {1}, DataType::UINT8),
            DatasetLayout::TensorShape("features", {2}, DataType::FP16),
            DatasetLayout::TensorShape("target", {1}, DataType::FP64),
        });

    uint8_t token = 7;
    uint16_t features[2] = {0x3e00, 0x4100};
    double target = 12.5;
    map<string, DatasetWriter::TensorView> tensors = {
        {"token", DatasetWriter::TensorView{.dataType = DataType::UINT8,
                                            .dimensions = {1},
                                            .data = &token,
                                            .numBytes = sizeof(token)}},
        {"features", DatasetWriter::TensorView{.dataType = DataType::FP16,
                                               .dimensions = {2},
                                               .data = features,
                                               .numBytes = sizeof(features)}},
        {"target", DatasetWriter::TensorView{.dataType = DataType::FP64,
                                             .dimensions = {1},
                                             .data = &target,
                                             .numBytes = sizeof(target)}},
    };

    DatasetWriter writer(datasetPath, layout, 8);
    writer.writeIndexedExample(tensors);
    writer.close();

    nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("record_size_bytes").get<uint64_t>(), 13);
    EXPECT_EQ(manifest.at("tensors").at("token").at("data_type").get<string>(), "uint8");
    EXPECT_EQ(manifest.at("tensors").at("features").at("data_type").get<string>(), "fp16");
    EXPECT_EQ(manifest.at("tensors").at("target").at("data_type").get<string>(), "fp64");

    DatasetShard shard;
    shard.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());
    EXPECT_EQ(shard.getDataType(), DataType::UINT8);
    vector<uint8_t> record(layout.recordSizeBytes());
    string label;
    string filename;
    shard.loadExample(record.data(), label, filename, ExampleType::TRAIN, 0);
    EXPECT_EQ(record.at(layout.tensor("token").offsetBytes), token);
    EXPECT_EQ(std::memcmp(record.data() + layout.tensor("features").offsetBytes, features, sizeof(features)), 0);
    double storedTarget = 0.0;
    std::memcpy(&storedTarget, record.data() + layout.tensor("target").offsetBytes, sizeof(storedTarget));
    EXPECT_DOUBLE_EQ(storedTarget, target);

    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsMissingTensor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("missing_tensor");
    DatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    map<string, DatasetWriter::TensorView> tensors = {{"seasonality_inputs", tensorView(seasonality, {2})},
                                                                       {"monotone_inputs", tensorView(monotone, {3})}};

    EXPECT_THROW(writer.writeIndexedExample(tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsExtraTensor) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("extra_tensor");
    DatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    vector<float> extra{99.0f};
    map<string, DatasetWriter::TensorView> tensors = exampleViews(seasonality, monotone, weight);
    tensors["extra"] = tensorView(extra, {1});

    EXPECT_THROW(writer.writeIndexedExample(tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsShapeMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("shape_mismatch");
    DatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    map<string, DatasetWriter::TensorView> tensors = exampleViews(seasonality, monotone, weight);
    tensors["monotone_inputs"].dimensions = {1, 3};

    EXPECT_THROW(writer.writeIndexedExample(tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsDtypeMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("dtype_mismatch");
    DatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    map<string, DatasetWriter::TensorView> tensors = exampleViews(seasonality, monotone, weight);
    tensors["daily_weight"].dataType = DataType::FP16;

    EXPECT_THROW(writer.writeIndexedExample(tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsByteCountMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("byte_count_mismatch");
    DatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{1.0f, 2.0f};
    vector<float> monotone{10.0f, 11.0f, 12.0f};
    vector<float> weight{0.5f};
    map<string, DatasetWriter::TensorView> tensors = exampleViews(seasonality, monotone, weight);
    tensors["seasonality_inputs"].numBytes = sizeof(float);

    EXPECT_THROW(writer.writeIndexedExample(tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsNonEmptyDatasetDirectory) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("non_empty_directory");
    std::filesystem::create_directories(datasetPath);
    std::ofstream(datasetPath / "stale_file") << "stale";

    EXPECT_THROW(DatasetWriter(datasetPath, testLayout(), 8), std::runtime_error);
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, WritesIndexedManifestAndGlobalRanges) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_manifest");
    DatasetLayout layout = testLayout();

    DatasetWriter writer(datasetPath, layout, 2);
    for (uint64_t i = 0; i < 5; ++i) {
        vector<float> seasonality{static_cast<float>(i), static_cast<float>(i + 1)};
        vector<float> monotone{static_cast<float>(i + 10), static_cast<float>(i + 11), static_cast<float>(i + 12)};
        vector<float> weight{static_cast<float>(i + 100)};
        writer.writeIndexedExample(exampleViews(seasonality, monotone, weight));
    }
    writer.close();

    nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("storage_mode").get<string>(), DatasetWriter::STORAGE_MODE_INDEXED);
    EXPECT_EQ(manifest.at("num_examples").get<uint64_t>(), 5);
    EXPECT_FALSE(manifest.contains("example_type_counts"));
    ASSERT_EQ(manifest.at("shards").size(), 3);
    EXPECT_EQ(manifest.at("shards").at(0).at("global_start").get<uint64_t>(), 0);
    EXPECT_EQ(manifest.at("shards").at(0).at("num_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(1).at("global_start").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(1).at("num_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(2).at("global_start").get<uint64_t>(), 4);
    EXPECT_EQ(manifest.at("shards").at(2).at("num_examples").get<uint64_t>(), 1);

    DatasetShard shard1;
    shard1.openShard((datasetPath / manifest.at("shards").at(1).at("file").get<string>()).string());
    EXPECT_EQ(shard1.getNumExamples(ExampleType::TRAIN), 2);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::VALIDATE), 0);
    EXPECT_EQ(shard1.getNumExamples(ExampleType::TEST), 0);

    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, WritesIndexedExamplesChunkWithExpectedCountAndCompactShards) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_chunked_preallocated");
    DatasetLayout layout = testLayout();

    vector<float> seasonality{0.0f, 1.0f, 10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f, 40.0f, 41.0f};
    vector<float> monotone{100.0f, 101.0f, 102.0f,
                           110.0f, 111.0f, 112.0f,
                           120.0f, 121.0f, 122.0f,
                           130.0f, 131.0f, 132.0f,
                           140.0f, 141.0f, 142.0f};
    vector<float> weight{1000.0f, 1001.0f, 1002.0f, 1003.0f, 1004.0f};
    map<string, DatasetWriter::TensorBatchView> batch = {
        {"seasonality_inputs", tensorBatchView(seasonality, {5, 2})},
        {"monotone_inputs", tensorBatchView(monotone, {5, 3})},
        {"daily_weight", tensorBatchView(weight, {5, 1})},
    };

    DatasetWriter writer(datasetPath,
                                          layout,
                                          2,
                                          5,
                                          true);
    EXPECT_EQ(writer.getExpectedNumExamples().value(), 5);
    EXPECT_TRUE(writer.getPreallocate());
    writer.writeIndexedExamples(batch);
    writer.close();

    nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("storage_mode").get<string>(), DatasetWriter::STORAGE_MODE_INDEXED);
    EXPECT_EQ(manifest.at("expected_num_examples").get<uint64_t>(), 5);
    EXPECT_TRUE(manifest.at("preallocated").get<bool>());
    ASSERT_EQ(manifest.at("shards").size(), 3);
    EXPECT_EQ(manifest.at("shards").at(0).at("global_start").get<uint64_t>(), 0);
    EXPECT_EQ(manifest.at("shards").at(0).at("capacity_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(0).at("num_examples").get<uint64_t>(), 2);
    EXPECT_EQ(manifest.at("shards").at(2).at("capacity_examples").get<uint64_t>(), 1);
    EXPECT_EQ(manifest.at("shards").at(2).at("num_examples").get<uint64_t>(), 1);
    EXPECT_EQ(manifest.at("shards").at(2).at("num_bytes").get<uint64_t>(), layout.recordSizeBytes());

    DatasetShard shard0;
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

    DatasetShard shard2;
    shard2.openShard((datasetPath / manifest.at("shards").at(2).at("file").get<string>()).string());
    shard2.loadExample(record.data(), label, filename, ExampleType::TRAIN, 0);
    EXPECT_EQ(readFloats(record, layout.tensor("seasonality_inputs").offsetBytes, 2), (vector<float>{40.0f, 41.0f}));

    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, PreallocateRequiresExpectedNumExamples) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("preallocate_requires_expected");
    EXPECT_THROW(DatasetWriter(datasetPath,
                                                testLayout(),
                                                8,
                                                std::nullopt,
                                                true),
                 std::runtime_error);
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, ExpectedNumExamplesMustBeSatisfiedOnClose) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("expected_count_close");
    {
        DatasetWriter writer(datasetPath,
                                              testLayout(),
                                              8,
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

TEST(DatasetWriterTest, RejectsIndexedExamplesBatchShapeMismatch) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("indexed_batch_shape_mismatch");
    DatasetWriter writer(datasetPath, testLayout(), 8);

    vector<float> seasonality{0.0f, 1.0f, 10.0f, 11.0f};
    vector<float> monotone{100.0f, 101.0f, 102.0f, 110.0f, 111.0f, 112.0f};
    vector<float> weight{1000.0f, 1001.0f};
    map<string, DatasetWriter::TensorBatchView> batch = {
        {"seasonality_inputs", tensorBatchView(seasonality, {2, 2})},
        {"monotone_inputs", tensorBatchView(monotone, {2, 3})},
        {"daily_weight", tensorBatchView(weight, {1, 2})},
    };

    EXPECT_THROW(writer.writeIndexedExamples(batch), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

namespace {

DatasetLayout windowedTestLayout() {
    return DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("dense", {2}, DataType::FP32)},
        vector<DatasetLayout::WindowedTensorSourceShape>{DatasetLayout::WindowedTensorSourceShape(
            "history_source", {1}, DataType::FP32, DataType::UINT64)},
        vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
            "history", {3, 1}, "history_source", DataType::INT32, 0.0, string("history_mask"))});
}

DatasetLayout affineWindowedTestLayout() {
    return DatasetLayout::fromTensorShapes(
        {},
        vector<DatasetLayout::WindowedTensorSourceShape>{DatasetLayout::WindowedTensorSourceShape(
            "history_source", {1}, DataType::FP32, DataType::UINT64)},
        vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
            "history",
            {3, 1},
            "history_source",
            DataType::INT64,
            0.0,
            string("history_mask"),
            DatasetLayout::WindowedTensorReferenceMode::AFFINE)});
}

}  // namespace

TEST(DatasetWriterTest, WritesWindowedTensorSourceAndReferencesIntoManifestAndRecords) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("windowed_source_and_refs");
    DatasetLayout layout = windowedTestLayout();

    uint64_t key1 = 101;
    vector<float> source1{1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t key2 = 202;
    vector<float> source2{10.0f, 11.0f, 12.0f};

    vector<float> dense{5.0f, 6.0f, 7.0f, 8.0f};
    uint64_t keys[2] = {key1, key2};
    int32_t starts[2] = {11, 20};
    map<string, DatasetWriter::TensorBatchView> tensors = {{"dense", tensorBatchView(dense, {2, 2})}};
    map<string, DatasetWriter::WindowedTensorReferenceBatchView> refs = {
        {"history", DatasetWriter::WindowedTensorReferenceBatchView{.keyDataType = DataType::UINT64,
                                                                                    .indexDataType = DataType::INT32,
                                                                                    .keys = keys,
                                                                                    .starts = starts,
                                                                                    .count = 2}}};

    DatasetWriter writer(datasetPath, layout, 8);
    writer.writeWindowSource("history_source",
                                     DatasetWriter::WindowedTensorSourceView{.dataType = DataType::FP32,
                                                                                              .key = &key1,
                                                                                              .startIndex = 10,
                                                                                              .dimensions = {4, 1},
                                                                                              .data = source1.data(),
                                                                                              .numBytes = source1.size() * sizeof(float)});
    writer.writeWindowSource("history_source",
                                     DatasetWriter::WindowedTensorSourceView{.dataType = DataType::FP32,
                                                                                              .key = &key2,
                                                                                              .startIndex = 20,
                                                                                              .dimensions = {3, 1},
                                                                                              .data = source2.data(),
                                                                                              .numBytes = source2.size() * sizeof(float)});
    writer.writeIndexedExamples(tensors, refs);
    writer.close();

    nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    ASSERT_TRUE(manifest.contains("windowed_tensors"));
    const nlohmann::json &history = manifest.at("windowed_tensors").at("history");
    EXPECT_EQ(history.at("shape").get<vector<uint64_t>>(), (vector<uint64_t>{3, 1}));
    EXPECT_EQ(history.at("source").get<string>(), "history_source");
    EXPECT_EQ(history.at("index_data_type").get<string>(), "int32");
    EXPECT_EQ(history.at("reference_offset_bytes").get<uint64_t>(), layout.windowedTensor("history").referenceOffsetBytes);
    const nlohmann::json &sourceStorage = manifest.at("window_sources").at("history_source").at("storage");
    EXPECT_EQ(sourceStorage.at("num_bytes").get<uint64_t>(), (source1.size() + source2.size()) * sizeof(float));
    ASSERT_EQ(sourceStorage.at("sequences").size(), 2);
    EXPECT_EQ(sourceStorage.at("sequences").at(0).at("start_index").get<int64_t>(), 10);
    EXPECT_EQ(sourceStorage.at("sequences").at(0).at("num_steps").get<uint64_t>(), 4);
    EXPECT_EQ(sourceStorage.at("sequences").at(1).at("offset_bytes").get<uint64_t>(), source1.size() * sizeof(float));

    const std::filesystem::path sourcePath = datasetPath / sourceStorage.at("file").get<string>();
    ASSERT_TRUE(std::filesystem::exists(sourcePath));
    std::ifstream sourceIn(sourcePath, std::ios::binary);
    vector<float> sourceValues(source1.size() + source2.size());
    sourceIn.read(reinterpret_cast<char *>(sourceValues.data()), static_cast<std::streamsize>(sourceValues.size() * sizeof(float)));
    EXPECT_EQ(sourceValues, (vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 11.0f, 12.0f}));

    DatasetShard shard;
    shard.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());
    vector<uint8_t> record(layout.recordSizeBytes());
    string label;
    string filename;
    shard.loadExample(record.data(), label, filename, ExampleType::TRAIN, 1);
    EXPECT_EQ(readFloats(record, layout.tensor("dense").offsetBytes, 2), (vector<float>{7.0f, 8.0f}));
    uint64_t storedKey = 0;
    int32_t storedStart = 0;
    const DatasetLayout::WindowedTensorSpec &historySpec = layout.windowedTensor("history");
    std::memcpy(&storedKey, record.data() + historySpec.referenceOffsetBytes, sizeof(storedKey));
    std::memcpy(&storedStart, record.data() + historySpec.referenceOffsetBytes + sizeof(storedKey), sizeof(storedStart));
    EXPECT_EQ(storedKey, key2);
    EXPECT_EQ(storedStart, 20);

    DatasetLayout parsedLayout = DatasetLayout::readManifest(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    layout.validateRequestedLayoutExact(parsedLayout);
    ASSERT_EQ(parsedLayout.windowedTensorSource("history_source").sourceSequences.size(), 2);

    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsWindowedWriteWithoutReferences) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("windowed_missing_refs");
    DatasetWriter writer(datasetPath,
                                          windowedTestLayout(),
                                          8);
    vector<float> dense{5.0f, 6.0f};
    map<string, DatasetWriter::TensorBatchView> tensors = {{"dense", tensorBatchView(dense, {1, 2})}};
    EXPECT_THROW(writer.writeIndexedExamples(tensors), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsDuplicateWindowedTensorSourceKey) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("windowed_duplicate_source_key");
    DatasetWriter writer(datasetPath,
                                          windowedTestLayout(),
                                          8);
    uint64_t key = 101;
    vector<float> source{1.0f, 2.0f, 3.0f};
    DatasetWriter::WindowedTensorSourceView view{.dataType = DataType::FP32,
                                                                 .key = &key,
                                                                 .startIndex = 0,
                                                                 .dimensions = {3, 1},
                                                                 .data = source.data(),
                                                                 .numBytes = source.size() * sizeof(float)};
    writer.writeWindowSource("history_source", view);
    EXPECT_THROW(writer.writeWindowSource("history_source", view), std::runtime_error);
    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, WritesCompactAffineReferenceSegmentsWithoutIndexedRecords) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("affine_reference_segments");
    const DatasetLayout layout = affineWindowedTestLayout();
    uint64_t key = 101;
    vector<float> source{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    DatasetWriter writer(datasetPath, layout, 8, 4);
    writer.writeWindowSource("history_source",
                             DatasetWriter::WindowedTensorSourceView{.dataType = DataType::FP32,
                                                                     .key = &key,
                                                                     .startIndex = 0,
                                                                     .dimensions = {9, 1},
                                                                     .data = source.data(),
                                                                     .numBytes = source.size() * sizeof(float)});
    writer.writeAffineExamples(
        2,
        {},
        {{"history",
          DatasetWriter::AffineWindowedTensorReferenceView{.keyDataType = DataType::UINT64,
                                                            .key = &key,
                                                            .base = 1,
                                                            .stride = 2,
                                                            .fieldOffset = -1}}});
    writer.writeAffineExamples(
        2,
        {},
        {{"history",
          DatasetWriter::AffineWindowedTensorReferenceView{.keyDataType = DataType::UINT64,
                                                            .key = &key,
                                                            .base = 5,
                                                            .stride = 2,
                                                            .fieldOffset = -1}}});
    writer.close();

    const nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("format").get<string>(), "thor.dataset.v1");
    EXPECT_EQ(manifest.at("record_size_bytes").get<uint64_t>(), 0);
    EXPECT_EQ(manifest.at("num_examples").get<uint64_t>(), 4);
    EXPECT_TRUE(manifest.at("shards").empty());
    ASSERT_EQ(manifest.at("affine_window_reference_segments").size(), 1);
    const nlohmann::json &segment = manifest.at("affine_window_reference_segments").at(0);
    EXPECT_EQ(segment.at("row_start").get<uint64_t>(), 0);
    EXPECT_EQ(segment.at("count").get<uint64_t>(), 4);
    const nlohmann::json &reference = segment.at("references").at("history");
    EXPECT_EQ(reference.at("base").get<int64_t>(), 1);
    EXPECT_EQ(reference.at("stride").get<int64_t>(), 2);
    EXPECT_EQ(reference.at("field_offset").get<int64_t>(), -1);

    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsAffineReferenceFormulaOverflowAtEitherEndpoint) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("affine_reference_overflow");
    uint64_t key = 101;
    DatasetWriter writer(datasetPath, affineWindowedTestLayout(), 8);

    EXPECT_THROW(
        writer.writeAffineExamples(
            2,
            {},
            {{"history",
              DatasetWriter::AffineWindowedTensorReferenceView{.keyDataType = DataType::UINT64,
                                                                .key = &key,
                                                                .base = std::numeric_limits<int64_t>::min(),
                                                                .stride = 1,
                                                                .fieldOffset = -1}}}),
        std::runtime_error);
    EXPECT_THROW(
        writer.writeAffineExamples(
            2,
            {},
            {{"history",
              DatasetWriter::AffineWindowedTensorReferenceView{.keyDataType = DataType::UINT64,
                                                                .key = &key,
                                                                .base = std::numeric_limits<int64_t>::max(),
                                                                .stride = 1,
                                                                .fieldOffset = 0}}}),
        std::runtime_error);

    writer.close();
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, WritesPackedRaggedSidecarAndPerRecordReferences) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("ragged_sidecar");
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("weight", {1}, DataType::FP32)},
        vector<DatasetLayout::RaggedTensorShape>{DatasetLayout::RaggedTensorShape("labels", {}, DataType::INT32)});

    vector<float> weights{1.0f, 2.0f, 3.0f};
    vector<int32_t> labels{10, 11, 20, 21, 22};
    vector<uint64_t> offsets{0, 2, 2, 5};

    DatasetWriter writer(datasetPath, layout, 8);
    writer.writeIndexedExamples(
        {{"weight", DatasetWriter::TensorBatchView{.dataType = DataType::FP32,
                                                    .dimensions = {3, 1},
                                                    .data = weights.data(),
                                                    .numBytes = weights.size() * sizeof(float)}}},
        {{"labels", DatasetWriter::RaggedTensorBatchView{.dataType = DataType::INT32,
                                                         .dimensions = {5},
                                                         .data = labels.data(),
                                                         .numBytes = labels.size() * sizeof(int32_t),
                                                         .offsetsDataType = DataType::UINT64,
                                                         .offsets = offsets.data(),
                                                         .count = 3}}});

    float finalWeight = 4.0f;
    vector<int32_t> finalLabels{30, 31};
    writer.writeIndexedExample(
        {{"weight", DatasetWriter::TensorView{.dataType = DataType::FP32,
                                               .dimensions = {1},
                                               .data = &finalWeight,
                                               .numBytes = sizeof(finalWeight)}}},
        {{"labels", DatasetWriter::RaggedTensorView{.dataType = DataType::INT32,
                                                    .dimensions = {2},
                                                    .data = finalLabels.data(),
                                                    .numBytes = finalLabels.size() * sizeof(int32_t)}}});
    writer.close();

    const nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("format").get<string>(), DatasetLayout::FORMAT);
    ASSERT_TRUE(manifest.contains("ragged_tensors"));
    const nlohmann::json &storage = manifest.at("ragged_tensors").at("labels").at("storage");
    EXPECT_EQ(storage.at("num_values").get<uint64_t>(), 7);
    EXPECT_EQ(storage.at("num_bytes").get<uint64_t>(), 7 * sizeof(int32_t));
    const std::filesystem::path valuesPath = datasetPath / storage.at("file").get<string>();
    ASSERT_EQ(std::filesystem::file_size(valuesPath), 7 * sizeof(int32_t));

    vector<int32_t> storedLabels(7);
    {
        std::ifstream in(valuesPath, std::ios::binary);
        in.read(reinterpret_cast<char *>(storedLabels.data()), static_cast<std::streamsize>(storedLabels.size() * sizeof(int32_t)));
        ASSERT_TRUE(in.good() || in.eof());
    }
    EXPECT_EQ(storedLabels, vector<int32_t>({10, 11, 20, 21, 22, 30, 31}));

    DatasetShard shard;
    shard.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());
    const DatasetLayout parsed = DatasetLayout::fromJson(manifest);
    const DatasetLayout::RaggedTensorSpec &spec = parsed.raggedTensor("labels");
    vector<std::pair<uint64_t, uint64_t>> expectedReferences{{0, 2}, {2, 0}, {2, 3}, {5, 2}};
    for (uint64_t row = 0; row < expectedReferences.size(); ++row) {
        vector<uint8_t> record(parsed.recordSizeBytes());
        string label;
        string filename;
        shard.loadExample(record.data(), label, filename, ExampleType::TRAIN, row);
        uint64_t startValue = 0;
        uint64_t valueCount = 0;
        std::memcpy(&startValue, record.data() + spec.referenceOffsetBytes, sizeof(startValue));
        std::memcpy(&valueCount, record.data() + spec.referenceOffsetBytes + sizeof(startValue), sizeof(valueCount));
        EXPECT_EQ(startValue, expectedReferences.at(row).first);
        EXPECT_EQ(valueCount, expectedReferences.at(row).second);
    }

    EXPECT_NO_THROW(layout.validateRequestedLayoutExact(parsed));
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RaggedBatchSupportsUint32OffsetsAndAllEmptyRows) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("ragged_empty_rows");
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        {}, vector<DatasetLayout::RaggedTensorShape>{DatasetLayout::RaggedTensorShape("tokens", {2}, DataType::FP32)});
    vector<uint32_t> offsets{0, 0, 0, 0};

    DatasetWriter writer(datasetPath, layout, 8);
    writer.writeIndexedExamples(
        {},
        {{"tokens", DatasetWriter::RaggedTensorBatchView{.dataType = DataType::FP32,
                                                         .dimensions = {0, 2},
                                                         .data = nullptr,
                                                         .numBytes = 0,
                                                         .offsetsDataType = DataType::UINT32,
                                                         .offsets = offsets.data(),
                                                         .count = 3}}});
    writer.close();

    const nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    EXPECT_EQ(manifest.at("num_examples").get<uint64_t>(), 3);
    const nlohmann::json &storage = manifest.at("ragged_tensors").at("tokens").at("storage");
    EXPECT_EQ(storage.at("num_values").get<uint64_t>(), 0);
    EXPECT_EQ(storage.at("num_bytes").get<uint64_t>(), 0);
    EXPECT_TRUE(std::filesystem::exists(datasetPath / storage.at("file").get<string>()));

    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RejectsMalformedRaggedBatchBeforeWritingSidecar) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("ragged_bad_offsets");
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        {}, vector<DatasetLayout::RaggedTensorShape>{DatasetLayout::RaggedTensorShape("labels", {}, DataType::INT32)});
    vector<int32_t> values{1, 2, 3};
    vector<uint64_t> badStart{1, 2, 3};
    vector<uint64_t> decreasing{0, 3, 2};
    vector<uint64_t> badFinal{0, 1, 2};

    DatasetWriter writer(datasetPath, layout, 8);
    auto makeView = [&](const vector<uint64_t> &offsets) {
        return DatasetWriter::RaggedTensorBatchView{.dataType = DataType::INT32,
                                                    .dimensions = {3},
                                                    .data = values.data(),
                                                    .numBytes = values.size() * sizeof(int32_t),
                                                    .offsetsDataType = DataType::UINT64,
                                                    .offsets = offsets.data(),
                                                    .count = 2};
    };
    EXPECT_THROW(writer.writeIndexedExamples({}, {{"labels", makeView(badStart)}}), std::runtime_error);
    EXPECT_THROW(writer.writeIndexedExamples({}, {{"labels", makeView(decreasing)}}), std::runtime_error);
    EXPECT_THROW(writer.writeIndexedExamples({}, {{"labels", makeView(badFinal)}}), std::runtime_error);
    EXPECT_EQ(writer.numExamples(), 0);
    writer.close();

    const nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    const nlohmann::json &storage = manifest.at("ragged_tensors").at("labels").at("storage");
    EXPECT_EQ(storage.at("num_values").get<uint64_t>(), 0);
    EXPECT_EQ(storage.at("num_bytes").get<uint64_t>(), 0);
    EXPECT_EQ(std::filesystem::file_size(datasetPath / storage.at("file").get<string>()), 0);

    std::filesystem::remove_all(datasetPath);
}


TEST(DatasetWriterTest, RaggedWindowedTensorStoresCompactReferencesIntoSharedWindowSource) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("ragged_windowed_source");
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        {},
        vector<DatasetLayout::WindowedTensorSourceShape>{
            DatasetLayout::WindowedTensorSourceShape("history_source", {2}, DataType::FP32, DataType::UINT64)},
        {},
        {},
        vector<DatasetLayout::RaggedWindowedTensorShape>{
            DatasetLayout::RaggedWindowedTensorShape("history", "history_source", DataType::INT64)});

    EXPECT_EQ(layout.recordSizeBytes(), 2 * sizeof(uint64_t));
    EXPECT_TRUE(layout.hasRaggedTensors());
    EXPECT_FALSE(layout.hasWindowedTensors());
    ASSERT_TRUE(layout.raggedTensor("history").sourceName.has_value());
    EXPECT_EQ(*layout.raggedTensor("history").sourceName, "history_source");

    uint64_t sourceKey = 7;
    vector<float> sourceValues(20);
    for (uint64_t i = 0; i < sourceValues.size(); ++i) sourceValues[i] = static_cast<float>(i);
    DatasetWriter writer(datasetPath, layout, 8, 3, true);
    writer.writeWindowSource(
        "history_source",
        DatasetWriter::WindowedTensorSourceView{
            .dataType = DataType::FP32,
            .key = &sourceKey,
            .startIndex = 100,
            .dimensions = {10, 2},
            .data = sourceValues.data(),
            .numBytes = sourceValues.size() * sizeof(float)});

    vector<uint64_t> keys{7, 7, 7};
    vector<int64_t> starts{100, 102, 107};
    vector<uint64_t> lengths{4, 3, 2};
    writer.writeIndexedExamples(
        {},
        std::map<std::string, DatasetWriter::RaggedTensorBatchView>{
            {"history", DatasetWriter::RaggedTensorBatchView{
                            .dataType = DataType::FP32,
                            .count = 3,
                            .storageMode = DatasetWriter::RaggedTensorBatchView::StorageMode::WINDOW_REFERENCE,
                            .keyDataType = DataType::UINT64,
                            .indexDataType = DataType::INT64,
                            .keys = keys.data(),
                            .starts = starts.data(),
                            .lengths = lengths.data()}}});
    writer.close();

    const nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    const nlohmann::json &sourceStorage = manifest.at("window_sources").at("history_source").at("storage");
    const nlohmann::json &raggedStorage = manifest.at("ragged_tensors").at("history").at("storage");
    EXPECT_EQ(raggedStorage.at("file").get<string>(), sourceStorage.at("file").get<string>());
    EXPECT_EQ(raggedStorage.at("num_bytes").get<uint64_t>(), sourceValues.size() * sizeof(float));
    EXPECT_EQ(raggedStorage.at("num_values").get<uint64_t>(), 10u);
    EXPECT_FALSE(std::filesystem::exists(datasetPath / "ragged_values"));

    DatasetShard shard;
    shard.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());
    const DatasetLayout parsed = DatasetLayout::fromJson(manifest);
    const DatasetLayout::RaggedTensorSpec &spec = parsed.raggedTensor("history");
    const vector<std::pair<uint64_t, uint64_t>> expected{{0, 4}, {2, 3}, {7, 2}};
    for (uint64_t row = 0; row < expected.size(); ++row) {
        vector<uint8_t> record(parsed.recordSizeBytes());
        string label;
        string filename;
        shard.loadExample(record.data(), label, filename, ExampleType::TRAIN, row);
        uint64_t startValue = 0;
        uint64_t valueCount = 0;
        std::memcpy(&startValue, record.data() + spec.referenceOffsetBytes, sizeof(startValue));
        std::memcpy(&valueCount, record.data() + spec.referenceOffsetBytes + sizeof(startValue), sizeof(valueCount));
        EXPECT_EQ(startValue, expected.at(row).first);
        EXPECT_EQ(valueCount, expected.at(row).second);
    }

    std::shared_ptr<IndexedDatasetReader> reader = IndexedDatasetReader::openDataset(datasetPath, layout);
    vector<IndexedRaggedTensorReference> references(1);
    vector<IndexedRaggedTensorReference *> destinations(reader->getRaggedTensorCount(), nullptr);
    destinations.at(reader->getLayoutRaggedTensorOrdinal("history")) = references.data();
    std::unique_ptr<IndexedDatasetReader::Session> session = reader->createSession(1);
    session->loadExampleInto(1, 0, {}, {}, {}, destinations);
    session->drain();
    EXPECT_EQ(references.at(0).startValue, 2u);
    EXPECT_EQ(references.at(0).valueCount, 3u);
    vector<float> packed(6, -1.0f);
    session->loadRaggedValuesInto(0, 2, 3, packed.data());
    EXPECT_EQ(packed, (vector<float>{4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}));

    EXPECT_NO_THROW(layout.validateRequestedLayoutExact(parsed));
    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RaggedWindowedTensorResolvesMultipleKeysIntoOneSharedSourceFile) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("ragged_windowed_multiple_keys");
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        {},
        vector<DatasetLayout::WindowedTensorSourceShape>{
            DatasetLayout::WindowedTensorSourceShape("history_source", {}, DataType::INT32, DataType::UINT64)},
        {},
        {},
        vector<DatasetLayout::RaggedWindowedTensorShape>{
            DatasetLayout::RaggedWindowedTensorShape("history", "history_source", DataType::INT64)});

    DatasetWriter writer(datasetPath, layout, 8, 2, true);
    uint64_t firstKey = 11;
    uint64_t secondKey = 22;
    vector<int32_t> firstValues{100, 101, 102};
    vector<int32_t> secondValues{200, 201, 202, 203};
    writer.writeWindowSource(
        "history_source",
        DatasetWriter::WindowedTensorSourceView{
            .dataType = DataType::INT32,
            .key = &firstKey,
            .startIndex = 10,
            .dimensions = {firstValues.size()},
            .data = firstValues.data(),
            .numBytes = firstValues.size() * sizeof(int32_t)});
    writer.writeWindowSource(
        "history_source",
        DatasetWriter::WindowedTensorSourceView{
            .dataType = DataType::INT32,
            .key = &secondKey,
            .startIndex = 50,
            .dimensions = {secondValues.size()},
            .data = secondValues.data(),
            .numBytes = secondValues.size() * sizeof(int32_t)});

    vector<uint64_t> keys{firstKey, secondKey};
    vector<int64_t> starts{11, 51};
    vector<uint64_t> lengths{2, 2};
    writer.writeIndexedExamples(
        {},
        {{"history", DatasetWriter::RaggedTensorBatchView{
                         .dataType = DataType::INT32,
                         .count = 2,
                         .storageMode = DatasetWriter::RaggedTensorBatchView::StorageMode::WINDOW_REFERENCE,
                         .keyDataType = DataType::UINT64,
                         .indexDataType = DataType::INT64,
                         .keys = keys.data(),
                         .starts = starts.data(),
                         .lengths = lengths.data()}}});
    writer.close();

    const nlohmann::json manifest = readJson(datasetPath / DatasetWriter::MANIFEST_FILENAME);
    const DatasetLayout parsed = DatasetLayout::fromJson(manifest);
    const DatasetLayout::RaggedTensorSpec &spec = parsed.raggedTensor("history");
    DatasetShard shard;
    shard.openShard((datasetPath / manifest.at("shards").at(0).at("file").get<string>()).string());
    const vector<std::pair<uint64_t, uint64_t>> expected{{1, 2}, {4, 2}};
    for (uint64_t row = 0; row < expected.size(); ++row) {
        vector<uint8_t> record(parsed.recordSizeBytes());
        string label;
        string filename;
        shard.loadExample(record.data(), label, filename, ExampleType::TRAIN, row);
        uint64_t startValue = 0;
        uint64_t valueCount = 0;
        std::memcpy(&startValue, record.data() + spec.referenceOffsetBytes, sizeof(startValue));
        std::memcpy(&valueCount,
                    record.data() + spec.referenceOffsetBytes + sizeof(startValue),
                    sizeof(valueCount));
        EXPECT_EQ(startValue, expected.at(row).first);
        EXPECT_EQ(valueCount, expected.at(row).second);
    }

    std::shared_ptr<IndexedDatasetReader> reader = IndexedDatasetReader::openDataset(datasetPath, layout);
    std::unique_ptr<IndexedDatasetReader::Session> session = reader->createSession(1);
    vector<int32_t> values(2, -1);
    session->loadRaggedValuesInto(0, 4, 2, values.data());
    EXPECT_EQ(values, (vector<int32_t>{201, 202}));

    std::filesystem::remove_all(datasetPath);
}

TEST(DatasetWriterTest, RaggedWindowedTensorRejectsMissingKeyAndOutOfBoundsWindow) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("ragged_windowed_bounds");
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        {},
        vector<DatasetLayout::WindowedTensorSourceShape>{
            DatasetLayout::WindowedTensorSourceShape("tokens", {}, DataType::INT32, DataType::UINT64)},
        {},
        {},
        vector<DatasetLayout::RaggedWindowedTensorShape>{
            DatasetLayout::RaggedWindowedTensorShape("tokens", "tokens", DataType::INT64)});
    DatasetWriter writer(datasetPath, layout, 8);
    uint64_t sourceKey = 3;
    vector<int32_t> values{10, 11, 12, 13, 14};
    writer.writeWindowSource(
        "tokens",
        DatasetWriter::WindowedTensorSourceView{
            .dataType = DataType::INT32,
            .key = &sourceKey,
            .startIndex = 10,
            .dimensions = {5},
            .data = values.data(),
            .numBytes = values.size() * sizeof(int32_t)});

    uint64_t missingKey = 4;
    int64_t start = 10;
    uint64_t length = 1;
    EXPECT_THROW(
        writer.writeIndexedExample(
            {},
            std::map<std::string, DatasetWriter::RaggedTensorView>{
                {"tokens", DatasetWriter::RaggedTensorView{
                               .dataType = DataType::INT32,
                               .storageMode = DatasetWriter::RaggedTensorView::StorageMode::WINDOW_REFERENCE,
                               .keyDataType = DataType::UINT64,
                               .indexDataType = DataType::INT64,
                               .key = &missingKey,
                               .start = &start,
                               .length = &length}}}),
        std::runtime_error);

    start = 13;
    length = 3;
    EXPECT_THROW(
        writer.writeIndexedExample(
            {},
            std::map<std::string, DatasetWriter::RaggedTensorView>{
                {"tokens", DatasetWriter::RaggedTensorView{
                               .dataType = DataType::INT32,
                               .storageMode = DatasetWriter::RaggedTensorView::StorageMode::WINDOW_REFERENCE,
                               .keyDataType = DataType::UINT64,
                               .indexDataType = DataType::INT64,
                               .key = &sourceKey,
                               .start = &start,
                               .length = &length}}}),
        std::runtime_error);

    writer.close();
    std::filesystem::remove_all(datasetPath);
}
