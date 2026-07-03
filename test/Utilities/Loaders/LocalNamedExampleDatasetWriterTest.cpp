#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"
#include "Utilities/Loaders/Shard.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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
