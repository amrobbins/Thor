#include "DeepLearning/Api/Data/DatasetLayout.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ThorImplementation::DataType;
using std::map;
using std::pair;
using std::string;
using std::vector;

namespace {

DatasetLayout::TensorSpec spec(const string &name,
                                     const vector<uint64_t> &shape,
                                     uint64_t offsetBytes,
                                     uint64_t numBytes,
                                     DataType dataType = DataType::FP32) {
    return DatasetLayout::TensorSpec{.name = name,
                                     .dataType = dataType,
                                     .dimensions = shape,
                                     .offsetBytes = offsetBytes,
                                     .numBytes = numBytes};
}

DatasetLayout validLayout() {
    return DatasetLayout(76,
                                   {spec("seasonality_inputs", {4}, 0, 16),
                                    spec("monotone_inputs", {8}, 16, 32),
                                    spec("daily_target", {3}, 48, 12),
                                    spec("aggregate_target", {3}, 60, 12),
                                    spec("daily_weight", {1}, 72, 4)});
}

}  // namespace

TEST(DatasetLayoutTest, FromTensorShapesCalculatesPackedRecordLayout) {
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{
            DatasetLayout::TensorShape("seasonality_inputs", {4}, DataType::FP32),
            DatasetLayout::TensorShape("monotone_inputs", {8}, DataType::FP32),
            DatasetLayout::TensorShape("daily_weight", {1}, DataType::FP32)});

    ASSERT_EQ(layout.recordSizeBytes(), 52);
    ASSERT_EQ(layout.tensors().size(), 3);
    EXPECT_EQ(layout.tensor("seasonality_inputs").offsetBytes, 0);
    EXPECT_EQ(layout.tensor("seasonality_inputs").numBytes, 16);
    EXPECT_EQ(layout.tensor("monotone_inputs").offsetBytes, 16);
    EXPECT_EQ(layout.tensor("monotone_inputs").numBytes, 32);
    EXPECT_EQ(layout.tensor("daily_weight").offsetBytes, 48);
    EXPECT_EQ(layout.tensor("daily_weight").numBytes, 4);
}

TEST(DatasetLayoutTest, ValidatesValidManualLayout) { EXPECT_NO_THROW(validLayout().validate()); }

TEST(DatasetLayoutTest, RejectsDuplicateTensorNames) {
    EXPECT_THROW(DatasetLayout(8, {spec("x", {1}, 0, 4), spec("x", {1}, 4, 4)}), std::runtime_error);
}

TEST(DatasetLayoutTest, RejectsOverlappingOffsets) {
    EXPECT_THROW(DatasetLayout(12, {spec("x", {2}, 0, 8), spec("y", {1}, 4, 4)}), std::runtime_error);
}

TEST(DatasetLayoutTest, AcceptsTightlyPackedUnalignedMixedDtypeOffsets) {
    EXPECT_NO_THROW(DatasetLayout(
        13,
        {spec("small", {1}, 0, 1, DataType::UINT8),
         spec("wide", {1}, 1, 8, DataType::FP64),
         spec("half", {2}, 9, 4, DataType::FP16)}));
}

TEST(DatasetLayoutTest, RejectsNumBytesThatDoNotMatchShapeAndDType) {
    EXPECT_THROW(DatasetLayout(8, {spec("x", {2}, 0, 4)}), std::runtime_error);
}

TEST(DatasetLayoutTest, RejectsTensorExtendingPastRecordSize) {
    EXPECT_THROW(DatasetLayout(4, {spec("x", {2}, 0, 8)}), std::runtime_error);
}

TEST(DatasetLayoutTest, RejectsMaxEndDifferentFromRecordSize) {
    EXPECT_THROW(DatasetLayout(12, {spec("x", {1}, 0, 4), spec("y", {1}, 4, 4)}), std::runtime_error);
}

TEST(DatasetLayoutTest, FromTensorShapesSupportsIndependentFieldDtypesWithoutPadding) {
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{
            DatasetLayout::TensorShape("small", {1}, DataType::UINT8),
            DatasetLayout::TensorShape("wide", {1}, DataType::FP64),
            DatasetLayout::TensorShape("half", {2}, DataType::FP16)});

    EXPECT_EQ(layout.tensor("small").dataType, DataType::UINT8);
    EXPECT_EQ(layout.tensor("small").offsetBytes, 0);
    EXPECT_EQ(layout.tensor("wide").dataType, DataType::FP64);
    EXPECT_EQ(layout.tensor("wide").offsetBytes, 1);
    EXPECT_EQ(layout.tensor("half").dataType, DataType::FP16);
    EXPECT_EQ(layout.tensor("half").offsetBytes, 9);
    EXPECT_EQ(layout.recordSizeBytes(), 13);
}

TEST(DatasetLayoutTest, RejectsEmptyShape) {
    EXPECT_THROW(DatasetLayout(4, {spec("x", {}, 0, 4)}), std::runtime_error);
}

TEST(DatasetLayoutTest, RejectsZeroDimension) {
    EXPECT_THROW(DatasetLayout(4, {spec("x", {0}, 0, 4)}), std::runtime_error);
}

TEST(DatasetLayoutTest, JsonRoundTripPreservesLayout) {
    DatasetLayout layout = validLayout();
    nlohmann::json j = layout.toJson();

    EXPECT_EQ(j.at("format").get<string>(), DatasetLayout::FORMAT);
    EXPECT_FALSE(j.contains("data_type"));
    EXPECT_EQ(j.at("tensors").at("monotone_inputs").at("data_type").get<string>(), "fp32");
    EXPECT_EQ(j.at("record_size_bytes").get<uint64_t>(), 76);
    EXPECT_EQ(j.at("tensors").at("monotone_inputs").at("shape").get<vector<uint64_t>>(), vector<uint64_t>({8}));

    DatasetLayout parsed = DatasetLayout::fromJson(j);
    layout.validateRequestedLayoutExact(parsed);
    parsed.validateRequestedLayoutExact(layout);
}

TEST(DatasetLayoutTest, ReadWriteManifestRoundTripPreservesLayout) {
    const std::filesystem::path manifestPath = std::filesystem::temp_directory_path() / "thor_dataset_layout_test_manifest.json";
    std::filesystem::remove(manifestPath);

    DatasetLayout layout = validLayout();
    layout.writeManifest(manifestPath);
    DatasetLayout parsed = DatasetLayout::readManifest(manifestPath);

    layout.validateRequestedLayoutExact(parsed);
    std::filesystem::remove(manifestPath);
}

TEST(DatasetLayoutTest, RejectsBadManifestFormat) {
    nlohmann::json j = validLayout().toJson();
    j["format"] = "wrong.format";
    EXPECT_THROW(DatasetLayout::fromJson(j), std::runtime_error);
}

TEST(DatasetLayoutTest, RejectsBadManifestDtype) {
    nlohmann::json j = validLayout().toJson();
    j["tensors"]["monotone_inputs"]["data_type"] = "not_a_dtype";
    EXPECT_THROW(DatasetLayout::fromJson(j), std::runtime_error);
}

TEST(DatasetLayoutTest, ExactLayoutValidationRejectsShapeMismatch) {
    DatasetLayout manifest = validLayout();
    DatasetLayout requested(76,
                                      {spec("seasonality_inputs", {4}, 0, 16),
                                       spec("monotone_inputs", {7}, 16, 28),
                                       spec("daily_target", {3}, 48, 12),
                                       spec("aggregate_target", {3}, 60, 12),
                                       spec("daily_weight", {1}, 72, 4)});
    EXPECT_THROW(manifest.validateRequestedLayoutExact(requested), std::runtime_error);
}

TEST(DatasetLayoutTest, ExactLayoutValidationRejectsMissingTensor) {
    DatasetLayout manifest = validLayout();
    DatasetLayout requested(76,
                                      {spec("seasonality_inputs", {4}, 0, 16),
                                       spec("monotone_inputs", {8}, 16, 32),
                                       spec("daily_target", {3}, 48, 12),
                                       spec("aggregate_target", {3}, 60, 12),
                                       spec("other_weight", {1}, 72, 4)});
    EXPECT_THROW(manifest.validateRequestedLayoutExact(requested), std::runtime_error);
}

TEST(DatasetLayoutTest, ExactLayoutValidationRejectsOffsetMismatch) {
    DatasetLayout manifest = validLayout();
    DatasetLayout requested(76,
                                      {spec("seasonality_inputs", {4}, 0, 16),
                                       spec("daily_target", {3}, 16, 12),
                                       spec("monotone_inputs", {8}, 28, 32),
                                       spec("aggregate_target", {3}, 60, 12),
                                       spec("daily_weight", {1}, 72, 4)});
    EXPECT_THROW(manifest.validateRequestedLayoutExact(requested), std::runtime_error);
}

TEST(DatasetLayoutTest, ExactLayoutValidationAcceptsDifferentTensorVectorOrder) {
    DatasetLayout manifest = validLayout();
    DatasetLayout requested(76,
                                      {spec("daily_weight", {1}, 72, 4),
                                       spec("aggregate_target", {3}, 60, 12),
                                       spec("daily_target", {3}, 48, 12),
                                       spec("monotone_inputs", {8}, 16, 32),
                                       spec("seasonality_inputs", {4}, 0, 16)});
    EXPECT_NO_THROW(manifest.validateRequestedLayoutExact(requested));
}

TEST(DatasetLayoutTest, FromTensorShapesCalculatesWindowedReferenceLayout) {
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("dense", {2}, DataType::FP32)},
        vector<DatasetLayout::WindowedTensorSourceShape>{DatasetLayout::WindowedTensorSourceShape(
            "history_source", {1}, DataType::FP32, DataType::UINT64)},
        vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
            "history", {5, 1}, "history_source", DataType::INT32, 0.0, string("history_mask"))});

    ASSERT_EQ(layout.recordSizeBytes(), 8 + 8 + 4);
    ASSERT_EQ(layout.tensors().size(), 1);
    ASSERT_EQ(layout.windowedTensors().size(), 1);
    EXPECT_EQ(layout.tensor("dense").offsetBytes, 0);
    const DatasetLayout::WindowedTensorSpec &history = layout.windowedTensor("history");
    EXPECT_EQ(history.referenceOffsetBytes, 8);
    EXPECT_EQ(history.referenceNumBytes, 12);
    EXPECT_EQ(history.windowLength(), 5);
    EXPECT_EQ(history.sourceStepDimensions(), vector<uint64_t>({1}));
    EXPECT_EQ(history.sourceStepNumBytes(), 4);
    EXPECT_EQ(history.outputNumBytes(), 20);
    EXPECT_EQ(history.maskName.value(), "history_mask");
}

TEST(DatasetLayoutTest, WindowedJsonRoundTripPreservesLayoutContractAndSourceStorage) {
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("dense", {2}, DataType::FP32)},
        vector<DatasetLayout::WindowedTensorSourceShape>{DatasetLayout::WindowedTensorSourceShape(
            "history_source", {1}, DataType::FP32, DataType::UINT64)},
        vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
            "history", {5, 1}, "history_source", DataType::INT32, -1.0, string("history_mask"))});

    nlohmann::json j = layout.toJson();
    j["window_sources"]["history_source"]["storage"] = nlohmann::json{
        {"file", "window_sources/window_source_000000.bin"},
        {"num_bytes", 12},
        {"sequences", nlohmann::json::array({nlohmann::json{{"key_hex", "0100000000000000"},
                                                              {"start_index", 10},
                                                              {"end_index_exclusive", 13},
                                                              {"offset_bytes", 0},
                                                              {"num_steps", 3},
                                                              {"num_bytes", 12}}})}};

    DatasetLayout parsed = DatasetLayout::fromJson(j);
    layout.validateRequestedLayoutExact(parsed);
    parsed.validateRequestedLayoutExact(layout);
    const DatasetLayout::WindowedTensorSpec &history = parsed.windowedTensor("history");
    EXPECT_EQ(history.sourceName, "history_source");
    const DatasetLayout::WindowedTensorSourceSpec &source = parsed.windowedTensorSource("history_source");
    ASSERT_TRUE(source.sourceFilename.has_value());
    EXPECT_EQ(source.sourceFilename.value(), "window_sources/window_source_000000.bin");
    ASSERT_EQ(source.sourceSequences.size(), 1);
    EXPECT_EQ(source.sourceSequences.front().keyHex, "0100000000000000");
    EXPECT_EQ(source.sourceSequences.front().startIndex, 10);
}

TEST(DatasetLayoutTest, MultipleWindowedTensorsMayShareOneSource) {
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        {},
        {DatasetLayout::WindowedTensorSourceShape("tokens", {}, DataType::UINT8, DataType::UINT64)},
        {DatasetLayout::WindowedTensorShape("examples", {8}, "tokens", DataType::INT64),
         DatasetLayout::WindowedTensorShape("labels", {8}, "tokens", DataType::INT64)});

    ASSERT_EQ(layout.windowedTensorSources().size(), 1);
    ASSERT_EQ(layout.windowedTensors().size(), 2);
    EXPECT_EQ(layout.windowedTensor("examples").sourceName, "tokens");
    EXPECT_EQ(layout.windowedTensor("labels").sourceName, "tokens");
    EXPECT_EQ(layout.windowedTensor("examples").dataType, DataType::UINT8);
    EXPECT_EQ(layout.windowedTensor("labels").dataType, DataType::UINT8);
    EXPECT_EQ(layout.recordSizeBytes(), 32);
}

TEST(DatasetLayoutTest, RejectsDuplicateWindowedTensorName) {
    EXPECT_THROW(DatasetLayout::fromTensorShapes(
                     vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("dense", {2}, DataType::FP32)},
                     vector<DatasetLayout::WindowedTensorSourceShape>{DatasetLayout::WindowedTensorSourceShape(
                         "history_source", {1}, DataType::FP32, DataType::UINT64)},
                     vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
                         "dense", {5, 1}, "history_source", DataType::INT32)}),
                 std::runtime_error);
}

TEST(DatasetLayoutTest, RejectsWindowedMaskNameCollision) {
    EXPECT_THROW(DatasetLayout::fromTensorShapes(
                     vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("dense", {2}, DataType::FP32)},
                     vector<DatasetLayout::WindowedTensorSourceShape>{DatasetLayout::WindowedTensorSourceShape(
                         "history_source", {1}, DataType::FP32, DataType::UINT64)},
                     vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
                         "history", {5, 1}, "history_source", DataType::INT32, 0.0, string("dense"))}),
                 std::runtime_error);
}

TEST(DatasetLayoutTest, AffineWindowReferencesRequireNoPerExampleRecordBytes) {
    DatasetLayout layout = DatasetLayout::fromTensorShapes(
        {},
        {DatasetLayout::WindowedTensorSourceShape("tokens", {}, DataType::UINT8, DataType::UINT64)},
        {DatasetLayout::WindowedTensorShape("examples",
                                            {8},
                                            "tokens",
                                            DataType::INT64,
                                            0.0,
                                            std::nullopt,
                                            DatasetLayout::WindowedTensorReferenceMode::AFFINE),
         DatasetLayout::WindowedTensorShape("labels",
                                            {8},
                                            "tokens",
                                            DataType::INT64,
                                            0.0,
                                            std::nullopt,
                                            DatasetLayout::WindowedTensorReferenceMode::AFFINE)});

    EXPECT_EQ(layout.recordSizeBytes(), 0u);
    EXPECT_TRUE(layout.hasAffineWindowedTensors());
    EXPECT_FALSE(layout.hasIndexedWindowedTensors());
    EXPECT_EQ(layout.windowedTensor("examples").referenceNumBytes, 0u);
    EXPECT_EQ(layout.windowedTensor("labels").referenceOffsetBytes, 0u);

    nlohmann::json persisted = layout.toJson();
    EXPECT_EQ(persisted.at("format").get<std::string>(), "thor.dataset.v1");
    EXPECT_EQ(persisted.at("windowed_tensors").at("examples").at("reference_mode").get<std::string>(), "affine");
    EXPECT_FALSE(persisted.at("windowed_tensors").at("examples").contains("reference_offset_bytes"));
    EXPECT_FALSE(persisted.at("windowed_tensors").at("examples").contains("reference_num_bytes"));
    EXPECT_EQ(DatasetLayout::fromJson(persisted).windowedTensor("labels").referenceMode,
              DatasetLayout::WindowedTensorReferenceMode::AFFINE);
}

TEST(DatasetLayoutTest, RejectsRetiredWindowDatasetVersions) {
    nlohmann::json manifest = DatasetLayout::fromTensorShapes(
        {},
        {DatasetLayout::WindowedTensorSourceShape("tokens", {}, DataType::UINT8, DataType::UINT64)},
        {DatasetLayout::WindowedTensorShape("examples",
                                            {8},
                                            "tokens",
                                            DataType::INT64,
                                            0.0,
                                            std::nullopt,
                                            DatasetLayout::WindowedTensorReferenceMode::AFFINE)})
                                      .toJson();
    nlohmann::json oldV1 = manifest;
    oldV1.at("windowed_tensors").at("examples").erase("reference_mode");
    EXPECT_THROW(DatasetLayout::fromJson(oldV1), std::runtime_error);

    manifest["format"] = "thor.dataset.v2";
    EXPECT_THROW(DatasetLayout::fromJson(manifest), std::runtime_error);
    manifest["format"] = "thor.dataset.v3";
    EXPECT_THROW(DatasetLayout::fromJson(manifest), std::runtime_error);
}
