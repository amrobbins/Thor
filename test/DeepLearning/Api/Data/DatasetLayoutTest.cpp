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
        vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
            "history", {5, 1}, DataType::FP32, DataType::UINT64, DataType::INT32, 0.0, string("history_mask"))});

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
        vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
            "history", {5, 1}, DataType::FP32, DataType::UINT64, DataType::INT32, -1.0, string("history_mask"))});

    nlohmann::json j = layout.toJson();
    j["windowed_tensors"]["history"]["source_storage"] = nlohmann::json{
        {"file", "windowed_tensor_sources/windowed_tensor_000000.bin"},
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
    ASSERT_TRUE(history.sourceFilename.has_value());
    EXPECT_EQ(history.sourceFilename.value(), "windowed_tensor_sources/windowed_tensor_000000.bin");
    ASSERT_EQ(history.sourceSequences.size(), 1);
    EXPECT_EQ(history.sourceSequences.front().keyHex, "0100000000000000");
    EXPECT_EQ(history.sourceSequences.front().startIndex, 10);
}

TEST(DatasetLayoutTest, RejectsDuplicateWindowedTensorName) {
    EXPECT_THROW(DatasetLayout::fromTensorShapes(
                     vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("dense", {2}, DataType::FP32)},
                     vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
                         "dense", {5, 1}, DataType::FP32, DataType::UINT64, DataType::INT32)}),
                 std::runtime_error);
}

TEST(DatasetLayoutTest, RejectsWindowedMaskNameCollision) {
    EXPECT_THROW(DatasetLayout::fromTensorShapes(
                     vector<DatasetLayout::TensorShape>{DatasetLayout::TensorShape("dense", {2}, DataType::FP32)},
                     vector<DatasetLayout::WindowedTensorShape>{DatasetLayout::WindowedTensorShape(
                         "history", {5, 1}, DataType::FP32, DataType::UINT64, DataType::INT32, 0.0, string("dense"))}),
                 std::runtime_error);
}
