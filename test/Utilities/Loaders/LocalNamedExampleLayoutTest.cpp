#include "Utilities/Loaders/LocalNamedExampleLayout.h"

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

LocalNamedExampleLayout::TensorSpec spec(const string &name, const vector<uint64_t> &shape, uint64_t offsetBytes, uint64_t numBytes) {
    return LocalNamedExampleLayout::TensorSpec{.name = name,
                                               .dataType = DataType::FP32,
                                               .dimensions = shape,
                                               .offsetBytes = offsetBytes,
                                               .numBytes = numBytes};
}

LocalNamedExampleLayout validLayout() {
    return LocalNamedExampleLayout(DataType::FP32,
                                   76,
                                   {spec("seasonality_inputs", {4}, 0, 16),
                                    spec("monotone_inputs", {8}, 16, 32),
                                    spec("daily_target", {3}, 48, 12),
                                    spec("aggregate_target", {3}, 60, 12),
                                    spec("daily_weight", {1}, 72, 4)});
}

}  // namespace

TEST(LocalNamedExampleLayoutTest, FromTensorShapesCalculatesPackedRecordLayout) {
    LocalNamedExampleLayout layout = LocalNamedExampleLayout::fromTensorShapes(
        vector<pair<string, vector<uint64_t>>>{{"seasonality_inputs", {4}}, {"monotone_inputs", {8}}, {"daily_weight", {1}}},
        DataType::FP32);

    ASSERT_EQ(layout.dataType(), DataType::FP32);
    ASSERT_EQ(layout.recordSizeBytes(), 52);
    ASSERT_EQ(layout.tensors().size(), 3);
    EXPECT_EQ(layout.tensor("seasonality_inputs").offsetBytes, 0);
    EXPECT_EQ(layout.tensor("seasonality_inputs").numBytes, 16);
    EXPECT_EQ(layout.tensor("monotone_inputs").offsetBytes, 16);
    EXPECT_EQ(layout.tensor("monotone_inputs").numBytes, 32);
    EXPECT_EQ(layout.tensor("daily_weight").offsetBytes, 48);
    EXPECT_EQ(layout.tensor("daily_weight").numBytes, 4);
}

TEST(LocalNamedExampleLayoutTest, ValidatesValidManualLayout) { EXPECT_NO_THROW(validLayout().validate()); }

TEST(LocalNamedExampleLayoutTest, RejectsDuplicateTensorNames) {
    EXPECT_THROW(LocalNamedExampleLayout(DataType::FP32, 8, {spec("x", {1}, 0, 4), spec("x", {1}, 4, 4)}), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, RejectsOverlappingOffsets) {
    EXPECT_THROW(LocalNamedExampleLayout(DataType::FP32, 12, {spec("x", {2}, 0, 8), spec("y", {1}, 4, 4)}), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, RejectsMisalignedOffsets) {
    EXPECT_THROW(LocalNamedExampleLayout(DataType::FP32, 8, {spec("x", {1}, 0, 4), spec("y", {1}, 5, 4)}), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, RejectsNumBytesThatDoNotMatchShapeAndDType) {
    EXPECT_THROW(LocalNamedExampleLayout(DataType::FP32, 8, {spec("x", {2}, 0, 4)}), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, RejectsTensorExtendingPastRecordSize) {
    EXPECT_THROW(LocalNamedExampleLayout(DataType::FP32, 4, {spec("x", {2}, 0, 8)}), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, RejectsMaxEndDifferentFromRecordSize) {
    EXPECT_THROW(LocalNamedExampleLayout(DataType::FP32, 12, {spec("x", {1}, 0, 4), spec("y", {1}, 4, 4)}), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, RejectsTensorDtypeThatDoesNotMatchLayoutDtype) {
    LocalNamedExampleLayout::TensorSpec x = spec("x", {1}, 0, 4);
    x.dataType = DataType::FP16;
    EXPECT_THROW(LocalNamedExampleLayout(DataType::FP32, 4, {x}), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, RejectsEmptyShape) {
    EXPECT_THROW(LocalNamedExampleLayout(DataType::FP32, 4, {spec("x", {}, 0, 4)}), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, RejectsZeroDimension) {
    EXPECT_THROW(LocalNamedExampleLayout(DataType::FP32, 4, {spec("x", {0}, 0, 4)}), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, JsonRoundTripPreservesLayout) {
    LocalNamedExampleLayout layout = validLayout();
    nlohmann::json j = layout.toJson();

    EXPECT_EQ(j.at("format").get<string>(), LocalNamedExampleLayout::FORMAT);
    EXPECT_EQ(j.at("data_type").get<string>(), "fp32");
    EXPECT_EQ(j.at("record_size_bytes").get<uint64_t>(), 76);
    EXPECT_EQ(j.at("tensors").at("monotone_inputs").at("shape").get<vector<uint64_t>>(), vector<uint64_t>({8}));

    LocalNamedExampleLayout parsed = LocalNamedExampleLayout::fromJson(j);
    layout.validateRequestedLayoutExact(parsed);
    parsed.validateRequestedLayoutExact(layout);
}

TEST(LocalNamedExampleLayoutTest, ReadWriteManifestRoundTripPreservesLayout) {
    const std::filesystem::path manifestPath = std::filesystem::temp_directory_path() / "thor_local_named_example_layout_test_manifest.json";
    std::filesystem::remove(manifestPath);

    LocalNamedExampleLayout layout = validLayout();
    layout.writeManifest(manifestPath);
    LocalNamedExampleLayout parsed = LocalNamedExampleLayout::readManifest(manifestPath);

    layout.validateRequestedLayoutExact(parsed);
    std::filesystem::remove(manifestPath);
}

TEST(LocalNamedExampleLayoutTest, RejectsBadManifestFormat) {
    nlohmann::json j = validLayout().toJson();
    j["format"] = "wrong.format";
    EXPECT_THROW(LocalNamedExampleLayout::fromJson(j), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, RejectsBadManifestDtype) {
    nlohmann::json j = validLayout().toJson();
    j["data_type"] = "not_a_dtype";
    EXPECT_THROW(LocalNamedExampleLayout::fromJson(j), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, ExactLayoutValidationRejectsShapeMismatch) {
    LocalNamedExampleLayout manifest = validLayout();
    LocalNamedExampleLayout requested(DataType::FP32,
                                      76,
                                      {spec("seasonality_inputs", {4}, 0, 16),
                                       spec("monotone_inputs", {7}, 16, 28),
                                       spec("daily_target", {3}, 48, 12),
                                       spec("aggregate_target", {3}, 60, 12),
                                       spec("daily_weight", {1}, 72, 4)});
    EXPECT_THROW(manifest.validateRequestedLayoutExact(requested), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, ExactLayoutValidationRejectsMissingTensor) {
    LocalNamedExampleLayout manifest = validLayout();
    LocalNamedExampleLayout requested(DataType::FP32,
                                      76,
                                      {spec("seasonality_inputs", {4}, 0, 16),
                                       spec("monotone_inputs", {8}, 16, 32),
                                       spec("daily_target", {3}, 48, 12),
                                       spec("aggregate_target", {3}, 60, 12),
                                       spec("other_weight", {1}, 72, 4)});
    EXPECT_THROW(manifest.validateRequestedLayoutExact(requested), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, ExactLayoutValidationRejectsOffsetMismatch) {
    LocalNamedExampleLayout manifest = validLayout();
    LocalNamedExampleLayout requested(DataType::FP32,
                                      76,
                                      {spec("seasonality_inputs", {4}, 0, 16),
                                       spec("daily_target", {3}, 16, 12),
                                       spec("monotone_inputs", {8}, 28, 32),
                                       spec("aggregate_target", {3}, 60, 12),
                                       spec("daily_weight", {1}, 72, 4)});
    EXPECT_THROW(manifest.validateRequestedLayoutExact(requested), std::runtime_error);
}

TEST(LocalNamedExampleLayoutTest, ExactLayoutValidationAcceptsDifferentTensorVectorOrder) {
    LocalNamedExampleLayout manifest = validLayout();
    LocalNamedExampleLayout requested(DataType::FP32,
                                      76,
                                      {spec("daily_weight", {1}, 72, 4),
                                       spec("aggregate_target", {3}, 60, 12),
                                       spec("daily_target", {3}, 48, 12),
                                       spec("monotone_inputs", {8}, 16, 32),
                                       spec("seasonality_inputs", {4}, 0, 16)});
    EXPECT_NO_THROW(manifest.validateRequestedLayoutExact(requested));
}
