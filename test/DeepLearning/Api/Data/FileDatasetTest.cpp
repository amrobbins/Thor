#include "DeepLearning/Api/Data/FileDataset.h"
#include "DeepLearning/Api/Data/DatasetWriter.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::filesystem::path makeTempDatasetPath(const std::string &name) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("thor_file_dataset_" + name + "_" + std::to_string(nonce));
}

DatasetLayout testLayout() {
    return DatasetLayout::fromTensorShapes(
        std::vector<DatasetLayout::TensorShape>{
            DatasetLayout::TensorShape("features", {1}, ThorImplementation::DataType::FP32)});
}

void writeIndexedDataset(const std::filesystem::path &datasetPath) {
    DatasetWriter writer(datasetPath, testLayout(), 2);
    float value = 1.0f;
    writer.writeIndexedExample({
        {"features",
         DatasetWriter::TensorView{
             .dataType = ThorImplementation::DataType::FP32,
             .dimensions = {1},
             .data = &value,
             .numBytes = sizeof(value),
         }},
    });
    writer.close();
}

nlohmann::json readManifest(const std::filesystem::path &manifestPath) {
    std::ifstream in(manifestPath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open test manifest: " + manifestPath.string());
    }
    nlohmann::json manifest;
    in >> manifest;
    return manifest;
}

void writeManifest(const std::filesystem::path &manifestPath, const nlohmann::json &manifest) {
    std::ofstream out(manifestPath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write test manifest: " + manifestPath.string());
    }
    out << manifest.dump(2) << '\n';
}

}  // namespace

TEST(FileDatasetTest, RejectsLegacySplitStorageManifestWithMigrationMessage) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("split_storage");
    writeIndexedDataset(datasetPath);

    const std::filesystem::path manifestPath = datasetPath / DatasetWriter::MANIFEST_FILENAME;
    nlohmann::json manifest = readManifest(manifestPath);
    manifest["storage_mode"] = "split";
    writeManifest(manifestPath, manifest);

    try {
        (void)Thor::FileDataset::open(datasetPath);
        FAIL() << "Expected legacy split manifest rejection.";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("legacy split dataset storage_mode='split'"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("DatasetSplitManifest"), std::string::npos);
    }

    std::filesystem::remove_all(datasetPath);
}

TEST(FileDatasetTest, RejectsLegacyManifestWithoutStorageModeWithMigrationMessage) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("missing_storage_mode");
    writeIndexedDataset(datasetPath);

    const std::filesystem::path manifestPath = datasetPath / DatasetWriter::MANIFEST_FILENAME;
    nlohmann::json manifest = readManifest(manifestPath);
    manifest.erase("storage_mode");
    writeManifest(manifestPath, manifest);

    try {
        (void)Thor::FileDataset::open(datasetPath);
        FAIL() << "Expected legacy manifest rejection.";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("without storage_mode"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("DatasetSplitManifest"), std::string::npos);
    }

    std::filesystem::remove_all(datasetPath);
}

TEST(FileDatasetTest, RejectsManifestWithoutDatasetIdWithMigrationMessage) {
    const std::filesystem::path datasetPath = makeTempDatasetPath("missing_dataset_id");
    writeIndexedDataset(datasetPath);

    const std::filesystem::path manifestPath = datasetPath / DatasetWriter::MANIFEST_FILENAME;
    nlohmann::json manifest = readManifest(manifestPath);
    manifest.erase("dataset_id");
    writeManifest(manifestPath, manifest);

    try {
        (void)Thor::FileDataset::open(datasetPath);
        FAIL() << "Expected manifest identity rejection.";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("missing required dataset_id"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("DatasetWriter"), std::string::npos);
    }

    std::filesystem::remove_all(datasetPath);
}
