#include "DeepLearning/Api/Data/FileDataset.h"
#include "DeepLearning/Api/Data/NamedDataset.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class DatasetLayout;

namespace {

template <typename T>
concept HasPublicDatasetLayout = requires(const T &value) { value.getLayout(); };

template <typename T>
concept HasPublicLayoutAssertion = requires(const T &value, const DatasetLayout &layout) {
    value.assertLayout(layout);
};

template <typename T>
concept HasPublicResidencyCache = requires(const T &value) {
    value.getDeviceDatasetResidencyCache();
};

static_assert(!HasPublicDatasetLayout<Thor::FileDataset>);
static_assert(!HasPublicLayoutAssertion<Thor::FileDataset>);
static_assert(!HasPublicResidencyCache<Thor::NamedDataset>);

std::vector<std::string> readNonEmptyLines(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open generated header list: " + path.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            lines.push_back(std::move(line));
        }
    }
    return lines;
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open public header: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

}  // namespace

TEST(PublicHeaderSurface, GeneratedHeaderListUsesOnlyTheExplicitApiRoot) {
    const std::filesystem::path headerListPath = std::filesystem::path(THOR_BUILD_DIR) / "headerlist.txt";
    const std::vector<std::string> headers = readNonEmptyLines(headerListPath);

    ASSERT_FALSE(headers.empty());
    EXPECT_TRUE(std::is_sorted(headers.begin(), headers.end()));
    EXPECT_EQ(std::set<std::string>(headers.begin(), headers.end()).size(), headers.size());

    for (const std::string& header : headers) {
        EXPECT_TRUE(header.starts_with("./DeepLearning/Api/")) << header;
        EXPECT_FALSE(header.starts_with("./DeepLearning/Implementation/")) << header;
        EXPECT_FALSE(header.starts_with("./Utilities/")) << header;
    }

    const std::set<std::string> exported(headers.begin(), headers.end());
    EXPECT_TRUE(exported.contains("./DeepLearning/Api/Data/Batch.h"));
    EXPECT_TRUE(exported.contains("./DeepLearning/Api/Data/BatchSession.h"));
    EXPECT_TRUE(exported.contains("./DeepLearning/Api/Data/ExampleType.h"));
    EXPECT_FALSE(exported.contains("./DeepLearning/Api/Loaders/Batch.h"));
    EXPECT_FALSE(exported.contains("./DeepLearning/Api/Loaders/IndexedNamedBatchSession.h"));
    EXPECT_FALSE(exported.contains("./DeepLearning/Api/Loaders/DeviceResidentNamedBatchSession.h"));
    EXPECT_FALSE(exported.contains("./DeepLearning/Api/Loaders/DeviceResidentWindowedNamedBatchSession.h"));
    EXPECT_FALSE(exported.contains("./DeepLearning/Api/Loaders/DeviceDatasetMaterialization.h"));
    EXPECT_FALSE(exported.contains("./DeepLearning/Api/Training/DeviceDatasetResidency.h"));
    EXPECT_FALSE(exported.contains("./DeepLearning/Api/Training/DeviceDatasetStorageSelection.h"));
    EXPECT_FALSE(exported.contains("./Utilities/Loaders/Shard.h"));
}

TEST(PublicHeaderSurface, PublicHeadersDoNotDependOnLegacyShardStorage) {
    const std::filesystem::path sourceRoot = SOURCE_DIR;
    const std::filesystem::path headerListPath = std::filesystem::path(THOR_BUILD_DIR) / "headerlist.txt";
    const std::vector<std::string> headers = readNonEmptyLines(headerListPath);

    for (const std::string& header : headers) {
        ASSERT_TRUE(header.starts_with("./")) << header;
        const std::filesystem::path headerPath = sourceRoot / header.substr(2);
        const std::string contents = readFile(headerPath);
        EXPECT_EQ(contents.find("#include \"Utilities/Loaders/Shard.h\""), std::string::npos) << header;
    }
}


TEST(PublicHeaderSurface, DatasetBackendsKeepPhysicalStorageAndResidencyPrivate) {
    const std::filesystem::path sourceRoot = SOURCE_DIR;
    const std::string namedDataset =
        readFile(sourceRoot / "DeepLearning/Api/Data/NamedDataset.h");
    EXPECT_EQ(namedDataset.find("DeviceDatasetResidencyCache"), std::string::npos);
    EXPECT_EQ(namedDataset.find("getDeviceDatasetResidencyCache"), std::string::npos);

    const std::string fileDataset =
        readFile(sourceRoot / "DeepLearning/Api/Data/FileDataset.h");
    EXPECT_EQ(fileDataset.find("DatasetLayout"), std::string::npos);
    EXPECT_EQ(fileDataset.find("IndexedLocalNamedExampleReader"), std::string::npos);
    EXPECT_EQ(fileDataset.find("getLayout"), std::string::npos);
    EXPECT_EQ(fileDataset.find("assertLayout"), std::string::npos);

    EXPECT_FALSE(std::filesystem::exists(sourceRoot / "DeepLearning/Api/Loaders"));
}
