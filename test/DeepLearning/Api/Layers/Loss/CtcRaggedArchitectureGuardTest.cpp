#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to read source file: " + path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

}  // namespace

TEST(CtcRaggedArchitecture, PublicApiHasOnlyCanonicalRaggedLabelsContract) {
    const std::filesystem::path header =
        std::filesystem::path(SOURCE_DIR) / "DeepLearning/Api/Layers/Loss/CtcLoss.h";
    const std::string contents = readTextFile(header);

    EXPECT_NE(contents.find("CtcLoss::Builder& labels(RaggedTensor labels)"), std::string::npos);
    EXPECT_NE(contents.find("RaggedTensor getRaggedLabels() const"), std::string::npos);

    const std::vector<std::string> forbiddenPublicTokens = {
        std::string("label") + "Lengths(Tensor",
        std::string("padded") + "Labels(Tensor",
        std::string("maxLabel") + "Length(",
    };
    for (const std::string& token : forbiddenPublicTokens) {
        EXPECT_EQ(contents.find(token), std::string::npos)
            << "Canonical CTC labels are RaggedTensor; do not reintroduce the old public compatibility path: " << token;
    }
}

TEST(CtcRaggedArchitecture, ActiveCtcSourcesDoNotReintroducePaddedLabelCompaction) {
    const std::vector<std::filesystem::path> files = {
        std::filesystem::path(SOURCE_DIR) / "DeepLearning/Api/Layers/Loss/CtcLoss.h",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning/Api/Layers/Loss/CtcLoss.cpp",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning/Implementation/Layers/Loss/CtcLoss.h",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning/Implementation/Layers/Loss/CtcLoss.cpp",
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Loss/CtcLoss.h",
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Loss/CtcLoss.cpp",
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Loss/CtcLossScale.cu",
    };

    const std::vector<std::string> forbiddenTokens = {
        std::string("launchCompact") + "PaddedCtcLabels",
        std::string("compact") + "PaddedCtcLabels",
        std::string("paddedLabels") + "ToPacked",
    };
    for (const std::filesystem::path& path : files) {
        if (!std::filesystem::exists(path)) {
            continue;
        }
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbiddenTokens) {
            EXPECT_EQ(contents.find(token), std::string::npos)
                << "Canonical CTC consumes packed ragged labels directly; padded-label compaction must not return in "
                << std::filesystem::relative(path, SOURCE_DIR).generic_string();
        }
    }
}
