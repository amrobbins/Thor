#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

std::string readSource(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open source file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

}  // namespace

TEST(RaggedAttentionArchitectureGuard, PublicSdpaKeepsRaggedTensorAsTheOnlyRaggedInputSurface) {
    const std::filesystem::path root = SOURCE_DIR;
    const std::string header =
        readSource(root / "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h");
    const std::string pythonBinding =
        readSource(root / "bindings/python/src/core/layers/scaled_dot_product_attention.cpp");

    EXPECT_NE(header.find("selfInput(RaggedTensor input)"), std::string::npos);
    EXPECT_NE(header.find("queryInput(RaggedTensor input)"), std::string::npos);
    EXPECT_NE(header.find("keyInput(RaggedTensor input)"), std::string::npos);
    EXPECT_NE(header.find("valueInput(RaggedTensor input)"), std::string::npos);

    for (const char* retiredPublicMethod : {
             "Builder& raggedOffsetsInput(",
             "Builder& queryRaggedOffsetsInput(",
             "Builder& keyValueRaggedOffsetsInput(",
             "getUseRaggedOffsets(",
             "getQueryRaggedOffsetsInput(",
             "getKeyValueRaggedOffsetsInput(",
         }) {
        EXPECT_EQ(header.find(retiredPublicMethod), std::string::npos) << retiredPublicMethod;
    }

    for (const char* retiredPythonArgument : {
             "\"ragged_offsets\"_a",
             "\"query_ragged_offsets\"_a",
             "\"key_value_ragged_offsets\"_a",
         }) {
        EXPECT_EQ(pythonBinding.find(retiredPythonArgument), std::string::npos) << retiredPythonArgument;
    }
}
