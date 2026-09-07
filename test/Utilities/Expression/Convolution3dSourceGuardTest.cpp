#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string readSourceFile(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Unable to read Conv3D source guard input: " + path.string());
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    return contents.str();
}

void expectSourceDoesNotContain(const std::filesystem::path& path, const std::vector<std::string>& forbidden_tokens) {
    const std::string contents = readSourceFile(path);
    for (const std::string& token : forbidden_tokens) {
        EXPECT_EQ(contents.find(token), std::string::npos)
            << path.generic_string() << " still contains legacy Conv3D token: " << token;
    }
}

}  // namespace

TEST(ConvolutionSpatial3d, SourceUsesCanonicalDescriptorInsteadOfLegacyGeometryFields) {
#ifndef SOURCE_DIR
    GTEST_SKIP() << "SOURCE_DIR is not defined for Conv3D source guard test.";
#else
    const std::filesystem::path source_root = SOURCE_DIR;

    expectSourceDoesNotContain(source_root / "Utilities/Expression/Expression.h",
                               {"int32_t conv_stride_d", "int32_t conv_stride_h", "int32_t conv_stride_w",
                                "int32_t conv_pad_d", "int32_t conv_pad_h", "int32_t conv_pad_w"});
    expectSourceDoesNotContain(source_root / "Utilities/Expression/EquationCompiler.cpp",
                               {"int32_t conv_stride_d", "int32_t conv_stride_h", "int32_t conv_stride_w",
                                "int32_t conv_pad_d", "int32_t conv_pad_h", "int32_t conv_pad_w"});
    expectSourceDoesNotContain(source_root / "Utilities/Expression/CompiledEquation.h",
                               {"int32_t conv_stride_d", "int32_t conv_stride_h", "int32_t conv_stride_w",
                                "int32_t conv_pad_d", "int32_t conv_pad_h", "int32_t conv_pad_w"});
#endif
}

TEST(ConvolutionSpatial3d, SourceDoesNotHardcodeLegacyThreeDimensionalCudnnGeometry) {
#ifndef SOURCE_DIR
    GTEST_SKIP() << "SOURCE_DIR is not defined for Conv3D source guard test.";
#else
    const std::filesystem::path stamped_equation =
        std::filesystem::path(SOURCE_DIR) / "Utilities/Expression/StampedEquation.cpp";
    const std::string contents = readSourceFile(stamped_equation);

    EXPECT_EQ(contents.find(".set_dilation({1, 1, 1})"), std::string::npos);
    EXPECT_NE(contents.find("convolutionFrontendPrePadding(const ConvolutionSpatial3d& spatial)"), std::string::npos);
    EXPECT_NE(contents.find("convolutionFrontendPostPadding(const ConvolutionSpatial3d& spatial)"), std::string::npos);
    EXPECT_NE(contents.find("convolutionFrontendDilations(const ConvolutionSpatial3d& spatial)"), std::string::npos);
#endif
}
