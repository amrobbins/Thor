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

bool isExpressionSourceFile(const std::filesystem::path& path) {
    const std::string extension = path.extension().string();
    return extension == ".h" || extension == ".hpp" || extension == ".cuh" || extension == ".cpp" ||
           extension == ".cc" || extension == ".cxx" || extension == ".cu";
}

}  // namespace

TEST(ExpressionReductionArchitecture, DenseExpressionSourcesDoNotUseCudnnReductionApis) {
    const std::filesystem::path expression_root = std::filesystem::path(SOURCE_DIR) / "Utilities" / "Expression";
    ASSERT_TRUE(std::filesystem::exists(expression_root));

    const std::vector<std::string> forbidden_tokens = {
        "cudnnReduceTensor(",
        "cudnnReduceTensorDescriptor_t",
        "cudnnCreateReduceTensorDescriptor",
        "cudnnSetReduceTensorDescriptor",
        "cudnnGetReductionWorkspaceSize",
        "cudnnGetReductionIndicesSize",
        "CUDNN_REDUCE_TENSOR_",
    };

    std::vector<std::string> violations;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(expression_root)) {
        if (!entry.is_regular_file() || !isExpressionSourceFile(entry.path())) {
            continue;
        }

        const std::string contents = readTextFile(entry.path());
        for (const std::string& token : forbidden_tokens) {
            if (contents.find(token) != std::string::npos) {
                violations.push_back(std::filesystem::relative(entry.path(), SOURCE_DIR).string() + ": " + token);
            }
        }
    }

    EXPECT_TRUE(violations.empty()) << "Dense expression reductions must use the central CUB reduction utilities.\n"
                                    << [&]() {
                                           std::ostringstream message;
                                           for (const std::string& violation : violations) {
                                               message << violation << '\n';
                                           }
                                           return message.str();
                                       }();
}

TEST(ExpressionReductionArchitecture, GeneralReductionsAreCentralizedUnderCubReduction) {
    const std::vector<std::filesystem::path> source_roots = {
        std::filesystem::path(SOURCE_DIR) / "Utilities",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning",
        std::filesystem::path(SOURCE_DIR) / "bindings",
    };
    for (const std::filesystem::path& source_root : source_roots) {
        ASSERT_TRUE(std::filesystem::exists(source_root));
    }

    // ReduceByKey is intentionally excluded: it performs keyed grouping, not tensor-axis reduction.
    const std::vector<std::string> forbidden_tokens = {
        "cub::DeviceReduce::Sum(",
        "cub::DeviceReduce::Min(",
        "cub::DeviceReduce::Max(",
        "cub::DeviceReduce::Reduce(",
        "cub::DeviceReduce::TransformReduce(",
        "cub::DeviceReduce::ArgMin(",
        "cub::DeviceReduce::ArgMax(",
        "cub::DeviceSegmentedReduce::Sum(",
        "cub::DeviceSegmentedReduce::Min(",
        "cub::DeviceSegmentedReduce::Max(",
        "cub::DeviceSegmentedReduce::Reduce(",
        "cub::DeviceSegmentedReduce::ArgMin(",
        "cub::DeviceSegmentedReduce::ArgMax(",
    };

    std::vector<std::string> violations;
    for (const std::filesystem::path& source_root : source_roots) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(source_root)) {
            if (!entry.is_regular_file() || !isExpressionSourceFile(entry.path())) {
                continue;
            }
            const std::filesystem::path relative = std::filesystem::relative(entry.path(), SOURCE_DIR);
            const std::string relative_string = relative.generic_string();
            if (relative_string.rfind("Utilities/TensorOperations/Cub/CubReduction", 0) == 0
                || relative_string.rfind("Utilities/TensorOperations/Cub/CubArgReduction", 0) == 0) {
                continue;
            }

            const std::string contents = readTextFile(entry.path());
            for (const std::string& token : forbidden_tokens) {
                if (contents.find(token) != std::string::npos) {
                    violations.push_back(relative_string + ": " + token);
                }
            }
        }
    }

    EXPECT_TRUE(violations.empty()) << "General value and arg reductions must use the central CUB reduction utility.\n"
                                    << [&]() {
                                           std::ostringstream message;
                                           for (const std::string& violation : violations) {
                                               message << violation << '\n';
                                           }
                                           return message.str();
                                       }();
}
