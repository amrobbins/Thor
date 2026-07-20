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

bool isSourceFile(const std::filesystem::path& path) {
    const std::string extension = path.extension().string();
    return extension == ".h" || extension == ".hpp" || extension == ".cuh" || extension == ".cpp" ||
           extension == ".cc" || extension == ".cxx" || extension == ".cu";
}

bool isBuildArtifactDirectory(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    return name == "build" || name == "_build" || name == "_deps" || name == "dist" ||
           name == "wheelhouse" || name == "__pycache__" || name == ".venv" || name == "venv" ||
           name == "_skbuild" || name.rfind("cmake-build-", 0) == 0;
}

template <typename Fn>
void forEachActiveSourceFile(const std::vector<std::filesystem::path>& sourceRoots, Fn&& fn) {
    for (const std::filesystem::path& sourceRoot : sourceRoots) {
        std::filesystem::recursive_directory_iterator entry(sourceRoot);
        const std::filesystem::recursive_directory_iterator end;
        while (entry != end) {
            if (entry->is_directory() && isBuildArtifactDirectory(entry->path())) {
                // In-tree CMake and wheel builds can contain installed copies of headers from an earlier build.
                // Those files are generated artifacts, not active Thor source, and must not influence this guard.
                entry.disable_recursion_pending();
            } else if (entry->is_regular_file() && isSourceFile(entry->path())) {
                fn(entry->path());
            }
            ++entry;
        }
    }
}

}  // namespace

TEST(ExpressionReductionArchitecture, SourceGuardsIgnoreGeneratedBuildTrees) {
    EXPECT_TRUE(isBuildArtifactDirectory("build"));
    EXPECT_TRUE(isBuildArtifactDirectory("cmake-build-debug"));
    EXPECT_TRUE(isBuildArtifactDirectory("_skbuild"));
    EXPECT_FALSE(isBuildArtifactDirectory("src"));
    EXPECT_FALSE(isBuildArtifactDirectory("Utilities"));
}

TEST(ExpressionReductionArchitecture, ActiveSourcesDoNotUseCudnnReductionApis) {
    const std::vector<std::filesystem::path> source_roots = {
        std::filesystem::path(SOURCE_DIR) / "Utilities",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning",
        std::filesystem::path(SOURCE_DIR) / "bindings",
    };
    for (const std::filesystem::path& source_root : source_roots) {
        ASSERT_TRUE(std::filesystem::exists(source_root));
    }

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
    forEachActiveSourceFile(source_roots, [&](const std::filesystem::path& path) {
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            if (contents.find(token) != std::string::npos) {
                violations.push_back(std::filesystem::relative(path, SOURCE_DIR).generic_string() + ": " + token);
            }
        }
    });

    EXPECT_TRUE(violations.empty()) << "Active Thor reductions must not use cuDNN reduction APIs.\n"
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
    forEachActiveSourceFile(source_roots, [&](const std::filesystem::path& path) {
        const std::filesystem::path relative = std::filesystem::relative(path, SOURCE_DIR);
        const std::string relative_string = relative.generic_string();
        if (relative_string.rfind("Utilities/TensorOperations/Cub/CubReduction", 0) == 0
            || relative_string.rfind("Utilities/TensorOperations/Cub/CubArgReduction", 0) == 0) {
            return;
        }

        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            if (contents.find(token) != std::string::npos) {
                violations.push_back(relative_string + ": " + token);
            }
        }
    });

    EXPECT_TRUE(violations.empty()) << "General value and arg reductions must use the central CUB reduction utility.\n"
                                    << [&]() {
                                           std::ostringstream message;
                                           for (const std::string& violation : violations) {
                                               message << violation << '\n';
                                           }
                                           return message.str();
                                       }();
}
