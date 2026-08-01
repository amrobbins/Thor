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
                entry.disable_recursion_pending();
            } else if (entry->is_regular_file() && isSourceFile(entry->path())) {
                fn(entry->path());
            }
            ++entry;
        }
    }
}

}  // namespace

TEST(CudaRuntimeMemoryApiArchitecture, ActiveSourcesDoNotUseDefaultStreamMemcpyOrMemsetApis) {
    const std::vector<std::filesystem::path> sourceRoots = {
        std::filesystem::path(SOURCE_DIR) / "Utilities",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning",
        std::filesystem::path(SOURCE_DIR) / "bindings",
        std::filesystem::path(SOURCE_DIR) / "test",
    };
    for (const std::filesystem::path& sourceRoot : sourceRoots) {
        ASSERT_TRUE(std::filesystem::exists(sourceRoot));
    }

    // Thor execution streams are cudaStreamNonBlocking, so work implicitly issued to stream 0
    // is not ordered with them. Use the Async API with an explicit Thor stream instead.
    // Keep the tokens split so this guard does not flag its own source text.
    const std::vector<std::string> forbiddenTokens = {
        std::string("cuda") + "Memcpy(",
        std::string("cuda") + "Memcpy2D(",
        std::string("cuda") + "Memcpy3D(",
        std::string("cuda") + "MemcpyPeer(",
        std::string("cuda") + "MemcpyToSymbol(",
        std::string("cuda") + "MemcpyFromSymbol(",
        std::string("cuda") + "Memset(",
        std::string("cuda") + "Memset2D(",
        std::string("cuda") + "Memset3D(",
    };

    std::vector<std::string> violations;
    forEachActiveSourceFile(sourceRoots, [&](const std::filesystem::path& path) {
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbiddenTokens) {
            if (contents.find(token) != std::string::npos) {
                violations.push_back(std::filesystem::relative(path, SOURCE_DIR).generic_string() + ": " + token);
            }
        }
    });

    EXPECT_TRUE(violations.empty())
        << "Thor uses cudaStreamNonBlocking execution streams; non-Async/default-stream CUDA memory APIs are unordered with them. "
           "Use the Async API with an explicit Thor stream.\n"
        << [&]() {
               std::ostringstream message;
               for (const std::string& violation : violations) {
                   message << violation << '\n';
               }
               return message.str();
           }();
}
