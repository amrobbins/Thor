#include "Utilities/Common/CudnnExecutionWorkspace.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

optional<filesystem::path> findThorSourceRootFrom(filesystem::path current) {
    while (!current.empty()) {
        if (filesystem::exists(current / "CMakeLists.txt") &&
            filesystem::exists(current / "Utilities" / "TensorOperations")) {
            return current;
        }
        const filesystem::path parent = current.parent_path();
        if (parent == current)
            break;
        current = parent;
    }
    return nullopt;
}

filesystem::path findThorSourceRoot() {
    if (const optional<filesystem::path> fromSource =
            findThorSourceRootFrom(filesystem::absolute(filesystem::path(__FILE__)).parent_path());
        fromSource.has_value()) {
        return fromSource.value();
    }
    if (const optional<filesystem::path> fromWorkingDirectory =
            findThorSourceRootFrom(filesystem::current_path());
        fromWorkingDirectory.has_value()) {
        return fromWorkingDirectory.value();
    }
    throw runtime_error("Could not locate Thor source root from source path or working directory.");
}

string readTextFile(const filesystem::path& path) {
    ifstream input(path);
    if (!input)
        throw runtime_error("Could not read " + path.string());
    ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

bool isCudnnImplementationSource(const filesystem::path& path) {
    const string filename = path.filename().string();
    return path.extension() == ".cpp" && filename.rfind("Cudnn", 0) == 0;
}

}  // namespace

TEST(CudnnWorkspaceCachePolicy, EveryFrontendGraphCacheUsesPlanOnlyCacheEntries) {
    const filesystem::path root = findThorSourceRoot();
    const filesystem::path tensorOperations = root / "Utilities" / "TensorOperations";

    const regex directWorkspaceTensor(
        R"(\bTensor\s+[A-Za-z_][A-Za-z0-9_]*[Ww]orkspace[A-Za-z0-9_]*\s*;)");
    const regex optionalWorkspaceTensor(
        R"(\b(?:std::)?optional\s*<\s*Tensor\s*>\s+[A-Za-z_][A-Za-z0-9_]*[Ww]orkspace[A-Za-z0-9_]*\s*;)");

    vector<filesystem::path> auditedCaches;
    for (const auto& entry : filesystem::recursive_directory_iterator(tensorOperations)) {
        if (!entry.is_regular_file() || !isCudnnImplementationSource(entry.path()))
            continue;

        const string source = readTextFile(entry.path());
        const bool containsGraphCache =
            source.find("GraphCache") != string::npos ||
            source.find("unordered_map<string, BuiltGraph>") != string::npos;
        if (!containsGraphCache)
            continue;

        auditedCaches.push_back(filesystem::relative(entry.path(), root));
        EXPECT_NE(source.find("CudnnCachedExecutionPlan<"), string::npos)
            << entry.path() << " has a cuDNN graph cache but does not use the shared plan-only cache entry.";
        EXPECT_EQ(source.find("struct BuiltGraph"), string::npos)
            << entry.path() << " defines a private BuiltGraph; use CudnnCachedExecutionPlan so workspace cannot enter the cache.";
        EXPECT_FALSE(regex_search(source, directWorkspaceTensor))
            << entry.path() << " owns a workspace Tensor inside a low-level cuDNN graph-cache implementation.";
        EXPECT_FALSE(regex_search(source, optionalWorkspaceTensor))
            << entry.path() << " owns an optional workspace Tensor inside a low-level cuDNN graph-cache implementation.";
    }

    // Keep the audit broad, but also make accidental loss of one of the current
    // cache implementations visible instead of silently passing an empty scan.
    EXPECT_GE(auditedCaches.size(), 5U);
}
