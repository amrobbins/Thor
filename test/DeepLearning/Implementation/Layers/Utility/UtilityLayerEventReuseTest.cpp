#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace {

optional<filesystem::path> findThorSourceRootFrom(filesystem::path current) {
    while (!current.empty()) {
        if (filesystem::exists(current / "CMakeLists.txt") &&
            filesystem::exists(current / "DeepLearning" / "Implementation" / "Layers" / "Utility" / "TensorFanout.h")) {
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
    if (const optional<filesystem::path> fromWorkingDirectory = findThorSourceRootFrom(filesystem::current_path());
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

vector<string> e3HotPathsUsingValueReturningPutEvent(const filesystem::path& root) {
    // E3-owned synchronization is recurring and must use an owner-scoped Event.
    // These patterns identify the value-returning overloads putEvent(),
    // putEvent(bool), and putEvent(bool, bool). Calls with an Event argument are
    // intentionally not matched.
    const regex noArgumentPutEvent("putEvent\\s*\\(\\s*\\)");
    const regex booleanOnlyPutEvent("putEvent\\s*\\(\\s*(true|false)\\s*(,\\s*(true|false)\\s*)?\\)");

    const vector<filesystem::path> auditedFiles = {
        root / "DeepLearning" / "Implementation" / "Layers" / "Utility" / "TensorFanout.h",
        root / "DeepLearning" / "Implementation" / "Layers" / "Utility" / "Split.h",
        root / "DeepLearning" / "Implementation" / "Layers" / "Utility" / "Concatenate.h",
        root / "DeepLearning" / "Implementation" / "Layers" / "Utility" / "RaggedConcatenate.h",
        root / "DeepLearning" / "Implementation" / "Layers" / "Utility" / "EinsumLayer.h",
        root / "DeepLearning" / "Implementation" / "Layers" / "Utility" / "DeviceCrossing.h",
        root / "DeepLearning" / "Implementation" / "Layers" / "NeuralNetwork" / "AdaptiveLayerNorm.cpp",
    };

    vector<string> matches;
    for (const filesystem::path& path : auditedFiles) {
        const string contents = readTextFile(path);
        if (regex_search(contents, noArgumentPutEvent) || regex_search(contents, booleanOnlyPutEvent)) {
            matches.push_back(filesystem::relative(path, root).generic_string());
        }
    }
    return matches;
}

}  // namespace

TEST(UtilityLayerEventReuse, E3OwnedHotPathsDoNotUseValueReturningPutEvent) {
    const vector<string> matches = e3HotPathsUsingValueReturningPutEvent(findThorSourceRoot());
    EXPECT_TRUE(matches.empty())
        << "E3 utility-layer recurring synchronization must use owner-scoped reusable Events; found value-returning putEvent in: "
        << ([&] {
               ostringstream out;
               for (size_t i = 0; i < matches.size(); ++i) {
                   if (i != 0)
                       out << ", ";
                   out << matches[i];
               }
               return out.str();
           })();
}
