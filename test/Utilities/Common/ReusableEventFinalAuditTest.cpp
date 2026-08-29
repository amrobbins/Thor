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
            filesystem::exists(current / "Utilities" / "Common" / "ReusableEventPool.h")) {
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

string stripCppComments(const string& source) {
    string result;
    result.reserve(source.size());
    bool inBlockComment = false;
    bool inLineComment = false;
    bool inString = false;
    bool inChar = false;
    bool escaped = false;

    for (size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        const char next = i + 1 < source.size() ? source[i + 1] : '\0';

        if (inBlockComment) {
            if (c == '*' && next == '/') {
                inBlockComment = false;
                ++i;
            } else if (c == '\n') {
                result.push_back('\n');
            }
            continue;
        }
        if (inLineComment) {
            if (c == '\n') {
                inLineComment = false;
                result.push_back('\n');
            }
            continue;
        }
        if (!inString && !inChar && c == '/' && next == '*') {
            inBlockComment = true;
            ++i;
            continue;
        }
        if (!inString && !inChar && c == '/' && next == '/') {
            inLineComment = true;
            ++i;
            continue;
        }

        result.push_back(c);
        if (escaped) {
            escaped = false;
            continue;
        }
        if ((inString || inChar) && c == '\\') {
            escaped = true;
            continue;
        }
        if (!inChar && c == '"') {
            inString = !inString;
        } else if (!inString && c == '\'') {
            inChar = !inChar;
        }
    }
    return result;
}

bool isAuditedSourceFile(const filesystem::path& path) {
    const string extension = path.extension().string();
    return extension == ".h" || extension == ".hpp" || extension == ".cpp" || extension == ".cc" || extension == ".cu" ||
           extension == ".cuh";
}

vector<string> activeNestedTemporaryDependencySites(const filesystem::path& root) {
    const regex nestedTemporaryDependency(R"(waitEvent\s*\([^;{}]*putEvent\s*\()", regex::ECMAScript);
    vector<string> matches;
    // Audit canonical production source roots only. In particular, do not scan
    // bindings/python/build: that tree contains generated/install-staging copies
    // of public headers and may legitimately be stale relative to the source tree
    // until the corresponding build/install target is rerun.
    for (const filesystem::path& sourceRoot : {
             filesystem::path("Utilities"),
             filesystem::path("DeepLearning"),
             filesystem::path("bindings") / "python" / "src",
         }) {
        for (const auto& entry : filesystem::recursive_directory_iterator(root / sourceRoot)) {
            if (!entry.is_regular_file() || !isAuditedSourceFile(entry.path()))
                continue;
            const string source = stripCppComments(readTextFile(entry.path()));
            if (regex_search(source, nestedTemporaryDependency)) {
                matches.push_back(filesystem::relative(entry.path(), root).generic_string());
            }
        }
    }
    return matches;
}

vector<string> e5RecurringPathsUsingValueReturningPutEvent(const filesystem::path& root) {
    const regex noArgumentPutEvent(R"(putEvent\s*\(\s*\))");
    const regex booleanOnlyPutEvent(R"(putEvent\s*\(\s*(true|false)\s*(,\s*(true|false)\s*)?\))");
    const vector<filesystem::path> auditedFiles = {
        root / "Utilities" / "Expression" / "FusedEquation.cpp",
        root / "DeepLearning" / "Implementation" / "Layers" / "NeuralNetwork" / "LayerNorm.cpp",
        root / "DeepLearning" / "Implementation" / "Layers" / "NeuralNetwork" / "RMSNorm.cpp",
        root / "DeepLearning" / "Implementation" / "Layers" / "NeuralNetwork" / "InstanceNorm.cpp",
        root / "DeepLearning" / "Implementation" / "Layers" / "NeuralNetwork" / "BatchNormalization.cpp",
    };

    vector<string> matches;
    for (const filesystem::path& path : auditedFiles) {
        const string source = stripCppComments(readTextFile(path));
        if (regex_search(source, noArgumentPutEvent) || regex_search(source, booleanOnlyPutEvent)) {
            matches.push_back(filesystem::relative(path, root).generic_string());
        }
    }
    return matches;
}

}  // namespace

TEST(ReusableEventFinalAudit, NoActiveNestedTemporaryDependencyEventsRemain) {
    const vector<string> matches = activeNestedTemporaryDependencySites(findThorSourceRoot());
    EXPECT_TRUE(matches.empty())
        << "Recurring stream dependencies must not create a temporary CUDA event inside waitEvent(...). Found: "
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

TEST(ReusableEventFinalAudit, E5DynamicAndNormalizationHotPathsDoNotCreateTemporaryEvents) {
    const filesystem::path root = findThorSourceRoot();
    const vector<string> matches = e5RecurringPathsUsingValueReturningPutEvent(root);
    EXPECT_TRUE(matches.empty())
        << "E5 recurring synchronization must use leased or owner-scoped reusable Events; found value-returning putEvent in: "
        << ([&] {
               ostringstream out;
               for (size_t i = 0; i < matches.size(); ++i) {
                   if (i != 0)
                       out << ", ";
                   out << matches[i];
               }
               return out.str();
           })();

    const string fusedEquation = readTextFile(root / "Utilities" / "Expression" / "FusedEquation.cpp");
    EXPECT_NE(fusedEquation.find("ReusableEventLeases helperCompletionEvents"), string::npos);
    EXPECT_NE(fusedEquation.find("helper_stream.putEvent(helperDoneEvent)"), string::npos);

    for (const string layer : {"LayerNorm", "RMSNorm", "InstanceNorm", "BatchNormalization"}) {
        const string header = readTextFile(
            root / "DeepLearning" / "Implementation" / "Layers" / "NeuralNetwork" / (layer + ".h"));
        const string source = readTextFile(
            root / "DeepLearning" / "Implementation" / "Layers" / "NeuralNetwork" / (layer + ".cpp"));
        EXPECT_NE(header.find("backwardCompletionEvents"), string::npos) << layer;
        EXPECT_NE(source.find("TrainableLayer::cleanup()"), string::npos) << layer;
    }
}
