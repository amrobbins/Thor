#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Metric.h"

#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

class LossEventReuseProbe : public Loss {
   public:
    LossEventReuseProbe() : Loss(DataType::FP32) {}

    void bindStreams(Stream compute, Stream labels) {
        stream = std::move(compute);
        labelsStream = std::move(labels);
    }

    void submitLabelsHandshake() {
        waitForLabelsReady();
        markLabelsReusableAfterCompute();
    }

    uint64_t labelsReadyEventId() const { return labelsReadyEvent.getId(); }
    uint64_t labelsReusableEventId() const { return labelsReusableEvent.getId(); }

   protected:
    void infer(optional<Tensor>, optional<Tensor>, Stream) override {}
    void backProp(optional<Tensor>, optional<Tensor>, optional<Tensor>, Stream) override {}
};

class MetricEventReuseProbe : public Metric {
   public:
    void bindStreams(Stream compute, Stream labels) {
        stream = std::move(compute);
        labelsStream = std::move(labels);
    }

    void submitLabelsHandshake() {
        waitForLabelsReady();
        markLabelsReusableAfterCompute();
    }

    uint64_t labelsReadyEventId() const { return labelsReadyEvent.getId(); }
    uint64_t labelsReusableEventId() const { return labelsReusableEvent.getId(); }

    string toDisplayString(Tensor) override { return {}; }

   protected:
    void computeMetric(Tensor, Tensor, Tensor, Stream, uint32_t) override {}
};

optional<filesystem::path> findThorSourceRootFrom(filesystem::path current) {
    while (!current.empty()) {
        if (filesystem::exists(current / "CMakeLists.txt") &&
            filesystem::exists(current / "DeepLearning" / "Implementation" / "Layers" / "Loss.h")) {
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

bool isAuditedSourceFile(const filesystem::path& path) {
    const string extension = path.extension().string();
    return extension == ".h" || extension == ".hpp" || extension == ".cpp" || extension == ".cc" || extension == ".cu" ||
           extension == ".cuh";
}

vector<string> lossAndMetricHotPathFilesWithTemporaryWaitEvents(const filesystem::path& root) {
    const regex temporaryWaitEvent("waitEvent\\s*\\([^;\\n]*putEvent\\s*\\(");
    vector<string> matches;

    const vector<filesystem::path> roots = {
        root / "DeepLearning" / "Implementation" / "Layers" / "Loss.h",
        root / "DeepLearning" / "Implementation" / "Layers" / "Metric.h",
        root / "DeepLearning" / "Implementation" / "Layers" / "Loss",
        root / "DeepLearning" / "Implementation" / "Layers" / "Metrics",
    };

    for (const filesystem::path& candidate : roots) {
        if (filesystem::is_regular_file(candidate)) {
            if (regex_search(readTextFile(candidate), temporaryWaitEvent))
                matches.push_back(filesystem::relative(candidate, root).generic_string());
            continue;
        }
        for (const auto& entry : filesystem::recursive_directory_iterator(candidate)) {
            if (!entry.is_regular_file() || !isAuditedSourceFile(entry.path()))
                continue;
            if (regex_search(readTextFile(entry.path()), temporaryWaitEvent))
                matches.push_back(filesystem::relative(entry.path(), root).generic_string());
        }
    }
    return matches;
}

}  // namespace

TEST(LossMetricEventReuse, LossLabelsHandshakeReusesOwnerScopedEvents) {
    Stream compute(0);
    Stream labels(0);
    LossEventReuseProbe probe;
    probe.bindStreams(compute, labels);

    probe.submitLabelsHandshake();
    const uint64_t readyId = probe.labelsReadyEventId();
    const uint64_t reusableId = probe.labelsReusableEventId();
    ASSERT_NE(readyId, 0u);
    ASSERT_NE(reusableId, 0u);
    ASSERT_NE(readyId, reusableId);

    for (uint32_t repetition = 0; repetition < 64; ++repetition) {
        probe.submitLabelsHandshake();
        EXPECT_EQ(probe.labelsReadyEventId(), readyId);
        EXPECT_EQ(probe.labelsReusableEventId(), reusableId);
    }

    labels.synchronize();
    probe.cleanup();
    EXPECT_EQ(probe.labelsReadyEventId(), 0u);
    EXPECT_EQ(probe.labelsReusableEventId(), 0u);
}

TEST(LossMetricEventReuse, MetricLabelsHandshakeReusesOwnerScopedEvents) {
    Stream compute(0);
    Stream labels(0);
    MetricEventReuseProbe probe;
    probe.bindStreams(compute, labels);

    probe.submitLabelsHandshake();
    const uint64_t readyId = probe.labelsReadyEventId();
    const uint64_t reusableId = probe.labelsReusableEventId();
    ASSERT_NE(readyId, 0u);
    ASSERT_NE(reusableId, 0u);
    ASSERT_NE(readyId, reusableId);

    for (uint32_t repetition = 0; repetition < 64; ++repetition) {
        probe.submitLabelsHandshake();
        EXPECT_EQ(probe.labelsReadyEventId(), readyId);
        EXPECT_EQ(probe.labelsReusableEventId(), reusableId);
    }

    labels.synchronize();
    probe.cleanup();
    EXPECT_EQ(probe.labelsReadyEventId(), 0u);
    EXPECT_EQ(probe.labelsReusableEventId(), 0u);
}

TEST(LossMetricEventReuse, LossAndMetricHotPathsDoNotCreateTemporaryWaitEvents) {
    const vector<string> matches = lossAndMetricHotPathFilesWithTemporaryWaitEvents(findThorSourceRoot());
    EXPECT_TRUE(matches.empty()) << "Loss/Metric recurring synchronization must use owner-scoped reusable Events; found: "
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
