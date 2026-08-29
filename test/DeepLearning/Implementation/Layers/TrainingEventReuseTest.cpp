#include "DeepLearning/Implementation/Layers/TrainableLayer.h"

#include "Utilities/Common/Stream.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
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

class TrainingEventReuseProbe final : public TrainableLayer {
   public:
    explicit TrainingEventReuseProbe(const TensorPlacement& placement) : TrainableLayer(placement, false) {}

    void bindStreams(Stream dataStream, Stream updateStream) {
        streams = {std::move(dataStream)};
        uniqueDataStreams = streams;
        gradientUpdateStream = std::move(updateStream);
        errorInputReadyEvents.resize(1);
    }

    void submitIncomingErrorDependency() {
        THOR_THROW_IF_FALSE(gradientUpdateStream.has_value());
        THOR_THROW_IF_FALSE(errorInputReadyEvents.size() == 1);
        streams[0].putEvent(errorInputReadyEvents[0]);
        gradientUpdateStream.value().waitEvent(errorInputReadyEvents[0]);
    }

    void submitWeightsUpdatedDependency() {
        THOR_THROW_IF_FALSE(gradientUpdateStream.has_value());
        gradientUpdateStream.value().putEvent(weightsAreUpToDateEvent);
        weightsAreUpToDateEventValid = true;
        for (const Stream& dataStream : uniqueDataStreams) {
            dataStream.waitEvent(weightsAreUpToDateEvent);
        }
        weightsAreUpToDateEventValid = false;
    }

    uint64_t incomingErrorEventId() const {
        return errorInputReadyEvents.empty() ? 0u : errorInputReadyEvents[0].getId();
    }
    uint64_t weightsUpdatedEventId() const { return weightsAreUpToDateEvent.getId(); }
    bool weightsDependencyIsPending() const { return weightsAreUpToDateEventValid; }

   protected:
    void computeFeatureOut(uint32_t) override {}
    string getLayerType() override { return "TrainingEventReuseProbe"; }
    uint64_t flopCountForward() override { return 0; }
    uint64_t flopCountBackward() override { return 0; }
};

optional<filesystem::path> findThorSourceRootFrom(filesystem::path current) {
    while (!current.empty()) {
        if (filesystem::exists(current / "CMakeLists.txt") &&
            filesystem::exists(current / "DeepLearning" / "Implementation" / "Layers" / "TrainableLayer.h")) {
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

vector<string> e4HotPathsUsingValueReturningPutEvent(const filesystem::path& root) {
    const regex noArgumentPutEvent("putEvent\\s*\\(\\s*\\)");
    const regex booleanOnlyPutEvent("putEvent\\s*\\(\\s*(true|false)\\s*(,\\s*(true|false)\\s*)?\\)");
    const vector<filesystem::path> auditedFiles = {
        root / "DeepLearning" / "Implementation" / "Layers" / "TrainableLayer.h",
        root / "DeepLearning" / "Implementation" / "Layers" / "CustomLayer.cpp",
        root / "DeepLearning" / "Implementation" / "Layers" / "NeuralNetwork" / "Embedding.cpp",
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

TEST(TrainingEventReuse, TrainableDependenciesReuseOwnerScopedCudaEventsAcrossPasses) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Training reusable-event test requires a GPU";

    TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
    Stream dataStream(0);
    Stream updateStream(0);
    TrainingEventReuseProbe probe(placement);
    probe.bindStreams(dataStream, updateStream);

    probe.submitIncomingErrorDependency();
    probe.submitWeightsUpdatedDependency();
    const uint64_t incomingErrorEventId = probe.incomingErrorEventId();
    const uint64_t weightsUpdatedEventId = probe.weightsUpdatedEventId();
    ASSERT_NE(incomingErrorEventId, 0u);
    ASSERT_NE(weightsUpdatedEventId, 0u);
    ASSERT_NE(incomingErrorEventId, weightsUpdatedEventId);
    EXPECT_FALSE(probe.weightsDependencyIsPending());

    for (uint32_t repetition = 0; repetition < 64; ++repetition) {
        probe.submitIncomingErrorDependency();
        probe.submitWeightsUpdatedDependency();
        EXPECT_EQ(probe.incomingErrorEventId(), incomingErrorEventId);
        EXPECT_EQ(probe.weightsUpdatedEventId(), weightsUpdatedEventId);
        EXPECT_FALSE(probe.weightsDependencyIsPending());
    }

    dataStream.synchronize();
    updateStream.synchronize();
    probe.cleanup();
    EXPECT_EQ(probe.incomingErrorEventId(), 0u);
    EXPECT_EQ(probe.weightsUpdatedEventId(), 0u);
}

TEST(TrainingEventReuse, E4OwnedHotPathsDoNotUseValueReturningPutEvent) {
    const vector<string> matches = e4HotPathsUsingValueReturningPutEvent(findThorSourceRoot());
    EXPECT_TRUE(matches.empty())
        << "E4 training/backward recurring synchronization must use owner-scoped reusable Events; found value-returning putEvent in: "
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
