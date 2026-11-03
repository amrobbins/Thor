// 1170: the network-level reduction tuning depth. Unset means Regular; Network::place() stamps every layer under the
// network's depth on the calling thread, whatever the process default or the caller's own scope is.
//
// The GPU tests place a network whose only reduction is a dense Sum metric over FP8 E4M3 values and count the contests
// the dense reduction selection cache runs during placement. Each network uses its own feature width so no other test in the
// process has tuned its reduction.

#include "DeepLearning/Api/Layers/Metrics/ReductionMetrics.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace Thor;
using Depth = ThorImplementation::CubReductionTuningDepth;

namespace {

uint64_t contestsSoFar() {
    return ThorImplementation::CubReductionInternal::denseReductionSelectionCache().tuningAttempts();
}

// Builds input [width] FP8 -> Sum metric -> output, places it (inference only, batch 4) at `depth` (nullopt: unset)
// and returns the contests the placement ran.
uint64_t placeFp8SumNetwork(uint64_t width, std::optional<Depth> depth) {
    Network network("fp8_sum_tuning_depth_" + std::to_string(width));
    network.setReductionTuningDepth(depth);
    NetworkInput values = NetworkInput::Builder()
                              .network(network)
                              .name("values")
                              .dimensions({width})
                              .dataType(DataType::FP8_E4M3)
                              .build();
    Sum sum = Sum::Builder().network(network).values(values.getFeatureOutput().value()).build();
    NetworkOutput::Builder().network(network).name("sum").inputTensor(sum.getMetric()).dataType(DataType::FP32).build();

    const uint64_t before = contestsSoFar();
    std::vector<Event> initializationDone;
    std::shared_ptr<PlacedNetwork> placed =
        network.place(/*batchSize=*/4, initializationDone, /*inferenceOnly=*/true, std::vector<int32_t>{0}, 1);
    for (Event& event : initializationDone) event.synchronize();
    EXPECT_NE(placed, nullptr);
    return contestsSoFar() - before;
}

struct DefaultDepthGuard {
    Depth saved = ThorImplementation::defaultCubReductionTuningDepth();
    ~DefaultDepthGuard() { ThorImplementation::setDefaultCubReductionTuningDepth(saved); }
};

}  // namespace

TEST(NetworkReductionTuningDepth, UnsetMeansRegularAndTheSettingRoundTrips) {
    Network network("tuning_depth_setting");
    EXPECT_FALSE(network.getReductionTuningDepthSetting().has_value());
    EXPECT_EQ(network.getReductionTuningDepth(), Depth::Regular);
    network.setReductionTuningDepth(Depth::Ultra);
    EXPECT_EQ(network.getReductionTuningDepthSetting(), std::optional<Depth>(Depth::Ultra));
    EXPECT_EQ(network.getReductionTuningDepth(), Depth::Ultra);
    network.setReductionTuningDepth(Depth::Quick);
    EXPECT_EQ(network.getReductionTuningDepth(), Depth::Quick);
    network.setReductionTuningDepth(std::nullopt);
    EXPECT_FALSE(network.getReductionTuningDepthSetting().has_value());
    EXPECT_EQ(network.getReductionTuningDepth(), Depth::Regular);
}

TEST(NetworkReductionTuningDepth, PlacementTunesAtTheNetworksDepth) {
    if (MachineEvaluator::instance().getNumGpus() == 0) GTEST_SKIP() << "placement requires a GPU";
    // Ultra first: an unset (Regular) and a Quick network with the same reduction reuse the Ultra selection.
    ASSERT_GE(placeFp8SumNetwork(50021, Depth::Ultra), 1U)
        << "no FP8 selection contest ran while placing an FP8 Sum metric network";
    EXPECT_EQ(placeFp8SumNetwork(50021, std::nullopt), 0U);
    EXPECT_EQ(placeFp8SumNetwork(50021, Depth::Quick), 0U);
    // Unset first (Regular): an Ultra network then runs its own contest; Quick reuses.
    EXPECT_GE(placeFp8SumNetwork(50033, std::nullopt), 1U);
    EXPECT_EQ(placeFp8SumNetwork(50033, Depth::Quick), 0U);
    EXPECT_GE(placeFp8SumNetwork(50033, Depth::Ultra), 1U);
}

TEST(NetworkReductionTuningDepth, AnUnsetNetworkIsRegularWhateverTheAmbientDepth) {
    if (MachineEvaluator::instance().getNumGpus() == 0) GTEST_SKIP() << "placement requires a GPU";
    DefaultDepthGuard guard;
    // Neither an Ultra process default nor an Ultra scope around place() makes an unset network tune at Ultra: it
    // tunes at Regular, so an Ultra network afterwards runs a contest of its own.
    ThorImplementation::setDefaultCubReductionTuningDepth(Depth::Ultra);
    {
        ThorImplementation::ScopedCubReductionTuningDepth ultra(Depth::Ultra);
        EXPECT_GE(placeFp8SumNetwork(50051, std::nullopt), 1U);
    }
    EXPECT_GE(placeFp8SumNetwork(50051, Depth::Ultra), 1U);
    // A Quick network under an Ultra scope tunes at Quick, and the scope is restored after placement.
    {
        ThorImplementation::ScopedCubReductionTuningDepth ultra(Depth::Ultra);
        EXPECT_GE(placeFp8SumNetwork(50069, Depth::Quick), 1U);
        EXPECT_EQ(ThorImplementation::currentCubReductionTuningDepth(), Depth::Ultra);
    }
    EXPECT_GE(placeFp8SumNetwork(50069, std::nullopt), 1U);  // Regular is deeper than the cached Quick
}

TEST(NetworkReductionTuningDepth, PlacementOnAWorkerThreadFollowsTheNetwork) {
    if (MachineEvaluator::instance().getNumGpus() == 0) GTEST_SKIP() << "placement requires a GPU";
    // Training runners place on their own threads: the network's depth applies there too.
    uint64_t ultra_contests = 0;
    std::thread worker([&] { ultra_contests = placeFp8SumNetwork(50077, Depth::Ultra); });
    worker.join();
    EXPECT_GE(ultra_contests, 1U);
    EXPECT_EQ(placeFp8SumNetwork(50077, std::nullopt), 0U);  // the worker's Ultra selection answers Regular
}
