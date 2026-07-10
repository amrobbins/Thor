#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"

#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"

#include <chrono>
#include <condition_variable>
#include <future>
#include <filesystem>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "Utilities/ComputeTopology/MachineEvaluator.h"

#pragma GCC diagnostic ignored "-Wsign-compare"
#include "gtest/gtest.h"
#pragma GCC diagnostic pop

using namespace ThorImplementation;
using namespace std;

namespace {

struct HostGate {
    mutex mtx;
    condition_variable cv;
    bool released = false;
};

struct WaitForHostGateArgs : public HostFunctionArgsBase {
    explicit WaitForHostGateArgs(shared_ptr<HostGate> gate) : gate(std::move(gate)) {}
    shared_ptr<HostGate> gate;
};

void waitForHostGate(void *rawArgs) {
    auto *args = static_cast<WaitForHostGateArgs *>(rawArgs);
    unique_lock<mutex> lock(args->gate->mtx);
    args->gate->cv.wait(lock, [&] { return args->gate->released; });
}

void releaseHostGate(const shared_ptr<HostGate> &gate) {
    {
        lock_guard<mutex> lock(gate->mtx);
        gate->released = true;
    }
    gate->cv.notify_all();
}

class ReleaseAllGates {
   public:
    explicit ReleaseAllGates(vector<shared_ptr<HostGate>> &gates) : gates(gates) {}
    ~ReleaseAllGates() {
        for (const auto &gate : gates)
            releaseHostGate(gate);
    }

   private:
    vector<shared_ptr<HostGate>> &gates;
};

class SynchronizationTestLayer final : public Layer {
   public:
    void setDataStream(Stream dataStream) { stream = std::move(dataStream); }

   protected:
    void infer(optional<Tensor>, optional<Tensor>, Stream) override {}
    void backProp(optional<Tensor>, optional<Tensor>, optional<Tensor>, Stream) override {}
};

class SynchronizationTestMultiConnectionLayer final : public MultiConnectionLayer {
   public:
    void setDataStreams(vector<Stream> dataStreams) { streams = std::move(dataStreams); }

   protected:
    void infer(optional<Tensor>, optional<Tensor>, Stream, unsigned int) override {}
    void backProp(optional<Tensor>, optional<Tensor>, optional<Tensor>, Stream, unsigned int) override {}
};

class SynchronizationTestTrainableLayer final : public TrainableLayer {
   public:
    explicit SynchronizationTestTrainableLayer(const TensorPlacement &placement) : TrainableLayer(placement, false) {}

    void setDataStreams(vector<Stream> dataStreams) { streams = std::move(dataStreams); }
    void setGradientStream(Stream updateStream) { gradientUpdateStream = std::move(updateStream); }

   protected:
    void computeFeatureOut(uint32_t) override {}
    string getLayerType() override { return "SynchronizationTestTrainableLayer"; }
    uint64_t flopCountForward() override { return 0; }
    uint64_t flopCountBackward() override { return 0; }
};

struct PlacedSynchronizationTarget {
    shared_ptr<Thor::PlacedNetwork> placedNetwork;
    Stream modelStream;
};

PlacedSynchronizationTarget makePlacedSynchronizationTarget(const string &networkName) {
    Thor::Network network(networkName);
    Thor::NetworkInput input = Thor::NetworkInput::Builder()
                                   .network(network)
                                   .name("input")
                                   .dimensions({4})
                                   .dataType(DataType::FP32)
                                   .build();
    Thor::FullyConnected fullyConnected = Thor::FullyConnected::Builder()
                                               .network(network)
                                               .featureInput(input.getFeatureOutput().value())
                                               .numOutputFeatures(3)
                                               .hasBias(false)
                                               .computeDataType(DataType::FP32)
                                               .outputDataType(DataType::FP32)
                                               .noActivation()
                                               .build();
    Thor::NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(fullyConnected.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();
    Thor::Sgd::Builder().network(network).initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();

    vector<Event> initDoneEvents;
    shared_ptr<Thor::PlacedNetwork> placedNetwork = network.place(2,
                                                                   initDoneEvents,
                                                                   /*inferenceOnly=*/false,
                                                                   vector<int32_t>{0},
                                                                   /*forcedNumStampsPerGpu=*/1);
    for (Event &event : initDoneEvents)
        event.synchronize();

    shared_ptr<TrainableLayer> physicalFullyConnected = dynamic_pointer_cast<TrainableLayer>(
        placedNetwork->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fullyConnected.getId()));
    THOR_THROW_IF_FALSE(physicalFullyConnected != nullptr);
    THOR_THROW_IF_FALSE(physicalFullyConnected->getGradientUpdateStream().has_value());
    return {placedNetwork, physicalFullyConnected->getGradientUpdateStream().value()};
}

void expectSynchronizationEventsCoverStreams(Layer &layer, const vector<Stream> &streamsToBlock, size_t expectedEventCount) {
    vector<shared_ptr<HostGate>> gates;
    gates.reserve(streamsToBlock.size());
    ReleaseAllGates releaseAll(gates);

    for (Stream stream : streamsToBlock) {
        auto gate = make_shared<HostGate>();
        gates.push_back(gate);
        stream.enqueueHostFunction(&waitForHostGate, make_unique<WaitForHostGateArgs>(gate));
    }

    vector<Event> synchronizeEvents = layer.getSynchronizeEvents();
    ASSERT_EQ(synchronizeEvents.size(), expectedEventCount);

    auto synchronization = async(launch::async, [events = std::move(synchronizeEvents)]() mutable {
        for (Event &event : events)
            event.synchronize();
    });

    EXPECT_EQ(synchronization.wait_for(chrono::milliseconds(100)), future_status::timeout);

    for (size_t i = 0; i < gates.size(); ++i) {
        releaseHostGate(gates[i]);
        if (i + 1 < gates.size())
            EXPECT_EQ(synchronization.wait_for(chrono::milliseconds(100)), future_status::timeout);
    }

    ASSERT_EQ(synchronization.wait_for(chrono::seconds(5)), future_status::ready);
    synchronization.get();
}

}  // namespace

TEST(LayerSynchronization, UnconnectedLayerHasNoSynchronizeEvents) {
    SynchronizationTestLayer layer;
    EXPECT_TRUE(layer.getSynchronizeEvents().empty());
}

TEST(LayerSynchronization, LayerEventCoversPreviouslyEnqueuedDataStreamWork) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Layer synchronization event test requires a GPU";

    Stream dataStream(0);
    SynchronizationTestLayer layer;
    layer.setDataStream(dataStream);

    expectSynchronizationEventsCoverStreams(layer, {dataStream}, 1);
}

TEST(LayerSynchronization, MultiConnectionLayerCoversEveryDistinctDataStream) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Layer synchronization event test requires a GPU";

    Stream dataStream0(0);
    Stream dataStream1(0);
    SynchronizationTestMultiConnectionLayer layer;
    layer.setDataStreams({dataStream0, dataStream1, dataStream0});

    expectSynchronizationEventsCoverStreams(layer, {dataStream0, dataStream1}, 2);
}

TEST(LayerSynchronization, TrainableLayerAlsoCoversGradientUpdateStream) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Layer synchronization event test requires a GPU";

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Stream dataStream0(0);
    Stream dataStream1(0);
    Stream gradientUpdateStream(0);
    SynchronizationTestTrainableLayer layer(gpuPlacement);
    layer.setDataStreams({dataStream0, dataStream1});
    layer.setGradientStream(gradientUpdateStream);

    expectSynchronizationEventsCoverStreams(layer, {dataStream0, dataStream1, gradientUpdateStream}, 3);
}

TEST(LayerSynchronization, NetworkInputIncludesItsUploadStream) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Layer synchronization event test requires a GPU";

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    NetworkInput input(gpuPlacement, DataType::FP32, vector<unsigned long>{4});

    vector<Event> synchronizeEvents = input.getSynchronizeEvents();
    EXPECT_EQ(synchronizeEvents.size(), 2u);
    for (Event &event : synchronizeEvents)
        event.synchronize();
}

TEST(LayerSynchronization, PlacedNetworkSynchronizeWaitsForModelStreamsWithoutDrainingUnrelatedWork) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Placed-network synchronization test requires a GPU";

    PlacedSynchronizationTarget target = makePlacedSynchronizationTarget("LayerSynchronizationBoundaryNetwork");
    Stream unrelatedStream(0);

    auto modelGate = make_shared<HostGate>();
    auto unrelatedGate = make_shared<HostGate>();
    target.modelStream.enqueueHostFunction(&waitForHostGate, make_unique<WaitForHostGateArgs>(modelGate));
    unrelatedStream.enqueueHostFunction(&waitForHostGate, make_unique<WaitForHostGateArgs>(unrelatedGate));

    auto synchronizeFuture = async(launch::async, [&] { target.placedNetwork->synchronize(); });
    vector<shared_ptr<HostGate>> gates{modelGate, unrelatedGate};
    ReleaseAllGates releaseAll(gates);

    EXPECT_EQ(synchronizeFuture.wait_for(chrono::milliseconds(100)), future_status::timeout)
        << "placed-network synchronization must wait for previously enqueued model work";

    releaseHostGate(modelGate);
    future_status afterModelRelease = synchronizeFuture.wait_for(chrono::seconds(10));
    EXPECT_EQ(afterModelRelease, future_status::ready)
        << "placed-network synchronization must not drain unrelated streams on the same CUDA device";
    if (afterModelRelease != future_status::ready) {
        releaseHostGate(unrelatedGate);
        ASSERT_EQ(synchronizeFuture.wait_for(chrono::seconds(10)), future_status::ready);
    }
    EXPECT_NO_THROW(synchronizeFuture.get());

    releaseHostGate(unrelatedGate);
    unrelatedStream.synchronize();
}


TEST(LayerSynchronization, PlacedNetworkSaveWaitsForModelStreams) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Placed-network save synchronization test requires a GPU";

    PlacedSynchronizationTarget target = makePlacedSynchronizationTarget("LayerSynchronizationSaveNetwork");

    auto modelGate = make_shared<HostGate>();
    target.modelStream.enqueueHostFunction(&waitForHostGate, make_unique<WaitForHostGateArgs>(modelGate));

    const auto uniqueSuffix = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archiveDirectory =
        filesystem::temp_directory_path() / ("thor_layer_sync_save_" + to_string(uniqueSuffix));
    filesystem::remove_all(archiveDirectory);

    auto saveFuture = async(launch::async, [&] {
        target.placedNetwork->save(archiveDirectory.string(), /*overwrite=*/true, /*saveOptimizerState=*/true);
    });
    vector<shared_ptr<HostGate>> gates{modelGate};
    ReleaseAllGates releaseAll(gates);

    EXPECT_EQ(saveFuture.wait_for(chrono::milliseconds(100)), future_status::timeout)
        << "save must wait for work already enqueued on the placed network's streams";

    releaseHostGate(modelGate);
    ASSERT_EQ(saveFuture.wait_for(chrono::seconds(30)), future_status::ready)
        << "save did not finish after the placed network's work completed";
    EXPECT_NO_THROW(saveFuture.get());

    filesystem::remove_all(archiveDirectory);
}
