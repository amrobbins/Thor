#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "test/DeepLearning/Implementation/Layers/LayerSynchronizationTestKernels.h"

#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Layers/Metrics/Mean.h"
#include "DeepLearning/Api/Layers/Metrics/WeightedMean.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
    bool entered = false;
    bool released = false;
};

struct WaitForHostGateArgs : public HostFunctionArgsBase {
    explicit WaitForHostGateArgs(shared_ptr<HostGate> gate) : gate(std::move(gate)) {}
    shared_ptr<HostGate> gate;
};

void waitForHostGate(void *rawArgs) {
    auto *args = static_cast<WaitForHostGateArgs *>(rawArgs);
    unique_lock<mutex> lock(args->gate->mtx);
    args->gate->entered = true;
    args->gate->cv.notify_all();
    args->gate->cv.wait(lock, [&] { return args->gate->released; });
}

bool waitForHostGateToEnter(const shared_ptr<HostGate> &gate, chrono::milliseconds timeout) {
    unique_lock<mutex> lock(gate->mtx);
    return gate->cv.wait_for(lock, timeout, [&] { return gate->entered; });
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

struct PlacedGradientStreamTarget {
    shared_ptr<Thor::PlacedNetwork> placedNetwork;
    vector<Stream> gradientUpdateStreams;
};

PlacedGradientStreamTarget makePlacedGradientStreamTarget(const string &networkName, uint32_t numTrainableLayers) {
    THOR_THROW_IF_FALSE(numTrainableLayers >= 1);

    Thor::Network network(networkName);
    Thor::NetworkInput input =
        Thor::NetworkInput::Builder().network(network).name("input").dimensions({4}).dataType(DataType::FP32).build();

    Thor::Tensor latest = input.getFeatureOutput().value();
    vector<uint64_t> fullyConnectedLayerIds;
    fullyConnectedLayerIds.reserve(numTrainableLayers);
    for (uint32_t i = 0; i < numTrainableLayers; ++i) {
        Thor::FullyConnected fullyConnected = Thor::FullyConnected::Builder()
                                                  .network(network)
                                                  .featureInput(latest)
                                                  .numOutputFeatures(4)
                                                  .hasBias(false)
                                                  .computeDataType(DataType::FP32)
                                                  .outputDataType(DataType::FP32)
                                                  .noActivation()
                                                  .build();
        latest = fullyConnected.getFeatureOutput().value();
        fullyConnectedLayerIds.push_back(fullyConnected.getId());
    }

    Thor::NetworkOutput::Builder().network(network).name("output").inputTensor(latest).dataType(DataType::FP32).build();
    Thor::Sgd::Builder().network(network).initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();

    vector<Event> initDoneEvents;
    shared_ptr<Thor::PlacedNetwork> placedNetwork = network.place(2,
                                                                  initDoneEvents,
                                                                  /*inferenceOnly=*/false,
                                                                  vector<int32_t>{0},
                                                                  /*forcedNumStampsPerGpu=*/1);
    for (Event &event : initDoneEvents)
        event.synchronize();

    vector<Stream> gradientUpdateStreams;
    gradientUpdateStreams.reserve(fullyConnectedLayerIds.size());
    for (uint64_t fullyConnectedLayerId : fullyConnectedLayerIds) {
        shared_ptr<TrainableLayer> physicalLayer =
            dynamic_pointer_cast<TrainableLayer>(placedNetwork->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fullyConnectedLayerId));
        THOR_THROW_IF_FALSE(physicalLayer != nullptr);
        THOR_THROW_IF_FALSE(physicalLayer->getGradientUpdateStream().has_value());
        gradientUpdateStreams.push_back(physicalLayer->getGradientUpdateStream().value());
    }

    return {placedNetwork, std::move(gradientUpdateStreams)};
}

PlacedSynchronizationTarget makePlacedSynchronizationTarget(const string &networkName) {
    Thor::Network network(networkName);
    Thor::NetworkInput input =
        Thor::NetworkInput::Builder().network(network).name("input").dimensions({4}).dataType(DataType::FP32).build();
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

    shared_ptr<TrainableLayer> physicalFullyConnected =
        dynamic_pointer_cast<TrainableLayer>(placedNetwork->getStampedNetwork(0).getPhysicalLayerFromApiLayer(fullyConnected.getId()));
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

    // This is the TrainableLayer half of the processingFinished contract:
    // StampedNetwork joins every stream returned by getProcessingStreams(), and
    // ProcessingFinishedEventJoinsDeferredSecondaryMetricStream below exercises
    // that join with a deliberately blocked secondary stream. Keep the gradient
    // stream in this list so trainable parameter-gradient/update work is covered
    // by the same batch-reuse barrier without having to gate a real optimizer
    // stream during host submission.
    vector<Stream> processingStreams = layer.getProcessingStreams();
    ASSERT_EQ(processingStreams.size(), 3u);
    EXPECT_EQ(processingStreams[0].getId(), dataStream0.getId());
    EXPECT_EQ(processingStreams[1].getId(), dataStream1.getId());
    EXPECT_EQ(processingStreams[2].getId(), gradientUpdateStream.getId());

    expectSynchronizationEventsCoverStreams(layer, {dataStream0, dataStream1, gradientUpdateStream}, 3);
}

TEST(LayerSynchronization, PlacedModelsOwnIndependentThreeStreamGradientUpdatePools) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Placed-network gradient stream-pool test requires a GPU";

    PlacedGradientStreamTarget firstModel = makePlacedGradientStreamTarget("GradientStreamModelA", 5);
    PlacedGradientStreamTarget secondModel = makePlacedGradientStreamTarget("GradientStreamModelB", 5);

    ASSERT_EQ(firstModel.gradientUpdateStreams.size(), 5u);
    ASSERT_EQ(secondModel.gradientUpdateStreams.size(), 5u);

    for (const vector<Stream> *modelStreams : {&firstModel.gradientUpdateStreams, &secondModel.gradientUpdateStreams}) {
        EXPECT_NE((*modelStreams)[0].getId(), (*modelStreams)[1].getId());
        EXPECT_NE((*modelStreams)[0].getId(), (*modelStreams)[2].getId());
        EXPECT_NE((*modelStreams)[1].getId(), (*modelStreams)[2].getId());
        EXPECT_EQ((*modelStreams)[3].getId(), (*modelStreams)[0].getId());
        EXPECT_EQ((*modelStreams)[4].getId(), (*modelStreams)[1].getId());
    }

    for (uint32_t firstIndex = 0; firstIndex < GradientUpdateStreamPool::MAX_STREAMS; ++firstIndex) {
        for (uint32_t secondIndex = 0; secondIndex < GradientUpdateStreamPool::MAX_STREAMS; ++secondIndex) {
            EXPECT_NE(firstModel.gradientUpdateStreams[firstIndex].getId(), secondModel.gradientUpdateStreams[secondIndex].getId());
        }
    }
}

TEST(LayerSynchronization, PlacedNetworkUsesLoaderPlacementsToElideDeviceInputRings) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Placed-network input placement test requires a GPU";

    PlacedSynchronizationTarget target = makePlacedSynchronizationTarget("DeviceInputPlacementNetwork");
    shared_ptr<NetworkInput> input = target.placedNetwork->getStampedNetwork(0).getNamedInput("input");
    ASSERT_NE(input, nullptr);
    EXPECT_EQ(input->getNumInputSlots(), 0u);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    target.placedNetwork->configureBatchInputPlacements({{"input", gpuPlacement}});
    target.placedNetwork->preallocateInputSlots(4);
    EXPECT_TRUE(input->isDeviceLoad());
    EXPECT_EQ(input->getNumInputSlots(), 0u);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    target.placedNetwork->configureBatchInputPlacements({{"input", cpuPlacement}});
    target.placedNetwork->preallocateInputSlots(4);
    EXPECT_FALSE(input->isDeviceLoad());
    EXPECT_EQ(input->getNumInputSlots(), 4u);
}

TEST(LayerSynchronization, NetworkInputIncludesItsUploadStream) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Layer synchronization event test requires a GPU";

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    NetworkInput input(gpuPlacement, DataType::FP32, vector<unsigned long>{4});

    vector<Stream> processingStreams = input.getProcessingStreams();
    ASSERT_EQ(processingStreams.size(), 1u)
        << "NetworkInput upload staging is slot-local and must stay outside the graph-processing barrier";
    EXPECT_EQ(processingStreams.front().getId(), input.getStream().getId());

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

    // Use GPU-side stream-memory-operation gates rather than blocking
    // cudaLaunchHostFunc callbacks or spinning kernels. Host callbacks can be
    // serialized across streams, while spinning kernels can occupy resources
    // required by the stream that releases them.
    ThorImplementation::Test::DeviceStreamGate modelGate(0);
    ThorImplementation::Test::DeviceStreamGate unrelatedGate(0);
    modelGate.enqueue(target.modelStream);
    unrelatedGate.enqueue(unrelatedStream);

    ASSERT_FALSE(modelGate.isComplete());
    ASSERT_FALSE(unrelatedGate.isComplete());

    auto synchronizeFuture = async(launch::async, [&] { target.placedNetwork->synchronize(); });

    EXPECT_EQ(synchronizeFuture.wait_for(chrono::milliseconds(100)), future_status::timeout)
        << "placed-network synchronization must wait for previously enqueued model work";

    modelGate.release();
    future_status afterModelRelease = synchronizeFuture.wait_for(chrono::seconds(10));
    EXPECT_EQ(afterModelRelease, future_status::ready)
        << "placed-network synchronization must not drain unrelated streams on the same CUDA device";
    if (afterModelRelease != future_status::ready) {
        unrelatedGate.release();
        ASSERT_EQ(synchronizeFuture.wait_for(chrono::seconds(10)), future_status::ready);
    }
    EXPECT_NO_THROW(synchronizeFuture.get());

    EXPECT_FALSE(unrelatedGate.isComplete()) << "the unrelated GPU-side gate must still be pending after placed-network synchronization";
    unrelatedGate.release();
    unrelatedStream.synchronize();
}

TEST(LayerSynchronization, ProcessingFinishedEventJoinsDeferredSecondaryMetricStream) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "Processing-finished stream-join regression requires a GPU";
    }

    constexpr uint32_t batchSize = 2;
    Thor::Network network("LayerSynchronizationProcessingJoinNetwork");
    Thor::NetworkInput values =
        Thor::NetworkInput::Builder().network(network).name("values").dimensions({1}).dataType(DataType::FP32).build();
    Thor::NetworkInput weights =
        Thor::NetworkInput::Builder().network(network).name("weights").dimensions({1}).dataType(DataType::FP32).build();

    // Mean is deliberately connected first so the shared values tensor receives
    // a TensorFanout and WeightedMean runs on a secondary fanout stream. The
    // weighted metric cannot enqueue its read of values until the independent
    // weights NetworkInput arrives later in StampedNetwork::sendPhysicalBatch().
    Thor::Mean mean = Thor::Mean::Builder().network(network).values(values.getFeatureOutput().value()).build();
    Thor::NetworkOutput::Builder().network(network).name("mean").inputTensor(mean.getMetric()).dataType(DataType::FP32).build();

    Thor::WeightedMean weightedMean = Thor::WeightedMean::Builder()
                                          .network(network)
                                          .values(values.getFeatureOutput().value())
                                          .weights(weights.getFeatureOutput().value())
                                          .build();
    Thor::NetworkOutput::Builder()
        .network(network)
        .name("weighted_mean")
        .inputTensor(weightedMean.getMetric())
        .dataType(DataType::FP32)
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<Thor::PlacedNetwork> placedNetwork = network.place(batchSize,
                                                                  initDoneEvents,
                                                                  /*inferenceOnly=*/true,
                                                                  vector<int32_t>{0},
                                                                  /*forcedNumStampsPerGpu=*/1);
    ASSERT_NE(placedNetwork, nullptr);
    for (Event &event : initDoneEvents)
        event.synchronize();

    placedNetwork->preallocateInputSlots(1);
    placedNetwork->preallocateOutputSlots(1);
    placedNetwork->synchronize();

    ThorImplementation::StampedNetwork &stamp = placedNetwork->getStampedNetwork(0);
    shared_ptr<ThorImplementation::Metric> physicalWeightedMean =
        dynamic_pointer_cast<ThorImplementation::Metric>(stamp.getPhysicalLayerFromApiLayer(weightedMean.getId()));
    shared_ptr<ThorImplementation::NetworkInput> physicalValues = stamp.getNamedInput("values");
    ASSERT_NE(physicalWeightedMean, nullptr);
    ASSERT_NE(physicalValues, nullptr);
    ASSERT_NE(physicalWeightedMean->getStream().getId(), physicalValues->getStream().getId())
        << "WeightedMean must be on the secondary TensorFanout stream for this regression.";

    ThorImplementation::Test::DeviceStreamGate weightedMetricGate(0);
    weightedMetricGate.enqueue(physicalWeightedMean->getStream());

    const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor valuesCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {batchSize, 1}));
    Tensor weightsCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {batchSize, 1}));
    valuesCpu.getMemPtr<float>()[0] = 2.0f;
    valuesCpu.getMemPtr<float>()[1] = 4.0f;
    weightsCpu.getMemPtr<float>()[0] = 1.0f;
    weightsCpu.getMemPtr<float>()[1] = 2.0f;

    Batch batch;
    batch.insert("values", valuesCpu);
    batch.insert("weights", weightsCpu);

    map<string, Tensor> outputs;
    map<string, Event> outputReadyEvents;
    Event processingFinished = placedNetwork->submitBatch(0,
                                                          batch,
                                                          outputs,
                                                          outputReadyEvents,
                                                          /*isInferenceOnly=*/true,
                                                          /*reusableProcessingFinishedEvent=*/nullptr,
                                                          /*waitForOutputsOnProcessingStream=*/false,
                                                          /*submitTiming=*/nullptr,
                                                          /*outputSlotIndex=*/0);

    auto processingWait = async(launch::async, [processingFinished]() mutable { processingFinished.synchronize(); });

    // The processing-finished boundary must include every secondary stream
    // declared by Layer::getProcessingStreams(). WeightedMean gives us a safe,
    // deterministic secondary-stream gate for that generic StampedNetwork
    // contract. TrainableLayerAlsoCoversGradientUpdateStream separately verifies
    // that a trainable layer declares its gradient/update stream, so the two tests
    // compose to cover gradient-stream participation without blocking a real
    // optimizer stream before host submission. Before the processing-stream fix,
    // this event was recorded on input 0 immediately, allowing the next queued
    // batch to overwrite statically connected values while a secondary consumer
    // was still waiting behind this gate.
    EXPECT_EQ(processingWait.wait_for(chrono::milliseconds(100)), future_status::timeout);

    weightedMetricGate.release();
    ASSERT_EQ(processingWait.wait_for(chrono::seconds(5)), future_status::ready);
    processingWait.get();
    outputReadyEvents.at("weighted_mean").synchronize();

    ASSERT_EQ(outputs.at("weighted_mean").getPlacement().getMemDevice(), TensorPlacement::MemDevices::CPU);
    EXPECT_NEAR(*outputs.at("weighted_mean").getMemPtr<float>(), 10.0f / 3.0f, 1e-6f);
}

TEST(LayerSynchronization, PlacedNetworkSaveWaitsForModelStreams) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Placed-network save synchronization test requires a GPU";

    PlacedSynchronizationTarget target = makePlacedSynchronizationTarget("LayerSynchronizationSaveNetwork");

    auto modelGate = make_shared<HostGate>();
    target.modelStream.enqueueHostFunction(&waitForHostGate, make_unique<WaitForHostGateArgs>(modelGate));

    const auto uniqueSuffix = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archiveDirectory = filesystem::temp_directory_path() / ("thor_layer_sync_save_" + to_string(uniqueSuffix));
    filesystem::remove_all(archiveDirectory);

    auto saveFuture = async(
        launch::async, [&] { target.placedNetwork->save(archiveDirectory.string(), /*overwrite=*/true, /*saveOptimizerState=*/true); });
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
