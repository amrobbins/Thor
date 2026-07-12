#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"

#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
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

class DelayedConnectedInputCaptureLayer : public Layer {
   public:
    explicit DelayedConnectedInputCaptureLayer(shared_ptr<HostGate> firstReadGate) : firstReadGate(std::move(firstReadGate)) {}

    void forward(std::optional<Tensor> forwardedFeatureInput, bool validationPass, uint32_t batchSize = 0) override {
        (void)validationPass;
        (void)batchSize;

        THOR_THROW_IF_FALSE(forwardedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(forwardedFeatureInput.value().getTensorId() == featureInput.value().getTensorId());

        forwardedTensorIds.push_back(forwardedFeatureInput.value().getTensorId());
        connectedTensorIds.push_back(featureInput.value().getTensorId());
        forwardedMemPtrs.push_back(forwardedFeatureInput.value().getMemPtr<void>());
        connectedMemPtrs.push_back(featureInput.value().getMemPtr<void>());
        THOR_THROW_IF_FALSE(forwardedMemPtrs.back() == connectedMemPtrs.back());

        if (captureOutputs.size() == invocationCount) {
            captureOutputs.push_back(featureInput.value().clone());
        }

        if (invocationCount == 0 && firstReadGate != nullptr) {
            stream.enqueueHostFunction(&waitForHostGate, std::make_unique<WaitForHostGateArgs>(firstReadGate));
        }

        // Intentionally copy from the connected featureInput member, not the forwarded argument.  That models a
        // stamped layer that captured the input tensor during connect/stamp and relies on Tensor backing indirection
        // to see later NetworkInput ping-pong swaps.
        captureOutputs[invocationCount].copyFromAsync(featureInput.value(), stream);
        ++invocationCount;
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream inferStream) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)inferStream;
        THOR_UNREACHABLE();
    }

    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream backPropStream) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)backPropStream;
        THOR_UNREACHABLE();
    }

    void synchronize() { stream.synchronize(); }

    vector<float> readCapture(uint32_t index) const {
        THOR_THROW_IF_FALSE(index < captureOutputs.size());
        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor cpu(cpuPlacement, captureOutputs[index].getDescriptor());
        Stream downloadStream(captureOutputs[index].getPlacement());
        cpu.copyFromAsync(captureOutputs[index], downloadStream);
        downloadStream.synchronize();

        const float *mem = cpu.getMemPtr<float>();
        return vector<float>(mem, mem + cpu.getTotalNumElements());
    }

    uint32_t invocationCount = 0;
    vector<uint64_t> forwardedTensorIds;
    vector<uint64_t> connectedTensorIds;
    vector<void *> forwardedMemPtrs;
    vector<void *> connectedMemPtrs;

   private:
    shared_ptr<HostGate> firstReadGate;
    vector<Tensor> captureOutputs;
};

Tensor makeFilledGpuTensor(TensorDescriptor descriptor, float value, Stream stream) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    Tensor tensor(gpuPlacement, descriptor);
    tensor.fill(value, stream);
    return tensor;
}

void expectAllEqual(const vector<float> &values, float expected) {
    for (uint32_t i = 0; i < values.size(); ++i) {
        ASSERT_EQ(values[i], expected) << "Mismatch at index " << i;
    }
}

}  // namespace

// Ping-pong was backed out.
// TEST(NetworkInput, ConnectedInputCopySeesBackingSwapsAndInactiveBackingWaitsForPriorConsumers) {
//     if (MachineEvaluator::instance().getNumGpus() == 0) {
//         GTEST_SKIP() << "NetworkInput ping-pong synchronization test requires a GPU";
//     }
//
//     TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
//     TensorDescriptor descriptor(DataType::FP32, {4096});
//     Stream setupStream(0);
//
//     Tensor batch1 = makeFilledGpuTensor(descriptor, 1.0f, setupStream);
//     Tensor batch2 = makeFilledGpuTensor(descriptor, 2.0f, setupStream);
//     Tensor batch3 = makeFilledGpuTensor(descriptor, 3.0f, setupStream);
//     setupStream.synchronize();
//
//     auto firstReadGate = make_shared<HostGate>();
//     NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions());
//     DelayedConnectedInputCaptureLayer capture(firstReadGate);
//     input.connectToNextLayer(&capture);
//
//     // Batch 1 uses backing A.  Its downstream read is intentionally delayed on the compute stream.
//     input.forward(batch1, false, 1);
//
//     // Batch 2 uses backing B while batch 1 still has queued consumers of backing A.
//     input.forward(batch2, false, 1);
//
//     // Batch 3 wants to reuse backing A.  Correct synchronization queues the upload behind batch 1's reusable event
//     // instead of overwriting backing A before the delayed batch-1 consumer has copied from it.  It should still
//     // return immediately on the host because the wait is enqueued on the upload stream, not synchronized here.
//     auto thirdForward = async(launch::async, [&] { input.forward(batch3, false, 1); });
//     if (thirdForward.wait_for(chrono::seconds(2)) != future_status::ready) {
//         releaseHostGate(firstReadGate);
//         thirdForward.wait();
//         FAIL() << "NetworkInput::forward blocked the host while waiting for the inactive backing to become reusable";
//     }
//     try {
//         thirdForward.get();
//     } catch (...) {
//         releaseHostGate(firstReadGate);
//         throw;
//     }
//
//     releaseHostGate(firstReadGate);
//     capture.synchronize();
//
//     ASSERT_EQ(capture.invocationCount, 3u);
//     ASSERT_EQ(capture.forwardedTensorIds.size(), 3u);
//     ASSERT_EQ(capture.connectedTensorIds.size(), 3u);
//     ASSERT_EQ(capture.forwardedMemPtrs.size(), 3u);
//     ASSERT_EQ(capture.connectedMemPtrs.size(), 3u);
//
//     // The logical tensor identity is stable for an already-connected/stamped downstream layer.
//     EXPECT_EQ(capture.forwardedTensorIds[0], capture.forwardedTensorIds[1]);
//     EXPECT_EQ(capture.forwardedTensorIds[1], capture.forwardedTensorIds[2]);
//     EXPECT_EQ(capture.connectedTensorIds[0], capture.forwardedTensorIds[0]);
//     EXPECT_EQ(capture.connectedTensorIds[1], capture.forwardedTensorIds[1]);
//     EXPECT_EQ(capture.connectedTensorIds[2], capture.forwardedTensorIds[2]);
//
//     // The backing allocation alternates, and the third batch reuses the first backing only after the first consumer
//     // has completed.
//     EXPECT_NE(capture.connectedMemPtrs[0], capture.connectedMemPtrs[1]);
//     EXPECT_EQ(capture.connectedMemPtrs[0], capture.connectedMemPtrs[2]);
//
//     expectAllEqual(capture.readCapture(0), 1.0f);
//     expectAllEqual(capture.readCapture(1), 2.0f);
//     expectAllEqual(capture.readCapture(2), 3.0f);
// }

namespace {

class RuntimeForwardedInputCaptureLayer : public Layer {
   public:
    void forward(std::optional<Tensor> forwardedFeatureInput, bool validationPass, uint32_t batchSize = 0) override {
        (void)validationPass;
        (void)batchSize;
        THOR_THROW_IF_FALSE(forwardedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());

        forwardedTensorIds.push_back(forwardedFeatureInput.value().getTensorId());
        connectedTensorIds.push_back(featureInput.value().getTensorId());
        forwardedMemPtrs.push_back(forwardedFeatureInput.value().getMemPtr<void>());
        connectedMemPtrs.push_back(featureInput.value().getMemPtr<void>());

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
        Tensor cpuCopy(cpuPlacement, forwardedFeatureInput.value().getDescriptor());
        cpuCopy.copyFromAsync(forwardedFeatureInput.value(), stream);
        capturedCopies.push_back(cpuCopy);
        ++invocationCount;
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream inferStream) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)inferStream;
        THOR_UNREACHABLE();
    }

    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream backPropStream) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)backPropStream;
        THOR_UNREACHABLE();
    }

    void synchronize() { stream.synchronize(); }

    vector<float> readCapture(uint32_t index) const {
        THOR_THROW_IF_FALSE(index < capturedCopies.size());
        // The copy into capturedCopies was enqueued on this layer's stream.  The
        // tests call synchronize() before reading, so the CPU tensor is ready.
        const float *mem = capturedCopies[index].getMemPtr<float>();
        return vector<float>(mem, mem + capturedCopies[index].getTotalNumElements());
    }

    uint32_t invocationCount = 0;
    vector<uint64_t> forwardedTensorIds;
    vector<uint64_t> connectedTensorIds;
    vector<void *> forwardedMemPtrs;
    vector<void *> connectedMemPtrs;

   private:
    vector<Tensor> capturedCopies;
};

class FixedOutputBackpropCaptureLayer : public Layer {
   public:
    explicit FixedOutputBackpropCaptureLayer(Tensor output) : output(std::move(output)) {}

    std::optional<Tensor> createFeatureOutputTensor() override { return output; }

    bool isBackPropStub() override { return false; }

    void backward(std::optional<Tensor> incomingErrorInput, uint32_t batchSize = 0) override {
        (void)batchSize;
        capturedErrors.push_back(incomingErrorInput);
    }

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream inferStream) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)inferStream;
        THOR_UNREACHABLE();
    }

    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream backPropStream) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)backPropStream;
        THOR_UNREACHABLE();
    }

    Tensor output;
    vector<std::optional<Tensor>> capturedErrors;
};

class BackpropSinkLayer : public Layer {
   public:
    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream inferStream) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)inferStream;
        THOR_UNREACHABLE();
    }

    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream backPropStream) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)backPropStream;
        THOR_UNREACHABLE();
    }
};

}  // namespace

// Network composition primitive: a pass-through NetworkInput binds an upstream
// physical tensor before downstream layers are connected. The downstream layer
// therefore sees the upstream tensor's stable identity at connect/stamp time and
// at runtime; forward() no longer substitutes a different tensor later.
TEST(NetworkInput, PassThroughBindsSourceTensorIdentityBeforeDownstreamConnect) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput pass-through test requires a GPU";
    }

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});
    Stream setupStream(0);
    Tensor source = makeFilledGpuTensor(descriptor, 7.0f, setupStream);
    setupStream.synchronize();

    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions(), NetworkInput::Mode::PassThrough, source);
    EXPECT_TRUE(input.isPassThrough());
    EXPECT_FALSE(input.requiresBatchInput());

    RuntimeForwardedInputCaptureLayer capture;
    input.connectToNextLayer(&capture);

    ASSERT_TRUE(input.getFeatureOutput().has_value());
    ASSERT_EQ(input.getFeatureOutput().value().getTensorId(), source.getTensorId());
    ASSERT_EQ(input.getFeatureOutput().value().getMemPtr<void>(), source.getMemPtr<void>());

    // These are intentionally no-ops for pass-through inputs; pass-through
    // composition must not allocate external-load staging slots.
    input.preallocateInputSlots(3);
    input.setActiveInputSlot(2);

    input.forward(source, false, 1);
    capture.synchronize();

    ASSERT_EQ(capture.invocationCount, 1u);
    ASSERT_EQ(capture.connectedTensorIds[0], source.getTensorId());
    ASSERT_EQ(capture.forwardedTensorIds[0], source.getTensorId());
    ASSERT_EQ(capture.connectedMemPtrs[0], source.getMemPtr<void>());
    ASSERT_EQ(capture.forwardedMemPtrs[0], source.getMemPtr<void>());
    expectAllEqual(capture.readCapture(0), 7.0f);
}

TEST(NetworkInput, PassThroughConnectsAndForwardsBackwardErrorAsIdentity) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput pass-through backward test requires a GPU";
    }

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});
    Stream setupStream(0);
    Tensor source = makeFilledGpuTensor(descriptor, 1.0f, setupStream);
    setupStream.synchronize();

    FixedOutputBackpropCaptureLayer producer(source);
    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions(), NetworkInput::Mode::PassThrough);
    BackpropSinkLayer sink;

    producer.connectToNextLayer(&input);
    ASSERT_TRUE(input.getErrorOutput().has_value());
    ASSERT_TRUE(producer.getErrorInput().has_value());
    const Tensor provisionalError = input.getErrorOutput().value();
    EXPECT_EQ(producer.getErrorInput().value(), provisionalError);

    input.connectToNextLayer(&sink);
    ASSERT_TRUE(sink.getErrorOutput().has_value());
    ASSERT_TRUE(input.getErrorInput().has_value());
    ASSERT_TRUE(input.getErrorOutput().has_value());
    ASSERT_TRUE(producer.getErrorInput().has_value());

    // The pass-through input is a differentiable identity edge: once the
    // downstream side exists, the provisional upstream error tensor is fused
    // into the downstream error tensor instead of materializing a copy.
    EXPECT_NE(provisionalError, sink.getErrorOutput().value());
    EXPECT_EQ(input.getErrorInput().value(), sink.getErrorOutput().value());
    EXPECT_EQ(input.getErrorOutput().value(), sink.getErrorOutput().value());
    EXPECT_EQ(producer.getErrorInput().value(), sink.getErrorOutput().value());

    input.backward(sink.getErrorOutput(), 1);

    ASSERT_EQ(producer.capturedErrors.size(), 1u);
    ASSERT_TRUE(producer.capturedErrors[0].has_value());
    EXPECT_EQ(producer.capturedErrors[0].value(), sink.getErrorOutput().value());
}

TEST(NetworkInput, ExternalLoadInputRemainsBackwardBoundary) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput backward-boundary test requires a GPU";
    }

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});

    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    BackpropSinkLayer sink;
    input.connectToNextLayer(&sink);

    EXPECT_FALSE(input.getErrorInput().has_value());
    EXPECT_FALSE(input.getErrorOutput().has_value());
    EXPECT_FALSE(sink.getErrorOutput().has_value());
}


// Default behavior remains conservative: regular NetworkInputs materialize into
// their owned feature tensor unless constructed in explicit pass-through mode.
TEST(NetworkInput, SamePlacementInputStillCopiesByDefault) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput same-placement default copy test requires a GPU";
    }

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});
    Stream setupStream(0);
    Tensor externalInput = makeFilledGpuTensor(descriptor, 3.0f, setupStream);
    setupStream.synchronize();

    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    EXPECT_FALSE(input.isPassThrough());
    EXPECT_TRUE(input.requiresBatchInput());
    RuntimeForwardedInputCaptureLayer capture;
    input.connectToNextLayer(&capture);

    input.forward(externalInput, false, 1);
    capture.synchronize();

    ASSERT_EQ(capture.invocationCount, 1u);
    ASSERT_EQ(capture.forwardedTensorIds[0], input.getFeatureOutput().value().getTensorId());
    ASSERT_NE(capture.forwardedTensorIds[0], externalInput.getTensorId());
    EXPECT_EQ(capture.forwardedMemPtrs[0], input.getFeatureOutput().value().getMemPtr<void>());
    expectAllEqual(capture.readCapture(0), 3.0f);
}

TEST(NetworkInput, DeviceLoadCopiesDirectlyWithoutAllocatingInputSlots) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput device-load test requires a GPU";
    }

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});
    Stream setupStream(0);
    Tensor deviceBatchTensor = makeFilledGpuTensor(descriptor, 5.0f, setupStream);
    setupStream.synchronize();

    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    RuntimeForwardedInputCaptureLayer capture;
    input.connectToNextLayer(&capture);

    EXPECT_EQ(input.getNumInputSlots(), 0u);
    input.configureBatchInputPlacement(gpuPlacement);
    EXPECT_TRUE(input.isDeviceLoad());
    input.preallocateInputSlots(4);
    input.setActiveInputSlot(3);
    EXPECT_EQ(input.getNumInputSlots(), 0u);

    input.forward(deviceBatchTensor, false, 1);
    capture.synchronize();

    ASSERT_EQ(capture.invocationCount, 1u);
    ASSERT_EQ(capture.forwardedTensorIds[0], input.getFeatureOutput().value().getTensorId());
    ASSERT_NE(capture.forwardedTensorIds[0], deviceBatchTensor.getTensorId());
    EXPECT_EQ(capture.forwardedMemPtrs[0], input.getFeatureOutput().value().getMemPtr<void>());
    expectAllEqual(capture.readCapture(0), 5.0f);
}

TEST(NetworkInput, DeviceLoadCanFallBackToHostStagingBeforeSubmission) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput device-load fallback test requires a GPU";
    }

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});

    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    RuntimeForwardedInputCaptureLayer capture;
    input.connectToNextLayer(&capture);

    input.configureBatchInputPlacement(gpuPlacement);
    input.preallocateInputSlots(4);
    ASSERT_TRUE(input.isDeviceLoad());
    ASSERT_EQ(input.getNumInputSlots(), 0u);

    input.configureBatchInputPlacement(cpuPlacement);
    input.preallocateInputSlots(4);
    EXPECT_FALSE(input.isDeviceLoad());
    EXPECT_EQ(input.getNumInputSlots(), 4u);
}

TEST(NetworkInput, RepeatedPlacementConfigurationIsIdempotentButModeChangeWithSlotsIsRejected) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput placement reconfiguration test requires a GPU";
    }

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});

    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    RuntimeForwardedInputCaptureLayer capture;
    input.connectToNextLayer(&capture);

    input.configureBatchInputPlacement(cpuPlacement);
    input.preallocateInputSlots(4);
    ASSERT_EQ(input.getNumInputSlots(), 4u);

    EXPECT_NO_THROW(input.configureBatchInputPlacement(cpuPlacement));
    EXPECT_THROW(input.configureBatchInputPlacement(gpuPlacement), std::logic_error);
}

TEST(NetworkInput, HostBatchPlacementRetainsRequestedStagingRingDepth) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput staged-load test requires a GPU";
    }

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});

    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    RuntimeForwardedInputCaptureLayer capture;
    input.connectToNextLayer(&capture);

    EXPECT_EQ(input.getNumInputSlots(), 0u);
    input.configureBatchInputPlacement(cpuPlacement);
    EXPECT_FALSE(input.isDeviceLoad());
    input.preallocateInputSlots(4);
    EXPECT_EQ(input.getNumInputSlots(), 4u);
}

TEST(NetworkInput, PassThroughRejectsDifferentRuntimeTensor) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput pass-through validation test requires a GPU";
    }

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});
    Stream setupStream(0);
    Tensor source = makeFilledGpuTensor(descriptor, 1.0f, setupStream);
    Tensor differentSource = makeFilledGpuTensor(descriptor, 2.0f, setupStream);
    setupStream.synchronize();

    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions(), NetworkInput::Mode::PassThrough, source);
    RuntimeForwardedInputCaptureLayer capture;
    input.connectToNextLayer(&capture);

    EXPECT_ANY_THROW(input.forward(differentSource, false, 1));
}

TEST(NetworkInput, PassThroughRejectsDescriptorOrPlacementMismatch) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput pass-through validation test requires a GPU";
    }

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4});
    TensorDescriptor wrongDescriptor(DataType::FP32, {5});
    Tensor cpuSource(cpuPlacement, descriptor);
    Tensor gpuWrongShape(gpuPlacement, wrongDescriptor);

    EXPECT_ANY_THROW(NetworkInput(gpuPlacement, DataType::FP32, descriptor.getDimensions(), NetworkInput::Mode::PassThrough, cpuSource));
    EXPECT_ANY_THROW(NetworkInput(gpuPlacement, DataType::FP32, descriptor.getDimensions(), NetworkInput::Mode::PassThrough, gpuWrongShape));
}
