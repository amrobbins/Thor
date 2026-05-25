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

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream backPropStream) override {
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

TEST(NetworkInput, ConnectedInputCopySeesBackingSwapsAndInactiveBackingWaitsForPriorConsumers) {
    if (MachineEvaluator::instance().getNumGpus() == 0) {
        GTEST_SKIP() << "NetworkInput ping-pong synchronization test requires a GPU";
    }

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {4096});
    Stream setupStream(0);

    Tensor batch1 = makeFilledGpuTensor(descriptor, 1.0f, setupStream);
    Tensor batch2 = makeFilledGpuTensor(descriptor, 2.0f, setupStream);
    Tensor batch3 = makeFilledGpuTensor(descriptor, 3.0f, setupStream);
    setupStream.synchronize();

    auto firstReadGate = make_shared<HostGate>();
    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    DelayedConnectedInputCaptureLayer capture(firstReadGate);
    input.connectToNextLayer(&capture);

    // Batch 1 uses backing A.  Its downstream read is intentionally delayed on the compute stream.
    input.forward(batch1, false, 1);

    // Batch 2 uses backing B while batch 1 still has queued consumers of backing A.
    input.forward(batch2, false, 1);

    // Batch 3 wants to reuse backing A.  Correct synchronization queues the upload behind batch 1's reusable event
    // instead of overwriting backing A before the delayed batch-1 consumer has copied from it.  It should still
    // return immediately on the host because the wait is enqueued on the upload stream, not synchronized here.
    auto thirdForward = async(launch::async, [&] { input.forward(batch3, false, 1); });
    if (thirdForward.wait_for(chrono::seconds(2)) != future_status::ready) {
        releaseHostGate(firstReadGate);
        thirdForward.wait();
        FAIL() << "NetworkInput::forward blocked the host while waiting for the inactive backing to become reusable";
    }
    try {
        thirdForward.get();
    } catch (...) {
        releaseHostGate(firstReadGate);
        throw;
    }

    releaseHostGate(firstReadGate);
    capture.synchronize();

    ASSERT_EQ(capture.invocationCount, 3u);
    ASSERT_EQ(capture.forwardedTensorIds.size(), 3u);
    ASSERT_EQ(capture.connectedTensorIds.size(), 3u);
    ASSERT_EQ(capture.forwardedMemPtrs.size(), 3u);
    ASSERT_EQ(capture.connectedMemPtrs.size(), 3u);

    // The logical tensor identity is stable for an already-connected/stamped downstream layer.
    EXPECT_EQ(capture.forwardedTensorIds[0], capture.forwardedTensorIds[1]);
    EXPECT_EQ(capture.forwardedTensorIds[1], capture.forwardedTensorIds[2]);
    EXPECT_EQ(capture.connectedTensorIds[0], capture.forwardedTensorIds[0]);
    EXPECT_EQ(capture.connectedTensorIds[1], capture.forwardedTensorIds[1]);
    EXPECT_EQ(capture.connectedTensorIds[2], capture.forwardedTensorIds[2]);

    // The backing allocation alternates, and the third batch reuses the first backing only after the first consumer
    // has completed.
    EXPECT_NE(capture.connectedMemPtrs[0], capture.connectedMemPtrs[1]);
    EXPECT_EQ(capture.connectedMemPtrs[0], capture.connectedMemPtrs[2]);

    expectAllEqual(capture.readCapture(0), 1.0f);
    expectAllEqual(capture.readCapture(1), 2.0f);
    expectAllEqual(capture.readCapture(2), 3.0f);
}
