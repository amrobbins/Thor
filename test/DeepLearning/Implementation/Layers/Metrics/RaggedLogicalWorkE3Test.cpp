#include "DeepLearning/Implementation/Layers/Metrics/RaggedAccuracy.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                         \
    do {                                                                                                              \
        int deviceCount = 0;                                                                                          \
        const cudaError_t status = cudaGetDeviceCount(&deviceCount);                                                  \
        if (status != cudaSuccess || deviceCount == 0) GTEST_SKIP() << "CUDA device is required for ragged tests."; \
    } while (false)

class PassiveEndpoint final : public Layer {
   public:
    void forward(std::optional<Tensor>, bool, uint32_t = 0) override {}
    void backward(std::optional<Tensor>, uint32_t = 0) override {}

   private:
    void infer(std::optional<Tensor>, std::optional<Tensor>, Stream) override {}
    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream) override {}
};

TEST(RaggedLogicalWork, Lwa4e3RaggedAccuracyUsesActivePrefixAndExcludesStructuralPartition) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 3;

    auto workForCapacity = [&](uint64_t capacity) {
        TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
        Stream stream(0);
        Tensor predictions(gpu, TensorDescriptor(DataType::FP32, {capacity, 1}));
        Tensor labels(gpu, TensorDescriptor(DataType::UINT8, {capacity, 1}));
        Tensor carrier(gpu, TensorDescriptor(DataType::UINT32, {1}));

        RaggedAccuracyMetric metric(RaggedAccuracyDetail::Kind::BINARY, batchSize, capacity);
        PassiveEndpoint predictionsSource, labelsSource, carrierSource, sink;
        EXPECT_FALSE(metric.connectToPreviousLayer(&predictionsSource,
                                                   predictions,
                                                   stream,
                                                   false,
                                                   static_cast<int>(Metric::ConnectionType::FORWARD))
                         .has_value());
        EXPECT_FALSE(metric.connectToPreviousLayer(&labelsSource,
                                                   labels,
                                                   stream,
                                                   false,
                                                   static_cast<int>(Metric::ConnectionType::LABELS))
                         .has_value());
        EXPECT_FALSE(metric.connectToPreviousLayer(&carrierSource,
                                                   carrier,
                                                   stream,
                                                   false,
                                                   static_cast<int>(Metric::ConnectionType::STRUCTURAL))
                         .has_value());
        metric.connectToNextLayer(&sink);
        metric.compile();

        const RowPartitionDescriptor descriptor(batchSize, capacity, DataType::UINT32);
        RowPartitionRuntime::publishHostState(carrier, descriptor, carrier.getTensorId(), {0, 2, 2, 5});

        // Five active binary predictions: 5*4 prediction bytes + 5*1 label
        // bytes + one FP32 semantic metric result. Structural partition bytes
        // and hidden aggregation workspaces are not logical tensor traffic.
        EXPECT_EQ(metric.logicalByteCountForward(0), 29U);
        EXPECT_EQ(metric.logicalByteCountForward(2), 14U);
        EXPECT_EQ(metric.logicalByteCountBackward(0), 0U);
        const std::vector<uint64_t> work{metric.logicalByteCountForward(0), metric.logicalByteCountForward(2)};

        RowPartitionRuntime::publishHostState(carrier, descriptor, carrier.getTensorId(), {0, 0, 0, 0});
        EXPECT_EQ(metric.logicalByteCountForward(0), sizeof(float));

        metric.cleanup();
        return work;
    };

    EXPECT_EQ(workForCapacity(8), workForCapacity(16));
}

}  // namespace
