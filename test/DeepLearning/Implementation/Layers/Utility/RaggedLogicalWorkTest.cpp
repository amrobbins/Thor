#include "DeepLearning/Implementation/Layers/Utility/PaddedDenseToRagged.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedConcatenate.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedToPaddedDense.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int device_count = 0;                                                                                          \
        const cudaError_t status = cudaGetDeviceCount(&device_count);                                                  \
        if (status != cudaSuccess || device_count == 0) GTEST_SKIP() << "CUDA device is required for ragged tests."; \
    } while (false)

class RaggedConcatenateLogicalWorkProbe final : public RaggedConcatenate {
   public:
    using RaggedConcatenate::RaggedConcatenate;

    void bind(const std::vector<Tensor>& values,
              Tensor partitionCarrier,
              Tensor output,
              Tensor outputGradient,
              const std::vector<std::optional<Tensor>>& inputGradients) {
        ASSERT_EQ(values.size(), inputGradients.size());
        ASSERT_EQ(featureInputs.size(), values.size() + 1);
        ASSERT_EQ(errorOutputs.size(), values.size() + 1);
        for (std::size_t i = 0; i < values.size(); ++i) {
            featureInputs[i] = values[i];
            errorOutputs[i] = inputGradients[i];
        }
        featureInputs[values.size()] = partitionCarrier;
        errorOutputs[values.size()] = std::nullopt;
        featureOutputs = {output};
        errorInputs = {outputGradient};
    }
};

class PaddedDenseToRaggedLogicalWorkProbe final : public PaddedDenseToRagged {
   public:
    using PaddedDenseToRagged::PaddedDenseToRagged;

    void bind(Tensor dense, Tensor offsets, Tensor values, Tensor dValues, Tensor dDense) {
        featureInputs = {dense, offsets};
        featureOutputs = {values};
        errorInputs = {dValues};
        errorOutputs = {dDense, std::nullopt};
    }
};

class RaggedToPaddedDenseLogicalWorkProbe final : public RaggedToPaddedDense {
   public:
    using RaggedToPaddedDense::RaggedToPaddedDense;

    void bind(Tensor values, Tensor offsets, Tensor dense, Tensor dDense, Tensor dValues) {
        featureInputs = {values, offsets};
        featureOutputs = {dense};
        errorInputs = {dDense};
        errorOutputs = {dValues, std::nullopt};
    }
};

TEST(RaggedLogicalWork, Lwa4e1RaggedConcatenateUsesActivePrefixAndAuthoredGradientsOnly) {
    REQUIRE_CUDA_DEVICE();
    TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint64_t batch_size = 3;
    constexpr uint64_t valid_rows = 2;

    auto work_for_capacity = [&](uint64_t capacity) {
        Tensor input0(gpu, TensorDescriptor(DataType::FP32, {capacity, 2}));
        Tensor input1(gpu, TensorDescriptor(DataType::FP32, {capacity, 3}));
        Tensor output(gpu, TensorDescriptor(DataType::FP32, {capacity, 5}));
        Tensor dOutput(gpu, TensorDescriptor(DataType::FP32, {capacity, 5}));
        Tensor dInput0(gpu, TensorDescriptor(DataType::FP32, {capacity, 2}));
        Tensor dInput1(gpu, TensorDescriptor(DataType::FP32, {capacity, 3}));
        Tensor carrier(gpu, TensorDescriptor(DataType::UINT32, {1}));

        const RowPartitionDescriptor partition(batch_size, capacity, DataType::UINT32);
        RowPartitionRuntime::publishHostState(carrier, partition, carrier.getTensorId(), {0, 2, 5, 7});

        RaggedConcatenateLogicalWorkProbe concatenate(/*valuesAxis=*/1, /*expectedValueInputs=*/2, batch_size);
        concatenate.bind({input0, input1}, carrier, output, dOutput, {dInput0, dInput1});

        constexpr uint64_t bytes_per_active_row = (2 + 3 + 5) * sizeof(float);
        EXPECT_EQ(concatenate.logicalByteCountForward(0), 7U * bytes_per_active_row);
        EXPECT_EQ(concatenate.logicalByteCountForward(valid_rows), 5U * bytes_per_active_row);
        EXPECT_EQ(concatenate.logicalByteCountBackward(0), 7U * bytes_per_active_row);
        EXPECT_EQ(concatenate.logicalByteCountBackward(valid_rows), 5U * bytes_per_active_row);

        const std::vector<uint64_t> observed = {
            concatenate.logicalByteCountForward(0), concatenate.logicalByteCountForward(valid_rows),
            concatenate.logicalByteCountBackward(0), concatenate.logicalByteCountBackward(valid_rows)};

        // A pruned logical gradient output must not be charged merely because the
        // physical split implementation keeps a discard buffer for that branch.
        concatenate.bind({input0, input1}, carrier, output, dOutput, {dInput0, std::nullopt});
        constexpr uint64_t pruned_backward_bytes_per_row = (5 + 2) * sizeof(float);
        EXPECT_EQ(concatenate.logicalByteCountBackward(valid_rows), 5U * pruned_backward_bytes_per_row);

        // Mutating only rows after the submitted prefix leaves partial-batch work unchanged.
        RowPartitionRuntime::publishHostState(carrier, partition, carrier.getTensorId(), {0, 2, 5, 8});
        EXPECT_EQ(concatenate.logicalByteCountForward(valid_rows), 5U * bytes_per_active_row);
        EXPECT_EQ(concatenate.logicalByteCountForward(0), 8U * bytes_per_active_row);

        return observed;
    };

    EXPECT_EQ(work_for_capacity(8), work_for_capacity(16));
}

TEST(RaggedLogicalWork, Lwa4e1DenseRaggedAdaptersIgnorePackedAndPaddedCapacity) {
    REQUIRE_CUDA_DEVICE();
    TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint64_t batch_size = 3;
    constexpr uint64_t channels = 4;
    constexpr uint64_t valid_rows = 2;

    auto work_for_capacities = [&](uint64_t packed_capacity, uint64_t padded_width) {
        const RaggedTensorDescriptor ragged(
            DataType::FP32, {channels}, batch_size, packed_capacity, padded_width, DataType::UINT32);
        const TensorDescriptor denseDescriptor(DataType::FP32, {batch_size, padded_width, channels});

        Tensor dense(gpu, denseDescriptor);
        Tensor values(gpu, ragged.getValuesDescriptor());
        Tensor offsets(gpu, ragged.getOffsetsDescriptor());
        Tensor dDense(gpu, denseDescriptor);
        Tensor dValues(gpu, ragged.getValuesDescriptor());

        // Device offsets intentionally remain untouched. Logical-work telemetry
        // must use only the authoritative host publication below.
        RowPartitionRuntime partition(offsets, ragged.getRowPartition());
        partition.setHostOffsets({0, 2, 5, 7});

        PaddedDenseToRaggedLogicalWorkProbe toRagged(denseDescriptor, ragged, ragged);
        toRagged.bind(dense, offsets, values, dValues, dDense);

        RaggedToPaddedDenseLogicalWorkProbe toDense(ragged, denseDescriptor, 0.0);
        toDense.bind(values, offsets, dense, dDense, dValues);

        constexpr uint64_t bytes_per_active_value = channels * sizeof(float) * 2U;
        EXPECT_EQ(toRagged.logicalByteCountForward(0), 7U * bytes_per_active_value);
        EXPECT_EQ(toRagged.logicalByteCountForward(valid_rows), 5U * bytes_per_active_value);
        EXPECT_EQ(toRagged.logicalByteCountBackward(0), 7U * bytes_per_active_value);
        EXPECT_EQ(toRagged.logicalByteCountBackward(valid_rows), 5U * bytes_per_active_value);

        EXPECT_EQ(toDense.logicalByteCountForward(0), 7U * bytes_per_active_value);
        EXPECT_EQ(toDense.logicalByteCountForward(valid_rows), 5U * bytes_per_active_value);
        EXPECT_EQ(toDense.logicalByteCountBackward(0), 7U * bytes_per_active_value);
        EXPECT_EQ(toDense.logicalByteCountBackward(valid_rows), 5U * bytes_per_active_value);

        const std::vector<uint64_t> observed = {
            toRagged.logicalByteCountForward(0), toRagged.logicalByteCountForward(valid_rows),
            toRagged.logicalByteCountBackward(0), toRagged.logicalByteCountBackward(valid_rows),
            toDense.logicalByteCountForward(0), toDense.logicalByteCountForward(valid_rows),
            toDense.logicalByteCountBackward(0), toDense.logicalByteCountBackward(valid_rows)};

        partition.setHostOffsets({0, 0, 0, 0});
        EXPECT_EQ(toRagged.logicalByteCountForward(0), 0U);
        EXPECT_EQ(toRagged.logicalByteCountBackward(0), 0U);
        EXPECT_EQ(toDense.logicalByteCountForward(0), 0U);
        EXPECT_EQ(toDense.logicalByteCountBackward(0), 0U);

        return observed;
    };

    // Same semantic partition, but both packed capacity and explicit dense
    // padding width grow. Neither is useful logical value traffic.
    EXPECT_EQ(work_for_capacities(8, 6), work_for_capacities(16, 10));
}

}  // namespace
