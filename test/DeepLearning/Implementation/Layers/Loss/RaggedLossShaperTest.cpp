#include "DeepLearning/Implementation/Layers/Loss/RaggedLossShaper.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int cudaDeviceCountForTest = 0;                                                                                \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                            \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                         \
            GTEST_SKIP() << "CUDA device is required for RaggedLossShaper execution tests.";                          \
        }                                                                                                              \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t tensorNumel(const Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dimension : tensor.getDimensions())
        numel *= dimension;
    return numel;
}

void writeCpuFloatingTensor(Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensorNumel(tensor), values.size());
    if (tensor.getDataType() == DataType::FP16) {
        half* ptr = static_cast<half*>(tensor.getMemPtr());
        for (size_t i = 0; i < values.size(); ++i)
            ptr[i] = __float2half(values[i]);
    } else if (tensor.getDataType() == DataType::FP32) {
        float* ptr = tensor.getMemPtr<float>();
        for (size_t i = 0; i < values.size(); ++i)
            ptr[i] = values[i];
    } else {
        FAIL() << "unsupported floating dtype";
    }
}

Tensor makeGpuFloatingTensor(DataType dtype,
                             const std::vector<uint64_t>& dimensions,
                             const std::vector<float>& values,
                             Stream& stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtype, dimensions));
    writeCpuFloatingTensor(host, values);
    Tensor device(gpuPlacement, TensorDescriptor(dtype, dimensions));
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

void overwriteGpuFloatingTensor(Tensor device, const std::vector<float>& values, Stream& stream) {
    Tensor host(cpuPlacement, device.getDescriptor());
    writeCpuFloatingTensor(host, values);
    device.copyFromAsync(host, stream);
    stream.synchronize();
}

std::vector<float> readGpuFloatingTensor(const Tensor& device, Stream& stream) {
    Tensor host(cpuPlacement, device.getDescriptor());
    host.copyFromAsync(device, stream);
    stream.synchronize();
    std::vector<float> values(tensorNumel(host));
    if (host.getDataType() == DataType::FP16) {
        const half* ptr = static_cast<const half*>(host.getMemPtr());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = __half2float(ptr[i]);
    } else if (host.getDataType() == DataType::FP32) {
        const float* ptr = host.getMemPtr<float>();
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = ptr[i];
    } else {
        ADD_FAILURE() << "unsupported floating dtype";
    }
    return values;
}

Tensor makeGpuOffsets(DataType dtype, const std::vector<uint64_t>& values, Stream& stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtype, {static_cast<uint64_t>(values.size())}));
    if (dtype == DataType::UINT32) {
        uint32_t* ptr = host.getMemPtr<uint32_t>();
        for (size_t i = 0; i < values.size(); ++i)
            ptr[i] = static_cast<uint32_t>(values[i]);
    } else if (dtype == DataType::UINT64) {
        uint64_t* ptr = host.getMemPtr<uint64_t>();
        for (size_t i = 0; i < values.size(); ++i)
            ptr[i] = values[i];
    } else {
        throw std::invalid_argument("offset dtype must be UINT32 or UINT64");
    }
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

void overwriteGpuOffsets(Tensor device, const std::vector<uint64_t>& values, Stream& stream) {
    Tensor replacement = makeGpuOffsets(device.getDataType(), values, stream);
    device.copyFromAsync(replacement, stream);
    stream.synchronize();
}

class PassiveEndpoint final : public Layer {
   public:
    void forward(std::optional<Tensor> input, bool validationPass, uint32_t batchSize = 0) override {
        (void)validationPass;
        lastForward = input;
        lastForwardBatchSize = batchSize;
    }

    void backward(std::optional<Tensor> error, uint32_t batchSize = 0) override {
        lastBackward = error;
        lastBackwardBatchSize = batchSize;
    }

    std::optional<Tensor> lastForward;
    std::optional<Tensor> lastBackward;
    uint32_t lastForwardBatchSize = 0;
    uint32_t lastBackwardBatchSize = 0;

   private:
    void infer(std::optional<Tensor>, std::optional<Tensor>, Stream) override {}
    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream) override {}
};

struct ShaperFixture {
    PassiveEndpoint valuesSource;
    PassiveEndpoint offsetsSource;
    PassiveEndpoint sink;
    RaggedLossShaper shaper;

    ShaperFixture(RaggedLossShaper::OutputLossType type,
                  uint64_t batchSize,
                  uint64_t maxTotalValues,
                  Tensor values,
                  Tensor offsets,
                  Stream stream)
        : shaper(type, batchSize, maxTotalValues) {
        shaper.connectToPreviousLayer(&valuesSource,
                                      values,
                                      stream,
                                      false,
                                      static_cast<int>(RaggedLossShaper::InputConnection::VALUES));
        shaper.connectToPreviousLayer(&offsetsSource,
                                      offsets,
                                      stream,
                                      false,
                                      static_cast<int>(RaggedLossShaper::InputConnection::OFFSETS));
        shaper.connectToNextLayer(&sink);
        shaper.compile();
        shaper.initialize();
    }

    ~ShaperFixture() {
        if (shaper.isCompiled())
            shaper.cleanup();
    }
};

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
}

void runPerExampleCase(DataType valueDType, DataType offsetsDType) {
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 8;
    constexpr uint64_t width = 2;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    const float tolerance = valueDType == DataType::FP16 ? 3.0e-2f : 1.0e-5f;
    Stream stream(0);

    // Rows contain 2, 0, 3, 0 tokens. Packed capacity after offsets[B] is poison.
    Tensor values = makeGpuFloatingTensor(valueDType,
                                          {maxTotalValues, width},
                                          {1.0f, 2.0f,
                                           3.0f, 4.0f,
                                           5.0f, 6.0f,
                                           7.0f, 8.0f,
                                           9.0f, 10.0f,
                                           poison, poison,
                                           poison, poison,
                                           poison, poison},
                                          stream);
    Tensor offsets = makeGpuOffsets(offsetsDType, {0, 2, 2, 5, 5}, stream);

    ShaperFixture fixture(RaggedLossShaper::OutputLossType::PER_EXAMPLE,
                          batchSize,
                          maxTotalValues,
                          values,
                          offsets,
                          stream);
    ASSERT_EQ(fixture.shaper.getFeatureOutput().value().getDimensions(), (std::vector<uint64_t>{batchSize, 1}));

    fixture.shaper.forward(values, false, batchSize);
    fixture.shaper.forward(offsets, false, batchSize);
    ASSERT_TRUE(fixture.sink.lastForward.has_value());
    EXPECT_EQ(fixture.sink.lastForwardBatchSize, batchSize);
    expectNear(readGpuFloatingTensor(fixture.sink.lastForward.value(), stream), {10.0f, 0.0f, 45.0f, 0.0f}, tolerance);
}

}  // namespace

TEST(RaggedLossShaper, RejectsPerOutputSemantics) {
    EXPECT_THROW(RaggedLossShaper(RaggedLossShaper::OutputLossType::PER_OUTPUT, 4, 8), std::invalid_argument);
}

TEST(RaggedLossShaper, RequestedBackpropagationIsPrunedForReportingInputs) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    PassiveEndpoint valuesSource;
    PassiveEndpoint offsetsSource;
    RaggedLossShaper shaper(RaggedLossShaper::OutputLossType::BATCH, 3, 6);

    Tensor values = makeGpuFloatingTensor(DataType::FP32, {6}, {1, 2, 3, 4, 5, 6}, stream);
    Tensor offsets = makeGpuOffsets(DataType::UINT32, {0, 2, 2, 3}, stream);

    std::optional<Tensor> valuesError;
    EXPECT_NO_THROW(valuesError = shaper.connectToPreviousLayer(
                        &valuesSource,
                        values,
                        stream,
                        true,
                        static_cast<int>(RaggedLossShaper::InputConnection::VALUES)));
    EXPECT_FALSE(valuesError.has_value());

    std::optional<Tensor> offsetsError;
    EXPECT_NO_THROW(offsetsError = shaper.connectToPreviousLayer(
                        &offsetsSource,
                        offsets,
                        stream,
                        true,
                        static_cast<int>(RaggedLossShaper::InputConnection::OFFSETS)));
    EXPECT_FALSE(offsetsError.has_value());
}

TEST(RaggedLossShaper, RawAliasesPackedValuesAndPreservesPartition) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor values = makeGpuFloatingTensor(DataType::FP32, {6}, {1, 2, 3, 99, 100, 101}, stream);
    Tensor offsets = makeGpuOffsets(DataType::UINT32, {0, 2, 2, 3}, stream);

    ShaperFixture fixture(RaggedLossShaper::OutputLossType::RAW, 3, 6, values, offsets, stream);
    ASSERT_TRUE(fixture.shaper.getFeatureOutput().has_value());
    EXPECT_EQ(fixture.shaper.getFeatureOutput().value(), values);
    ASSERT_TRUE(fixture.shaper.getRawOutputOffsets().has_value());
    EXPECT_EQ(fixture.shaper.getRawOutputOffsets().value(), offsets);

    fixture.shaper.forward(values, false, 3);
    fixture.shaper.forward(offsets, false, 3);
    ASSERT_TRUE(fixture.sink.lastForward.has_value());
    EXPECT_EQ(fixture.sink.lastForward.value(), values);
    expectNear(readGpuFloatingTensor(values, stream), {1, 2, 3, 99, 100, 101}, 0.0f);
}

TEST(RaggedLossShaper, PerExampleSumsActiveTokensAndAllTrailingValuesForBothOffsetWidths) {
    REQUIRE_CUDA_DEVICE();
    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        runPerExampleCase(DataType::FP32, offsetsDType);
        runPerExampleCase(DataType::FP16, offsetsDType);
    }
}

TEST(RaggedLossShaper, BatchAveragesPerRowSumsOverValidLogicalExamples) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 7;

    // Only the first two logical rows are valid. Invalid tail rows are canonical
    // empty rows. Row sums are 3 and 7, so BATCH = (3 + 7) / 2 = 5.
    Tensor values = makeGpuFloatingTensor(DataType::FP32,
                                          {maxTotalValues},
                                          {1, 2, 7, std::numeric_limits<float>::quiet_NaN(),
                                           std::numeric_limits<float>::quiet_NaN(),
                                           std::numeric_limits<float>::quiet_NaN(),
                                           std::numeric_limits<float>::quiet_NaN()},
                                          stream);
    Tensor offsets = makeGpuOffsets(DataType::UINT64, {0, 2, 3, 3, 3}, stream);
    ShaperFixture fixture(RaggedLossShaper::OutputLossType::BATCH,
                          batchSize,
                          maxTotalValues,
                          values,
                          offsets,
                          stream);

    fixture.shaper.forward(values, false, 2);
    fixture.shaper.forward(offsets, false, 2);
    ASSERT_TRUE(fixture.sink.lastForward.has_value());
    EXPECT_EQ(fixture.sink.lastForwardBatchSize, 2U);
    expectNear(readGpuFloatingTensor(fixture.sink.lastForward.value(), stream), {5.0f}, 1.0e-5f);
}

TEST(RaggedLossShaper, EmptyValidRowsStillCountInBatchDenominator) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 6;

    // Four valid rows with row sums 0, 6, 0, 0 => BATCH is 1.5, not 6.
    Tensor values = makeGpuFloatingTensor(DataType::FP32,
                                          {maxTotalValues},
                                          {6, 99, 100, 101, 102, 103},
                                          stream);
    Tensor offsets = makeGpuOffsets(DataType::UINT32, {0, 0, 1, 1, 1}, stream);
    ShaperFixture fixture(RaggedLossShaper::OutputLossType::BATCH,
                          batchSize,
                          maxTotalValues,
                          values,
                          offsets,
                          stream);

    fixture.shaper.forward(values, false, batchSize);
    fixture.shaper.forward(offsets, false, batchSize);
    expectNear(readGpuFloatingTensor(fixture.sink.lastForward.value(), stream), {1.5f}, 1.0e-5f);
}

TEST(RaggedLossShaper, AllEmptyBatchProducesExactZeros) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 5;
    Tensor values = makeGpuFloatingTensor(DataType::FP32,
                                          {maxTotalValues, 2},
                                          std::vector<float>(maxTotalValues * 2, std::numeric_limits<float>::quiet_NaN()),
                                          stream);
    Tensor offsets = makeGpuOffsets(DataType::UINT64, {0, 0, 0, 0}, stream);

    {
        ShaperFixture fixture(RaggedLossShaper::OutputLossType::PER_EXAMPLE,
                              batchSize,
                              maxTotalValues,
                              values,
                              offsets,
                              stream);
        fixture.shaper.forward(values, false, batchSize);
        fixture.shaper.forward(offsets, false, batchSize);
        expectNear(readGpuFloatingTensor(fixture.sink.lastForward.value(), stream), {0, 0, 0}, 0.0f);
    }
    {
        ShaperFixture fixture(RaggedLossShaper::OutputLossType::BATCH,
                              batchSize,
                              maxTotalValues,
                              values,
                              offsets,
                              stream);
        fixture.shaper.forward(values, false, batchSize);
        fixture.shaper.forward(offsets, false, batchSize);
        expectNear(readGpuFloatingTensor(fixture.sink.lastForward.value(), stream), {0}, 0.0f);
    }
}

TEST(RaggedLossShaper, ReusesStampedPlansAcrossShortLongShortPartitions) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 7;
    Tensor values = makeGpuFloatingTensor(DataType::FP32, {maxTotalValues}, {2, 3, 4, 5, 6, 7, 8}, stream);
    Tensor offsets = makeGpuOffsets(DataType::UINT32, {0, 1, 1, 2}, stream);
    ShaperFixture fixture(RaggedLossShaper::OutputLossType::PER_EXAMPLE,
                          batchSize,
                          maxTotalValues,
                          values,
                          offsets,
                          stream);

    auto run = [&](const std::vector<float>& packed,
                   const std::vector<uint64_t>& partition,
                   const std::vector<float>& expected) {
        overwriteGpuFloatingTensor(values, packed, stream);
        overwriteGpuOffsets(offsets, partition, stream);
        fixture.shaper.forward(values, false, batchSize);
        fixture.shaper.forward(offsets, false, batchSize);
        expectNear(readGpuFloatingTensor(fixture.sink.lastForward.value(), stream), expected, 1.0e-5f);
    };

    run({2, 3, 99, 99, 99, 99, 99}, {0, 1, 1, 2}, {2, 0, 3});
    run({1, 2, 3, 4, 5, 6, 99}, {0, 2, 5, 6}, {3, 12, 6});
    run({7, 8, 99, 99, 99, 99, 99}, {0, 0, 1, 2}, {0, 7, 8});
}

TEST(RaggedLossShaper, InputsMustAgreeOnLogicalBatchCardinality) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor values = makeGpuFloatingTensor(DataType::FP32, {4}, {1, 2, 3, 4}, stream);
    Tensor offsets = makeGpuOffsets(DataType::UINT32, {0, 1, 2}, stream);
    ShaperFixture fixture(RaggedLossShaper::OutputLossType::BATCH, 2, 4, values, offsets, stream);

    fixture.shaper.forward(values, false, 1);
    EXPECT_THROW(fixture.shaper.forward(offsets, false, 2), std::invalid_argument);
}
