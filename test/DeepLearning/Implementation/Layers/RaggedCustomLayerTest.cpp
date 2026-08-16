#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int cudaDeviceCountForTest = 0;                                                                                \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                            \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                         \
            GTEST_SKIP() << "CUDA device is required for RaggedCustomLayer execution tests.";                         \
        }                                                                                                              \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t tensorNumel(const Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dimension : tensor.getDimensions()) {
        numel *= dimension;
    }
    return numel;
}

Tensor makeGpuFp32Tensor(const std::vector<uint64_t>& dimensions, const std::vector<float>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, dimensions));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("makeGpuFp32Tensor value count mismatch.");
    }
    float* ptr = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dimensions));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

Tensor makeGpuU32Tensor(const std::vector<uint64_t>& dimensions, const std::vector<uint32_t>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::UINT32, dimensions));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("makeGpuU32Tensor value count mismatch.");
    }
    uint32_t* ptr = cpu.getMemPtr<uint32_t>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::UINT32, dimensions));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

void overwriteGpuFp32Tensor(Tensor& gpu, const std::vector<float>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, gpu.getDimensions()));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("overwriteGpuFp32Tensor value count mismatch.");
    }
    float* ptr = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
}

std::vector<float> readGpuFp32Tensor(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const float* ptr = cpu.getMemPtr<float>();
    return std::vector<float>(ptr, ptr + tensorNumel(cpu));
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1.0e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], atol) << "index " << i;
    }
}

class PassiveEndpoint final : public Layer {
   public:
    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        (void)validationPass;
        (void)batchSize;
        lastForward = featureInput;
    }

    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        (void)batchSize;
        lastBackward = errorInput;
    }

    std::optional<Tensor> lastForward;
    std::optional<Tensor> lastBackward;

   private:
    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)stream;
    }

    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream stream) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)stream;
    }
};

class FixedTrainableVectorParameter final : public PhysicalParameter {
   public:
    FixedTrainableVectorParameter(std::string name, std::vector<float> initialValues)
        : PhysicalParameter(std::move(name), true), initialValues(std::move(initialValues)) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& x = context.getInput("x");
        if (x.getDataType() != DataType::FP32) {
            throw std::runtime_error("FixedTrainableVectorParameter supports FP32 only.");
        }
        storage = Tensor(x.getPlacement(), TensorDescriptor(DataType::FP32, {static_cast<uint64_t>(initialValues.size())}));
        Tensor host(cpuPlacement, TensorDescriptor(DataType::FP32, {static_cast<uint64_t>(initialValues.size())}));
        float* ptr = host.getMemPtr<float>();
        for (size_t i = 0; i < initialValues.size(); ++i) ptr[i] = initialValues[i];
        Stream initStream = gradientUpdateStream.has_value() ? gradientUpdateStream.value() : Stream(x.getPlacement());
        storage->copyFromAsync(host, initStream);
        initStream.synchronize();
    }

   private:
    std::vector<float> initialValues;
};

DynamicExpression buildTrainableScaleRaggedExpression(uint64_t batchSize, uint64_t fullCapacityRows, uint64_t width) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression offsets = Expression::input(RaggedCustomLayer::RAGGED_OFFSETS_INPUT_NAME, std::nullopt, DataType::UINT32);
    const Expression y = (x * scale).withRaggedRuntimeExtent(offsets, batchSize, fullCapacityRows, width);
    return DynamicExpression::fromExpressionDefinition(
        ExpressionDefinition::fromOutputs(Expression::outputs({{"y", y}})));
}

DynamicExpression buildTwoInputTwoOutputRaggedExpression(uint64_t batchSize, uint64_t fullCapacityRows) {
    const RaggedTensorDescriptor descriptor(DataType::FP32, {4}, batchSize, fullCapacityRows, DataType::UINT32);
    const Expression offsets = Expression::input("offsets", std::nullopt, DataType::UINT32);
    const RaggedExpression lhs(Expression::input("lhs", std::nullopt, DataType::FP32), offsets, descriptor);
    const RaggedExpression rhs(Expression::input("rhs", std::nullopt, DataType::FP32), offsets, descriptor);
    const RaggedExpression wide = lhs + rhs;
    const RaggedExpression narrow = wide.sliceLastDimension(1, 2);

    const ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({
        {"wide", wide.getValues()},
        {"narrow", narrow.getValues()},
    }));
    return DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

TEST(RaggedCustomLayer, MultiInputMultiOutputPreservesActivePrefixAndCanonicalizesEveryTail) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchSize = 3;
    constexpr uint64_t fullCapacityRows = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint64_t inputWidth = 4;
    constexpr uint64_t narrowWidth = 2;

    Stream stream(0);
    std::vector<float> lhsValues(fullCapacityRows * inputWidth, 12345.0f);
    std::vector<float> rhsValues(fullCapacityRows * inputWidth, -23456.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t column = 0; column < inputWidth; ++column) {
            lhsValues[row * inputWidth + column] = static_cast<float>(10 * row + column + 1);
            rhsValues[row * inputWidth + column] = static_cast<float>(100 + 7 * row - 2 * column);
        }
    }

    Tensor lhs = makeGpuFp32Tensor({fullCapacityRows, inputWidth}, lhsValues, stream);
    Tensor rhs = makeGpuFp32Tensor({fullCapacityRows, inputWidth}, rhsValues, stream);
    Tensor offsets = makeGpuU32Tensor({batchSize + 1}, {0, 3, 3, 5}, stream);
    RowPartitionRuntime(offsets, RowPartitionDescriptor(batchSize, fullCapacityRows, DataType::UINT32))
        .setHostActiveValueCount(activeRows);

    RaggedCustomLayer layer(buildTwoInputTwoOutputRaggedExpression(batchSize, fullCapacityRows),
                            {"lhs", "rhs", "offsets"},
                            {"wide", "narrow"},
                            gpuPlacement,
                            std::vector<std::shared_ptr<PhysicalParameter>>{},
                            false,
                            fullCapacityRows,
                            {inputWidth, inputWidth},
                            {inputWidth, narrowWidth},
                            {0, 1},
                            2,
                            -1,
                            {{DataType::FP32, {inputWidth}, false}, {DataType::FP32, {narrowWidth}, false}});

    PassiveEndpoint lhsSource;
    PassiveEndpoint rhsSource;
    PassiveEndpoint offsetsSource;
    PassiveEndpoint wideSink;
    PassiveEndpoint narrowSink;

    ASSERT_TRUE(layer.connectToPreviousLayer(&lhsSource, lhs, stream, true, 0).has_value());
    ASSERT_TRUE(layer.connectToPreviousLayer(&rhsSource, rhs, stream, true, 1).has_value());
    ASSERT_FALSE(layer.connectToPreviousLayer(&offsetsSource, offsets, stream, false, 2).has_value());
    layer.connectToNextLayer(&wideSink, 0, 0);
    layer.connectToNextLayer(&narrowSink, 1, 0);

    layer.compile();
    layer.initialize();

    layer.forward(lhs, false, batchSize);
    layer.forward(rhs, false, batchSize);
    layer.forward(offsets, false, batchSize);
    ASSERT_TRUE(wideSink.lastForward.has_value());
    ASSERT_TRUE(narrowSink.lastForward.has_value());

    const std::vector<float> wide = readGpuFp32Tensor(wideSink.lastForward.value(), stream);
    const std::vector<float> narrow = readGpuFp32Tensor(narrowSink.lastForward.value(), stream);

    std::vector<float> expectedWide(fullCapacityRows * inputWidth, 0.0f);
    std::vector<float> expectedNarrow(fullCapacityRows * narrowWidth, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t column = 0; column < inputWidth; ++column) {
            expectedWide[row * inputWidth + column] = lhsValues[row * inputWidth + column] + rhsValues[row * inputWidth + column];
        }
        expectedNarrow[row * narrowWidth] = expectedWide[row * inputWidth + 1];
        expectedNarrow[row * narrowWidth + 1] = expectedWide[row * inputWidth + 2];
    }
    expectNear(wide, expectedWide);
    expectNear(narrow, expectedNarrow);

    ASSERT_TRUE(wideSink.getErrorOutput().has_value());
    ASSERT_TRUE(narrowSink.getErrorOutput().has_value());
    Tensor wideGradient = wideSink.getErrorOutput().value();
    Tensor narrowGradient = narrowSink.getErrorOutput().value();
    overwriteGpuFp32Tensor(wideGradient, std::vector<float>(fullCapacityRows * inputWidth, 1.0f), stream);
    overwriteGpuFp32Tensor(narrowGradient, std::vector<float>(fullCapacityRows * narrowWidth, 2.0f), stream);

    layer.backward(wideGradient, batchSize);
    layer.backward(narrowGradient, batchSize);

    const auto inputGradients = layer.getErrorOutputs();
    ASSERT_EQ(inputGradients.size(), 3u);
    ASSERT_TRUE(inputGradients[0].has_value());
    ASSERT_TRUE(inputGradients[1].has_value());
    ASSERT_FALSE(inputGradients[2].has_value());

    std::vector<float> expectedInputGradient(fullCapacityRows * inputWidth, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        expectedInputGradient[row * inputWidth] = 1.0f;
        expectedInputGradient[row * inputWidth + 1] = 3.0f;
        expectedInputGradient[row * inputWidth + 2] = 3.0f;
        expectedInputGradient[row * inputWidth + 3] = 1.0f;
    }
    expectNear(readGpuFp32Tensor(inputGradients[0].value(), stream), expectedInputGradient);
    expectNear(readGpuFp32Tensor(inputGradients[1].value(), stream), expectedInputGradient);

    layer.cleanup();
}

TEST(RaggedCustomLayer, GpuOffsetsRequireHostActiveValueCountOnRowPartition) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchSize = 3;
    constexpr uint64_t fullCapacityRows = 8;
    constexpr uint64_t inputWidth = 4;

    Stream stream(0);
    Tensor lhs = makeGpuFp32Tensor({fullCapacityRows, inputWidth}, std::vector<float>(fullCapacityRows * inputWidth, 1.0f), stream);
    Tensor rhs = makeGpuFp32Tensor({fullCapacityRows, inputWidth}, std::vector<float>(fullCapacityRows * inputWidth, 2.0f), stream);
    Tensor offsets = makeGpuU32Tensor({batchSize + 1}, {0, 3, 3, 5}, stream);
    ASSERT_FALSE(RowPartitionRuntime(offsets, RowPartitionDescriptor(batchSize, fullCapacityRows, DataType::UINT32))
                     .getHostActiveValueCountIfAvailable()
                     .has_value());

    RaggedCustomLayer layer(buildTwoInputTwoOutputRaggedExpression(batchSize, fullCapacityRows),
                            {"lhs", "rhs", "offsets"},
                            {"wide", "narrow"},
                            gpuPlacement,
                            std::vector<std::shared_ptr<PhysicalParameter>>{},
                            true,
                            fullCapacityRows,
                            {inputWidth, inputWidth},
                            {inputWidth, 2},
                            {0, 1},
                            2);

    PassiveEndpoint lhsSource;
    PassiveEndpoint rhsSource;
    PassiveEndpoint offsetsSource;
    PassiveEndpoint wideSink;
    PassiveEndpoint narrowSink;

    layer.connectToPreviousLayer(&lhsSource, lhs, stream, false, 0);
    layer.connectToPreviousLayer(&rhsSource, rhs, stream, false, 1);
    layer.connectToPreviousLayer(&offsetsSource, offsets, stream, false, 2);
    layer.connectToNextLayer(&wideSink, 0, 0);
    layer.connectToNextLayer(&narrowSink, 1, 0);
    layer.compile();
    layer.initialize();

    layer.forward(lhs, false, batchSize);
    layer.forward(rhs, false, batchSize);
    EXPECT_THROW(layer.forward(offsets, false, batchSize), std::runtime_error);

    layer.cleanup();
}

TEST(RaggedCustomLayer, TrainableParameterGradientAndSgdUpdateIgnoreInactivePackedCapacity) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchSize = 3;
    constexpr uint64_t fullCapacityRows = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint64_t width = 4;
    constexpr float learningRate = 0.03f;

    Stream stream(0);
    std::vector<float> xValues(fullCapacityRows * width, 100000.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < width; ++col) {
            xValues[row * width + col] = static_cast<float>(1 + row * width + col) * 0.1f;
        }
    }
    Tensor x = makeGpuFp32Tensor({fullCapacityRows, width}, xValues, stream);
    Tensor offsets = makeGpuU32Tensor({batchSize + 1}, {0, 3, 3, 5}, stream);
    RowPartitionRuntime(offsets, RowPartitionDescriptor(batchSize, fullCapacityRows, DataType::UINT32))
        .setHostActiveValueCount(activeRows);

    const std::vector<float> initialScale{2.0f, -1.0f, 0.5f, 3.0f};
    auto scale = std::make_shared<FixedTrainableVectorParameter>("scale", initialScale);
    scale->setOptimizer(
        std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(1234, learningRate, 0.0f, 0.0f, false)));

    RaggedCustomLayer layer(buildTrainableScaleRaggedExpression(batchSize, fullCapacityRows, width),
                            {"x", RaggedCustomLayer::RAGGED_OFFSETS_INPUT_NAME},
                            {"y"},
                            gpuPlacement,
                            std::vector<std::shared_ptr<PhysicalParameter>>{scale},
                            false,
                            fullCapacityRows,
                            {width},
                            {width},
                            {0},
                            1,
                            -1,
                            {{DataType::FP32, {width}, false}});

    PassiveEndpoint xSource;
    PassiveEndpoint offsetsSource;
    PassiveEndpoint sink;
    ASSERT_TRUE(layer.connectToPreviousLayer(&xSource, x, stream, true, 0).has_value());
    ASSERT_FALSE(layer.connectToPreviousLayer(&offsetsSource, offsets, stream, false, 1).has_value());
    layer.connectToNextLayer(&sink, 0, 0);
    layer.compile();
    layer.initialize();

    layer.forward(x, false, batchSize);
    layer.forward(offsets, false, batchSize);
    ASSERT_TRUE(sink.lastForward.has_value());
    const std::vector<float> actualY = readGpuFp32Tensor(sink.lastForward.value(), stream);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < width; ++col) {
            EXPECT_NEAR(actualY[row * width + col], xValues[row * width + col] * initialScale[col], 1e-5f);
        }
    }
    for (uint64_t i = activeRows * width; i < actualY.size(); ++i) EXPECT_EQ(actualY[i], 0.0f);

    ASSERT_TRUE(sink.getErrorOutput().has_value());
    Tensor dY = sink.getErrorOutput().value();
    std::vector<float> dYValues(fullCapacityRows * width, -50000.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < width; ++col) {
            dYValues[row * width + col] = 0.25f * static_cast<float>(1 + ((row + col) % 5));
        }
    }
    overwriteGpuFp32Tensor(dY, dYValues, stream);
    layer.backward(dY, batchSize);

    ASSERT_TRUE(layer.getGradientUpdateStream().has_value());
    Stream gradientStream = layer.getGradientUpdateStream().value();
    gradientStream.synchronize();
    stream.synchronize();

    const auto inputGradients = layer.getErrorOutputs();
    ASSERT_EQ(inputGradients.size(), 2u);
    ASSERT_TRUE(inputGradients[0].has_value());
    ASSERT_FALSE(inputGradients[1].has_value());
    const std::vector<float> actualDX = readGpuFp32Tensor(inputGradients[0].value(), stream);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < width; ++col) {
            EXPECT_NEAR(actualDX[row * width + col], dYValues[row * width + col] * initialScale[col], 1e-5f);
        }
    }
    for (uint64_t i = activeRows * width; i < actualDX.size(); ++i) EXPECT_EQ(actualDX[i], 0.0f);

    std::vector<float> dScale(width, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < width; ++col) {
            dScale[col] += xValues[row * width + col] * dYValues[row * width + col];
        }
    }
    const float step = learningRate / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    std::vector<float> expectedScale = initialScale;
    for (uint64_t col = 0; col < width; ++col) expectedScale[col] -= step * dScale[col];

    ASSERT_TRUE(scale->getStorage().has_value());
    const std::vector<float> actualScale = readGpuFp32Tensor(scale->getStorage().value(), gradientStream);
    expectNear(actualScale, expectedScale, 2e-5f);

    layer.cleanup();
}
