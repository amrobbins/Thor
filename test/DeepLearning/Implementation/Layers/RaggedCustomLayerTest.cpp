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
#include "test/DeepLearning/RaggedTestUtils.h"

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

DynamicExpression buildAffineRaggedExpression(uint64_t batchSize,
                                              uint64_t fullCapacityRows,
                                              uint64_t width,
                                              float scale,
                                              float bias) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression offsets =
        Expression::input(RaggedCustomLayer::RAGGED_OFFSETS_INPUT_NAME, std::nullopt, DataType::UINT32);
    const Expression y = ((x * Expression(scale)) + Expression(bias))
                             .withRaggedRuntimeExtent(offsets, batchSize, fullCapacityRows, width);
    return DynamicExpression::fromExpressionDefinition(
        ExpressionDefinition::fromOutputs(Expression::outputs({{"y", y}})));
}

DynamicExpression buildSquareShiftRaggedExpression(uint64_t batchSize,
                                                   uint64_t fullCapacityRows,
                                                   uint64_t width,
                                                   float shift) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression offsets =
        Expression::input(RaggedCustomLayer::RAGGED_OFFSETS_INPUT_NAME, std::nullopt, DataType::UINT32);
    const Expression y = ((x * x) + Expression(shift))
                             .withRaggedRuntimeExtent(offsets, batchSize, fullCapacityRows, width);
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

TEST(RaggedCustomLayer, MultiInputMultiOutputPreservesActivePrefixWithPoisonedInactiveStorage) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchSize = 3;
    constexpr uint64_t fullCapacityRows = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint64_t inputWidth = 4;
    constexpr uint64_t narrowWidth = 2;

    Stream stream(0);
    std::vector<float> lhsValues(fullCapacityRows * inputWidth, 0.0f);
    std::vector<float> rhsValues(fullCapacityRows * inputWidth, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t column = 0; column < inputWidth; ++column) {
            lhsValues[row * inputWidth + column] = static_cast<float>(10 * row + column + 1);
            rhsValues[row * inputWidth + column] = static_cast<float>(100 + 7 * row - 2 * column);
        }
    }

    ThorTest::poisonInactiveRows(
        lhsValues, activeRows, inputWidth, ThorTest::RaggedInactivePoison::PositiveFinite);
    ThorTest::poisonInactiveRows(
        rhsValues, activeRows, inputWidth, ThorTest::RaggedInactivePoison::NaN);

    Tensor lhs = makeGpuFp32Tensor({fullCapacityRows, inputWidth}, lhsValues, stream);
    Tensor rhs = makeGpuFp32Tensor({fullCapacityRows, inputWidth}, rhsValues, stream);
    Tensor offsets = makeGpuU32Tensor({batchSize + 1}, {0, 3, 3, 5}, stream);

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

    std::vector<float> expectedWide(activeRows * inputWidth, 0.0f);
    std::vector<float> expectedNarrow(activeRows * narrowWidth, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t column = 0; column < inputWidth; ++column) {
            expectedWide[row * inputWidth + column] = lhsValues[row * inputWidth + column] + rhsValues[row * inputWidth + column];
        }
        expectedNarrow[row * narrowWidth] = expectedWide[row * inputWidth + 1];
        expectedNarrow[row * narrowWidth + 1] = expectedWide[row * inputWidth + 2];
    }
    expectNear(ThorTest::logicalActivePrefix(wide, activeRows, inputWidth), expectedWide);
    expectNear(ThorTest::logicalActivePrefix(narrow, activeRows, narrowWidth), expectedNarrow);

    ASSERT_TRUE(wideSink.getErrorOutput().has_value());
    ASSERT_TRUE(narrowSink.getErrorOutput().has_value());
    Tensor wideGradient = wideSink.getErrorOutput().value();
    Tensor narrowGradient = narrowSink.getErrorOutput().value();
    std::vector<float> wideGradientValues(fullCapacityRows * inputWidth, 1.0f);
    std::vector<float> narrowGradientValues(fullCapacityRows * narrowWidth, 2.0f);
    ThorTest::poisonInactiveRows(
        wideGradientValues, activeRows, inputWidth, ThorTest::RaggedInactivePoison::NaN);
    ThorTest::poisonInactiveRows(
        narrowGradientValues, activeRows, narrowWidth, ThorTest::RaggedInactivePoison::NegativeFinite);
    overwriteGpuFp32Tensor(wideGradient, wideGradientValues, stream);
    overwriteGpuFp32Tensor(narrowGradient, narrowGradientValues, stream);

    layer.backward(wideGradient, batchSize);
    layer.backward(narrowGradient, batchSize);

    const auto inputGradients = layer.getErrorOutputs();
    ASSERT_EQ(inputGradients.size(), 3u);
    ASSERT_TRUE(inputGradients[0].has_value());
    ASSERT_TRUE(inputGradients[1].has_value());
    ASSERT_FALSE(inputGradients[2].has_value());

    std::vector<float> expectedInputGradient(activeRows * inputWidth, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        expectedInputGradient[row * inputWidth] = 1.0f;
        expectedInputGradient[row * inputWidth + 1] = 3.0f;
        expectedInputGradient[row * inputWidth + 2] = 3.0f;
        expectedInputGradient[row * inputWidth + 3] = 1.0f;
    }
    expectNear(ThorTest::logicalActivePrefix(readGpuFp32Tensor(inputGradients[0].value(), stream), activeRows, inputWidth),
               expectedInputGradient);
    expectNear(ThorTest::logicalActivePrefix(readGpuFp32Tensor(inputGradients[1].value(), stream), activeRows, inputWidth),
               expectedInputGradient);

    layer.cleanup();
}

TEST(RaggedCustomLayer, GpuOffsetsDirectlyDriveRuntimeExtentWithoutHostActiveValueCount) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchSize = 3;
    constexpr uint64_t fullCapacityRows = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint64_t inputWidth = 4;

    Stream stream(0);
    std::vector<float> lhsValues(fullCapacityRows * inputWidth, 1.0f);
    std::vector<float> rhsValues(fullCapacityRows * inputWidth, 2.0f);
    ThorTest::poisonInactiveRows(
        lhsValues, activeRows, inputWidth, ThorTest::RaggedInactivePoison::PositiveFinite);
    ThorTest::poisonInactiveRows(rhsValues, activeRows, inputWidth, ThorTest::RaggedInactivePoison::NaN);
    Tensor lhs = makeGpuFp32Tensor({fullCapacityRows, inputWidth}, lhsValues, stream);
    Tensor rhs = makeGpuFp32Tensor({fullCapacityRows, inputWidth}, rhsValues, stream);
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
    EXPECT_NO_THROW(layer.forward(offsets, false, batchSize));
    ASSERT_TRUE(wideSink.lastForward.has_value());
    ASSERT_TRUE(narrowSink.lastForward.has_value());

    const std::vector<float> expectedWide(activeRows * inputWidth, 3.0f);
    const std::vector<float> expectedNarrow(activeRows * 2, 3.0f);
    expectNear(ThorTest::logicalActivePrefix(readGpuFp32Tensor(wideSink.lastForward.value(), stream), activeRows, inputWidth),
               expectedWide);
    expectNear(ThorTest::logicalActivePrefix(readGpuFp32Tensor(narrowSink.lastForward.value(), stream), activeRows, 2),
               expectedNarrow);

    layer.cleanup();
}

TEST(RaggedCustomLayer, ChainedActiveAwareExpressionsIgnorePoisonedInactiveStorageForwardAndBackward) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchSize = 3;
    constexpr uint64_t fullCapacityRows = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint64_t width = 3;

    Stream stream(0);
    std::vector<float> xValues(fullCapacityRows * width, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < width; ++col) {
            xValues[row * width + col] = 0.1f * static_cast<float>(1 + row * width + col);
        }
    }
    ThorTest::poisonInactiveRows(xValues, activeRows, width, ThorTest::RaggedInactivePoison::NaN);
    Tensor x = makeGpuFp32Tensor({fullCapacityRows, width}, xValues, stream);
    Tensor offsets = makeGpuU32Tensor({batchSize + 1}, {0, 2, 2, 5}, stream);
    ASSERT_FALSE(RowPartitionRuntime(offsets, RowPartitionDescriptor(batchSize, fullCapacityRows, DataType::UINT32))
                     .getHostActiveValueCountIfAvailable()
                     .has_value());

    RaggedCustomLayer first(buildAffineRaggedExpression(batchSize, fullCapacityRows, width, 2.0f, 1.0f),
                            {"x", RaggedCustomLayer::RAGGED_OFFSETS_INPUT_NAME},
                            {"y"},
                            gpuPlacement,
                            false,
                            fullCapacityRows,
                            width,
                            width,
                            0,
                            1);
    RaggedCustomLayer second(buildSquareShiftRaggedExpression(batchSize, fullCapacityRows, width, -3.0f),
                             {"x", RaggedCustomLayer::RAGGED_OFFSETS_INPUT_NAME},
                             {"y"},
                             gpuPlacement,
                             false,
                             fullCapacityRows,
                             width,
                             width,
                             0,
                             1);

    PassiveEndpoint firstInputSource;
    PassiveEndpoint firstOffsetsSource;
    PassiveEndpoint firstSink;
    ASSERT_TRUE(first.connectToPreviousLayer(&firstInputSource, x, stream, true, 0).has_value());
    ASSERT_FALSE(first.connectToPreviousLayer(&firstOffsetsSource, offsets, stream, false, 1).has_value());
    first.connectToNextLayer(&firstSink, 0, 0);
    first.compile();
    first.initialize();

    first.forward(x, false, batchSize);
    first.forward(offsets, false, batchSize);
    ASSERT_TRUE(firstSink.lastForward.has_value());
    Tensor middle = firstSink.lastForward.value();
    std::vector<float> middleValues = readGpuFp32Tensor(middle, stream);

    std::vector<float> expectedMiddle(activeRows * width, 0.0f);
    for (uint64_t i = 0; i < expectedMiddle.size(); ++i) {
        expectedMiddle[i] = 2.0f * xValues[i] + 1.0f;
    }
    expectNear(ThorTest::logicalActivePrefix(middleValues, activeRows, width), expectedMiddle);

    // The first producer makes no promise about its inactive storage. Force that
    // storage dirty before the next active-aware consumer to prove the consumer
    // really is bounded by offsets[B], not by physical capacity.
    ThorTest::poisonInactiveRows(middleValues, activeRows, width, ThorTest::RaggedInactivePoison::NegativeFinite);
    overwriteGpuFp32Tensor(middle, middleValues, stream);

    PassiveEndpoint secondInputSource;
    PassiveEndpoint secondOffsetsSource;
    PassiveEndpoint secondSink;
    ASSERT_TRUE(second.connectToPreviousLayer(&secondInputSource, middle, stream, true, 0).has_value());
    ASSERT_FALSE(second.connectToPreviousLayer(&secondOffsetsSource, offsets, stream, false, 1).has_value());
    second.connectToNextLayer(&secondSink, 0, 0);
    second.compile();
    second.initialize();

    second.forward(middle, false, batchSize);
    second.forward(offsets, false, batchSize);
    ASSERT_TRUE(secondSink.lastForward.has_value());
    const std::vector<float> actualOutput = readGpuFp32Tensor(secondSink.lastForward.value(), stream);
    std::vector<float> expectedOutput(activeRows * width, 0.0f);
    for (uint64_t i = 0; i < expectedOutput.size(); ++i) {
        expectedOutput[i] = expectedMiddle[i] * expectedMiddle[i] - 3.0f;
    }
    expectNear(ThorTest::logicalActivePrefix(actualOutput, activeRows, width), expectedOutput);

    ASSERT_TRUE(secondSink.getErrorOutput().has_value());
    Tensor dOutput = secondSink.getErrorOutput().value();
    std::vector<float> dOutputValues(fullCapacityRows * width, 0.0f);
    for (uint64_t i = 0; i < activeRows * width; ++i) {
        dOutputValues[i] = 0.25f + 0.05f * static_cast<float>(i % width);
    }
    ThorTest::poisonInactiveRows(dOutputValues, activeRows, width, ThorTest::RaggedInactivePoison::NaN);
    overwriteGpuFp32Tensor(dOutput, dOutputValues, stream);
    second.backward(dOutput, batchSize);

    const auto secondInputGradients = second.getErrorOutputs();
    ASSERT_EQ(secondInputGradients.size(), 2u);
    ASSERT_TRUE(secondInputGradients[0].has_value());
    ASSERT_FALSE(secondInputGradients[1].has_value());
    Tensor dMiddle = secondInputGradients[0].value();
    std::vector<float> dMiddleValues = readGpuFp32Tensor(dMiddle, stream);
    std::vector<float> expectedDMiddle(activeRows * width, 0.0f);
    for (uint64_t i = 0; i < expectedDMiddle.size(); ++i) {
        expectedDMiddle[i] = dOutputValues[i] * 2.0f * expectedMiddle[i];
    }
    expectNear(ThorTest::logicalActivePrefix(dMiddleValues, activeRows, width), expectedDMiddle);

    // Backward producers have the same undefined-tail contract. Poison the
    // logical dMiddle result, then stage it into the exact downstream-error
    // tensor registered for the first layer. These two layers are driven
    // independently in this test so that the intermediate storage can be
    // deliberately poisoned between consumers; CustomLayer::backward()
    // intentionally accepts only tensors belonging to a stamped application.
    ThorTest::poisonInactiveRows(dMiddleValues, activeRows, width, ThorTest::RaggedInactivePoison::PositiveFinite);
    ASSERT_TRUE(firstSink.getErrorOutput().has_value());
    Tensor firstDownstreamGradient = firstSink.getErrorOutput().value();
    overwriteGpuFp32Tensor(firstDownstreamGradient, dMiddleValues, stream);
    first.backward(firstDownstreamGradient, batchSize);

    const auto firstInputGradients = first.getErrorOutputs();
    ASSERT_EQ(firstInputGradients.size(), 2u);
    ASSERT_TRUE(firstInputGradients[0].has_value());
    ASSERT_FALSE(firstInputGradients[1].has_value());
    const std::vector<float> actualDX = readGpuFp32Tensor(firstInputGradients[0].value(), stream);
    std::vector<float> expectedDX(activeRows * width, 0.0f);
    for (uint64_t i = 0; i < expectedDX.size(); ++i) {
        expectedDX[i] = expectedDMiddle[i] * 2.0f;
    }
    expectNear(ThorTest::logicalActivePrefix(actualDX, activeRows, width), expectedDX);

    second.cleanup();
    first.cleanup();
}

TEST(RaggedCustomLayer, TrainableParameterGradientAndSgdUpdateIgnoreInactivePackedCapacity) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchSize = 3;
    constexpr uint64_t fullCapacityRows = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint64_t width = 4;
    constexpr float learningRate = 0.03f;

    Stream stream(0);
    std::vector<float> xValues(fullCapacityRows * width, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < width; ++col) {
            xValues[row * width + col] = static_cast<float>(1 + row * width + col) * 0.1f;
        }
    }
    ThorTest::poisonInactiveRows(
        xValues, activeRows, width, ThorTest::RaggedInactivePoison::PositiveFinite);
    Tensor x = makeGpuFp32Tensor({fullCapacityRows, width}, xValues, stream);
    Tensor offsets = makeGpuU32Tensor({batchSize + 1}, {0, 3, 3, 5}, stream);

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
    ASSERT_TRUE(sink.getErrorOutput().has_value());
    Tensor dY = sink.getErrorOutput().value();
    std::vector<float> dYValues(fullCapacityRows * width, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < width; ++col) {
            dYValues[row * width + col] = 0.25f * static_cast<float>(1 + ((row + col) % 5));
        }
    }
    ThorTest::poisonInactiveRows(
        dYValues, activeRows, width, ThorTest::RaggedInactivePoison::NegativeFinite);
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
