#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int cudaDeviceCountForTest = 0;                                                                                \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                            \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                         \
            GTEST_SKIP() << "CUDA device is required for RaggedCustomLoss execution tests.";                          \
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
    switch (tensor.getDataType()) {
        case DataType::FP8_E4M3: {
            __nv_fp8_e4m3* ptr = tensor.getMemPtr<__nv_fp8_e4m3>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = __nv_fp8_e4m3(values[i]);
            break;
        }
        case DataType::FP8_E5M2: {
            __nv_fp8_e5m2* ptr = tensor.getMemPtr<__nv_fp8_e5m2>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = __nv_fp8_e5m2(values[i]);
            break;
        }
        case DataType::FP16: {
            half* ptr = static_cast<half*>(tensor.getMemPtr());
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = __float2half(values[i]);
            break;
        }
        case DataType::BF16: {
            __nv_bfloat16* ptr = tensor.getMemPtr<__nv_bfloat16>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = __float2bfloat16(values[i]);
            break;
        }
        case DataType::FP32: {
            float* ptr = tensor.getMemPtr<float>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = values[i];
            break;
        }
        case DataType::INT32: {
            int32_t* ptr = tensor.getMemPtr<int32_t>();
            for (size_t i = 0; i < values.size(); ++i)
                ptr[i] = static_cast<int32_t>(values[i]);
            break;
        }
        default:
            FAIL() << "unsupported numeric dtype";
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
    switch (host.getDataType()) {
        case DataType::FP8_E4M3: {
            const __nv_fp8_e4m3* ptr = host.getMemPtr<__nv_fp8_e4m3>();
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(ptr[i]);
            break;
        }
        case DataType::FP8_E5M2: {
            const __nv_fp8_e5m2* ptr = host.getMemPtr<__nv_fp8_e5m2>();
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(ptr[i]);
            break;
        }
        case DataType::FP16: {
            const half* ptr = static_cast<const half*>(host.getMemPtr());
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = __half2float(ptr[i]);
            break;
        }
        case DataType::BF16: {
            const __nv_bfloat16* ptr = host.getMemPtr<__nv_bfloat16>();
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = __bfloat162float(ptr[i]);
            break;
        }
        case DataType::FP32: {
            const float* ptr = host.getMemPtr<float>();
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = ptr[i];
            break;
        }
        case DataType::INT32: {
            const int32_t* ptr = host.getMemPtr<int32_t>();
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(ptr[i]);
            break;
        }
        default:
            ADD_FAILURE() << "unsupported numeric dtype";
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

void overwriteGpuOffsets(Tensor& device, const std::vector<uint64_t>& values, Stream& stream) {
    Tensor replacement = makeGpuOffsets(device.getDataType(), values, stream);
    device.copyFromAsync(replacement, stream);
    stream.synchronize();
}

DynamicExpressionBuild compileOutputs(const Outputs& outputs,
                                      const DynamicExpression::TensorMap& stampInputs,
                                      const DynamicExpression::TensorMap& preallocatedOutputs,
                                      Stream& stream) {
    return DynamicExpressionBuild{
        .equation = std::make_shared<FusedEquation>(FusedEquation::compile(outputs.physicalOutputs(), stream.getGpuNum())),
        .stamp_inputs = stampInputs,
        .tensor_scalar_inputs = {},
        .preallocated_outputs = preallocatedOutputs,
        .requested_output_shapes = {},
    };
}

DynamicExpression makeSquaredErrorLossExpression(DataType lossDType) {
    return DynamicExpression(
        {"predictions", "labels"},
        {"loss"},
        [lossDType](const DynamicExpression::TensorMap& inputs,
                    const DynamicExpression::TensorMap& outputs,
                    Stream& stream) -> DynamicExpressionBuild {
            if (inputs.at("predictions").getDimensions() != inputs.at("labels").getDimensions())
                throw std::invalid_argument("test squared-error loss requires equal shapes");
            Expression predictions = Expression::input("predictions", DataType::FP32, DataType::FP32);
            Expression labels = Expression::input("labels", DataType::FP32, DataType::FP32);
            Expression diff = predictions - labels;
            Expression loss = (diff * diff).withOutputDType(lossDType);
            return compileOutputs(Expression::outputs({{"loss", loss}}), inputs, outputs, stream);
        });
}

DynamicExpression makeSquaredErrorGradientExpression() {
    return DynamicExpression(
        {"predictions", "labels"},
        {"predictions_grad"},
        [](const DynamicExpression::TensorMap& inputs,
           const DynamicExpression::TensorMap& outputs,
           Stream& stream) -> DynamicExpressionBuild {
            const DataType predictionDType = inputs.at("predictions").getDataType();
            Expression predictions = Expression::input("predictions", DataType::FP32, DataType::FP32);
            Expression labels = Expression::input("labels", DataType::FP32, DataType::FP32);
            Expression scale(2.0f * Loss::getLossScalingFactor());
            Expression gradient = ((predictions - labels) * scale).withOutputDType(predictionDType);
            return compileOutputs(Expression::outputs({{"predictions_grad", gradient}}), inputs, outputs, stream);
        });
}

DynamicExpression makeWeightedSquaredErrorLossExpression(DataType lossDType) {
    return DynamicExpression(
        {"predictions", "labels", "example_weights"},
        {"loss"},
        [lossDType](const DynamicExpression::TensorMap& inputs,
                    const DynamicExpression::TensorMap& outputs,
                    Stream& stream) -> DynamicExpressionBuild {
            Expression predictions = Expression::input("predictions", DataType::FP32, DataType::FP32);
            Expression labels = Expression::input("labels", DataType::FP32, DataType::FP32);
            Expression weights = Expression::input("example_weights", DataType::FP32, DataType::FP32);
            Expression diff = predictions - labels;
            Expression loss = (diff * diff * weights).withOutputDType(lossDType);
            return compileOutputs(Expression::outputs({{"loss", loss}}), inputs, outputs, stream);
        });
}

DynamicExpression makeWeightedSquaredErrorGradientExpression() {
    return DynamicExpression(
        {"predictions", "labels", "example_weights"},
        {"predictions_grad"},
        [](const DynamicExpression::TensorMap& inputs,
           const DynamicExpression::TensorMap& outputs,
           Stream& stream) -> DynamicExpressionBuild {
            const DataType predictionDType = inputs.at("predictions").getDataType();
            Expression predictions = Expression::input("predictions", DataType::FP32, DataType::FP32);
            Expression labels = Expression::input("labels", DataType::FP32, DataType::FP32);
            Expression weights = Expression::input("example_weights", DataType::FP32, DataType::FP32);
            Expression scale(2.0f * Loss::getLossScalingFactor());
            Expression gradient = ((predictions - labels) * weights * scale).withOutputDType(predictionDType);
            return compileOutputs(Expression::outputs({{"predictions_grad", gradient}}), inputs, outputs, stream);
        });
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

void expectNearPrefix(const std::vector<float>& actual,
                      const std::vector<float>& expected,
                      size_t prefixSize,
                      float tolerance) {
    ASSERT_GE(actual.size(), prefixSize);
    ASSERT_EQ(expected.size(), prefixSize);
    for (size_t i = 0; i < prefixSize; ++i)
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
}

void runActivePrefixCase(DataType predictionDType, DataType labelDType, DataType offsetsDType) {
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 9;
    constexpr uint64_t activeValues = 5;
    constexpr uint64_t width = 2;
    Stream stream(0);
    std::vector<float> predictions(maxTotalValues * width, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> labels(maxTotalValues * width, -1234.0f);
    for (uint64_t row = 0; row < activeValues; ++row) {
        for (uint64_t column = 0; column < width; ++column) {
            const size_t index = row * width + column;
            predictions[index] = static_cast<float>(row + column + 1);
            labels[index] = static_cast<float>(static_cast<int64_t>(row) - static_cast<int64_t>(column));
        }
    }

    Tensor predictionsTensor = makeGpuFloatingTensor(predictionDType, {maxTotalValues, width}, predictions, stream);
    Tensor labelsTensor = makeGpuFloatingTensor(labelDType, {maxTotalValues, width}, labels, stream);
    Tensor offsetsTensor = makeGpuOffsets(offsetsDType, {0, 2, 2, 5, 5}, stream);

    RaggedCustomLoss loss(makeSquaredErrorLossExpression(DataType::FP32),
                          makeSquaredErrorGradientExpression(),
                          batchSize,
                          maxTotalValues);
    PassiveEndpoint predictionsSource;
    PassiveEndpoint labelsSource;
    PassiveEndpoint offsetsSource;
    PassiveEndpoint lossSink;

    ASSERT_TRUE(loss.connectToPreviousLayer(&predictionsSource,
                                            predictionsTensor,
                                            stream,
                                            true,
                                            static_cast<int>(RaggedCustomLoss::InputConnection::PREDICTIONS))
                    .has_value());
    ASSERT_FALSE(loss.connectToPreviousLayer(&labelsSource,
                                             labelsTensor,
                                             stream,
                                             false,
                                             static_cast<int>(RaggedCustomLoss::InputConnection::LABELS))
                     .has_value());
    ASSERT_FALSE(loss.connectToPreviousLayer(&offsetsSource,
                                             offsetsTensor,
                                             stream,
                                             false,
                                             static_cast<int>(RaggedCustomLoss::InputConnection::OFFSETS))
                     .has_value());
    loss.connectToNextLayer(&lossSink);
    loss.compile();
    loss.initialize();

    ASSERT_TRUE(loss.getFeatureOutput().has_value());
    ASSERT_TRUE(loss.getErrorOutput().has_value());
    Tensor rawLoss = loss.getFeatureOutput().value();
    Tensor gradient = loss.getErrorOutput().value();
    overwriteGpuFloatingTensor(rawLoss, std::vector<float>(maxTotalValues * width, -777.0f), stream);
    overwriteGpuFloatingTensor(gradient, std::vector<float>(maxTotalValues * width, -4.0f), stream);

    loss.forward(predictionsTensor, false, batchSize);
    loss.forward(labelsTensor, false, batchSize);
    loss.forward(offsetsTensor, false, batchSize);

    ASSERT_TRUE(lossSink.lastForward.has_value());
    ASSERT_TRUE(predictionsSource.lastBackward.has_value());
    const std::vector<float> actualLoss = readGpuFloatingTensor(lossSink.lastForward.value(), stream);
    const std::vector<float> actualGradient = readGpuFloatingTensor(predictionsSource.lastBackward.value(), stream);

    const std::vector<float> quantizedPredictions = readGpuFloatingTensor(predictionsTensor, stream);
    const std::vector<float> quantizedLabels = readGpuFloatingTensor(labelsTensor, stream);
    std::vector<float> expectedLoss(activeValues * width);
    std::vector<float> expectedGradient(activeValues * width);
    for (size_t i = 0; i < expectedLoss.size(); ++i) {
        const float diff = quantizedPredictions[i] - quantizedLabels[i];
        expectedLoss[i] = diff * diff;
        expectedGradient[i] = 2.0f * diff * Loss::getLossScalingFactor();
    }
    expectNearPrefix(actualLoss, expectedLoss, expectedLoss.size(), 1.0e-4f);
    const float gradientTolerance =
        (predictionDType == DataType::FP8_E4M3 || predictionDType == DataType::FP8_E5M2) ? 1.0f :
        (predictionDType == DataType::FP16 || predictionDType == DataType::BF16) ? 2.0e-2f : 1.0e-5f;
    expectNearPrefix(actualGradient, expectedGradient, expectedGradient.size(), gradientTolerance);

    for (size_t i = expectedLoss.size(); i < actualLoss.size(); ++i)
        EXPECT_EQ(actualLoss[i], -777.0f) << "inactive raw-loss element " << i << " was written";
    for (size_t i = expectedGradient.size(); i < actualGradient.size(); ++i)
        EXPECT_EQ(actualGradient[i], -4.0f) << "inactive prediction-gradient element " << i << " was written";

    EXPECT_EQ(lossSink.lastForwardBatchSize, batchSize);
    EXPECT_EQ(predictionsSource.lastBackwardBatchSize, batchSize);
    loss.cleanup();
}

}  // namespace

TEST(RaggedCustomLoss, ActivePrefixForwardBackwardSupportsDenseRegressionPredictionDTypesAndBothOffsetWidths) {
    REQUIRE_CUDA_DEVICE();
    for (DataType predictionDType :
         {DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16, DataType::BF16, DataType::FP32}) {
        for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64})
            runActivePrefixCase(predictionDType, DataType::FP32, offsetsDType);
    }
}

TEST(RaggedCustomLoss, ActivePrefixForwardBackwardSupportsExpressionConvertibleIntegerLabels) {
    REQUIRE_CUDA_DEVICE();
    runActivePrefixCase(DataType::BF16, DataType::INT32, DataType::UINT32);
}

TEST(RaggedCustomLoss, PackedScalarExampleWeightsScaleLossAndGradientWithoutTouchingInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 7;
    constexpr uint64_t activeValues = 5;
    constexpr uint64_t width = 2;

    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Stream stream(0);
        Tensor predictions = makeGpuFloatingTensor(
            DataType::FP32, {maxTotalValues, width}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, NAN, NAN, NAN, NAN}, stream);
        Tensor labels = makeGpuFloatingTensor(
            DataType::FP32, {maxTotalValues, width}, std::vector<float>(maxTotalValues * width, 0.0f), stream);
        Tensor offsets = makeGpuOffsets(offsetsDType, {0, 2, 2, 5}, stream);
        Tensor weights = makeGpuFloatingTensor(
            DataType::FP32, {maxTotalValues, 1}, {0.5f, 0.5f, 2.0f, 2.0f, 2.0f, NAN, NAN}, stream);

        RaggedCustomLoss loss(makeWeightedSquaredErrorLossExpression(DataType::FP32),
                              makeWeightedSquaredErrorGradientExpression(),
                              batchSize,
                              maxTotalValues,
                              "predictions",
                              "labels",
                              "loss",
                              "predictions_grad",
                              DataType::FP32,
                              std::nullopt,
                              std::string("example_weights"));
        PassiveEndpoint predictionsSource, labelsSource, offsetsSource, weightsSource, lossSink;
        loss.connectToPreviousLayer(&predictionsSource, predictions, stream, true,
                                    static_cast<int>(RaggedCustomLoss::InputConnection::PREDICTIONS));
        loss.connectToPreviousLayer(&labelsSource, labels, stream, false,
                                    static_cast<int>(RaggedCustomLoss::InputConnection::LABELS));
        loss.connectToPreviousLayer(&offsetsSource, offsets, stream, false,
                                    static_cast<int>(RaggedCustomLoss::InputConnection::OFFSETS));
        EXPECT_FALSE(loss.connectToPreviousLayer(&weightsSource, weights, stream, true,
                                                 static_cast<int>(RaggedCustomLoss::InputConnection::EXAMPLE_WEIGHTS))
                         .has_value());
        loss.connectToNextLayer(&lossSink);
        loss.compile();
        loss.initialize();

        overwriteGpuFloatingTensor(loss.getFeatureOutput().value(), std::vector<float>(maxTotalValues * width, -77.0f), stream);
        overwriteGpuFloatingTensor(loss.getErrorOutput().value(), std::vector<float>(maxTotalValues * width, -55.0f), stream);
        loss.forward(weights, false, batchSize);
        loss.forward(offsets, false, batchSize);
        loss.forward(labels, false, batchSize);
        loss.forward(predictions, false, batchSize);

        const std::vector<float> actualLoss = readGpuFloatingTensor(lossSink.lastForward.value(), stream);
        const std::vector<float> actualGradient = readGpuFloatingTensor(predictionsSource.lastBackward.value(), stream);
        const std::vector<float> predictionValues = readGpuFloatingTensor(predictions, stream);
        const std::vector<float> tokenWeights{0.5f, 0.5f, 2.0f, 2.0f, 2.0f};
        for (uint64_t token = 0; token < activeValues; ++token) {
            for (uint64_t column = 0; column < width; ++column) {
                const size_t i = token * width + column;
                const float expectedLoss = predictionValues[i] * predictionValues[i] * tokenWeights[token];
                const float expectedGradient =
                    2.0f * predictionValues[i] * tokenWeights[token] * Loss::getLossScalingFactor();
                EXPECT_NEAR(actualLoss[i], expectedLoss, 1.0e-5f);
                EXPECT_NEAR(actualGradient[i], expectedGradient, 1.0e-5f);
            }
        }
        for (size_t i = activeValues * width; i < actualLoss.size(); ++i) {
            EXPECT_EQ(actualLoss[i], -77.0f);
            EXPECT_EQ(actualGradient[i], -55.0f);
        }
        EXPECT_FALSE(weightsSource.lastBackward.has_value());
        loss.cleanup();
    }
}

TEST(RaggedCustomLoss, AllEmptyBatchDoesNotReadOrWritePackedCapacity) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 7;

    Stream stream(0);
    Tensor predictions = makeGpuFloatingTensor(
        DataType::FP32, {maxTotalValues}, std::vector<float>(maxTotalValues, std::numeric_limits<float>::quiet_NaN()), stream);
    Tensor labels = makeGpuFloatingTensor(DataType::FP32, {maxTotalValues}, std::vector<float>(maxTotalValues, 1234.0f), stream);
    Tensor offsets = makeGpuOffsets(DataType::UINT64, {0, 0, 0, 0}, stream);

    RaggedCustomLoss loss(makeSquaredErrorLossExpression(DataType::FP32),
                          makeSquaredErrorGradientExpression(),
                          batchSize,
                          maxTotalValues);
    PassiveEndpoint predictionsSource, labelsSource, offsetsSource, lossSink;
    loss.connectToPreviousLayer(&predictionsSource, predictions, stream, true,
                                static_cast<int>(RaggedCustomLoss::InputConnection::PREDICTIONS));
    loss.connectToPreviousLayer(&labelsSource, labels, stream, false,
                                static_cast<int>(RaggedCustomLoss::InputConnection::LABELS));
    loss.connectToPreviousLayer(&offsetsSource, offsets, stream, false,
                                static_cast<int>(RaggedCustomLoss::InputConnection::OFFSETS));
    loss.connectToNextLayer(&lossSink);
    loss.compile();
    loss.initialize();

    overwriteGpuFloatingTensor(loss.getFeatureOutput().value(), std::vector<float>(maxTotalValues, -77.0f), stream);
    overwriteGpuFloatingTensor(loss.getErrorOutput().value(), std::vector<float>(maxTotalValues, -55.0f), stream);

    loss.forward(predictions, false, batchSize);
    loss.forward(labels, false, batchSize);
    loss.forward(offsets, false, batchSize);

    EXPECT_EQ(readGpuFloatingTensor(lossSink.lastForward.value(), stream), std::vector<float>(maxTotalValues, -77.0f));
    EXPECT_EQ(readGpuFloatingTensor(predictionsSource.lastBackward.value(), stream), std::vector<float>(maxTotalValues, -55.0f));
    loss.cleanup();
}

TEST(RaggedCustomLoss, ReusesStampedPlansAcrossShortLongShortRuntimeExtents) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 8;

    Stream stream(0);
    Tensor predictions = makeGpuFloatingTensor(DataType::FP32, {maxTotalValues}, std::vector<float>(maxTotalValues, 0.0f), stream);
    Tensor labels = makeGpuFloatingTensor(DataType::FP32, {maxTotalValues}, std::vector<float>(maxTotalValues, 0.0f), stream);
    Tensor offsets = makeGpuOffsets(DataType::UINT32, {0, 1, 1, 2}, stream);

    RaggedCustomLoss loss(makeSquaredErrorLossExpression(DataType::FP32),
                          makeSquaredErrorGradientExpression(),
                          batchSize,
                          maxTotalValues);
    PassiveEndpoint predictionsSource, labelsSource, offsetsSource, lossSink;
    loss.connectToPreviousLayer(&predictionsSource, predictions, stream, true,
                                static_cast<int>(RaggedCustomLoss::InputConnection::PREDICTIONS));
    loss.connectToPreviousLayer(&labelsSource, labels, stream, false,
                                static_cast<int>(RaggedCustomLoss::InputConnection::LABELS));
    loss.connectToPreviousLayer(&offsetsSource, offsets, stream, false,
                                static_cast<int>(RaggedCustomLoss::InputConnection::OFFSETS));
    loss.connectToNextLayer(&lossSink);
    loss.compile();
    loss.initialize();

    const std::vector<std::vector<uint64_t>> partitions{{0, 1, 1, 2}, {0, 3, 5, 7}, {0, 0, 1, 1}};
    const std::vector<uint64_t> activeCounts{2, 7, 1};
    for (size_t pass = 0; pass < partitions.size(); ++pass) {
        std::vector<float> predictionValues(maxTotalValues, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> labelValues(maxTotalValues, -999.0f);
        for (uint64_t i = 0; i < activeCounts[pass]; ++i) {
            predictionValues[i] = static_cast<float>(10 * pass + i + 1);
            labelValues[i] = static_cast<float>(static_cast<int>(pass) - static_cast<int>(i));
        }
        overwriteGpuFloatingTensor(predictions, predictionValues, stream);
        overwriteGpuFloatingTensor(labels, labelValues, stream);
        overwriteGpuOffsets(offsets, partitions[pass], stream);
        overwriteGpuFloatingTensor(loss.getFeatureOutput().value(), std::vector<float>(maxTotalValues, -77.0f), stream);
        overwriteGpuFloatingTensor(loss.getErrorOutput().value(), std::vector<float>(maxTotalValues, -55.0f), stream);

        loss.forward(labels, false, batchSize);
        loss.forward(offsets, false, batchSize);
        loss.forward(predictions, false, batchSize);

        const std::vector<float> actualLoss = readGpuFloatingTensor(lossSink.lastForward.value(), stream);
        const std::vector<float> actualGradient = readGpuFloatingTensor(predictionsSource.lastBackward.value(), stream);
        for (uint64_t i = 0; i < activeCounts[pass]; ++i) {
            const float diff = predictionValues[i] - labelValues[i];
            EXPECT_NEAR(actualLoss[i], diff * diff, 1.0e-5f);
            EXPECT_NEAR(actualGradient[i], 2.0f * diff * Loss::getLossScalingFactor(), 1.0e-5f);
        }
        for (uint64_t i = activeCounts[pass]; i < maxTotalValues; ++i) {
            EXPECT_EQ(actualLoss[i], -77.0f);
            EXPECT_EQ(actualGradient[i], -55.0f);
        }
    }
    loss.cleanup();
}

TEST(RaggedCustomLoss, PartialBatchCardinalityUsesLogicalRowsRatherThanPackedCapacity) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t batchSize = 4;
    constexpr uint64_t maxTotalValues = 12;
    constexpr uint32_t validExamples = 2;

    Stream stream(0);
    Tensor predictions = makeGpuFloatingTensor(DataType::FP32, {maxTotalValues}, std::vector<float>(maxTotalValues, 2.0f), stream);
    Tensor labels = makeGpuFloatingTensor(DataType::FP32, {maxTotalValues}, std::vector<float>(maxTotalValues, 1.0f), stream);
    // Invalid logical tail rows are empty in a partial batch; active packed data
    // belongs only to the first two valid examples.
    Tensor offsets = makeGpuOffsets(DataType::UINT32, {0, 2, 3, 3, 3}, stream);

    RaggedCustomLoss loss(makeSquaredErrorLossExpression(DataType::FP32),
                          makeSquaredErrorGradientExpression(),
                          batchSize,
                          maxTotalValues);
    PassiveEndpoint predictionsSource, labelsSource, offsetsSource, lossSink;
    loss.connectToPreviousLayer(&predictionsSource, predictions, stream, true,
                                static_cast<int>(RaggedCustomLoss::InputConnection::PREDICTIONS));
    loss.connectToPreviousLayer(&labelsSource, labels, stream, false,
                                static_cast<int>(RaggedCustomLoss::InputConnection::LABELS));
    loss.connectToPreviousLayer(&offsetsSource, offsets, stream, false,
                                static_cast<int>(RaggedCustomLoss::InputConnection::OFFSETS));
    loss.connectToNextLayer(&lossSink);
    loss.compile();
    loss.initialize();

    loss.forward(predictions, false, validExamples);
    loss.forward(labels, false, validExamples);
    loss.forward(offsets, false, validExamples);

    EXPECT_EQ(lossSink.lastForwardBatchSize, validExamples);
    EXPECT_EQ(predictionsSource.lastBackwardBatchSize, validExamples);
    EXPECT_NE(predictionsSource.lastBackwardBatchSize, maxTotalValues);
    loss.cleanup();
}
