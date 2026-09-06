#include "DeepLearning/Implementation/Layers/Loss/SparseCategoricalCrossEntropyWithLogits.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int cudaDeviceCountForTest = 0;                                                                                \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                            \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                         \
            GTEST_SKIP() << "CUDA device is required for SparseCategoricalCrossEntropyWithLogits execution tests.";  \
        }                                                                                                              \
    } while (false)

class PassiveEndpoint final : public Layer {
   public:
    void forward(optional<Tensor> input, bool validationPass, uint32_t batchSize = 0) override {
        (void)validationPass;
        lastForward = input;
        lastForwardBatchSize = batchSize;
    }

    void backward(optional<Tensor> error, uint32_t batchSize = 0) override {
        lastBackward = error;
        lastBackwardBatchSize = batchSize;
    }

    optional<Tensor> lastForward;
    optional<Tensor> lastBackward;
    uint32_t lastForwardBatchSize = 0;
    uint32_t lastBackwardBatchSize = 0;

   private:
    void infer(optional<Tensor>, optional<Tensor>, Stream) override {}
    void backProp(optional<Tensor>, optional<Tensor>, optional<Tensor>, Stream) override {}
};

void copyFloatVectorToGpu(const vector<float>& values, Tensor gpuTensor, Stream stream) {
    Tensor cpuTensor(TensorPlacement(TensorPlacement::MemDevices::CPU), gpuTensor.getDescriptor());
    ASSERT_EQ(cpuTensor.getTotalNumElements(), values.size());
    copy(values.begin(), values.end(), static_cast<float*>(cpuTensor.getMemPtr()));
    gpuTensor.copyFromAsync(cpuTensor, stream);
    stream.synchronize();
}

vector<float> copyFloatVectorFromGpu(const Tensor& gpuTensor, Stream stream) {
    Tensor cpuTensor(TensorPlacement(TensorPlacement::MemDevices::CPU), gpuTensor.getDescriptor());
    cpuTensor.copyFromAsync(gpuTensor, stream);
    stream.synchronize();
    vector<float> result(cpuTensor.getTotalNumElements());
    copy(static_cast<float*>(cpuTensor.getMemPtr()),
         static_cast<float*>(cpuTensor.getMemPtr()) + result.size(),
         result.begin());
    return result;
}

void copyLabelsToGpu(const vector<uint32_t>& values, Tensor gpuTensor, Stream stream) {
    Tensor cpuTensor(TensorPlacement(TensorPlacement::MemDevices::CPU), gpuTensor.getDescriptor());
    ASSERT_EQ(cpuTensor.getTotalNumElements(), values.size());
    copy(values.begin(), values.end(), static_cast<uint32_t*>(cpuTensor.getMemPtr()));
    gpuTensor.copyFromAsync(cpuTensor, stream);
    stream.synchronize();
}

void copyMaskToGpu(const vector<uint8_t>& values, Tensor gpuTensor, Stream stream) {
    Tensor cpuTensor(TensorPlacement(TensorPlacement::MemDevices::CPU), gpuTensor.getDescriptor());
    ASSERT_EQ(cpuTensor.getTotalNumElements(), values.size());
    copy(values.begin(), values.end(), static_cast<uint8_t*>(cpuTensor.getMemPtr()));
    gpuTensor.copyFromAsync(cpuTensor, stream);
    stream.synchronize();
}

void copyActiveCountToGpu(uint64_t activeCount, DataType dataType, Tensor gpuTensor, Stream stream) {
    Tensor cpuTensor(TensorPlacement(TensorPlacement::MemDevices::CPU), gpuTensor.getDescriptor());
    if (dataType == DataType::UINT32)
        *static_cast<uint32_t*>(cpuTensor.getMemPtr()) = static_cast<uint32_t>(activeCount);
    else
        *static_cast<uint64_t*>(cpuTensor.getMemPtr()) = activeCount;
    gpuTensor.copyFromAsync(cpuTensor, stream);
    stream.synchronize();
}

float expectedLoss(const float* logits, uint32_t numClasses, uint32_t label) {
    float rowMax = -numeric_limits<float>::infinity();
    for (uint32_t c = 0; c < numClasses; ++c)
        rowMax = max(rowMax, logits[c]);
    float sumExp = 0.0f;
    for (uint32_t c = 0; c < numClasses; ++c)
        sumExp += expf(logits[c] - rowMax);
    return logf(sumExp) + rowMax - logits[label];
}

}  // namespace

TEST(SparseCategoricalCrossEntropyWithLogits, RaggedManagedActiveCountUsesPackedPrefixNotLogicalBatchCardinality) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint32_t logicalBatchSize = 3;
    constexpr uint32_t rowCapacity = 8;
    constexpr uint32_t activeRows = 5;
    constexpr uint32_t numClasses = 4;
    constexpr float lossSentinel = -77.0f;
    constexpr float gradientSentinel = -55.0f;

    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    for (DataType activeCountDataType : {DataType::UINT32, DataType::UINT64}) {
        Stream predictionsStream(0);
        Stream labelsStream(0);
        Stream maskStream(0);
        Stream activeCountStream(0);

        Tensor predictions(gpuPlacement, TensorDescriptor(DataType::FP32, {rowCapacity, numClasses}));
        Tensor labels(gpuPlacement, TensorDescriptor(DataType::UINT32, {rowCapacity}));
        Tensor mask(gpuPlacement, TensorDescriptor(DataType::UINT8, {rowCapacity}));
        Tensor activeCount(gpuPlacement, TensorDescriptor(activeCountDataType, {1}));

        vector<float> logitsHost(static_cast<size_t>(rowCapacity) * numClasses,
                                 numeric_limits<float>::quiet_NaN());
        vector<uint32_t> labelsHost(rowCapacity, numeric_limits<uint32_t>::max());
        vector<uint8_t> maskHost(rowCapacity, 0xff);
        for (uint32_t row = 0; row < activeRows; ++row) {
            labelsHost[row] = (row + 1) % numClasses;
            maskHost[row] = 1;
            for (uint32_t c = 0; c < numClasses; ++c)
                logitsHost[static_cast<size_t>(row) * numClasses + c] =
                    0.2f * static_cast<float>(row) - 0.35f * static_cast<float>(c) + 0.1f;
        }

        copyFloatVectorToGpu(logitsHost, predictions, predictionsStream);
        copyLabelsToGpu(labelsHost, labels, labelsStream);
        copyMaskToGpu(maskHost, mask, maskStream);
        copyActiveCountToGpu(activeRows, activeCountDataType, activeCount, activeCountStream);

        SparseCategoricalCrossEntropyWithLogits loss(
            DataType::FP32, nullopt, nullopt, logicalBatchSize);
        PassiveEndpoint predictionsSource;
        PassiveEndpoint labelsSource;
        PassiveEndpoint maskSource;
        PassiveEndpoint activeCountSource;
        PassiveEndpoint lossSink;

        ASSERT_TRUE(loss.connectToPreviousLayer(&predictionsSource,
                                                predictions,
                                                predictionsStream,
                                                true,
                                                static_cast<int>(Loss::ConnectionType::FORWARD_BACKWARD))
                        .has_value());
        ASSERT_FALSE(loss.connectToPreviousLayer(&labelsSource,
                                                 labels,
                                                 labelsStream,
                                                 false,
                                                 static_cast<int>(Loss::ConnectionType::LABELS))
                         .has_value());
        ASSERT_FALSE(loss.connectToPreviousLayer(&maskSource,
                                                 mask,
                                                 maskStream,
                                                 false,
                                                 SparseCategoricalCrossEntropyWithLogits::MASK_CONNECTION_TYPE)
                         .has_value());
        ASSERT_FALSE(loss.connectToPreviousLayer(&activeCountSource,
                                                 activeCount,
                                                 activeCountStream,
                                                 false,
                                                 SparseCategoricalCrossEntropyWithLogits::ACTIVE_COUNT_CONNECTION_TYPE)
                         .has_value());
        loss.connectToNextLayer(&lossSink);
        loss.compile();
        loss.initialize();

        ASSERT_TRUE(loss.getFeatureOutput().has_value());
        ASSERT_TRUE(loss.getErrorOutput().has_value());
        copyFloatVectorToGpu(vector<float>(rowCapacity, lossSentinel), loss.getFeatureOutput().value(), predictionsStream);
        copyFloatVectorToGpu(vector<float>(static_cast<size_t>(rowCapacity) * numClasses, gradientSentinel),
                             loss.getErrorOutput().value(),
                             predictionsStream);

        // activeRows (5) intentionally exceeds logicalBatchSize (3). A dense
        // partial-batch mask would incorrectly erase tokens 3 and 4.
        loss.forward(labels, false, logicalBatchSize);
        loss.forward(mask, false, logicalBatchSize);
        loss.forward(activeCount, false, logicalBatchSize);
        loss.forward(predictions, false, logicalBatchSize);

        Stream::deviceSynchronize(0);
        const vector<float> actualLoss = copyFloatVectorFromGpu(lossSink.lastForward.value(), predictionsStream);
        const vector<float> actualGradient = copyFloatVectorFromGpu(predictionsSource.lastBackward.value(), predictionsStream);

        EXPECT_EQ(lossSink.lastForwardBatchSize, logicalBatchSize);
        EXPECT_EQ(predictionsSource.lastBackwardBatchSize, logicalBatchSize);
        for (uint32_t row = 0; row < activeRows; ++row) {
            const size_t rowOffset = static_cast<size_t>(row) * numClasses;
            EXPECT_NEAR(actualLoss[row], expectedLoss(logitsHost.data() + rowOffset, numClasses, labelsHost[row]), 1.0e-5f);

            float rowMax = -numeric_limits<float>::infinity();
            for (uint32_t c = 0; c < numClasses; ++c)
                rowMax = max(rowMax, logitsHost[rowOffset + c]);
            float sumExp = 0.0f;
            for (uint32_t c = 0; c < numClasses; ++c)
                sumExp += expf(logitsHost[rowOffset + c] - rowMax);
            for (uint32_t c = 0; c < numClasses; ++c) {
                const float probability = expf(logitsHost[rowOffset + c] - rowMax) / sumExp;
                const float expectedGradient = probability - (c == labelsHost[row] ? 1.0f : 0.0f);
                EXPECT_NEAR(actualGradient[rowOffset + c], expectedGradient, 1.0e-5f);
            }
        }
        for (uint32_t row = activeRows; row < rowCapacity; ++row) {
            EXPECT_EQ(actualLoss[row], lossSentinel);
            for (uint32_t c = 0; c < numClasses; ++c)
                EXPECT_EQ(actualGradient[static_cast<size_t>(row) * numClasses + c], gradientSentinel);
        }

        loss.cleanup();
    }
}
