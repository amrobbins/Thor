#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Loss/SparseCategoricalCrossEntropyWithLogitsLoss.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

struct SparseCeReference {
    vector<float> loss;
    vector<float> gradient;
};

SparseCeReference referenceSparseCategoricalCrossEntropyWithLogits(const vector<float> &logits,
                                                                   const vector<uint32_t> &labels,
                                                                   const vector<uint8_t> *mask,
                                                                   uint32_t numRows,
                                                                   uint32_t numClasses,
                                                                   uint32_t lossScalingFactor,
                                                                   float lossWeight,
                                                                   bool hasIgnoreIndex,
                                                                   uint32_t ignoreIndex) {
    SparseCeReference reference;
    reference.loss.assign(numRows, 0.0f);
    reference.gradient.assign(static_cast<size_t>(numRows) * numClasses, 0.0f);

    for (uint32_t row = 0; row < numRows; ++row) {
        const uint32_t label = labels[row];
        bool valid = true;
        if (hasIgnoreIndex && label == ignoreIndex)
            valid = false;
        if (valid && mask != nullptr)
            valid = (*mask)[row] > 0;
        if (valid && label >= numClasses)
            valid = false;
        if (!valid)
            continue;

        const size_t rowOffset = static_cast<size_t>(row) * numClasses;
        float rowMax = -numeric_limits<float>::infinity();
        for (uint32_t c = 0; c < numClasses; ++c) {
            const float z = logits[rowOffset + c];
            rowMax = std::max(rowMax, std::isfinite(z) ? z : -numeric_limits<float>::infinity());
        }

        float sumExp = 0.0f;
        for (uint32_t c = 0; c < numClasses; ++c) {
            const float z = logits[rowOffset + c];
            sumExp += std::exp((std::isfinite(z) ? z : -numeric_limits<float>::infinity()) - rowMax);
        }

        reference.loss[row] = (std::log(sumExp) + rowMax - logits[rowOffset + label]) * lossWeight;
        const float invSumExp = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
        for (uint32_t c = 0; c < numClasses; ++c) {
            const float z = logits[rowOffset + c];
            const float probability = std::exp((std::isfinite(z) ? z : -numeric_limits<float>::infinity()) - rowMax) * invSumExp;
            reference.gradient[rowOffset + c] =
                (probability - (c == label ? 1.0f : 0.0f)) * static_cast<float>(lossScalingFactor) * lossWeight;
        }
    }

    return reference;
}

void assertVectorNear(const vector<float> &actual, const vector<float> &expected, float tolerance, const char *name) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::abs(actual[i] - expected[i]);
        if (diff >= tolerance || !std::isfinite(diff)) {
            printf("%s[%zu] expected=%0.9f actual=%0.9f diff=%0.9f\n", name, i, expected[i], actual[i], diff);
        }
        ASSERT_LT(diff, tolerance);
    }
}

}  // namespace

TEST(SparseCategoricalCrossEntropyWithLogitsLoss, ForwardAndBackwardFp32UInt32MatchReference) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);

    constexpr uint32_t numRows = 4;
    constexpr uint32_t numClasses = 5;
    constexpr uint32_t lossScalingFactor = 3;
    constexpr float lossWeight = 0.25f;

    const vector<float> logitsHost = {1.5f, -0.25f, 0.75f, 2.0f, -1.0f,
                                      -2.0f, 0.5f, 3.0f, 1.25f, -0.75f,
                                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                      4.0f, 1.0f, -3.0f, 0.5f, 2.25f};
    const vector<uint32_t> labelsHost = {3, 2, 0, 4};

    Tensor labels(cpuPlacement, TensorDescriptor(DataType::UINT32, {numRows}));
    Tensor labelsD = labels.clone(gpuPlacement);
    Tensor logits(cpuPlacement, TensorDescriptor(DataType::FP32, {numRows, numClasses}));
    Tensor logitsD = logits.clone(gpuPlacement);
    Tensor loss(cpuPlacement, TensorDescriptor(DataType::FP32, {numRows}));
    Tensor lossD = loss.clone(gpuPlacement);
    Tensor gradient(cpuPlacement, TensorDescriptor(DataType::FP32, {numRows, numClasses}));
    Tensor gradientD = gradient.clone(gpuPlacement);

    std::copy(labelsHost.begin(), labelsHost.end(), static_cast<uint32_t *>(labels.getMemPtr()));
    std::copy(logitsHost.begin(), logitsHost.end(), static_cast<float *>(logits.getMemPtr()));

    labelsD.copyFromAsync(labels, stream);
    logitsD.copyFromAsync(logits, stream);

    launchSparseCategoricalCrossEntropyWithLogits<uint32_t, float, float, uint8_t>(labelsD.getMemPtr(),
                                                                                  logitsD.getMemPtr(),
                                                                                  nullptr,
                                                                                  lossD.getMemPtr(),
                                                                                  gradientD.getMemPtr(),
                                                                                  numClasses,
                                                                                  numRows,
                                                                                  true,
                                                                                  lossScalingFactor,
                                                                                  lossWeight,
                                                                                  false,
                                                                                  0,
                                                                                  false,
                                                                                  stream);

    loss.copyFromAsync(lossD, stream);
    gradient.copyFromAsync(gradientD, stream);
    stream.synchronize();

    vector<float> actualLoss(numRows);
    vector<float> actualGradient(static_cast<size_t>(numRows) * numClasses);
    std::copy(static_cast<float *>(loss.getMemPtr()), static_cast<float *>(loss.getMemPtr()) + actualLoss.size(), actualLoss.begin());
    std::copy(static_cast<float *>(gradient.getMemPtr()),
              static_cast<float *>(gradient.getMemPtr()) + actualGradient.size(),
              actualGradient.begin());

    SparseCeReference reference = referenceSparseCategoricalCrossEntropyWithLogits(
        logitsHost, labelsHost, nullptr, numRows, numClasses, lossScalingFactor, lossWeight, false, 0);

    assertVectorNear(actualLoss, reference.loss, 1.0e-5f, "loss");
    assertVectorNear(actualGradient, reference.gradient, 1.0e-5f, "gradient");
}

TEST(SparseCategoricalCrossEntropyWithLogitsLoss, ForwardAndBackwardHonorMaskAndIgnoreIndex) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);

    constexpr uint32_t numRows = 5;
    constexpr uint32_t numClasses = 4;
    constexpr uint32_t ignoreIndex = 99;
    constexpr uint32_t lossScalingFactor = 2;
    constexpr float lossWeight = 0.5f;

    const vector<float> logitsHost = {0.25f, 1.25f, -0.5f, 2.0f,
                                      1.0f, -1.0f, 0.0f, 0.5f,
                                      -0.75f, 2.5f, 0.25f, 1.5f,
                                      3.0f, -2.0f, 1.0f, 0.0f,
                                      -1.0f, 0.75f, 1.75f, -0.25f};
    const vector<uint32_t> labelsHost = {3, ignoreIndex, 1, 2, 0};
    const vector<uint8_t> maskHost = {1, 1, 0, 1, 1};

    Tensor labels(cpuPlacement, TensorDescriptor(DataType::UINT32, {numRows}));
    Tensor labelsD = labels.clone(gpuPlacement);
    Tensor logits(cpuPlacement, TensorDescriptor(DataType::FP32, {numRows, numClasses}));
    Tensor logitsD = logits.clone(gpuPlacement);
    Tensor mask(cpuPlacement, TensorDescriptor(DataType::UINT8, {numRows}));
    Tensor maskD = mask.clone(gpuPlacement);
    Tensor loss(cpuPlacement, TensorDescriptor(DataType::FP32, {numRows}));
    Tensor lossD = loss.clone(gpuPlacement);
    Tensor gradient(cpuPlacement, TensorDescriptor(DataType::FP32, {numRows, numClasses}));
    Tensor gradientD = gradient.clone(gpuPlacement);

    std::copy(labelsHost.begin(), labelsHost.end(), static_cast<uint32_t *>(labels.getMemPtr()));
    std::copy(logitsHost.begin(), logitsHost.end(), static_cast<float *>(logits.getMemPtr()));
    std::copy(maskHost.begin(), maskHost.end(), static_cast<uint8_t *>(mask.getMemPtr()));

    labelsD.copyFromAsync(labels, stream);
    logitsD.copyFromAsync(logits, stream);
    maskD.copyFromAsync(mask, stream);

    launchSparseCategoricalCrossEntropyWithLogits<uint32_t, float, float, uint8_t>(labelsD.getMemPtr(),
                                                                                  logitsD.getMemPtr(),
                                                                                  maskD.getMemPtr(),
                                                                                  lossD.getMemPtr(),
                                                                                  gradientD.getMemPtr(),
                                                                                  numClasses,
                                                                                  numRows,
                                                                                  true,
                                                                                  lossScalingFactor,
                                                                                  lossWeight,
                                                                                  true,
                                                                                  ignoreIndex,
                                                                                  true,
                                                                                  stream);

    loss.copyFromAsync(lossD, stream);
    gradient.copyFromAsync(gradientD, stream);
    stream.synchronize();

    vector<float> actualLoss(numRows);
    vector<float> actualGradient(static_cast<size_t>(numRows) * numClasses);
    std::copy(static_cast<float *>(loss.getMemPtr()), static_cast<float *>(loss.getMemPtr()) + actualLoss.size(), actualLoss.begin());
    std::copy(static_cast<float *>(gradient.getMemPtr()),
              static_cast<float *>(gradient.getMemPtr()) + actualGradient.size(),
              actualGradient.begin());

    SparseCeReference reference = referenceSparseCategoricalCrossEntropyWithLogits(
        logitsHost, labelsHost, &maskHost, numRows, numClasses, lossScalingFactor, lossWeight, true, ignoreIndex);

    assertVectorNear(actualLoss, reference.loss, 1.0e-5f, "loss");
    assertVectorNear(actualGradient, reference.gradient, 1.0e-5f, "gradient");
}

TEST(SparseCategoricalCrossEntropyWithLogitsLoss, ForwardOnlyMatchesReferenceWithNullGradient) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);

    constexpr uint32_t numRows = 3;
    constexpr uint32_t numClasses = 6;
    constexpr float lossWeight = 1.75f;

    const vector<float> logitsHost = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f,
                                      3.0f, 2.0f, 1.0f, 0.0f, -1.0f, -2.0f,
                                      0.125f, -0.5f, 2.5f, 1.5f, -1.5f, 0.75f};
    const vector<uint32_t> labelsHost = {5, 0, 2};

    Tensor labels(cpuPlacement, TensorDescriptor(DataType::UINT32, {numRows}));
    Tensor labelsD = labels.clone(gpuPlacement);
    Tensor logits(cpuPlacement, TensorDescriptor(DataType::FP32, {numRows, numClasses}));
    Tensor logitsD = logits.clone(gpuPlacement);
    Tensor loss(cpuPlacement, TensorDescriptor(DataType::FP32, {numRows}));
    Tensor lossD = loss.clone(gpuPlacement);

    std::copy(labelsHost.begin(), labelsHost.end(), static_cast<uint32_t *>(labels.getMemPtr()));
    std::copy(logitsHost.begin(), logitsHost.end(), static_cast<float *>(logits.getMemPtr()));

    labelsD.copyFromAsync(labels, stream);
    logitsD.copyFromAsync(logits, stream);

    launchSparseCategoricalCrossEntropyWithLogits<uint32_t, float, float, uint8_t>(labelsD.getMemPtr(),
                                                                                  logitsD.getMemPtr(),
                                                                                  nullptr,
                                                                                  lossD.getMemPtr(),
                                                                                  nullptr,
                                                                                  numClasses,
                                                                                  numRows,
                                                                                  false,
                                                                                  1,
                                                                                  lossWeight,
                                                                                  false,
                                                                                  0,
                                                                                  false,
                                                                                  stream);

    loss.copyFromAsync(lossD, stream);
    stream.synchronize();

    vector<float> actualLoss(numRows);
    std::copy(static_cast<float *>(loss.getMemPtr()), static_cast<float *>(loss.getMemPtr()) + actualLoss.size(), actualLoss.begin());

    SparseCeReference reference = referenceSparseCategoricalCrossEntropyWithLogits(
        logitsHost, labelsHost, nullptr, numRows, numClasses, 1, lossWeight, false, 0);

    assertVectorNear(actualLoss, reference.loss, 1.0e-5f, "loss");
}

TEST(SparseCategoricalCrossEntropyWithLogitsLoss, RaggedActiveCountGuardsPoisonedInactiveCapacityShortLongShortAndEmpty) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);

    constexpr uint32_t rowCapacity = 6;
    constexpr uint32_t numClasses = 4;
    constexpr uint32_t lossScalingFactor = 3;
    constexpr float lossWeight = 0.25f;
    constexpr float lossSentinel = -77.0f;
    constexpr float gradientSentinel = -55.0f;

    static_assert(kRaggedSparseCategoricalCrossEntropyWithLogitsPartitionRequirement ==
                  RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);

    Tensor labels(cpuPlacement, TensorDescriptor(DataType::UINT32, {rowCapacity}));
    Tensor labelsD = labels.clone(gpuPlacement);
    Tensor logits(cpuPlacement, TensorDescriptor(DataType::FP32, {rowCapacity, numClasses}));
    Tensor logitsD = logits.clone(gpuPlacement);
    Tensor mask(cpuPlacement, TensorDescriptor(DataType::UINT8, {rowCapacity}));
    Tensor maskD = mask.clone(gpuPlacement);
    Tensor loss(cpuPlacement, TensorDescriptor(DataType::FP32, {rowCapacity}));
    Tensor lossD = loss.clone(gpuPlacement);
    Tensor gradient(cpuPlacement, TensorDescriptor(DataType::FP32, {rowCapacity, numClasses}));
    Tensor gradientD = gradient.clone(gpuPlacement);

    const vector<uint32_t> activeCounts = {2, 5, 1, 0};
    for (DataType activeCountDataType : {DataType::UINT32, DataType::UINT64}) {
        Tensor activeCount(cpuPlacement, TensorDescriptor(activeCountDataType, {1}));
        Tensor activeCountD = activeCount.clone(gpuPlacement);

        for (size_t pass = 0; pass < activeCounts.size(); ++pass) {
            const uint32_t activeRows = activeCounts[pass];
            vector<uint32_t> labelsHost(rowCapacity, numeric_limits<uint32_t>::max());
            vector<float> logitsHost(static_cast<size_t>(rowCapacity) * numClasses,
                                     numeric_limits<float>::quiet_NaN());
            vector<uint8_t> maskHost(rowCapacity, 0xff);
            for (uint32_t row = 0; row < activeRows; ++row) {
                labelsHost[row] = (row + static_cast<uint32_t>(pass)) % numClasses;
                maskHost[row] = 1;
                for (uint32_t c = 0; c < numClasses; ++c) {
                    logitsHost[static_cast<size_t>(row) * numClasses + c] =
                        static_cast<float>(10 * pass + 3 * row) * 0.1f + static_cast<float>(c) * 0.35f - 0.4f;
                }
            }

            std::copy(labelsHost.begin(), labelsHost.end(), static_cast<uint32_t *>(labels.getMemPtr()));
            std::copy(logitsHost.begin(), logitsHost.end(), static_cast<float *>(logits.getMemPtr()));
            std::copy(maskHost.begin(), maskHost.end(), static_cast<uint8_t *>(mask.getMemPtr()));
            std::fill_n(static_cast<float *>(loss.getMemPtr()), rowCapacity, lossSentinel);
            std::fill_n(static_cast<float *>(gradient.getMemPtr()), static_cast<size_t>(rowCapacity) * numClasses, gradientSentinel);
            if (activeCountDataType == DataType::UINT32)
                *static_cast<uint32_t *>(activeCount.getMemPtr()) = activeRows;
            else
                *static_cast<uint64_t *>(activeCount.getMemPtr()) = activeRows;

            labelsD.copyFromAsync(labels, stream);
            logitsD.copyFromAsync(logits, stream);
            maskD.copyFromAsync(mask, stream);
            lossD.copyFromAsync(loss, stream);
            gradientD.copyFromAsync(gradient, stream);
            activeCountD.copyFromAsync(activeCount, stream);

            launchRaggedSparseCategoricalCrossEntropyWithLogits<uint32_t, float, float, uint8_t>(
                labelsD.getMemPtr(),
                logitsD.getMemPtr(),
                maskD.getMemPtr(),
                lossD.getMemPtr(),
                gradientD.getMemPtr(),
                activeCountD.getMemPtr(),
                activeCountDataType,
                numClasses,
                rowCapacity,
                true,
                lossScalingFactor,
                lossWeight,
                false,
                0,
                true,
                stream);

            loss.copyFromAsync(lossD, stream);
            gradient.copyFromAsync(gradientD, stream);
            stream.synchronize();

            vector<float> actualLoss(rowCapacity);
            vector<float> actualGradient(static_cast<size_t>(rowCapacity) * numClasses);
            std::copy(static_cast<float *>(loss.getMemPtr()),
                      static_cast<float *>(loss.getMemPtr()) + actualLoss.size(),
                      actualLoss.begin());
            std::copy(static_cast<float *>(gradient.getMemPtr()),
                      static_cast<float *>(gradient.getMemPtr()) + actualGradient.size(),
                      actualGradient.begin());

            const vector<float> activeLogits(logitsHost.begin(), logitsHost.begin() + static_cast<size_t>(activeRows) * numClasses);
            const vector<uint32_t> activeLabels(labelsHost.begin(), labelsHost.begin() + activeRows);
            const vector<uint8_t> activeMask(maskHost.begin(), maskHost.begin() + activeRows);
            const SparseCeReference reference = referenceSparseCategoricalCrossEntropyWithLogits(
                activeLogits, activeLabels, &activeMask, activeRows, numClasses, lossScalingFactor, lossWeight, false, 0);

            for (uint32_t row = 0; row < activeRows; ++row) {
                EXPECT_NEAR(actualLoss[row], reference.loss[row], 1.0e-5f)
                    << "activeCountDataType=" << static_cast<int>(activeCountDataType) << " pass=" << pass << " row=" << row;
                for (uint32_t c = 0; c < numClasses; ++c) {
                    const size_t i = static_cast<size_t>(row) * numClasses + c;
                    EXPECT_NEAR(actualGradient[i], reference.gradient[i], 1.0e-5f)
                        << "activeCountDataType=" << static_cast<int>(activeCountDataType) << " pass=" << pass
                        << " row=" << row << " class=" << c;
                }
            }
            for (uint32_t row = activeRows; row < rowCapacity; ++row) {
                EXPECT_EQ(actualLoss[row], lossSentinel)
                    << "inactive loss row was written for activeCountDataType=" << static_cast<int>(activeCountDataType)
                    << " pass=" << pass << " row=" << row;
                for (uint32_t c = 0; c < numClasses; ++c) {
                    const size_t i = static_cast<size_t>(row) * numClasses + c;
                    EXPECT_EQ(actualGradient[i], gradientSentinel)
                        << "inactive gradient row was written for activeCountDataType=" << static_cast<int>(activeCountDataType)
                        << " pass=" << pass << " row=" << row << " class=" << c;
                }
            }
        }
    }
}
