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
