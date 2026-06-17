#include "Utilities/TensorOperations/Loss/CtcLoss.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

namespace ThorImplementation {

namespace {


__global__ void compactPaddedCtcLabelsKernel(const int* paddedLabels,
                                             const int* labelLengths,
                                             int* packedLabels,
                                             uint32_t batchSize,
                                             uint32_t maxLabelLength) {
    // v1 intentionally uses one tiny device-side control kernel. CTC target labels are short
    // (deterministic cuDNN rejects label lengths >= 256), so this avoids host sync/fallback
    // while keeping the public API padded and batch-shaped.
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    uint32_t packedOffset = 0;
    for (uint32_t b = 0; b < batchSize; ++b) {
        const int length = labelLengths[b];
        if (length < 0 || length > static_cast<int>(maxLabelLength)) {
            asm("trap;");
        }
        for (int i = 0; i < length; ++i) {
            packedLabels[packedOffset++] = paddedLabels[b * maxLabelLength + static_cast<uint32_t>(i)];
        }
    }
}

__global__ void scaleFloatTensor(float* values, uint64_t numElements, float scale) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (; index < numElements; index += stride) {
        values[index] *= scale;
    }
}

__global__ void scaleCtcGradientTensor(float* gradients,
                                       const int* inputLengths,
                                       uint32_t batchSize,
                                       uint32_t maxTimeSteps,
                                       uint32_t numClasses,
                                       float scale) {
    const uint64_t numElements = static_cast<uint64_t>(batchSize) * maxTimeSteps * numClasses;
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (; index < numElements; index += stride) {
        const uint32_t t = static_cast<uint32_t>((index / numClasses) % maxTimeSteps);
        const uint32_t b = static_cast<uint32_t>(index / (static_cast<uint64_t>(maxTimeSteps) * numClasses));
        const int validLength = inputLengths[b];
        if (validLength < 0 || validLength > static_cast<int>(maxTimeSteps)) {
            asm("trap;");
        }
        if (t < static_cast<uint32_t>(validLength)) {
            gradients[index] *= scale;
        } else {
            gradients[index] = 0.0f;
        }
    }
}

uint32_t blocksForElements(uint64_t numElements, uint32_t blockSize) {
    uint64_t blocks = (numElements + blockSize - 1) / blockSize;
    if (blocks > 65535)
        blocks = 65535;
    return static_cast<uint32_t>(blocks);
}

void launchScaleFloatTensor(float* values, uint64_t numElements, float scale, Stream stream) {
    if (numElements == 0 || scale == 1.0f)
        return;
    THOR_THROW_IF_FALSE(values != nullptr);
    constexpr uint32_t blockSize = 256;
    ScopedGpu scopedGpu(stream.getGpuNum());
    scaleFloatTensor<<<blocksForElements(numElements, blockSize), blockSize, 0, stream.getStream()>>>(values, numElements, scale);
}

void launchScaleCtcGradientTensor(float* gradients,
                                  const int* inputLengths,
                                  uint32_t batchSize,
                                  uint32_t maxTimeSteps,
                                  uint32_t numClasses,
                                  float scale,
                                  Stream stream) {
    THOR_THROW_IF_FALSE(gradients != nullptr);
    THOR_THROW_IF_FALSE(inputLengths != nullptr);
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(maxTimeSteps > 0);
    THOR_THROW_IF_FALSE(numClasses > 0);
    const uint64_t numElements = static_cast<uint64_t>(batchSize) * maxTimeSteps * numClasses;
    constexpr uint32_t blockSize = 256;
    ScopedGpu scopedGpu(stream.getGpuNum());
    scaleCtcGradientTensor<<<blocksForElements(numElements, blockSize), blockSize, 0, stream.getStream()>>>(
        gradients, inputLengths, batchSize, maxTimeSteps, numClasses, scale);
}

}  // namespace


void launchCompactPaddedCtcLabels(const int* paddedLabels,
                                  const int* labelLengths,
                                  int* packedLabels,
                                  uint32_t batchSize,
                                  uint32_t maxLabelLength,
                                  Stream stream) {
    THOR_THROW_IF_FALSE(paddedLabels != nullptr);
    THOR_THROW_IF_FALSE(labelLengths != nullptr);
    THOR_THROW_IF_FALSE(packedLabels != nullptr);
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(maxLabelLength > 0);
    ScopedGpu scopedGpu(stream.getGpuNum());
    compactPaddedCtcLabelsKernel<<<1, 1, 0, stream.getStream()>>>(paddedLabels, labelLengths, packedLabels, batchSize, maxLabelLength);
}

void launchScaleCtcLossOutputs(float* costs,
                               float* gradients,
                               const int* inputLengths,
                               uint32_t batchSize,
                               uint32_t maxTimeSteps,
                               uint32_t numClasses,
                               uint64_t numCostElements,
                               bool scaleGradients,
                               float lossScale,
                               float gradientScale,
                               Stream stream) {
    launchScaleFloatTensor(costs, numCostElements, lossScale, stream);
    if (scaleGradients) {
        launchScaleCtcGradientTensor(gradients, inputLengths, batchSize, maxTimeSteps, numClasses, gradientScale, stream);
    }
}

}  // namespace ThorImplementation
