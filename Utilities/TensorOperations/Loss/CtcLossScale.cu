#include "Utilities/TensorOperations/Loss/CtcLoss.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"

namespace ThorImplementation {

namespace {

__global__ void scaleFloatTensor(float* values, uint64_t numElements, float scale) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (; index < numElements; index += stride) {
        values[index] *= scale;
    }
}

__global__ void correctCtcEmptyTargetRows(const float* activations,
                                          const int* labelLengths,
                                          const int* inputLengths,
                                          float* costs,
                                          float* gradients,
                                          uint32_t maxTimeSteps,
                                          uint32_t numClasses) {
    const uint32_t batch = blockIdx.x;
    if (threadIdx.x != 0 || labelLengths[batch] != 0)
        return;

    const int inputLength = inputLengths[batch];
    if (inputLength < 0 || inputLength > static_cast<int>(maxTimeSteps)) {
        asm("trap;");
        return;
    }

    float cost = 0.0f;
    const uint64_t batchBase = static_cast<uint64_t>(batch) * maxTimeSteps * numClasses;
    for (uint32_t t = 0; t < maxTimeSteps; ++t) {
        const uint64_t timeBase = batchBase + static_cast<uint64_t>(t) * numClasses;
        if (t >= static_cast<uint32_t>(inputLength)) {
            for (uint32_t c = 0; c < numClasses; ++c)
                gradients[timeBase + c] = 0.0f;
            continue;
        }

        float maxActivation = activations[timeBase];
        for (uint32_t c = 1; c < numClasses; ++c)
            maxActivation = fmaxf(maxActivation, activations[timeBase + c]);

        float expSum = 0.0f;
        for (uint32_t c = 0; c < numClasses; ++c)
            expSum += expf(activations[timeBase + c] - maxActivation);
        const float logNormalizer = maxActivation + logf(expSum);

        // cuDNN CTC fixes the blank class at index zero. For an empty target,
        // the all-blank path is the only valid alignment.
        cost += logNormalizer - activations[timeBase];
        for (uint32_t c = 0; c < numClasses; ++c) {
            const float probability = expf(activations[timeBase + c] - logNormalizer);
            gradients[timeBase + c] = probability - (c == 0 ? 1.0f : 0.0f);
        }
    }
    costs[batch] = cost;
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

void launchCorrectCtcEmptyTargetRows(const float* activations,
                                     const int* labelLengths,
                                     const int* inputLengths,
                                     float* costs,
                                     float* gradients,
                                     uint32_t batchSize,
                                     uint32_t maxTimeSteps,
                                     uint32_t numClasses,
                                     Stream stream) {
    THOR_THROW_IF_FALSE(activations != nullptr);
    THOR_THROW_IF_FALSE(labelLengths != nullptr);
    THOR_THROW_IF_FALSE(inputLengths != nullptr);
    THOR_THROW_IF_FALSE(costs != nullptr);
    THOR_THROW_IF_FALSE(gradients != nullptr);
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(maxTimeSteps > 0);
    THOR_THROW_IF_FALSE(numClasses > 1);

    ScopedGpu scopedGpu(stream.getGpuNum());
    correctCtcEmptyTargetRows<<<batchSize, 1, 0, stream.getStream()>>>(
        activations, labelLengths, inputLengths, costs, gradients, maxTimeSteps, numClasses);
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
