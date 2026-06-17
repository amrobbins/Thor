#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>

#include <cudnn.h>

namespace ThorImplementation {

// cuDNN CTC v1 policy for Thor:
//   * cuDNN only; no native/CPU fallback.
//   * v9 descriptor setup with the CUDA-graph-friendly v8 workspace/compute APIs.
//   * deterministic algorithm only until Thor has an explicit nondeterminism policy knob.
//   * fp32 only because cuDNN returns NOT_SUPPORTED for non-FLOAT CTC compute/data types.
//   * labels, label lengths, and input lengths are device-resident int32 arrays.
//
// Thor stores the dense activation tensor as physical [batchSize, maxTimeSteps, numClasses].
// The cuDNN tensor descriptor presents that same contiguous memory as logical [maxTimeSteps, batchSize, numClasses]
// using custom strides, so no transpose/copy is inserted.
// The blank-index convention is intentionally not abstracted here because cuDNN's CTC
// API does not expose a blank-index parameter. The public Thor layer should hard-code
// and document the cuDNN convention, or insert an explicit class permutation later.
enum class CtcLossAlgorithm { DETERMINISTIC };
enum class CtcLossNormalization { SOFTMAX };
enum class CtcLossOobGradientMode { ZERO, SKIP };

struct CudnnCtcLossConfig {
    uint32_t maxTimeSteps = 0;
    uint32_t batchSize = 0;
    uint32_t numClasses = 0;
    uint32_t maxLabelLength = 0;
    DataType dataType = DataType::FP32;
    CtcLossAlgorithm algorithm = CtcLossAlgorithm::DETERMINISTIC;
    CtcLossNormalization normalization = CtcLossNormalization::SOFTMAX;
    CtcLossOobGradientMode oobGradientMode = CtcLossOobGradientMode::ZERO;
};


void launchCompactPaddedCtcLabels(const int* paddedLabels,
                                  const int* labelLengths,
                                  int* packedLabels,
                                  uint32_t batchSize,
                                  uint32_t maxLabelLength,
                                  Stream stream);

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
                               Stream stream);

class CudnnCtcLossPlan {
   public:
    CudnnCtcLossPlan(const CudnnCtcLossConfig &config, Stream stream);
    ~CudnnCtcLossPlan();

    CudnnCtcLossPlan(const CudnnCtcLossPlan &) = delete;
    CudnnCtcLossPlan &operator=(const CudnnCtcLossPlan &) = delete;

    CudnnCtcLossPlan(CudnnCtcLossPlan &&other) noexcept;
    CudnnCtcLossPlan &operator=(CudnnCtcLossPlan &&other) noexcept;

    static void validateConfig(const CudnnCtcLossConfig &config);

    size_t getWorkspaceSizeInBytes() const { return workspaceSizeInBytes; }
    const CudnnCtcLossConfig &getConfig() const { return config; }

    // activations:    physical [B, T, C], fp32, device memory; cuDNN sees logical [T, B, C]
    // labels:         concatenated int32 target labels, device memory
    // labelLengths:   [B], int32, device memory
    // inputLengths:   [B], int32, device memory
    // costs:          [B], fp32, device memory
    // gradients:      physical [B, T, C], fp32, device memory; cuDNN sees logical [T, B, C]
    // workspace:      device memory of at least getWorkspaceSizeInBytes() bytes
    void run(void *probabilities,
             const int *labels,
             const int *labelLengths,
             const int *inputLengths,
             void *costs,
             void *gradients,
             void *workspace,
             size_t workspaceSizeBytes,
             Stream stream) const;

   private:
    void destroy() noexcept;

    CudnnCtcLossConfig config;
    cudnnTensorDescriptor_t probabilitiesDesc = nullptr;
    cudnnTensorDescriptor_t gradientsDesc = nullptr;
    cudnnCTCLossDescriptor_t ctcLossDesc = nullptr;
    size_t workspaceSizeInBytes = 0;
};

}  // namespace ThorImplementation
