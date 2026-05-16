#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ThorImplementation {

/**
 * Internal backend primitive for RMSNorm + blockwise 16-wide randomized/Hadamard transform + per-CTA abs max.
 *
 * This mirrors the cuDNN Frontend FE-OSS RMSNorm+RHT+AbsMax support surface: rank-2 row-major BF16
 * x/w/o tensors, FP32 abs max, N divisible by 16, and one abs max value per rowsPerCta rows. This is
 * deliberately not a Thor API layer mode or user-visible epilogue. It is backend quantization plumbing
 * that a future low-precision lowering pass can select internally when it sees the right RMSNorm ->
 * quantization -> GEMM pattern.
 *
 */
struct CudnnRmsNormRhtAbsMaxDescriptor {
    uint64_t outerSize = 0;
    uint64_t normalizedFeatureCount = 0;

    TensorDescriptor::DataType inputDataType = TensorDescriptor::DataType::BF16;
    TensorDescriptor::DataType outputDataType = TensorDescriptor::DataType::BF16;
    TensorDescriptor::DataType parameterDataType = TensorDescriptor::DataType::BF16;
    TensorDescriptor::DataType absMaxDataType = TensorDescriptor::DataType::FP32;

    float epsilon = 1.0e-5f;
    uint32_t numThreads = 0;  // 0: resolve from NVIDIA's published table/fallback constraints.
    uint32_t rowsPerCta = 0;  // 0: use the default heuristic.
    std::string debugName = "thor_rms_norm_rht_abs_max";

    [[nodiscard]] uint32_t resolvedNumThreads() const;
    [[nodiscard]] uint32_t resolvedRowsPerCta() const;
    [[nodiscard]] uint64_t absMaxElementCount() const;

    void validate() const;
    std::string cacheKey(std::string_view passName, int gpuNum) const;
};

struct CudnnRmsNormRhtAbsMaxForwardArgs {
    Tensor x;
    Tensor scale;
    Tensor y;

    // Per-CTA absolute maximum values for the transformed output. This is FP32 scale metadata,
    // not an index tensor.
    Tensor absMax;
};

class CudnnRmsNormRhtAbsMax {
   public:
    static CudnnRmsNormRhtAbsMax& instance();

    void forward(const CudnnRmsNormRhtAbsMaxDescriptor& descriptor, const CudnnRmsNormRhtAbsMaxForwardArgs& args, Stream stream);
    void warmForward(const CudnnRmsNormRhtAbsMaxDescriptor& descriptor, int gpuNum);

    void clearCache();
    size_t cachedGraphCount() const;

    static bool frontendAvailable();

   private:
    CudnnRmsNormRhtAbsMax() = default;
};

}  // namespace ThorImplementation
