#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ThorImplementation {

enum class AttentionTensorLayout {
    BHSD,
    BSHD,
};

/**
 * cuDNN Frontend scaled-dot-product attention wrapper for Thor tensors.
 *
 * Semantic tensor order is always [B, H, S, D].  The physical layout is expressed by the supplied strides,
 * so BHSD, BSHD, interleaved projections, packed/ragged layouts, and future Thor materialization layouts can all
 * be represented without changing the public API.
 */
struct AttentionTensorSpec {
    std::vector<int64_t> dimensions;
    std::vector<int64_t> strides;
    DataType dataType = DataType::FP16;
    bool ragged = false;

    static AttentionTensorSpec bhsd(
        int64_t batch, int64_t heads, int64_t sequenceLength, int64_t headDim, DataType dataType);

    static AttentionTensorSpec bshd(
        int64_t batch, int64_t heads, int64_t sequenceLength, int64_t headDim, DataType dataType);

    static AttentionTensorSpec fromLayout(AttentionTensorLayout layout,
                                          int64_t batch,
                                          int64_t heads,
                                          int64_t sequenceLength,
                                          int64_t headDim,
                                          DataType dataType);

    std::string toString() const;
};

enum class AttentionMaskKind {
    None,
    CausalTopLeft,
    CausalBottomRight,
    SlidingWindowTopLeft,
    SlidingWindowBottomRight,
};

struct AttentionDropoutConfig {
    float probability = 0.0f;
    bool usePhilox = true;
    bool dumpMaskForDebug = false;
};

struct AttentionPagedKvConfig {
    int64_t maxSequenceLengthKv = 0;
};

struct CudnnAttentionDescriptor {
    AttentionTensorSpec q;
    AttentionTensorSpec k;
    AttentionTensorSpec v;
    AttentionTensorSpec o;

    // Additive bias is a score-space tensor in semantic [B,Hq,Sq,Skv] order.
    // Production forward supports independently broadcasting any score dimension by setting
    // B/Hq/Sq/Skv to 1. Production backward sends only dense or batch/head-broadcast bias
    // directly to cuDNN; sequence-broadcast bias is materialized to dense by autodiff and
    // dense dBias is reduced back to the public bias shape.
    std::optional<AttentionTensorSpec> bias;

    // Backward dBias is also score-space, but Thor currently exposes only the full dense dBias tensor.
    // Its dtype is resolved from the runtime output tensor so the cuDNN graph matches the execution-stage
    // allocation instead of assuming the Q/K/V IO dtype.
    std::optional<AttentionTensorSpec> dBias;

    DataType computeDataType = DataType::FP32;
    DataType intermediateDataType = DataType::FP32;

    // Default is the usual 1 / sqrt(Dqk).  Set attentionScale explicitly when a model uses a custom softcap/scale policy.
    std::optional<float> attentionScale;

    AttentionMaskKind maskKind = AttentionMaskKind::None;
    int64_t diagonalLeftBound = 0;
    int64_t diagonalRightBound = 0;

    bool generateStats = false;
    bool deterministicBackward = false;
    bool usePaddingMask = false;
    bool useAlibiMask = false;
    bool useBias = false;
    bool usePagedKvCache = false;
    bool useFp8 = false;

    AttentionDropoutConfig dropout;
    AttentionPagedKvConfig pagedKv;

    std::string debugName = "thor_sdpa";

    int64_t batchSize() const;
    int64_t queryHeads() const;
    int64_t keyValueHeads() const;
    int64_t queryLength() const;
    int64_t keyValueLength() const;
    int64_t qkHeadDim() const;
    int64_t vHeadDim() const;

    void validateForward() const;
    void validateBackward() const;
    std::string cacheKey(std::string_view passName, int gpuNum) const;
};

struct CudnnAttentionForwardArgs {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor o;

    // Required when descriptor.generateStats is true.  Shape is cuDNN-controlled; use FP32.
    std::optional<Tensor> stats;

    // Optional tensors enabled by descriptor flags.
    std::optional<Tensor> bias;
    std::optional<Tensor> seqLenQ;
    std::optional<Tensor> seqLenKv;
    std::optional<Tensor> raggedOffsetQ;
    std::optional<Tensor> raggedOffsetK;
    std::optional<Tensor> raggedOffsetV;
    std::optional<Tensor> raggedOffsetO;
    std::optional<Tensor> dropoutSeed;
    std::optional<Tensor> dropoutOffset;
    std::optional<Tensor> dropoutMask;
    std::optional<Tensor> dropoutScale;
    std::optional<Tensor> pageTableK;
    std::optional<Tensor> pageTableV;

    // FP8 only.  Scale/descale tensors are scalar FP32 tensors on device.
    std::optional<Tensor> descaleQ;
    std::optional<Tensor> descaleK;
    std::optional<Tensor> descaleV;
    std::optional<Tensor> descaleS;
    std::optional<Tensor> scaleS;
    std::optional<Tensor> scaleO;
    std::optional<Tensor> amaxS;
    std::optional<Tensor> amaxO;
};

struct CudnnAttentionBackwardArgs {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor o;
    Tensor dO;
    Tensor stats;
    Tensor dQ;
    Tensor dK;
    Tensor dV;

    std::optional<Tensor> bias;
    std::optional<Tensor> dBias;
    std::optional<Tensor> seqLenQ;
    std::optional<Tensor> seqLenKv;
    std::optional<Tensor> raggedOffsetQ;
    std::optional<Tensor> raggedOffsetK;
    std::optional<Tensor> raggedOffsetV;
    std::optional<Tensor> raggedOffsetO;
    std::optional<Tensor> raggedOffsetDO;
    std::optional<Tensor> raggedOffsetDQ;
    std::optional<Tensor> raggedOffsetDK;
    std::optional<Tensor> raggedOffsetDV;
    std::optional<Tensor> dropoutSeed;
    std::optional<Tensor> dropoutOffset;

    // FP8 backward.  Kept in the same struct so the planner can cache a single logical attention signature.
    std::optional<Tensor> descaleQ;
    std::optional<Tensor> descaleK;
    std::optional<Tensor> descaleV;
    std::optional<Tensor> descaleO;
    std::optional<Tensor> descaleDO;
    std::optional<Tensor> descaleS;
    std::optional<Tensor> descaleDP;
    std::optional<Tensor> scaleS;
    std::optional<Tensor> scaleDQ;
    std::optional<Tensor> scaleDK;
    std::optional<Tensor> scaleDV;
    std::optional<Tensor> scaleDP;
    std::optional<Tensor> amaxDQ;
    std::optional<Tensor> amaxDK;
    std::optional<Tensor> amaxDV;
    std::optional<Tensor> amaxDP;
};

/**
 * Thread-safe cached executor for cuDNN SDPA graphs.
 *
 * This is deliberately a low-level TensorOperation first.  It is the right boundary to wire into Thor's expression
 * scheduler as a first-class Attention execution stage, and it is also directly testable against CPU/PyTorch or
 * decomposed Thor matmul-softmax-matmul reference paths.
 */
class CudnnScaledDotProductAttention {
   public:
    static CudnnScaledDotProductAttention& instance();

    void forward(const CudnnAttentionDescriptor& descriptor, const CudnnAttentionForwardArgs& args, Stream stream);
    void backward(const CudnnAttentionDescriptor& descriptor, const CudnnAttentionBackwardArgs& args, Stream stream);

    // Useful after shape inference / stamping when a model will repeatedly run the same attention shape.
    void warmForward(const CudnnAttentionDescriptor& descriptor, int gpuNum);
    void warmBackward(const CudnnAttentionDescriptor& descriptor, int gpuNum);

    void clearCache();
    size_t cachedGraphCount() const;

    static bool frontendAvailable();

   private:
    CudnnScaledDotProductAttention() = default;
};

}  // namespace ThorImplementation
