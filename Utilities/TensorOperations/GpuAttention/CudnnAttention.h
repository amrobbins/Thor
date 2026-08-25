#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/CudnnFrontendPlan.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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


struct CudnnRaggedAttentionScratch {
    Tensor seqLenQ;
    Tensor seqLenKv;
    Tensor qElementOffsets;
    Tensor kElementOffsets;
    Tensor vElementOffsets;
    Tensor oElementOffsets;
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
    // Canonical Thor token row partitions for the sides that are physically ragged.
    // Fully-ragged attention supplies both. Mixed attention supplies only the
    // partition for its ragged sequence domain; the opposite dense domain is
    // represented by uniform sequence lengths derived from the descriptor.
    // Canonical offset dtype is UINT32 or UINT64.
    std::optional<Tensor> qRowPartitionOffsets;
    std::optional<Tensor> kvRowPartitionOffsets;
    // Preallocated backend metadata scratch. Required when any side is ragged.
    std::optional<CudnnRaggedAttentionScratch> raggedScratch;
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
    // Canonical Thor token row partitions for the sides that are physically ragged.
    // Fully-ragged attention supplies both. Mixed attention supplies only the
    // partition for its ragged sequence domain; the opposite dense domain is
    // represented by uniform sequence lengths derived from the descriptor.
    // Canonical offset dtype is UINT32 or UINT64.
    std::optional<Tensor> qRowPartitionOffsets;
    std::optional<Tensor> kvRowPartitionOffsets;
    // Preallocated backend metadata scratch. Required when any side is ragged.
    std::optional<CudnnRaggedAttentionScratch> raggedScratch;
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
 * Move-only, operation-local cuDNN Frontend SDPA executable.
 *
 * The process-global repository retains only CudnnFrontendPlanSelection recipes.
 * Every stamped/placed attention execution receives its own finalized Frontend
 * graph/plan and caller-owned workspace.
 */
class CudnnAttentionExecutablePlan final : public AcceleratorBackendLocalExecutionStateTag {
   public:
    CudnnAttentionExecutablePlan(const CudnnAttentionExecutablePlan&) = delete;
    CudnnAttentionExecutablePlan& operator=(const CudnnAttentionExecutablePlan&) = delete;
    CudnnAttentionExecutablePlan(CudnnAttentionExecutablePlan&&) noexcept = default;
    CudnnAttentionExecutablePlan& operator=(CudnnAttentionExecutablePlan&&) noexcept = default;
    ~CudnnAttentionExecutablePlan() = default;

    [[nodiscard]] const CudnnAttentionDescriptor& descriptor() const noexcept { return descriptor_; }
    [[nodiscard]] const CudnnFrontendPlanSelection& selection() const noexcept { return executable_.selection(); }
    [[nodiscard]] uint64_t workspaceBytes() const noexcept { return executable_.workspaceBytes(); }
    [[nodiscard]] uintptr_t executableId() const noexcept { return executable_.executableId(); }
    [[nodiscard]] int gpuNum() const noexcept { return gpu_num_; }
    [[nodiscard]] bool isForward() const noexcept { return pass_ == Pass::Forward; }
    [[nodiscard]] bool isBackward() const noexcept { return pass_ == Pass::Backward; }

   private:
    enum class Pass { Forward, Backward };

    CudnnAttentionExecutablePlan(CudnnAttentionDescriptor descriptor,
                                 Pass pass,
                                 int gpuNum,
                                 CudnnFrontendExecutablePlan executable)
        : descriptor_(std::move(descriptor)), pass_(pass), gpu_num_(gpuNum), executable_(std::move(executable)) {}

    CudnnAttentionDescriptor descriptor_;
    Pass pass_;
    int gpu_num_ = -1;
    CudnnFrontendExecutablePlan executable_;

    friend class CudnnScaledDotProductAttention;
};

/**
 * cuDNN Frontend SDPA selection repository and operation-local executor.
 *
 * Preparation is placement/stamping work.  It may consult the immutable global
 * selection cache and replay/deserialise a selected recipe into a fresh local
 * executable. Runtime forward/backward accept only an already-prepared plan and
 * never consult selection state or construct Frontend objects.
 */
class CudnnScaledDotProductAttention {
   public:
    static CudnnScaledDotProductAttention& instance();

    [[nodiscard]] CudnnAttentionExecutablePlan prepareForward(const CudnnAttentionDescriptor& descriptor,
                                                               const CudnnAttentionForwardArgs& args,
                                                               Stream stream);
    [[nodiscard]] CudnnAttentionExecutablePlan prepareBackward(const CudnnAttentionDescriptor& descriptor,
                                                                const CudnnAttentionBackwardArgs& args,
                                                                Stream stream);

    void forward(const CudnnAttentionExecutablePlan& plan,
                 const CudnnAttentionForwardArgs& args,
                 std::optional<Tensor>& workspace,
                 Stream stream);
    void backward(const CudnnAttentionExecutablePlan& plan,
                  const CudnnAttentionBackwardArgs& args,
                  std::optional<Tensor>& workspace,
                  Stream stream);

    // Selection-only helpers for descriptor validation/tests and ahead-of-time
    // warming. They never retain a live executable graph.
    [[nodiscard]] uint64_t forwardWorkspaceSizeInBytes(const CudnnAttentionDescriptor& descriptor, int gpuNum);
    [[nodiscard]] uint64_t backwardWorkspaceSizeInBytes(const CudnnAttentionDescriptor& descriptor, int gpuNum);
    [[nodiscard]] uint64_t forwardWorkspaceSizeInBytes(const CudnnAttentionDescriptor& descriptor,
                                                       const CudnnAttentionForwardArgs& args,
                                                       int gpuNum);
    [[nodiscard]] uint64_t backwardWorkspaceSizeInBytes(const CudnnAttentionDescriptor& descriptor,
                                                        const CudnnAttentionBackwardArgs& args,
                                                        int gpuNum);

    void warmForward(const CudnnAttentionDescriptor& descriptor, int gpuNum);
    void warmBackward(const CudnnAttentionDescriptor& descriptor, int gpuNum);

    void clearSelectionCache();
    [[nodiscard]] size_t cachedSelectionCount() const;
    [[nodiscard]] uint64_t selectionCacheHitCount() const;
    [[nodiscard]] uint64_t selectionCacheMissCount() const;

    static bool frontendAvailable();

   private:
    CudnnScaledDotProductAttention() = default;
};

static_assert(AcceleratorBackendLocalExecutionState<CudnnAttentionExecutablePlan>);
static_assert(!std::is_copy_constructible_v<CudnnAttentionExecutablePlan>);
static_assert(!std::is_copy_assignable_v<CudnnAttentionExecutablePlan>);
static_assert(std::is_move_constructible_v<CudnnAttentionExecutablePlan>);
static_assert(std::is_move_assignable_v<CudnnAttentionExecutablePlan>);

}  // namespace ThorImplementation
