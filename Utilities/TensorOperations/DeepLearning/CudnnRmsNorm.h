#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/CudnnFrontendPlan.h"
#include "Utilities/Common/Stream.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace ThorImplementation {

enum class CudnnRmsNormFusedActivation { NONE, SWISH };

const char* toString(CudnnRmsNormFusedActivation activation);
CudnnRmsNormFusedActivation cudnnRmsNormFusedActivationFromString(std::string_view value);

/**
 * cuDNN Frontend RMSNorm wrapper.
 *
 * Thor exposes RMSNorm over an arbitrary contiguous trailing normalized shape. The frontend graph is built
 * with NVIDIA's canonical 4D view:
 *
 *   x/y:   [outer, hidden, 1, 1]
 *   scale: [1, hidden, 1, 1]
 *   stats: [outer, 1, 1, 1]
 *
 * where hidden is the product of the normalized trailing dimensions and outer is the product of the remaining
 * leading dimensions. No tensor materialization is required for this view; cuDNN only sees packed dimensions
 * and strides for the contiguous allocation already owned by Thor.
 */
struct CudnnRmsNormDescriptor {
    uint64_t outerSize = 0;
    uint64_t normalizedFeatureCount = 0;

    DataType inputDataType = DataType::FP16;
    DataType outputDataType = DataType::FP16;
    DataType parameterDataType = DataType::FP32;
    DataType computeDataType = DataType::FP32;

    float epsilon = 1.0e-5f;
    bool training = true;
    CudnnRmsNormFusedActivation fusedActivation = CudnnRmsNormFusedActivation::NONE;
    std::string debugName = "thor_rms_norm";

    // Optional explicit cuDNN 4-D physical view. When unset, Thor uses the
    // legacy contiguous [outer, hidden, 1, 1] view. T8C uses this to expose
    // retained padded ragged storage directly as [B,C,1,W], with statistics
    // [B,1,1,W], so normalization is per timestep across channels without a
    // transpose/materialization. All six fields must be set together.
    std::optional<std::array<int64_t, 4>> ioDimensions;
    std::optional<std::array<int64_t, 4>> ioStrides;
    std::optional<std::array<int64_t, 4>> parameterDimensions;
    std::optional<std::array<int64_t, 4>> parameterStrides;
    std::optional<std::array<int64_t, 4>> statsDimensions;
    std::optional<std::array<int64_t, 4>> statsStrides;

    void validateForward() const;
    void validateBackward() const;
    std::string cacheKey(std::string_view passName, int gpuNum) const;
};

struct CudnnRmsNormForwardArgs {
    Tensor x;
    Tensor scale;
    Tensor y;

    // Required when descriptor.training is true. FP32 tensor with outerSize elements.
    std::optional<Tensor> invVariance;
};

struct CudnnRmsNormBackwardArgs {
    Tensor dy;
    Tensor x;
    Tensor scale;
    Tensor invVariance;
    Tensor dx;
    Tensor dscale;
};

/**
 * One finalized RMSNorm execution plan owned by one independently executable
 * operation/connection. The descriptor travels with the local executable so the
 * hot path never consults process-global selection state.
 */
class CudnnRmsNormExecutablePlan final : public AcceleratorBackendLocalExecutionStateTag {
   public:
    CudnnRmsNormExecutablePlan(const CudnnRmsNormExecutablePlan&) = delete;
    CudnnRmsNormExecutablePlan& operator=(const CudnnRmsNormExecutablePlan&) = delete;
    CudnnRmsNormExecutablePlan(CudnnRmsNormExecutablePlan&&) noexcept = default;
    CudnnRmsNormExecutablePlan& operator=(CudnnRmsNormExecutablePlan&&) noexcept = default;
    ~CudnnRmsNormExecutablePlan() = default;

    [[nodiscard]] const CudnnRmsNormDescriptor& descriptor() const noexcept { return descriptor_; }
    [[nodiscard]] const CudnnFrontendPlanSelection& selection() const noexcept { return executable_.selection(); }
    [[nodiscard]] uint64_t workspaceBytes() const noexcept { return executable_.workspaceBytes(); }
    [[nodiscard]] uintptr_t executableId() const noexcept { return executable_.executableId(); }
    [[nodiscard]] int gpuNum() const noexcept { return gpu_num_; }
    [[nodiscard]] bool isForward() const noexcept { return pass_ == Pass::Forward; }
    [[nodiscard]] bool isBackward() const noexcept { return pass_ == Pass::Backward; }

   private:
    enum class Pass { Forward, Backward };

    CudnnRmsNormExecutablePlan(CudnnRmsNormDescriptor descriptor,
                               Pass pass,
                               int gpuNum,
                               CudnnFrontendExecutablePlan executable)
        : descriptor_(std::move(descriptor)), pass_(pass), gpu_num_(gpuNum), executable_(std::move(executable)) {}

    CudnnRmsNormDescriptor descriptor_;
    Pass pass_;
    int gpu_num_ = -1;
    CudnnFrontendExecutablePlan executable_;

    friend class CudnnRmsNorm;
};

class CudnnRmsNorm {
   public:
    static CudnnRmsNorm& instance();

    // Preparation is the only phase allowed to consult the process-global
    // selection cache or construct/replay a Frontend executable graph. Every
    // call returns a fresh operation-local finalized executable.
    [[nodiscard]] CudnnRmsNormExecutablePlan prepareForward(const CudnnRmsNormDescriptor& descriptor, Stream stream);
    [[nodiscard]] CudnnRmsNormExecutablePlan prepareBackward(const CudnnRmsNormDescriptor& descriptor, Stream stream);

    void forward(const CudnnRmsNormExecutablePlan& plan,
                 const CudnnRmsNormForwardArgs& args,
                 std::optional<Tensor>& workspace,
                 Stream stream);
    void backward(const CudnnRmsNormExecutablePlan& plan,
                  const CudnnRmsNormBackwardArgs& args,
                  std::optional<Tensor>& workspace,
                  Stream stream);

    // Selection-only preparation helpers. These may populate the immutable
    // process-global selection cache, but never retain an executable graph.
    [[nodiscard]] uint64_t forwardWorkspaceSizeInBytes(const CudnnRmsNormDescriptor& descriptor, int gpuNum);
    [[nodiscard]] uint64_t backwardWorkspaceSizeInBytes(const CudnnRmsNormDescriptor& descriptor, int gpuNum);
    void warmForward(const CudnnRmsNormDescriptor& descriptor, int gpuNum);
    void warmBackward(const CudnnRmsNormDescriptor& descriptor, int gpuNum);

    void clearSelectionCache();
    [[nodiscard]] size_t cachedSelectionCount() const;
    [[nodiscard]] uint64_t selectionCacheHitCount() const;
    [[nodiscard]] uint64_t selectionCacheMissCount() const;

    static bool frontendAvailable();

   private:
    CudnnRmsNorm() = default;
};

static_assert(AcceleratorBackendLocalExecutionState<CudnnRmsNormExecutablePlan>);
static_assert(!std::is_copy_constructible_v<CudnnRmsNormExecutablePlan>);
static_assert(!std::is_copy_assignable_v<CudnnRmsNormExecutablePlan>);
static_assert(std::is_move_constructible_v<CudnnRmsNormExecutablePlan>);
static_assert(std::is_move_assignable_v<CudnnRmsNormExecutablePlan>);

}  // namespace ThorImplementation
