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

/**
 * cuDNN Frontend LayerNorm wrapper.
 *
 * Thor exposes LayerNorm over an arbitrary contiguous trailing normalized shape.  The frontend graph is built
 * with NVIDIA's canonical 4D view:
 *
 *   x/y:        [outer, hidden, 1, 1]
 *   scale/bias: [1, hidden, 1, 1]
 *   stats:      [outer, 1, 1, 1]
 *
 * where hidden is the product of the normalized trailing dimensions and outer is the product of the remaining
 * leading dimensions.  No tensor materialization is required for this view; cuDNN only sees packed dimensions
 * and strides for the contiguous allocation already owned by Thor.
 */
struct CudnnLayerNormDescriptor {
    uint64_t outerSize = 0;
    uint64_t normalizedFeatureCount = 0;

    DataType inputDataType = DataType::FP16;
    DataType outputDataType = DataType::FP16;
    DataType parameterDataType = DataType::FP32;
    DataType computeDataType = DataType::FP32;

    float epsilon = 1.0e-5f;
    bool training = true;
    std::string debugName = "thor_layer_norm";

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

struct CudnnLayerNormForwardArgs {
    Tensor x;
    Tensor scale;
    Tensor bias;
    Tensor y;

    // Required when descriptor.training is true. FP32 tensors with outerSize elements.
    std::optional<Tensor> mean;
    std::optional<Tensor> invVariance;
};

struct CudnnLayerNormBackwardArgs {
    Tensor dy;
    Tensor x;
    Tensor scale;
    Tensor mean;
    Tensor invVariance;
    Tensor dx;
    Tensor dscale;
    Tensor dbias;
};

/**
 * One finalized LayerNorm execution plan owned by one independently executable
 * LayerNorm operation/connection.  The descriptor is retained locally so
 * execution can validate its tensor pack without consulting process-global
 * state.  The underlying Frontend executable is move-only and operation-local.
 */
class CudnnLayerNormExecutablePlan final : public AcceleratorBackendLocalExecutionStateTag {
   public:
    CudnnLayerNormExecutablePlan(const CudnnLayerNormExecutablePlan&) = delete;
    CudnnLayerNormExecutablePlan& operator=(const CudnnLayerNormExecutablePlan&) = delete;
    CudnnLayerNormExecutablePlan(CudnnLayerNormExecutablePlan&&) noexcept = default;
    CudnnLayerNormExecutablePlan& operator=(CudnnLayerNormExecutablePlan&&) noexcept = default;
    ~CudnnLayerNormExecutablePlan() = default;

    [[nodiscard]] const CudnnLayerNormDescriptor& descriptor() const noexcept { return descriptor_; }
    [[nodiscard]] const CudnnFrontendPlanSelection& selection() const noexcept { return executable_.selection(); }
    [[nodiscard]] uint64_t workspaceBytes() const noexcept { return executable_.workspaceBytes(); }
    [[nodiscard]] uintptr_t executableId() const noexcept { return executable_.executableId(); }
    [[nodiscard]] int gpuNum() const noexcept { return gpu_num_; }
    [[nodiscard]] bool isForward() const noexcept { return pass_ == Pass::Forward; }
    [[nodiscard]] bool isBackward() const noexcept { return pass_ == Pass::Backward; }

   private:
    enum class Pass { Forward, Backward };

    CudnnLayerNormExecutablePlan(CudnnLayerNormDescriptor descriptor,
                                 Pass pass,
                                 int gpuNum,
                                 CudnnFrontendExecutablePlan executable)
        : descriptor_(std::move(descriptor)), pass_(pass), gpu_num_(gpuNum), executable_(std::move(executable)) {}

    CudnnLayerNormDescriptor descriptor_;
    Pass pass_;
    int gpu_num_ = -1;
    CudnnFrontendExecutablePlan executable_;

    friend class CudnnLayerNorm;
};

class CudnnLayerNorm {
   public:
    static CudnnLayerNorm& instance();

    // Preparation is the only phase allowed to consult the process-global
    // selection cache or construct a Frontend executable graph. Each call
    // returns a new operation-local finalized executable.
    [[nodiscard]] CudnnLayerNormExecutablePlan prepareForward(const CudnnLayerNormDescriptor& descriptor, Stream stream);
    [[nodiscard]] CudnnLayerNormExecutablePlan prepareBackward(const CudnnLayerNormDescriptor& descriptor, Stream stream);

    void forward(const CudnnLayerNormExecutablePlan& plan,
                 const CudnnLayerNormForwardArgs& args,
                 std::optional<Tensor>& workspace,
                 Stream stream);
    void backward(const CudnnLayerNormExecutablePlan& plan,
                  const CudnnLayerNormBackwardArgs& args,
                  std::optional<Tensor>& workspace,
                  Stream stream);

    // These diagnostics describe immutable global selection recipes only.
    void clearSelectionCache();
    [[nodiscard]] size_t cachedSelectionCount() const;
    [[nodiscard]] uint64_t selectionCacheHitCount() const;
    [[nodiscard]] uint64_t selectionCacheMissCount() const;

    static bool frontendAvailable();

   private:
    CudnnLayerNorm() = default;
};

static_assert(AcceleratorBackendLocalExecutionState<CudnnLayerNormExecutablePlan>);
static_assert(!std::is_copy_constructible_v<CudnnLayerNormExecutablePlan>);
static_assert(!std::is_copy_assignable_v<CudnnLayerNormExecutablePlan>);
static_assert(std::is_move_constructible_v<CudnnLayerNormExecutablePlan>);
static_assert(std::is_move_assignable_v<CudnnLayerNormExecutablePlan>);

}  // namespace ThorImplementation
