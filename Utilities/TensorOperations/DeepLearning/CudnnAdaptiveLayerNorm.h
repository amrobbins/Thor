#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/CudnnFrontendPlan.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace ThorImplementation {

/**
 * cuDNN Frontend Adaptive LayerNorm wrapper.
 *
 * cuDNN's Adaptive LayerNorm shape contract is different from ordinary LayerNorm:
 * scale and bias vary per batch sample, but are broadcast across the non-normalized
 * leading feature dimensions. Thor exposes this as:
 *
 *   data feature shape:       [leading..., normalized...]
 *   scale/bias feature shape: [normalized...]
 *
 * Once the network batch dimension is included, the frontend graph is built with
 * NVIDIA's canonical 3D view:
 *
 *   x/y:        [batch, leading, hidden]
 *   scale/bias: [batch, 1, hidden]
 *   stats:      [batch, leading, 1]
 *
 * where hidden is the product of the normalized trailing dimensions and leading is
 * the product of the non-normalized feature dimensions. No tensor materialization is
 * required for this view.
 */
struct CudnnAdaptiveLayerNormDescriptor {
    uint64_t batchSize = 0;
    uint64_t leadingFeatureCount = 0;
    uint64_t normalizedFeatureCount = 0;

    DataType inputDataType = DataType::FP16;
    DataType outputDataType = DataType::FP16;
    DataType scaleBiasDataType = DataType::FP32;
    DataType computeDataType = DataType::FP32;

    float epsilon = 1.0e-5f;
    bool training = true;
    std::string debugName = "thor_adaptive_layer_norm";

    void validateForward() const;
    void validateBackward() const;
    std::string cacheKey(std::string_view passName, int gpuNum) const;
};

struct CudnnAdaptiveLayerNormForwardArgs {
    Tensor x;
    Tensor scale;
    Tensor bias;
    Tensor y;

    // Required when descriptor.training is true. FP32 tensors with batchSize * leadingFeatureCount elements.
    std::optional<Tensor> mean;
    std::optional<Tensor> invVariance;
};

struct CudnnAdaptiveLayerNormBackwardArgs {
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
 * One finalized AdaptiveLayerNorm execution plan owned by one independently
 * executable AdaptiveLayerNorm layer/application. The descriptor is retained
 * locally so runtime execution never consults process-global selection state.
 */
class CudnnAdaptiveLayerNormExecutablePlan final : public AcceleratorBackendLocalExecutionStateTag {
   public:
    CudnnAdaptiveLayerNormExecutablePlan(const CudnnAdaptiveLayerNormExecutablePlan&) = delete;
    CudnnAdaptiveLayerNormExecutablePlan& operator=(const CudnnAdaptiveLayerNormExecutablePlan&) = delete;
    CudnnAdaptiveLayerNormExecutablePlan(CudnnAdaptiveLayerNormExecutablePlan&&) noexcept = default;
    CudnnAdaptiveLayerNormExecutablePlan& operator=(CudnnAdaptiveLayerNormExecutablePlan&&) noexcept = default;
    ~CudnnAdaptiveLayerNormExecutablePlan() = default;

    [[nodiscard]] const CudnnAdaptiveLayerNormDescriptor& descriptor() const noexcept { return descriptor_; }
    [[nodiscard]] const CudnnFrontendPlanSelection& selection() const noexcept { return executable_.selection(); }
    [[nodiscard]] uint64_t workspaceBytes() const noexcept { return executable_.workspaceBytes(); }
    [[nodiscard]] uintptr_t executableId() const noexcept { return executable_.executableId(); }
    [[nodiscard]] int gpuNum() const noexcept { return gpu_num_; }
    [[nodiscard]] bool isForward() const noexcept { return pass_ == Pass::Forward; }
    [[nodiscard]] bool isBackward() const noexcept { return pass_ == Pass::Backward; }

   private:
    enum class Pass { Forward, Backward };

    CudnnAdaptiveLayerNormExecutablePlan(CudnnAdaptiveLayerNormDescriptor descriptor,
                                         Pass pass,
                                         int gpuNum,
                                         CudnnFrontendExecutablePlan executable)
        : descriptor_(std::move(descriptor)), pass_(pass), gpu_num_(gpuNum), executable_(std::move(executable)) {}

    CudnnAdaptiveLayerNormDescriptor descriptor_;
    Pass pass_;
    int gpu_num_ = -1;
    CudnnFrontendExecutablePlan executable_;

    friend class CudnnAdaptiveLayerNorm;
};

class CudnnAdaptiveLayerNorm {
   public:
    static CudnnAdaptiveLayerNorm& instance();

    // Preparation is the only phase allowed to consult the process-global
    // selection cache or construct/replay a Frontend executable graph. Each
    // call returns a fresh operation-local executable.
    [[nodiscard]] CudnnAdaptiveLayerNormExecutablePlan prepareForward(const CudnnAdaptiveLayerNormDescriptor& descriptor, Stream stream);
    [[nodiscard]] CudnnAdaptiveLayerNormExecutablePlan prepareBackward(const CudnnAdaptiveLayerNormDescriptor& descriptor, Stream stream);

    void forward(const CudnnAdaptiveLayerNormExecutablePlan& plan,
                 const CudnnAdaptiveLayerNormForwardArgs& args,
                 std::optional<Tensor>& workspace,
                 Stream stream);
    void backward(const CudnnAdaptiveLayerNormExecutablePlan& plan,
                  const CudnnAdaptiveLayerNormBackwardArgs& args,
                  std::optional<Tensor>& workspace,
                  Stream stream);

    // Diagnostics describe immutable global selection recipes only.
    void clearSelectionCache();
    [[nodiscard]] size_t cachedSelectionCount() const;
    [[nodiscard]] uint64_t selectionCacheHitCount() const;
    [[nodiscard]] uint64_t selectionCacheMissCount() const;

    static bool frontendAvailable();

   private:
    CudnnAdaptiveLayerNorm() = default;
};

static_assert(AcceleratorBackendLocalExecutionState<CudnnAdaptiveLayerNormExecutablePlan>);
static_assert(!std::is_copy_constructible_v<CudnnAdaptiveLayerNormExecutablePlan>);
static_assert(!std::is_copy_assignable_v<CudnnAdaptiveLayerNormExecutablePlan>);
static_assert(std::is_move_constructible_v<CudnnAdaptiveLayerNormExecutablePlan>);
static_assert(std::is_move_assignable_v<CudnnAdaptiveLayerNormExecutablePlan>);

}  // namespace ThorImplementation
