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
 * cuDNN Frontend InstanceNorm wrapper.
 *
 * Thor API tensors do not include the batch dimension.  Physical tensors do.  This wrapper treats any packed physical
 * tensor with dimensions [N, C, spatial...] as NVIDIA's canonical 4D instance-normalization view:
 *
 *   x/y:         [N, C, S, 1]
 *   scale/bias:  [1, C, 1, 1]
 *   stats:       [N, C, 1, 1]
 *
 * where S is the product of all spatial dimensions.  This is a metadata-only view over Thor's contiguous tensor storage.
 */
struct CudnnInstanceNormDescriptor {
    uint64_t batchSize = 0;
    uint64_t channelCount = 0;
    uint64_t spatialElementCount = 0;

    DataType inputDataType = DataType::FP16;
    DataType outputDataType = DataType::FP16;
    DataType parameterDataType = DataType::FP32;
    DataType computeDataType = DataType::FP32;

    float epsilon = 1.0e-5f;
    bool training = true;
    std::string debugName = "thor_instance_norm";

    void validateForward() const;
    void validateBackward() const;
    std::string cacheKey(std::string_view passName, int gpuNum) const;
};

struct CudnnInstanceNormForwardArgs {
    Tensor x;
    Tensor scale;
    Tensor bias;
    Tensor y;

    // Required when descriptor.training is true.  FP32 tensors with batchSize * channelCount elements.
    std::optional<Tensor> mean;
    std::optional<Tensor> invVariance;
};

struct CudnnInstanceNormBackwardArgs {
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
 * One finalized InstanceNorm execution plan owned by one independently
 * executable InstanceNorm connection. The descriptor is retained locally so
 * execution validates its tensor pack without consulting process-global state.
 */
class CudnnInstanceNormExecutablePlan final : public AcceleratorBackendLocalExecutionStateTag {
   public:
    CudnnInstanceNormExecutablePlan(const CudnnInstanceNormExecutablePlan&) = delete;
    CudnnInstanceNormExecutablePlan& operator=(const CudnnInstanceNormExecutablePlan&) = delete;
    CudnnInstanceNormExecutablePlan(CudnnInstanceNormExecutablePlan&&) noexcept = default;
    CudnnInstanceNormExecutablePlan& operator=(CudnnInstanceNormExecutablePlan&&) noexcept = default;
    ~CudnnInstanceNormExecutablePlan() = default;

    [[nodiscard]] const CudnnInstanceNormDescriptor& descriptor() const noexcept { return descriptor_; }
    [[nodiscard]] const CudnnFrontendPlanSelection& selection() const noexcept { return executable_.selection(); }
    [[nodiscard]] uint64_t workspaceBytes() const noexcept { return executable_.workspaceBytes(); }
    [[nodiscard]] uintptr_t executableId() const noexcept { return executable_.executableId(); }
    [[nodiscard]] int gpuNum() const noexcept { return gpu_num_; }
    [[nodiscard]] bool isForward() const noexcept { return pass_ == Pass::Forward; }
    [[nodiscard]] bool isBackward() const noexcept { return pass_ == Pass::Backward; }

   private:
    enum class Pass { Forward, Backward };

    CudnnInstanceNormExecutablePlan(CudnnInstanceNormDescriptor descriptor,
                                    Pass pass,
                                    int gpuNum,
                                    CudnnFrontendExecutablePlan executable)
        : descriptor_(std::move(descriptor)), pass_(pass), gpu_num_(gpuNum), executable_(std::move(executable)) {}

    CudnnInstanceNormDescriptor descriptor_;
    Pass pass_;
    int gpu_num_ = -1;
    CudnnFrontendExecutablePlan executable_;

    friend class CudnnInstanceNorm;
};

class CudnnInstanceNorm {
   public:
    static CudnnInstanceNorm& instance();

    // Preparation is the only phase allowed to consult the process-global
    // selection cache or construct a Frontend executable graph. Each call
    // returns a fresh operation-local executable.
    [[nodiscard]] CudnnInstanceNormExecutablePlan prepareForward(const CudnnInstanceNormDescriptor& descriptor, Stream stream);
    [[nodiscard]] CudnnInstanceNormExecutablePlan prepareBackward(const CudnnInstanceNormDescriptor& descriptor, Stream stream);

    void forward(const CudnnInstanceNormExecutablePlan& plan,
                 const CudnnInstanceNormForwardArgs& args,
                 std::optional<Tensor>& workspace,
                 Stream stream);
    void backward(const CudnnInstanceNormExecutablePlan& plan,
                  const CudnnInstanceNormBackwardArgs& args,
                  std::optional<Tensor>& workspace,
                  Stream stream);

    // Diagnostics describe immutable global selection recipes only.
    void clearSelectionCache();
    [[nodiscard]] size_t cachedSelectionCount() const;
    [[nodiscard]] uint64_t selectionCacheHitCount() const;
    [[nodiscard]] uint64_t selectionCacheMissCount() const;

    static bool frontendAvailable();

   private:
    CudnnInstanceNorm() = default;
};

static_assert(AcceleratorBackendLocalExecutionState<CudnnInstanceNormExecutablePlan>);
static_assert(!std::is_copy_constructible_v<CudnnInstanceNormExecutablePlan>);
static_assert(!std::is_copy_assignable_v<CudnnInstanceNormExecutablePlan>);
static_assert(std::is_move_constructible_v<CudnnInstanceNormExecutablePlan>);
static_assert(std::is_move_assignable_v<CudnnInstanceNormExecutablePlan>);

}  // namespace ThorImplementation
