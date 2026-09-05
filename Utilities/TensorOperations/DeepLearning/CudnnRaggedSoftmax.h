#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/AcceleratorBackendCachePolicy.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cudnn.h>

namespace ThorImplementation {

inline constexpr RaggedPartitionRequirement kCudnnRaggedSoftmaxPartitionRequirement =
    RaggedPartitionRequirement::HOST_EXTENT;

enum class CudnnRaggedSoftmaxKind {
    Softmax,
    LogSoftmax,
};

/**
 * Exact-prefix ordinary ragged Softmax/LogSoftmax descriptor.
 *
 * Packed storage has physical shape
 *
 *   [maxTotalValues, D0, D1, ..., Dk]
 *
 * with k >= 0. Ordinary softmax normalizes only the final trailing dimension
 * Dk. At execution Thor already knows activeValueCount on the host; the
 * execution state exposes only
 *
 *   [activeValueCount * product(D0..D{k-1}), Dk, 1, 1]
 *
 * to fixed-function cuDNN. Inactive packed capacity is therefore outside the
 * cuDNN tensor descriptor and is neither read nor canonicalized.
 */
struct CudnnRaggedSoftmaxDescriptor {
    uint64_t maxTotalValues = 0;
    std::vector<uint64_t> trailingDimensions;
    DataType dataType = DataType::FP16;
    CudnnRaggedSoftmaxKind kind = CudnnRaggedSoftmaxKind::Softmax;
    std::string debugName = "thor_ragged_softmax";

    void validate() const;

    [[nodiscard]] uint64_t outerPerValue() const;
    [[nodiscard]] uint64_t channelCount() const;
    [[nodiscard]] uint64_t capacityElementCount() const;
};

struct CudnnRaggedSoftmaxForwardArgs {
    Tensor x;
    Tensor y;
    uint64_t activeValueCount = 0;
};

struct CudnnRaggedSoftmaxBackwardArgs {
    // y is the already-computed Softmax output for Softmax, or LogSoftmax
    // output for LogSoftmax, matching cudnnSoftmaxBackward's contract.
    Tensor y;
    Tensor dy;
    Tensor dx;
    uint64_t activeValueCount = 0;
};

/**
 * Operation-local fixed-function cuDNN execution state.
 *
 * Preparation allocates the cuDNN tensor descriptor. Runtime execution only
 * updates that descriptor's dimensions to the exact host-known active prefix
 * and calls cudnnSoftmaxForward/cudnnSoftmaxBackward. There is no Frontend
 * graph/plan selection, compilation, tuning, workspace allocation, or device
 * extent read in the run path.
 *
 * Like Thor's other stamped execution-local state, one instance is not
 * intended for concurrent host calls. Sequential asynchronous forward/backward
 * launches are safe because cuDNN consumes descriptor metadata during the API
 * call before returning.
 */
class CudnnRaggedSoftmaxExecutionState final : public AcceleratorBackendLocalExecutionStateTag {
   public:
    CudnnRaggedSoftmaxExecutionState(const CudnnRaggedSoftmaxExecutionState&) = delete;
    CudnnRaggedSoftmaxExecutionState& operator=(const CudnnRaggedSoftmaxExecutionState&) = delete;
    CudnnRaggedSoftmaxExecutionState(CudnnRaggedSoftmaxExecutionState&& other) noexcept;
    CudnnRaggedSoftmaxExecutionState& operator=(CudnnRaggedSoftmaxExecutionState&& other) noexcept;
    ~CudnnRaggedSoftmaxExecutionState();

    [[nodiscard]] const CudnnRaggedSoftmaxDescriptor& descriptor() const noexcept { return descriptor_; }
    [[nodiscard]] int gpuNum() const noexcept { return gpu_num_; }

   private:
    CudnnRaggedSoftmaxExecutionState(CudnnRaggedSoftmaxDescriptor descriptor, int gpuNum, cudnnTensorDescriptor_t ioDescriptor)
        : descriptor_(std::move(descriptor)), gpu_num_(gpuNum), io_descriptor_(ioDescriptor) {}

    void configureActiveValueCount(uint64_t activeValueCount);
    void release() noexcept;

    CudnnRaggedSoftmaxDescriptor descriptor_;
    int gpu_num_ = -1;
    cudnnTensorDescriptor_t io_descriptor_ = nullptr;

    friend class CudnnRaggedSoftmax;
};

class CudnnRaggedSoftmax {
   public:
    static CudnnRaggedSoftmax& instance();

    // Preparation is placement/stamping work. Runtime execution never creates
    // a descriptor or plan and never reads offsets/device extent state.
    [[nodiscard]] CudnnRaggedSoftmaxExecutionState prepare(const CudnnRaggedSoftmaxDescriptor& descriptor, Stream stream) const;

    void forward(CudnnRaggedSoftmaxExecutionState& state,
                 const CudnnRaggedSoftmaxForwardArgs& args,
                 Stream stream) const;
    void backward(CudnnRaggedSoftmaxExecutionState& state,
                  const CudnnRaggedSoftmaxBackwardArgs& args,
                  Stream stream) const;

   private:
    CudnnRaggedSoftmax() = default;
};

static_assert(AcceleratorBackendLocalExecutionState<CudnnRaggedSoftmaxExecutionState>);
static_assert(!std::is_copy_constructible_v<CudnnRaggedSoftmaxExecutionState>);
static_assert(!std::is_copy_assignable_v<CudnnRaggedSoftmaxExecutionState>);
static_assert(std::is_move_constructible_v<CudnnRaggedSoftmaxExecutionState>);
static_assert(std::is_move_assignable_v<CudnnRaggedSoftmaxExecutionState>);

}  // namespace ThorImplementation
