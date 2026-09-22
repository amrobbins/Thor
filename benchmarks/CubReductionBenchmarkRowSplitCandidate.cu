#include "benchmarks/CubReductionBenchmarkCandidate.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionBenchmarking {
namespace {

using namespace CubReductionInternal;

constexpr uint32_t ROW_SPLIT_MAX_BLOCK_THREADS = 256;

[[nodiscard]] uint32_t chooseFirstStageThreads(uint64_t inner_size) {
    uint32_t threads = 32;
    while (threads < ROW_SPLIT_MAX_BLOCK_THREADS && static_cast<uint64_t>(threads) < inner_size) {
        threads *= 2;
    }
    return threads;
}

template <int TargetWaves>
[[nodiscard]] constexpr std::string_view candidateName() {
    static_assert(TargetWaves == 1 || TargetWaves == 2 || TargetWaves == 4 || TargetWaves == 8);
    if constexpr (TargetWaves == 1) {
        return "tiled_row_split_w1";
    } else if constexpr (TargetWaves == 2) {
        return "tiled_row_split_w2";
    } else if constexpr (TargetWaves == 4) {
        return "tiled_row_split_w4";
    } else {
        return "tiled_row_split_w8";
    }
}

[[nodiscard]] uint64_t ceilDiv(uint64_t numerator, uint64_t denominator) {
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

[[nodiscard]] bool supportsRowSplit(const TensorDescriptor& descriptor,
                                    const std::vector<uint32_t>& axes) {
    const DataType dtype = descriptor.getDataType();
    if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
        return false;
    }

    const std::vector<uint64_t>& dimensions = descriptor.getDimensions();
    if (dimensions.size() != 3 || axes.size() != 1 || axes[0] != 1) {
        return false;
    }

    const CubReductionGeometry geometry =
        CubReduction::analyzeValueGeometry(CubReductionOp::Sum, dimensions, axes);
    return geometry.path == CubReductionPath::TiledFixedSegment && geometry.outer_size != 0
           && geometry.reduction_size != 0 && geometry.inner_size != 0;
}

template <typename InputT>
__global__ void rowSplitFirstStageSumKernel(const InputT* input,
                                            float* partials,
                                            uint64_t outer_size,
                                            uint64_t reduction_size,
                                            uint64_t inner_size,
                                            uint64_t shards_per_output) {
    const uint64_t work = static_cast<uint64_t>(blockIdx.x);
    const uint64_t total_work = outer_size * shards_per_output;
    if (work >= total_work) {
        return;
    }

    const uint64_t outer = work / shards_per_output;
    const uint64_t shard = work - outer * shards_per_output;
    const uint64_t base_rows = reduction_size / shards_per_output;
    const uint64_t extra_rows = reduction_size % shards_per_output;
    const uint64_t row_begin = shard * base_rows + (shard < extra_rows ? shard : extra_rows);
    const uint64_t row_count = base_rows + static_cast<uint64_t>(shard < extra_rows);
    const uint64_t row_end = row_begin + row_count;
    const uint64_t outer_input_base = outer * reduction_size * inner_size;
    const uint64_t partial_base = work * inner_size;

    // Threads own components, not rows.  Therefore every warp issues contiguous global reads for each source row,
    // while independent CTAs split the enormous reduction dimension.  This preserves the coalescing property of the
    // production full-row reducers but exposes row parallelism when there are too few output vectors to occupy the GPU.
    for (uint64_t component = static_cast<uint64_t>(threadIdx.x); component < inner_size;
         component += static_cast<uint64_t>(blockDim.x)) {
        float sum = 0.0f;
        uint64_t input_index = outer_input_base + row_begin * inner_size + component;
        for (uint64_t row = row_begin; row < row_end; ++row) {
            sum += ToFp32<InputT>{}(input[input_index]);
            input_index += inner_size;
        }
        partials[partial_base + component] = sum;
    }
}

__global__ void rowSplitFinalizeSumKernel(const float* partials,
                                           void* output,
                                           DataType output_dtype,
                                           uint64_t outer_size,
                                           uint64_t inner_size,
                                           uint64_t shards_per_output) {
    const uint64_t output_elements = outer_size * inner_size;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);
    for (uint64_t output_index = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x)
                                     + static_cast<uint64_t>(threadIdx.x);
         output_index < output_elements;
         output_index += grid_stride) {
        const uint64_t outer = output_index / inner_size;
        const uint64_t component = output_index - outer * inner_size;
        float sum = 0.0f;
        uint64_t partial_index = outer * shards_per_output * inner_size + component;
        for (uint64_t shard = 0; shard < shards_per_output; ++shard) {
            sum += partials[partial_index];
            partial_index += inner_size;
        }
        storeFp32AsRuntimeDType(output, output_dtype, output_index, sum);
    }
}

template <int TargetWaves>
class StampedRowSplitCandidate final : public StampedReductionCandidate {
   public:
    StampedRowSplitCandidate(Tensor input,
                             std::vector<uint32_t> axes,
                             int gpu_num)
        : input_(std::move(input)),
          geometry_(CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_.getDimensions(), axes)),
          output_(input_.getPlacement(), TensorDescriptor(input_.getDataType(), geometry_.output_dimensions)) {
        if (geometry_.path != CubReductionPath::TiledFixedSegment || geometry_.rank != 3) {
            throw std::logic_error("Row-split candidate requires rank-3 TiledFixedSegment geometry.");
        }

        int multiprocessors = 0;
        const cudaError_t status = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, gpu_num);
        if (status != cudaSuccess || multiprocessors <= 0) {
            throw std::runtime_error("Row-split candidate could not query the GPU multiprocessor count.");
        }

        const uint64_t target_blocks = static_cast<uint64_t>(multiprocessors) * static_cast<uint64_t>(TargetWaves);
        const uint64_t desired_shards = std::max<uint64_t>(1, ceilDiv(target_blocks, geometry_.outer_size));
        shards_per_output_ = std::min<uint64_t>(geometry_.reduction_size, desired_shards);
        first_stage_blocks_ = geometry_.outer_size * shards_per_output_;

        block_threads_ = chooseFirstStageThreads(geometry_.inner_size);
        workspace_ = Tensor(input_.getPlacement(),
                            TensorDescriptor(DataType::FP32,
                                             {first_stage_blocks_ * geometry_.inner_size}));
    }

    void runOn(Stream& stream) const override {
        auto launch_first_stage = [&]<typename InputT>() {
            rowSplitFirstStageSumKernel<InputT>
                <<<static_cast<unsigned int>(first_stage_blocks_), block_threads_, 0, stream.getStream()>>>(
                    input_.getMemPtr<InputT>(),
                    workspace_.getMemPtr<float>(),
                    geometry_.outer_size,
                    geometry_.reduction_size,
                    geometry_.inner_size,
                    shards_per_output_);
        };

        switch (input_.getDataType()) {
            case DataType::FP16:
                launch_first_stage.template operator()<__half>();
                break;
            case DataType::BF16:
                launch_first_stage.template operator()<__nv_bfloat16>();
                break;
            case DataType::FP32:
                launch_first_stage.template operator()<float>();
                break;
            default:
                throw std::logic_error("Row-split candidate received an unsupported dtype.");
        }
        const cudaError_t first_status = cudaGetLastError();
        if (first_status != cudaSuccess) {
            throw std::runtime_error(std::string("Row-split first-stage launch failed: ")
                                     + cudaGetErrorString(first_status));
        }

        const uint64_t output_elements = geometry_.outer_size * geometry_.inner_size;
        const uint64_t finalize_blocks = ceilDiv(output_elements, ROW_SPLIT_MAX_BLOCK_THREADS);
        rowSplitFinalizeSumKernel
            <<<static_cast<unsigned int>(finalize_blocks), ROW_SPLIT_MAX_BLOCK_THREADS, 0, stream.getStream()>>>(
                workspace_.getMemPtr<float>(),
                output_.getMemPtr<void>(),
                output_.getDataType(),
                geometry_.outer_size,
                geometry_.inner_size,
                shards_per_output_);
        const cudaError_t final_status = cudaGetLastError();
        if (final_status != cudaSuccess) {
            throw std::runtime_error(std::string("Row-split finalize launch failed: ")
                                     + cudaGetErrorString(final_status));
        }
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override { return workspace_.getArraySizeInBytes(); }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "tiled_row_split_two_stage";
        metadata.strategy = candidateName<TargetWaves>();
        metadata.vector_elements_per_load = 1;
        metadata.block_threads = block_threads_;
        metadata.first_stage_blocks = first_stage_blocks_;
        metadata.shards_per_output = shards_per_output_;
        return metadata;
    }

   private:
    Tensor input_;
    CubReductionGeometry geometry_;
    mutable Tensor output_;
    mutable Tensor workspace_;
    uint64_t shards_per_output_ = 1;
    uint64_t first_stage_blocks_ = 1;
    uint32_t block_threads_ = ROW_SPLIT_MAX_BLOCK_THREADS;
};

template <int TargetWaves>
class RowSplitCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override { return candidateName<TargetWaves>(); }

    [[nodiscard]] bool supports(CubReductionOp op,
                                const TensorDescriptor& input_descriptor,
                                const std::vector<uint32_t>& axes) const override {
        return op == CubReductionOp::Sum && supportsRowSplit(input_descriptor, axes);
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream& stream) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "tiled_row_split candidate requires dense contiguous rank-3 [outer,reduction,inner] SUM geometry.");
        }
        return std::make_unique<StampedRowSplitCandidate<TargetWaves>>(input, axes, stream.getGpuNum());
    }
};

const ReductionCandidateRegistrar<RowSplitCandidate<1>> row_split_w1_registrar;
const ReductionCandidateRegistrar<RowSplitCandidate<2>> row_split_w2_registrar;
const ReductionCandidateRegistrar<RowSplitCandidate<4>> row_split_w4_registrar;
const ReductionCandidateRegistrar<RowSplitCandidate<8>> row_split_w8_registrar;

}  // namespace
}  // namespace ThorImplementation::CubReductionBenchmarking
