#include "benchmarks/CubReductionBenchmarkCandidate.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cuda/std/functional>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionBenchmarking {
namespace {

using namespace CubReductionInternal;

[[nodiscard]] bool isExactFullRowWidth(uint64_t inner_size) {
    switch (inner_size) {
        case 32:
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] bool supportsForcedDirectComponentTiled(const TensorDescriptor& descriptor,
                                                       const std::vector<uint32_t>& axes) {
    const DataType dtype = descriptor.getDataType();
    if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
        return false;
    }

    const CubReductionGeometry geometry =
        CubReduction::analyzeValueGeometry(CubReductionOp::Sum, descriptor.getDimensions(), axes);
    if (geometry.path != CubReductionPath::TiledFixedSegment || geometry.inner_size <= 32
        || geometry.inner_size >= 4096 || isExactFullRowWidth(geometry.inner_size)) {
        return false;
    }

    // This candidate asks one narrow question: are the async full-row families the source of the low-precision
    // throughput cliffs exposed by the stage-cost calibration? Force the pre-existing direct component-tiled fallback
    // over the same dense [outer,reduction,inner] geometry. Nothing here participates in production selection.
    return true;
}

class StampedForcedDirectComponentTiledCandidate final : public StampedReductionCandidate {
   public:
    StampedForcedDirectComponentTiledCandidate(Tensor input,
                                                std::vector<uint32_t> axes)
        : input_(std::move(input)),
          geometry_(CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_.getDimensions(), axes)),
          output_(input_.getPlacement(), TensorDescriptor(input_.getDataType(), geometry_.output_dimensions)) {
        if (geometry_.path != CubReductionPath::TiledFixedSegment) {
            throw std::logic_error("Forced direct component-tiled candidate requires TiledFixedSegment geometry.");
        }
    }

    void runOn(Stream& stream) const override {
        auto launch = [&]<typename InputT>() {
            launchDirectTiledFixedSegmentReductionForRowLanes<InputT,
                                                               cuda::std::plus<float>,
                                                               IdentityFp32,
                                                               AdditiveFinalizeFp32,
                                                               1>(input_.getMemPtr<InputT>(),
                                                                  output_.getMemPtr<void>(),
                                                                  output_.getDataType(),
                                                                  geometry_,
                                                                  cuda::std::plus<float>{},
                                                                  0.0f,
                                                                  IdentityFp32{},
                                                                  AdditiveFinalizeFp32{1.0f},
                                                                  1.0f,
                                                                  stream.getStream());
        };
        switch (input_.getDataType()) {
            case DataType::FP16:
                launch.template operator()<__half>();
                break;
            case DataType::BF16:
                launch.template operator()<__nv_bfloat16>();
                break;
            case DataType::FP32:
                launch.template operator()<float>();
                break;
            default:
                throw std::logic_error("Forced direct component-tiled candidate received an unsupported dtype.");
        }
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override { return 0; }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "tiled_direct_component";
        metadata.strategy = "forced_direct_component_tiled_row_lanes1";
        metadata.block_threads = static_cast<uint32_t>(TILED_REDUCTION_BLOCK_THREADS);

        constexpr uint64_t components_per_warp = TILED_REDUCTION_WARP_THREADS;
        const uint64_t component_tiles = ceilDivideU64(geometry_.inner_size, components_per_warp);
        const int warps_per_tile = chooseDirectTiledReductionWarpsPerTile<1>(geometry_);
        const int groups_per_block = TILED_REDUCTION_WARPS_PER_BLOCK / warps_per_tile;
        const uint64_t total_work = geometry_.outer_size * component_tiles;
        metadata.first_stage_blocks =
            ceilDivideU64(total_work, static_cast<uint64_t>(groups_per_block));
        metadata.shards_per_output = component_tiles;
        return metadata;
    }

   private:
    Tensor input_;
    CubReductionGeometry geometry_;
    mutable Tensor output_;
};

class ForcedDirectComponentTiledCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override { return "tiled_direct_component_fallback"; }

    [[nodiscard]] bool supports(CubReductionOp op,
                                const TensorDescriptor& input_descriptor,
                                const std::vector<uint32_t>& axes) const override {
        return op == CubReductionOp::Sum && supportsForcedDirectComponentTiled(input_descriptor, axes);
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream&) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "tiled_direct_component_fallback requires supported dense contiguous TiledFixedSegment SUM geometry.");
        }
        return std::make_unique<StampedForcedDirectComponentTiledCandidate>(input, axes);
    }
};

const ReductionCandidateRegistrar<ForcedDirectComponentTiledCandidate> forced_direct_component_tiled_registrar;

}  // namespace
}  // namespace ThorImplementation::CubReductionBenchmarking
