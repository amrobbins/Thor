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
        || geometry.inner_size >= 4096
        || (isExactFullRowWidth(geometry.inner_size) && geometry.inner_size != 256)) {
        return false;
    }

    // This candidate asks one narrow question: are the async/full-row families the source of the low-precision
    // throughput cliffs exposed by the stage-cost calibration? Force the pre-existing direct component-tiled fallback
    // over the same dense [outer,reduction,inner] geometry. Exact width 256 is intentionally admitted as a diagnostic
    // for tiled_middle_large [256,512,256]: RowLanes=1 splits each 256-component retained row into eight independent
    // 32-component tiles, exposing 8x more work warps than the production full-row x8 ownership. Nothing here
    // participates in production selection.
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

// Benchmark-only geometry experiment for tiled_middle_large [256,512,256]. One cooperative block owns one
// (outer, retained-component-tile) result tile. x is always one physical warp spanning contiguous retained components;
// y is represented by WarpsPerOutput physical warps that split the reduction rows. Every thread issues one aligned
// 16-byte packet load per visited reduction row. The cooperating warps combine their FP32 packet accumulators through
// shared memory only -- deliberately no warp-shuffle reduction. Blocks grow beyond the normal 256-thread ceiling only
// when the additional warps cooperate on this same output tile.
template <typename InputT, int WarpsPerOutput, int ItemsPerLane>
__global__ void cooperativeVectorTiledMiddleReductionKernel(const InputT* input,
                                                             void* output,
                                                             DataType output_dtype,
                                                             uint64_t outer_size,
                                                             uint64_t reduction_size,
                                                             uint64_t inner_size,
                                                             uint64_t output_outer_stride,
                                                             uint64_t output_inner_stride) {
    static_assert(WarpsPerOutput == 4 || WarpsPerOutput == 8 || WarpsPerOutput == 16 || WarpsPerOutput == 32);
    static_assert(sizeof(InputT) * ItemsPerLane == 16, "candidate requires one 16-byte packet per thread");
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);

    // [reduction warp][packet item][component lane]. Keeping lane as the unit-stride dimension makes every shared
    // store/load issued by a physical warp bank-friendly while each thread owns the same packet items throughout.
    __shared__ float partials[WarpsPerOutput][ItemsPerLane][TILED_REDUCTION_WARP_THREADS];

    const int warp_in_output = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t component_tiles = ceilDivideU64(inner_size, components_per_tile);
    const uint64_t total_work = outer_size * component_tiles;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t outer_index = work_index / component_tiles;
        const uint64_t component_tile = work_index - outer_index * component_tiles;
        const uint64_t packet_component_begin =
            component_tile * components_per_tile
            + static_cast<uint64_t>(lane) * static_cast<uint64_t>(ItemsPerLane);

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = 0.0f;
        }

        // This first experiment intentionally supports only exact packet tiles, so every participating thread can issue
        // the same aligned 16-byte vector transaction without tails or predicates in the hot loop.
        if (packet_component_begin + static_cast<uint64_t>(ItemsPerLane) <= inner_size) {
            uint64_t row_base =
                (outer_index * reduction_size + static_cast<uint64_t>(warp_in_output)) * inner_size
                + packet_component_begin;
            const uint64_t row_stride = static_cast<uint64_t>(WarpsPerOutput) * inner_size;
            for (uint64_t row = static_cast<uint64_t>(warp_in_output); row < reduction_size;
                 row += static_cast<uint64_t>(WarpsPerOutput)) {
                const PackedInputValues<InputT, ItemsPerLane> values =
                    loadVectorizedInputPacket<InputT, ItemsPerLane>(input + row_base);
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item) {
                    local[item] += ToFp32<InputT>{}(values.values[item]);
                }
                row_base += row_stride;
            }
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            partials[warp_in_output][item][lane] = local[item];
        }
        __syncthreads();

        // Block-wide shared-memory tree. Additional warps are useful here: every one reduces the number of source rows
        // handled serially, then participates in combining the same output tile. No warp shuffle is used.
#pragma unroll
        for (int stride = WarpsPerOutput / 2; stride > 0; stride >>= 1) {
            if (warp_in_output < stride) {
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item) {
                    partials[warp_in_output][item][lane] += partials[warp_in_output + stride][item][lane];
                }
            }
            __syncthreads();
        }

        if (warp_in_output == 0) {
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                const uint64_t component = packet_component_begin + static_cast<uint64_t>(item);
                if (component < inner_size) {
                    storeFp32AsRuntimeDType(output,
                                            output_dtype,
                                            tiledReductionOutputIndex(outer_index,
                                                                      component,
                                                                      output_outer_stride,
                                                                      output_inner_stride),
                                            partials[0][item][lane]);
                }
            }
        }
        __syncthreads();
    }
}

template <int WarpsPerOutput>
[[nodiscard]] std::string_view cooperativeTiledMiddleCandidateName() {
    if constexpr (WarpsPerOutput == 4) {
        return "tiled_middle_cooperative_y4";
    } else if constexpr (WarpsPerOutput == 8) {
        return "tiled_middle_cooperative_y8";
    } else if constexpr (WarpsPerOutput == 16) {
        return "tiled_middle_cooperative_y16";
    } else {
        static_assert(WarpsPerOutput == 32);
        return "tiled_middle_cooperative_y32";
    }
}

template <int WarpsPerOutput>
class StampedCooperativeTiledMiddleCandidate final : public StampedReductionCandidate {
   public:
    StampedCooperativeTiledMiddleCandidate(Tensor input, std::vector<uint32_t> axes)
        : input_(std::move(input)),
          geometry_(CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_.getDimensions(), axes)),
          output_(input_.getPlacement(), TensorDescriptor(input_.getDataType(), geometry_.output_dimensions)) {}

    void runOn(Stream& stream) const override {
        constexpr int block_threads = WarpsPerOutput * TILED_REDUCTION_WARP_THREADS;
        const auto launch = [&]<typename InputT, int ItemsPerLane>() {
            constexpr uint64_t components_per_tile =
                static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);
            const uint64_t component_tiles = ceilDivideU64(geometry_.inner_size, components_per_tile);
            const uint64_t total_work = geometry_.outer_size * component_tiles;
            const unsigned int grid_blocks = static_cast<unsigned int>(
                std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));
            cooperativeVectorTiledMiddleReductionKernel<InputT, WarpsPerOutput, ItemsPerLane>
                <<<grid_blocks, block_threads, 0, stream.getStream()>>>(input_.getMemPtr<InputT>(),
                                                                        output_.getMemPtr<void>(),
                                                                        output_.getDataType(),
                                                                        geometry_.outer_size,
                                                                        geometry_.reduction_size,
                                                                        geometry_.inner_size,
                                                                        geometry_.tiled_output_outer_stride,
                                                                        geometry_.tiled_output_inner_stride);
            CUDA_CHECK(cudaGetLastError());
        };

        switch (input_.getDataType()) {
            case DataType::FP16:
                launch.template operator()<__half, 8>();
                break;
            case DataType::BF16:
                launch.template operator()<__nv_bfloat16, 8>();
                break;
            case DataType::FP32:
                launch.template operator()<float, 4>();
                break;
            default:
                throw std::logic_error("Cooperative tiled-middle candidate received an unsupported dtype.");
        }
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override { return 0; }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "tiled_middle_cooperative_shared";
        metadata.strategy = cooperativeTiledMiddleCandidateName<WarpsPerOutput>();
        metadata.vector_elements_per_load = input_.getDataType() == DataType::FP32 ? 4u : 8u;
        metadata.block_threads = static_cast<uint32_t>(WarpsPerOutput * TILED_REDUCTION_WARP_THREADS);
        const uint64_t components_per_tile =
            static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS)
            * static_cast<uint64_t>(input_.getDataType() == DataType::FP32 ? 4 : 8);
        const uint64_t component_tiles = ceilDivideU64(geometry_.inner_size, components_per_tile);
        metadata.first_stage_blocks = geometry_.outer_size * component_tiles;
        metadata.shards_per_output = component_tiles;
        return metadata;
    }

   private:
    Tensor input_;
    CubReductionGeometry geometry_;
    mutable Tensor output_;
};

template <int WarpsPerOutput>
class CooperativeTiledMiddleCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override {
        return cooperativeTiledMiddleCandidateName<WarpsPerOutput>();
    }

    [[nodiscard]] bool supports(CubReductionOp op,
                                const TensorDescriptor& input_descriptor,
                                const std::vector<uint32_t>& axes) const override {
        if (op != CubReductionOp::Sum) {
            return false;
        }
        const DataType dtype = input_descriptor.getDataType();
        if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
            return false;
        }

        // Benchmark the cooperative geometry across the full exact-K=256 TiledFixedSegment regime. K=256 is an
        // exact 16-byte packet tile for both low precision (8 values/lane, one component tile/output) and FP32
        // (4 values/lane, two component tiles/output), so this sweep changes only reduction/output parallelism and
        // does not mix in packet tails or awkward-row handling. Production selection remains unchanged.
        const CubReductionGeometry geometry =
            CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_descriptor.getDimensions(), axes);
        return geometry.path == CubReductionPath::TiledFixedSegment && geometry.inner_size == 256;
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream&) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "cooperative tiled-middle candidate requires dense contiguous exact-K=256 TiledFixedSegment SUM geometry.");
        }
        return std::make_unique<StampedCooperativeTiledMiddleCandidate<WarpsPerOutput>>(input, axes);
    }
};


// General adaptive cooperative tiled-value candidate. This exercises the production-ready primitive in
// CubReductionOperation.cuh without changing production selection. Geometry is derived from the actual number of
// retained-component tiles and reduction rows; no retained width maps to a hard-coded block shape. Input packets whose
// start lies inside the row use Thor's 128-byte allocation padding to eliminate scalar read-tail handling.
class StampedAdaptiveCooperativeTiledValueCandidate final : public StampedReductionCandidate {
   public:
    StampedAdaptiveCooperativeTiledValueCandidate(Tensor input, std::vector<uint32_t> axes)
        : input_(std::move(input)),
          geometry_(CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_.getDimensions(), axes)),
          output_(input_.getPlacement(), TensorDescriptor(input_.getDataType(), geometry_.output_dimensions)) {
        if (geometry_.path != CubReductionPath::TiledFixedSegment) {
            throw std::logic_error("Adaptive cooperative tiled-value candidate requires TiledFixedSegment geometry.");
        }
    }

    void runOn(Stream& stream) const override {
        const auto launch = [&]<typename InputT>() {
            launchAdaptiveCooperativeTiledFixedSegmentReduction<InputT,
                                                                cuda::std::plus<float>,
                                                                IdentityFp32,
                                                                AdditiveFinalizeFp32>(input_.getMemPtr<InputT>(),
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
                throw std::logic_error("Adaptive cooperative tiled-value candidate received an unsupported dtype.");
        }
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override { return 0; }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "tiled_cooperative_adaptive";
        metadata.strategy = "adaptive_output_parallelism_16b";
        metadata.vector_elements_per_load = input_.getDataType() == DataType::FP32 ? 4u : 8u;

        uint32_t warps = 1;
        uint64_t components_per_tile = 1;
        if (input_.getDataType() == DataType::FP32) {
            warps = static_cast<uint32_t>(chooseAdaptiveCooperativeTiledWarpsPerTile<float, 16>(geometry_));
            components_per_tile = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * 4ull;
        } else if (input_.getDataType() == DataType::FP16) {
            warps = static_cast<uint32_t>(chooseAdaptiveCooperativeTiledWarpsPerTile<__half, 16>(geometry_));
            components_per_tile = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * 8ull;
        } else {
            warps = static_cast<uint32_t>(chooseAdaptiveCooperativeTiledWarpsPerTile<__nv_bfloat16, 16>(geometry_));
            components_per_tile = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * 8ull;
        }
        const uint64_t component_tiles = ceilDivideU64(geometry_.inner_size, components_per_tile);
        const uint32_t block_warps =
            warps < static_cast<uint32_t>(TILED_REDUCTION_WARPS_PER_BLOCK)
                ? static_cast<uint32_t>(TILED_REDUCTION_WARPS_PER_BLOCK)
                : warps;
        const uint32_t groups_per_block = block_warps / warps;
        metadata.block_threads = block_warps * static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS);
        metadata.first_stage_blocks = ceilDivideU64(
            geometry_.outer_size * component_tiles, static_cast<uint64_t>(groups_per_block));
        metadata.shards_per_output = component_tiles;
        return metadata;
    }

   private:
    Tensor input_;
    CubReductionGeometry geometry_;
    mutable Tensor output_;
};

class AdaptiveCooperativeTiledValueCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override { return "tiled_middle_cooperative_adaptive"; }

    [[nodiscard]] bool supports(CubReductionOp op,
                                const TensorDescriptor& input_descriptor,
                                const std::vector<uint32_t>& axes) const override {
        if (op != CubReductionOp::Sum) {
            return false;
        }
        const DataType dtype = input_descriptor.getDataType();
        if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
            return false;
        }
        const CubReductionGeometry geometry =
            CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_descriptor.getDimensions(), axes);
        return geometry.path == CubReductionPath::TiledFixedSegment && geometry.outer_size != 0
               && geometry.reduction_size != 0 && geometry.inner_size != 0;
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream&) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "adaptive cooperative tiled-value candidate requires dense contiguous TiledFixedSegment SUM geometry.");
        }
        return std::make_unique<StampedAdaptiveCooperativeTiledValueCandidate>(input, axes);
    }
};


// Packet-adaptive variant of the cooperative tiled-value primitive. This keeps the proven full-warp ownership and
// warp-count policy, preferring 16-byte packets but allowing 8- or 4-byte packets when the retained width would
// otherwise leave at least half of a warp's packet capacity idle. Narrow packets are used only for naturally aligned
// row strides; awkward rows deliberately remain on the 16-byte alignment-safe path. Production selection is unchanged.
class StampedPacketAdaptiveCooperativeTiledValueCandidate final : public StampedReductionCandidate {
   public:
    StampedPacketAdaptiveCooperativeTiledValueCandidate(Tensor input, std::vector<uint32_t> axes)
        : input_(std::move(input)),
          geometry_(CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_.getDimensions(), axes)),
          output_(input_.getPlacement(), TensorDescriptor(input_.getDataType(), geometry_.output_dimensions)) {
        if (geometry_.path != CubReductionPath::TiledFixedSegment) {
            throw std::logic_error("Packet-adaptive cooperative tiled-value candidate requires TiledFixedSegment geometry.");
        }
    }

    void runOn(Stream& stream) const override {
        const auto launch = [&]<typename InputT>() {
            launchPacketAdaptiveCooperativeTiledFixedSegmentReduction<InputT,
                                                                      cuda::std::plus<float>,
                                                                      IdentityFp32,
                                                                      AdditiveFinalizeFp32>(input_.getMemPtr<InputT>(),
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
                throw std::logic_error("Packet-adaptive cooperative tiled-value candidate received an unsupported dtype.");
        }
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override { return 0; }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "tiled_cooperative_adaptive_packet";
        metadata.strategy = "adaptive_output_parallelism_packet_16_8_4";

        const auto fill = [&]<typename InputT>() {
            const size_t packet_bytes = chooseAdaptiveCooperativeTiledPacketBytes<InputT>(geometry_);
            uint32_t warps = 1;
            uint64_t components_per_tile = 1;
            switch (packet_bytes) {
                case 16:
                    warps = static_cast<uint32_t>(chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, 16>(geometry_));
                    components_per_tile = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS)
                                          * static_cast<uint64_t>(adaptiveCooperativeTiledItemsPerLane<InputT, 16>());
                    break;
                case 8:
                    warps = static_cast<uint32_t>(chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, 8>(geometry_));
                    components_per_tile = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS)
                                          * static_cast<uint64_t>(adaptiveCooperativeTiledItemsPerLane<InputT, 8>());
                    break;
                case 4:
                    warps = static_cast<uint32_t>(chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, 4>(geometry_));
                    components_per_tile = static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS)
                                          * static_cast<uint64_t>(adaptiveCooperativeTiledItemsPerLane<InputT, 4>());
                    break;
                default:
                    throw std::logic_error("Packet-adaptive cooperative candidate selected an invalid packet width.");
            }

            metadata.vector_elements_per_load = static_cast<uint32_t>(packet_bytes / sizeof(InputT));
            const uint64_t component_tiles = ceilDivideU64(geometry_.inner_size, components_per_tile);
            const uint32_t block_warps =
                warps < static_cast<uint32_t>(TILED_REDUCTION_WARPS_PER_BLOCK)
                    ? static_cast<uint32_t>(TILED_REDUCTION_WARPS_PER_BLOCK)
                    : warps;
            const uint32_t groups_per_block = block_warps / warps;
            metadata.block_threads = block_warps * static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS);
            metadata.first_stage_blocks = ceilDivideU64(
                geometry_.outer_size * component_tiles, static_cast<uint64_t>(groups_per_block));
            metadata.shards_per_output = component_tiles;
        };

        switch (input_.getDataType()) {
            case DataType::FP16:
                fill.template operator()<__half>();
                break;
            case DataType::BF16:
                fill.template operator()<__nv_bfloat16>();
                break;
            case DataType::FP32:
                fill.template operator()<float>();
                break;
            default:
                throw std::logic_error("Packet-adaptive cooperative tiled-value candidate received an unsupported dtype.");
        }
        return metadata;
    }

   private:
    Tensor input_;
    CubReductionGeometry geometry_;
    mutable Tensor output_;
};

class PacketAdaptiveCooperativeTiledValueCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override {
        return "tiled_middle_cooperative_adaptive_packet";
    }

    [[nodiscard]] bool supports(CubReductionOp op,
                                const TensorDescriptor& input_descriptor,
                                const std::vector<uint32_t>& axes) const override {
        if (op != CubReductionOp::Sum) {
            return false;
        }
        const DataType dtype = input_descriptor.getDataType();
        if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
            return false;
        }
        const CubReductionGeometry geometry =
            CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_descriptor.getDimensions(), axes);
        return geometry.path == CubReductionPath::TiledFixedSegment && geometry.outer_size != 0
               && geometry.reduction_size != 0 && geometry.inner_size != 0;
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream&) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "packet-adaptive cooperative tiled-value candidate requires dense contiguous TiledFixedSegment SUM geometry.");
        }
        return std::make_unique<StampedPacketAdaptiveCooperativeTiledValueCandidate>(input, axes);
    }
};

// Two-stage benchmark-only geometry for low-output/deep tiled reductions. The first stage keeps the proven
// cooperative y8 packet geometry (256 threads, 16-byte loads) but shards the reduction axis across independent CTAs.
// This exposes device-wide parallelism without growing unrelated blocks beyond 256 threads. Each CTA reduces one
// contiguous row shard for one retained-component tile and writes one FP32 partial vector. Those partials are then
// presented as an ordinary dense [outer, shards, inner] tensor and reduced by CubReduction itself. Stage two therefore
// has no candidate-specific kernel or hard-coded geometry: whatever ordinary reducer Thor selects for that shape is
// exactly what executes. The first stage uses shared memory only for inter-warp cooperation; no warp-shuffle reduction
// is used.
constexpr int SHARDED_TILED_MIDDLE_WARPS = 8;
constexpr int SHARDED_TILED_MIDDLE_BLOCK_THREADS =
    SHARDED_TILED_MIDDLE_WARPS * TILED_REDUCTION_WARP_THREADS;

template <int ItemsPerLane>
__device__ inline void storeFp32PacketVectorized(float* destination, const float (&values)[ItemsPerLane]) {
    static_assert(ItemsPerLane == 4 || ItemsPerLane == 8);
    static_assert((ItemsPerLane * sizeof(float)) % 16 == 0);
    PackedInputValues<float, ItemsPerLane> packed{};
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        packed.values[item] = values[item];
    }
    using RawT = RawVectorPacket<sizeof(float) * ItemsPerLane>;
    *reinterpret_cast<RawT*>(destination) = cuda::std::bit_cast<RawT>(packed);
}

template <typename InputT, int ItemsPerLane, int RowsPerShard>
__global__ void shardedCooperativeTiledMiddleFirstStageKernel(const InputT* input,
                                                               float* workspace,
                                                               uint64_t outer_size,
                                                               uint64_t reduction_size,
                                                               uint64_t inner_size,
                                                               uint64_t shards_per_output) {
    static_assert(RowsPerShard == 128 || RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024);
    static_assert(sizeof(InputT) * ItemsPerLane == 16, "first stage requires one 16-byte input packet per thread");
    constexpr uint64_t components_per_tile =
        static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);

    __shared__ float partials[SHARDED_TILED_MIDDLE_WARPS][ItemsPerLane][TILED_REDUCTION_WARP_THREADS];

    const int warp_in_output = static_cast<int>(threadIdx.x) / TILED_REDUCTION_WARP_THREADS;
    const int lane = static_cast<int>(threadIdx.x) % TILED_REDUCTION_WARP_THREADS;
    const uint64_t component_tiles = ceilDivideU64(inner_size, components_per_tile);
    const uint64_t total_work = outer_size * component_tiles * shards_per_output;
    const uint64_t grid_stride = static_cast<uint64_t>(gridDim.x);

    for (uint64_t work_index = static_cast<uint64_t>(blockIdx.x); work_index < total_work; work_index += grid_stride) {
        const uint64_t shard = work_index % shards_per_output;
        const uint64_t outer_component_work = work_index / shards_per_output;
        const uint64_t component_tile = outer_component_work % component_tiles;
        const uint64_t outer_index = outer_component_work / component_tiles;
        const uint64_t packet_component_begin =
            component_tile * components_per_tile
            + static_cast<uint64_t>(lane) * static_cast<uint64_t>(ItemsPerLane);
        const uint64_t row_begin = shard * static_cast<uint64_t>(RowsPerShard);
        const uint64_t shard_limit = row_begin + static_cast<uint64_t>(RowsPerShard);
        const uint64_t row_end = shard_limit < reduction_size ? shard_limit : reduction_size;

        float local[ItemsPerLane];
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] = 0.0f;
        }

        if (packet_component_begin + static_cast<uint64_t>(ItemsPerLane) <= inner_size) {
            const uint64_t first_row = row_begin + static_cast<uint64_t>(warp_in_output);
            if (first_row < row_end) {
                uint64_t input_index =
                    (outer_index * reduction_size + first_row) * inner_size + packet_component_begin;
                const uint64_t row_stride = static_cast<uint64_t>(SHARDED_TILED_MIDDLE_WARPS) * inner_size;
                for (uint64_t row = first_row; row < row_end;
                     row += static_cast<uint64_t>(SHARDED_TILED_MIDDLE_WARPS)) {
                    const PackedInputValues<InputT, ItemsPerLane> values =
                        loadVectorizedInputPacket<InputT, ItemsPerLane>(input + input_index);
#pragma unroll
                    for (int item = 0; item < ItemsPerLane; ++item) {
                        local[item] += ToFp32<InputT>{}(values.values[item]);
                    }
                    input_index += row_stride;
                }
            }
        }

#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            partials[warp_in_output][item][lane] = local[item];
        }
        __syncthreads();

#pragma unroll
        for (int stride = SHARDED_TILED_MIDDLE_WARPS / 2; stride > 0; stride >>= 1) {
            if (warp_in_output < stride) {
#pragma unroll
                for (int item = 0; item < ItemsPerLane; ++item) {
                    partials[warp_in_output][item][lane] += partials[warp_in_output + stride][item][lane];
                }
            }
            __syncthreads();
        }

        if (warp_in_output == 0 && packet_component_begin + static_cast<uint64_t>(ItemsPerLane) <= inner_size) {
            float reduced[ItemsPerLane];
#pragma unroll
            for (int item = 0; item < ItemsPerLane; ++item) {
                reduced[item] = partials[0][item][lane];
            }
            float* destination = workspace
                                 + (outer_index * shards_per_output + shard) * inner_size
                                 + packet_component_begin;
            storeFp32PacketVectorized<ItemsPerLane>(destination, reduced);
        }
        __syncthreads();
    }
}

template <int RowsPerShard>
[[nodiscard]] constexpr std::string_view shardedTiledMiddleCandidateName() {
    static_assert(RowsPerShard == 128 || RowsPerShard == 256 || RowsPerShard == 512 || RowsPerShard == 1024);
    if constexpr (RowsPerShard == 128) {
        return "tiled_middle_sharded_y8_rows128";
    } else if constexpr (RowsPerShard == 256) {
        return "tiled_middle_sharded_y8_rows256";
    } else if constexpr (RowsPerShard == 512) {
        return "tiled_middle_sharded_y8_rows512";
    } else {
        return "tiled_middle_sharded_y8_rows1024";
    }
}

template <int RowsPerShard>
class StampedShardedTiledMiddleCandidate final : public StampedReductionCandidate {
   public:
    StampedShardedTiledMiddleCandidate(Tensor input, std::vector<uint32_t> axes, const Stream& stream)
        : input_(std::move(input)),
          geometry_(CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_.getDimensions(), axes)),
          output_(input_.getPlacement(), TensorDescriptor(input_.getDataType(), geometry_.output_dimensions)) {
        if (geometry_.path != CubReductionPath::TiledFixedSegment) {
            throw std::logic_error("Sharded tiled-middle candidate requires TiledFixedSegment geometry.");
        }
        const uint64_t packet_elements = input_.getDataType() == DataType::FP32 ? 4ull : 8ull;
        if (geometry_.inner_size % packet_elements != 0) {
            throw std::logic_error("Sharded tiled-middle candidate requires a retained width divisible by its 16-byte packet width.");
        }

        shards_per_output_ = ceilDivideU64(geometry_.reduction_size, static_cast<uint64_t>(RowsPerShard));
        workspace_ = Tensor(input_.getPlacement(),
                            TensorDescriptor(DataType::FP32,
                                             {geometry_.outer_size, shards_per_output_, geometry_.inner_size}));

        // CubReduction's canonical tiled output for [outer, shards, inner] is [outer, 1, inner]. Alias the public
        // candidate output with that shape so the ordinary stage writes directly into the correctly shaped/typed
        // output allocation while the candidate still exposes the original reduction's public dimensions.
        Tensor stage_output = output_;
        stage_output.reshape({geometry_.outer_size, 1, geometry_.inner_size});
        second_stage_ = CubReduction(CubReductionOp::Sum, 1, input_.getDataType()).stamp(workspace_, stage_output, stream);
    }

    void runOn(Stream& stream) const override {
        const auto launch_first_stage = [&]<typename InputT, int ItemsPerLane>() {
            constexpr uint64_t components_per_tile =
                static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS) * static_cast<uint64_t>(ItemsPerLane);
            const uint64_t component_tiles = ceilDivideU64(geometry_.inner_size, components_per_tile);
            const uint64_t total_work = geometry_.outer_size * component_tiles * shards_per_output_;
            const unsigned int grid_blocks = static_cast<unsigned int>(
                std::min<uint64_t>(total_work, TILED_REDUCTION_MAX_GRID_BLOCKS));
            shardedCooperativeTiledMiddleFirstStageKernel<InputT, ItemsPerLane, RowsPerShard>
                <<<grid_blocks, SHARDED_TILED_MIDDLE_BLOCK_THREADS, 0, stream.getStream()>>>(
                    input_.getMemPtr<InputT>(),
                    workspace_.getMemPtr<float>(),
                    geometry_.outer_size,
                    geometry_.reduction_size,
                    geometry_.inner_size,
                    shards_per_output_);
            CUDA_CHECK(cudaGetLastError());
        };

        switch (input_.getDataType()) {
            case DataType::FP16:
                launch_first_stage.template operator()<__half, 8>();
                break;
            case DataType::BF16:
                launch_first_stage.template operator()<__nv_bfloat16, 8>();
                break;
            case DataType::FP32:
                launch_first_stage.template operator()<float, 4>();
                break;
            default:
                throw std::logic_error("Sharded tiled-middle candidate received an unsupported dtype.");
        }

        THOR_THROW_IF_FALSE(second_stage_ != nullptr);
        second_stage_->runOn(stream);
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override {
        THOR_THROW_IF_FALSE(second_stage_ != nullptr);
        return workspace_.getArraySizeInBytes() + second_stage_->getWorkspaceSizeInBytes();
    }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "tiled_middle_sharded_then_cub";
        metadata.strategy = shardedTiledMiddleCandidateName<RowsPerShard>();
        metadata.vector_elements_per_load = input_.getDataType() == DataType::FP32 ? 4u : 8u;
        metadata.block_threads = SHARDED_TILED_MIDDLE_BLOCK_THREADS;
        const uint64_t components_per_tile =
            static_cast<uint64_t>(TILED_REDUCTION_WARP_THREADS)
            * static_cast<uint64_t>(input_.getDataType() == DataType::FP32 ? 4 : 8);
        const uint64_t component_tiles = ceilDivideU64(geometry_.inner_size, components_per_tile);
        metadata.first_stage_blocks = geometry_.outer_size * component_tiles * shards_per_output_;
        metadata.shards_per_output = shards_per_output_;
        return metadata;
    }

   private:
    Tensor input_;
    CubReductionGeometry geometry_;
    mutable Tensor output_;
    mutable Tensor workspace_;
    std::shared_ptr<StampedCubReduction> second_stage_;
    uint64_t shards_per_output_ = 1;
};

template <int RowsPerShard>
class ShardedTiledMiddleCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override {
        return shardedTiledMiddleCandidateName<RowsPerShard>();
    }

    [[nodiscard]] bool supports(CubReductionOp op,
                                const TensorDescriptor& input_descriptor,
                                const std::vector<uint32_t>& axes) const override {
        if (op != CubReductionOp::Sum) {
            return false;
        }
        const DataType dtype = input_descriptor.getDataType();
        if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
            return false;
        }
        const CubReductionGeometry geometry =
            CubReduction::analyzeValueGeometry(CubReductionOp::Sum, input_descriptor.getDimensions(), axes);
        const uint64_t packet_elements = dtype == DataType::FP32 ? 4ull : 8ull;
        return geometry.path == CubReductionPath::TiledFixedSegment
               && geometry.inner_size % packet_elements == 0
               && geometry.reduction_size > static_cast<uint64_t>(RowsPerShard);
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream& stream) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument(
                "sharded tiled-middle candidate requires dense contiguous packet-aligned deep TiledFixedSegment SUM geometry.");
        }
        return std::make_unique<StampedShardedTiledMiddleCandidate<RowsPerShard>>(input, axes, stream);
    }
};

const ReductionCandidateRegistrar<AdaptiveCooperativeTiledValueCandidate> adaptive_cooperative_tiled_value_registrar;
const ReductionCandidateRegistrar<PacketAdaptiveCooperativeTiledValueCandidate>
    packet_adaptive_cooperative_tiled_value_registrar;

const ReductionCandidateRegistrar<ShardedTiledMiddleCandidate<128>> sharded_tiled_middle_rows128_registrar;
const ReductionCandidateRegistrar<ShardedTiledMiddleCandidate<256>> sharded_tiled_middle_rows256_registrar;
const ReductionCandidateRegistrar<ShardedTiledMiddleCandidate<512>> sharded_tiled_middle_rows512_registrar;
const ReductionCandidateRegistrar<ShardedTiledMiddleCandidate<1024>> sharded_tiled_middle_rows1024_registrar;

const ReductionCandidateRegistrar<CooperativeTiledMiddleCandidate<4>> cooperative_tiled_middle_y4_registrar;
const ReductionCandidateRegistrar<CooperativeTiledMiddleCandidate<8>> cooperative_tiled_middle_y8_registrar;
const ReductionCandidateRegistrar<CooperativeTiledMiddleCandidate<16>> cooperative_tiled_middle_y16_registrar;
const ReductionCandidateRegistrar<CooperativeTiledMiddleCandidate<32>> cooperative_tiled_middle_y32_registrar;

const ReductionCandidateRegistrar<ForcedDirectComponentTiledCandidate> forced_direct_component_tiled_registrar;

}  // namespace
}  // namespace ThorImplementation::CubReductionBenchmarking
