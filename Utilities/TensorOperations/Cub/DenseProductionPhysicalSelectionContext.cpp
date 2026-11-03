#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"

#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/Common/Stream.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionInternal {

DenseRKValueOperation denseRKValueOperation(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum: return DenseRKValueOperation::Sum;
        case CubReductionOp::Mean: return DenseRKValueOperation::Mean;
        case CubReductionOp::L1Norm: return DenseRKValueOperation::L1;
        case CubReductionOp::L2Norm: return DenseRKValueOperation::L2;
        case CubReductionOp::SumSquares: return DenseRKValueOperation::SumSquares;
        case CubReductionOp::Product: return DenseRKValueOperation::Product;
        case CubReductionOp::Min: return DenseRKValueOperation::Minimum;
        case CubReductionOp::Max: return DenseRKValueOperation::Maximum;
    }
    throw std::logic_error("Invalid DenseRK VALUE operation");
}

DenseProductionPhysicalSelectionContext::DenseProductionPhysicalSelectionContext(
    DataType original_input_dtype, DenseRKValueOperation operation, const Stream& stream,
    uint64_t original_reduction_size, float output_scale,
    std::optional<CubReductionOp> original_value_operation, uint64_t workspace_budget_bytes,
    CubReductionTuningDepth tuning_depth) {
    cudaDeviceProp properties{};
    const cudaError_t status = cudaGetDeviceProperties(&properties, static_cast<int>(stream.getGpuNum()));
    if (status != cudaSuccess || properties.multiProcessorCount <= 0 || properties.warpSize <= 0
        || properties.maxThreadsPerMultiProcessor <= 0 || properties.maxBlocksPerMultiProcessor <= 0) {
        throw std::runtime_error(std::string("Failed to query CUDA device geometry for dense VALUE planning: ")
                                 + cudaGetErrorString(status));
    }

    limits_ = Limits{static_cast<uint32_t>(properties.maxThreadsPerMultiProcessor),
                     static_cast<uint32_t>(properties.maxBlocksPerMultiProcessor),
                     static_cast<uint32_t>(properties.warpSize)};
    context_.tuning_stream = &stream;
    context_.gpu_ordinal = stream.getGpuNum();
    context_.original_value_operation = original_value_operation;
    context_.original_reduction_size = original_reduction_size;
    context_.output_scale = output_scale;
    context_.max_workspace_bytes = workspace_budget_bytes;
    context_.tuning_depth = tuning_depth;
    context_.rk.original_input_dtype = original_input_dtype;
    context_.rk.value_operation = operation;
    context_.rk.multiprocessors = static_cast<uint32_t>(properties.multiProcessorCount);
    context_.rk.warp_size = limits_.warp_size;
    context_.rk.max_threads_per_sm = limits_.max_threads_per_sm;
    context_.rk.max_blocks_per_sm = limits_.max_blocks_per_sm;
    context_.rk.occupancy_query = queryProductionRKOccupancy;
    context_.rk.occupancy_query_context = &limits_;
    // Non-FP8 RK still uses its existing calibrated/generic production coverage policy.
    // Original-input FP8 RK VALUE is intercepted by the retained selector.
    context_.rk.allow_generic_rk_fallback = true;
}

DenseRKOccupancyInfo DenseProductionPhysicalSelectionContext::queryProductionRKOccupancy(
    const DenseRKFamilyPhysicalPlan& plan, void* user_context) {
    using namespace CubDevicePrimitiveSupport;
    CubKernelOccupancyInfo exact;
    switch (plan.implementation) {
        case DenseRKProductionImplementation::KParallelPass: {
            const KParallelTiledStagePlan physical{
                plan.progress == DenseRKProgress::Complete ? TiledRKStageTopology::Complete
                                                            : TiledRKStageTopology::Staged,
                plan.packet_bytes,
                plan.block_threads,
                plan.shards_per_output};
            exact = queryKParallelTiledStageOccupancy(plan.input_dtype, physical);
            break;
        }
        case DenseRKProductionImplementation::RCooperativeAlignedStaged:
            exact = queryCooperativeShardedTiledFirstStageOccupancy(plan.input_dtype, plan.rows_per_shard);
            break;
        case DenseRKProductionImplementation::RCooperativeRotatedStaged:
            exact = queryAwkwardAlignmentRotatedShardedFirstStageOccupancy(plan.input_dtype, plan.rows_per_shard);
            break;
        case DenseRKProductionImplementation::RCooperativeFlatRowsStaged:
            exact = queryNarrowLowPrecisionFlatRCooperativeFirstStageOccupancy(plan.input_dtype, plan.rows_per_shard);
            break;
        case DenseRKProductionImplementation::RCooperativeAlignedComplete: {
            const auto* fallback = static_cast<const Limits*>(user_context);
            if (fallback == nullptr || plan.block_threads == 0 || fallback->warp_size == 0) {
                throw std::logic_error("Packet-adaptive production occupancy is missing device geometry.");
            }
            const uint32_t thread_limited = fallback->max_threads_per_sm / plan.block_threads;
            const uint32_t blocks = std::min(fallback->max_blocks_per_sm, thread_limited);
            const uint32_t warps_per_block = plan.block_threads / fallback->warp_size;
            if (blocks == 0 || warps_per_block == 0) {
                throw std::logic_error("Packet-adaptive production occupancy found no resident blocks.");
            }
            return DenseRKOccupancyInfo{blocks, blocks * warps_per_block};
        }
        default:
            throw std::logic_error("Production DenseRK occupancy received a non-family implementation.");
    }
    if (exact.max_active_blocks_per_sm <= 0 || exact.max_active_warps_per_sm <= 0) {
        throw std::logic_error("Production DenseRK exact occupancy returned zero residency.");
    }
    return DenseRKOccupancyInfo{static_cast<uint32_t>(exact.max_active_blocks_per_sm),
                                static_cast<uint32_t>(exact.max_active_warps_per_sm)};
}

}  // namespace ThorImplementation::CubReductionInternal
