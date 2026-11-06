#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
namespace {
KParallelTiledStagePlan makeStagePlan(uint32_t packet, uint32_t threads, uint64_t shards) {
    return {shards == 1 ? TiledRKStageTopology::Complete : TiledRKStageTopology::Staged,
            packet, threads, shards};
}
}

uint32_t alignedKResidency(DataType input_dtype, DataType output_dtype,
                           const CubReductionStageSemantics& semantics,
                           uint32_t packet_bytes, uint32_t threads,
                           uint64_t shards_per_output) {
    if (input_dtype == DataType::FP32) {
        // 1192: one FP32 kernel per (packet, CTA, Staged / Complete); PRODUCT runs only with the identity transform.
        static_cast<void>(output_dtype);
        if (semantics.combine == CubReductionStageCombine::Product &&
            semantics.input_transform != CubReductionStageInputTransform::Identity) return 0u;
        const auto occ = queryFp32AlignedKOccupancy(packet_bytes, threads, shards_per_output > 1);
        return occ.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occ.max_active_blocks_per_sm) : 0u;
    }
    if (input_dtype == DataType::FP16 || input_dtype == DataType::BF16) {  // 1246: FP32's rule, per format
        static_cast<void>(output_dtype);
        if (semantics.combine == CubReductionStageCombine::Product &&
            semantics.input_transform != CubReductionStageInputTransform::Identity) return 0u;
        const auto occ = queryF16AlignedKOccupancy(input_dtype, packet_bytes, threads, shards_per_output > 1);
        return occ.max_active_blocks_per_sm > 0 ? static_cast<uint32_t>(occ.max_active_blocks_per_sm) : 0u;
    }
    // Unlike the generic KParallel query, this uses the retained transformed
    // and operation-aware physical specialization, including Complete's
    // output-dtype-sensitive shared-memory footprint.
    const auto occ = queryFp8AlignedKStageOccupancy(
        input_dtype, output_dtype, semantics.input_transform, semantics.combine,
        makeStagePlan(packet_bytes, threads, shards_per_output));
    return occ.max_active_blocks_per_sm > 0
        ? static_cast<uint32_t>(occ.max_active_blocks_per_sm) : 0u;
}

void launchAlignedK(const PhysicalKernel& kernel, const CubReductionStageSemantics& semantics,
                    const Tensor& input, Tensor& stage_output,
                    const CubReductionGeometry& geometry, float output_scale, Stream& stream) {
    if (kernel.family != PhysicalFamily::AlignedK ||
        !structurallyValid(kernel, geometry.outer_size, geometry.reduction_size, geometry.inner_size))
        throw std::logic_error("Invalid retained AlignedK physical stage");
    if (kernel.element_bytes == 4) {
        launchFp32AlignedKStage(semantics, input, stage_output, geometry, static_cast<uint32_t>(kernel.a),
                                static_cast<uint32_t>(kernel.b), kernel.c, output_scale, stream);
        return;
    }
    if (kernel.element_bytes == 2) {  // 1246
        launchF16AlignedKStage(semantics, input, stage_output, geometry, static_cast<uint32_t>(kernel.a),
                               static_cast<uint32_t>(kernel.b), kernel.c, output_scale, stream);
        return;
    }
    launchKParallelTiledStage(semantics, input, stage_output, geometry,
        makeStagePlan(static_cast<uint32_t>(kernel.a), static_cast<uint32_t>(kernel.b), kernel.c),
        output_scale, stream);
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
