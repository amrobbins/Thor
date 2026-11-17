#pragma once

// Compact, production-owned launch ABI.  Keep the full XGBoost model and the
// benchmark ranking implementation OUT of the four independent family object
// files. Each adapter invokes the same CUDA entry point used by the
// benchmark: no additional CUDA template instantiations or device objects.
// The executable FP32 continuation is deliberately not part of this first-stage
// descriptor; an autotune plan must own and benchmark that continuation too.

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalIdentity.h"
#include <cstdint>
#include <vector>

namespace ThorImplementation::ReductionInternal::RK {

// Only exact physically compiled variants may report residency. Returns 0 on
// unsupported/absent specializations; any other failure is propagated.
[[nodiscard]] uint32_t oneWindowResidency(DataType input_dtype, uint64_t k,
                                            ReductionStageCombine combine,
                                            uint32_t outputs_per_warp,
                                            uint32_t warps_per_output, uint32_t packet_bytes);
[[nodiscard]] uint32_t deepRSmallKResidency(DataType input_dtype, uint64_t k,
                                             const ReductionStageSemantics& semantics,
                                             uint64_t rows_per_shard);
[[nodiscard]] uint32_t alignedKResidency(DataType input_dtype, DataType output_dtype,
                                          const ReductionStageSemantics& semantics,
                                          uint32_t packet_bytes, uint32_t threads,
                                          uint64_t shards_per_output);
[[nodiscard]] uint32_t runtimeKResidency(DataType input_dtype,
                                          const ReductionStageSemantics& semantics,
                                          const PhysicalKernel& kernel);

// One-pass launchers. Complete variants write final output; staged variants
// write FP32 [outer,shards,K] partials, for a separately planned continuation.
// No kernel selection, allocation, or synchronization in these functions.
void launchOneWindow(const PhysicalKernel& kernel, const ReductionStageSemantics& semantics,
                     const StridedSpan& input, const StridedSpan& output, const ReductionGeometry& geometry,
                     float output_scale, Stream& stream);
void launchDeepRSmallK(const PhysicalKernel& kernel, const ReductionStageSemantics& semantics,
                       const StridedSpan& input, const StridedSpan& fp32_partials, const ReductionGeometry& geometry,
                       Stream& stream);
void launchAlignedK(const PhysicalKernel& kernel, const ReductionStageSemantics& semantics,
                    const StridedSpan& input, const StridedSpan& stage_output, const ReductionGeometry& geometry,
                    float output_scale, Stream& stream);
void launchRuntimeK(const PhysicalKernel& kernel, const ReductionStageSemantics& semantics,
                    const StridedSpan& input, const StridedSpan& stage_output, const ReductionGeometry& geometry,
                    float output_scale, Stream& stream);

void launchPhysicalFirstStage(const PhysicalKernel& kernel, const ReductionStageSemantics& semantics,
                              const StridedSpan& input, const StridedSpan& stage_output,
                              const ReductionGeometry& geometry, float output_scale, Stream& stream);

}  // namespace ThorImplementation::ReductionInternal::RK
