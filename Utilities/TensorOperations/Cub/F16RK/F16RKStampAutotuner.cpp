#include "Utilities/TensorOperations/Cub/F16RK/F16RKStampAutotuner.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKArgCandidates.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKDescriptorSelection.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKIsolatedSite.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKProductionCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"
#include "Utilities/TensorOperations/Cub/ReductionAutotuner.h"
#include "Utilities/TensorOperations/Cub/ReductionGpuMeasurementLock.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionInternal::F16RK {
namespace {
using Fp8RK::PhysicalFamily;
using Fp8RK::PhysicalKernel;

struct Shape {
    uint64_t outer = 1, r = 0, k = 1;
};
uint64_t checkedMul(uint64_t a, uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("FP16 / BF16 RK stamp shape overflow");
    return a * b;
}
bool isF16(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }
Shape shapeAtSite(const DenseReductionProblem& problem, DenseReductionSite site) {
    Shape shape;
    for (size_t i = 0; i < problem.runs.size(); ++i) {
        if (i < site.run_index) shape.outer = checkedMul(shape.outer, problem.runs[i].extent);
        else if (i == site.run_index) shape.r = problem.runs[i].extent;
        else shape.k = checkedMul(shape.k, problem.runs[i].extent);
    }
    return shape;
}
// The FP16 / BF16 VALUE domain, checked before the cache so a hit cannot make an invalid request acceptable: an
// original FP16 / BF16 RK site with value-only output.
Shape validateValueSite(const DenseReductionProblem& problem, DenseReductionSite site, DataType input_dtype,
                        uint64_t original_reduction_size, float output_scale) {
    if (!isOriginalF16RKSite(problem, site, input_dtype) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Value)
        throw std::invalid_argument("FP16 / BF16 RK selection needs an original FP16 / BF16 VALUE RK site");
    if (!problem.final_output.produce_value || problem.final_output.produce_index || original_reduction_size == 0 ||
        !std::isfinite(output_scale))
        throw std::invalid_argument("Invalid FP16 / BF16 RK VALUE selection semantics");
    return shapeAtSite(problem, site);
}
// 1251: the FP16 / BF16 ARG domain: an original FP16 / BF16 ARG RK site.
Shape validateArgSite(const DenseReductionProblem& problem, DenseReductionSite site, DataType input_dtype,
                      uint64_t original_reduction_size) {
    if (!isOriginalF16RKSite(problem, site, input_dtype) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Arg || original_reduction_size == 0)
        throw std::invalid_argument("FP16 / BF16 RK ARG selection needs an original FP16 / BF16 ARG RK site");
    return shapeAtSite(problem, site);
}
const char* familyName(PhysicalFamily family) {
    switch (family) {
        case PhysicalFamily::OneWindow: return "f16_one_window";
        case PhysicalFamily::DeepRSmallK: return "f16_deep_r_small_k";
        case PhysicalFamily::AlignedK: return "f16_aligned_k";
        case PhysicalFamily::RuntimeK: return "f16_runtime_k";
    }
    throw std::logic_error("Unknown physical FP16 / BF16 RK family");
}
// The input alignment the selection honours: the original input's, at most 16 (and at least the element).
uint64_t siteAlignment(uint64_t input_alignment_bytes) {
    if (input_alignment_bytes < 2 || (input_alignment_bytes & (input_alignment_bytes - 1)) != 0)
        throw std::invalid_argument("FP16 / BF16 RK input alignment must be a power of two of at least 2");
    return std::min<uint64_t>(input_alignment_bytes, 16);
}
}  // namespace

bool isOriginalF16RKSite(const DenseReductionProblem& problem, DenseReductionSite site,
                         DataType original_input_dtype) {
    return isF16(original_input_dtype) && problem.input_state == DenseReductionInputState::OriginalInput &&
           site.run_index < problem.runs.size() &&
           problem.runs[site.run_index].kind == DenseReductionRunKind::OriginalReduction &&
           classifyDenseReducerFamily(problem, site) == DenseReducerFamily::RK;
}

F16RKSelection autotuneValueSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                 CubReductionOp operation, uint64_t original_reduction_size, float output_scale,
                                 const Tensor& representative_input, const Stream& stream,
                                 uint64_t workspace_budget_bytes, CubReductionTuningDepth tuning_depth,
                                 uint64_t input_alignment_bytes) {
    const DataType input_dtype = representative_input.getDataType();
    if (!isF16(input_dtype)) throw std::invalid_argument("FP16 / BF16 RK autotuning requires FP16 or BF16 input");
    const Shape shape = validateValueSite(problem, site, input_dtype, original_reduction_size, output_scale);
    const uint64_t alignment = siteAlignment(input_alignment_bytes);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    if (!representative_input.isDenseContiguous() ||
        representative_input.getTotalNumElements() != checkedMul(checkedMul(shape.outer, shape.r), shape.k) ||
        representative_input.getPlacement().getDeviceNum() != stream.getGpuNum())
        throw std::invalid_argument(
            "FP16 / BF16 RK autotune input must match contiguous selected-site geometry and stream");
    // The pass role: Complete or First on original input (the transform here, the finalizer on the last pass).
    const auto semantics = makeValueReductionStageSemantics(
        operation, deriveDensePassRole(problem, advanceDenseReductionProblem(problem, site, 1)),
        original_reduction_size);
    const auto device = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));

    F16RKSite candidate_site;
    candidate_site.outer = shape.outer;
    candidate_site.r = shape.r;
    candidate_site.k = shape.k;
    candidate_site.sm_count = device.sm_count;
    candidate_site.input_dtype = input_dtype;
    candidate_site.semantics = semantics;
    candidate_site.scratch_budget_bytes = workspace_budget_bytes;
    auto pool = f16RKCandidatePool(candidate_site, [&](const PhysicalKernel& kernel) {
        return f16RKCompiledResidency(kernel, input_dtype, shape.k, semantics);
    });
    std::erase_if(pool, [&](const F16RKPoolEntry& entry) { return f16RKPassInputAlignment(entry.kernel) > alignment; });
    if (pool.empty())
        throw std::runtime_error("No retained FP16 / BF16 RK pass runs at this input's " + std::to_string(alignment) +
                                 "-byte alignment");
    const auto ranked = f16RKRankedKernels(pool, candidate_site, f16RKShortlistLength(tuning_depth));

    // The production policy selects each staged candidate's FP32 continuation (outside the GPU lock), as for FP8 and
    // FP32, with the original semantics and the tuning depth the retained FP32 routing requires.
    const DenseProductionPhysicalSelectionContext continuation_context(
        input_dtype, denseRKValueOperation(operation), stream, original_reduction_size, output_scale, operation,
        workspace_budget_bytes, tuning_depth);
    std::vector<Fp8RK::SelectedIsolatedBenchmark> executables;
    std::vector<ReductionAutotuning::Candidate> candidates;
    for (const PhysicalKernel& kernel : ranked) {
        Fp8RK::IsolatedRKSite isolated = Fp8RK::planIsolatedRKSite(problem, site, kernel, semantics);
        try {
            auto executable = Fp8RK::makeSelectedIsolatedBenchmark(problem, site, std::move(isolated), kernel,
                                                                   continuation_context.context(), output_scale,
                                                                   workspace_budget_bytes, stream);
            candidates.push_back(executable.asAutotuneCandidate(std::string(familyName(kernel.family)) + "#" +
                                                                std::to_string(candidates.size())));
            executables.push_back(std::move(executable));
        } catch (const Fp8RK::UnavailableIsolatedCandidate&) {
            continue;
        }
    }
    if (candidates.empty())
        throw std::runtime_error("No legal retained FP16 / BF16 RK candidate under the workspace budget");

    const TensorDescriptor output_descriptor(executables.front().site.site_output_dtype, {shape.outer, shape.k});
    const Fp8RK::Fp8RKTuningPlan tuning = Fp8RK::fp8RKTuningPlan(tuning_depth);
    Stream tuning_stream = stream;
    const auto result = [&] {
        ScopedReductionGpuMeasurement measurement(static_cast<int>(stream.getGpuNum()));
        ReductionAutotuning::Options options;
        options.measurement_count = tuning.rounds;
        options.trim_count = tuning.trimmed;
        return ReductionAutotuning::autotune(candidates, representative_input, output_descriptor, tuning_stream,
                                             options);
    }();
    if (result.winner_index >= executables.size())
        throw std::logic_error("FP16 / BF16 RK autotuner returned an out-of-range winner");
    F16RKSelection selection;
    selection.physical = executables[result.winner_index].first_physical;
    selection.winner_index = result.winner_index;
    selection.winner_name = result.winner_name;
    selection.candidate_count = candidates.size();
    return selection;
}

F16RKSelection selectCachedValueSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                     DataType input_dtype, CubReductionOp operation,
                                     uint64_t original_reduction_size, float output_scale, const Stream& stream,
                                     uint64_t workspace_budget_bytes, CubReductionTuningDepth tuning_depth,
                                     uint64_t input_alignment_bytes) {
    const Shape shape = validateValueSite(problem, site, input_dtype, original_reduction_size, output_scale);
    const uint64_t alignment = siteAlignment(input_alignment_bytes);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const auto identity = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    // FP8 RK's descriptor (every field that can change the shortlist, continuation or winner; the input dtype
    // separates FP16 from BF16), under kind F16RK, with the alignment.
    DenseReductionSelectionKey key = Fp8RK::toDenseReductionSelectionKey(Fp8RK::makeFp8RKDescriptorSelectionKey(
        problem, site, operation, original_reduction_size, output_scale, input_dtype, identity,
        workspace_budget_bytes, tuning_depth));
    key.kind = DenseReductionSelectionKind::F16RK;
    key.site.push_back(alignment);
    return denseReductionSelectionCache().getOrTune<F16RKSelection>(key, [&] {
        // A temporary representative input, allocated and released under the measurement lock; the contest takes the
        // lock separately (as FP8 and FP32 RK's miss paths).
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
        Tensor temporary;
        try {
            {
                ScopedReductionGpuMeasurement preparation(static_cast<int>(stream.getGpuNum()));
                temporary = Tensor(placement, TensorDescriptor(input_dtype, {shape.outer, shape.r, shape.k}));
                Stream fill_stream = stream;
                temporary.fill(1.0, fill_stream);
                fill_stream.synchronize();
            }
            auto winner = autotuneValueSite(problem, site, operation, original_reduction_size, output_scale, temporary,
                                            stream, workspace_budget_bytes, tuning_depth, alignment);
            {
                ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
                temporary.dropReference();
            }
            return winner;
        } catch (...) {
            ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
            temporary.dropReference();
            throw;
        }
    });
}

F16RKSelection autotuneArgSite(const DenseReductionProblem& problem, DenseReductionSite site, bool maximum,
                               uint64_t original_reduction_size, const Tensor& representative_input,
                               const Stream& stream, uint64_t workspace_budget_bytes,
                               CubReductionTuningDepth tuning_depth, uint64_t input_alignment_bytes) {
    const DataType input_dtype = representative_input.getDataType();
    if (!isF16(input_dtype)) throw std::invalid_argument("FP16 / BF16 RK ARG autotuning requires FP16 or BF16 input");
    const Shape shape = validateArgSite(problem, site, input_dtype, original_reduction_size);
    const uint64_t alignment = siteAlignment(input_alignment_bytes);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    if (!representative_input.isDenseContiguous() ||
        representative_input.getTotalNumElements() != checkedMul(checkedMul(shape.outer, shape.r), shape.k) ||
        representative_input.getPlacement().getDeviceNum() != stream.getGpuNum())
        throw std::invalid_argument(
            "FP16 / BF16 RK ARG autotune input must match contiguous selected-site geometry and stream");
    const bool final_site = !hasDenseReductionSites(advanceDenseReductionProblem(problem, site, 1));

    // The VALUE MIN / MAX pool (identity transform): the ARG twins share its identities.
    auto semantics = makeValueReductionStageSemantics(maximum ? CubReductionOp::Max : CubReductionOp::Min,
                                                      DensePassRole::Complete, shape.r);
    semantics.combine = maximum ? CubReductionStageCombine::Maximum : CubReductionStageCombine::Minimum;
    semantics.input_transform = CubReductionStageInputTransform::Identity;
    const auto device = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    F16RKSite candidate_site;
    candidate_site.outer = shape.outer;
    candidate_site.r = shape.r;
    candidate_site.k = shape.k;
    candidate_site.sm_count = device.sm_count;
    candidate_site.input_dtype = input_dtype;
    candidate_site.semantics = semantics;
    const auto value_pool = f16RKCandidatePool(candidate_site, [&](const PhysicalKernel& kernel) {
        return f16RKCompiledResidency(kernel, input_dtype, shape.k, semantics);
    });
    auto generated = f16RKArgCandidates(value_pool, shape.outer, shape.r, shape.k, workspace_budget_bytes);
    F16RKSelection selection;
    if (generated.empty()) return selection;  // candidate_count 0: no 2-byte ARG pass
    std::erase_if(generated, [&](const F16RKArgCandidate& candidate) {
        return f16RKPassInputAlignment(candidate.identity) > alignment;
    });
    if (generated.empty())
        throw std::runtime_error("No retained FP16 / BF16 RK ARG pass runs at this input's " +
                                 std::to_string(alignment) + "-byte alignment");

    // Rank with the ARG aggregate of this pass's outputs (1 index, 2 value and index).
    const bool produce_value = !final_site || problem.final_output.produce_value;
    const bool produce_index = !final_site || problem.final_output.produce_index;
    const uint32_t aggregate = produce_value ? 2U : 1U;
    const auto ranked = f16RKRankedKernels(f16RKArgRankerPool(generated), candidate_site,
                                           f16RKShortlistLength(tuning_depth), aggregate);

    // Outputs exactly as the stamped pass writes them: the requested outputs on the final site, else the FP32 value +
    // carried index intermediate. One UINT8 output slot holds the value then (256-byte aligned) the index.
    const DataType value_dtype = final_site ? problem.final_output.value_dtype : DataType::FP32;
    const DataType index_dtype = final_site ? problem.final_output.index_dtype : problem.carried_arg_index_dtype;
    const std::vector<uint64_t> output_dims{shape.outer, shape.k};
    const uint64_t value_bytes = produce_value ? TensorDescriptor(value_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t index_offset = (value_bytes + 255) & ~uint64_t{255};
    const uint64_t index_bytes = produce_index ? TensorDescriptor(index_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t domain_stride = denseArgOriginalIndexStride(problem.runs[site.run_index]).value_or(0);
    const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({shape.outer, shape.r, shape.k}, {1});
    std::vector<PhysicalKernel> passes;
    std::vector<ReductionAutotuning::Candidate> candidates;
    for (const PhysicalKernel& identity : ranked) {
        ReductionAutotuning::Candidate candidate;
        candidate.name = std::string(familyName(identity.family)) + "_arg#" + std::to_string(candidates.size());
        candidate.scratch_bytes = f16RKArgScratchBytes(identity, shape.outer, shape.r, shape.k);
        const bool staged = candidate.scratch_bytes != 0;
        candidate.launch = [=](const Tensor& input, Tensor& output, Tensor& scratch, Stream& run_stream) {
            Tensor site_input = input;
            site_input.reshape({shape.outer, shape.r, shape.k});
            std::optional<Tensor> value, index;
            if (produce_value) value = output.aliasByteStorageAs(value_dtype, output_dims, 0);
            if (produce_index) index = output.aliasByteStorageAs(index_dtype, output_dims, index_offset);
            uint64_t* packed = staged ? static_cast<uint64_t*>(scratch.getMemPtr<void>()) : nullptr;
            launchF16RKArgPass(identity, site_input, value ? &*value : nullptr, index ? &*index : nullptr, geometry,
                               packed, maximum, domain_stride, run_stream);
        };
        candidates.push_back(std::move(candidate));
        passes.push_back(identity);
    }
    const Fp8RK::Fp8RKTuningPlan tuning = Fp8RK::fp8RKTuningPlan(tuning_depth);
    Stream tuning_stream = stream;
    const auto result = [&] {
        ScopedReductionGpuMeasurement measurement(static_cast<int>(stream.getGpuNum()));
        ReductionAutotuning::Options options;
        options.measurement_count = tuning.rounds;
        options.trim_count = tuning.trimmed;
        return ReductionAutotuning::autotune(candidates, representative_input,
                                             TensorDescriptor(DataType::UINT8, {index_offset + index_bytes}),
                                             tuning_stream, options);
    }();
    if (result.winner_index >= passes.size())
        throw std::logic_error("FP16 / BF16 RK ARG autotuner returned an out-of-range winner");
    selection.physical = passes[result.winner_index];
    selection.winner_index = result.winner_index;
    selection.winner_name = result.winner_name;
    selection.candidate_count = candidates.size();
    return selection;
}

F16RKSelection selectCachedArgSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                   DataType input_dtype, bool maximum, uint64_t original_reduction_size,
                                   const Stream& stream, uint64_t workspace_budget_bytes,
                                   CubReductionTuningDepth tuning_depth, uint64_t input_alignment_bytes) {
    const Shape shape = validateArgSite(problem, site, input_dtype, original_reduction_size);
    const uint64_t alignment = siteAlignment(input_alignment_bytes);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const auto identity = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    DenseReductionSelectionKey key = Fp8RK::toDenseReductionSelectionKey(Fp8RK::makeFp8RKDescriptorSelectionKey(
        problem, site, maximum ? CubReductionOp::Max : CubReductionOp::Min, original_reduction_size, 1.0f,
        input_dtype, identity, workspace_budget_bytes, tuning_depth));
    key.kind = DenseReductionSelectionKind::F16RK;
    key.site.push_back(alignment);
    return denseReductionSelectionCache().getOrTune<F16RKSelection>(key, [&] {
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
        Tensor temporary;
        try {
            {
                ScopedReductionGpuMeasurement preparation(static_cast<int>(stream.getGpuNum()));
                temporary = Tensor(placement, TensorDescriptor(input_dtype, {shape.outer, shape.r, shape.k}));
                Stream fill_stream = stream;
                temporary.fill(1.0, fill_stream);
                fill_stream.synchronize();
            }
            auto winner = autotuneArgSite(problem, site, maximum, original_reduction_size, temporary, stream,
                                          workspace_budget_bytes, tuning_depth, alignment);
            {
                ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
                temporary.dropReference();
            }
            return winner;
        } catch (...) {
            ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
            temporary.dropReference();
            throw;
        }
    });
}

}  // namespace ThorImplementation::CubReductionInternal::F16RK
