#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"

#include "Utilities/TensorOperations/Cub/ReductionAutotuner.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
namespace {
[[nodiscard]] size_t reserveBytes(size_t& cursor, size_t bytes) {
    if (!bytes) return 0;
    if (cursor > std::numeric_limits<size_t>::max() - 31)
        throw std::overflow_error("FP8 benchmark scratch alignment overflow");
    cursor = (cursor + 31) & ~size_t{31};
    if (bytes > std::numeric_limits<size_t>::max() - cursor)
        throw std::overflow_error("FP8 benchmark scratch size overflow");
    const size_t offset = cursor;
    cursor += bytes;
    return offset;
}

[[nodiscard]] size_t fp32Bytes(const std::vector<uint64_t>& dimensions) {
    return TensorDescriptor(DataType::FP32, dimensions).getArraySizeInBytes();
}
}  // namespace

SelectedIsolatedBenchmark makeSelectedIsolatedBenchmark(
    const DenseReductionProblem& original_problem, DenseReductionSite nominated_site,
    IsolatedRKSite site, PhysicalKernel first_physical,
    const DensePhysicalSelectionContext& fp32_selector,
    float output_scale, uint64_t workspace_budget_bytes, const Stream& stream) {
    if (!std::isfinite(output_scale))
        throw std::invalid_argument("Nonfinite FP8 isolated benchmark output scale");
    if (first_physical.staged() != site.layout.has_fp32_continuation)
        throw std::invalid_argument("FP8 isolated benchmark first-stage topology mismatch");
    if (site.layout.partial_bytes > std::numeric_limits<size_t>::max())
        throw std::overflow_error("FP8 benchmark partial tensor size exceeds size_t");
    // Reject an impossible minimum budget before selecting any FP32 dependencies.
    if (site.layout.partial_bytes > workspace_budget_bytes)
        throw UnavailableIsolatedCandidate("FP8 first-stage partial tensor exceeds benchmark workspace budget");
    // Sequential selection through the production channel (including staged
    // FP32 passes), before the caller enters a GPU timing contest.
    SelectedIsolatedContinuation chain = selectIsolatedFp32Continuation(
        original_problem, nominated_site, site, fp32_selector);
    SelectedIsolatedBenchmark bench;
    bench.site = std::move(site);
    bench.first_physical = first_physical;
    bench.first_geometry = CubReduction::analyzeGeometry(
        {bench.site.outer, bench.site.r, bench.site.k}, std::vector<uint32_t>{1});
    bench.output_scale = bench.site.layout.last_reduction_site ? output_scale : 1.0f;

    if (bench.site.layout.has_fp32_continuation) {
        bench.first_partial_offset = reserveBytes(bench.scratch_bytes,
                                                  static_cast<size_t>(bench.site.layout.partial_bytes));
    }
    bench.continuation.reserve(chain.fp32_passes.size());
    for (size_t i = 0; i < chain.fp32_passes.size(); ++i) {
        const auto& selected = chain.fp32_passes[i].selected;
        BenchmarkContinuationPass pass;
        pass.selected = selected;
        pass.layout = describeDensePhysicalPass(*selected.physical_plan);
        pass.last_pass = i + 1 == chain.fp32_passes.size();
        // A nonterminal physical FP32 pass must keep FP32, even if it finished
        // its own shard extent. Only final mathematical reduction converts.
        const bool final_math_pass = !hasDenseReductionSites(selected.next_problem);
        pass.output_dtype = final_math_pass ? original_problem.final_output.value_dtype : DataType::FP32;
        pass.semantics = bench.site.continuation_semantics;
        if (!final_math_pass) {
            pass.semantics.finalize = CubReductionStageFinalize::Identity;
            pass.semantics.finalize_divisor = 1;
        }
        pass.semantics.input_transform = CubReductionStageInputTransform::Identity;
        if (!pass.last_pass)
            pass.output_offset = reserveBytes(bench.scratch_bytes,
                                              fp32Bytes(pass.layout.intermediate_dimensions));
        pass.physical_scratch_bytes = queryDensePhysicalPassScratchBytes(
            *selected.physical_plan, pass.semantics, DataType::FP32, nullptr,
            pass.layout.geometry.input_elements, pass.output_dtype, nullptr,
            pass.layout.geometry, final_math_pass ? bench.output_scale : 1.0f, stream);
        pass.physical_scratch_offset = reserveBytes(bench.scratch_bytes, pass.physical_scratch_bytes);
        bench.continuation.push_back(std::move(pass));
    }
    if (bench.scratch_bytes > workspace_budget_bytes)
        throw UnavailableIsolatedCandidate("Selected FP8 benchmark continuation exceeds workspace budget");
    return bench;
}

void SelectedIsolatedBenchmark::launch(const Tensor& input, Tensor& output,
                                       const Tensor& scratch, Stream& stream) const {
    // 1192: FP32 first stages (element_bytes 4, the FP32 RK selector) time the same program on FP32 input.
    const bool input_matches = first_physical.element_bytes == 4
                                   ? input.getDataType() == DataType::FP32
                                   : (input.getDataType() == DataType::FP8_E4M3 ||
                                      input.getDataType() == DataType::FP8_E5M2);
    if (!input_matches || !input.isDenseContiguous() || !output.isDenseContiguous() ||
        input.getTotalNumElements() != first_geometry.input_elements ||
        output.getTotalNumElements() != first_geometry.output_elements ||
        output.getDataType() != site.site_output_dtype)
        throw std::invalid_argument("Selected FP8 benchmark input/output mismatch");
    if (scratch_bytes && (!scratch.isInitialized() || scratch.getDataType() != DataType::UINT8 ||
                          scratch.getArraySizeInBytes() < scratch_bytes))
        throw std::invalid_argument("Selected FP8 benchmark scratch too small");
    Tensor first_input = input;
    first_input.reshape({site.outer, site.r, site.k});
    Tensor final_output = output;
    final_output.reshape(first_geometry.output_dimensions);
    if (continuation.empty()) {
        launchPhysicalFirstStage(first_physical, site.first_stage_semantics, first_input,
                                 final_output, first_geometry, output_scale, stream);
        return;
    }

    Tensor previous = scratch.aliasByteStorageAs(DataType::FP32,
        {site.outer, site.layout.shards, site.k}, first_partial_offset);
    launchPhysicalFirstStage(first_physical, site.first_stage_semantics, first_input,
                             previous, first_geometry, 1.0f, stream);
    for (const BenchmarkContinuationPass& pass : continuation) {
        Tensor pass_input = previous;
        pass_input.reshape(pass.layout.input_dimensions);
        Tensor pass_output = pass.last_pass ? final_output : scratch.aliasByteStorageAs(
            DataType::FP32, pass.layout.intermediate_dimensions, pass.output_offset);
        if (pass.last_pass) {
            if (pass_output.getDataType() != pass.output_dtype ||
                pass_output.getTotalNumElements() != pass.layout.geometry.output_elements)
                throw std::logic_error("FP8 benchmark final FP32 pass output shape/dtype mismatch");
            pass_output.reshape(pass.layout.geometry.output_dimensions);
        }
        Tensor pass_scratch = scratch;
        if (pass.physical_scratch_bytes)
            pass_scratch = scratch.aliasByteStorageAs(DataType::UINT8,
                {static_cast<uint64_t>(pass.physical_scratch_bytes)}, pass.physical_scratch_offset);
        launchDensePhysicalPass(*pass.selected.physical_plan, pass.semantics,
                                pass_scratch, pass.physical_scratch_bytes,
                                pass_input, pass_output, pass.layout.geometry,
                                pass.last_pass ? this->output_scale : 1.0f, stream);
        previous = std::move(pass_output);
    }
}

ReductionAutotuning::Candidate SelectedIsolatedBenchmark::asAutotuneCandidate(std::string name) const {
    ReductionAutotuning::Candidate candidate;
    candidate.name = std::move(name);
    candidate.scratch_bytes = scratch_bytes;
    candidate.launch = [program = *this](const Tensor& input, Tensor& output, Tensor& scratch, Stream& stream) {
        program.launch(input, output, scratch, stream);
    };
    return candidate;
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
