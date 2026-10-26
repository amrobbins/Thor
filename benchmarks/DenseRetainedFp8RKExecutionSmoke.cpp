// Exercise the ordinary flat physical-pass materializer and stamped runOn
// path with a deterministic injected FP8 physical choice. Production already
// routes eligible original FP8 RK through the retained selector by default.
#include "Utilities/TensorOperations/Cub/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
namespace C = ThorImplementation::CubReductionInternal;
namespace F = ThorImplementation::CubReductionInternal::Fp8RK;

// Test-only access to the same production constructor and stamp-owned buffers.
// There is no alternative executor: this builds a normal StampedCubReduction.
namespace ThorImplementation {
struct DenseRetainedFp8RKExecutionProbe {
    static std::shared_ptr<StampedCubReduction> stamp(
        const Tensor& input, const Tensor& output, const std::vector<uint32_t>& axes,
        C::DenseReductionPlan plan, uint64_t total_reduction, Stream& stream,
        CubReductionOp op) {
        C::DenseExecutablePlan passes = C::materializeDenseValueExecutablePlan(
            op, input, output, plan, total_reduction, 1.0f, stream);
        return std::shared_ptr<StampedCubReduction>(new StampedCubReduction(
            op,
            CubReduction::analyzeGeometry(input.getDimensions(), axes),
            input, output, std::move(passes), "P2-C physical-pass integration",
            true, true, total_reduction, 1.0f, stream));
    }
    static const std::vector<C::DenseExecutablePass>& passes(const StampedCubReduction& stamp) {
        return stamp.dense_executable_plan.passes;
    }
    static bool hasLegacyStampScratch(const StampedCubReduction& stamp) {
        return stamp.temp_storage.isInitialized();
    }
};
}

namespace {
struct Choice {
    C::DensePhysicalSelectionContext production;
    F::PhysicalKernel first;
    unsigned retained_selections = 0;
    unsigned other_selections = 0;
};

C::DenseReductionCandidate proposal(const C::DenseReductionProblem& problem,
                                    C::DenseReductionSite site, C::DenseReducerFamily family,
                                    const C::DenseRKPlanningContext&, void* opaque) {
    auto& choice = *static_cast<Choice*>(opaque);
    if (problem.input_state == C::DenseReductionInputState::OriginalInput &&
        family == C::DenseReducerFamily::RK) {
        ++choice.retained_selections;
        return C::makeRetainedFp8RKPhysicalCandidate(problem, site, choice.first,
                                                     choice.production);
    }
    ++choice.other_selections;
    // This is THE production FP32 physical selector. No private completion or
    // injected continuation policy is permitted.
    return C::selectDensePhysicalPass(problem, site, choice.production);
}

[[nodiscard]] F::PhysicalKernel deepR(uint64_t r, uint64_t k) {
    const uint64_t n = r / 128 + (r % 128 != 0);
    return {F::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, n, n * k * 4};
}
[[nodiscard]] F::PhysicalKernel complete() {
    return {F::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 1, 0};
}
[[nodiscard]] F::PhysicalKernel runtimeComplete(uint64_t r) {
    return {F::PhysicalFamily::RuntimeK, static_cast<uint64_t>(F::RuntimeLayout::PacketComplete),
            1, 32, 1, r, 1, 6, 1, 0};
}

void require(bool cond, const std::string& message) {
    if (!cond) throw std::logic_error(message);
}

void verify(DataType dtype, bool staged, bool multi_axis, bool runtime_complete = false,
            CubReductionOp op = CubReductionOp::Sum) {
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(placement), alternate(placement);
    const uint64_t r = staged ? 2048 : (multi_axis ? 64 : 16);
    const std::vector<uint64_t> input_dims = multi_axis
        ? std::vector<uint64_t>{r, 2, 16}
        : std::vector<uint64_t>{r, 32};
    const std::vector<uint32_t> axes = multi_axis
        ? std::vector<uint32_t>{0, 2} : std::vector<uint32_t>{0};
    const uint64_t total_reduction = r * (multi_axis ? 16 : 1);
    C::DenseReductionOutputSpec final_output;
    final_output.value_dtype = DataType::FP32;
    const auto problem = C::makeInitialDenseReductionProblem(
        input_dims, axes, C::DenseReductionAggregateKind::Value, final_output);
    C::DenseProductionPhysicalSelectionContext production(
        dtype, C::denseRKValueOperation(op), stream, total_reduction, 1.0f, op);
    Choice choice{production.context(), staged ? deepR(r, 32) :
        (runtime_complete ? runtimeComplete(r) : complete())};
    C::DensePhysicalSelectionContext injected = choice.production;
    injected.proposal_override = proposal;
    injected.proposal_override_context = &choice;
    const auto plan = C::planDenseReduction(problem, injected);
    require(!plan.steps.empty(), "P2-C planner returned no physical passes");
    const auto* first = dynamic_cast<const C::DenseRetainedFp8RKPhysicalPlan*>(
        plan.steps.front().selected_candidate.physical_plan.get());
    require(first != nullptr, "P2-C first physical pass is not retained FP8 RK");
    require(choice.retained_selections == 1, "P2-C did not choose exactly one original FP8 RK pass");
    if (staged) {
        require(plan.steps.size() >= 2 && first->output_dimensions ==
                (std::vector<uint64_t>{1, 16, 32}), "P2-C staged output is not real FP32 [O,N,K]");
        require(plan.steps[1].input_problem == plan.steps[0].selected_candidate.next_problem,
                "P2-C physical output did not become the next ordinary problem");
    }
    if (multi_axis && staged) {
        require(plan.steps[1].selected_candidate.family == C::DenseReducerFamily::KR,
                "P2-C planner illegally forced hypothetical RK continuation before KR");
    }
    require(!C::hasDenseReductionSites(plan.final_problem), "P2-C ordinary planner left work unfinished");

    Tensor input(placement, TensorDescriptor(dtype, input_dims));
    input.fill(1.0f, stream);
    stream.synchronize(); // The second stamp will run on an independent stream.
    const auto geometry = CubReduction::analyzeGeometry(input_dims, axes);
    Tensor result_a(placement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
    Tensor result_b(placement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
    auto stamp_a = DenseRetainedFp8RKExecutionProbe::stamp(
        input, result_a, axes, plan, total_reduction, stream, op);
    auto stamp_b = DenseRetainedFp8RKExecutionProbe::stamp(
        input, result_b, axes, plan, total_reduction, stream, op);
    require(stamp_a->usesDenseExecutablePlan(),
            "P2-C accidentally constructed the special isolated executor");
    require(stamp_a->getDenseExecutablePassCount() == plan.steps.size(),
            "P2-C has hidden production continuation passes");
    const auto& a = DenseRetainedFp8RKExecutionProbe::passes(*stamp_a);
    const auto& b = DenseRetainedFp8RKExecutionProbe::passes(*stamp_b);
    require(!DenseRetainedFp8RKExecutionProbe::hasLegacyStampScratch(*stamp_a) &&
            !DenseRetainedFp8RKExecutionProbe::hasLegacyStampScratch(*stamp_b),
            "scratch-free stamp still owns the legacy one-byte dummy Tensor");
    size_t accounted_workspace = 0;
    size_t actual_workspace = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        accounted_workspace += a[i].workspaceBytes();
        require(a[i].temp_storage.isInitialized() == (a[i].temp_storage_bytes != 0) &&
                b[i].temp_storage.isInitialized() == (b[i].temp_storage_bytes != 0),
                "a physical pass allocated scratch inconsistent with its query");
        actual_workspace += a[i].intermediate_bytes +
            (a[i].temp_storage.isInitialized() ? a[i].temp_storage.getArraySizeInBytes() : 0);
        if (a[i].temp_storage_bytes)
            require(a[i].temp_storage.getMemPtr<void>() != b[i].temp_storage.getMemPtr<void>(),
                    "P2-C independent stamps share a physical scratch buffer");
    }
    require(accounted_workspace == stamp_a->getWorkspaceSizeInBytes() &&
            actual_workspace == accounted_workspace,
            "P2-C flat physical-pass workspace accounting is inconsistent with allocations");
    require(dynamic_cast<const C::DenseRetainedFp8RKPhysicalPlan*>(a[0].physical_plan.get()) != nullptr &&
            a[0].temp_storage_bytes == 0 && a[0].output.getDataType() == DataType::FP32,
            "P2-C retained first pass does not have its own FP32 output and zero scratch");
    if (staged) {
        require(a[0].output.getDimensions() == first->output_dimensions &&
                a[0].intermediate_bytes == 16 * 32 * sizeof(float),
                "P2-C staged physical output is misallocated or double counted");
    }
    if (a.size() > 1) {
        require(a[0].output.getMemPtr<void>() != b[0].output.getMemPtr<void>(),
                "P2-C separate stamps share a mutable intermediate allocation");
        require(dynamic_cast<const C::DenseRetainedFp8RKPhysicalPlan*>(a[1].physical_plan.get()) == nullptr,
                "P2-C continuation is not independently selected FP32");
    }
    const unsigned proposals_before_run = choice.retained_selections + choice.other_selections;
    stamp_a->runOn(stream);
    stamp_b->runOn(alternate);
    stream.synchronize();
    alternate.synchronize();
    require(proposals_before_run == choice.retained_selections + choice.other_selections,
            "P2-C selected a kernel during runOn");
    // With input fixed to +1, all operations have a closed-form reference.
    // In particular MEAN divides by the *original* domain exactly once and
    // L2 applies its final square root after all staged FP32 passes.
    const float expected = [&] {
        switch (op) {
            case CubReductionOp::Mean:
            case CubReductionOp::Product:
            case CubReductionOp::Min:
            case CubReductionOp::Max: return 1.0f;
            case CubReductionOp::L2Norm:
                return std::sqrt(static_cast<float>(total_reduction));
            case CubReductionOp::Sum:
            case CubReductionOp::L1Norm:
            case CubReductionOp::SumSquares: return static_cast<float>(total_reduction);
        }
        throw std::logic_error("Unknown physical-pass VALUE operation");
    }();
    for (const auto& result : {result_a, result_b}) {
        std::vector<float> actual(result.getTotalNumElements());
        const cudaError_t err = cudaMemcpy(actual.data(), result.getMemPtr<void>(),
                                            actual.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        for (float value : actual)
            require(std::isfinite(value) && std::abs(value - expected) < 0.1f,
                    "P2-C ordinary physical pass list produced an incorrect value");
    }
    // Verify the runtime float scale is applied once on the final pass for
    // both Complete families as well as a staged FP32 continuation.
    stamp_a->runOn(stream, -0.5f);
    stream.synchronize();
    std::vector<float> scaled_actual(result_a.getTotalNumElements());
    const auto scaled_status = cudaMemcpy(scaled_actual.data(), result_a.getMemPtr<void>(),
                                         scaled_actual.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (scaled_status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(scaled_status));
    for (float value : scaled_actual)
        require(std::isfinite(value) &&
                std::abs(value + expected * 0.5f) < 0.1f,
                "P2-C scaled physical pass returned an incorrect result");
    std::cout << "P2-C dtype=" << static_cast<unsigned>(dtype)
              << " first=" << (staged ? "staged" : (runtime_complete ? "runtime_complete" : "complete"))
              << " multi_axis=" << multi_axis
              << " op=" << static_cast<unsigned>(op)
              << " passes=" << a.size()
              << " workspace=" << stamp_a->getWorkspaceSizeInBytes() << " passed\n";
    for (size_t i = 0; i < a.size(); ++i) {
        const auto& pass = a[i];
        const bool retained = dynamic_cast<const C::DenseRetainedFp8RKPhysicalPlan*>(
            pass.physical_plan.get()) != nullptr;
        std::cout << "  physical_pass=" << i << " kind=" << (retained ? "retained_fp8_rk" : "production_fp32")
                  << " input_dtype=" << static_cast<unsigned>(pass.input.getDataType())
                  << " output_dtype=" << static_cast<unsigned>(pass.output.getDataType())
                  << " output_values=" << pass.output.getTotalNumElements()
                  << " intermediate_bytes=" << pass.intermediate_bytes
                  << " scratch_bytes=" << pass.temp_storage_bytes << '\n';
    }
}
}  // namespace

int main() {
    try {
        for (auto dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            verify(dtype, false, false);
            verify(dtype, true, false);
            verify(dtype, false, true);
            verify(dtype, true, true);
            verify(dtype, false, false, true);  // RuntimeK PacketComplete
            // Deterministic DeepRSmallK staged first stages and independently
            // selected FP32 completions for the newly enabled semantics.
            for (CubReductionOp op : {CubReductionOp::Mean, CubReductionOp::Product,
                                      CubReductionOp::L2Norm, CubReductionOp::Min,
                                      CubReductionOp::Max, CubReductionOp::L1Norm,
                                      CubReductionOp::SumSquares}) {
                verify(dtype, true, false, false, op);
                verify(dtype, true, true, false, op);
            }
        }
        std::cout << "P2-C ordinary stamped FP8 physical-pass execution smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P2-C ordinary stamped FP8 physical-pass execution failed: " << error.what() << '\n';
        return 1;
    }
}
