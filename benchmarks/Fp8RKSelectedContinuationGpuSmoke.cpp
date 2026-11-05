// GPU smoke for the P1-C benchmark-only chain. The live FP8 autotuner and
// ordinary production dispatcher are intentionally not changed by P1-C.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"

#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
namespace C = ThorImplementation::CubReductionInternal;
namespace F = ThorImplementation::CubReductionInternal::Fp8RK;

int main() {
    try {
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
        Stream stream(placement);
        constexpr uint64_t R = 2048, K = 32;
        // Exact DeepRSmallK identity from the existing isolated-site host smoke.
        const F::PhysicalKernel first{F::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0,
                                      6, 16, 2048};
        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            C::DenseReductionOutputSpec spec;
            spec.value_dtype = DataType::FP32;
            const auto problem = C::makeInitialDenseReductionProblem(
                {1, R, K}, {1}, C::DenseReductionAggregateKind::Value, spec);
            const C::DenseReductionSite nominated = C::enumerateDenseReductionSites(problem).front();
            const auto semantics = C::makeValueReductionStageSemantics(CubReductionOp::Sum,
                                                                         C::DensePassRole::Complete, R);
            auto site = F::planIsolatedRKSite(problem, nominated, first, semantics);
            // 1229: with the original semantics the retained FP32 continuation selectors need (1225 / 1227).
            C::DenseProductionPhysicalSelectionContext selector(dtype, C::DenseRKValueOperation::Sum, stream, R, 1.0f,
                                                                CubReductionOp::Sum);
            const auto chain = F::selectIsolatedFp32Continuation(problem, nominated, site, selector.context());
            if (chain.fp32_passes.empty() ||
                chain.first_stage_output_problem.runs[nominated.run_index].extent != 16 ||
                C::hasDenseReductionSites(chain.completed_site_problem))
                throw std::logic_error("Official FP32 selection failed to finish staged candidate");
            const auto bench = F::makeSelectedIsolatedBenchmark(
                problem, nominated, std::move(site), first, selector.context(),
                1.0f, uint64_t{256} * 1024 * 1024, stream);
            if (bench.scratch_bytes < 2048 || bench.continuation.size() != chain.fp32_passes.size())
                throw std::logic_error("Selected physical-pass benchmark scratch accounting is incomplete");
            Tensor input(placement, TensorDescriptor(dtype, {1, R, K}));
            input.fill(1.0f, stream);
            Tensor output(placement, TensorDescriptor(DataType::FP32, {1, K}));
            Tensor scratch(placement, TensorDescriptor(DataType::UINT8, {bench.scratch_bytes}));
            // Launch twice without selecting, allocation or GPU timing inside run.
            bench.launch(input, output, scratch, stream);
            bench.launch(input, output, scratch, stream);
            std::vector<float> actual(K);
            const auto status = cudaMemcpyAsync(actual.data(), output.getMemPtr<void>(),
                                                 actual.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                                 stream.getStream());
            if (status != cudaSuccess) throw std::runtime_error("P1-C benchmark result download failed");
            stream.synchronize();
            for (const float value : actual)
                if (!std::isfinite(value) || std::abs(value - static_cast<float>(R)) > 0.01f)
                    throw std::logic_error("Selected FP32 physical continuation returned an incorrect sum");
            std::cout << "P1-C dtype=" << static_cast<int>(dtype)
                      << " continuation_passes=" << bench.continuation.size()
                      << " scratch_bytes=" << bench.scratch_bytes << " passed\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "P1-C selected continuation GPU smoke failed: " << error.what() << '\n';
        return 1;
    }
}
