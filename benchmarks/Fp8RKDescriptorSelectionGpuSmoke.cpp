// P2-A integration: real descriptor-only miss; Tensor and descriptor hits
// share exactly one cached first-stage recipe, for both FP8 storage formats.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectionCache.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ThorImplementation;
namespace C = ThorImplementation::CubReductionInternal;
namespace RK = ThorImplementation::CubReductionInternal::Fp8RK;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::logic_error(message);
}
} // namespace

int main() {
    try {
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
        Stream stream(placement);
        constexpr uint64_t O = 16, R = 512, K = 64;
        constexpr uint64_t budget = uint64_t{256} * 1024 * 1024;
        C::DenseReductionOutputSpec final_output;
        final_output.value_dtype = DataType::FP32;
        const auto problem = C::makeInitialDenseReductionProblem(
            {O, R, K}, {1}, C::DenseReductionAggregateKind::Value, final_output);
        const auto site = C::enumerateDenseReductionSites(problem).at(0);

        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            // Assert that the production context carries complete semantics
            // for the official physical selector and FP32 continuation policy.
            C::DenseProductionPhysicalSelectionContext context(
                dtype, C::DenseRKValueOperation::Sum, stream, R, 1.0f, CubReductionOp::Sum, budget);
            require(context.context().original_value_operation == CubReductionOp::Sum &&
                    context.context().original_reduction_size == R &&
                    context.context().output_scale == 1.0f &&
                    context.context().max_workspace_bytes == budget,
                    "Production selector context lost the original VALUE semantics");

            RK::fp8RKSelectionCache().clear();
            const auto before = RK::fp8RKSelectionCache().tuningAttempts();
            const auto descriptor_miss = RK::selectCachedIsolatedRKSite(
                problem, site, CubReductionOp::Sum, R, 1.0f, dtype, stream, budget);
            require(RK::fp8RKSelectionCache().tuningAttempts() == before + 1 &&
                    RK::fp8RKSelectionCache().size() == 1 &&
                    descriptor_miss.candidate_count > 0 && descriptor_miss.candidate_count <= 28,
                    "Descriptor-only first miss did not tune/cache exactly once");
            const auto descriptor_hit = RK::selectCachedIsolatedRKSite(
                problem, site, CubReductionOp::Sum, R, 1.0f, dtype, stream, budget);
            require(descriptor_hit.physical == descriptor_miss.physical &&
                    RK::fp8RKSelectionCache().tuningAttempts() == before + 1,
                    "Descriptor cache hit changed physical recipe or retuned");

            Tensor real_input(placement, TensorDescriptor(dtype, {O, R, K}));
            real_input.fill(1.0, stream);
            const auto tensor_hit = RK::selectCachedIsolatedRKSite(
                problem, site, CubReductionOp::Sum, R, 1.0f, real_input, stream, budget);
            require(tensor_hit.physical == descriptor_miss.physical &&
                    tensor_hit.winner_index == descriptor_miss.winner_index &&
                    RK::fp8RKSelectionCache().tuningAttempts() == before + 1,
                    "Tensor and descriptor overloads did not reuse identical cache key");

            bool rejected = false;
            try {
                (void)RK::selectCachedIsolatedRKSite(
                    problem, site, CubReductionOp::Sum, 0, 1.0f, dtype, stream, budget);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Invalid descriptor must be rejected even if a related key is cached");
            rejected = false;
            try {
                (void)RK::selectCachedIsolatedRKSite(
                    problem, site, CubReductionOp::Sum, R, std::numeric_limits<float>::quiet_NaN(),
                    dtype, stream, budget);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Nonfinite scale must be rejected");
            rejected = false;
            try {
                (void)RK::selectCachedIsolatedRKSite(
                    problem, site, CubReductionOp::Sum, R, 1.0f,
                    DataType::FP32, stream, budget);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Non-FP8 descriptor must be rejected");

            std::cout << "P2-A dtype=" << static_cast<int>(dtype)
                      << " candidates=" << descriptor_miss.candidate_count
                      << " winner=" << descriptor_miss.winner_name
                      << " cache_misses=1 passed\n";
        }
        std::cout << "FP8 descriptor selection GPU smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FP8 descriptor selection GPU smoke failed: " << error.what() << '\n';
        return 1;
    }
}
