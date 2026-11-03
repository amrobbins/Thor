// Host-only P2-A contract: descriptor and Tensor requests use the same
// allocation-free key, while the cache never invokes the miss callback on a hit.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKDescriptorSelection.h"

#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace ThorImplementation;
namespace C = ThorImplementation::CubReductionInternal;
namespace RK = ThorImplementation::CubReductionInternal::Fp8RK;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::logic_error(message);
}
RK::Fp8RKSelection recipe() {
    RK::Fp8RKSelection result;
    result.physical.family = RK::PhysicalFamily::RuntimeK;
    result.candidate_count = 28;
    result.winner_index = 15;
    result.winner_name = "runtime_k#15";
    return result;
}
C::DenseReductionProblem problem() {
    C::DenseReductionProblem p;
    p.runs = {{C::DenseReductionRunKind::Retained, 16, 0},
              {C::DenseReductionRunKind::OriginalReduction, 512, 0},
              {C::DenseReductionRunKind::Retained, 64, 0}};
    p.final_output.value_dtype = DataType::FP32;
    return p;
}
} // namespace

int main() {
    try {
        constexpr auto operation = static_cast<CubReductionOp>(0); // public CubReductionOp::Sum
        constexpr uint64_t budget = uint64_t{256} * 1024 * 1024;
        RK::Fp8RKDeviceSelectionIdentity device;
        device.uuid[0] = 0x51;
        device.uuid[15] = 0xA7;
        device.ordinal = 0;
        device.sm_count = 170;
        device.compute_major = 12;
        device.compute_minor = 0;
        auto p = problem();
        auto key = RK::makeFp8RKDescriptorSelectionKey(
            p, {1}, operation, 512, 1.0f, DataType::FP8_E4M3, device, budget);
        const auto equivalent = RK::makeFp8RKDescriptorSelectionKey(
            p, {1}, operation, 512, 1.0f, DataType::FP8_E4M3, device, budget);
        require(key == equivalent, "Equal descriptor requests must have identical keys");
        require(key.device_uuid == device.uuid && key.device_ordinal == device.ordinal &&
                key.sm_count == 170 && key.compute_major == 12 && key.compute_minor == 0,
                "Device identity must be preserved in key");
        require(key.input_dtype == 22 && key.output_dtype == 11 &&
                key.output_scale_bits == std::bit_cast<uint32_t>(1.0f) &&
                key.site_run_index == 1 && key.runs.size() == 3 &&
                key.runs[1].kind == static_cast<uint8_t>(C::DenseReductionRunKind::OriginalReduction),
                "Full descriptor semantics must be preserved in key");

        C::DenseReductionSelectionCache cache;
        unsigned constructed_representatives = 0;
        auto first = cache.getOrTune<RK::Fp8RKSelection>(RK::toDenseReductionSelectionKey(key), [&] {
            ++constructed_representatives;
            return recipe();
        });
        auto cached = cache.getOrTune<RK::Fp8RKSelection>(RK::toDenseReductionSelectionKey(equivalent), [&]() -> RK::Fp8RKSelection {
            ++constructed_representatives;
            throw std::logic_error("Cache hit must never construct benchmark input or select continuation");
        });
        require(constructed_representatives == 1 && cache.tuningAttempts() == 1 &&
                cached.physical == first.physical && cache.size() == 1,
                "Cache hit must reuse immutable first-stage selection without preparations");

        auto distinct = key;
        distinct.input_dtype = static_cast<uint8_t>(DataType::FP8_E5M2);
        require(!(distinct == key), "FP8 dtypes must have distinct keys");
        distinct = key;
        distinct.original_reduction_size++;
        require(!(distinct == key), "Full mathematical reduction size must be in key");
        distinct = key;
        distinct.output_scale_bits = std::bit_cast<uint32_t>(2.0f);
        require(!(distinct == key), "Output scale must be in key");
        distinct = key;
        distinct.workspace_budget_bytes++;
        require(!(distinct == key), "Workspace budget must be in key");
        distinct = key;
        distinct.device_uuid[0]++;
        require(!(distinct == key), "GPU UUID must be in key");
        distinct = key;
        distinct.site_run_index++;
        require(!(distinct == key), "Nominated site must be in key");
        distinct = key;
        distinct.runs.front().extent++;
        require(!(distinct == key), "Original problem runs must be in key");
        p.finalizer_pending = false;
        distinct = RK::makeFp8RKDescriptorSelectionKey(
            p, {1}, operation, 512, 1.0f, DataType::FP8_E4M3, device, budget);
        require(!(distinct == key), "Finalizer semantics must be in key");

        std::cout << "FP8 descriptor selection host smoke passed (key, hit, ownership)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FP8 descriptor selection host smoke failed: " << error.what() << '\n';
        return 1;
    }
}
