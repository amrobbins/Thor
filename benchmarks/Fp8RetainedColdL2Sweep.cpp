// Retained-only FP8 dense RK cold-L2 census. No experimental CUDA kernels.
// This is deliberately separate from production stamping and from the older
// calibration benchmark that was deleted in the cleanup.
#include "benchmarks/Fp8BenchmarkArenaCursor.h"
#include "benchmarks/ColdTiming.h"
#include "benchmarks/Fp8OracleBatchQueue.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKArgCandidates.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKCandidates.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKArgCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCarriedArgCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKBoundSelection.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8GeneralRParallelSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKRuntimeProposals.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>
#include <cxxabi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>
#include <variant>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <tuple>
#include <utility>
#include <vector>

namespace T = ThorImplementation;
namespace C = ThorImplementation::CubReductionInternal;
namespace F = ThorImplementation::CubReductionInternal::Fp8RK;
namespace G = ThorImplementation::CubReductionInternal::Fp8RK::Fp8GeneralRParallelPolicy;
namespace B = ThorImplementation::CubReductionBenchmarking;
namespace OW = ThorImplementation::CubReductionInternal::Fp8RK::Fp8OneWindowPolicy;
namespace DR = ThorImplementation::CubReductionInternal::Fp8RK::Fp8DeepRSmallKPolicy;
namespace AK = ThorImplementation::CubReductionInternal::Fp8RK::Fp8AlignedKPolicy;
namespace RK = ThorImplementation::CubReductionInternal::Fp8RK::Fp8RuntimeKPolicy;

namespace {
void check(cudaError_t s, const char* what) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
}
constexpr uint64_t MiB = 1ULL << 20;
constexpr int Passes = 5;
struct Op {
    T::CubReductionOp operation;
    C::CubReductionStageCombine combine;
    C::CubReductionStageInputTransform transform;
    const char* name;
};
constexpr std::array<Op, 12> ops{{
    {T::CubReductionOp::Sum,C::CubReductionStageCombine::Sum,C::CubReductionStageInputTransform::Identity,"sum_identity"},
    {T::CubReductionOp::Sum,C::CubReductionStageCombine::Sum,C::CubReductionStageInputTransform::AbsoluteValue,"sum_abs"},
    {T::CubReductionOp::Sum,C::CubReductionStageCombine::Sum,C::CubReductionStageInputTransform::Square,"sum_square"},
    {T::CubReductionOp::Product,C::CubReductionStageCombine::Product,C::CubReductionStageInputTransform::Identity,"product_identity"},
    {T::CubReductionOp::Product,C::CubReductionStageCombine::Product,C::CubReductionStageInputTransform::AbsoluteValue,"product_abs"},
    {T::CubReductionOp::Product,C::CubReductionStageCombine::Product,C::CubReductionStageInputTransform::Square,"product_square"},
    {T::CubReductionOp::Min,C::CubReductionStageCombine::Minimum,C::CubReductionStageInputTransform::Identity,"min_identity"},
    {T::CubReductionOp::Min,C::CubReductionStageCombine::Minimum,C::CubReductionStageInputTransform::AbsoluteValue,"min_abs"},
    {T::CubReductionOp::Min,C::CubReductionStageCombine::Minimum,C::CubReductionStageInputTransform::Square,"min_square"},
    {T::CubReductionOp::Max,C::CubReductionStageCombine::Maximum,C::CubReductionStageInputTransform::Identity,"max_identity"},
    {T::CubReductionOp::Max,C::CubReductionStageCombine::Maximum,C::CubReductionStageInputTransform::AbsoluteValue,"max_abs"},
    {T::CubReductionOp::Max,C::CubReductionStageCombine::Maximum,C::CubReductionStageInputTransform::Square,"max_square"}}};
// ARG: production FP8 RK argmin/argmax (family arg_production): since ARG-A2 the stamped production plan
// (planDenseReduction + the retained ARG selection), timed with StampedCubArgReduction::runOnInputForBenchmark so every
// measurement reads a fresh cold input view without re-stamping; configuration = the planned passes. With
// --onewindow-runtime-k every OneWindow ARG twin is also timed directly (family one_window_arg). Compare with VALUE
// min_identity/max_identity (arg_value_compare.py).
struct ArgOp {
    T::CubArgReductionOp operation;
    const char* name;
};
constexpr std::array<ArgOp, 2> arg_ops{{
    {T::CubArgReductionOp::ArgMin,"argmin"},
    {T::CubArgReductionOp::ArgMax,"argmax"}}};
bool knownOperation(std::string_view name) {
    return std::any_of(ops.begin(), ops.end(), [&](const Op& op) { return name == op.name; }) ||
           std::any_of(arg_ops.begin(), arg_ops.end(), [&](const ArgOp& op) { return name == op.name; });
}
struct Settings {
    std::string family;
    // Also time every OneWindow unit directly (family one_window_rk), not only the selector's OneWindow variants. Since
    // P5-J2 family one_window runs the same bucketed kernels through the selector.
    bool onewindow_runtime_k = false;
    // Optional cell filters (empty: everything). --focus=onewindow sets all of them.
    std::vector<std::string> operations;
    std::vector<T::DataType> dtypes{T::DataType::FP8_E4M3, T::DataType::FP8_E5M2};
    std::vector<uint64_t> target_mib{24, 96, 192};
    bool onewindow_shapes = false;
    bool uncovered_shapes = false;  // replace the grid with uncoveredKShapes()
    bool replacement_shapes = false;  // replace the grid with replacementShapes()
    bool all_shapes = false;  // replace the grid with allShapes() (grid + uncovered K + K = 2)
    bool arg_produce_value = true;  // --arg-outputs=index: ARG writes only the UINT32 index
    uint64_t arena_bytes = 12ULL << 30;
    uint64_t max_cells = std::numeric_limits<uint64_t>::max();
    bool historical_fp8_output = false;
    bool targets_given = false;  // --targets-mib was passed
    // 1133 --training: the ranker training dataset (runTraining).
    bool training = false;
    bool continuation_audit = false;  // 1133b --continuation-audit
    bool dtypes_given = false;  // --dtypes was passed
    uint64_t training_seed = 1133;
    uint64_t training_random = 400;  // log-uniform random geometries on top of allShapes()
    uint32_t training_semantics_per_geometry = 2;  // even: VALUE, odd: ARG
    uint32_t training_heldout_percent = 10;  // of (R, K) shapes
    bool training_stamp = false;  // 1162 --stamp: time production's own contest at every --training cell
    // 1211 --carried-index=uint32|uint64 (with --training --dtypes=fp32 --ops=argmin,argmax): the ARG cells are
    // carried-index (intermediate) sites, timing the carried FP32 RK ARG passes over values plus a carried index.
    std::optional<T::DataType> carried_index;
    // 1164 --training timing (benchmarks/ColdTiming.h, the KR sweep's method): per-candidate launches per sample
    // from a probe (about interval_us each, at most max_launches) and rotating output views.
    double interval_us = 1000.0;
    uint64_t max_launches = 256;
    uint64_t output_ring_bytes = 3ULL << 30;
};
std::vector<std::string> splitList(std::string_view text) {
    std::vector<std::string> items;
    while (!text.empty()) {
        const size_t comma = text.find(',');
        if (comma != 0) items.emplace_back(text.substr(0, comma));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return items;
}

Settings parse(int argc, char** argv) {
    Settings s;
    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        if (a.starts_with("--family=")) s.family = std::string(a.substr(9));
        else if (a.starts_with("--max-cells=")) s.max_cells = std::stoull(std::string(a.substr(12)));
        else if (a.starts_with("--arena-gib=")) s.arena_bytes = std::stoull(std::string(a.substr(12))) << 30;
        else if (a == "--output=same_fp8") s.historical_fp8_output = true;
        else if (a == "--output=fp32") s.historical_fp8_output = false;
        else if (a == "--onewindow-runtime-k") s.onewindow_runtime_k = true;
        else if (a.starts_with("--ops=")) s.operations = splitList(a.substr(6));
        else if (a.starts_with("--dtypes=")) {
            s.dtypes.clear();
            s.dtypes_given = true;
            for (const auto& d : splitList(a.substr(9))) {
                if (d == "e4m3") s.dtypes.push_back(T::DataType::FP8_E4M3);
                else if (d == "e5m2") s.dtypes.push_back(T::DataType::FP8_E5M2);
                else if (d == "fp32") s.dtypes.push_back(T::DataType::FP32);  // 1194 / 1205: --training only
                else if (d == "fp16") s.dtypes.push_back(T::DataType::FP16);  // 1247: --training only
                else if (d == "bf16") s.dtypes.push_back(T::DataType::BF16);  // 1247: --training only
                else throw std::invalid_argument("Unknown dtype (e4m3|e5m2|fp32|fp16|bf16): " + d);
            }
        } else if (a.starts_with("--targets-mib=")) {
            s.target_mib.clear();
            s.targets_given = true;
            for (const auto& t : splitList(a.substr(14))) s.target_mib.push_back(std::stoull(t));
        } else if (a == "--shapes=onewindow") s.onewindow_shapes = true;
        else if (a == "--shapes=uncovered") s.uncovered_shapes = true;
        else if (a == "--shapes=replacement") s.replacement_shapes = true;
        else if (a == "--shapes=all") s.all_shapes = true;
        else if (a == "--training") s.training = true;
        else if (a == "--continuation-audit") s.continuation_audit = true;
        else if (a.starts_with("--training-seed=")) s.training_seed = std::stoull(std::string(a.substr(16)));
        else if (a.starts_with("--training-random=")) s.training_random = std::stoull(std::string(a.substr(18)));
        else if (a.starts_with("--training-semantics=")) {
            s.training_semantics_per_geometry = static_cast<uint32_t>(std::stoul(std::string(a.substr(21))));
            if (s.training_semantics_per_geometry == 0) throw std::invalid_argument("--training-semantics must be >= 1");
        } else if (a.starts_with("--training-heldout-percent=")) {
            s.training_heldout_percent = static_cast<uint32_t>(std::stoul(std::string(a.substr(27))));
            if (s.training_heldout_percent > 100) throw std::invalid_argument("--training-heldout-percent must be <= 100");
        }
        else if (a == "--stamp") s.training_stamp = true;
        else if (a == "--carried-index=uint32") s.carried_index = T::DataType::UINT32;
        else if (a == "--carried-index=uint64") s.carried_index = T::DataType::UINT64;
        else if (a.starts_with("--interval-us=")) s.interval_us = std::stod(std::string(a.substr(14)));
        else if (a.starts_with("--max-launches=")) s.max_launches = std::stoull(std::string(a.substr(15)));
        else if (a.starts_with("--output-ring-gib=")) s.output_ring_bytes = std::stoull(std::string(a.substr(18))) << 30;
        else if (a == "--arg-outputs=value_index") s.arg_produce_value = true;
        else if (a == "--arg-outputs=index") s.arg_produce_value = false;
        else if (a == "--focus=arg") {
            // ARG-A0 baseline: today's production FP8 argmin/argmax next to VALUE min/max (the line-rate reference)
            // on every RK shape the sweep knows (the grid, the uncovered-K shapes and K = 2), one format, all
            // three working-set sizes. Compare with arg_value_compare.py.
            s.all_shapes = true;
            s.dtypes = {T::DataType::FP8_E4M3};
            s.operations = {"argmin", "argmax", "min_identity", "max_identity"};
            s.target_mib = {24, 96, 192};
            s.onewindow_runtime_k = true;  // OneWindow ARG twins next to the VALUE OneWindow units
        }
        else if (a == "--focus=replacement") {
            // P5-J: bucketed OneWindow as the only OneWindow. The OneWindow-relevant grid shapes, the uncovered-K
            // shapes and K = 2 (absent from the grid), all three working-set sizes, with the --focus=onewindow ops,
            // format and families. Compare with onewindow_runtime_k_compare.py --replace-all.
            s.replacement_shapes = true;
            s.dtypes = {T::DataType::FP8_E4M3};
            s.operations = {"sum_identity", "product_identity", "max_identity", "sum_square"};
            s.target_mib = {24, 96, 192};
            s.onewindow_runtime_k = true;
        }
        else if (a == "--focus=uncovered") {
            // Bucketed OneWindow validation at K outside the compiled OneWindow set: the uncovered-K shape list
            // (instead of the grid) with the --focus=onewindow ops, format, sizes and families.
            s.uncovered_shapes = true;
            s.dtypes = {T::DataType::FP8_E4M3};
            s.operations = {"sum_identity", "product_identity", "max_identity", "sum_square"};
            s.target_mib = {96, 192};
            s.onewindow_runtime_k = true;
        }
        else if (a == "--focus=onewindow") {
            // OneWindow work: the OneWindow-relevant shapes, one FP8 format (both formats measured identically in
            // every P5 comparison), one kernel per physical operation plus the runtime-transform path, the two
            // larger working-set sizes (24 MiB cells are ~15 us and noise-bound), all families (the K-coverage
            // comparison needs the best production candidate), and the bucketed OneWindow configurations.
            s.onewindow_shapes = true;
            s.dtypes = {T::DataType::FP8_E4M3};
            s.operations = {"sum_identity", "product_identity", "max_identity", "sum_square"};
            s.target_mib = {96, 192};
            s.onewindow_runtime_k = true;
        }
        else if (a == "--help") {
            std::cout << "Usage: thor_fp8_retained_cold_l2_sweep"
                      << " [--family=one_window|deep_r_small_k|aligned_k|runtime_k|one_window_rk]"
                      << " [--output=fp32|same_fp8] [--max-cells=N] [--arena-gib=12] [--onewindow-runtime-k]\n"
                      << "  --onewindow-runtime-k also times every OneWindow unit directly (one_window runs the same\n"
                      << "  bucketed kernels through the selector since P5-J2)\n"
                      << "  (family one_window_rk, configuration oO_wW_pP_cB[_la]); --family=one_window_rk times\n"
                      << "  only those.\n"
                      << "  Cell filters: --ops=sum_identity,max_abs,... --dtypes=e4m3,e5m2 --targets-mib=24,96,192\n"
                      << "  --shapes=onewindow (the OneWindow-relevant grid shapes), --shapes=uncovered (K outside the\n"
                      << "  compiled OneWindow set, R*K <= 64 Ki, instead of the grid).\n"
                      << "  --focus=onewindow = --shapes=onewindow --dtypes=e4m3 --targets-mib=96,192\n"
                      << "  --ops=sum_identity,product_identity,max_identity,sum_square --onewindow-runtime-k\n"
                      << "  --focus=uncovered = the same with --shapes=uncovered.\n"
                      << "  --focus=replacement = OneWindow-relevant grid + uncovered-K + K=2 shapes, 24/96/192 MiB.\n"
                      << "  --shapes=all = grid + uncovered-K + K=2 shapes.\n"
                      << "  ARG: --ops=argmin,argmax (family arg_production, production path, UINT32 index;\n"
                      << "  --arg-outputs=value_index|index, default value_index).\n"
                      << "  --focus=arg = --shapes=all --dtypes=e4m3 --targets-mib=24,96,192\n"
                      << "  --ops=argmin,argmax,min_identity,max_identity --onewindow-runtime-k (one_window_arg units)\n"
                      << "  Ranker training dataset (1133): --training [--training-seed=1133] [--training-random=400]\n"
                      << "  [--training-semantics=2] [--training-heldout-percent=10]; targets default 96,192 MiB.\n"
                      << "  --training --dtypes=fp32 (1194): FP32 cells of the FP32 RK pool, alone (one ranker per\n"
                      << "  dtype); working sets stay in bytes (a quarter of FP8's elements). 1205: VALUE and ARG cells\n"
                      << "  (the FP32 RK ARG candidates), as FP8's.\n"
                      << "  --training --dtypes=fp16,bf16 (1247): FP16 / BF16 cells of the 2-byte RK pool (one\n"
                      << "  ranker for both formats, apart from FP8 and FP32); half FP8's elements per byte. 1251:\n"
                      << "  VALUE and ARG cells (the 2-byte RK ARG candidates), as FP32's.\n"
                      << "  --carried-index=uint32|uint64 (1211, with --training --dtypes=fp32 --ops=argmin,argmax):\n"
                      << "  the ARG cells are carried-index (intermediate) sites: the carried FP32 RK ARG passes\n"
                      << "  (AlignedK, rotated) over FP32 values plus a carried index of that width; the working set\n"
                      << "  counts both.\n"
                      << "  --stamp (with --training, 1162): also run production's own RK contest at every cell (Regular\n"
                      << "  depth) and print '#stamp,cell_id,stamp_ms,candidates,winner'; analyse with fp8_rk_stamp.py.\n"
                      << "  VALUE cells whose semantics no public operation reaches (product/min/max of |x| or x^2)\n"
                      << "  are not stamped.\n"
                      << "  --training timing (1164, the KR sweep's): [--interval-us=1000] [--max-launches=256]\n"
                      << "  [--output-ring-gib=3]: launches per sample from each candidate's probe, rotating outputs.\n"
                      << "  --ops/--dtypes/--targets-mib/--max-cells still filter. See FP8_RETAINED_COLD_L2_SWEEP.md.\n"
                      << "  Continuation audit (1133b): --continuation-audit (default ops sum_identity,max_identity,argmax,\n"
                      << "  e4m3, 96,192 MiB; analyse with fp8_continuation_audit.py).\n";
            std::exit(0);
        } else throw std::invalid_argument("Unknown benchmark option: " + std::string(a));
    }
    if (!s.family.empty() && s.family != "one_window" && s.family != "deep_r_small_k" &&
        s.family != "aligned_k" && s.family != "runtime_k" && s.family != "one_window_rk")
        throw std::invalid_argument("Unknown retained family filter");
    if (s.family == "one_window_rk") s.onewindow_runtime_k = true;
    for (const auto& name : s.operations) {
        if (!knownOperation(name)) throw std::invalid_argument("Unknown operation: " + name);
    }
    if (s.dtypes.empty() || s.target_mib.empty()) throw std::invalid_argument("Empty dtype or target list");
    if (static_cast<int>(s.onewindow_shapes) + static_cast<int>(s.uncovered_shapes) +
            static_cast<int>(s.replacement_shapes) + static_cast<int>(s.all_shapes) > 1)
        throw std::invalid_argument(
            "Choose one of --shapes=onewindow, --shapes=uncovered, --shapes=replacement, --shapes=all");
    if (s.arena_bytes < (1ULL << 30) || s.arena_bytes > (24ULL << 30))
        throw std::invalid_argument("Arena size must be 1..24 GiB");
    // 1194: FP32 cells exist in --training only (the FP32 RK pools), and one ranker per dtype: FP32 alone.
    const bool fp32 = std::find(s.dtypes.begin(), s.dtypes.end(), T::DataType::FP32) != s.dtypes.end();
    if (fp32 && (!s.training || s.dtypes.size() != 1))
        throw std::invalid_argument("--dtypes=fp32 runs with --training only, and alone (one ranker per dtype)");
    // 1247: FP16 / BF16 cells exist in --training only (the 2-byte RK pool), together and apart from FP8 and FP32;
    // 1251: VALUE and ARG cells.
    const auto f16 = [](T::DataType d) { return d == T::DataType::FP16 || d == T::DataType::BF16; };
    if (std::any_of(s.dtypes.begin(), s.dtypes.end(), f16)) {
        if (!s.training || !std::all_of(s.dtypes.begin(), s.dtypes.end(), f16))
            throw std::invalid_argument("--dtypes=fp16,bf16 runs with --training only, apart from e4m3/e5m2/fp32");
        if (s.carried_index) throw std::invalid_argument("--carried-index runs with --dtypes=fp32 only");
    }
    if (s.carried_index) {
        const bool arg_only = !s.operations.empty() &&
            std::all_of(s.operations.begin(), s.operations.end(), [](const std::string& name) {
                return std::any_of(arg_ops.begin(), arg_ops.end(), [&](const ArgOp& op) { return name == op.name; });
            });
        if (!fp32 || !arg_only)
            throw std::invalid_argument("--carried-index runs with --training --dtypes=fp32 and ARG operations only");
    }
    return s;
}

// The same deterministic R/K fast-pruning census used for Thor1012 model
// training and kernel A/B checks. Preserve this as a stable regression grid.
using Shape = std::pair<uint64_t, uint64_t>;
std::vector<Shape> shapes() {
    std::vector<Shape> result{
        {16,16},{64,17},{112,17},{224,32},{352,32},{640,64},{1280,16},{2048,32},
        {4608,13},{7168,32},{112,256},{352,256},{1280,256},{7168,256},{112,512},
        {352,512},{1792,512},{7168,510},{4608,513},{640,840},{4608,896},
        {112,2049},{56,3},{112,8},{144,16},{56,32},
        {8,3},{24,5},{32,7},{48,8},{72,9},{96,11},{128,13},{176,16},
        {192,17},{256,19},{288,31},{384,32},{512,63},{768,64},{1024,127},
        {1536,128},{2304,129},{3072,255},{4096,256},{9216,497},{8192,512},
        {1536,511},{3584,509},{6144,895},{28672,31},{25600,3},
        {10,4},{20,6},{28,10},{36,12},{44,14},{60,15},{84,18},{104,20},
        {120,23},{152,29},{184,33},{216,47},{272,49},{336,59},{416,65},
        {480,73},{560,95},{672,111},{832,131},{960,159},{1152,191},
        {1344,223},{1472,239},{1664,257},{1920,287},{2176,319},{2432,383},
        {2816,447},{3328,495},{3712,507},{4352,521},{4992,575},{5632,639},
        {6656,703},{7552,767},{8704,831},{9984,959},{11264,1023},
        {12800,1151},{14336,1279},{16896,1535},{19456,2047},{22528,3071},
        {24576,4095},{27648,27},{29184,35},{31744,5},{34560,21}};
    constexpr std::array<uint64_t, 36> rs{{4,8,12,16,24,32,40,56,64,80,96,112,
        128,176,224,288,352,448,512,640,768,896,1024,1280,1536,2048,
        3072,4096,6144,8192,10240,14336,18432,24576,32768,49152}};
    constexpr std::array<uint64_t, 28> ks{{3,4,7,8,11,16,17,19,31,32,33,63,64,65,
        127,128,129,255,256,257,497,510,512,513,895,1024,2049,4095}};
    for (uint64_t r : rs) for (uint64_t k : ks) {
        const uint64_t hash = (r * 11400714819323198485ULL) ^ (k * 14029467366897019727ULL);
        if ((hash % 7) == 0) result.emplace_back(r,k);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
// --shapes=uncovered: K values outside the 25 compiled OneWindow K (and up to 384, past which other families win), each
// at the R values that keep the reduced window R*K <= 64 Ki bytes, where OneWindow-type kernels compete.
std::vector<Shape> uncoveredKShapes() {
    constexpr std::array<uint64_t, 25> ks{{6,10,12,14,18,20,21,24,27,28,33,40,48,56,65,80,96,100,111,144,192,200,
        257,320,384}};
    constexpr std::array<uint64_t, 7> rs{{4,12,32,96,256,768,2048}};
    std::vector<Shape> result;
    for (uint64_t k : ks) for (uint64_t r : rs) if (r * k <= 65536) result.emplace_back(r, k);
    return result;  // 148 shapes
}
// --shapes=onewindow: grid (R, K) shapes where the best compiled or bucketed OneWindow configuration came within 10%
// of the best configuration for at least one cell of the full 1092 sweep (134 of 247 shapes; compiled and uncovered K).
bool oneWindowRelevantShape(uint64_t r, uint64_t k) {
    static constexpr std::array<std::pair<uint16_t, uint16_t>, 134> relevant{{
        {8,3},{16,3},{24,3},{56,3},{112,3},{896,3},{1024,3},{14336,3},{25600,3},{10,4},{24,4},{224,4},{768,4},
        {896,4},{1280,4},{18432,4},{24,5},{20,6},{32,7},{288,7},{512,7},{640,7},{2048,7},{8192,7},{49152,7},{48,8},
        {112,8},{288,8},{448,8},{1280,8},{24576,8},{72,9},{28,10},{96,11},{896,11},{36,12},{128,13},{4608,13},
        {44,14},{60,15},{16,16},{112,16},{144,16},{176,16},{352,16},{1280,16},{1536,16},{14336,16},{32,17},{64,17},
        {112,17},{128,17},{176,17},{192,17},{288,17},{84,18},{128,19},{256,19},{104,20},{120,23},{152,29},{64,31},
        {96,31},{288,31},{640,31},{1024,31},{32,32},{56,32},{112,32},{176,32},{224,32},{352,32},{384,32},{2048,32},
        {3072,32},{4096,32},{7168,32},{10240,32},{14336,32},{16,33},{184,33},{216,47},{272,49},{336,59},{24,63},
        {512,63},{768,63},{4,64},{16,64},{224,64},{448,64},{640,64},{768,64},{896,64},{8192,64},{10240,64},{24,65},
        {80,65},{128,65},{560,95},{672,111},{112,127},{1024,127},{1536,127},{32,128},{288,128},{896,128},{1536,128},
        {40,129},{64,129},{176,129},{352,129},{2304,129},{224,255},{640,255},{112,256},{128,256},{176,256},
        {352,256},{640,256},{1280,256},{4096,256},{24,497},{112,497},{128,497},{176,497},{352,497},{512,497},
        {96,512},{112,512},{224,512},{352,512},{1280,512},{1792,512}}};
    return std::any_of(relevant.begin(), relevant.end(), [&](const auto& s) { return s.first == r && s.second == k; });
}
// --shapes=replacement: everything where OneWindow (compiled or bucketed) can matter: the OneWindow-relevant grid
// shapes, the uncovered-K shapes, and K = 2, which the grid lacks. K = 1 is not here: a dense trailing reduction is
// ContiguousFixedSegment, not an RK site.
std::vector<Shape> replacementShapes() {
    std::vector<Shape> result;
    for (const Shape& shape : shapes())
        if (oneWindowRelevantShape(shape.first, shape.second)) result.push_back(shape);
    const auto uncovered = uncoveredKShapes();
    result.insert(result.end(), uncovered.begin(), uncovered.end());
    for (uint64_t r : {4,8,16,32,64,128,256,512,1024,2048,4096}) result.emplace_back(r, 2);
    return result;  // 134 + 148 + 11 shapes
}
// --shapes=all: every RK shape the sweep knows: the grid, the uncovered-K shapes and K = 2, without duplicates.
std::vector<Shape> allShapes() {
    std::vector<Shape> result = shapes();
    const auto uncovered = uncoveredKShapes();
    result.insert(result.end(), uncovered.begin(), uncovered.end());
    for (uint64_t r : {4,8,16,32,64,128,256,512,1024,2048,4096}) result.emplace_back(r, 2);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

uint64_t nearTargetOuter(uint64_t r, uint64_t k, uint64_t bytes) {
    // The historical fast sweep used ceil(target_bytes/(R*K)), not a fixed
    // output cardinality. Retain that choice for geometry/traffic matching.
    return std::max<uint64_t>(1, (bytes + (r*k) / 2) / (r*k));
}
const char* dtypeName(T::DataType dtype) {
    return dtype == T::DataType::FP8_E4M3 ? "fp8_e4m3" : dtype == T::DataType::FP32 ? "fp32"
           : dtype == T::DataType::FP16   ? "fp16" : dtype == T::DataType::BF16 ? "bf16" : "fp8_e5m2";
}
const char* familyName(F::PhysicalFamily f) {
    switch (f) {
        case F::PhysicalFamily::OneWindow:return "one_window";
        case F::PhysicalFamily::DeepRSmallK:return "deep_r_small_k";
        case F::PhysicalFamily::AlignedK:return "aligned_k";
        case F::PhysicalFamily::RuntimeK:return "runtime_k";
    }
    return "unknown";
}
std::string physicalName(const F::PhysicalKernel& p) {
    switch (p.family) {
        case F::PhysicalFamily::OneWindow:return "o"+std::to_string(p.a)+"_w"+std::to_string(p.b)+"_p"+std::to_string(p.c);
        case F::PhysicalFamily::DeepRSmallK:return "p16_rows"+std::to_string(p.a);
        case F::PhysicalFamily::AlignedK:return "p"+std::to_string(p.a)+"_cta"+std::to_string(p.b)+"_shards"+std::to_string(p.c);
        case F::PhysicalFamily::RuntimeK:
            return std::string(p.a == uint64_t(F::RuntimeLayout::PacketComplete) ? "packet_complete" :
                p.a == uint64_t(F::RuntimeLayout::RotatedExact) ? "rotated_exact" : "rotated_prefix")+
                "_w"+std::to_string(p.b)+"_tile"+std::to_string(p.c)+"_rows"+std::to_string(p.e);
    }
    return "unknown";
}
F::PhysicalKernel bind(const G::CandidatePlan& c) {
    F::PhysicalKernel k{};
    k.family = static_cast<F::PhysicalFamily>(c.selection.mode);
    k.resident_ctas_per_sm = c.resident_ctas_per_sm;
    k.first_stage_ctas = c.first_stage_work_groups;
    k.first_stage_scratch_bytes = c.scratch_bytes;
    switch (c.selection.mode) {
        case G::Mode::OneWindow: {
            const auto v = std::get<G::OneWindowConfiguration>(c.selection.configuration).variant;
            k.a=v.outputs_per_warp; k.b=v.warps_per_output; k.c=v.packet_bytes; break;
        }
        case G::Mode::DeepRSmallK:
            k.a=std::get<G::DeepRSmallKConfiguration>(c.selection.configuration).variant.rows_per_shard; break;
        case G::Mode::AlignedK: {
            const auto v=std::get<G::AlignedKConfiguration>(c.selection.configuration).variant;
            k.a=v.packet; k.b=v.threads; k.c=v.shards_per_output; break;
        }
        case G::Mode::RuntimeK: {
            const auto v=std::get<G::RuntimeKConfiguration>(c.selection.configuration).variant;
            k.a=v.layout == RK::Layout::PacketComplete ? uint64_t(F::RuntimeLayout::PacketComplete) :
                v.layout == RK::Layout::RotatedExact ? uint64_t(F::RuntimeLayout::RotatedExact) :
                uint64_t(F::RuntimeLayout::RotatedPrefix);
            k.b=v.warps_per_output; k.c=v.tile_components; k.d=v.tile_count;
            k.e=v.rows_per_shard; k.f=v.shards_per_output; break;
        }
    }
    return k;
}

class Arena {
public:
    Arena(T::TensorPlacement placement, Stream& stream, uint64_t bytes)
       : bytes_(placement, T::TensorDescriptor(T::DataType::UINT8, {bytes})),
         cursor_(bytes), workspace_(placement, T::TensorDescriptor(T::DataType::UINT8, {2ULL << 30})),
         partials_(placement, T::TensorDescriptor(T::DataType::UINT8, {1ULL << 30})) {
        check(cudaMemsetAsync(bytes_.getMemPtr<void>(), 0x30, bytes, stream.getStream()), "cold arena initialization");
        stream.synchronize();
    }
    T::Tensor input(T::DataType dtype, uint64_t outer, uint64_t r, uint64_t k) {
        const std::vector<uint64_t> dims{outer,r,k};
        const T::TensorDescriptor d(dtype,dims);
        auto view = cursor_.getBenchmarkTensorView(d.getArraySizeInBytes(), B::Fp8BenchmarkArenaCursor::Lifetime::StreamingInput);
        return bytes_.aliasByteStorageAs(dtype, dims, view.offset);
    }
    [[nodiscard]] std::pair<T::Tensor,T::Tensor> outputAndScratch(
        T::DataType output_dtype, uint64_t outer, uint64_t k, uint64_t scratch_bytes) const {
        const std::vector<uint64_t> dims{outer,k};
        const uint64_t output_bytes=T::TensorDescriptor(output_dtype,dims).getArraySizeInBytes();
        const auto align256=[](uint64_t n){return (n+255ULL)&~255ULL;};  // 1163: 256-byte starts
        const uint64_t scratch_offset=align256(output_bytes+128);
        if (scratch_offset > workspace_.getArraySizeInBytes()-128 ||
            scratch_bytes > workspace_.getArraySizeInBytes()-scratch_offset-128)
            throw std::runtime_error("Retained FP8 cold-L2 output/scratch exceeds 2 GiB workspace");
        T::Tensor out=workspace_.aliasByteStorageAs(output_dtype,dims,0);
        T::Tensor scratch;
        if (scratch_bytes) scratch=workspace_.aliasByteStorageAs(
            T::DataType::UINT8,{scratch_bytes},scratch_offset);
        return {std::move(out),std::move(scratch)};
    }
    // ARG outputs: value (when produced) at the start of the workspace, the UINT32 index after it, then scratch, with
    // the same 256-byte starts (1163; 32 before) and 128-byte gaps as outputAndScratch.
    struct ArgOutputs {
        std::optional<T::Tensor> value, index;
        T::Tensor scratch;
    };
    [[nodiscard]] ArgOutputs argOutputs(std::optional<T::DataType> value_dtype, uint64_t outer, uint64_t k,
                                        uint64_t scratch_bytes) const {
        const std::vector<uint64_t> dims{outer,k};
        const auto align256=[](uint64_t n){return (n+255ULL)&~255ULL;};  // 1163: 256-byte starts
        const uint64_t value_bytes=value_dtype?T::TensorDescriptor(*value_dtype,dims).getArraySizeInBytes():0;
        const uint64_t index_offset=value_dtype?align256(value_bytes+128):0;
        const uint64_t index_bytes=T::TensorDescriptor(T::DataType::UINT32,dims).getArraySizeInBytes();
        const uint64_t scratch_offset=align256(index_offset+index_bytes+128);
        if (scratch_offset > workspace_.getArraySizeInBytes()-128 ||
            scratch_bytes > workspace_.getArraySizeInBytes()-scratch_offset-128)
            throw std::runtime_error("Retained FP8 cold-L2 ARG outputs/scratch exceed 2 GiB workspace");
        ArgOutputs result;
        if (value_dtype) result.value=workspace_.aliasByteStorageAs(*value_dtype,dims,0);
        result.index=workspace_.aliasByteStorageAs(T::DataType::UINT32,dims,index_offset);
        if (scratch_bytes) result.scratch=workspace_.aliasByteStorageAs(T::DataType::UINT8,{scratch_bytes},scratch_offset);
        return result;
    }
    // ARG staged twins: FP32 value + UINT32 index partials [outer, shards, k] in a separate 1 GiB buffer.
    [[nodiscard]] std::pair<T::Tensor,T::Tensor> argPartials(uint64_t outer, uint64_t shards, uint64_t k) const {
        const std::vector<uint64_t> dims{outer,shards,k};
        const uint64_t value_bytes=T::TensorDescriptor(T::DataType::FP32,dims).getArraySizeInBytes();
        const uint64_t index_offset=(value_bytes+128+255)&~255ULL;  // 1163: 256-byte starts
        if (index_offset+value_bytes+128 > partials_.getArraySizeInBytes())
            throw std::runtime_error("ARG partials exceed the 1 GiB partial buffer");
        return {partials_.aliasByteStorageAs(T::DataType::FP32,dims,0),
                partials_.aliasByteStorageAs(T::DataType::UINT32,dims,index_offset)};
    }
    // Packed ARG partials (4 bytes per [outer, shards, k] lane candidate) in the 1 GiB partial buffer; null for 0 bytes.
    [[nodiscard]] void* packedArgScratch(uint64_t bytes) {
        if (bytes == 0) return nullptr;
        if (bytes > partials_.getArraySizeInBytes()) throw std::runtime_error("Packed ARG partials exceed 1 GiB");
        return partials_.getMemPtr<void>();
    }
    // Raw bases of the 1 GiB partial buffer and the 2 GiB output/scratch workspace (--continuation-audit copies).
    [[nodiscard]] uint8_t* rawPartials() { return partials_.getMemPtr<uint8_t>(); }
    [[nodiscard]] uint8_t* rawWorkspace() { return workspace_.getMemPtr<uint8_t>(); }
    uint64_t wraps() const { return cursor_.wraps(); }
private:
    T::Tensor bytes_;
    B::Fp8BenchmarkArenaCursor cursor_;
    T::Tensor workspace_;
    T::Tensor partials_;
};

struct Event {
    cudaEvent_t handle = nullptr;
    Event() { check(cudaEventCreate(&handle),"create timing event"); }
    ~Event() { if (handle) (void)cudaEventDestroy(handle); }
    Event(const Event&)=delete;
    Event& operator=(const Event&)=delete;
    Event(Event&& e) noexcept : handle(std::exchange(e.handle,nullptr)) {}
    Event& operator=(Event&&)=delete;
};
struct Timed {
    std::string family;
    std::optional<F::SelectedIsolatedBenchmark> executable;  // production retained candidate
    std::function<void(const T::Tensor&,T::Tensor&,Stream&)> direct;  // bucketed OneWindow (no continuation)
    std::string name;
    size_t rank = 0;  // 0: not in the learned ranking
    std::array<double,Passes> milliseconds{};
    // 1211: a carried-index pass (values, carried index, outputs); last, so the aggregate initializers above stand.
    std::function<void(const T::Tensor&,const T::Tensor&,T::Tensor&,Stream&)> direct_carried;
};
struct Cell {
    T::Tensor input,output,scratch;
    // 1211: a carried-index cell streams a second cold view, the carried index, next to its values.
    std::optional<T::DataType> carried_dtype;
    T::Tensor carried;
    std::vector<Timed> candidates;
    std::vector<Event> starts, stops;
    // 1164 (--training): rotating output views (empty: the shared `output`) and, per candidate, the launches per sample
    // and the probe time (empty: one launch per sample, as --continuation-audit still times).
    std::vector<T::Tensor> outputs;
    size_t next_output = 0;
    std::vector<uint64_t> launches;
    std::vector<double> probe_ms;
    T::DataType dtype;
    uint64_t outer,r,k;
    std::string operation;
    explicit Cell(T::Tensor input_, T::DataType dtype_, uint64_t o, uint64_t rr, uint64_t kk,
                  std::string op) : input(std::move(input_)), dtype(dtype_),outer(o),r(rr),k(kk),operation(std::move(op)) {}
};

// All callback-proven, structurally legal retained choices. The extended
// 128-place candidate set is measured, not just the learned top 28, so it
// remains possible to detect a missing fastest kernel after cleanup.
std::vector<G::CandidatePlan> retainedChoices(uint64_t outer,uint64_t r,uint64_t k,
    uint32_t sm_count,T::DataType dtype,T::DataType output_dtype,
    const C::CubReductionStageSemantics& semantics, std::string_view family_filter) {
    const auto residence=[](auto&& fn)->uint32_t { try { return fn(); }
        catch(const std::logic_error&) {return 0;} };
    const auto ow = [&](OW::Variant v) {if (!family_filter.empty() && family_filter != "one_window") return 0u;
        return residence([&] {
        return F::oneWindowResidency(dtype,k,semantics.combine,v.outputs_per_warp,v.warps_per_output,v.packet_bytes);});};
    const auto dr = [&](DR::Variant v) {if (!family_filter.empty() && family_filter != "deep_r_small_k") return 0u;
        return residence([&] {
        return F::deepRSmallKResidency(dtype,k,semantics,v.rows_per_shard);});};
    const auto ak = [&](AK::Variant v) {if (!family_filter.empty() && family_filter != "aligned_k") return 0u;
        return residence([&] {
        return F::alignedKResidency(dtype,v.shards_per_output == 1 ? output_dtype : T::DataType::FP32,
            semantics,v.packet,v.threads,v.shards_per_output);});};
    const auto runtime_proposals = (!family_filter.empty() && family_filter != "runtime_k")
        ? std::vector<F::PhysicalKernel>{}
        : F::enumerateRetainedRuntimeK({outer,r,k,sm_count}, [&](const F::PhysicalKernel& p) {
              return residence([&] {return F::runtimeKResidency(dtype,semantics,p);});
          });
    std::vector<RK::Candidate> runtime;
    runtime.reserve(runtime_proposals.size());
    for (const auto& p : runtime_proposals) {
        const auto layout = p.a == uint64_t(F::RuntimeLayout::PacketComplete) ? RK::Layout::PacketComplete :
            p.a == uint64_t(F::RuntimeLayout::RotatedExact) ? RK::Layout::RotatedExact : RK::Layout::RotatedPrefix;
        const RK::Candidate c{{layout,static_cast<uint32_t>(p.b),static_cast<uint32_t>(p.c),
                               static_cast<uint32_t>(p.d),p.e,p.f},
                               p.first_stage_ctas,p.first_stage_scratch_bytes,p.resident_ctas_per_sm};
        if (RK::legal(c,outer,r,k,sm_count)) runtime.push_back(c);
    }
    const G::RankerSemantics typed{
        dtype==T::DataType::FP8_E4M3 ? G::Fp8Format::E4M3 : G::Fp8Format::E5M2,
        semantics.combine==C::CubReductionStageCombine::Sum ? G::Combine::Sum :
        semantics.combine==C::CubReductionStageCombine::Product ? G::Combine::Product :
        semantics.combine==C::CubReductionStageCombine::Minimum ? G::Combine::Min : G::Combine::Max,
        semantics.input_transform==C::CubReductionStageInputTransform::Identity ? G::InputTransform::Identity :
        semantics.input_transform==C::CubReductionStageInputTransform::AbsoluteValue ? G::InputTransform::Abs : G::InputTransform::Square};
    G::GeneralShortlistOptions options;
    options.max_candidates=128;
    options.per_family_limit=128;
    options.max_scratch_bytes=std::numeric_limits<uint64_t>::max();
    return G::generalShortlist(outer,r,k,sm_count,ow,dr,ak,runtime,typed,options);
}

uint64_t outerNear(uint64_t r, uint64_t k, uint64_t bytes) {
    if (r > std::numeric_limits<uint64_t>::max()/k) throw std::overflow_error("RK shape overflow");
    return nearTargetOuter(r,k,bytes);
}

void run(const Settings& settings) {
    const T::TensorPlacement placement(T::TensorPlacement::MemDevices::GPU,0);
    ScopedGpu gpu(0);
    Stream stream(placement);
    int sm=0;
    check(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0),"get SM count");
    if (sm<=0) throw std::runtime_error("No CUDA SMs");
    Arena arena(placement,stream,settings.arena_bytes);
    B::Fp8OracleBatchQueue queue(128,Passes,[&]{stream.synchronize();});
    const auto start=std::chrono::steady_clock::now();
    std::cout << "# cross_family_oracle=1 fast_pruning=1 retained_production_kernels=1 "
              << "performance_only=1 correctness_checks=0 operations=4 combines x 3 transforms output="
              << (settings.historical_fp8_output?"same_fp8":"fp32")
              << " staged_partials=fp32 continuation=production_selected_FP32 samples=5 interleaved=1"
              << " dataset=fast_cold_ring_pruning";
    if (!settings.family.empty()) std::cout << " family_filter=" << settings.family;
    if (settings.onewindow_runtime_k) std::cout << " onewindow_runtime_k=1";
    if (settings.onewindow_shapes) std::cout << " shapes=onewindow";
    if (settings.uncovered_shapes) std::cout << " shapes=uncovered";
    if (settings.replacement_shapes) std::cout << " shapes=replacement";
    if (settings.all_shapes) std::cout << " shapes=all";
    if (std::any_of(settings.operations.begin(),settings.operations.end(),[](const std::string& name) {
            return std::any_of(arg_ops.begin(),arg_ops.end(),[&](const ArgOp& op) { return name == op.name; });}))
        std::cout << " arg_path=stamped arg_index=uint32 arg_outputs=" << (settings.arg_produce_value?"value_index":"index")
                  << " arg_value_output=" << (settings.historical_fp8_output?"same_fp8":"fp32");
    if (!settings.operations.empty()) {
        std::cout << " ops=";
        for (size_t i=0;i<settings.operations.size();++i) std::cout << (i?",":"") << settings.operations[i];
    }
    std::cout << " dtypes=";
    for (size_t i=0;i<settings.dtypes.size();++i) std::cout << (i?",":"") << dtypeName(settings.dtypes[i]);
    std::cout << " targets_mib=";
    for (size_t i=0;i<settings.target_mib.size();++i) std::cout << (i?",":"") << settings.target_mib[i];
    std::cout << '\n';
    std::cout << "family,outer,R,K,dtype,configuration,operation,rank12,rank16,rank20,rank24,rank28,rank32,rank40,rank48,rank128,"
                 "total_median_ms,sample0_ms,sample1_ms,sample2_ms,sample3_ms,sample4_ms\n";
    uint64_t prepared_cells=0;
    uint64_t collected_cells=0;
    // Interleaved event timing of every candidate of one prepared cell (outputs already bound).
    const auto enqueue=[&](const std::shared_ptr<Cell>& cell) {
        const size_t count=cell->candidates.size();
        cell->starts.reserve(Passes*count);cell->stops.reserve(Passes*count);
        for (size_t i=0;i<Passes*count;++i) {
            cell->starts.emplace_back();cell->stops.emplace_back();
        }
        auto launch=[cell,&stream,&arena](size_t index) {
            cell->input=arena.input(cell->dtype,cell->outer,cell->r,cell->k);
            auto& candidate=cell->candidates[index];
            if (candidate.direct) candidate.direct(cell->input,cell->output,stream);
            else candidate.executable->launch(cell->input,cell->output,cell->scratch,stream);
        };
        auto submit=[cell,launch,&stream](int pass) {
            const size_t n=cell->candidates.size();
            for (size_t visit=0;visit<n;++visit) {
                const size_t index=(visit+static_cast<size_t>(pass))%n;
                const size_t event=index*Passes+pass;
                check(cudaEventRecord(cell->starts[event].handle,stream.getStream()),"start event");
                launch(index);
                check(cudaEventRecord(cell->stops[event].handle,stream.getStream()),"stop event");
            }
        };
        auto collect=[cell,&collected_cells] {
            for (size_t index=0;index<cell->candidates.size();++index) {
                auto& item=cell->candidates[index];
                for (int pass=0;pass<Passes;++pass) {
                    float ms=0;
                    check(cudaEventElapsedTime(&ms,cell->starts[index*Passes+pass].handle,
                                             cell->stops[index*Passes+pass].handle),"event elapsed");
                    item.milliseconds[pass]=ms;
                }
                auto sorted=item.milliseconds;
                std::sort(sorted.begin(),sorted.end());
                std::cout << item.family<<','<<cell->outer<<','<<cell->r<<','<<cell->k
                          <<','<<dtypeName(cell->dtype)<<','<<item.name<<','<<cell->operation;
                for (size_t budget : {12u,16u,20u,24u,28u,32u,40u,48u,128u})
                    std::cout << ',' << (item.rank<=budget?item.rank:0);
                std::cout << ',' << std::fixed << std::setprecision(7) << sorted[Passes/2];
                for (double ms:item.milliseconds) std::cout << ','<<ms;
                std::cout << '\n';
            }
            std::cout << "# cross_family_shape_completed,outer="<<cell->outer<<",R="<<cell->r
                      <<",K="<<cell->k<<",dtype="<<dtypeName(cell->dtype)
                      <<",operation="<<cell->operation
                      <<",oracle_candidates="<<cell->candidates.size()<<'\n';
            if (++collected_cells % 64 == 0)
                std::cerr << "[fp8-retained-cold-l2] completed " << collected_cells
                          << " workload cells\n";
        };
        queue.add({[launch] {for(int i=0;i<8;++i)launch(0);},submit,collect});
        ++prepared_cells;
    };
    const auto grid=settings.uncovered_shapes?uncoveredKShapes():
                    settings.replacement_shapes?replacementShapes():
                    settings.all_shapes?allShapes():shapes();
    for (T::DataType dtype : settings.dtypes) {
        for (const Op& op : ops) {
            if (!settings.operations.empty() &&
                std::find(settings.operations.begin(),settings.operations.end(),op.name)==settings.operations.end())
                continue;
            for (const auto& [r,k] : grid) {
                if (settings.onewindow_shapes && !oneWindowRelevantShape(r,k)) continue;
                // 1229: the FP32 continuation is the retained FP32 selectors' (1225 / 1227), which time candidates and
                // need the original operation and reduction size (R: the cell reduces one axis).
                C::DenseProductionPhysicalSelectionContext fp32_context(
                    dtype, C::denseRKValueOperation(op.operation), stream, r, 1.0f, op.operation);
                for (uint64_t target_mib : settings.target_mib) {
                    const uint64_t target=target_mib*MiB;
                    if (prepared_cells>=settings.max_cells) goto finished;
                    // Inputs and output/scratch workspace have stable backing
                    // buffers. Cells own only alias handles and CUDA events.
                    const uint64_t outer=outerNear(r,k,target);
                    const auto problem=[&]{
                        C::DenseReductionOutputSpec spec;
                        spec.value_dtype=settings.historical_fp8_output?dtype:T::DataType::FP32;
                        return C::makeInitialDenseReductionProblem(
                            {outer,r,k},{1},C::DenseReductionAggregateKind::Value,spec);
                    }();
                    const auto site=C::enumerateDenseReductionSites(problem).front();
                    auto semantics=C::makeValueReductionStageSemantics(op.operation,C::DensePassRole::Complete,r);
                    semantics.combine=op.combine;
                    semantics.input_transform=op.transform;
                    auto plans=retainedChoices(outer,r,k,static_cast<uint32_t>(sm),dtype,
                            settings.historical_fp8_output?dtype:T::DataType::FP32,
                            semantics,settings.family);
                    auto cell=std::make_shared<Cell>(arena.input(dtype,outer,r,k),dtype,outer,r,k,op.name);
                    uint64_t max_scratch=0;
                    for (size_t rank=0;rank<plans.size();++rank) {
                        auto physical=bind(plans[rank]);
                        if (!settings.family.empty() && settings.family!=familyName(physical.family)) continue;
                        if (!F::structurallyValid(physical,outer,r,k))
                            throw std::logic_error("Invalid retained physical identity in cold-L2 sweep");
                        auto isolated=F::planIsolatedRKSite(problem,site,physical,semantics);
                        try {
                            auto executable=F::makeSelectedIsolatedBenchmark(problem,site,std::move(isolated),physical,
                                fp32_context.context(),1.0f,std::numeric_limits<uint64_t>::max(),stream);
                            max_scratch=std::max<uint64_t>(max_scratch,executable.scratch_bytes);
                            cell->candidates.push_back({familyName(physical.family),std::move(executable),{},
                                physicalName(physical),rank+1,{}});
                        } catch (const F::UnavailableIsolatedCandidate& e) {
                            // Missing candidates silently corrupt the measured oracle
                            // and can falsely validate the learned 28-item prefix.
                            throw std::runtime_error("Retained candidate " + physicalName(physical) +
                                " cannot construct complete FP32 continuation: " + e.what());
                        }
                    }
                    if (settings.onewindow_runtime_k &&
                        (settings.family.empty() || settings.family=="one_window" ||
                         settings.family=="one_window_rk")) {
                        // OneWindow units directly: every compiled unit whose bucket holds K, at every legal
                        // warps-per-output. Not part of the learned ranking (rank 0).
                        const auto geometry=T::CubReduction::analyzeValueGeometry(op.operation,{outer,r,k},{1});
                        for (const C::Fp8OneWindowRuntimeKUnit unit : C::fp8OneWindowRuntimeKUnits(k)) {
                            const uint32_t o=unit.outputs_per_warp;
                            for (uint32_t w : {1u,2u,4u,8u}) {
                                if ((o>1 && w!=1) || r<w) continue;
                                const std::string name="o"+std::to_string(o)+"_w"+std::to_string(w)+"_p"+
                                    std::to_string(unit.packet_bytes)+"_c"+std::to_string(unit.max_components_per_lane)+
                                    (unit.load_ahead?"_la":"");
                                cell->candidates.push_back({"one_window_rk",std::nullopt,
                                    [geometry,unit,w,semantics](const T::Tensor& in,T::Tensor& out,Stream& s) {
                                        C::launchFp8OneWindowRuntimeKComplete(in,out,geometry,unit,w,s,semantics,1.0f);
                                    },name,0,{}});
                            }
                        }
                    }
                    if (cell->candidates.empty()) continue;
                    auto [output,scratch]=arena.outputAndScratch(
                        settings.historical_fp8_output?dtype:T::DataType::FP32,
                        outer,k,max_scratch);
                    cell->output=std::move(output);
                    cell->scratch=std::move(scratch);
                    enqueue(cell);
                }
            }
        }
        // ARG ops run only when requested by name (--ops or --focus=arg), so the default sweep is unchanged.
        for (const ArgOp& op : arg_ops) {
            if (std::find(settings.operations.begin(),settings.operations.end(),op.name)==settings.operations.end())
                continue;
            if (!settings.family.empty())
                throw std::invalid_argument("--family filters retained VALUE families; it cannot select ARG ops");
            T::CubArgReductionOutputOptions outputs;
            outputs.produce_value=settings.arg_produce_value;
            // --arg-outputs=index: no value dtype (CubArgReduction rejects a value dtype without a value output).
            if (settings.arg_produce_value)
                outputs.value_output_dtype=settings.historical_fp8_output?dtype:T::DataType::FP32;
            outputs.index_output_dtype=T::DataType::UINT32;
            const T::CubArgReduction reduction(op.operation,1u,outputs);
            const std::optional<T::DataType> value_dtype=settings.arg_produce_value?
                std::optional<T::DataType>(reduction.resolveValueOutputDataType(dtype)):std::nullopt;
            for (const auto& [r,k] : grid) {
                if (settings.onewindow_shapes && !oneWindowRelevantShape(r,k)) continue;
                for (uint64_t target_mib : settings.target_mib) {
                    if (prepared_cells>=settings.max_cells) goto finished;
                    const uint64_t outer=outerNear(r,k,MiB*target_mib);
                    const std::vector<uint64_t> dims{outer,r,k};
                    const T::CubReductionGeometry geometry=T::CubArgReduction::analyzeDenseGeometry(dims,{1});
                    auto out=arena.argOutputs(value_dtype,outer,k,0);
                    auto cell=std::make_shared<Cell>(arena.input(dtype,outer,r,k),dtype,outer,r,k,op.name);
                    if (out.value) cell->output=*out.value;
                    // Production: stamp once (selection/autotune happen here, outside timing) on preallocated outputs.
                    auto stamped=reduction.stamp(cell->input,out.value,out.index,stream);
                    std::string name;
                    for (const std::string& pass : stamped->getPlannedPassNames()) name+=(name.empty()?"":"+")+pass;
                    if (name.empty()) throw std::logic_error("FP8 ARG production stamp has no planned passes");
                    cell->candidates.push_back({"arg_production",std::nullopt,
                        [stamped](const T::Tensor& in,T::Tensor&,Stream& s) { stamped->runOnInputForBenchmark(in,s); },
                        std::move(name),0,{}});
                    const bool maximum=op.operation==T::CubArgReductionOp::ArgMax;
                    if (settings.onewindow_runtime_k) {
                        // Every OneWindow ARG twin directly (not ranked), named like the VALUE one_window_rk units.
                        for (const C::Fp8OneWindowRuntimeKUnit unit : C::fp8OneWindowRuntimeKUnits(k)) {
                            for (uint32_t w : {1u,2u,4u,8u}) {
                                if (r<w || !C::fp8OneWindowArgSupports(outer,r,k,unit,w)) continue;  // as one_window_rk
                                const std::string unit_name="o"+std::to_string(unit.outputs_per_warp)+"_w"+
                                    std::to_string(w)+"_p"+std::to_string(unit.packet_bytes)+"_c"+
                                    std::to_string(unit.max_components_per_lane)+(unit.load_ahead?"_la":"");
                                cell->candidates.push_back({"one_window_arg",std::nullopt,
                                    [geometry,unit,w,maximum,value=out.value,index=out.index]
                                    (const T::Tensor& in,T::Tensor&,Stream& s) mutable {
                                        C::launchFp8OneWindowArgComplete(in,value?&*value:nullptr,index?&*index:nullptr,
                                                                         geometry,unit,w,maximum,1,s);
                                    },unit_name,0,{}});
                            }
                        }
                        // Every DeepRSmallK ARG depth: the first stage into 4-byte packed partials plus the packed
                        // continuation (what production runs since 1116), named like the VALUE deep_r_small_k candidates.
                        for (uint64_t rows : F::Fp8DeepRSmallKPolicy::compiled_rows) {
                            if (!C::fp8DeepRSmallKArgSupports(outer,r,k,rows)) continue;
                            const uint64_t shards=(r+rows-1)/rows;
                            uint32_t* packed=static_cast<uint32_t*>(arena.argPartials(outer,shards,k).first.getMemPtr<void>());
                            cell->candidates.push_back({"deep_r_small_k_arg",std::nullopt,
                                [geometry,outer,r,k,rows,shards,maximum,packed,
                                 value=out.value,index=out.index](const T::Tensor& in,T::Tensor&,Stream& s) mutable {
                                    C::launchFp8DeepRSmallKArgFirstStagePacked(in,packed,geometry,rows,shards,maximum,s);
                                    C::launchFp8ArgPackedContinuation(packed,value?&*value:nullptr,
                                        index?&*index:nullptr,in.getDataType(),outer,r,k,shards,rows,maximum,1,s);
                                },"p16_rows"+std::to_string(rows),0,{}});
                        }
                        // RuntimeK ARG: the PacketComplete twin at W 1/2/4/8 (tile: VALUE's fewest-tile width,
                        // capped at 256, 1127), then the rotated kernel (8-byte packets) at every rotated depth below R
                        // (staged, plus the carried-index reduction of the shards) and the single-shard Complete pass.
                        {
                            const uint32_t value_tile=static_cast<uint32_t>((k+(k+495)/496-1)/((k+495)/496));
                            const uint32_t tile=C::fp8PacketCompleteArgTileComponents(k,value_tile);
                            for (uint32_t w : {1u,2u,4u,8u}) {
                                if (w>r || !C::fp8PacketCompleteArgSupports(outer,r,k,w,tile)) continue;
                                cell->candidates.push_back({"runtime_k_arg",std::nullopt,
                                    [geometry,w,tile,maximum,value=out.value,index=out.index]
                                    (const T::Tensor& in,T::Tensor&,Stream& s) mutable {
                                        C::launchFp8PacketCompleteArg(in,value?&*value:nullptr,index?&*index:nullptr,
                                                                      geometry,w,tile,maximum,1,s);
                                    },"packet_complete_w"+std::to_string(w)+"_tile"+std::to_string(tile),0,{}});
                            }
                        }
                        for (uint32_t packet : {8u}) {
                            std::vector<uint64_t> depths(F::Fp8RuntimeKPolicy::rotated_row_depths.begin(),
                                                         F::Fp8RuntimeKPolicy::rotated_row_depths.end());
                            depths.erase(std::remove_if(depths.begin(),depths.end(),
                                [r](uint64_t rows) { return rows>=r; }),depths.end());
                            depths.push_back(r);  // single shard: the Complete pass
                            for (uint64_t rows : depths) {
                                if (!C::fp8RuntimeKArgSupports(outer,r,k,rows,packet)) continue;
                                const uint64_t shards=(r+rows-1)/rows;
                                const std::string unit_name="p"+std::to_string(packet)+
                                    (shards==1?std::string("_complete"):"_rows"+std::to_string(rows));
                                if (shards==1) {
                                    cell->candidates.push_back({"runtime_k_arg",std::nullopt,
                                        [geometry,rows,packet,maximum,value=out.value,index=out.index]
                                        (const T::Tensor& in,T::Tensor&,Stream& s) mutable {
                                            C::launchFp8RuntimeKArgPass(in,value?&*value:nullptr,index?&*index:nullptr,
                                                                        geometry,rows,packet,maximum,1,s);
                                        },unit_name,0,{}});
                                    continue;
                                }
                                uint32_t* packed=static_cast<uint32_t*>(arena.argPartials(outer,shards,k).first.getMemPtr<void>());
                                cell->candidates.push_back({"runtime_k_arg",std::nullopt,
                                    [geometry,outer,r,k,rows,shards,packet,maximum,packed,
                                     value=out.value,index=out.index](const T::Tensor& in,T::Tensor&,Stream& s) mutable {
                                        C::launchFp8RuntimeKArgPass(in,nullptr,nullptr,geometry,rows,packet,maximum,1,s,
                                                                    packed);
                                        C::launchFp8ArgPackedContinuation(packed,value?&*value:nullptr,
                                            index?&*index:nullptr,in.getDataType(),outer,r,k,shards,rows,maximum,1,s);
                                    },unit_name,0,{}});
                            }
                        }
                        // AlignedK ARG: every VALUE AlignedK variant (packet, CTA, shards), named like aligned_k.
                        for (const AK::Variant v : AK::candidates(outer,r,k)) {
                            if (!C::fp8AlignedKArgSupports(outer,r,k,v.packet,v.threads,v.shards_per_output)) continue;
                            const uint32_t packet=v.packet, threads=v.threads;
                            const uint64_t shards=v.shards_per_output;
                            const std::string unit_name="p"+std::to_string(packet)+"_cta"+std::to_string(threads)+
                                "_shards"+std::to_string(shards);
                            if (shards==1) {
                                cell->candidates.push_back({"aligned_k_arg",std::nullopt,
                                    [geometry,packet,threads,maximum,value=out.value,index=out.index]
                                    (const T::Tensor& in,T::Tensor&,Stream& s) mutable {
                                        C::launchFp8AlignedKArgPass(in,value?&*value:nullptr,index?&*index:nullptr,
                                                                    geometry,packet,threads,1,maximum,1,s);
                                    },unit_name,0,{}});
                                continue;
                            }
                            uint32_t* packed=static_cast<uint32_t*>(arena.argPartials(outer,shards,k).first.getMemPtr<void>());
                            cell->candidates.push_back({"aligned_k_arg",std::nullopt,
                                [geometry,outer,r,k,packet,threads,shards,maximum,packed,
                                 value=out.value,index=out.index](const T::Tensor& in,T::Tensor&,Stream& s) mutable {
                                    C::launchFp8AlignedKArgPass(in,nullptr,nullptr,geometry,packet,threads,shards,
                                                                maximum,1,s,packed);
                                    C::launchFp8ArgPackedContinuation(packed,value?&*value:nullptr,
                                        index?&*index:nullptr,in.getDataType(),outer,r,k,shards,0,maximum,1,s);
                                },unit_name,0,{}});
                        }
                    }
                    enqueue(cell);
                }
            }
        }
    }
finished:
    queue.flush();
    const auto t=queue.timing();
    const double wall=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
    std::cerr<<"[fp8-retained-cold-l2] cells="<<queue.completedCells()
             <<" batches="<<queue.batches()<<" wall_ms="<<wall
             <<" submit_ms="<<t.submit_ms<<" stream_wait_ms="<<t.wait_ms
             <<" collect_ms="<<t.collect_ms<<" warmup_ms="<<t.warmup_ms
             <<" arena_wraps="<<arena.wraps()<<'\n';
}
// ---------------------------------------------------------------------------------------------------------------
// 1133 --training: the ranker training dataset. Every legal retained candidate of a cell is timed end to end (VALUE:
// first stage + its production FP32 continuation; ARG: the single retained pass, packed staged passes including their
// packed continuation, exactly as launchRetainedFp8ArgPlan runs it in production). Each row carries the C++ ranker
// features (rankerFeatures of the candidate at its local rank), so training never re-derives features in Python.
// Semantics are sampled per geometry (not crossed): each geometry gets one VALUE cell and one ARG cell by default.
namespace Training {
// 1211: the carried kinds are the ARG kinds at a carried-index site (benchmarks/fp8_ranker/dataset.py KINDS 3 / 4).
enum class Kind { Value, ArgIndex, ArgValueIndex, CarriedArgIndex, CarriedArgValueIndex };
const char* kindName(Kind kind) {
    switch (kind) {
        case Kind::Value: return "value";
        case Kind::ArgIndex: return "arg_index";
        case Kind::ArgValueIndex: return "arg_value_index";
        case Kind::CarriedArgIndex: return "carried_arg_index";
        case Kind::CarriedArgValueIndex: return "carried_arg_value_index";
    }
    return "value";
}
uint64_t mix(uint64_t x) {  // splitmix64 finalizer: deterministic across platforms
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
// Uniform double in [0, 1) from a 64-bit hash.
double unit(uint64_t h) { return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0); }
struct Geometry {
    uint64_t outer, r, k, target_mib;
    bool random;
};
struct CellSpec {
    uint64_t id;
    Geometry geometry;
    bool heldout;
    Kind kind;
    T::DataType dtype;
    size_t value_op = 0;  // index into ops (Kind::Value)
    size_t arg_op = 0;    // index into arg_ops (ARG kinds)
};
// Held out by (R, K) shape, so every working-set size of a held-out shape is held out together (geometry-separated).
bool heldOutShape(uint64_t r, uint64_t k, uint64_t seed, uint32_t percent) {
    return mix(mix(seed ^ 0x48454C44ULL) ^ mix(r * 0x100000001B3ULL + k)) % 100 < percent;
}
// 1194: the working set is in bytes, so an FP32 cell (--dtypes=fp32, alone) has a quarter of FP8's elements; 1247: an
// FP16 / BF16 cell half.
std::vector<Geometry> geometries(const Settings& s, const std::vector<uint64_t>& targets) {
    std::vector<Geometry> result;
    // 1211: a carried-index cell streams its carried index too, so its working set counts both.
    const bool f16 = s.dtypes.front() == T::DataType::FP16 || s.dtypes.front() == T::DataType::BF16;
    const uint64_t element_bytes = f16 ? 2 : s.dtypes.front() != T::DataType::FP32 ? 1
                                   : s.carried_index == T::DataType::UINT64 ? 12
                                   : s.carried_index == T::DataType::UINT32 ? 8 : 4;
    for (const auto& [r, k] : allShapes())
        for (uint64_t t : targets) result.push_back({outerNear(r, k, t * MiB / element_bytes), r, k, t, false});
    // Log-uniform (R, K): R in [2, 2^16], K in [2, 2^12]; working set drawn from the target list.
    for (uint64_t i = 0, drawn = 0; drawn < s.training_random && i < s.training_random * 64; ++i) {
        const uint64_t h = mix(s.training_seed * 0x2545F4914F6CDD1DULL + i);
        const uint64_t r = static_cast<uint64_t>(std::llround(std::exp2(1.0 + 15.0 * unit(mix(h ^ 1)))));
        const uint64_t k = static_cast<uint64_t>(std::llround(std::exp2(1.0 + 11.0 * unit(mix(h ^ 2)))));
        const uint64_t t = targets[mix(h ^ 3) % targets.size()];
        if (r < 2 || k < 2 || r * k * element_bytes > t * MiB) continue;
        result.push_back({outerNear(r, k, t * MiB / element_bytes), r, k, t, true});
        ++drawn;
    }
    std::sort(result.begin(), result.end(), [](const Geometry& a, const Geometry& b) {
        return std::tie(a.r, a.k, a.outer) < std::tie(b.r, b.k, b.outer);
    });
    result.erase(std::unique(result.begin(), result.end(), [](const Geometry& a, const Geometry& b) {
        return a.r == b.r && a.k == b.k && a.outer == b.outer;
    }), result.end());
    return result;
}
std::vector<CellSpec> cells(const Settings& s, const std::vector<uint64_t>& targets) {
    std::vector<CellSpec> result;
    std::vector<size_t> value_ops;
    for (size_t i = 0; i < ops.size(); ++i)
        if (s.operations.empty() || std::find(s.operations.begin(), s.operations.end(), ops[i].name) != s.operations.end())
            value_ops.push_back(i);
    const bool arg_allowed = s.operations.empty() ||
        std::any_of(arg_ops.begin(), arg_ops.end(), [&](const ArgOp& op) {
            return std::find(s.operations.begin(), s.operations.end(), op.name) != s.operations.end(); });
    std::vector<size_t> arg_op_indices;
    for (size_t i = 0; i < arg_ops.size(); ++i)
        if (s.operations.empty() ||
            std::find(s.operations.begin(), s.operations.end(), arg_ops[i].name) != s.operations.end())
            arg_op_indices.push_back(i);
    uint64_t id = 0;
    for (const Geometry& g : geometries(s, targets)) {
        const bool heldout = heldOutShape(g.r, g.k, s.training_seed, s.training_heldout_percent);
        for (uint32_t j = 0; j < s.training_semantics_per_geometry; ++j) {
            const uint64_t h = mix(mix(s.training_seed) ^ mix(g.outer * 0x9E3779B1ULL ^ g.r * 0x85EBCA77ULL ^ g.k) ^ j);
            CellSpec c{id, g, heldout, Kind::Value, s.dtypes[mix(h ^ 11) % s.dtypes.size()]};
            // Alternate VALUE and ARG per geometry (even j: VALUE, odd j: ARG), so both kinds see every geometry.
            // 1205: FP32 cells alternate too (the FP32 RK ARG candidates; 1194-1204 were VALUE only).
            const bool want_arg = (j % 2) == 1;
            if ((want_arg && arg_allowed) || value_ops.empty()) {
                if (!arg_allowed) continue;
                c.kind = (mix(h ^ 13) & 1) ? Kind::ArgValueIndex : Kind::ArgIndex;
                if (s.carried_index)
                    c.kind = c.kind == Kind::ArgIndex ? Kind::CarriedArgIndex : Kind::CarriedArgValueIndex;
                c.arg_op = arg_op_indices[mix(h ^ 17) % arg_op_indices.size()];
            } else {
                c.value_op = value_ops[mix(h ^ 19) % value_ops.size()];
            }
            result.push_back(c);
            ++id;
        }
    }
    // 1140: a seeded shuffle, so any --max-cells prefix samples the whole grid (in (R, K) order a 200-cell slice
    // held only R <= 8 and could not size the run). Cell ids keep the grid order.
    std::stable_sort(result.begin(), result.end(), [&](const CellSpec& a, const CellSpec& b) {
        return mix(s.training_seed ^ mix(a.id)) < mix(s.training_seed ^ mix(b.id));
    });
    return result;
}
G::RankerSemantics typedSemantics(T::DataType dtype, C::CubReductionStageCombine combine,
                                  C::CubReductionStageInputTransform transform) {
    return {dtype == T::DataType::FP8_E4M3 ? G::Fp8Format::E4M3 : G::Fp8Format::E5M2,
            combine == C::CubReductionStageCombine::Sum ? G::Combine::Sum :
            combine == C::CubReductionStageCombine::Product ? G::Combine::Product :
            combine == C::CubReductionStageCombine::Minimum ? G::Combine::Min : G::Combine::Max,
            transform == C::CubReductionStageInputTransform::Identity ? G::InputTransform::Identity :
            transform == C::CubReductionStageInputTransform::AbsoluteValue ? G::InputTransform::Abs :
                                                                             G::InputTransform::Square};
}
// A unique configuration name for an ARG pass (the identity alone can merge OneWindow units that differ only in
// components-per-lane or load-ahead).
std::string argPlanName(const C::DenseRetainedFp8ArgPhysicalPlan& plan, const F::PhysicalKernel& identity) {
    std::string name = physicalName(identity);
    if (plan.family == F::PhysicalFamily::OneWindow)
        name += "_c" + std::to_string(plan.unit.max_components_per_lane) + (plan.unit.load_ahead ? "_la" : "");
    if (plan.family == F::PhysicalFamily::RuntimeK && plan.tile_components == 0)
        name += "_p" + std::to_string(plan.packet_bytes);
    return name;
}
std::string recordPrefix(const CellSpec& c, const char* op_name, const char* family, const std::string& configuration,
                         uint32_t local_rank, size_t list_position,
                         const std::array<float, F::Fp8GeometryRankerModel::FEATURE_COUNT>& features, bool staged,
                         std::optional<float> margin = std::nullopt) {
    std::ostringstream line;
    line << kindName(c.kind) << ',' << c.id << ',' << (c.heldout ? 1 : 0) << ',' << (c.geometry.random ? 1 : 0) << ','
         << c.geometry.outer << ',' << c.geometry.r << ',' << c.geometry.k << ',' << c.geometry.target_mib << ','
         << dtypeName(c.dtype) << ',' << op_name << ',' << family << ',' << configuration << ',' << local_rank << ','
         << list_position << ',' << (staged ? 1 : 0) << ',' << std::setprecision(9)
         << (margin ? *margin : F::Fp8GeometryRankerModel::margin(features));  // 1194: FP32 passes its model's
    // f0..f24: the training CSV keeps the 25 geometry/physical features; the aggregate kind (feature 25) is the kind
    // column (benchmarks/fp8_ranker/dataset.py appends it).
    for (size_t i = 0; i + 1 < features.size(); ++i) line << ',' << std::setprecision(9) << features[i];
    return line.str();
}
}  // namespace Training

// Interleaved cold-L2 timing of one prepared cell whose candidates' `name` is a complete CSV record prefix: prints
// "<name>,median,samples..." per candidate and progress/ETA to stderr (--training, --continuation-audit).
struct RecordQueue {
    Stream& stream;
    Arena& arena;
    B::Fp8OracleBatchQueue& queue;
    uint64_t total;
    std::chrono::steady_clock::time_point start;
    const char* tag;
    uint64_t collected = 0, candidates = 0;
    // 1164: >0 (--training): probe() sets each candidate's launches per sample, about interval_us each, and the rows
    // carry launches_per_sample and probe_ms.
    double interval_us = 0.0;
    uint64_t max_launches = 1;
    // One launch: a fresh cold input view and the cell's next output view.
    void launchOnce(Cell& cell, size_t index) {
        cell.input=arena.input(cell.dtype,cell.outer,cell.r,cell.k);
        if (cell.carried_dtype) cell.carried=arena.input(*cell.carried_dtype,cell.outer,cell.r,cell.k);
        T::Tensor& out=cell.outputs.empty()?cell.output:cell.outputs[cell.next_output++%cell.outputs.size()];
        auto& candidate=cell.candidates[index];
        if (candidate.direct_carried) candidate.direct_carried(cell.input,cell.carried,out,stream);
        else if (candidate.direct) candidate.direct(cell.input,out,stream);
        else candidate.executable->launch(cell.input,out,cell.scratch,stream);
    }
    // 1164: per candidate an untimed launch (loads the kernel) and a timed probe launch, as the KR sweep.
    void probe(Cell& cell) {
        const size_t n=cell.candidates.size();
        std::vector<Event> events(2*n);
        for (size_t i=0;i<n;++i) {
            launchOnce(cell,i);
            check(cudaEventRecord(events[2*i].handle,stream.getStream()),"probe start");
            launchOnce(cell,i);
            check(cudaEventRecord(events[2*i+1].handle,stream.getStream()),"probe stop");
        }
        stream.synchronize();
        cell.launches.assign(n,1);
        cell.probe_ms.assign(n,0.0);
        for (size_t i=0;i<n;++i) {
            float ms=0;
            check(cudaEventElapsedTime(&ms,events[2*i].handle,events[2*i+1].handle),"probe elapsed");
            cell.probe_ms[i]=ms;
            cell.launches[i]=B::ColdTiming::launchesPerSample(1000.0*ms,interval_us,max_launches);
        }
    }
    void enqueue(const std::shared_ptr<Cell>& cell) {
        const size_t count=cell->candidates.size();
        candidates+=count;
        for (size_t i=0;i<Passes*count;++i) { cell->starts.emplace_back(); cell->stops.emplace_back(); }
        auto launch=[cell,this](size_t index) { launchOnce(*cell,index); };
        auto submit=[cell,launch,this](int pass) {
            const size_t n=cell->candidates.size();
            for (size_t visit=0;visit<n;++visit) {
                const size_t index=(visit+static_cast<size_t>(pass))%n;
                const size_t event=index*Passes+pass;
                const uint64_t launches=cell->launches.empty()?1:cell->launches[index];
                check(cudaEventRecord(cell->starts[event].handle,stream.getStream()),"start event");
                for (uint64_t l=0;l<launches;++l) launch(index);
                check(cudaEventRecord(cell->stops[event].handle,stream.getStream()),"stop event");
            }
        };
        auto collect=[cell,this] {
            for (size_t index=0;index<cell->candidates.size();++index) {
                auto& item=cell->candidates[index];
                for (int pass=0;pass<Passes;++pass) {
                    float ms=0;
                    check(cudaEventElapsedTime(&ms,cell->starts[index*Passes+pass].handle,
                                               cell->stops[index*Passes+pass].handle),"event elapsed");
                    const uint64_t launches=cell->launches.empty()?1:cell->launches[index];
                    item.milliseconds[pass]=ms/static_cast<double>(launches);  // per launch
                }
                auto sorted=item.milliseconds;
                std::sort(sorted.begin(),sorted.end());
                std::cout << item.name << ',' << std::fixed << std::setprecision(7) << sorted[Passes/2];
                for (double ms:item.milliseconds) std::cout << ',' << ms;
                if (interval_us>0.0)
                    std::cout << ',' << (cell->launches.empty()?1:cell->launches[index]) << ','
                              << (cell->probe_ms.empty()?0.0:cell->probe_ms[index]);
                std::cout << std::defaultfloat << '\n';
            }
            ++collected;
            if (collected % 25 == 0 || collected == total) {
                const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                const double eta=elapsed/static_cast<double>(collected)*static_cast<double>(total-collected);
                std::cerr << '[' << tag << "] " << collected << '/' << total << " cells, elapsed "
                          << std::fixed << std::setprecision(0) << elapsed << " s, eta " << eta << " s\n"
                          << std::defaultfloat;
            }
        };
        queue.add({[launch]{for(int i=0;i<8;++i)launch(0);},submit,collect});
    }
};

void runTraining(const Settings& settings) {
    const T::TensorPlacement placement(T::TensorPlacement::MemDevices::GPU,0);
    ScopedGpu gpu(0);
    Stream stream(placement);
    int sm=0;
    check(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0),"get SM count");
    if (sm<=0) throw std::runtime_error("No CUDA SMs");
    if (!settings.family.empty()) throw std::invalid_argument("--training times every family; --family is not allowed");
    const std::vector<uint64_t> targets=settings.targets_given?settings.target_mib:std::vector<uint64_t>{96,192};
    const auto specs=Training::cells(settings,targets);
    const uint64_t total=std::min<uint64_t>(specs.size(),settings.max_cells);
    Arena arena(placement,stream,settings.arena_bytes);
    // 1164: rotating output views, as the KR sweep (a shared output stays in L2 between launches).
    namespace CT=B::ColdTiming;
    CT::Ring outputs(placement,settings.output_ring_bytes);
    check(cudaMemsetAsync(outputs.storage().getMemPtr<void>(),0,settings.output_ring_bytes,stream.getStream()),
          "output ring initialization");
    stream.synchronize();
    const auto outputViews=[&](Cell& cell,T::DataType dtype,const std::vector<uint64_t>& dims) {
        const uint64_t bytes=T::TensorDescriptor(dtype,dims).getArraySizeInBytes();
        for (uint64_t v=0,count=CT::viewCount(bytes,CT::OutputReuseBytes,outputs.bytes());v<count;++v)
            cell.outputs.push_back(outputs.view(dtype,dims,outputs.take(bytes)));
    };
    B::Fp8OracleBatchQueue queue(128,Passes,[&]{stream.synchronize();});
    const auto start=std::chrono::steady_clock::now();
    std::cout << "# fp8_ranker_training=1 version=1164 method=ring_batched_probe interval_us=" << settings.interval_us
              << " max_launches=" << settings.max_launches << " output_ring_bytes=" << settings.output_ring_bytes
              << " output_reuse_bytes=" << CT::OutputReuseBytes << " seed=" << settings.training_seed
              << " random_geometries=" << settings.training_random
              << " semantics_per_geometry=" << settings.training_semantics_per_geometry
              << " heldout_percent=" << settings.training_heldout_percent << " cells=" << total
              << " cells_available=" << specs.size()
              << " samples=" << Passes << " interleaved=1 cold_l2=1 value_output=fp32"
              << " value_continuation=production_selected_FP32 arg_index=uint32 arg_value=fp32 sm_count=" << sm
              << " carried_index=" << (!settings.carried_index ? "none"
                                       : *settings.carried_index == T::DataType::UINT64 ? "uint64" : "uint32")
              << " input_dtypes=";
    for (size_t i=0;i<settings.dtypes.size();++i) std::cout << (i?",":"") << dtypeName(settings.dtypes[i]);
    std::cout << " targets_mib=";
    for (size_t i=0;i<targets.size();++i) std::cout << (i?",":"") << targets[i];
    std::cout << "\nkind,cell_id,heldout,random_geometry,outer,R,K,target_mib,dtype,operation,family,configuration,"
                 "local_rank,list_position,staged,frozen_margin";
    for (int f=0;f<25;++f) std::cout << ",f" << f;
    std::cout << ",total_median_ms,sample0_ms,sample1_ms,sample2_ms,sample3_ms,sample4_ms,launches_per_sample,"
                 "probe_ms\n";

    if (settings.training_stamp) std::cout << "#stamp,cell_id,stamp_ms,candidates,winner\n";
    // 1162 --stamp: production's own contest at this cell (the cache-bypassing RK entry points CubReduction stamps
    // through, Regular depth, unlimited workspace like the timed candidates), on a cold arena view of the same constant
    // input. The winner is named like the CSV configuration, so the analysis reads its timed median as the pick.
    const auto stampRow=[&](uint64_t cell_id,const std::function<F::Fp8RKSelection()>& contest,
                            const std::function<std::string(const F::Fp8RKSelection&)>& name) {
        stream.synchronize();
        const auto t0=std::chrono::steady_clock::now();
        const F::Fp8RKSelection selection=contest();
        stream.synchronize();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        std::cout << "#stamp," << cell_id << ',' << std::fixed << std::setprecision(3) << ms << std::defaultfloat
                  << ',' << selection.candidate_count << ','
                  << (selection.candidate_count ? name(selection) : std::string("none")) << '\n';
    };

    RecordQueue records(stream,arena,queue,total,start,"fp8-ranker-training");
    records.interval_us=settings.interval_us;
    records.max_launches=settings.max_launches;
    // 1164: each candidate's launches per sample from its probe (samples are ms per launch).
    const auto enqueue=[&](const std::shared_ptr<Cell>& cell) {
        records.probe(*cell);
        records.enqueue(cell);
    };

    // One FP32 continuation context per (dtype, VALUE op, R). 1229: the continuation is the retained FP32 selectors'
    // (1225 / 1227), which time candidates and need the original operation and reduction size.
    std::map<std::tuple<int,size_t,uint64_t>,std::unique_ptr<C::DenseProductionPhysicalSelectionContext>> contexts;
    const auto context=[&](T::DataType dtype,size_t op,uint64_t r)->const C::DensePhysicalSelectionContext& {
        auto& slot=contexts[{static_cast<int>(dtype),op,r}];
        if (!slot) slot=std::make_unique<C::DenseProductionPhysicalSelectionContext>(
            dtype,C::denseRKValueOperation(ops[op].operation),stream,r,1.0f,ops[op].operation);
        return slot->context();
    };
    for (uint64_t n=0;n<total;++n) {
        const Training::CellSpec& spec=specs[n];
        const uint64_t outer=spec.geometry.outer, r=spec.geometry.r, k=spec.geometry.k;
        const T::DataType dtype=spec.dtype;
        if (spec.kind==Training::Kind::Value) {
            const Op& op=ops[spec.value_op];
            const auto problem=[&]{
                C::DenseReductionOutputSpec out;
                out.value_dtype=T::DataType::FP32;
                return C::makeInitialDenseReductionProblem({outer,r,k},{1},C::DenseReductionAggregateKind::Value,out);
            }();
            const auto site=C::enumerateDenseReductionSites(problem).front();
            auto semantics=C::makeValueReductionStageSemantics(op.operation,C::DensePassRole::Complete,r);
            semantics.combine=op.combine;
            semantics.input_transform=op.transform;
            const auto typed=Training::typedSemantics(dtype,op.combine,op.transform);
            // 1166: the public operation whose original-input stage is exactly this cell's semantics (sum of |x| is
            // L1Norm, sum of x^2 SumSquares). 1162-1165 passed op.operation, so 8 of the 12 VALUE combinations were
            // stamped as their identity form (e.g. min_square as MIN). Product, MIN and MAX of |x| or x^2 have no
            // public operation, so no production stamp reaches them: those cells get no '#stamp' line.
            std::optional<T::CubReductionOp> public_op;
            if (op.transform==C::CubReductionStageInputTransform::Identity) public_op=op.operation;
            else if (op.operation==T::CubReductionOp::Sum)
                public_op=op.transform==C::CubReductionStageInputTransform::AbsoluteValue?T::CubReductionOp::L1Norm
                                                                                         :T::CubReductionOp::SumSquares;
            if (settings.training_stamp && public_op) {
                const auto stamped=C::makeValueReductionStageSemantics(*public_op,C::DensePassRole::Complete,r);
                if (stamped.combine!=semantics.combine || stamped.input_transform!=semantics.input_transform)
                    throw std::logic_error(std::string("--stamp operation does not match the cell's semantics: ")+op.name);
                stampRow(spec.id,[&]{
                    if (dtype==T::DataType::FP16||dtype==T::DataType::BF16)  // 1247: the 2-byte RK selector's contest
                        return C::F16RK::autotuneValueSite(problem,site,*public_op,r,1.0f,
                                                           arena.input(dtype,outer,r,k),stream,
                                                           std::numeric_limits<uint64_t>::max(),
                                                           T::CubReductionTuningDepth::Regular);
                    if (dtype==T::DataType::FP32)  // 1194: the FP32 RK selector's contest
                        return C::Fp32RK::autotuneValueSite(problem,site,*public_op,r,1.0f,
                                                            arena.input(dtype,outer,r,k),stream,
                                                            std::numeric_limits<uint64_t>::max(),
                                                            T::CubReductionTuningDepth::Regular);
                    return F::autotuneIsolatedRKSite(problem,site,*public_op,r,1.0f,arena.input(dtype,outer,r,k),
                                                     stream,std::numeric_limits<uint64_t>::max(),
                                                     T::CubReductionTuningDepth::Regular);
                },[](const F::Fp8RKSelection& selection){ return physicalName(selection.physical); });
            }
            if (dtype==T::DataType::FP16||dtype==T::DataType::BF16) {
                // 1247: every entry of the FP16 / BF16 RK pool (F16RK::f16RKCandidatePool, the selector's
                // candidates), at its pool position and local rank, with the 2-byte ranker's features and margin.
                C::F16RK::F16RKSite f16_site;
                f16_site.outer=outer;
                f16_site.r=r;
                f16_site.k=k;
                f16_site.sm_count=static_cast<uint32_t>(sm);
                f16_site.input_dtype=dtype;
                f16_site.semantics=semantics;
                const auto pool=C::F16RK::f16RKCandidatePool(f16_site,[&](const F::PhysicalKernel& p) {
                    return C::F16RK::f16RKCompiledResidency(p,dtype,k,semantics);
                });
                auto cell=std::make_shared<Cell>(arena.input(dtype,outer,r,k),dtype,outer,r,k,op.name);
                uint64_t max_scratch=0;
                for (size_t position=0;position<pool.size();++position) {
                    const auto& entry=pool[position];
                    const F::PhysicalKernel& physical=entry.kernel;
                    const auto features=C::F16RK::f16RKRankerFeatures(entry,f16_site,0);
                    auto isolated=F::planIsolatedRKSite(problem,site,physical,semantics);
                    try {
                        auto executable=F::makeSelectedIsolatedBenchmark(problem,site,std::move(isolated),physical,
                            context(dtype,spec.value_op,r),1.0f,std::numeric_limits<uint64_t>::max(),stream);
                        max_scratch=std::max<uint64_t>(max_scratch,executable.scratch_bytes);
                        Timed timed{familyName(physical.family),std::move(executable),{},
                            Training::recordPrefix(spec,op.name,familyName(physical.family),physicalName(physical),
                                                   entry.local_rank,position+1,features,physical.staged(),
                                                   C::F16RK::F16RKRankerModel::margin(features)),
                            position+1,{}};
                        cell->candidates.push_back(std::move(timed));
                    } catch (const F::UnavailableIsolatedCandidate& e) {
                        throw std::runtime_error("FP16 / BF16 candidate " + physicalName(physical) +
                            " cannot construct complete FP32 continuation: " + e.what());
                    }
                }
                if (cell->candidates.empty())
                    throw std::logic_error("FP16 / BF16 training cell has no retained VALUE candidate");
                auto [output,scratch]=arena.outputAndScratch(T::DataType::FP32,outer,k,max_scratch);
                cell->output=std::move(output);
                cell->scratch=std::move(scratch);
                outputViews(*cell,T::DataType::FP32,{outer,k});
                enqueue(cell);
                continue;
            }
            if (dtype==T::DataType::FP32) {
                // 1194: every entry of the FP32 RK pool (Fp32RK::fp32RKCandidatePool, the selector's candidates), at
                // its pool position and local rank, with the FP32 ranker's features and margin.
                const C::Fp32RK::Fp32RKSite fp32_site{outer,r,k,static_cast<uint32_t>(sm),semantics};
                const auto pool=C::Fp32RK::fp32RKCandidatePool(fp32_site,[&](const F::PhysicalKernel& p) {
                    return C::Fp32RK::fp32RKCompiledResidency(p,k,semantics);
                });
                auto cell=std::make_shared<Cell>(arena.input(dtype,outer,r,k),dtype,outer,r,k,op.name);
                uint64_t max_scratch=0;
                for (size_t position=0;position<pool.size();++position) {
                    const auto& entry=pool[position];
                    const F::PhysicalKernel& physical=entry.kernel;
                    const auto features=C::Fp32RK::fp32RKRankerFeatures(entry,fp32_site,0);
                    auto isolated=F::planIsolatedRKSite(problem,site,physical,semantics);
                    try {
                        auto executable=F::makeSelectedIsolatedBenchmark(problem,site,std::move(isolated),physical,
                            context(dtype,spec.value_op,r),1.0f,std::numeric_limits<uint64_t>::max(),stream);
                        max_scratch=std::max<uint64_t>(max_scratch,executable.scratch_bytes);
                        Timed timed{familyName(physical.family),std::move(executable),{},
                            Training::recordPrefix(spec,op.name,familyName(physical.family),physicalName(physical),
                                                   entry.local_rank,position+1,features,physical.staged(),
                                                   C::Fp32RK::Fp32RKRankerModel::margin(features)),
                            position+1,{}};
                        cell->candidates.push_back(std::move(timed));
                    } catch (const F::UnavailableIsolatedCandidate& e) {
                        throw std::runtime_error("FP32 candidate " + physicalName(physical) +
                            " cannot construct complete FP32 continuation: " + e.what());
                    }
                }
                if (cell->candidates.empty()) throw std::logic_error("FP32 training cell has no retained VALUE candidate");
                auto [output,scratch]=arena.outputAndScratch(T::DataType::FP32,outer,k,max_scratch);
                cell->output=std::move(output);
                cell->scratch=std::move(scratch);
                outputViews(*cell,T::DataType::FP32,{outer,k});
                enqueue(cell);
                continue;
            }
            const auto plans=retainedChoices(outer,r,k,static_cast<uint32_t>(sm),dtype,T::DataType::FP32,semantics,"");
            auto cell=std::make_shared<Cell>(arena.input(dtype,outer,r,k),dtype,outer,r,k,op.name);
            uint64_t max_scratch=0;
            for (size_t position=0;position<plans.size();++position) {
                const auto& plan=plans[position];
                const auto physical=bind(plan);
                if (!F::structurallyValid(physical,outer,r,k))
                    throw std::logic_error("Invalid retained physical identity in training sweep");
                const auto features=G::Detail::rankerFeatures(plan,outer,r,k,typed,plan.local_rank,
                                                              G::RankerAggregate::Value);
                auto isolated=F::planIsolatedRKSite(problem,site,physical,semantics);
                try {
                    auto executable=F::makeSelectedIsolatedBenchmark(problem,site,std::move(isolated),physical,
                        context(dtype,spec.value_op,r),1.0f,std::numeric_limits<uint64_t>::max(),stream);
                    max_scratch=std::max<uint64_t>(max_scratch,executable.scratch_bytes);
                    Timed timed{familyName(physical.family),std::move(executable),{},
                        Training::recordPrefix(spec,op.name,familyName(physical.family),physicalName(physical),
                                               plan.local_rank,position+1,features,plan.staged()),
                        position+1,{}};
                    cell->candidates.push_back(std::move(timed));
                } catch (const F::UnavailableIsolatedCandidate& e) {
                    // A missing candidate would silently corrupt the oracle the labels are measured against.
                    throw std::runtime_error("Retained candidate " + physicalName(physical) +
                        " cannot construct complete FP32 continuation: " + e.what());
                }
            }
            if (cell->candidates.empty()) throw std::logic_error("FP8 training cell has no retained VALUE candidate");
            auto [output,scratch]=arena.outputAndScratch(T::DataType::FP32,outer,k,max_scratch);
            cell->output=std::move(output);
            cell->scratch=std::move(scratch);
            outputViews(*cell,T::DataType::FP32,{outer,k});
            enqueue(cell);
            continue;
        }
        // ARG: the candidate generator production will rank (1135): raw twins of the VALUE MIN/MAX candidates (with
        // their local ranks), the single-shard rotated pass and the fallback depth.
        const ArgOp& op=arg_ops[spec.arg_op];
        const bool maximum=op.operation==T::CubArgReductionOp::ArgMax;
        const T::CubReductionOp value_op=maximum?T::CubReductionOp::Max:T::CubReductionOp::Min;
        auto semantics=C::makeValueReductionStageSemantics(value_op,C::DensePassRole::Complete,r);
        semantics.combine=maximum?C::CubReductionStageCombine::Maximum:C::CubReductionStageCombine::Minimum;
        semantics.input_transform=C::CubReductionStageInputTransform::Identity;
        const auto typed=Training::typedSemantics(dtype,semantics.combine,semantics.input_transform);
        const bool produce_value=spec.kind==Training::Kind::ArgValueIndex||
                                 spec.kind==Training::Kind::CarriedArgValueIndex;
        if (spec.kind==Training::Kind::CarriedArgIndex||spec.kind==Training::Kind::CarriedArgValueIndex) {
            // 1211: a carried-index site: every carried FP32 RK ARG pass (Fp32RK::fp32RKCarriedArgCandidates) over the
            // values and a cold carried index of the same shape, as launchFp32RKCarriedArgPass runs it (staged
            // passes include the wide continuation); row stride 1 (an original run). The FP32 ranker's features with
            // the carried aggregate (3 index, 4 value and index) and the installed model's margin (unvalidated there).
            const T::DataType carried_dtype=*settings.carried_index;
            const auto args=C::Fp32RK::fp32RKCarriedArgCandidates(outer,r,k,std::numeric_limits<uint64_t>::max());
            if (args.empty()) throw std::logic_error("FP32 carried training cell has no ARG candidate");
            const auto ranker_pool=C::Fp32RK::fp32RKArgRankerPool(args);
            const C::Fp32RK::Fp32RKSite fp32_site{outer,r,k,static_cast<uint32_t>(sm),semantics};
            const uint32_t aggregate=produce_value?4U:3U;
            const std::vector<uint64_t> dims{outer,r,k};
            const T::CubReductionGeometry geometry=T::CubArgReduction::analyzeDenseGeometry(dims,{1});
            const std::vector<uint64_t> out_dims{outer,k};
            const uint64_t value_bytes=
                produce_value?T::TensorDescriptor(T::DataType::FP32,out_dims).getArraySizeInBytes():0;
            const uint64_t index_offset=CT::align256(value_bytes);
            const uint64_t index_bytes=T::TensorDescriptor(T::DataType::UINT32,out_dims).getArraySizeInBytes();
            auto cell=std::make_shared<Cell>(arena.input(dtype,outer,r,k),dtype,outer,r,k,op.name);
            cell->carried_dtype=carried_dtype;
            outputViews(*cell,T::DataType::UINT8,{index_offset+index_bytes});
            for (size_t position=0;position<args.size();++position) {
                const auto& candidate=args[position];
                const auto features=C::Fp32RK::fp32RKRankerFeatures(ranker_pool[position],fp32_site,aggregate);
                void* wide=arena.packedArgScratch(candidate.scratch_bytes);
                const F::PhysicalKernel identity=candidate.identity;
                Timed timed;
                timed.family=familyName(identity.family);
                timed.direct_carried=[identity,geometry,wide,maximum,produce_value,out_dims,index_offset]
                    (const T::Tensor& in,const T::Tensor& carried,T::Tensor& out,Stream& s) {
                        std::optional<T::Tensor> value;
                        if (produce_value) value=out.aliasByteStorageAs(T::DataType::FP32,out_dims,0);
                        T::Tensor index=out.aliasByteStorageAs(T::DataType::UINT32,out_dims,index_offset);
                        C::Fp32RK::launchFp32RKCarriedArgPass(identity,in,carried,value?&*value:nullptr,&index,
                                                              geometry,wide,maximum,1,s);
                    };
                timed.name=Training::recordPrefix(spec,op.name,familyName(identity.family),physicalName(identity),
                                                  candidate.local_rank,position+1,features,candidate.scratch_bytes!=0,
                                                  C::Fp32RK::Fp32RKRankerModel::margin(features));
                timed.rank=position+1;
                cell->candidates.push_back(std::move(timed));
            }
            enqueue(cell);
            continue;
        }
        if (dtype==T::DataType::FP16||dtype==T::DataType::BF16) {
            // 1251: every FP16 / BF16 RK ARG candidate (F16RK::f16RKArgCandidates over the 2-byte VALUE MIN / MAX pool,
            // the selector's candidates), at its position and local rank, with the 2-byte ranker's ARG features and
            // margin, timed as launchF16RKArgPass runs it (staged passes include their packed continuation).
            C::F16RK::F16RKSite f16_site;
            f16_site.outer=outer;
            f16_site.r=r;
            f16_site.k=k;
            f16_site.sm_count=static_cast<uint32_t>(sm);
            f16_site.input_dtype=dtype;
            f16_site.semantics=semantics;
            const auto pool=C::F16RK::f16RKCandidatePool(f16_site,[&](const F::PhysicalKernel& p) {
                return C::F16RK::f16RKCompiledResidency(p,dtype,k,semantics);
            });
            const auto args=C::F16RK::f16RKArgCandidates(pool,outer,r,k,std::numeric_limits<uint64_t>::max());
            if (args.empty()) throw std::logic_error("FP16 / BF16 training cell has no ARG candidate");
            const auto ranker_pool=C::F16RK::f16RKArgRankerPool(args);
            const uint32_t aggregate=produce_value?2U:1U;
            if (settings.training_stamp) {
                C::DenseReductionOutputSpec out_spec;
                out_spec.produce_value=produce_value;
                out_spec.value_dtype=T::DataType::FP32;
                out_spec.produce_index=true;
                out_spec.index_dtype=T::DataType::UINT32;
                const auto problem=C::makeInitialDenseReductionProblem({outer,r,k},{1},
                                                                       C::DenseReductionAggregateKind::Arg,out_spec);
                const auto site=C::enumerateDenseReductionSites(problem).front();
                stampRow(spec.id,[&]{
                    return C::F16RK::autotuneArgSite(problem,site,maximum,r,arena.input(dtype,outer,r,k),stream,
                                                     std::numeric_limits<uint64_t>::max(),
                                                     T::CubReductionTuningDepth::Regular);
                },[](const F::Fp8RKSelection& selection){ return physicalName(selection.physical); });
            }
            const std::vector<uint64_t> dims{outer,r,k};
            const T::CubReductionGeometry geometry=T::CubArgReduction::analyzeDenseGeometry(dims,{1});
            const std::vector<uint64_t> out_dims{outer,k};
            const uint64_t value_bytes=
                produce_value?T::TensorDescriptor(T::DataType::FP32,out_dims).getArraySizeInBytes():0;
            const uint64_t index_offset=CT::align256(value_bytes);
            const uint64_t index_bytes=T::TensorDescriptor(T::DataType::UINT32,out_dims).getArraySizeInBytes();
            auto cell=std::make_shared<Cell>(arena.input(dtype,outer,r,k),dtype,outer,r,k,op.name);
            outputViews(*cell,T::DataType::UINT8,{index_offset+index_bytes});
            for (size_t position=0;position<args.size();++position) {
                const auto& candidate=args[position];
                const auto features=C::F16RK::f16RKRankerFeatures(ranker_pool[position],f16_site,aggregate);
                uint64_t* packed=static_cast<uint64_t*>(arena.packedArgScratch(candidate.scratch_bytes));
                const F::PhysicalKernel identity=candidate.identity;
                cell->candidates.push_back({familyName(identity.family),std::nullopt,
                    [identity,geometry,packed,maximum,produce_value,out_dims,index_offset]
                    (const T::Tensor& in,T::Tensor& out,Stream& s) {
                        std::optional<T::Tensor> value;
                        if (produce_value) value=out.aliasByteStorageAs(T::DataType::FP32,out_dims,0);
                        T::Tensor index=out.aliasByteStorageAs(T::DataType::UINT32,out_dims,index_offset);
                        C::F16RK::launchF16RKArgPass(identity,in,value?&*value:nullptr,&index,geometry,packed,
                                                     maximum,1,s);
                    },
                    Training::recordPrefix(spec,op.name,familyName(identity.family),physicalName(identity),
                                           candidate.local_rank,position+1,features,candidate.scratch_bytes!=0,
                                           C::F16RK::F16RKRankerModel::margin(features)),
                    position+1,{}});
            }
            enqueue(cell);
            continue;
        }
        if (dtype==T::DataType::FP32) {
            // 1205: every FP32 RK ARG candidate (Fp32RK::fp32RKArgCandidates over the VALUE MIN / MAX pool, the
            // selector's candidates), at its position and local rank, with the FP32 ranker's ARG features and margin,
            // timed as launchFp32RKArgPass runs it (staged passes include their packed continuation).
            const C::Fp32RK::Fp32RKSite fp32_site{outer,r,k,static_cast<uint32_t>(sm),semantics};
            const auto pool=C::Fp32RK::fp32RKCandidatePool(fp32_site,[&](const F::PhysicalKernel& p) {
                return C::Fp32RK::fp32RKCompiledResidency(p,k,semantics);
            });
            const auto args=C::Fp32RK::fp32RKArgCandidates(pool,outer,r,k,std::numeric_limits<uint64_t>::max());
            if (args.empty()) throw std::logic_error("FP32 training cell has no ARG candidate");
            const auto ranker_pool=C::Fp32RK::fp32RKArgRankerPool(args);
            const uint32_t aggregate=produce_value?2U:1U;
            if (settings.training_stamp) {
                C::DenseReductionOutputSpec out_spec;
                out_spec.produce_value=produce_value;
                out_spec.value_dtype=T::DataType::FP32;
                out_spec.produce_index=true;
                out_spec.index_dtype=T::DataType::UINT32;
                const auto problem=C::makeInitialDenseReductionProblem({outer,r,k},{1},
                                                                       C::DenseReductionAggregateKind::Arg,out_spec);
                const auto site=C::enumerateDenseReductionSites(problem).front();
                stampRow(spec.id,[&]{
                    return C::Fp32RK::autotuneArgSite(problem,site,maximum,r,arena.input(dtype,outer,r,k),stream,
                                                      std::numeric_limits<uint64_t>::max(),
                                                      T::CubReductionTuningDepth::Regular);
                },[](const F::Fp8RKSelection& selection){ return physicalName(selection.physical); });
            }
            const std::vector<uint64_t> dims{outer,r,k};
            const T::CubReductionGeometry geometry=T::CubArgReduction::analyzeDenseGeometry(dims,{1});
            const std::vector<uint64_t> out_dims{outer,k};
            const uint64_t value_bytes=
                produce_value?T::TensorDescriptor(T::DataType::FP32,out_dims).getArraySizeInBytes():0;
            const uint64_t index_offset=CT::align256(value_bytes);
            const uint64_t index_bytes=T::TensorDescriptor(T::DataType::UINT32,out_dims).getArraySizeInBytes();
            auto cell=std::make_shared<Cell>(arena.input(dtype,outer,r,k),dtype,outer,r,k,op.name);
            outputViews(*cell,T::DataType::UINT8,{index_offset+index_bytes});
            for (size_t position=0;position<args.size();++position) {
                const auto& candidate=args[position];
                const auto features=C::Fp32RK::fp32RKRankerFeatures(ranker_pool[position],fp32_site,aggregate);
                uint64_t* packed=static_cast<uint64_t*>(arena.packedArgScratch(candidate.scratch_bytes));
                const F::PhysicalKernel identity=candidate.identity;
                cell->candidates.push_back({familyName(identity.family),std::nullopt,
                    [identity,geometry,packed,maximum,produce_value,out_dims,index_offset]
                    (const T::Tensor& in,T::Tensor& out,Stream& s) {
                        std::optional<T::Tensor> value;
                        if (produce_value) value=out.aliasByteStorageAs(T::DataType::FP32,out_dims,0);
                        T::Tensor index=out.aliasByteStorageAs(T::DataType::UINT32,out_dims,index_offset);
                        C::Fp32RK::launchFp32RKArgPass(identity,in,value?&*value:nullptr,&index,geometry,packed,
                                                       maximum,1,s);
                    },
                    Training::recordPrefix(spec,op.name,familyName(identity.family),physicalName(identity),
                                           candidate.local_rank,position+1,features,candidate.scratch_bytes!=0,
                                           C::Fp32RK::Fp32RKRankerModel::margin(features)),
                    position+1,{}});
            }
            enqueue(cell);
            continue;
        }
        const auto value_plans=retainedChoices(outer,r,k,static_cast<uint32_t>(sm),dtype,T::DataType::FP32,semantics,"");
        std::vector<C::RetainedFp8ValueSource> sources;
        sources.reserve(value_plans.size());
        for (const auto& plan : value_plans) sources.push_back({bind(plan),plan.local_rank});
        const auto args=C::retainedFp8ArgCandidates(sources,dtype,outer,r,k);
        if (args.empty()) throw std::logic_error("FP8 training cell has no retained ARG candidate");
        if (settings.training_stamp) {
            C::DenseReductionOutputSpec out_spec;
            out_spec.produce_value=produce_value;
            out_spec.value_dtype=T::DataType::FP32;
            out_spec.produce_index=true;
            out_spec.index_dtype=T::DataType::UINT32;
            const auto problem=C::makeInitialDenseReductionProblem({outer,r,k},{1},C::DenseReductionAggregateKind::Arg,
                                                                   out_spec);
            const auto site=C::enumerateDenseReductionSites(problem).front();
            stampRow(spec.id,[&]{
                return F::autotuneArgRKSite(problem,site,maximum,r,arena.input(dtype,outer,r,k),stream,
                                            std::numeric_limits<uint64_t>::max(),T::CubReductionTuningDepth::Regular);
            },[&](const F::Fp8RKSelection& selection){
                const auto plan=C::retainedFp8ArgPlanForIdentity(selection.physical,dtype,outer,r,k);
                return plan ? Training::argPlanName(*plan,selection.physical) : physicalName(selection.physical);
            });
        }
        const std::vector<uint64_t> dims{outer,r,k};
        const T::CubReductionGeometry geometry=T::CubArgReduction::analyzeDenseGeometry(dims,{1});
        // 1164: one rotating UINT8 output view per launch holds the FP32 value (when produced) and, 256-byte aligned
        // after it, the UINT32 index, as production's ARG contest lays them out.
        const std::vector<uint64_t> out_dims{outer,k};
        const uint64_t value_bytes=produce_value?T::TensorDescriptor(T::DataType::FP32,out_dims).getArraySizeInBytes():0;
        const uint64_t index_offset=CT::align256(value_bytes);
        const uint64_t index_bytes=T::TensorDescriptor(T::DataType::UINT32,out_dims).getArraySizeInBytes();
        auto cell=std::make_shared<Cell>(arena.input(dtype,outer,r,k),dtype,outer,r,k,op.name);
        outputViews(*cell,T::DataType::UINT8,{index_offset+index_bytes});
        for (size_t position=0;position<args.size();++position) {
            const auto& candidate=args[position];
            const auto features=G::Detail::rankerFeatures(F::rankerPlanForKernel(candidate.identity,candidate.local_rank),
                                                  outer,r,k,typed,candidate.local_rank,
                                                  produce_value?G::RankerAggregate::ArgValueIndex:
                                                                G::RankerAggregate::ArgIndex);
            uint32_t* packed=static_cast<uint32_t*>(arena.packedArgScratch(C::retainedFp8ArgScratchBytes(candidate.plan)));
            auto plan=std::make_shared<const C::DenseRetainedFp8ArgPhysicalPlan>(candidate.plan);
            cell->candidates.push_back({familyName(candidate.plan.family),std::nullopt,
                [plan,geometry,packed,maximum,produce_value,out_dims,index_offset]
                (const T::Tensor& in,T::Tensor& out,Stream& s) {
                    std::optional<T::Tensor> value;
                    if (produce_value) value=out.aliasByteStorageAs(T::DataType::FP32,out_dims,0);
                    T::Tensor index=out.aliasByteStorageAs(T::DataType::UINT32,out_dims,index_offset);
                    C::launchRetainedFp8ArgPlan(*plan,in,value?&*value:nullptr,&index,geometry,packed,maximum,1,s);
                },
                Training::recordPrefix(spec,op.name,familyName(candidate.plan.family),
                                       Training::argPlanName(candidate.plan,candidate.identity),candidate.local_rank,
                                       position+1,features,candidate.plan.shards>1),
                position+1,{}});
        }
        enqueue(cell);
    }
    queue.flush();
    const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cerr << "[fp8-ranker-training] cells=" << queue.completedCells() << " candidates=" << records.candidates
              << " wall_s=" << wall << " arena_wraps=" << arena.wraps() << '\n';
}
// 1133b --continuation-audit: are the production continuations fast? For every candidate of every cell, Complete
// candidates are timed once (variant complete); staged candidates three ways, interleaved with the rest of the cell:
//   first      the first stage alone, writing its partials;
//   full       first stage + production continuation (what --training times);
//   first_copy the first stage followed by a device-to-device copy of its partials (reads and writes every partial
//              byte, so a pessimistic stand-in for a bandwidth-bound continuation over the same, L2-warm, partials).
// continuation = full - first; copy reference = first_copy - first. fp8_continuation_audit.py reports the
// continuations against that reference and how much the measured oracle would change if they ran at copy speed.
std::string continuationName(const F::SelectedIsolatedBenchmark& executable) {
    std::string name;
    for (const auto& pass : executable.continuation) {
        const auto& plan=*pass.selected.physical_plan;
        const char* mangled=typeid(plan).name();
        int status=0;
        std::unique_ptr<char,void(*)(void*)> demangled(abi::__cxa_demangle(mangled,nullptr,nullptr,&status),std::free);
        std::string type=status==0&&demangled?demangled.get():mangled;
        const size_t colon=type.rfind("::");
        if (colon!=std::string::npos) type=type.substr(colon+2);
        const char* family=pass.selected.family==C::DenseReducerFamily::R?"R":
                           pass.selected.family==C::DenseReducerFamily::KR?"KR":"RK";
        name+=(name.empty()?"":"+")+std::string(family)+":"+type;
    }
    return name;
}

void runContinuationAudit(const Settings& settings) {
    const T::TensorPlacement placement(T::TensorPlacement::MemDevices::GPU,0);
    ScopedGpu gpu(0);
    Stream stream(placement);
    int sm=0;
    check(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0),"get SM count");
    if (sm<=0) throw std::runtime_error("No CUDA SMs");
    if (!settings.family.empty()) throw std::invalid_argument("--continuation-audit times every family; --family is not allowed");
    const std::vector<uint64_t> targets=settings.targets_given?settings.target_mib:std::vector<uint64_t>{96,192};
    const std::vector<T::DataType> dtypes=settings.dtypes_given?settings.dtypes:std::vector<T::DataType>{T::DataType::FP8_E4M3};
    const std::vector<std::string> names=settings.operations.empty()?
        std::vector<std::string>{"sum_identity","max_identity","argmax"}:settings.operations;
    struct Spec { uint64_t outer,r,k,target; T::DataType dtype; std::string op; };
    std::vector<Spec> specs;
    for (const T::DataType dtype : dtypes)
        for (const std::string& op : names)
            for (const auto& [r,k] : allShapes())
                for (uint64_t t : targets) specs.push_back({outerNear(r,k,t*MiB),r,k,t,dtype,op});
    const uint64_t total=std::min<uint64_t>(specs.size(),settings.max_cells);
    Arena arena(placement,stream,settings.arena_bytes);
    B::Fp8OracleBatchQueue queue(128,Passes,[&]{stream.synchronize();});
    const auto start=std::chrono::steady_clock::now();
    std::cout << "# fp8_continuation_audit=1 version=1133b cells=" << total << " samples=" << Passes
              << " interleaved=1 cold_l2=1 value_output=fp32 arg_index=uint32 arg_outputs="
              << (settings.arg_produce_value?"value_index":"index") << " sm_count=" << sm << '\n'
              << "kind,outer,R,K,target_mib,dtype,operation,family,configuration,shards,partial_bytes,continuation,"
                 "variant,total_median_ms,sample0_ms,sample1_ms,sample2_ms,sample3_ms,sample4_ms\n";
    RecordQueue records(stream,arena,queue,total,start,"fp8-continuation-audit");
    // 1229: per (dtype, VALUE op, R), with the original semantics the retained FP32 continuation selectors need.
    std::map<std::tuple<int,size_t,uint64_t>,std::unique_ptr<C::DenseProductionPhysicalSelectionContext>> contexts;
    uint8_t* const partials_base=arena.rawPartials();
    uint8_t* const copy_target=arena.rawWorkspace()+(1ULL<<30);  // ARG outputs sit at the workspace start
    for (uint64_t n=0;n<total;++n) {
        const Spec& spec=specs[n];
        const uint64_t outer=spec.outer, r=spec.r, k=spec.k;
        const auto prefix=[&](const char* kind,const char* family,const std::string& configuration,uint64_t shards,
                              uint64_t partial_bytes,const std::string& continuation,const char* variant) {
            std::ostringstream line;
            line << kind << ',' << outer << ',' << r << ',' << k << ',' << spec.target << ',' << dtypeName(spec.dtype)
                 << ',' << spec.op << ',' << family << ',' << configuration << ',' << shards << ',' << partial_bytes
                 << ',' << continuation << ',' << variant;
            return line.str();
        };
        auto cell=std::make_shared<Cell>(arena.input(spec.dtype,outer,r,k),spec.dtype,outer,r,k,spec.op);
        Cell* const raw=cell.get();  // candidate closures must not own their cell
        const auto value_op=std::find_if(ops.begin(),ops.end(),[&](const Op& op){return spec.op==op.name;});
        if (value_op!=ops.end()) {
            const size_t op_index=static_cast<size_t>(value_op-ops.begin());
            auto& slot=contexts[{static_cast<int>(spec.dtype),op_index,r}];
            if (!slot) slot=std::make_unique<C::DenseProductionPhysicalSelectionContext>(
                spec.dtype,C::denseRKValueOperation(value_op->operation),stream,r,1.0f,value_op->operation);
            const auto problem=[&]{
                C::DenseReductionOutputSpec out;
                out.value_dtype=T::DataType::FP32;
                return C::makeInitialDenseReductionProblem({outer,r,k},{1},C::DenseReductionAggregateKind::Value,out);
            }();
            const auto site=C::enumerateDenseReductionSites(problem).front();
            auto semantics=C::makeValueReductionStageSemantics(value_op->operation,C::DensePassRole::Complete,r);
            semantics.combine=value_op->combine;
            semantics.input_transform=value_op->transform;
            uint64_t max_scratch=0;
            for (const auto& plan : retainedChoices(outer,r,k,static_cast<uint32_t>(sm),spec.dtype,T::DataType::FP32,
                                                    semantics,"")) {
                const auto physical=bind(plan);
                auto executable=F::makeSelectedIsolatedBenchmark(problem,site,F::planIsolatedRKSite(problem,site,physical,
                    semantics),physical,slot->context(),1.0f,std::numeric_limits<uint64_t>::max(),stream);
                max_scratch=std::max<uint64_t>(max_scratch,executable.scratch_bytes);
                const std::string configuration=physicalName(physical);
                const char* family=familyName(physical.family);
                if (executable.continuation.empty()) {
                    cell->candidates.push_back({family,std::move(executable),{},
                        prefix("value",family,configuration,1,0,"","complete"),0,{}});
                    continue;
                }
                const uint64_t shards=executable.site.layout.shards;
                const uint64_t partial_bytes=outer*shards*k*sizeof(float);
                const std::string continuation=continuationName(executable);
                auto shared=std::make_shared<const F::SelectedIsolatedBenchmark>(executable);
                const auto first=[shared,raw,shards](const T::Tensor& in,Stream& s) {
                    T::Tensor first_input=in;
                    first_input.reshape({shared->site.outer,shared->site.r,shared->site.k});
                    T::Tensor partials=raw->scratch.aliasByteStorageAs(T::DataType::FP32,
                        {shared->site.outer,shards,shared->site.k},shared->first_partial_offset);
                    F::launchPhysicalFirstStage(shared->first_physical,shared->site.first_stage_semantics,first_input,
                                                partials,shared->first_geometry,1.0f,s);
                };
                cell->candidates.push_back({family,std::nullopt,[first](const T::Tensor& in,T::Tensor&,Stream& s){first(in,s);},
                    prefix("value",family,configuration,shards,partial_bytes,continuation,"first"),0,{}});
                cell->candidates.push_back({family,std::move(executable),{},
                    prefix("value",family,configuration,shards,partial_bytes,continuation,"full"),0,{}});
                if (partial_bytes<=(1ULL<<30))
                    cell->candidates.push_back({family,std::nullopt,
                        [first,shared,raw,partial_bytes,partials_base](const T::Tensor& in,T::Tensor&,Stream& s){
                            first(in,s);
                            check(cudaMemcpyAsync(partials_base,raw->scratch.getMemPtr<uint8_t>()+shared->first_partial_offset,
                                                  partial_bytes,cudaMemcpyDeviceToDevice,s.getStream()),"partials copy");
                        },prefix("value",family,configuration,shards,partial_bytes,continuation,"first_copy"),0,{}});
            }
            auto [output,scratch]=arena.outputAndScratch(T::DataType::FP32,outer,k,max_scratch);
            cell->output=std::move(output);
            cell->scratch=std::move(scratch);
        } else {
            const auto arg_op=std::find_if(arg_ops.begin(),arg_ops.end(),[&](const ArgOp& op){return spec.op==op.name;});
            if (arg_op==arg_ops.end()) throw std::invalid_argument("Unknown operation: "+spec.op);
            const bool maximum=arg_op->operation==T::CubArgReductionOp::ArgMax;
            auto semantics=C::makeValueReductionStageSemantics(maximum?T::CubReductionOp::Max:T::CubReductionOp::Min,
                                                               C::DensePassRole::Complete,r);
            semantics.combine=maximum?C::CubReductionStageCombine::Maximum:C::CubReductionStageCombine::Minimum;
            semantics.input_transform=C::CubReductionStageInputTransform::Identity;
            std::vector<C::RetainedFp8ValueSource> sources;
            for (const auto& plan : retainedChoices(outer,r,k,static_cast<uint32_t>(sm),spec.dtype,T::DataType::FP32,
                                                    semantics,""))
                sources.push_back({bind(plan),plan.local_rank});
            const std::vector<uint64_t> dims{outer,r,k};
            const T::CubReductionGeometry geometry=T::CubArgReduction::analyzeDenseGeometry(dims,{1});
            auto out=arena.argOutputs(settings.arg_produce_value?std::optional<T::DataType>(T::DataType::FP32):
                                                                 std::nullopt,outer,k,0);
            for (const auto& candidate : C::retainedFp8ArgCandidates(sources,spec.dtype,outer,r,k)) {
                auto plan=std::make_shared<const C::DenseRetainedFp8ArgPhysicalPlan>(candidate.plan);
                const uint64_t partial_bytes=C::retainedFp8ArgScratchBytes(*plan);
                uint32_t* packed=static_cast<uint32_t*>(arena.packedArgScratch(partial_bytes));
                const char* family=familyName(plan->family);
                const std::string configuration=Training::argPlanName(*plan,candidate.identity);
                auto full=[plan,geometry,packed,maximum,value=out.value,index=out.index]
                          (const T::Tensor& in,T::Tensor&,Stream& s) mutable {
                    C::launchRetainedFp8ArgPlan(*plan,in,value?&*value:nullptr,index?&*index:nullptr,geometry,packed,
                                                maximum,1,s);
                };
                if (!plan->packed_partials) {
                    cell->candidates.push_back({family,std::nullopt,full,
                        prefix("arg",family,configuration,1,0,"","complete"),0,{}});
                    continue;
                }
                // The packed first stage alone: launchRetainedFp8ArgPlan's staged branch without the continuation.
                const auto first=[plan,geometry,packed,maximum](const T::Tensor& in,Stream& s) {
                    if (plan->family==F::PhysicalFamily::DeepRSmallK)
                        C::launchFp8DeepRSmallKArgFirstStagePacked(in,packed,geometry,plan->rows_per_shard,plan->shards,
                                                                   maximum,s);
                    else if (plan->family==F::PhysicalFamily::RuntimeK)
                        C::launchFp8RuntimeKArgPass(in,nullptr,nullptr,geometry,plan->rows_per_shard,plan->packet_bytes,
                                                    maximum,1,s,packed);
                    else
                        C::launchFp8AlignedKArgPass(in,nullptr,nullptr,geometry,plan->packet_bytes,plan->cta_threads,
                                                    plan->shards,maximum,1,s,packed);
                };
                cell->candidates.push_back({family,std::nullopt,[first](const T::Tensor& in,T::Tensor&,Stream& s){first(in,s);},
                    prefix("arg",family,configuration,plan->shards,partial_bytes,"packed","first"),0,{}});
                cell->candidates.push_back({family,std::nullopt,full,
                    prefix("arg",family,configuration,plan->shards,partial_bytes,"packed","full"),0,{}});
                cell->candidates.push_back({family,std::nullopt,
                    [first,packed,partial_bytes,copy_target](const T::Tensor& in,T::Tensor&,Stream& s){
                        first(in,s);
                        check(cudaMemcpyAsync(copy_target,packed,partial_bytes,cudaMemcpyDeviceToDevice,s.getStream()),
                              "packed partials copy");
                    },prefix("arg",family,configuration,plan->shards,partial_bytes,"packed","first_copy"),0,{}});
            }
        }
        if (cell->candidates.empty()) throw std::logic_error("Continuation audit cell has no candidate");
        records.enqueue(cell);
    }
    queue.flush();
    const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cerr << "[fp8-continuation-audit] cells=" << queue.completedCells() << " timed=" << records.candidates
              << " wall_s=" << wall << '\n';
}
} // namespace

int main(int argc,char**argv) {
    try {
        const Settings settings=parse(argc,argv);
        if (settings.training) runTraining(settings);
        else if (settings.continuation_audit) runContinuationAudit(settings);
        else run(settings);
        return 0;
    }
    catch (const std::exception& e) {std::cerr<<"FP8 retained cold-L2 sweep failed: "<<e.what()<<'\n';return 1;}
}
