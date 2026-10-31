// Retained-only FP8 dense RK cold-L2 census. No experimental CUDA kernels.
// This is deliberately separate from production stamping and from the older
// calibration benchmark that was deleted in the cleanup.
#include "benchmarks/Fp8BenchmarkArenaCursor.h"
#include "benchmarks/Fp8OracleBatchQueue.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKBoundSelection.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8GeneralRParallelSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKRuntimeProposals.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

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
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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
    uint64_t arena_bytes = 12ULL << 30;
    uint64_t max_cells = std::numeric_limits<uint64_t>::max();
    bool historical_fp8_output = false;
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
            for (const auto& d : splitList(a.substr(9))) {
                if (d == "e4m3") s.dtypes.push_back(T::DataType::FP8_E4M3);
                else if (d == "e5m2") s.dtypes.push_back(T::DataType::FP8_E5M2);
                else throw std::invalid_argument("Unknown FP8 dtype (e4m3|e5m2): " + d);
            }
        } else if (a.starts_with("--targets-mib=")) {
            s.target_mib.clear();
            for (const auto& t : splitList(a.substr(14))) s.target_mib.push_back(std::stoull(t));
        } else if (a == "--shapes=onewindow") s.onewindow_shapes = true;
        else if (a == "--shapes=uncovered") s.uncovered_shapes = true;
        else if (a == "--shapes=replacement") s.replacement_shapes = true;
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
                      << "  --focus=replacement = OneWindow-relevant grid + uncovered-K + K=2 shapes, 24/96/192 MiB.\n";
            std::exit(0);
        } else throw std::invalid_argument("Unknown benchmark option: " + std::string(a));
    }
    if (!s.family.empty() && s.family != "one_window" && s.family != "deep_r_small_k" &&
        s.family != "aligned_k" && s.family != "runtime_k" && s.family != "one_window_rk")
        throw std::invalid_argument("Unknown retained family filter");
    if (s.family == "one_window_rk") s.onewindow_runtime_k = true;
    for (const auto& name : s.operations) {
        if (std::none_of(ops.begin(), ops.end(), [&](const Op& op) { return name == op.name; }))
            throw std::invalid_argument("Unknown operation: " + name);
    }
    if (s.dtypes.empty() || s.target_mib.empty()) throw std::invalid_argument("Empty dtype or target list");
    if (static_cast<int>(s.onewindow_shapes) + static_cast<int>(s.uncovered_shapes) +
            static_cast<int>(s.replacement_shapes) > 1)
        throw std::invalid_argument("Choose one of --shapes=onewindow, --shapes=uncovered, --shapes=replacement");
    if (s.arena_bytes < (1ULL << 30) || s.arena_bytes > (24ULL << 30))
        throw std::invalid_argument("Arena size must be 1..24 GiB");
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

uint64_t nearTargetOuter(uint64_t r, uint64_t k, uint64_t bytes) {
    // The historical fast sweep used ceil(target_bytes/(R*K)), not a fixed
    // output cardinality. Retain that choice for geometry/traffic matching.
    return std::max<uint64_t>(1, (bytes + (r*k) / 2) / (r*k));
}
const char* dtypeName(T::DataType dtype) { return dtype == T::DataType::FP8_E4M3 ? "fp8_e4m3" : "fp8_e5m2"; }
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
         cursor_(bytes), workspace_(placement, T::TensorDescriptor(T::DataType::UINT8, {2ULL << 30})) {
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
        const auto align32=[](uint64_t n){return (n+31ULL)&~31ULL;};
        const uint64_t scratch_offset=align32(output_bytes+128);
        if (scratch_offset > workspace_.getArraySizeInBytes()-128 ||
            scratch_bytes > workspace_.getArraySizeInBytes()-scratch_offset-128)
            throw std::runtime_error("Retained FP8 cold-L2 output/scratch exceeds 2 GiB workspace");
        T::Tensor out=workspace_.aliasByteStorageAs(output_dtype,dims,0);
        T::Tensor scratch;
        if (scratch_bytes) scratch=workspace_.aliasByteStorageAs(
            T::DataType::UINT8,{scratch_bytes},scratch_offset);
        return {std::move(out),std::move(scratch)};
    }
    uint64_t wraps() const { return cursor_.wraps(); }
private:
    T::Tensor bytes_;
    B::Fp8BenchmarkArenaCursor cursor_;
    T::Tensor workspace_;
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
};
struct Cell {
    T::Tensor input,output,scratch;
    std::vector<Timed> candidates;
    std::vector<Event> starts, stops;
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
    const auto grid=settings.uncovered_shapes?uncoveredKShapes():
                    settings.replacement_shapes?replacementShapes():shapes();
    for (T::DataType dtype : settings.dtypes) {
        for (const Op& op : ops) {
            if (!settings.operations.empty() &&
                std::find(settings.operations.begin(),settings.operations.end(),op.name)==settings.operations.end())
                continue;
            C::DenseProductionPhysicalSelectionContext fp32_context(
                dtype, C::denseRKValueOperation(op.operation), stream);
            for (const auto [r,k] : grid) {
                if (settings.onewindow_shapes && !oneWindowRelevantShape(r,k)) continue;
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
} // namespace

int main(int argc,char**argv) {
    try {run(parse(argc,argv));return 0;}
    catch (const std::exception& e) {std::cerr<<"FP8 retained cold-L2 sweep failed: "<<e.what()<<'\n';return 1;}
}
