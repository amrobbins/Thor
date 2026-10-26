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
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
struct Settings {
    std::string family;
    uint64_t arena_bytes = 12ULL << 30;
    uint64_t max_cells = std::numeric_limits<uint64_t>::max();
    bool historical_fp8_output = false;
};
Settings parse(int argc, char** argv) {
    Settings s;
    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        if (a.starts_with("--family=")) s.family = std::string(a.substr(9));
        else if (a.starts_with("--max-cells=")) s.max_cells = std::stoull(std::string(a.substr(12)));
        else if (a.starts_with("--arena-gib=")) s.arena_bytes = std::stoull(std::string(a.substr(12))) << 30;
        else if (a == "--output=same_fp8") s.historical_fp8_output = true;
        else if (a == "--output=fp32") s.historical_fp8_output = false;
        else if (a == "--help") {
            std::cout << "Usage: thor_fp8_retained_cold_l2_sweep [--family=one_window|deep_r_small_k|aligned_k|runtime_k]"
                      << " [--output=fp32|same_fp8] [--max-cells=N] [--arena-gib=12]\n";
            std::exit(0);
        } else throw std::invalid_argument("Unknown benchmark option: " + std::string(a));
    }
    if (!s.family.empty() && s.family != "one_window" && s.family != "deep_r_small_k" &&
        s.family != "aligned_k" && s.family != "runtime_k")
        throw std::invalid_argument("Unknown retained family filter");
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
uint64_t nearTargetOuter(uint64_t r, uint64_t k, uint64_t bytes) {
    // The historical fast sweep used ceil(target_bytes/(R*K)), not a fixed
    // output cardinality. Retain that choice for geometry/traffic matching.
    return std::max<uint64_t>(1, (bytes + (r*k) / 2) / (r*k));
}
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
    F::PhysicalKernel physical;
    F::SelectedIsolatedBenchmark executable;
    std::string name;
    size_t rank = 0;
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
    std::cout << '\n';
    std::cout << "family,outer,R,K,dtype,configuration,operation,rank12,rank16,rank20,rank24,rank28,rank32,rank40,rank48,rank128,"
                 "total_median_ms,sample0_ms,sample1_ms,sample2_ms,sample3_ms,sample4_ms\n";
    uint64_t prepared_cells=0;
    uint64_t collected_cells=0;
    const auto grid=shapes();
    for (T::DataType dtype : {T::DataType::FP8_E4M3,T::DataType::FP8_E5M2}) {
        for (const Op& op : ops) {
            C::DenseProductionPhysicalSelectionContext fp32_context(
                dtype, C::denseRKValueOperation(op.operation), stream);
            for (const auto [r,k] : grid) {
                for (uint64_t target : {24*MiB,96*MiB,192*MiB}) {
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
                            cell->candidates.push_back({physical,std::move(executable),
                                physicalName(physical),rank+1,{}});
                        } catch (const F::UnavailableIsolatedCandidate& e) {
                            // Missing candidates silently corrupt the measured oracle
                            // and can falsely validate the learned 28-item prefix.
                            throw std::runtime_error("Retained candidate " + physicalName(physical) +
                                " cannot construct complete FP32 continuation: " + e.what());
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
                        cell->candidates[index].executable.launch(cell->input,cell->output,cell->scratch,stream);
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
                            std::cout << familyName(item.physical.family)<<','<<cell->outer<<','<<cell->r<<','<<cell->k
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
