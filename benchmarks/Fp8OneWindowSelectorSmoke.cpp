// Standalone host-only invariant test:
// g++ -O2 -std=c++23 -I. benchmarks/Fp8OneWindowSelectorSmoke.cpp -o /tmp/ow-selector-smoke
#include "benchmarks/Fp8OneWindowSelector.h"
#include "benchmarks/Fp8GeneralRParallelSelector.h"
#include <cassert>
#include <iostream>
#include <variant>
using namespace ThorImplementation::CubReductionBenchmarking;
int main() {
    constexpr auto resident6 = [](Fp8OneWindowPolicy::Variant) {return 6u;};
    for (unsigned r=2; r<=2048; ++r) {
        for (unsigned k : {3u,4u,5u,7u,8u,9u,11u,13u,15u,16u,17u,19u,31u,32u,
                           63u,64u,127u,128u,129u,255u,256u,497u,512u}) {
            for (unsigned outer : {1u,17u,256u,4096u,65536u,1000000u}) {
                auto selected = Fp8OneWindowPolicy::select(outer, r, k, 170, resident6);
                assert(selected.has_value());
                assert(Fp8OneWindowPolicy::legal(*selected,r,k));
                bool promoted = false;
                for (const auto v: Fp8OneWindowPolicy::candidates) promoted |= v == *selected;
                assert(promoted);
                const auto general = Fp8GeneralRParallelPolicy::select(outer,r,k,170,resident6);
                assert(general && general->mode == Fp8GeneralRParallelPolicy::Mode::OneWindow);
                assert(std::get<Fp8GeneralRParallelPolicy::OneWindowConfiguration>(general->configuration).variant == *selected);
            }
        }
    }
    assert(!Fp8OneWindowPolicy::select(1,32,496,170,resident6));
    assert(!Fp8GeneralRParallelPolicy::select(1,32,496,170,resident6));
    assert(!Fp8OneWindowPolicy::select(0,32,32,170,resident6));
    assert(!Fp8OneWindowPolicy::select(1,32,32,0,resident6));
    constexpr auto resident0 = [](Fp8OneWindowPolicy::Variant) {return 0u;};
    assert(!Fp8GeneralRParallelPolicy::select(1,32,32,170,resident0));
    // New local selector is opt-in only; the generic select() above stays frozen.
    constexpr auto resident4 = [](Fp8DeepRSmallKPolicy::Variant) { return 4u; };
    constexpr auto resident0_deep = [](Fp8DeepRSmallKPolicy::Variant) { return 0u; };
    for (unsigned k : {1u,3u,7u,15u,31u,64u,127u,255u,256u}) {
        for (unsigned r : {65u,128u,256u,257u,512u,777u,1024u,1025u,4096u,16384u}) {
            for (unsigned outer : {1u,11u,512u,2000u,8192u,65536u}) {
                const auto selected = Fp8GeneralRParallelPolicy::selectDeepRSmallK(
                    outer, r, k, 170, resident4);
                assert(selected && selected->mode == Fp8GeneralRParallelPolicy::Mode::DeepRSmallK);
                const auto v = std::get<Fp8GeneralRParallelPolicy::DeepRSmallKConfiguration>(
                    selected->configuration).variant;
                assert(Fp8DeepRSmallKPolicy::legal(v,outer,r,k));
                assert(Fp8DeepRSmallKPolicy::packet_bytes == 16);
                const auto cost = Fp8DeepRSmallKPolicy::describe(v,outer,r,k,170,4);
                assert(cost.shards_per_output >= 2 && cost.full_shard_rounds >= 1);
                assert(cost.total_score > 0 && cost.fp32_partial_mb > 0);
            }
        }
    }
    // The generalized wrapper exposes completed local modes without an
    // implicit OneWindow/DeepRSmallK crossover decision.
    for (auto mode : {Fp8GeneralRParallelPolicy::Mode::OneWindow,
                      Fp8GeneralRParallelPolicy::Mode::DeepRSmallK}) {
        const auto local = Fp8GeneralRParallelPolicy::selectForMode(
            mode, 4096, 16384, 3, 170, resident6, resident4);
        assert(local && local->mode == mode);
        if (mode == Fp8GeneralRParallelPolicy::Mode::OneWindow)
            assert(std::holds_alternative<Fp8GeneralRParallelPolicy::OneWindowConfiguration>(local->configuration));
        else
            assert(std::holds_alternative<Fp8GeneralRParallelPolicy::DeepRSmallKConfiguration>(local->configuration));
    }
    for (auto mode : {Fp8GeneralRParallelPolicy::Mode::AlignedK,
                      Fp8GeneralRParallelPolicy::Mode::RuntimeK})
        assert(!Fp8GeneralRParallelPolicy::selectForMode(mode,4096,16384,3,170,resident6,resident4));
    assert(!Fp8GeneralRParallelPolicy::selectDeepRSmallK(1,64,3,170,resident4));
    assert(!Fp8GeneralRParallelPolicy::selectDeepRSmallK(0,512,7,170,resident4));
    assert(!Fp8GeneralRParallelPolicy::selectDeepRSmallK(10,512,0,170,resident4));
    assert(!Fp8GeneralRParallelPolicy::selectDeepRSmallK(10,512,257,170,resident4));
    assert(!Fp8GeneralRParallelPolicy::selectDeepRSmallK(10,512,7,0,resident4));
    assert(!Fp8GeneralRParallelPolicy::selectDeepRSmallK(10,512,7,170,resident0_deep));
    assert(Fp8DeepRSmallKPolicy::legal({1024}, 10, 4096, 31));
    assert(!Fp8DeepRSmallKPolicy::legal({1024}, 10, 1024, 31));
    constexpr auto only1024 = [](Fp8DeepRSmallKPolicy::Variant v) {
        return v.rows_per_shard == 1024 ? 6u : 0u;
    };
    assert(std::get<Fp8GeneralRParallelPolicy::DeepRSmallKConfiguration>(
        Fp8GeneralRParallelPolicy::selectDeepRSmallK(4096, 16384, 3, 170, only1024)->configuration)
        .variant.rows_per_shard == 1024);
    const auto deep_through_general = Fp8GeneralRParallelPolicy::selectForMode(
        Fp8GeneralRParallelPolicy::Mode::DeepRSmallK,4096,16384,3,170,resident0,only1024);
    assert(deep_through_general &&
        std::get<Fp8GeneralRParallelPolicy::DeepRSmallKConfiguration>(
            deep_through_general->configuration).variant.rows_per_shard == 1024);
    assert(!Fp8GeneralRParallelPolicy::selectForMode(
        Fp8GeneralRParallelPolicy::Mode::OneWindow,4096,16384,3,170,resident0,only1024));
    assert(!Fp8DeepRSmallKPolicy::legal({64}, UINT64_MAX, 1024, 31));
    constexpr auto only64 = [](Fp8DeepRSmallKPolicy::Variant v) {return v.rows_per_shard==64 ? 2u : 0u;};
    assert(std::get<Fp8GeneralRParallelPolicy::DeepRSmallKConfiguration>(
        Fp8GeneralRParallelPolicy::selectDeepRSmallK(10,512,255,170,only64)->configuration)
        .variant.rows_per_shard == 64);
    // Deliberately lean nine-combination AlignedK inventory and accepted
    // initial policy. Cross-mode/general default selection remains unchanged.
    namespace AK = Fp8AlignedKPolicy;
    assert(AK::candidates(17, 96, 32).size() == 9);  // 3 Complete + 6 Staged32
    assert(AK::candidates(17, 384, 18).size() == 7); // p2 Complete32 + 3 CTAs x 2 staged
    assert(AK::candidates(17, 384, 56).size() == 15); // 3 Complete + 6 x 2 staged
    assert(AK::candidates(17, 32, 56).size() == 3);  // 3 Complete only
    assert(AK::candidates(17, 448, 56).size() == 15); // Staged128 means 4 shards/112 rows
    assert(AK::legal({2, 32, 1}, 17, 448, 56));
    assert(AK::legal({8, 32, 1}, 17, 448, 56));
    assert(AK::legal({8, 64, 1}, 17, 448, 56));
    assert(!AK::legal({2, 64, 1}, 17, 448, 56));
    assert(!AK::legal({2, 128, 1}, 17, 448, 56));
    assert(!AK::legal({8, 128, 1}, 17, 448, 56));
    assert(AK::legal({2, 128, 4}, 17, 448, 56));
    assert(!AK::legal({4, 32, 4}, 17, 448, 56));
    assert(!AK::legal({8, 256, 4}, 17, 448, 56));
    assert(!AK::legal({8, 64, 7}, 17, 448, 56));
    for (const auto& v : AK::candidates(17, 6144, 56)) {
        assert(AK::legal(v, 17, 6144, 56));
        const auto desc = AK::describe(v, 17, 6144, 56, 170, 6);
        assert(desc.first_stage_ctas == 17 * v.shards_per_output * desc.component_tiles);
        assert(desc.rows_per_shard >= 1 && desc.cta_work_waves >= 1);
        assert(desc.intermediate_bytes == (v.shards_per_output == 1 ? 0ULL :
            17ULL * v.shards_per_output * 56ULL * sizeof(float)));
        Fp8GeneralRParallelPolicy::Configuration typed =
            Fp8GeneralRParallelPolicy::AlignedKConfiguration{v};
        assert(std::get<Fp8GeneralRParallelPolicy::AlignedKConfiguration>(typed).variant == v);
    }
    assert(!AK::legal({8, 64, 3}, 17, 160, 18));
    assert(!AK::legal({4, 64, 1}, 17, 160, 36));
    assert(!AK::legal({16, 64, 1}, 17, 160, 384));
    assert(!AK::legal({2, 64, 1}, UINT64_MAX, 160, 384));
    // AlignedK initial heuristic: validate legal inventory, nonresident exclusions,
    // and explicit typed mode selection without asserting near-oracle performance.
    constexpr auto aligned_resident = [](AK::Variant) { return 6u; };
    for (unsigned r : {32u,96u,224u,448u,1536u,7168u}) {
        for (unsigned k : {18u,24u,56u,128u,320u,672u,960u}) {
            for (unsigned outer : {1u,31u,1000u,8000u}) {
                auto pick = AK::select(outer,r,k,170,aligned_resident);
                assert(pick && AK::legal(*pick,outer,r,k));
                bool found = false;
                for (const auto& v : AK::candidates(outer,r,k)) found |= v == *pick;
                assert(found);
            }
        }
    }
    constexpr auto aligned_nonresident = [](AK::Variant) { return 0u; };
    assert(!AK::select(31,448,56,170,aligned_nonresident));
    assert(!AK::select(31,448,56,0,aligned_resident));
    assert(!AK::select(0,448,56,170,aligned_resident));
    assert(!AK::select(31,448,0,170,aligned_resident));
    assert(!AK::select(UINT64_MAX,448,56,170,aligned_resident));
    const auto aligned_choice = AK::select(4096,96,256,170,aligned_resident);
    assert(aligned_choice && aligned_choice->shards_per_output == 1);
    assert(AK::experimentalSelect(4096,96,256,170,aligned_resident) == aligned_choice);
    const auto nonresident8 = [](AK::Variant v) { return v.packet == 8 ? 0u : 6u; };
    const auto legal_fallback = AK::select(4096,96,256,170,nonresident8);
    assert(legal_fallback && legal_fallback->packet == 2);
    // Old two-residency callers are preserved and never invent AlignedK
    // occupancy. The explicit three-residency path selects AlignedK.
    assert(!Fp8GeneralRParallelPolicy::selectForMode(
        Fp8GeneralRParallelPolicy::Mode::AlignedK,4096,96,256,170,resident6,resident4));
    const auto via_general = Fp8GeneralRParallelPolicy::selectForMode(
        Fp8GeneralRParallelPolicy::Mode::AlignedK,4096,96,256,170,
        resident6,resident4,aligned_resident);
    assert(via_general && via_general->mode == Fp8GeneralRParallelPolicy::Mode::AlignedK);
    assert(std::get<Fp8GeneralRParallelPolicy::AlignedKConfiguration>(
        via_general->configuration).variant == *aligned_choice);
    const auto via_helper = Fp8GeneralRParallelPolicy::selectAlignedK(
        4096,96,256,170,aligned_resident);
    assert(via_helper && via_helper->mode == Fp8GeneralRParallelPolicy::Mode::AlignedK);
    assert(std::get<Fp8GeneralRParallelPolicy::AlignedKConfiguration>(
        via_helper->configuration).variant == *aligned_choice);
    assert(!Fp8GeneralRParallelPolicy::selectAlignedK(4096,96,256,170,aligned_nonresident));
    assert(!Fp8GeneralRParallelPolicy::selectAlignedK(4096,96,257,170,aligned_resident));
    assert(!Fp8GeneralRParallelPolicy::selectForMode(
        Fp8GeneralRParallelPolicy::Mode::AlignedK,4096,96,256,170,
        resident6,resident4,aligned_nonresident));
    // RuntimeK: typed selection, physical residency, lean compiled depths and
    // fail-closed malformed/unsupported geometry. No generalized crossover.
    namespace RK = Fp8RuntimeKPolicy;
    const RK::Candidate exact512{{RK::Layout::RotatedExact,0,512,1,512,24},
        504,21ULL*24*511*4,2};
    const RK::Candidate prefix512{{RK::Layout::RotatedPrefix,0,497,2,512,24},
        1008,21ULL*24*511*4,3};
    const RK::Candidate promoted1024{{RK::Layout::RotatedExact,0,512,1,1024,12},
        252,21ULL*12*511*4,2};
    const RK::Candidate unpromoted2048{{RK::Layout::RotatedExact,0,512,1,2048,6},
        126,21ULL*6*511*4,2};
    assert(RK::legal(exact512,21,12288,511,170));
    assert(RK::legal(prefix512,21,12288,511,170));
    assert(RK::legal(promoted1024,21,12288,511,170));
    assert(!RK::legal(unpromoted2048,21,12288,511,170));
    const std::array physical{exact512,prefix512,promoted1024,unpromoted2048};
    const auto runtime = Fp8GeneralRParallelPolicy::selectRuntimeK(21,12288,511,170,physical);
    assert(runtime && runtime->mode == Fp8GeneralRParallelPolicy::Mode::RuntimeK);
    const auto selected_runtime = std::get<Fp8GeneralRParallelPolicy::RuntimeKConfiguration>(
        runtime->configuration).variant;
    assert(selected_runtime == exact512.variant || selected_runtime == prefix512.variant ||
        selected_runtime == promoted1024.variant);
    const auto shortlist = RK::shortlist(21,12288,511,170,physical,5);
    assert(shortlist.size() == 3);
    assert(shortlist.front() == selected_runtime);
    assert(std::find(shortlist.begin(),shortlist.end(),unpromoted2048.variant) == shortlist.end());
    const auto top_two = RK::shortlist(21,12288,511,170,physical,2);
    assert(top_two.size() == 2 && top_two.front() == selected_runtime);
    assert(RK::shortlist(21,12288,511,170,physical,0).empty());
    assert(RK::shortlist(0,12288,511,170,physical,5).empty());
    assert(RK::shortlist(21,12288,511,0,physical,5).empty());
    const std::array duplicated{exact512,exact512,unpromoted2048};
    assert(RK::shortlist(21,12288,511,170,duplicated,5).size() == 1);
    const auto selected_from_general = Fp8GeneralRParallelPolicy::selectForMode(
        Fp8GeneralRParallelPolicy::Mode::RuntimeK,21,12288,511,170,
        resident6,resident4,aligned_resident,std::span<const RK::Candidate>(physical));
    assert(selected_from_general && selected_from_general->configuration == runtime->configuration);
    const auto promoted_only = Fp8GeneralRParallelPolicy::selectRuntimeK(21,12288,511,170,
        std::span<const RK::Candidate>(&promoted1024,1));
    assert(promoted_only &&
        std::get<Fp8GeneralRParallelPolicy::RuntimeKConfiguration>(
            promoted_only->configuration).variant == promoted1024.variant);
    assert(!Fp8GeneralRParallelPolicy::selectRuntimeK(21,12288,511,170,
        std::span<const RK::Candidate>(&unpromoted2048,1)));
    assert(!Fp8GeneralRParallelPolicy::selectRuntimeK(0,12288,511,170,physical));
    assert(!Fp8GeneralRParallelPolicy::selectRuntimeK(21,12288,511,0,physical));
    RK::Candidate bad_occupancy = exact512;
    bad_occupancy.resident_ctas_per_sm = 0;
    assert(!RK::legal(bad_occupancy,21,12288,511,170));
    RK::Candidate bad_grid = exact512;
    bad_grid.first_stage_ctas = 1;
    assert(!RK::legal(bad_grid,21,12288,511,170));
    RK::Candidate bad_scratch = exact512;
    bad_scratch.scratch_bytes -= 4;
    assert(!RK::legal(bad_scratch,21,12288,511,170));
    RK::Candidate invalid_overflow = exact512;
    assert(!RK::legal(invalid_overflow,UINT64_MAX,12288,511,170));
    // Existing three- and two-residency callers continue to fail closed for RuntimeK.
    assert(!Fp8GeneralRParallelPolicy::selectForMode(
        Fp8GeneralRParallelPolicy::Mode::RuntimeK,21,12288,511,170,
        resident6,resident4,aligned_resident));
    // The default general select() still selects only OneWindow.
    const auto default_choice = Fp8GeneralRParallelPolicy::select(
        4096,96,256,170,resident6);
    assert(default_choice && default_choice->mode == Fp8GeneralRParallelPolicy::Mode::OneWindow);
    std::cout << "OneWindow/DeepRSmallK/AlignedK local selection passes\n";
}
