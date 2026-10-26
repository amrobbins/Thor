#include "benchmarks/Fp8BenchmarkArenaCursor.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>

using ThorImplementation::CubReductionBenchmarking::Fp8BenchmarkArenaCursor;

int main() {
    using C = Fp8BenchmarkArenaCursor;
    C arena(1024);
    auto i0 = arena.getBenchmarkTensorView(33, C::Lifetime::StreamingInput);
    assert(i0.offset == 0 && i0.bytes == 33);
    auto i1 = arena.getBenchmarkTensorView(65, C::Lifetime::StreamingInput);
    assert(i1.offset == 192 && i1.bytes == 65);
    assert(i0.offset % 32 == 0 && i1.offset % 32 == 0);
    assert(i0.offset + i0.bytes + 128 <= i1.offset);

    // Stage sizes vary; their overlapping suffix reservations all leave a
    // 128-byte guard and maintain stable addresses for stamped continuations.
    auto s0 = arena.getBenchmarkTensorView(40, C::Lifetime::ReusableStage);
    auto s1 = arena.getBenchmarkTensorView(200, C::Lifetime::ReusableStage);
    assert(s0.offset == 1024 - 64 - 128);
    assert(s1.offset == 1024 - 224 - 128);
    assert(arena.workspaceBytes() == 352);
    assert(s0.offset >= s1.offset && s0.offset + 40 <= 1024 - 128);
    assert(s1.offset + 200 <= 1024 - 128);

    // A streaming allocation near the workspace wraps, rather than aliases
    // a staged output; the cursor continues across geometry boundaries.
    auto i2 = arena.getBenchmarkTensorView(300, C::Lifetime::StreamingInput);
    assert(i2.offset == 0 && arena.wraps() == 1);
    assert(i2.offset + i2.bytes + 128 <= s1.offset);
    // Reusable workspace only grows as necessary; never return stage-written
    // bytes to the cold input rotation, including across workload boundaries.
    assert(arena.workspaceBytes() == 352);
    auto i3 = arena.getBenchmarkTensorView(160, C::Lifetime::StreamingInput);
    assert(i3.offset == 0 && arena.wraps() == 2);
    auto s2 = arena.getBenchmarkTensorView(480, C::Lifetime::ReusableStage);
    assert(s2.offset == 416);
    assert(arena.workspaceBytes() == 608);
    auto i4 = arena.getBenchmarkTensorView(64, C::Lifetime::StreamingInput);
    assert(i4.offset == 0 && arena.wraps() == 3);
    assert(i4.offset + i4.bytes + 128 <= s2.offset);
    bool threw = false;
    try { (void)arena.getBenchmarkTensorView(700, C::Lifetime::StreamingInput); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    threw = false;
    try { (void)arena.getBenchmarkTensorView(1024, C::Lifetime::ReusableStage); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    return 0;
}
