// The cold-input benchmark arena's cursor (benchmarks/Fp8BenchmarkArenaCursor.h, used by the cold-L2 sweep that
// trains the FP8 RK ranker): streaming inputs advance and wrap below the stage workspace, stage views are stable
// overlapping suffixes, every view has an aligned start and a 128-byte guard, and the workspace only grows. Host-only.

#include "benchmarks/Fp8BenchmarkArenaCursor.h"

#include "gtest/gtest.h"

#include <stdexcept>

using Cursor = ThorImplementation::ReductionBenchmarking::Fp8BenchmarkArenaCursor;

TEST(Fp8BenchmarkArenaCursor, StreamingInputsAndStagesShareOneArena) {
    Cursor arena(1024, 32);  // 32-byte starts
    const auto i0 = arena.getBenchmarkTensorView(33, Cursor::Lifetime::StreamingInput);
    const auto i1 = arena.getBenchmarkTensorView(65, Cursor::Lifetime::StreamingInput);
    EXPECT_EQ(i0.offset, 0u);
    EXPECT_EQ(i0.bytes, 33u);
    EXPECT_EQ(i1.offset, 192u);
    EXPECT_EQ(i1.bytes, 65u);
    EXPECT_LE(i0.offset + i0.bytes + 128, i1.offset);

    // Stages of different sizes are overlapping suffixes with a 128-byte guard and stable addresses.
    const auto s0 = arena.getBenchmarkTensorView(40, Cursor::Lifetime::ReusableStage);
    const auto s1 = arena.getBenchmarkTensorView(200, Cursor::Lifetime::ReusableStage);
    EXPECT_EQ(s0.offset, 1024u - 64 - 128);
    EXPECT_EQ(s1.offset, 1024u - 224 - 128);
    EXPECT_EQ(arena.workspaceBytes(), 352u);
    EXPECT_GE(s0.offset, s1.offset);
    EXPECT_LE(s0.offset + 40, 1024u - 128);
    EXPECT_LE(s1.offset + 200, 1024u - 128);

    // An input that would reach the workspace wraps instead of aliasing a stage.
    const auto i2 = arena.getBenchmarkTensorView(300, Cursor::Lifetime::StreamingInput);
    EXPECT_EQ(i2.offset, 0u);
    EXPECT_EQ(arena.wraps(), 1u);
    EXPECT_LE(i2.offset + i2.bytes + 128, s1.offset);
    EXPECT_EQ(arena.workspaceBytes(), 352u);
    const auto i3 = arena.getBenchmarkTensorView(160, Cursor::Lifetime::StreamingInput);
    EXPECT_EQ(i3.offset, 0u);
    EXPECT_EQ(arena.wraps(), 2u);
    // The workspace grows only when a stage needs it, and stage bytes never return to the input ring.
    const auto s2 = arena.getBenchmarkTensorView(480, Cursor::Lifetime::ReusableStage);
    EXPECT_EQ(s2.offset, 416u);
    EXPECT_EQ(arena.workspaceBytes(), 608u);
    const auto i4 = arena.getBenchmarkTensorView(64, Cursor::Lifetime::StreamingInput);
    EXPECT_EQ(i4.offset, 0u);
    EXPECT_EQ(arena.wraps(), 3u);
    EXPECT_LE(i4.offset + i4.bytes + 128, s2.offset);

    EXPECT_THROW((void)arena.getBenchmarkTensorView(700, Cursor::Lifetime::StreamingInput), std::runtime_error);
    EXPECT_THROW((void)arena.getBenchmarkTensorView(1024, Cursor::Lifetime::ReusableStage), std::runtime_error);
}

TEST(Fp8BenchmarkArenaCursor, ViewsStartOn256ByteBoundariesByDefault) {
    Cursor arena(8192);
    const auto a0 = arena.getBenchmarkTensorView(33, Cursor::Lifetime::StreamingInput);
    const auto a1 = arena.getBenchmarkTensorView(129, Cursor::Lifetime::StreamingInput);
    const auto a2 = arena.getBenchmarkTensorView(1000, Cursor::Lifetime::StreamingInput);
    EXPECT_EQ(a0.offset, 0u);
    EXPECT_EQ(a1.offset, 256u);
    EXPECT_EQ(a2.offset, 768u);
    EXPECT_LE(a1.offset + a1.bytes + 128, a2.offset);
    const auto stage = arena.getBenchmarkTensorView(300, Cursor::Lifetime::ReusableStage);
    EXPECT_EQ(stage.offset % 256, 0u);
    EXPECT_LE(stage.offset + 300 + 128, 8192u);
    EXPECT_THROW(Cursor bad(1024, 48), std::invalid_argument);  // not a power of two
}
