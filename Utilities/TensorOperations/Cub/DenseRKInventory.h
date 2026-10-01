#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace ThorImplementation::CubReductionInternal {

/**
 * Durable strategy inventory for ordinary dense [O,R,K] reductions.
 *
 * These are the only strategies admitted to ReducersDenseRK. Packet width, CTA geometry, aligned/rotated/flat-row access, and
 * Complete/Staged progress are attributes beneath one of these strategies; they are not additional reducer families.
 */
enum class DenseRKStrategy : uint8_t {
    KParallel = 0,
    RCooperative = 1,
};

enum class DenseRKAccess : uint8_t {
    Aligned = 0,
    Rotated = 1,
    FlatRows = 2,
};

enum class DenseRKSemantics : uint8_t {
    Value = 0,
    Arg = 1,
};

/**
 * FamilyImplementation means the implementation belongs to the modern RK inventory and therefore represents exactly
 * one physical pass. LegacyOverlap is reserved for migration debt outside ordinary VALUE RK (currently ARG fallback).
 */
enum class DenseRKImplementationStatus : uint8_t {
    FamilyImplementation = 0,
    LegacyOverlap = 1,
};

enum class DenseRKProductionImplementation : uint8_t {
    // Modern VALUE implementations.
    KParallelPass = 0,
    RCooperativeAlignedComplete,
    RCooperativeAlignedStaged,
    RCooperativeRotatedStaged,
    RCooperativeFlatRowsStaged,


    // Modern ARG implementations. VALUE and ARG share the RK strategy vocabulary even when their CUDA kernels differ.
    AlignedAsyncNarrowTiledArg,
    AlignedCooperativeTiledArg,


    // Conservative ARG fallback retained until the shared RK family owns ARG execution.
    DirectTiledFixedSegmentArg,
};

struct DenseRKProductionInventoryEntry {
    DenseRKProductionImplementation implementation;
    DenseRKImplementationStatus status;
    DenseRKSemantics semantics;

    // Populated only for admitted modern family implementations. Non-family paths intentionally receive no strategy name.
    std::optional<DenseRKStrategy> strategy;
    std::optional<DenseRKAccess> access;

    // Physical progress supported by this implementation today. These are independent of semantic pass role.
    bool supports_complete = false;
    bool supports_staged = false;

    // True only when today's implementation still internally owns/forces a successor pass. Modern family
    // implementations must keep this false; only temporary compatibility/legacy paths may still own a successor.
    bool owns_successor = false;

    std::string_view symbol;
};

/** Exactly {KParallel, RCooperative}; no legacy implementation can extend this list. */
[[nodiscard]] std::span<const DenseRKStrategy> denseRKStrategies();

/** Production ordinary-dense RK implementation inventory. */
[[nodiscard]] std::span<const DenseRKProductionInventoryEntry> denseRKProductionInventory();

[[nodiscard]] const DenseRKProductionInventoryEntry& denseRKProductionInventoryEntry(
    DenseRKProductionImplementation implementation);

[[nodiscard]] bool denseRKIsFamilyImplementation(DenseRKProductionImplementation implementation);

/**
 * The benchmark harness exercises the production modern RK inventory directly.
 * No benchmark-only RK CUDA implementation is retained after the VALUE cutover.
 */
enum class DenseRKBenchmarkImplementation : uint8_t {
    FamilyProductionHarness = 0,
};

struct DenseRKBenchmarkInventoryEntry {
    DenseRKBenchmarkImplementation implementation;
    std::optional<DenseRKStrategy> strategy;
    bool owns_cuda_kernel = false;
    bool legacy_reference = false;
    std::string_view source_file;
};

[[nodiscard]] std::span<const DenseRKBenchmarkInventoryEntry> denseRKBenchmarkInventory();

/**
 * Special dense-view RK implementations are intentionally outside this ordinary-dense inventory for this overhaul.
 * Current examples include payload-transpose, pitched tiled, and wide shared-transpose output kernels. Their existing
 * ordained view routing remains unchanged until the later dense-view cleanup.
 */

}  // namespace ThorImplementation::CubReductionInternal
