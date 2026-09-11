#pragma once

#include <algorithm>
#include <cstdint>

namespace DeviceResidentRowGrouping {

constexpr uint32_t kThreadsPerCta = 256;
constexpr uint32_t kMaxRowsPerCta = kThreadsPerCta;
constexpr uint64_t kBaseTargetPayloadBytesPerCta = 1024;
constexpr uint64_t kWideRowThresholdBytes = 512;
constexpr uint64_t kExact512RowGroupingThreshold = 8192;
constexpr uint64_t kExact1024RowGroupingThreshold = 4096;
constexpr uint64_t kMeasuredPowerOfTwoTargetPayloadBytesPerCta = 2048;
constexpr uint64_t kWideRowMediumBatchThreshold = 8192;
constexpr uint64_t kWideRowLargeBatchThreshold = 16384;
constexpr uint64_t kWideRowMediumTargetPayloadBytesPerCta = 2048;
constexpr uint64_t kWideRowLargeTargetPayloadBytesPerCta = 4096;
constexpr uint64_t kTargetCtasForNarrowRows = 512;

// Return the largest power of two <= value, clamped to the row-grouping
// specializations supported by the residency kernels. A zero input means that
// the caller has less than one unit of batch-derived grouping available.
inline uint32_t floorPowerOfTwoRowsPerCta(uint64_t value) {
    if (value == 0) return 1;

    uint32_t result = 1;
    while (result < kMaxRowsPerCta &&
           static_cast<uint64_t>(result) * 2 <= value) {
        result *= 2;
    }
    return result;
}

// Most rows use the general payload budget below, but the measured grouping
// sweeps show two exact power-of-two widths can profitably share more rows per
// CTA earlier than their neighboring widths. Keep these exceptions narrow:
//
//   * 512-byte rows use a 2 KiB CTA payload budget from 8192 rows onward,
//     yielding 4 rows/CTA instead of 2.
//   * 1024-byte rows use a 2 KiB CTA payload budget from 4096 rows onward,
//     yielding 2 rows/CTA at 4096 while preserving the already-measured 8192
//     and 16384 behavior.
//
// Neighboring widths (511/513 and 1023/1025) remain on the general rule; the
// benchmark showed that broadening these exceptions would introduce regressions.
inline uint64_t targetPayloadBytesPerCta(uint64_t logicalRows,
                                         uint64_t rowBytes) {
    if (rowBytes == 512 && logicalRows >= kExact512RowGroupingThreshold) {
        return kMeasuredPowerOfTwoTargetPayloadBytesPerCta;
    }
    if (rowBytes == 1024 && logicalRows >= kExact1024RowGroupingThreshold &&
        logicalRows < kWideRowLargeBatchThreshold) {
        return kMeasuredPowerOfTwoTargetPayloadBytesPerCta;
    }
    if (rowBytes <= kWideRowThresholdBytes) {
        return kBaseTargetPayloadBytesPerCta;
    }
    if (logicalRows >= kWideRowLargeBatchThreshold) {
        return kWideRowLargeTargetPayloadBytesPerCta;
    }
    if (logicalRows >= kWideRowMediumBatchThreshold) {
        return kWideRowMediumTargetPayloadBytesPerCta;
    }
    return kBaseTargetPayloadBytesPerCta;
}

// Size the fixed 256-thread CTA from the two dimensions of useful work rather
// than from an SM count or wave-count target:
//
//   * logicalRows increases row packing once the input already contains enough
//     independent rows to launch roughly 512 CTAs at the narrower grouping.
//   * rowBytes limits how many rows share a CTA according to a payload budget.
//     The general budget remains 1 KiB for <=512-byte rows. For wider rows it
//     grows to 2 KiB at 8192 rows and 4 KiB at 16384 rows. Two measured exact
//     power-of-two widths (512 and 1024 bytes) opt into 2 KiB earlier where the
//     forced grouping sweep showed a gain without neighboring-width regressions.
//
// Both limits are rounded down to the power-of-two specializations implemented
// by the kernels, then the more conservative (smaller) grouping wins. Grid size
// remains ceil(logicalRows / rowsPerCta); there is no device-wide CTA cap and no
// dependency on SM count.
//
// Examples:
//   logicalRows=2048, rowBytes<=256  -> 4 rows/CTA
//   logicalRows=4096, rowBytes=32    -> 8 rows/CTA
//   logicalRows=4096, rowBytes=1024  -> 2 rows/CTA
//   logicalRows=8192, rowBytes=512   -> 4 rows/CTA
//   logicalRows=8192, rowBytes=1024  -> 2 rows/CTA
//   logicalRows=16384,rowBytes=1024  -> 4 rows/CTA
//   logicalRows=16384,rowBytes=2048  -> 2 rows/CTA
//   logicalRows=16384,rowBytes=4096  -> 1 row/CTA
inline uint32_t selectRowsPerCta(uint64_t logicalRows, uint64_t rowBytes) {
    if (logicalRows == 0 || rowBytes == 0) return 1;

    const uint64_t payloadBudget = targetPayloadBytesPerCta(logicalRows, rowBytes);
    const uint64_t maxRowsByPayload =
        std::max<uint64_t>(1, payloadBudget / rowBytes);
    const uint32_t rowsByPayload = floorPowerOfTwoRowsPerCta(maxRowsByPayload);

    const uint64_t batchGroupingUnits = logicalRows / kTargetCtasForNarrowRows;
    const uint32_t rowsByBatch = std::max<uint32_t>(
        4, floorPowerOfTwoRowsPerCta(batchGroupingUnits));

    return std::min(rowsByBatch, rowsByPayload);
}

}  // namespace DeviceResidentRowGrouping
