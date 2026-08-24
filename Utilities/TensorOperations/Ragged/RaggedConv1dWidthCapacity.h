#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace ThorImplementation {

/**
 * Build the finite width-capacity family used by padded ragged Conv1D.
 *
 * The family is determined entirely from placement-time max_values_per_row and
 * never grows at runtime. Small widths use powers of two beginning at 8.
 * Starting at width 128, each power-of-two octave is split into quarters:
 *
 *   B, 1.25B, 1.5B, 1.75B, 2B
 *
 * The exact max_values_per_row is always appended as the final capacity when it
 * is not already canonical. This keeps runtime padding bounded while ensuring
 * the structural maximum is representable without introducing a new shape.
 *
 * Examples:
 *   5   -> {5}
 *   9   -> {8, 9}
 *   64  -> {8, 16, 32, 64}
 *   371 -> {8, 16, 32, 64, 128, 160, 192, 224, 256, 320, 371}
 *   818 -> {8, 16, 32, 64, 128, 160, 192, 224, 256, 320, 384,
 *           448, 512, 640, 768, 818}
 */
[[nodiscard]] std::vector<uint64_t> makeRaggedConv1dWidthCapacities(uint64_t maxValuesPerRow);

/**
 * Select the smallest placement-defined width that can contain the current
 * maximum active row length. An all-empty batch selects width zero and executes
 * as a no-op; zero is not part of the placement-time capacity family.
 */
[[nodiscard]] uint64_t chooseRaggedConv1dWidthCapacity(uint64_t maxActiveRowLength,
                                                       std::span<const uint64_t> widthCapacities);

}  // namespace ThorImplementation
