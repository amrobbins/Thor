#include "Utilities/TensorOperations/Ragged/RaggedConv1dWidthCapacity.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ThorImplementation {
namespace {

constexpr uint64_t MIN_CANONICAL_WIDTH = 8;
constexpr uint64_t QUARTER_OCTAVE_START_WIDTH = 128;

void appendIfWithin(std::vector<uint64_t>& capacities, uint64_t candidate, uint64_t maxValuesPerRow) {
    if (candidate <= maxValuesPerRow && (capacities.empty() || capacities.back() != candidate)) {
        capacities.push_back(candidate);
    }
}

}  // namespace

std::vector<uint64_t> makeRaggedConv1dWidthCapacities(uint64_t maxValuesPerRow) {
    if (maxValuesPerRow == 0) {
        throw std::invalid_argument("Ragged Conv1D max_values_per_row must be non-zero when building width capacities.");
    }
    if (maxValuesPerRow < MIN_CANONICAL_WIDTH) {
        return {maxValuesPerRow};
    }

    std::vector<uint64_t> capacities;
    for (uint64_t width = MIN_CANONICAL_WIDTH;
         width < QUARTER_OCTAVE_START_WIDTH && width <= maxValuesPerRow;
         width *= 2) {
        capacities.push_back(width);
    }
    appendIfWithin(capacities, QUARTER_OCTAVE_START_WIDTH, maxValuesPerRow);

    uint64_t octaveBase = QUARTER_OCTAVE_START_WIDTH;
    while (octaveBase < maxValuesPerRow) {
        const uint64_t quarter = octaveBase / 4;
        const uint64_t remaining = maxValuesPerRow - octaveBase;
        for (uint64_t quarterStep = 1; quarterStep <= 3; ++quarterStep) {
            const uint64_t delta = quarter * quarterStep;
            if (delta > remaining) {
                break;
            }
            appendIfWithin(capacities, octaveBase + delta, maxValuesPerRow);
        }

        if (octaveBase > std::numeric_limits<uint64_t>::max() / 2) {
            break;
        }
        const uint64_t nextOctave = octaveBase * 2;
        if (nextOctave > maxValuesPerRow) {
            break;
        }
        appendIfWithin(capacities, nextOctave, maxValuesPerRow);
        octaveBase = nextOctave;
    }

    appendIfWithin(capacities, maxValuesPerRow, maxValuesPerRow);
    return capacities;
}

uint64_t chooseRaggedConv1dWidthCapacity(uint64_t maxActiveRowLength,
                                         std::span<const uint64_t> widthCapacities) {
    if (maxActiveRowLength == 0) {
        return 0;
    }
    if (widthCapacities.empty()) {
        throw std::invalid_argument("Ragged Conv1D width-capacity family must be non-empty.");
    }

    const auto capacity = std::lower_bound(widthCapacities.begin(), widthCapacities.end(), maxActiveRowLength);
    if (capacity == widthCapacities.end()) {
        throw std::invalid_argument(
            "Ragged Conv1D max_active_row_length exceeds the placement-time max_values_per_row capacity family.");
    }
    return *capacity;
}

}  // namespace ThorImplementation
