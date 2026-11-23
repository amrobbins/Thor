#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ThorImplementation {

// Sequence concatenate consumes authoritative host partition structure but no
// device offsets representation. StampedNetwork derives a compact copy plan
// from that host state before physical execution.
inline constexpr RaggedPartitionRequirement kRaggedSequenceConcatenatePartitionRequirement =
    RaggedPartitionRequirement::HOST_EXTENT;

// A copy span: valueCount values from value sourceBegin of input inputIndex to value destinationBegin of the output.
// A plan's spans follow one another through the output in order, each starting where the one before it ends, so a span
// need not hold its count: it is where the next span starts (the last's, the active output's end). A plan is one of
// two formats, fixed for a concatenate when it is compiled:
//
// - PACKED, 8 bytes a span: the source as 32 bits; the destination in the low 32 - b bits of the second word and the
//   input index in its top b bits, b = raggedSequenceCopySpanInputBits(inputs). For concatenates whose output holds
//   at most 2^(32 - b) values and whose inputs at most 2^32 each (raggedSequenceCopySpanFormatFor).
// - WIDE, 32 bytes a span: every field in full.
enum class RaggedSequenceCopySpanFormat : uint8_t { PACKED, WIDE };

struct RaggedSequenceCopySpanPacked {
    uint32_t sourceBegin;
    uint32_t destinationBeginAndInput;
};
static_assert(sizeof(RaggedSequenceCopySpanPacked) == 8);

struct RaggedSequenceCopySpan64 {
    uint32_t inputIndex;
    uint32_t padding;
    uint64_t sourceBegin;
    uint64_t destinationBegin;
    uint64_t valueCount;
};
static_assert(sizeof(RaggedSequenceCopySpan64) == 32);

// The bits a packed span gives its input index: the fewest that name every input, at least 1.
[[nodiscard]] constexpr uint32_t raggedSequenceCopySpanInputBits(uint64_t inputs) noexcept {
    uint32_t bits = 1;
    while (bits < 32 && (uint64_t{1} << bits) < inputs) ++bits;
    return bits;
}

[[nodiscard]] constexpr std::size_t raggedSequenceCopySpanBytes(RaggedSequenceCopySpanFormat format) noexcept {
    return format == RaggedSequenceCopySpanFormat::PACKED ? sizeof(RaggedSequenceCopySpanPacked) : sizeof(RaggedSequenceCopySpan64);
}

// The format of a concatenate of `inputs` inputs whose output holds at most outputCapacityValues values and whose
// largest input at most largestInputCapacityValues.
[[nodiscard]] constexpr RaggedSequenceCopySpanFormat raggedSequenceCopySpanFormatFor(uint64_t inputs,
                                                                                    uint64_t outputCapacityValues,
                                                                                    uint64_t largestInputCapacityValues) noexcept {
    const uint32_t destinationBits = 32 - raggedSequenceCopySpanInputBits(inputs);
    return inputs <= (uint64_t{1} << 31) && outputCapacityValues <= (uint64_t{1} << destinationBits) &&
                   largestInputCapacityValues <= (uint64_t{1} << 32)
               ? RaggedSequenceCopySpanFormat::PACKED
               : RaggedSequenceCopySpanFormat::WIDE;
}

// A copy plan is built from runs: a run is one input's non-empty row, which goes to the output whole. The kernel gives
// every span of a plan the same lanes, those that move the plan's mean span in one pass, so a run far longer than the
// mean would be walked by its few lanes alone, pass after pass. A plan therefore holds no span longer than twice its
// mean run: a longer run is cut into spans of that length, which their lanes move side by side.
[[nodiscard]] constexpr uint64_t raggedSequenceCopySpanValueLimit(uint64_t activeValues, uint64_t runs) noexcept {
    if (runs == 0) return 1;
    const uint64_t meanRun = activeValues / runs + (activeValues % runs != 0 ? 1 : 0);
    return meanRun > std::numeric_limits<uint64_t>::max() / 2 ? std::numeric_limits<uint64_t>::max() : 2 * meanRun;
}

// The most spans a plan of `runs` runs holds: cutting adds fewer than one span per two runs (a run of n values is at
// most 1 + (n - 1) / limit spans, and the limit is at least twice the values per run).
[[nodiscard]] constexpr uint64_t raggedSequenceCopySpanCapacity(uint64_t runs) noexcept { return runs + runs / 2; }

// Write a span of a plan whose inputs' indices take inputBits bits. A packed span keeps no count (see above).
inline void setRaggedSequenceCopySpan(RaggedSequenceCopySpanPacked &span,
                                      uint32_t inputIndex,
                                      uint64_t sourceBegin,
                                      uint64_t destinationBegin,
                                      uint64_t /*valueCount*/,
                                      uint32_t inputBits) {
    const uint32_t destinationBits = 32 - inputBits;
    if (inputBits == 0 || inputBits > 31 || inputIndex >= (uint64_t{1} << inputBits) ||
        sourceBegin > std::numeric_limits<uint32_t>::max() || destinationBegin >= (uint64_t{1} << destinationBits)) {
        throw std::overflow_error("RaggedSequenceConcatenate packed copy span overflow.");
    }
    span = RaggedSequenceCopySpanPacked{static_cast<uint32_t>(sourceBegin),
                                        static_cast<uint32_t>(destinationBegin) | (inputIndex << destinationBits)};
}

inline void setRaggedSequenceCopySpan(RaggedSequenceCopySpan64 &span,
                                      uint32_t inputIndex,
                                      uint64_t sourceBegin,
                                      uint64_t destinationBegin,
                                      uint64_t valueCount,
                                      uint32_t /*inputBits*/) {
    span = RaggedSequenceCopySpan64{inputIndex, 0, sourceBegin, destinationBegin, valueCount};
}

// Appends the run of valueCount values from value sourceBegin of input inputIndex to value destinationBegin of the
// output, as spans of at most valueLimit values (raggedSequenceCopySpanValueLimit of the plan), to the spanCount spans
// already in `spans`, which holds spanCapacity; inputBits is raggedSequenceCopySpanInputBits of the plan's inputs. A
// plan's runs are appended in output order. Returns the new span count.
template <typename SpanT>
uint64_t appendRaggedSequenceCopyRun(SpanT *spans,
                                     uint64_t spanCount,
                                     uint64_t spanCapacity,
                                     uint32_t inputIndex,
                                     uint64_t sourceBegin,
                                     uint64_t destinationBegin,
                                     uint64_t valueCount,
                                     uint64_t valueLimit,
                                     uint32_t inputBits) {
    if (valueLimit == 0) throw std::invalid_argument("RaggedSequenceConcatenate copy span limit must be positive.");
    for (uint64_t done = 0; done < valueCount;) {
        const uint64_t values = valueCount - done < valueLimit ? valueCount - done : valueLimit;
        if (spanCount >= spanCapacity) throw std::length_error("RaggedSequenceConcatenate copy plan exceeds its span capacity.");
        setRaggedSequenceCopySpan(spans[spanCount++], inputIndex, sourceBegin + done, destinationBegin + done, values, inputBits);
        done += values;
    }
    return spanCount;
}

}  // namespace ThorImplementation

// Sequence-axis concatenate for canonical rank-1 ragged tensors.
//
// StampedNetwork derives the authoritative output partition and the copy spans
// (the inputs' non-empty rows, the long ones cut: see
// raggedSequenceCopySpanValueLimit) from authoritative host input partitions
// in O(batch * inputs). CUDA consumes only the inputs' packed-value arrays plus that compact
// span table; it never reads or reconstructs row offsets. `input_values` are
// the inputs' packed values, in input order, passed to the kernel as launch
// parameters. `copy_spans` is a device table of spans in `span_format` (packed
// spans with raggedSequenceCopySpanInputBits(input_values.size()) input bits),
// on a 16-byte boundary, that cover the first `active_output_values` values of
// the output in order. Those also size the launch: every span gets the lanes
// that move the table's mean span in one pass, and a longer span takes more
// passes.
void launchRaggedSequenceConcatenate(void *output_values,
                                     const std::vector<void *> &input_values,
                                     const void *copy_spans,
                                     uint64_t span_count,
                                     std::size_t value_element_size_bytes,
                                     uint64_t elements_per_value,
                                     ThorImplementation::RaggedSequenceCopySpanFormat span_format,
                                     uint64_t active_output_values,
                                     Stream stream);

// Backward split reuses the exact same copy-span table. `input_gradients` is
// parallel to the logical inputs; the spans of a null entry are skipped. Only
// values named by non-empty spans are written, so inactive gradient capacity
// remains untouched.
void launchRaggedSequenceConcatenateBackward(const std::vector<void *> &input_gradients,
                                             const void *output_gradient,
                                             const void *copy_spans,
                                             uint64_t span_count,
                                             std::size_t value_element_size_bytes,
                                             uint64_t elements_per_value,
                                             ThorImplementation::RaggedSequenceCopySpanFormat span_format,
                                             uint64_t active_output_values,
                                             Stream stream);
