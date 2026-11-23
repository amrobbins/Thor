#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

#include "DeepLearning/Implementation/Data/Residency/DeviceResidentPacketGrid.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

using ThorImplementation::RaggedSequenceCopySpan64;
using ThorImplementation::RaggedSequenceCopySpanFormat;
using ThorImplementation::RaggedSequenceCopySpanPacked;
using namespace DeviceResidentPacketGrid;

namespace {

// The inputs whose value arrays one launch is given. More inputs take a launch per kMaxInputsPerLaunch of them, each
// over the whole span table.
constexpr uint32_t kMaxInputsPerLaunch = 16;
// The most packets a lane takes in a pass, once the launch is past kSpreadMaxBlocks blocks at that depth: aligned
// packets, and shifted ones (a shifted packet holds two 16-byte words until its store).
constexpr uint32_t kMaxPacketsPerLane = 4;
constexpr uint32_t kMaxShiftedPacketsPerLane = 2;
// The most lanes a span is given, as a power of two: 4096 blocks. The lanes of a span are consecutive threads of the
// grid, so a striding grid (kMaxBlocks blocks) must hold a whole number of spans' lanes.
constexpr uint32_t kMaxLanesPerSpanLog2 = 20;
static_assert((kMaxBlocks * kThreadsPerBlock) % (1ULL << kMaxLanesPerSpanLog2) == 0);

// What a launch is given: the value arrays of inputs firstInput to firstInput + inputs - 1 (a null one is skipped), the
// packed output, and the span table. Kernel parameters, so a thread reads its span's array from constant memory. A
// value is packetsPerValue packets of the launch's kernel (1-byte packets, so its bytes, for sequenceBytesKernel). The
// spans cover the output's first activeOutputValues values; a packed span's destination is its second word's low
// destinationBits bits.
struct SequenceLaunch {
    void *input[kMaxInputsPerLaunch];
    void *output;
    const void *spans;
    uint64_t spanCount;
    uint64_t activeOutputValues;
    uint64_t packetsPerValue;
    uint32_t firstInput;
    uint32_t inputs;
    uint32_t lanesPerSpanLog2;
    uint32_t destinationBits;
};

// A span of the table: `values` values, from value `source` of input `input`, at value `destination` of the output.
struct Span {
    uint32_t input;
    uint64_t source;
    uint64_t destination;
    uint64_t values;
};

static_assert(sizeof(RaggedSequenceCopySpanPacked) == sizeof(uint2));
static_assert(sizeof(RaggedSequenceCopySpan64) == 2 * sizeof(uint4));

// A packed span is its 8-byte word and the next span's (the last span's next is the active output's end), read with
// no test between them, so both are in flight at once; they share a 32-byte sector three times in four. Its count is
// where the next span starts less where it starts.
__device__ __forceinline__ Span loadSpan(const RaggedSequenceCopySpanPacked *__restrict__ spans,
                                         uint64_t index,
                                         const SequenceLaunch &launch) {
    const uint2 *const words = reinterpret_cast<const uint2 *>(spans);
    const bool last = index + 1 == launch.spanCount;
    const uint2 word = words[index];
    const uint2 next = words[last ? index : index + 1];
    // An input index takes at least one bit, so destinationBits is at most 31.
    const uint32_t destinationMask = (1U << launch.destinationBits) - 1U;
    const uint64_t destination = word.y & destinationMask;
    const uint64_t end = last ? launch.activeOutputValues : static_cast<uint64_t>(next.y & destinationMask);
    return Span{word.y >> launch.destinationBits, word.x, destination, end - destination};
}

// A wide span is its two 16-byte words (the table is on a 16-byte boundary).
__device__ __forceinline__ Span loadSpan(const RaggedSequenceCopySpan64 *__restrict__ spans, uint64_t index, const SequenceLaunch &) {
    const uint4 *const words = reinterpret_cast<const uint4 *>(spans) + 2 * index;
    const uint4 first = words[0];
    const uint4 second = words[1];
    return Span{first.x,
                (static_cast<uint64_t>(first.w) << 32U) | first.z,
                (static_cast<uint64_t>(second.y) << 32U) | second.x,
                (static_cast<uint64_t>(second.w) << 32U) | second.z};
}

/**
 * Copy the spans of a sequence-axis concatenate. A span is a run of one input's packed values and the run of the
 * packed output it goes to; the forward copies input to output (OutputIsDestination), the backward output to input.
 * A value is packetsPerValue packets, and every value array starts on a packet boundary, so every span of both sides
 * is whole aligned packets (as in the ragged materialization, DeviceResidentRaggedMaterializationKernel.cu, whose
 * grid this is). The launcher takes it for every table but the large ones of 1- and 2-byte packets.
 *
 * Spans differ in length, so the grid is over spans: every span is given 1 << lanesPerSpanLog2 consecutive threads of
 * the grid, its lanes, which are part of a block (256 >> lanesPerSpanLog2 spans to a block) or several whole blocks.
 * In one pass the lanes of a span take its next lanes x PacketsPerLane packets, consecutive lanes consecutive packets,
 * PacketsPerLane packets each; a span longer than that takes more passes. The launcher sizes a pass to the table's
 * mean span, so a table of equal spans is one pass however long they are.
 *
 * A thread issues its span load, then the value loads of a pass (the first use of the span) before it stores any. The
 * span's value array comes from the kernel parameters. A packet past the span's end reads the span's first packet and
 * is not stored.
 */
template <typename SpanT, typename PacketT, typename LengthT, uint32_t PacketsPerLane, bool OutputIsDestination>
__global__ void __launch_bounds__(kThreadsPerBlock) sequencePacketsKernel(const __grid_constant__ SequenceLaunch launch) {
    const uint32_t thread = blockIdx.x * kThreadsPerBlock + threadIdx.x;
    const uint32_t lanesPerSpan = 1U << launch.lanesPerSpanLog2;
    const LengthT lane = static_cast<LengthT>(thread & (lanesPerSpan - 1U));
    const LengthT packetsPerPass = static_cast<LengthT>(lanesPerSpan) * PacketsPerLane;
    const LengthT packetsPerValue = static_cast<LengthT>(launch.packetsPerValue);
    const uint64_t spanStride = (gridDim.x * kThreadsPerBlock) >> launch.lanesPerSpanLog2;
    for (uint64_t index = thread >> launch.lanesPerSpanLog2; index < launch.spanCount; index += spanStride) {
        const Span span = loadSpan(static_cast<const SpanT *>(launch.spans), index, launch);
        // A span of an input that another launch holds, or of an input without a gradient, is not this launch's.
        const uint32_t input = span.input - launch.firstInput;
        if (input >= launch.inputs) continue;
        PacketT *const inputValues = static_cast<PacketT *>(launch.input[input]);
        if (inputValues == nullptr) continue;
        PacketT *const inputSpan = inputValues + span.source * launch.packetsPerValue;
        PacketT *const outputSpan = static_cast<PacketT *>(launch.output) + span.destination * launch.packetsPerValue;
        const PacketT *const source = OutputIsDestination ? inputSpan : outputSpan;
        PacketT *const destination = OutputIsDestination ? outputSpan : inputSpan;

        const LengthT spanPackets = static_cast<LengthT>(span.values) * packetsPerValue;
        LengthT pass = 0;
        while (pass < spanPackets) {
            // Value loads only: no loaded packet is read before the stores.
            LengthT packet[PacketsPerLane];
            bool inSpan[PacketsPerLane];
            PacketT value[PacketsPerLane];
#pragma unroll
            for (uint32_t k = 0; k < PacketsPerLane; ++k) {
                packet[k] = pass + static_cast<LengthT>(k * lanesPerSpan) + lane;
                inSpan[k] = packet[k] < spanPackets;
                value[k] = source[inSpan[k] ? packet[k] : 0];
            }
#pragma unroll
            for (uint32_t k = 0; k < PacketsPerLane; ++k) {
                if (inSpan[k]) destination[packet[k]] = value[k];
            }
            // Break before the increment so the 32-bit packet count cannot wrap on the last pass.
            if (spanPackets - pass <= packetsPerPass) break;
            pass += packetsPerPass;
        }
    }
}

/**
 * Copy the spans of a sequence-axis concatenate in 16-byte packets whatever byte they start and end at (the launcher
 * takes it for large tables of values on a 1- or 2-byte boundary). The grid, the lanes and the passes are
 * sequencePacketsKernel's; a lane's share of a span is copyRowBytes (DeviceResidentPacketGrid.h), the ragged
 * materialization's: the span's whole 16-byte packets of the destination with their source words shifted, and its
 * head and tail in aligned pieces.
 *
 * Each side of a span is addressed from the 16-byte boundary at or before its first byte, so the arrays may start at
 * any byte and a span's byte counts stay small. A word that holds a span's first or last bytes may begin up to 15
 * bytes before the array or end up to 15 bytes past it: inside the 16-byte boundaries an allocation starts and ends
 * on, or the padding every tensor is allocated with.
 */
template <typename SpanT, typename LengthT, uint32_t PacketsPerLane, bool OutputIsDestination>
__global__ void __launch_bounds__(kThreadsPerBlock) sequenceBytesKernel(const __grid_constant__ SequenceLaunch launch) {
    constexpr uintptr_t kWordMask = sizeof(uint4) - 1;
    const uint32_t thread = blockIdx.x * kThreadsPerBlock + threadIdx.x;
    const uint32_t lanesPerSpan = 1U << launch.lanesPerSpanLog2;
    const uint32_t lane = thread & (lanesPerSpan - 1U);
    const uint64_t spanStride = (gridDim.x * kThreadsPerBlock) >> launch.lanesPerSpanLog2;
    for (uint64_t index = thread >> launch.lanesPerSpanLog2; index < launch.spanCount; index += spanStride) {
        const Span span = loadSpan(static_cast<const SpanT *>(launch.spans), index, launch);
        const uint32_t input = span.input - launch.firstInput;
        if (input >= launch.inputs) continue;
        if (launch.input[input] == nullptr) continue;
        const uintptr_t inputSpan = reinterpret_cast<uintptr_t>(launch.input[input]) + span.source * launch.packetsPerValue;
        const uintptr_t outputSpan = reinterpret_cast<uintptr_t>(launch.output) + span.destination * launch.packetsPerValue;
        const uintptr_t source = OutputIsDestination ? inputSpan : outputSpan;
        const uintptr_t destination = OutputIsDestination ? outputSpan : inputSpan;
        const LengthT firstByte = static_cast<LengthT>(destination & kWordMask);
        const LengthT endByte = firstByte + static_cast<LengthT>(span.values) * static_cast<LengthT>(launch.packetsPerValue);
        copyRowBytes<LengthT, PacketsPerLane>(reinterpret_cast<const uint4 *>(source & ~kWordMask),
                                              reinterpret_cast<uint4 *>(destination & ~kWordMask),
                                              firstByte,
                                              endByte,
                                              source & kWordMask,
                                              lane,
                                              lanesPerSpan);
    }
}

// The lanes of a span, as a power of two: the fewest whose pass holds the table's mean span.
uint32_t lanesPerSpanLog2For(uint64_t meanSpanPackets, uint32_t packetsPerLane) {
    uint32_t lanesPerSpanLog2 = 0;
    while (lanesPerSpanLog2 < kMaxLanesPerSpanLog2 && (static_cast<uint64_t>(packetsPerLane) << lanesPerSpanLog2) < meanSpanPackets) {
        ++lanesPerSpanLog2;
    }
    return lanesPerSpanLog2;
}

// Blocks of a launch over spanCount spans: the blocks that hold their lanes, up to kMaxBlocks (the kernel strides over
// the rest).
uint32_t blocksForSpans(uint64_t spanCount, uint32_t lanesPerSpanLog2) {
    constexpr uint32_t kThreadsPerBlockLog2 = 8;
    static_assert(kThreadsPerBlock == 1U << kThreadsPerBlockLog2);
    if (lanesPerSpanLog2 >= kThreadsPerBlockLog2) {
        const uint32_t blocksPerSpanLog2 = lanesPerSpanLog2 - kThreadsPerBlockLog2;
        return static_cast<uint32_t>(spanCount > (kMaxBlocks >> blocksPerSpanLog2) ? kMaxBlocks : spanCount << blocksPerSpanLog2);
    }
    const uint64_t spansPerBlock = kThreadsPerBlock >> lanesPerSpanLog2;
    return static_cast<uint32_t>(std::min(kMaxBlocks, (spanCount + spansPerBlock - 1) / spansPerBlock));
}

template <typename SpanT, typename PacketT, typename LengthT, uint32_t PacketsPerLane, bool OutputIsDestination>
void launchWithPacketsPerLane(SequenceLaunch launch, uint32_t lanesPerSpanLog2, cudaStream_t stream) {
    launch.lanesPerSpanLog2 = lanesPerSpanLog2;
    const uint32_t blocks = blocksForSpans(launch.spanCount, lanesPerSpanLog2);
    sequencePacketsKernel<SpanT, PacketT, LengthT, PacketsPerLane, OutputIsDestination><<<blocks, kThreadsPerBlock, 0, stream>>>(launch);
    CUDA_CHECK(cudaGetLastError());
}

// A lane takes the fewest packets (1, 2, 4) that keep the launch within kSpreadMaxBlocks blocks, as in the ragged
// materialization: more packets per lane are fewer lanes per span, so fewer blocks, down to one lane a span. A span's
// packets are counted in 32 bits when the active output holds few enough of them that a pass cannot wrap; a larger
// output takes 64-bit counts, one packet per lane: out of the ordinary range.
template <typename SpanT, typename PacketT, bool OutputIsDestination>
void launchWithPacket(SequenceLaunch launch, uint64_t activeOutputValues, cudaStream_t stream) {
    launch.packetsPerValue /= sizeof(PacketT);
    const uint64_t activeOutputPackets = activeOutputValues * launch.packetsPerValue;
    const uint64_t meanSpanPackets = std::max<uint64_t>(1, (activeOutputPackets + launch.spanCount - 1) / launch.spanCount);
    if (activeOutputPackets > std::numeric_limits<uint32_t>::max() - (static_cast<uint64_t>(kMaxPacketsPerLane) << kMaxLanesPerSpanLog2)) {
        launchWithPacketsPerLane<SpanT, PacketT, uint64_t, 1, OutputIsDestination>(launch, lanesPerSpanLog2For(meanSpanPackets, 1), stream);
        return;
    }
    uint32_t packetsPerLane = 1;
    while (packetsPerLane < kMaxPacketsPerLane &&
           lanesPerSpanLog2For(meanSpanPackets, packetsPerLane) != 0 &&
           blocksForSpans(launch.spanCount, lanesPerSpanLog2For(meanSpanPackets, packetsPerLane)) > kSpreadMaxBlocks) {
        packetsPerLane *= 2;
    }
    const uint32_t lanesPerSpanLog2 = lanesPerSpanLog2For(meanSpanPackets, packetsPerLane);
    if (packetsPerLane == 1) {
        launchWithPacketsPerLane<SpanT, PacketT, uint32_t, 1, OutputIsDestination>(launch, lanesPerSpanLog2, stream);
    } else if (packetsPerLane == 2) {
        launchWithPacketsPerLane<SpanT, PacketT, uint32_t, 2, OutputIsDestination>(launch, lanesPerSpanLog2, stream);
    } else {
        launchWithPacketsPerLane<SpanT, PacketT, uint32_t, kMaxPacketsPerLane, OutputIsDestination>(launch, lanesPerSpanLog2, stream);
    }
}

template <typename SpanT, typename LengthT, uint32_t PacketsPerLane, bool OutputIsDestination>
void launchBytesWithPacketsPerLane(SequenceLaunch launch, uint32_t lanesPerSpanLog2, cudaStream_t stream) {
    launch.lanesPerSpanLog2 = lanesPerSpanLog2;
    const uint32_t blocks = blocksForSpans(launch.spanCount, lanesPerSpanLog2);
    sequenceBytesKernel<SpanT, LengthT, PacketsPerLane, OutputIsDestination><<<blocks, kThreadsPerBlock, 0, stream>>>(launch);
    CUDA_CHECK(cudaGetLastError());
}

// Spans that start and end at any byte: 16-byte packets with the spans' heads and tails in aligned pieces
// (sequenceBytesKernel), sized as the ragged materialization sizes its rows. A pass holds the mean span's packets, with
// no fewer lanes than the mean span has pieces (a head and a tail are 2 pieces each on average, and a span has no more
// pieces than bytes). A lane takes two packets where that makes a launch past kSpreadMaxBlocks blocks smaller. A span's
// bytes are counted in 32 bits when the active output's fit, and otherwise in 64 bits, one packet per lane.
template <typename SpanT, bool OutputIsDestination>
void launchBytes(const SequenceLaunch &launch, uint64_t activeOutputValues, cudaStream_t stream) {
    const uint64_t activeOutputBytes = activeOutputValues * launch.packetsPerValue;
    const uint64_t meanSpanBytes = std::max<uint64_t>(1, (activeOutputBytes + launch.spanCount - 1) / launch.spanCount);
    const uint64_t meanSpanPackets = meanSpanBytes / sizeof(uint4);
    const uint64_t meanSpanPieces = std::min<uint64_t>(meanSpanBytes, 4);
    const auto lanesPerSpanLog2At = [&](uint32_t packetsPerLane) {
        return std::max(lanesPerSpanLog2For(meanSpanPackets, packetsPerLane), lanesPerSpanLog2For(meanSpanPieces, 1));
    };
    if (activeOutputBytes > std::numeric_limits<uint32_t>::max() - sizeof(uint4)) {
        launchBytesWithPacketsPerLane<SpanT, uint64_t, 1, OutputIsDestination>(launch, lanesPerSpanLog2At(1), stream);
        return;
    }
    if (blocksForSpans(launch.spanCount, lanesPerSpanLog2At(1)) <= kSpreadMaxBlocks ||
        lanesPerSpanLog2At(kMaxShiftedPacketsPerLane) == lanesPerSpanLog2At(1)) {
        launchBytesWithPacketsPerLane<SpanT, uint32_t, 1, OutputIsDestination>(launch, lanesPerSpanLog2At(1), stream);
    } else {
        launchBytesWithPacketsPerLane<SpanT, uint32_t, kMaxShiftedPacketsPerLane, OutputIsDestination>(
            launch, lanesPerSpanLog2At(kMaxShiftedPacketsPerLane), stream);
    }
}

// launch.packetsPerValue holds the value's bytes on the way in; launchWithPacket divides it by its packet.
//
// Packets of 1 and 2 bytes are the faster way for a table that the most of them a lane takes keep within
// kSpreadMaxBlocks blocks. A larger table of them is bound by its instructions and moves in 16-byte packets instead
// (the ragged materialization's rule, measured there on both sides of the bound, patch 1553).
template <typename SpanT, bool OutputIsDestination>
void launchWithSpans(const SequenceLaunch &launch, uint64_t packetBytes, uint64_t activeOutputValues, cudaStream_t stream) {
    if (packetBytes < sizeof(uint32_t)) {
        const uint64_t activeOutputPackets = activeOutputValues * (launch.packetsPerValue / packetBytes);
        const uint64_t meanSpanPackets = std::max<uint64_t>(1, (activeOutputPackets + launch.spanCount - 1) / launch.spanCount);
        if (blocksForSpans(launch.spanCount, lanesPerSpanLog2For(meanSpanPackets, kMaxPacketsPerLane)) > kSpreadMaxBlocks) {
            launchBytes<SpanT, OutputIsDestination>(launch, activeOutputValues, stream);
            return;
        }
    }
    switch (packetBytes) {
        case 16:
            launchWithPacket<SpanT, uint4, OutputIsDestination>(launch, activeOutputValues, stream);
            return;
        case 8:
            launchWithPacket<SpanT, uint2, OutputIsDestination>(launch, activeOutputValues, stream);
            return;
        case 4:
            launchWithPacket<SpanT, uint32_t, OutputIsDestination>(launch, activeOutputValues, stream);
            return;
        case 2:
            launchWithPacket<SpanT, uint16_t, OutputIsDestination>(launch, activeOutputValues, stream);
            return;
        default:
            launchWithPacket<SpanT, uint8_t, OutputIsDestination>(launch, activeOutputValues, stream);
            return;
    }
}

// Both directions: inputs are the inputs' value arrays (forward) or their gradients (backward, null where an input
// takes none); output is the packed output (forward) or its gradient (backward).
template <bool OutputIsDestination>
void launchSequence(const std::vector<void *> &inputs,
                    const void *output,
                    const void *copySpans,
                    uint64_t spanCount,
                    std::size_t valueElementSizeBytes,
                    uint64_t elementsPerValue,
                    RaggedSequenceCopySpanFormat spanFormat,
                    uint64_t activeOutputValues,
                    Stream stream) {
    if (valueElementSizeBytes == 0 || elementsPerValue == 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate values must have non-zero element geometry.");
    }
    if (spanFormat != RaggedSequenceCopySpanFormat::PACKED && spanFormat != RaggedSequenceCopySpanFormat::WIDE) {
        throw std::invalid_argument("RaggedSequenceConcatenate copy spans must be PACKED or WIDE.");
    }
    if (inputs.empty() || inputs.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("RaggedSequenceConcatenate needs at least one input.");
    }
    if (activeOutputValues == 0) {
        if (spanCount != 0) throw std::invalid_argument("RaggedSequenceConcatenate zero active prefix requires an empty copy plan.");
        return;
    }
    if (spanCount == 0 || copySpans == nullptr) {
        throw std::invalid_argument("RaggedSequenceConcatenate non-empty active prefix requires copy spans.");
    }
    if (reinterpret_cast<uintptr_t>(copySpans) % sizeof(uint4) != 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate copy spans must start on a 16-byte boundary.");
    }
    if (output == nullptr) throw std::invalid_argument("RaggedSequenceConcatenate requires the packed output.");
    if (elementsPerValue > std::numeric_limits<uint64_t>::max() / valueElementSizeBytes) {
        throw std::invalid_argument("RaggedSequenceConcatenate value byte count overflow.");
    }
    const uint64_t valueBytes = valueElementSizeBytes * elementsPerValue;
    if (activeOutputValues > std::numeric_limits<uint64_t>::max() / valueBytes) {
        throw std::invalid_argument("RaggedSequenceConcatenate active output byte count overflow.");
    }

    // The widest packet, up to 16 bytes, on whose boundaries every value of the output and of every input starts.
    uint64_t alignment = valueBytes | reinterpret_cast<uintptr_t>(output) | 16;
    bool anyInput = false;
    for (const void *input : inputs) {
        if (input == nullptr) {
            if (OutputIsDestination) throw std::invalid_argument("RaggedSequenceConcatenate requires every input's values.");
            continue;
        }
        anyInput = true;
        alignment |= reinterpret_cast<uintptr_t>(input);
    }
    if (!anyInput) return;
    const uint64_t packetBytes = alignment & (~alignment + 1);

    ScopedGpu scopedGpu(stream.getGpuNum());
    SequenceLaunch sequenceLaunch{};
    sequenceLaunch.output = const_cast<void *>(output);
    sequenceLaunch.spans = copySpans;
    sequenceLaunch.spanCount = spanCount;
    sequenceLaunch.activeOutputValues = activeOutputValues;
    sequenceLaunch.packetsPerValue = valueBytes;
    sequenceLaunch.destinationBits = 32 - ThorImplementation::raggedSequenceCopySpanInputBits(inputs.size());
    for (uint64_t first = 0; first < inputs.size(); first += kMaxInputsPerLaunch) {
        sequenceLaunch.firstInput = static_cast<uint32_t>(first);
        sequenceLaunch.inputs = static_cast<uint32_t>(std::min<uint64_t>(kMaxInputsPerLaunch, inputs.size() - first));
        bool anyOfLaunch = false;
        for (uint32_t i = 0; i < kMaxInputsPerLaunch; ++i) {
            sequenceLaunch.input[i] = i < sequenceLaunch.inputs ? inputs[first + i] : nullptr;
            anyOfLaunch |= sequenceLaunch.input[i] != nullptr;
        }
        if (!anyOfLaunch) continue;
        if (spanFormat == RaggedSequenceCopySpanFormat::PACKED) {
            launchWithSpans<RaggedSequenceCopySpanPacked, OutputIsDestination>(sequenceLaunch, packetBytes, activeOutputValues, stream.getStream());
        } else {
            launchWithSpans<RaggedSequenceCopySpan64, OutputIsDestination>(sequenceLaunch, packetBytes, activeOutputValues, stream.getStream());
        }
    }
}

}  // namespace

void launchRaggedSequenceConcatenate(void *output_values,
                                     const std::vector<void *> &input_values,
                                     const void *copy_spans,
                                     uint64_t span_count,
                                     std::size_t value_element_size_bytes,
                                     uint64_t elements_per_value,
                                     RaggedSequenceCopySpanFormat span_format,
                                     uint64_t active_output_values,
                                     Stream stream) {
    launchSequence<true>(input_values, output_values, copy_spans, span_count, value_element_size_bytes, elements_per_value,
                         span_format, active_output_values, stream);
}

void launchRaggedSequenceConcatenateBackward(const std::vector<void *> &input_gradients,
                                             const void *output_gradient,
                                             const void *copy_spans,
                                             uint64_t span_count,
                                             std::size_t value_element_size_bytes,
                                             uint64_t elements_per_value,
                                             RaggedSequenceCopySpanFormat span_format,
                                             uint64_t active_output_values,
                                             Stream stream) {
    launchSequence<false>(input_gradients, output_gradient, copy_spans, span_count, value_element_size_bytes, elements_per_value,
                          span_format, active_output_values, stream);
}
