#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                               \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                            \
            GTEST_SKIP() << "CUDA device is required for RaggedSequenceConcatenate kernel tests.";                      \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

Tensor makeGpuBytes(const std::vector<uint8_t> &values, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(std::max<size_t>(values.size(), 16))}));
    uint8_t *hostValues = host.getMemPtr<uint8_t>();
    for (size_t i = 0; i < host.getTotalNumElements(); ++i) hostValues[i] = i < values.size() ? values[i] : 0;
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

std::vector<uint8_t> copyGpuBytes(const Tensor &device, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT8, device.getDimensions()));
    host.copyFromAsync(device, stream);
    stream.synchronize();
    const uint8_t *hostValues = host.getMemPtr<uint8_t>();
    return std::vector<uint8_t>(hostValues, hostValues + host.getTotalNumElements());
}

uint64_t splitMix64(uint64_t value) {
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

// A batch of rows of every input, concatenated on the sequence axis: lengths[row][input] values of valueBytes bytes.
// The copy plan is the one StampedNetwork derives (row by row, input by input, the run of every non-empty row of an
// input, cut at the plan's span limit unless the case keeps its runs whole; rows from activeRows on are capacity and
// get no span). Every array holds extraValues values past its active ones and
// starts offsetBytes into its tensor, so a case chooses the packet the kernel can move. Checked byte for byte: the
// output against the inputs, the gradients against the output's gradient, everything no span names untouched.
struct SequenceCase {
    std::string name;
    uint64_t valueBytes = 4;
    std::vector<std::vector<uint64_t>> lengths;
    uint64_t activeRows = 0;  // 0: every row
    bool wideSpans = false;
    uint64_t extraValues = 3;
    uint64_t outputOffsetBytes = 0;
    std::vector<uint64_t> inputOffsetBytes;  // empty: none
    std::vector<bool> nullGradient;          // empty: none
    bool cutLongRuns = true;                 // false: a span per run, however long (more passes for its lanes)
};

// The rows of `inputs` inputs: row r of input i has lengthOf(r, i) values.
template <typename LengthOf>
std::vector<std::vector<uint64_t>> makeLengths(uint64_t rows, uint32_t inputs, LengthOf &&lengthOf) {
    std::vector<std::vector<uint64_t>> lengths(rows, std::vector<uint64_t>(inputs, 0));
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint32_t input = 0; input < inputs; ++input) lengths[row][input] = lengthOf(row, input);
    }
    return lengths;
}

void runSequenceCase(const SequenceCase &testCase) {
    SCOPED_TRACE(testCase.name);
    constexpr uint8_t sentinel = 0xD3;
    const size_t inputs = testCase.lengths.front().size();
    const uint64_t activeRows = testCase.activeRows == 0 ? testCase.lengths.size() : testCase.activeRows;
    std::vector<uint64_t> inputOffsetBytes = testCase.inputOffsetBytes;
    inputOffsetBytes.resize(inputs, 0);
    std::vector<bool> nullGradient = testCase.nullGradient;
    nullGradient.resize(inputs, false);

    // The runs, and each input's and the output's values (all rows: the inactive ones are capacity).
    std::vector<RaggedSequenceCopySpan64> runs;
    std::vector<uint64_t> inputValues(inputs, 0);
    uint64_t outputValues = 0;
    uint64_t activeOutputValues = 0;
    for (uint64_t row = 0; row < testCase.lengths.size(); ++row) {
        for (uint32_t input = 0; input < inputs; ++input) {
            const uint64_t length = testCase.lengths[row][input];
            if (row < activeRows && length != 0) runs.push_back({input, 0, inputValues[input], outputValues, length});
            inputValues[input] += length;
            outputValues += length;
        }
        if (row + 1 == activeRows) activeOutputValues = outputValues;
    }
    // The plan, in both span formats.
    const uint64_t valueLimit = testCase.cutLongRuns ? raggedSequenceCopySpanValueLimit(activeOutputValues, runs.size()) : activeOutputValues + 1;
    const uint64_t spanCapacity = testCase.cutLongRuns ? raggedSequenceCopySpanCapacity(runs.size()) : runs.size();
    const uint32_t inputBits = raggedSequenceCopySpanInputBits(inputs);
    std::vector<RaggedSequenceCopySpanPacked> packed(spanCapacity);
    std::vector<RaggedSequenceCopySpan64> spans64(spanCapacity);
    uint64_t spanCount = 0;
    for (const RaggedSequenceCopySpan64 &run : runs) {
        const uint64_t before = spanCount;
        spanCount = appendRaggedSequenceCopyRun(spans64.data(), before, spanCapacity, run.inputIndex, run.sourceBegin, run.destinationBegin,
                                                run.valueCount, valueLimit, inputBits);
        ASSERT_EQ(appendRaggedSequenceCopyRun(packed.data(), before, spanCapacity, run.inputIndex, run.sourceBegin, run.destinationBegin,
                                              run.valueCount, valueLimit, inputBits),
                  spanCount);
    }
    packed.resize(spanCount);
    spans64.resize(spanCount);
    if (!testCase.cutLongRuns) ASSERT_EQ(spanCount, runs.size());
    const RaggedSequenceCopySpanFormat spanFormat = testCase.wideSpans ? RaggedSequenceCopySpanFormat::WIDE : RaggedSequenceCopySpanFormat::PACKED;
    std::vector<uint8_t> spanBytes(spanCount * raggedSequenceCopySpanBytes(spanFormat));
    if (!spanBytes.empty()) {
        std::memcpy(spanBytes.data(), testCase.wideSpans ? static_cast<const void *>(spans64.data()) : static_cast<const void *>(packed.data()),
                    spanBytes.size());
    }

    Stream stream(0);
    const Tensor spanTable = makeGpuBytes(spanBytes, stream);
    const uint64_t valueBytes = testCase.valueBytes;
    const auto arrayBytes = [&](uint64_t values, uint64_t offset) { return offset + (values + testCase.extraValues) * valueBytes; };

    std::vector<std::vector<uint8_t>> inputBytes(inputs);
    std::vector<Tensor> inputTensors;
    std::vector<void *> inputPointers;
    for (size_t input = 0; input < inputs; ++input) {
        inputBytes[input].resize(arrayBytes(inputValues[input], inputOffsetBytes[input]));
        for (uint64_t byte = 0; byte < inputBytes[input].size(); ++byte) {
            inputBytes[input][byte] = static_cast<uint8_t>(splitMix64(byte * 131 + input * 977 + 5) >> 24U);
        }
        inputTensors.push_back(makeGpuBytes(inputBytes[input], stream));
        inputPointers.push_back(inputTensors.back().getMemPtr<uint8_t>() + inputOffsetBytes[input]);
    }
    Tensor output = makeGpuBytes(std::vector<uint8_t>(arrayBytes(outputValues, testCase.outputOffsetBytes), sentinel), stream);
    void *const outputPointer = output.getMemPtr<uint8_t>() + testCase.outputOffsetBytes;

    launchRaggedSequenceConcatenate(outputPointer, inputPointers, spanTable.getMemPtr(), spanCount, 1, valueBytes, spanFormat,
                                    activeOutputValues, stream);
    stream.synchronize();
    std::vector<uint8_t> expectedOutput(arrayBytes(outputValues, testCase.outputOffsetBytes), sentinel);
    for (const RaggedSequenceCopySpan64 &span : spans64) {
        std::memcpy(expectedOutput.data() + testCase.outputOffsetBytes + span.destinationBegin * valueBytes,
                    inputBytes[span.inputIndex].data() + inputOffsetBytes[span.inputIndex] + span.sourceBegin * valueBytes,
                    span.valueCount * valueBytes);
    }
    std::vector<uint8_t> outputBytes = copyGpuBytes(output, stream);
    outputBytes.resize(expectedOutput.size());
    uint64_t wrong = 0;
    for (size_t byte = 0; byte < expectedOutput.size(); ++byte) wrong += outputBytes[byte] != expectedOutput[byte];
    EXPECT_EQ(wrong, 0U) << "forward";

    // Backward: the output's gradient back to the inputs' gradients; an input without one is not touched.
    std::vector<uint8_t> upstream(arrayBytes(outputValues, testCase.outputOffsetBytes));
    for (uint64_t byte = 0; byte < upstream.size(); ++byte) upstream[byte] = static_cast<uint8_t>(splitMix64(byte * 257 + 11) >> 16U);
    const Tensor upstreamTensor = makeGpuBytes(upstream, stream);
    std::vector<Tensor> gradientTensors;
    std::vector<void *> gradientPointers;
    for (size_t input = 0; input < inputs; ++input) {
        gradientTensors.push_back(makeGpuBytes(std::vector<uint8_t>(inputBytes[input].size(), sentinel), stream));
        gradientPointers.push_back(nullGradient[input] ? nullptr : gradientTensors.back().getMemPtr<uint8_t>() + inputOffsetBytes[input]);
    }
    launchRaggedSequenceConcatenateBackward(gradientPointers, upstreamTensor.getMemPtr<uint8_t>() + testCase.outputOffsetBytes,
                                            spanTable.getMemPtr(), spanCount, valueBytes, 1, spanFormat, activeOutputValues, stream);
    stream.synchronize();
    wrong = 0;
    for (size_t input = 0; input < inputs; ++input) {
        std::vector<uint8_t> expected(inputBytes[input].size(), sentinel);
        if (!nullGradient[input]) {
            for (const RaggedSequenceCopySpan64 &span : spans64) {
                if (span.inputIndex != input) continue;
                std::memcpy(expected.data() + inputOffsetBytes[input] + span.sourceBegin * valueBytes,
                            upstream.data() + testCase.outputOffsetBytes + span.destinationBegin * valueBytes, span.valueCount * valueBytes);
            }
        }
        std::vector<uint8_t> gradientBytes = copyGpuBytes(gradientTensors[input], stream);
        gradientBytes.resize(expected.size());
        for (size_t byte = 0; byte < expected.size(); ++byte) wrong += gradientBytes[byte] != expected[byte];
    }
    EXPECT_EQ(wrong, 0U) << "backward";
}

// Row lengths uniform in [1, 2 mean - 1], a share of them empty.
uint64_t randomLength(uint64_t row, uint32_t input, uint64_t mean, uint32_t emptyPerMille = 0) {
    const uint64_t draw = splitMix64(row * 0xd1b54a32d192ed03ULL + input * 0x8cb92ba72f3d8dd7ULL + mean);
    if (draw % 1000 < emptyPerMille) return 0;
    return 1 + (draw >> 20U) % (2 * mean - 1);
}

}  // namespace

TEST(RaggedSequenceConcatenate, ForwardAndBackwardFollowTheCopyPlanAndLeaveInactiveCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    for (const bool wideSpans : {false, true}) {
        SequenceCase testCase;
        testCase.name = wideSpans ? "FP32 pairs, wide spans" : "FP32 pairs, packed spans";
        testCase.valueBytes = 8;
        // Rows with an empty side, an empty row, and two rows of capacity past the active ones.
        testCase.lengths = {{2, 1}, {0, 3}, {4, 0}, {0, 0}, {1, 1}, {5, 2}, {3, 3}};
        testCase.activeRows = 5;
        testCase.wideSpans = wideSpans;
        runSequenceCase(testCase);
        testCase.name += ", second input without a gradient";
        testCase.nullGradient = {false, true};
        runSequenceCase(testCase);
    }
}

TEST(RaggedSequenceConcatenate, AllEmptyRowsLaunchNothingAndTouchNothing) {
    REQUIRE_CUDA_DEVICE();
    SequenceCase testCase;
    testCase.name = "every active row empty";
    testCase.lengths = {{0, 0}, {0, 0}, {0, 0}, {2, 5}};
    testCase.activeRows = 3;
    runSequenceCase(testCase);
}

TEST(RaggedSequenceConcatenate, ValuesMoveInPacketsOfTheirWidthAndAlignment) {
    REQUIRE_CUDA_DEVICE();
    // 300 rows of 3 inputs with random lengths around 6 values, a fifth of them empty: values of every packet width
    // (16, 8, 4, 2 and 1 bytes), then 16-byte values in arrays that start off a 16-byte boundary.
    const auto lengths = makeLengths(300, 3, [](uint64_t row, uint32_t input) { return randomLength(row, input, 6, 200); });
    for (const uint64_t valueBytes : {1ULL, 2ULL, 3ULL, 4ULL, 6ULL, 8ULL, 12ULL, 16ULL, 20ULL, 33ULL, 48ULL, 100ULL}) {
        SequenceCase testCase;
        testCase.name = std::to_string(valueBytes) + "-byte values";
        testCase.valueBytes = valueBytes;
        testCase.lengths = lengths;
        testCase.wideSpans = valueBytes % 3 == 0;
        runSequenceCase(testCase);
    }
    for (const uint64_t offset : {8ULL, 4ULL, 2ULL, 1ULL}) {
        SequenceCase testCase;
        testCase.name = "16-byte values, the output " + std::to_string(offset) + " bytes off";
        testCase.valueBytes = 16;
        testCase.lengths = lengths;
        testCase.outputOffsetBytes = offset;
        runSequenceCase(testCase);
        testCase.name = "16-byte values, the second input " + std::to_string(offset) + " bytes off";
        testCase.outputOffsetBytes = 0;
        testCase.inputOffsetBytes = {0, offset, 0};
        runSequenceCase(testCase);
    }
}

TEST(RaggedSequenceConcatenate, SpansOfEveryLengthTakeTheirLanesAndPasses) {
    REQUIRE_CUDA_DEVICE();
    // The lanes of a span follow the table's mean span, from one lane (a packet) to eight blocks of them (5000
    // packets, four to a lane); a span longer than a pass takes more.
    for (const uint64_t mean : {1ULL, 2ULL, 5ULL, 16ULL, 37ULL, 128ULL, 300ULL, 1000ULL, 5000ULL}) {
        SequenceCase testCase;
        testCase.valueBytes = 16;
        testCase.name = "every row of " + std::to_string(mean) + " packets";
        testCase.lengths = makeLengths(40, 2, [mean](uint64_t, uint32_t) { return mean; });
        runSequenceCase(testCase);
        if (mean == 1) continue;
        testCase.name = "rows around " + std::to_string(mean) + " packets";
        testCase.lengths = makeLengths(40, 2, [mean](uint64_t row, uint32_t input) { return randomLength(row, input, mean, 100); });
        runSequenceCase(testCase);
    }
    {
        SequenceCase testCase;
        testCase.name = "a one-value input beside a long one";
        testCase.valueBytes = 4;
        testCase.lengths = makeLengths(500, 2, [](uint64_t row, uint32_t input) { return input == 0 ? 1 : randomLength(row, input, 200); });
        runSequenceCase(testCase);
    }
    {
        // One row of each input holds 30000 values among rows of 4. Cut by the plan, it is 790 spans of 38 values beside
        // the short rows' spans of 4; kept whole, it is one span and about a thousand passes for its lanes.
        SequenceCase testCase;
        testCase.name = "one long row among short ones, cut";
        testCase.valueBytes = 4;
        testCase.lengths = makeLengths(2000, 2, [](uint64_t row, uint32_t input) { return row == 700 + input ? 30000 : 4; });
        runSequenceCase(testCase);
        testCase.name = "one long row among short ones, whole";
        testCase.cutLongRuns = false;
        runSequenceCase(testCase);
    }
}

TEST(RaggedSequenceConcatenate, PlansCutRunsLongerThanTwiceTheMean) {
    // The limit is twice the mean run, rounded up; a plan without runs has none to cut.
    EXPECT_EQ(raggedSequenceCopySpanValueLimit(43, 8), 12U);
    EXPECT_EQ(raggedSequenceCopySpanValueLimit(40, 8), 10U);
    EXPECT_EQ(raggedSequenceCopySpanValueLimit(1, 1), 2U);
    EXPECT_EQ(raggedSequenceCopySpanValueLimit(0, 0), 1U);
    EXPECT_EQ(raggedSequenceCopySpanCapacity(8), 12U);
    EXPECT_EQ(raggedSequenceCopySpanCapacity(1), 1U);

    // A run of 36 values at a limit of 12, after two spans already in the plan: three packed spans that follow one
    // another in the input and in the output, the input index (1 of 3 inputs: 2 bits) in the top bits.
    const uint32_t inputBits = raggedSequenceCopySpanInputBits(3);
    EXPECT_EQ(inputBits, 2U);
    std::vector<RaggedSequenceCopySpanPacked> spans(6, RaggedSequenceCopySpanPacked{9, 9});
    EXPECT_EQ(appendRaggedSequenceCopyRun(spans.data(), 2, spans.size(), 1, 100, 500, 36, 12, inputBits), 5U);
    for (uint32_t piece = 0; piece < 3; ++piece) {
        EXPECT_EQ(spans[2 + piece].sourceBegin, 100U + 12U * piece);
        EXPECT_EQ(spans[2 + piece].destinationBeginAndInput, (500U + 12U * piece) | (1U << 30U));
    }
    EXPECT_EQ(spans[5].sourceBegin, 9U);
    // A run that is not a multiple of the limit ends in a shorter span; a run within the limit is one span.
    std::vector<RaggedSequenceCopySpan64> wide(4);
    EXPECT_EQ(appendRaggedSequenceCopyRun(wide.data(), 0, wide.size(), 3, 7, 11, 25, 12, inputBits), 3U);
    EXPECT_EQ(wide[2].sourceBegin, 31U);
    EXPECT_EQ(wide[2].destinationBegin, 35U);
    EXPECT_EQ(wide[2].valueCount, 1U);
    EXPECT_EQ(appendRaggedSequenceCopyRun(wide.data(), 3, wide.size(), 0, 0, 36, 12, 12, inputBits), 4U);
    EXPECT_EQ(wide[3].valueCount, 12U);
    // No room, no limit, and packed spans past what they hold: a source past 32 bits, a destination past 30 bits
    // (2 input bits), an input past 2 bits.
    EXPECT_THROW(appendRaggedSequenceCopyRun(wide.data(), 4, wide.size(), 0, 0, 0, 1, 12, inputBits), std::length_error);
    EXPECT_THROW(appendRaggedSequenceCopyRun(wide.data(), 0, wide.size(), 0, 0, 0, 1, 0, inputBits), std::invalid_argument);
    EXPECT_THROW(appendRaggedSequenceCopyRun(spans.data(), 0, spans.size(), 0, 1ULL << 32U, 0, 1, 12, inputBits), std::overflow_error);
    EXPECT_THROW(appendRaggedSequenceCopyRun(spans.data(), 0, spans.size(), 0, 0, 1ULL << 30U, 1, 12, inputBits), std::overflow_error);
    EXPECT_THROW(appendRaggedSequenceCopyRun(spans.data(), 0, spans.size(), 4, 0, 0, 1, 12, inputBits), std::overflow_error);

    // The format a concatenate takes: packed while its output and inputs fit, wide past that.
    EXPECT_EQ(raggedSequenceCopySpanInputBits(1), 1U);
    EXPECT_EQ(raggedSequenceCopySpanInputBits(2), 1U);
    EXPECT_EQ(raggedSequenceCopySpanInputBits(16), 4U);
    EXPECT_EQ(raggedSequenceCopySpanInputBits(17), 5U);
    EXPECT_EQ(raggedSequenceCopySpanFormatFor(2, 1ULL << 31U, 1ULL << 32U), RaggedSequenceCopySpanFormat::PACKED);
    EXPECT_EQ(raggedSequenceCopySpanFormatFor(2, (1ULL << 31U) + 1, 10), RaggedSequenceCopySpanFormat::WIDE);
    EXPECT_EQ(raggedSequenceCopySpanFormatFor(16, 1ULL << 28U, 10), RaggedSequenceCopySpanFormat::PACKED);
    EXPECT_EQ(raggedSequenceCopySpanFormatFor(17, 1ULL << 28U, 10), RaggedSequenceCopySpanFormat::WIDE);
    EXPECT_EQ(raggedSequenceCopySpanFormatFor(2, 100, (1ULL << 32U) + 1), RaggedSequenceCopySpanFormat::WIDE);
    EXPECT_EQ(raggedSequenceCopySpanBytes(RaggedSequenceCopySpanFormat::PACKED), 8U);
    EXPECT_EQ(raggedSequenceCopySpanBytes(RaggedSequenceCopySpanFormat::WIDE), 32U);

    // The capacity holds every plan: runs of all lengths, among them ones far longer than the rest.
    for (uint64_t seed = 0; seed < 2000; ++seed) {
        const uint64_t runCount = 1 + splitMix64(seed) % 40;
        std::vector<uint64_t> runValues(runCount);
        uint64_t values = 0;
        for (uint64_t run = 0; run < runCount; ++run) {
            const uint64_t draw = splitMix64(seed * 1000 + run);
            runValues[run] = draw % 7 == 0 ? 1 + (draw >> 8U) % 5000 : 1 + (draw >> 8U) % 6;
            values += runValues[run];
        }
        const uint64_t limit = raggedSequenceCopySpanValueLimit(values, runCount);
        std::vector<RaggedSequenceCopySpan64> plan(raggedSequenceCopySpanCapacity(runCount));
        uint64_t count = 0;
        uint64_t begin = 0;
        for (const uint64_t run : runValues) {
            count = appendRaggedSequenceCopyRun(plan.data(), count, plan.size(), 0, begin, begin, run, limit, 1);
            begin += run;
        }
        uint64_t next = 0;
        uint64_t wrong = 0;
        for (uint64_t span = 0; span < count; ++span) {
            wrong += plan[span].destinationBegin != next || plan[span].valueCount == 0 || plan[span].valueCount > limit;
            next += plan[span].valueCount;
        }
        ASSERT_EQ(wrong, 0U) << "seed " << seed;
        ASSERT_EQ(next, values) << "seed " << seed;
    }
}

TEST(RaggedSequenceConcatenate, LargeTablesDeepenTheLanes) {
    REQUIRE_CUDA_DEVICE();
    // Enough spans that a lane takes two and four packets a pass (past 768 blocks at the depth before), and a table
    // of one-packet spans, which stays at one.
    {
        SequenceCase testCase;
        testCase.name = "20000 rows x 2 of about 5 4-byte values";
        testCase.valueBytes = 4;
        testCase.lengths = makeLengths(20000, 2, [](uint64_t row, uint32_t input) { return randomLength(row, input, 5, 50); });
        runSequenceCase(testCase);
    }
    {
        SequenceCase testCase;
        testCase.name = "60000 rows x 2 of about 5 4-byte values";
        testCase.valueBytes = 4;
        testCase.lengths = makeLengths(60000, 2, [](uint64_t row, uint32_t input) { return randomLength(row, input, 5, 50); });
        runSequenceCase(testCase);
    }
    {
        // 1-byte packets, four to a lane: 500 blocks, so the table stays on them.
        SequenceCase testCase;
        testCase.name = "2000 rows x 2 of about 80 1-byte values";
        testCase.valueBytes = 1;
        testCase.lengths = makeLengths(2000, 2, [](uint64_t row, uint32_t input) { return randomLength(row, input, 80); });
        runSequenceCase(testCase);
    }
    {
        SequenceCase testCase;
        testCase.name = "150000 rows x 2 of one 16-byte value";
        testCase.valueBytes = 16;
        testCase.lengths = makeLengths(150000, 2, [](uint64_t row, uint32_t input) { return (row + input) % 5 == 0 ? 0 : 1; });
        testCase.wideSpans = true;
        runSequenceCase(testCase);
    }
}

TEST(RaggedSequenceConcatenate, LargeTablesOfOneAndTwoByteValuesMoveInSixteenBytePackets) {
    REQUIRE_CUDA_DEVICE();
    // Tables of 1- and 2-byte packets past 768 blocks at four packets per lane take the 16-byte kernel: spans that
    // start and end at any byte of arrays that start at any byte, their heads and tails in aligned pieces.
    struct ByteCase {
        const char *name;
        uint64_t valueBytes;
        uint64_t rows;
        uint64_t mean;
        uint32_t emptyPerMille;
        bool wideSpans;
    };
    for (const ByteCase &byteCase : {
             ByteCase{"110000 rows x 2 of about 3 bytes: spans inside one packet", 1, 110000, 3, 100, false},
             ByteCase{"9000 rows x 2 of about 40 bytes", 1, 9000, 40, 50, true},
             ByteCase{"4000 rows x 2 of about 3 33-byte values", 33, 4000, 3, 0, false},
             ByteCase{"15000 rows x 2 of about 10 6-byte values (2-byte packets)", 6, 15000, 10, 50, false},
             ByteCase{"40 rows x 2 of about 60000 bytes: lanes over several blocks, two packets each", 1, 40, 60000, 0, true},
             ByteCase{"3000 rows x 2 of about 250 3-byte values", 3, 3000, 250, 20, false},
         }) {
        SequenceCase testCase;
        testCase.name = byteCase.name;
        testCase.valueBytes = byteCase.valueBytes;
        testCase.wideSpans = byteCase.wideSpans;
        const uint64_t mean = byteCase.mean;
        const uint32_t emptyPerMille = byteCase.emptyPerMille;
        testCase.lengths = makeLengths(byteCase.rows, 2, [mean, emptyPerMille](uint64_t row, uint32_t input) {
            return randomLength(row, input, mean, emptyPerMille);
        });
        runSequenceCase(testCase);
        testCase.name += ", arrays off their boundaries, no gradient for the first input";
        testCase.outputOffsetBytes = 3;
        testCase.inputOffsetBytes = {1, 7};
        testCase.nullGradient = {true, false};
        runSequenceCase(testCase);
    }
    {
        // A long row cut by the plan among short ones, at 2-byte packets.
        SequenceCase testCase;
        testCase.name = "one row of 600000 2-byte values among 20000 rows of 6";
        testCase.valueBytes = 2;
        testCase.lengths = makeLengths(20000, 2, [](uint64_t row, uint32_t input) { return row == 7000 + input ? 600000 : 6; });
        runSequenceCase(testCase);
    }
}

TEST(RaggedSequenceConcatenate, MoreInputsThanALaunchHoldsAndNullableGradients) {
    REQUIRE_CUDA_DEVICE();
    for (const uint32_t inputs : {16U, 17U, 40U}) {
        SequenceCase testCase;
        testCase.name = std::to_string(inputs) + " inputs of odd-width values";
        testCase.valueBytes = 5;
        testCase.lengths = makeLengths(70, inputs, [](uint64_t row, uint32_t input) { return randomLength(row, input, 4, 300); });
        testCase.activeRows = 61;
        for (uint32_t input = 0; input < inputs; ++input) testCase.nullGradient.push_back(input % 3 == 1 || (input >= 16 && input < 32));
        runSequenceCase(testCase);
    }
}

TEST(RaggedSequenceConcatenate, RejectsMalformedLaunches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor values = makeGpuBytes(std::vector<uint8_t>(256, 1), stream);
    Tensor output = makeGpuBytes(std::vector<uint8_t>(256, 2), stream);
    const std::vector<RaggedSequenceCopySpanPacked> spans{{0, 0}};
    std::vector<uint8_t> spanBytes(sizeof(RaggedSequenceCopySpanPacked) + 16);
    std::memcpy(spanBytes.data(), spans.data(), sizeof(RaggedSequenceCopySpanPacked));
    Tensor spanTable = makeGpuBytes(spanBytes, stream);
    const std::vector<void *> inputs{values.getMemPtr(), values.getMemPtr<uint8_t>() + 64};
    const std::vector<void *> inputsWithNull{values.getMemPtr(), nullptr};
    void *const table = spanTable.getMemPtr();

    constexpr RaggedSequenceCopySpanFormat kPacked = RaggedSequenceCopySpanFormat::PACKED;
    EXPECT_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), inputs, table, 1, 0, 1, kPacked, 1, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), inputs, table, 1, 4, 0, kPacked, 1, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), inputs, table, 1, 4, 1, static_cast<RaggedSequenceCopySpanFormat>(7), 1, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), inputs, table, 1, 4, 1, kPacked, 0, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), inputs, table, 0, 4, 1, kPacked, 1, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), inputs, nullptr, 1, 4, 1, kPacked, 1, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), inputs, spanTable.getMemPtr<uint8_t>() + 4, 1, 4, 1, kPacked, 1, stream),
                 std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenate(nullptr, inputs, table, 1, 4, 1, kPacked, 1, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), inputsWithNull, table, 1, 4, 1, kPacked, 1, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), {}, table, 1, 4, 1, kPacked, 1, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenateBackward(inputs, output.getMemPtr(), table, 1, 4, 1, kPacked, 0, stream), std::invalid_argument);
    EXPECT_THROW(launchRaggedSequenceConcatenateBackward(inputs, nullptr, table, 1, 4, 1, kPacked, 1, stream), std::invalid_argument);
    // A null gradient is not an error, and neither is an empty plan for an empty active prefix.
    EXPECT_NO_THROW(launchRaggedSequenceConcatenateBackward(inputsWithNull, output.getMemPtr(), table, 1, 4, 1, kPacked, 1, stream));
    EXPECT_NO_THROW(launchRaggedSequenceConcatenate(output.getMemPtr(), inputs, nullptr, 0, 4, 1, kPacked, 0, stream));
    stream.synchronize();
}
