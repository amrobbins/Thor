// The embedding sparse gradient's runs kernel (the run-length encode of the sorted row keys and the bucketing of the
// runs, in one pass) against a host encode and bucketing: every row type, runs of every bucket alone, mixed, at the
// buckets' edges, in long stretches of one kind and across many tiles (2,048 keys), with and without a last run of
// the sentinel row, key counts read on the device (0 included), and scratch cleared by the launch or by the caller.
// Every output is checked past its count: nothing there may change. The low bucket's entries (run, start, length,
// key's low 32 bits) are checked word for word. The reducers' grids, set on the device, are covered by the Embedding
// tests.

#include "Utilities/TensorOperations/Embedding/ReduceStageController.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                     \
    do {                                                                                          \
        int cuda_device_count_for_test = 0;                                                       \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test); \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {             \
            GTEST_SKIP() << "CUDA device is required for the embedding runs tests.";              \
        }                                                                                         \
    } while (false)

const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
constexpr uint64_t kTile = 2048;
constexpr uint32_t kStale = 0x5A5A5A5AU;

template <typename KeyT>
constexpr DataType keyDataTypeOf() {
    return sizeof(KeyT) == 2 ? DataType::UINT16 : sizeof(KeyT) == 4 ? DataType::UINT32 : DataType::UINT64;
}

struct Buckets {
    uint32_t lowRunMax;
    uint32_t ultraHighRunMin;
    uint32_t ultraHighTokensPerPartial;
};

constexpr Buckets kDefaultBuckets{16U, 512U, 1024U};
constexpr Buckets kWideBuckets{32U, 4096U, 1024U};
constexpr Buckets kTightBuckets{1U, 3U, 2U};  // the narrowest the embedding allows: most runs ultra-high

enum class Lengths { Singles, High, Ultra, Mixed, Edges, Stretches, OneRun, Long };

struct RunsCase {
    uint64_t capacity;
    std::optional<uint64_t> active;  // the device key count; none: the capacity, no device count
    DataType activeDataType = DataType::UINT32;
    bool sentinel = false;  // the keys end with a run of the sentinel row
    Lengths lengths = Lengths::Mixed;
    Buckets buckets = kDefaultBuckets;
    bool scratchAlreadyClear = false;  // the caller zeroes the scratch and the launch is told not to
};

template <typename KeyT>
void expectRuns(const RunsCase& c, uint64_t seed) {
    SCOPED_TRACE("capacity " + std::to_string(c.capacity) + " active " +
                 (c.active ? std::to_string(*c.active) : std::string("none")) + " sentinel " + std::to_string(c.sentinel) +
                 " lengths " + std::to_string(static_cast<int>(c.lengths)) + " low max " + std::to_string(c.buckets.lowRunMax));
    Stream stream(0);
    const uint64_t vocabularySize = sizeof(KeyT) == 2 ? 60000ULL : 3000000000ULL;
    const Buckets b = c.buckets;
    const uint64_t n = c.active ? std::min(*c.active, c.capacity) : c.capacity;

    const TensorDescriptor keysDescriptor(keyDataTypeOf<KeyT>(), {c.capacity});
    const TensorDescriptor wordsDescriptor(DataType::UINT32, {c.capacity});
    const TensorDescriptor oneKeyDescriptor(keyDataTypeOf<KeyT>(), {1});
    const TensorDescriptor oneWordDescriptor(DataType::UINT32, {1});
    const TensorDescriptor entriesDescriptor(DataType::UINT32, {c.capacity, 4});
    Tensor keysCpu(cpu, keysDescriptor), staleKeysCpu(cpu, keysDescriptor), staleCpu(cpu, wordsDescriptor),
        staleEntriesCpu(cpu, entriesDescriptor);

    // Sorted keys: runs of increasing keys below the vocabulary, then the sentinel's run when asked.
    std::mt19937_64 rng(seed);
    const uint64_t sentinelLength = c.sentinel ? 1 + rng() % 3000 : 0;
    const uint64_t body = c.capacity > sentinelLength ? c.capacity - sentinelLength : 0;
    const uint32_t highSpan = b.ultraHighRunMin - b.lowRunMax - 1U;
    uint64_t key = rng() % 7;
    uint64_t stretch = 0;
    uint32_t stretchKind = 0;
    uint64_t i = 0;
    while (i < body) {
        uint64_t length = 1;
        switch (c.lengths) {
            case Lengths::Singles: length = 1; break;
            case Lengths::High: length = b.lowRunMax + 1U + rng() % highSpan; break;
            case Lengths::Ultra: length = b.ultraHighRunMin + rng() % 5000U; break;
            case Lengths::Mixed: {
                const uint32_t pick = rng() % 10U;
                length = pick < 6U   ? 1U + rng() % b.lowRunMax
                         : pick < 9U ? b.lowRunMax + 1U + rng() % highSpan
                                     : b.ultraHighRunMin + rng() % 9000U;
                break;
            }
            case Lengths::Edges: {
                const uint64_t edges[6] = {b.lowRunMax,       b.lowRunMax + 1U, b.ultraHighRunMin - 1U,
                                           b.ultraHighRunMin, 1U,               b.ultraHighRunMin + b.ultraHighTokensPerPartial};
                length = edges[rng() % 6U];
                break;
            }
            case Lengths::Stretches:
                if (stretch == 0) {
                    stretch = 1 + rng() % (2 * kTile + 900);
                    stretchKind = rng() % 3U;
                }
                --stretch;
                length = stretchKind == 0U ? 1U : stretchKind == 1U ? b.lowRunMax + 1U : b.ultraHighRunMin + rng() % 3000U;
                break;
            case Lengths::OneRun: length = body; break;
            case Lengths::Long: length = 1 + rng() % (3 * kTile); break;
        }
        for (uint64_t k = 0; k < length && i < body; ++k, ++i) keysCpu.getMemPtr<KeyT>()[i] = static_cast<KeyT>(key);
        key = std::min(key + 1 + rng() % 3, vocabularySize - 1);
    }
    for (; i < c.capacity; ++i) keysCpu.getMemPtr<KeyT>()[i] = static_cast<KeyT>(c.sentinel ? vocabularySize : key);
    const KeyT staleKey = static_cast<KeyT>(0xA5A5A5A5A5A5A5A5ULL);
    for (uint64_t k = 0; k < c.capacity; ++k) {
        staleKeysCpu.getMemPtr<KeyT>()[k] = staleKey;
        staleCpu.getMemPtr<uint32_t>()[k] = kStale;
    }
    for (uint64_t k = 0; k < 4 * c.capacity; ++k) staleEntriesCpu.getMemPtr<uint32_t>()[k] = kStale;

    Tensor keys(gpu, keysDescriptor), rows(gpu, keysDescriptor), numRows(gpu, oneKeyDescriptor);
    Tensor runCounts(gpu, wordsDescriptor), runOffsets(gpu, wordsDescriptor), high(gpu, wordsDescriptor), ultra(gpu, wordsDescriptor),
        partialCounts(gpu, wordsDescriptor), partialOffsets(gpu, wordsDescriptor);
    Tensor low(gpu, entriesDescriptor);
    Tensor numRuns(gpu, oneWordDescriptor), numLow(gpu, oneWordDescriptor), numHigh(gpu, oneWordDescriptor),
        numUltra(gpu, oneWordDescriptor), numPartials(gpu, oneWordDescriptor);
    Tensor scratch(gpu, TensorDescriptor(DataType::UINT64, {embeddingSparseGradientRunsScratchWords(c.capacity)}));
    keys.copyFromAsync(keysCpu, stream);
    rows.copyFromAsync(staleKeysCpu, stream);
    for (Tensor* list : {&runCounts, &runOffsets, &high, &ultra, &partialCounts, &partialOffsets}) list->copyFromAsync(staleCpu, stream);
    low.copyFromAsync(staleEntriesCpu, stream);
    for (Tensor* one : {&numRows, &numRuns, &numLow, &numHigh, &numUltra, &numPartials}) one->memsetAsync(stream, 0x77);
    // The launch must not depend on what the scratch held, unless it is told the scratch is clear.
    scratch.memsetAsync(stream, c.scratchAlreadyClear ? 0 : 0x5E);

    std::optional<Tensor> activeCount;
    const void* activePointer = nullptr;
    if (c.active) {
        Tensor activeCpu(cpu, TensorDescriptor(c.activeDataType, {1}));
        if (c.activeDataType == DataType::UINT64) {
            activeCpu.getMemPtr<uint64_t>()[0] = *c.active;
        } else {
            activeCpu.getMemPtr<uint32_t>()[0] = static_cast<uint32_t>(*c.active);
        }
        activeCount = Tensor(gpu, activeCpu.getDescriptor());
        activeCount->copyFromAsync(activeCpu, stream);
        activePointer = activeCount->getMemPtr();
    }

    launchEncodeAndBucketizeEmbeddingSparseGradientRuns(keys.getMemPtr(),
                                                        keyDataTypeOf<KeyT>(),
                                                        c.capacity,
                                                        activePointer,
                                                        c.activeDataType,
                                                        rows.getMemPtr(),
                                                        runCounts.getMemPtr<uint32_t>(),
                                                        runOffsets.getMemPtr<uint32_t>(),
                                                        numRuns.getMemPtr<uint32_t>(),
                                                        numRows.getMemPtr(),
                                                        low.getMemPtr<uint32_t>(),
                                                        high.getMemPtr<uint32_t>(),
                                                        ultra.getMemPtr<uint32_t>(),
                                                        partialCounts.getMemPtr<uint32_t>(),
                                                        partialOffsets.getMemPtr<uint32_t>(),
                                                        numPartials.getMemPtr<uint32_t>(),
                                                        numLow.getMemPtr<uint32_t>(),
                                                        numHigh.getMemPtr<uint32_t>(),
                                                        numUltra.getMemPtr<uint32_t>(),
                                                        scratch.getMemPtr<uint64_t>(),
                                                        vocabularySize,
                                                        b.lowRunMax,
                                                        b.ultraHighRunMin,
                                                        b.ultraHighTokensPerPartial,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        {},
                                                        {},
                                                        {},
                                                        {},
                                                        stream,
                                                        /*clearScratch=*/!c.scratchAlreadyClear);

    std::vector<Tensor> lists;
    for (Tensor* list : {&runCounts, &runOffsets, &high, &ultra, &partialCounts, &partialOffsets}) {
        lists.emplace_back(cpu, wordsDescriptor);
        lists.back().copyFromAsync(*list, stream);
    }
    Tensor lowCpu(cpu, entriesDescriptor);
    lowCpu.copyFromAsync(low, stream);
    std::vector<Tensor> ones;
    for (Tensor* one : {&numRuns, &numLow, &numHigh, &numUltra, &numPartials}) {
        ones.emplace_back(cpu, oneWordDescriptor);
        ones.back().copyFromAsync(*one, stream);
    }
    Tensor rowsCpu(cpu, keysDescriptor), numRowsCpu(cpu, oneKeyDescriptor);
    rowsCpu.copyFromAsync(rows, stream);
    numRowsCpu.copyFromAsync(numRows, stream);
    stream.synchronize();

    // The host's runs and buckets.
    std::vector<KeyT> expectedRows;
    std::vector<uint32_t> expectedCounts, expectedOffsets, expectedLow, expectedHigh, expectedUltra, expectedPartialCounts,
        expectedPartialOffsets;
    const KeyT* input = keysCpu.getMemPtr<KeyT>();
    for (uint64_t k = 0; k < n; ++k) {
        if (k == 0 || input[k] != input[k - 1]) {
            expectedRows.push_back(input[k]);
            expectedOffsets.push_back(static_cast<uint32_t>(k));
            expectedCounts.push_back(0U);
        }
        ++expectedCounts.back();
    }
    const uint64_t runs = expectedRows.size();
    const uint64_t validRuns = runs != 0 && static_cast<uint64_t>(expectedRows.back()) == vocabularySize ? runs - 1 : runs;
    uint32_t partials = 0;
    for (uint64_t r = 0; r < validRuns; ++r) {
        const uint32_t tokens = expectedCounts[r];
        if (tokens >= b.ultraHighRunMin) {
            const uint32_t runPartials = (tokens + b.ultraHighTokensPerPartial - 1U) / b.ultraHighTokensPerPartial;
            expectedUltra.push_back(static_cast<uint32_t>(r));
            expectedPartialCounts.push_back(runPartials);
            expectedPartialOffsets.push_back(partials);
            partials += runPartials;
        } else if (tokens > b.lowRunMax) {
            expectedHigh.push_back(static_cast<uint32_t>(r));
        } else {
            for (const uint32_t word : {static_cast<uint32_t>(r), expectedOffsets[r], tokens, static_cast<uint32_t>(expectedRows[r])}) {
                expectedLow.push_back(word);
            }
        }
    }

    EXPECT_EQ(ones[0].getMemPtr<uint32_t>()[0], runs);
    EXPECT_EQ(static_cast<uint64_t>(numRowsCpu.getMemPtr<KeyT>()[0]), validRuns);
    EXPECT_EQ(ones[1].getMemPtr<uint32_t>()[0], expectedLow.size() / 4);
    EXPECT_EQ(ones[2].getMemPtr<uint32_t>()[0], expectedHigh.size());
    EXPECT_EQ(ones[3].getMemPtr<uint32_t>()[0], expectedUltra.size());
    EXPECT_EQ(ones[4].getMemPtr<uint32_t>()[0], partials);

    uint64_t wrongRows = 0;
    for (uint64_t r = 0; r < c.capacity; ++r) {
        const KeyT want = r < runs ? expectedRows[r] : staleKey;  // nothing past the runs
        if (rowsCpu.getMemPtr<KeyT>()[r] != want) ++wrongRows;
    }
    EXPECT_EQ(wrongRows, 0U) << "rows";
    uint64_t wrongLow = 0;
    uint64_t firstWrongLow = 0;
    for (uint64_t w = 0; w < 4 * c.capacity; ++w) {
        const uint32_t want = w < expectedLow.size() ? expectedLow[w] : kStale;  // nothing past the bucket's size
        if (lowCpu.getMemPtr<uint32_t>()[w] != want && wrongLow++ == 0) firstWrongLow = w;
    }
    EXPECT_EQ(wrongLow, 0U) << "low run entries: first wrong word " << firstWrongLow << " of " << expectedLow.size();
    const char* names[6] = {
        "run counts", "run offsets", "high runs", "ultra-high runs", "ultra-high partial counts", "ultra-high partial offsets"};
    const std::vector<uint32_t>* expected[6] = {&expectedCounts, &expectedOffsets,       &expectedHigh,
                                                &expectedUltra,  &expectedPartialCounts, &expectedPartialOffsets};
    for (uint32_t l = 0; l < 6; ++l) {
        uint64_t wrong = 0;
        uint64_t firstWrong = 0;
        for (uint64_t r = 0; r < c.capacity; ++r) {
            const uint32_t want = r < expected[l]->size() ? (*expected[l])[r] : kStale;  // nothing past the list's size
            if (lists[l].getMemPtr<uint32_t>()[r] != want && wrong++ == 0) firstWrong = r;
        }
        EXPECT_EQ(wrong, 0U) << names[l] << ": first wrong entry " << firstWrong << " of " << expected[l]->size();
    }
}

template <typename KeyT>
void expectEveryLayout(uint64_t& seed) {
    struct Size {
        uint64_t capacity;
        bool sentinel;
    };
    for (const Size size : {Size{1, false},
                            Size{1, true},
                            Size{2, true},
                            Size{31, false},
                            Size{kTile - 1, true},
                            Size{kTile, false},
                            Size{kTile, true},
                            Size{kTile + 1, false},
                            Size{3 * kTile + 77, true},
                            Size{9 * kTile, false},
                            Size{9 * kTile, true}}) {
        for (const Lengths lengths : {Lengths::Singles, Lengths::High, Lengths::Ultra, Lengths::Mixed, Lengths::Edges, Lengths::Stretches,
                                      Lengths::OneRun, Lengths::Long}) {
            const Buckets buckets = seed % 3 == 0 ? kDefaultBuckets : seed % 3 == 1 ? kWideBuckets : kTightBuckets;
            expectRuns<KeyT>(RunsCase{size.capacity, std::nullopt, DataType::UINT32, size.sentinel, lengths, buckets}, seed);
            ++seed;
        }
    }
}

}  // namespace

TEST(EmbeddingSparseGradientRuns, ScratchIsThreeWordsATile) {
    EXPECT_EQ(embeddingSparseGradientRunsScratchWords(1), 3U);
    EXPECT_EQ(embeddingSparseGradientRunsScratchWords(2048), 3U);
    EXPECT_EQ(embeddingSparseGradientRunsScratchWords(2049), 6U);
    EXPECT_EQ(embeddingSparseGradientRunsScratchWords(100000), 147U);
    EXPECT_THROW((void)embeddingSparseGradientRunsScratchWords(0), std::invalid_argument);
    EXPECT_THROW((void)embeddingSparseGradientRunsScratchWords(uint64_t{1} << 30), std::invalid_argument);
}

TEST(EmbeddingSparseGradientRuns, RejectsMalformedLaunches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const TensorDescriptor words(DataType::UINT32, {1000});
    Tensor keys(gpu, words), rows(gpu, words), counts(gpu, words), offsets(gpu, words), low(gpu, words), high(gpu, words),
        ultra(gpu, words), partialCounts(gpu, words), partialOffsets(gpu, words);
    Tensor one(gpu, TensorDescriptor(DataType::UINT32, {1}));
    Tensor scratch(gpu, TensorDescriptor(DataType::UINT64, {embeddingSparseGradientRunsScratchWords(1000)}));
    const auto launch = [&](const void* keysPointer,
                            void* rowsPointer,
                            uint64_t* scratchPointer,
                            uint64_t capacity,
                            DataType keyDataType,
                            const void* activePointer,
                            DataType activeDataType,
                            uint32_t tokensPerPartial,
                            uint32_t lowOffsetWords) {
        launchEncodeAndBucketizeEmbeddingSparseGradientRuns(keysPointer,
                                                            keyDataType,
                                                            capacity,
                                                            activePointer,
                                                            activeDataType,
                                                            rowsPointer,
                                                            counts.getMemPtr<uint32_t>(),
                                                            offsets.getMemPtr<uint32_t>(),
                                                            one.getMemPtr<uint32_t>(),
                                                            one.getMemPtr(),
                                                            low.getMemPtr<uint32_t>() + lowOffsetWords,
                                                            high.getMemPtr<uint32_t>(),
                                                            ultra.getMemPtr<uint32_t>(),
                                                            partialCounts.getMemPtr<uint32_t>(),
                                                            partialOffsets.getMemPtr<uint32_t>(),
                                                            one.getMemPtr<uint32_t>(),
                                                            one.getMemPtr<uint32_t>(),
                                                            one.getMemPtr<uint32_t>(),
                                                            one.getMemPtr<uint32_t>(),
                                                            scratchPointer,
                                                            1000,
                                                            16U,
                                                            512U,
                                                            tokensPerPartial,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            {},
                                                            {},
                                                            {},
                                                            {},
                                                            stream);
    };
    void* k = keys.getMemPtr();
    void* r = rows.getMemPtr();
    uint64_t* s = scratch.getMemPtr<uint64_t>();
    EXPECT_THROW(launch(nullptr, r, s, 1000, DataType::UINT32, nullptr, DataType::UINT32, 1024U, 0U), std::invalid_argument);
    EXPECT_THROW(launch(k, r, nullptr, 1000, DataType::UINT32, nullptr, DataType::UINT32, 1024U, 0U), std::invalid_argument);
    EXPECT_THROW(launch(k, k, s, 1000, DataType::UINT32, nullptr, DataType::UINT32, 1024U, 0U), std::invalid_argument);
    EXPECT_THROW(launch(k, r, s, 0, DataType::UINT32, nullptr, DataType::UINT32, 1024U, 0U), std::invalid_argument);
    EXPECT_THROW(launch(k, r, s, uint64_t{1} << 30, DataType::UINT32, nullptr, DataType::UINT32, 1024U, 0U), std::invalid_argument);
    EXPECT_THROW(launch(k, r, s, 1000, DataType::UINT8, nullptr, DataType::UINT32, 1024U, 0U), std::invalid_argument);
    EXPECT_THROW(launch(k, r, s, 1000, DataType::FP32, nullptr, DataType::UINT32, 1024U, 0U), std::invalid_argument);
    EXPECT_THROW(launch(k, r, s, 1000, DataType::UINT32, one.getMemPtr(), DataType::UINT16, 1024U, 0U), std::invalid_argument);
    EXPECT_THROW(launch(k, r, s, 1000, DataType::UINT32, nullptr, DataType::UINT32, 0U, 0U), std::invalid_argument);
    EXPECT_THROW(launch(k, r, s, 1000, DataType::UINT32, nullptr, DataType::UINT32, 1024U, 1U), std::invalid_argument);  // unaligned
    stream.synchronize();
}

TEST(EmbeddingSparseGradientRuns, EncodesAndBucketsEveryRowTypeAndRunLayout) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 1;
    expectEveryLayout<uint16_t>(seed);
    expectEveryLayout<uint32_t>(seed);
    expectEveryLayout<uint64_t>(seed);
}

TEST(EmbeddingSparseGradientRuns, EncodesOnlyTheDeviceKeyCount) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 1000;
    for (const DataType countType : {DataType::UINT32, DataType::UINT64}) {
        for (const uint64_t active :
             {uint64_t{0}, uint64_t{1}, kTile - 1, kTile, kTile + 1, 4 * kTile + 9, 9 * kTile, 20 * kTile}) {
            for (const Lengths lengths : {Lengths::Mixed, Lengths::Long, Lengths::Stretches}) {
                for (const bool sentinel : {false, true}) {
                    expectRuns<uint32_t>(RunsCase{9 * kTile + 1, active, countType, sentinel, lengths, kDefaultBuckets}, seed++);
                    expectRuns<uint16_t>(RunsCase{5 * kTile, active, countType, sentinel, lengths, kTightBuckets}, seed++);
                }
            }
        }
    }
}

// The scratch zeroed by the caller (in a graph, by the materialize kernel) and no memset by the launch.
TEST(EmbeddingSparseGradientRuns, EncodesAndBucketsOnScratchTheCallerCleared) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 2000;
    for (const Lengths lengths : {Lengths::Singles, Lengths::Mixed, Lengths::Edges, Lengths::Stretches}) {
        expectRuns<uint32_t>(RunsCase{9 * kTile, std::nullopt, DataType::UINT32, true, lengths, kDefaultBuckets, true}, seed++);
        expectRuns<uint16_t>(RunsCase{4 * kTile, uint64_t{2 * kTile + 1}, DataType::UINT64, false, lengths, kTightBuckets, true},
                             seed++);
        expectRuns<uint64_t>(RunsCase{5 * kTile, uint64_t{0}, DataType::UINT32, true, lengths, kWideBuckets, true}, seed++);
    }
}

// Many tiles: more than run at once, so the lookbacks meet rows not yet written.
TEST(EmbeddingSparseGradientRuns, EncodesAndBucketsLargeInputs) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 5000;
    expectRuns<uint32_t>(RunsCase{3000001, std::nullopt, DataType::UINT32, false, Lengths::Singles, kDefaultBuckets}, seed++);
    expectRuns<uint32_t>(RunsCase{3000001, std::nullopt, DataType::UINT32, true, Lengths::Mixed, kDefaultBuckets}, seed++);
    expectRuns<uint64_t>(RunsCase{1048576, uint64_t{700001}, DataType::UINT64, true, Lengths::Stretches, kWideBuckets}, seed++);
    expectRuns<uint32_t>(RunsCase{700000, std::nullopt, DataType::UINT32, true, Lengths::Edges, kTightBuckets}, seed++);
    expectRuns<uint32_t>(RunsCase{700000, std::nullopt, DataType::UINT32, false, Lengths::Ultra, kTightBuckets}, seed++);
}
