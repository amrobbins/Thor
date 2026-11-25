// Thor's radix sort of (key, value) pairs against std::stable_sort: every key type, digit layouts from 1 to 64 key
// bits, tiles of 512 and 4096 pairs, counts around a tile's edges, and item counts read on the device. Both
// pairs of arrays are checked past the item count: nothing there may change.

#include "Utilities/TensorOperations/Sort/RadixSort.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
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
            GTEST_SKIP() << "CUDA device is required for radix sort tests.";                      \
        }                                                                                         \
    } while (false)

const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);

template <typename KeyT>
constexpr DataType keyDataTypeOf() {
    return sizeof(KeyT) == 2 ? DataType::UINT16 : sizeof(KeyT) == 4 ? DataType::UINT32 : DataType::UINT64;
}

enum class Keys { Random, FewValues, Ascending, Skewed };

struct SortCase {
    uint64_t capacity;
    uint32_t keyBits;
    uint32_t tilePairs;              // 0: the plan chooses
    std::optional<uint64_t> active;  // the device item count; none: the capacity, no device count
    DataType activeDataType = DataType::UINT32;
    Keys keys = Keys::Random;
    bool scratchAlreadyClear = false;  // the caller zeroes the scratch and the launch is told not to
};

template <typename KeyT>
void expectSorted(const SortCase& c, uint64_t seed) {
    SCOPED_TRACE("capacity " + std::to_string(c.capacity) + " bits " + std::to_string(c.keyBits) + " tile " +
                 std::to_string(c.tilePairs) + " active " +
                 (c.active ? std::to_string(*c.active) : std::string("none")) + " keys " +
                 std::to_string(static_cast<int>(c.keys)));
    Stream stream(0);
    const RadixSortPairsPlan plan = planRadixSortPairs(keyDataTypeOf<KeyT>(), c.capacity, c.keyBits, c.tilePairs);
    const uint64_t n = c.active ? std::min(*c.active, c.capacity) : c.capacity;

    const TensorDescriptor keysDescriptor(keyDataTypeOf<KeyT>(), {c.capacity});
    const TensorDescriptor valuesDescriptor(DataType::UINT32, {c.capacity});
    Tensor keysCpu(cpu, keysDescriptor), valuesCpu(cpu, valuesDescriptor), staleKeysCpu(cpu, keysDescriptor),
        staleValuesCpu(cpu, valuesDescriptor);
    std::mt19937_64 rng(seed);
    const uint64_t mask = c.keyBits == 64 ? ~uint64_t{0} : (uint64_t{1} << c.keyBits) - 1;
    const KeyT staleKey = static_cast<KeyT>(0xA5A5A5A5A5A5A5A5ULL);
    for (uint64_t i = 0; i < c.capacity; ++i) {
        uint64_t key = 0;
        switch (c.keys) {
            case Keys::Random: key = rng(); break;
            case Keys::FewValues: key = rng() % 3; break;
            case Keys::Ascending: key = i * 7; break;
            case Keys::Skewed: key = rng() % 5 == 0 ? mask : rng() % 17; break;
        }
        keysCpu.getMemPtr<KeyT>()[i] = static_cast<KeyT>(key & mask);
        valuesCpu.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(rng());
        staleKeysCpu.getMemPtr<KeyT>()[i] = staleKey;
        staleValuesCpu.getMemPtr<uint32_t>()[i] = 0x5A5A5A5AU;
    }

    Tensor keys(gpu, keysDescriptor), keysAlternate(gpu, keysDescriptor), values(gpu, valuesDescriptor),
        valuesAlternate(gpu, valuesDescriptor);
    Tensor scratch(gpu, TensorDescriptor(DataType::UINT32, {plan.scratchWords}));
    keys.copyFromAsync(keysCpu, stream);
    values.copyFromAsync(valuesCpu, stream);
    keysAlternate.copyFromAsync(staleKeysCpu, stream);
    valuesAlternate.copyFromAsync(staleValuesCpu, stream);
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

    // Twice on the same scratch: the second launch sorts the first one's output back from scrambled state.
    launchRadixSortPairs(plan,
                         keys.getMemPtr(),
                         keysAlternate.getMemPtr(),
                         values.getMemPtr<uint32_t>(),
                         valuesAlternate.getMemPtr<uint32_t>(),
                         scratch.getMemPtr<uint32_t>(),
                         activePointer,
                         c.activeDataType,
                         stream,
                         /*clearScratch=*/!c.scratchAlreadyClear);
    Tensor sortedKeysCpu(cpu, keysDescriptor), sortedValuesCpu(cpu, valuesDescriptor),
        otherKeysCpu(cpu, keysDescriptor), otherValuesCpu(cpu, valuesDescriptor);
    sortedKeysCpu.copyFromAsync(plan.sortedInAlternate ? keysAlternate : keys, stream);
    sortedValuesCpu.copyFromAsync(plan.sortedInAlternate ? valuesAlternate : values, stream);
    otherKeysCpu.copyFromAsync(plan.sortedInAlternate ? keys : keysAlternate, stream);
    otherValuesCpu.copyFromAsync(plan.sortedInAlternate ? values : valuesAlternate, stream);
    stream.synchronize();

    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0U);
    const KeyT* inputKeys = keysCpu.getMemPtr<KeyT>();
    const uint32_t* inputValues = valuesCpu.getMemPtr<uint32_t>();
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return inputKeys[a] < inputKeys[b]; });

    uint64_t wrong = 0;
    uint64_t firstWrong = 0;
    for (uint64_t i = 0; i < n; ++i) {
        if (sortedKeysCpu.getMemPtr<KeyT>()[i] != inputKeys[order[i]] ||
            sortedValuesCpu.getMemPtr<uint32_t>()[i] != inputValues[order[i]]) {
            if (wrong++ == 0) firstWrong = i;
        }
    }
    EXPECT_EQ(wrong, 0U) << "first wrong pair at " << firstWrong;

    // Past the item count nothing is written: the (keys, values) side keeps the input, the alternates their stale
    // bytes.
    const bool inputSideIsSorted = !plan.sortedInAlternate;
    uint64_t touched = 0;
    for (uint64_t i = n; i < c.capacity; ++i) {
        const KeyT inputSideKey = (inputSideIsSorted ? sortedKeysCpu : otherKeysCpu).getMemPtr<KeyT>()[i];
        const uint32_t inputSideValue = (inputSideIsSorted ? sortedValuesCpu : otherValuesCpu).getMemPtr<uint32_t>()[i];
        const KeyT alternateKey = (inputSideIsSorted ? otherKeysCpu : sortedKeysCpu).getMemPtr<KeyT>()[i];
        const uint32_t alternateValue = (inputSideIsSorted ? otherValuesCpu : sortedValuesCpu).getMemPtr<uint32_t>()[i];
        if (inputSideKey != inputKeys[i] || inputSideValue != inputValues[i] || alternateKey != staleKey ||
            alternateValue != 0x5A5A5A5AU) {
            ++touched;
        }
    }
    EXPECT_EQ(touched, 0U) << "pairs past the item count were written";
}

}  // namespace

TEST(RadixSortPairs, PlansDigitsTilesAndScratch) {
    const RadixSortPairsPlan plan = planRadixSortPairs(DataType::UINT32, 100000, 21);
    EXPECT_EQ(plan.passes, 3U);
    EXPECT_EQ(plan.digitBits[0], 7U);
    EXPECT_EQ(plan.digitBits[1], 7U);
    EXPECT_EQ(plan.digitBits[2], 7U);
    EXPECT_EQ(plan.digitShift[0], 0U);
    EXPECT_EQ(plan.digitShift[1], 7U);
    EXPECT_EQ(plan.digitShift[2], 14U);
    EXPECT_EQ(plan.tilePairs, 512U);
    EXPECT_EQ(plan.tiles, 196U);
    EXPECT_EQ(plan.histogramTiles, 25U);
    EXPECT_EQ(plan.scratchWords, 3U * 256U + 3U * 196U * 128U);
    EXPECT_TRUE(plan.sortedInAlternate);

    const RadixSortPairsPlan uneven = planRadixSortPairs(DataType::UINT32, 131073, 17);
    EXPECT_EQ(uneven.passes, 3U);
    EXPECT_EQ(uneven.digitBits[0], 6U);
    EXPECT_EQ(uneven.digitBits[1], 6U);
    EXPECT_EQ(uneven.digitBits[2], 5U);
    EXPECT_EQ(uneven.digitShift[2], 12U);
    EXPECT_EQ(uneven.tilePairs, 512U);  // up to 262144 pairs with digits of up to 7 bits
    EXPECT_EQ(uneven.tiles, 257U);

    EXPECT_EQ(planRadixSortPairs(DataType::UINT32, 262144, 21).tilePairs, 512U);
    EXPECT_EQ(planRadixSortPairs(DataType::UINT32, 262145, 21).tilePairs, 4096U);
    EXPECT_EQ(planRadixSortPairs(DataType::UINT32, 131072, 16).tilePairs, 512U);  // 8-bit digits
    EXPECT_EQ(planRadixSortPairs(DataType::UINT32, 131073, 16).tilePairs, 4096U);
    EXPECT_FALSE(planRadixSortPairs(DataType::UINT16, 1000, 16).sortedInAlternate);
    EXPECT_EQ(planRadixSortPairs(DataType::UINT64, 1000, 64).passes, 8U);
    EXPECT_EQ(planRadixSortPairs(DataType::UINT32, 1000, 8, 4096).tilePairs, 4096U);
    EXPECT_EQ(planRadixSortPairs(DataType::UINT32, 1000, 8, 4096).tiles, 1U);

    EXPECT_THROW((void)planRadixSortPairs(DataType::INT32, 1000, 8), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairs(DataType::UINT32, 0, 8), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairs(DataType::UINT32, uint64_t{1} << 30, 8), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairs(DataType::UINT16, 1000, 17), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairs(DataType::UINT32, 1000, 0), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairs(DataType::UINT32, 1000, 8, 1024), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairs(DataType::UINT32, 1000, 8, 2048), std::invalid_argument);
}

TEST(RadixSortPairs, RejectsMalformedLaunches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    RadixSortPairsPlan plan = planRadixSortPairs(DataType::UINT32, 64, 8);
    Tensor a(gpu, TensorDescriptor(DataType::UINT32, {64})), b(gpu, TensorDescriptor(DataType::UINT32, {64})),
        c(gpu, TensorDescriptor(DataType::UINT32, {64})), d(gpu, TensorDescriptor(DataType::UINT32, {64}));
    Tensor scratch(gpu, TensorDescriptor(DataType::UINT32, {plan.scratchWords}));
    auto launch = [&](const RadixSortPairsPlan& p,
                      void* keys,
                      void* keysAlternate,
                      uint32_t* values,
                      uint32_t* valuesAlternate,
                      uint32_t* scratchWords,
                      const void* active,
                      DataType activeType) {
        launchRadixSortPairs(p, keys, keysAlternate, values, valuesAlternate, scratchWords, active, activeType, stream);
    };
    uint32_t* pa = a.getMemPtr<uint32_t>();
    uint32_t* pb = b.getMemPtr<uint32_t>();
    uint32_t* pc = c.getMemPtr<uint32_t>();
    uint32_t* pd = d.getMemPtr<uint32_t>();
    uint32_t* ps = scratch.getMemPtr<uint32_t>();
    EXPECT_THROW(launch(plan, nullptr, pb, pc, pd, ps, nullptr, DataType::UINT32), std::invalid_argument);
    EXPECT_THROW(launch(plan, pa, pb, pc, pd, nullptr, nullptr, DataType::UINT32), std::invalid_argument);
    EXPECT_THROW(launch(plan, pa, pa, pc, pd, ps, nullptr, DataType::UINT32), std::invalid_argument);
    EXPECT_THROW(launch(plan, pa, pb, pc, pc, ps, nullptr, DataType::UINT32), std::invalid_argument);
    EXPECT_THROW(launch(plan, pa, pb, pc, pd, ps, pa, DataType::UINT16), std::invalid_argument);
    RadixSortPairsPlan edited = plan;
    edited.tiles += 1;
    EXPECT_THROW(launch(edited, pa, pb, pc, pd, ps, nullptr, DataType::UINT32), std::invalid_argument);
    stream.synchronize();
}

TEST(RadixSortPairs, SortsEveryKeyTypeAndDigitLayout) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 1;
    for (const uint32_t tilePairs : {512U, 4096U}) {
        const uint64_t capacity = 5 * uint64_t{tilePairs} / 2 + 904;  // three tiles, the last one part full
        for (const Keys keys : {Keys::Random, Keys::FewValues, Keys::Ascending, Keys::Skewed}) {
            for (const uint32_t bits : {1U, 9U, 16U})
                expectSorted<uint16_t>(SortCase{capacity, bits, tilePairs, {}, DataType::UINT32, keys}, ++seed);
            for (const uint32_t bits : {3U, 8U, 17U, 21U, 32U}) {
                expectSorted<uint32_t>(SortCase{capacity, bits, tilePairs, {}, DataType::UINT32, keys}, ++seed);
            }
            for (const uint32_t bits : {21U, 40U, 64U})
                expectSorted<uint64_t>(SortCase{capacity, bits, tilePairs, {}, DataType::UINT32, keys}, ++seed);
        }
    }
}

TEST(RadixSortPairs, SortsCountsAroundATilesEdges) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 1000;
    for (const uint32_t tilePairs : {512U, 4096U}) {
        const uint64_t tile = tilePairs;
        for (const uint64_t capacity : {uint64_t{1}, uint64_t{5}, tile - 1, tile, tile + 1, 3 * tile + 77, 6 * tile}) {
            expectSorted<uint32_t>(SortCase{capacity, 21, tilePairs, {}}, ++seed);
            expectSorted<uint16_t>(SortCase{capacity, 12, tilePairs, {}, DataType::UINT32, Keys::Skewed}, ++seed);
        }
    }
}

TEST(RadixSortPairs, SortsOnlyTheDeviceItemCount) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 2000;
    for (const uint32_t tilePairs : {512U, 4096U}) {
        const uint64_t tile = tilePairs;
        for (const DataType activeType : {DataType::UINT32, DataType::UINT64}) {
            expectSorted<uint32_t>(SortCase{9 * tile + 1, 21, tilePairs, 4 * tile + 9, activeType}, ++seed);
            expectSorted<uint32_t>(SortCase{5 * tile, 17, tilePairs, 0, activeType}, ++seed);
            expectSorted<uint32_t>(SortCase{5 * tile, 17, tilePairs, 1, activeType}, ++seed);
            expectSorted<uint16_t>(SortCase{4 * tile, 16, tilePairs, 2 * tile, activeType}, ++seed);
            expectSorted<uint64_t>(SortCase{4 * tile + 3, 33, tilePairs, 9 * tile, activeType},
                                   ++seed);  // past the capacity
            expectSorted<uint32_t>(SortCase{4 * tile + 3, 21, tilePairs, 4 * tile + 3, activeType, Keys::FewValues},
                                   ++seed);
        }
    }
}

// The scratch zeroed by the caller (in a graph, by an earlier kernel) and no memset by the launch.
TEST(RadixSortPairs, SortsOnScratchTheCallerCleared) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 3000;
    for (const uint32_t tilePairs : {512U, 4096U}) {
        const uint64_t tile = tilePairs;
        expectSorted<uint32_t>(SortCase{9 * tile + 1, 21, tilePairs, {}, DataType::UINT32, Keys::Random, true}, ++seed);
        expectSorted<uint32_t>(SortCase{9 * tile + 1, 17, tilePairs, 4 * tile + 9, DataType::UINT64, Keys::Skewed, true},
                               ++seed);
        expectSorted<uint16_t>(SortCase{4 * tile, 16, tilePairs, {}, DataType::UINT32, Keys::FewValues, true}, ++seed);
        expectSorted<uint64_t>(SortCase{3 * tile + 77, 40, tilePairs, {}, DataType::UINT32, Keys::Random, true}, ++seed);
    }
    expectSorted<uint32_t>(SortCase{1000003, 21, 0, {}, DataType::UINT32, Keys::Random, true}, ++seed);
}

// Many tiles: the lookback runs through tiles that publish in any order. Twice each, to the same result.
TEST(RadixSortPairs, SortsLargeInputs) {
    REQUIRE_CUDA_DEVICE();
    for (const uint32_t tilePairs : {0U, 512U}) {
        expectSorted<uint32_t>(SortCase{3000001, 21, tilePairs, {}}, 31);
        expectSorted<uint32_t>(SortCase{3000001, 21, tilePairs, {}}, 31);
        expectSorted<uint32_t>(SortCase{1000003, 32, tilePairs, 777777, DataType::UINT64, Keys::Skewed}, 32);
        expectSorted<uint16_t>(SortCase{2000000, 16, tilePairs, {}, DataType::UINT32, Keys::FewValues}, 33);
        expectSorted<uint64_t>(SortCase{700001, 45, tilePairs, {}}, 34);
    }
}
