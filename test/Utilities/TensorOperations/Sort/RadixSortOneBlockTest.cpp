// Thor's one-block radix sort of (key, value) pairs against std::stable_sort: every key type, digit layouts from 1 to
// 64 key bits, capacities around a warp's 128 pairs and up to a block's most, and item counts read on the device.
// The pairs past the item count are checked: nothing there may change.

#include "Utilities/TensorOperations/Sort/RadixSortOneBlock.h"

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

enum class Keys { Random, FewValues, Ascending, Skewed, Descending };

struct OneBlockCase {
    uint64_t capacity;
    uint32_t keyBits;
    std::optional<uint64_t> active;  // the device item count; none: the capacity, no device count
    DataType activeDataType = DataType::UINT32;
    Keys keys = Keys::Random;
};

template <typename KeyT>
void expectSorted(const OneBlockCase& c, uint64_t seed) {
    SCOPED_TRACE("capacity " + std::to_string(c.capacity) + " bits " + std::to_string(c.keyBits) + " active " +
                 (c.active ? std::to_string(*c.active) : std::string("none")) + " keys " +
                 std::to_string(static_cast<int>(c.keys)));
    Stream stream(0);
    const RadixSortOneBlockPlan plan = planRadixSortPairsOneBlock(keyDataTypeOf<KeyT>(), c.capacity, c.keyBits);
    const uint64_t n = c.active ? std::min(*c.active, c.capacity) : c.capacity;

    const TensorDescriptor keysDescriptor(keyDataTypeOf<KeyT>(), {c.capacity});
    const TensorDescriptor valuesDescriptor(DataType::UINT32, {c.capacity});
    Tensor keysCpu(cpu, keysDescriptor), valuesCpu(cpu, valuesDescriptor);
    std::mt19937_64 rng(seed);
    const uint64_t mask = c.keyBits == 64 ? ~uint64_t{0} : (uint64_t{1} << c.keyBits) - 1;
    for (uint64_t i = 0; i < c.capacity; ++i) {
        uint64_t key = 0;
        switch (c.keys) {
            case Keys::Random: key = rng(); break;
            case Keys::FewValues: key = rng() % 3; break;
            case Keys::Ascending: key = i * 7; break;
            case Keys::Skewed: key = rng() % 5 == 0 ? mask : rng() % 17; break;
            case Keys::Descending: key = (c.capacity - i) * 3; break;
        }
        keysCpu.getMemPtr<KeyT>()[i] = static_cast<KeyT>(key & mask);
        valuesCpu.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(rng());
    }

    Tensor keys(gpu, keysDescriptor), values(gpu, valuesDescriptor);
    keys.copyFromAsync(keysCpu, stream);
    values.copyFromAsync(valuesCpu, stream);

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

    // Twice: the second launch sorts sorted pairs, which must stay as they are.
    for (int launch = 0; launch < 2; ++launch) {
        launchRadixSortPairsOneBlock(
            plan, keys.getMemPtr(), values.getMemPtr<uint32_t>(), activePointer, c.activeDataType, stream);
    }
    Tensor sortedKeysCpu(cpu, keysDescriptor), sortedValuesCpu(cpu, valuesDescriptor);
    sortedKeysCpu.copyFromAsync(keys, stream);
    sortedValuesCpu.copyFromAsync(values, stream);
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

    uint64_t touched = 0;
    for (uint64_t i = n; i < c.capacity; ++i) {
        if (sortedKeysCpu.getMemPtr<KeyT>()[i] != inputKeys[i] ||
            sortedValuesCpu.getMemPtr<uint32_t>()[i] != inputValues[i]) {
            ++touched;
        }
    }
    EXPECT_EQ(touched, 0U) << "pairs past the item count were written";
}

template <typename KeyT>
void expectSortedForKeyType(const std::vector<uint32_t>& keyBits, uint64_t& seed) {
    const uint64_t most = radixSortOneBlockMaxPairs(keyDataTypeOf<KeyT>());
    for (const uint32_t bits : keyBits) {
        for (const uint64_t capacity : {uint64_t{1},    uint64_t{2},    uint64_t{31},   uint64_t{32},  uint64_t{33},
                                        uint64_t{127},  uint64_t{128},  uint64_t{129},  uint64_t{500}, uint64_t{1024},
                                        uint64_t{2047}, uint64_t{2048}, uint64_t{2049}, uint64_t{3000}, uint64_t{4095},
                                        uint64_t{4096}}) {
            if (capacity > most) continue;
            for (const Keys keys : {Keys::Random, Keys::FewValues, Keys::Ascending, Keys::Skewed, Keys::Descending}) {
                expectSorted<KeyT>(OneBlockCase{capacity, bits, std::nullopt, DataType::UINT32, keys}, ++seed);
            }
        }
    }
}

}  // namespace

TEST(RadixSortPairsOneBlock, PlansDigitsAndBlock) {
    EXPECT_EQ(radixSortOneBlockMaxPairs(DataType::UINT16), 4096U);
    EXPECT_EQ(radixSortOneBlockMaxPairs(DataType::UINT32), 4096U);
    EXPECT_EQ(radixSortOneBlockMaxPairs(DataType::UINT64), 2048U);
    EXPECT_THROW((void)radixSortOneBlockMaxPairs(DataType::INT32), std::invalid_argument);

    const RadixSortOneBlockPlan plan = planRadixSortPairsOneBlock(DataType::UINT32, 4096, 21);
    EXPECT_EQ(plan.capacity, 4096U);
    EXPECT_EQ(plan.passes, 3U);
    EXPECT_EQ(plan.digitBits[0], 7U);
    EXPECT_EQ(plan.digitBits[1], 7U);
    EXPECT_EQ(plan.digitBits[2], 7U);
    EXPECT_EQ(plan.digitShift[0], 0U);
    EXPECT_EQ(plan.digitShift[1], 7U);
    EXPECT_EQ(plan.digitShift[2], 14U);
    EXPECT_EQ(plan.warps, 32U);

    const RadixSortOneBlockPlan uneven = planRadixSortPairsOneBlock(DataType::UINT32, 1000, 17);
    EXPECT_EQ(uneven.passes, 3U);
    EXPECT_EQ(uneven.digitBits[0], 6U);
    EXPECT_EQ(uneven.digitBits[1], 6U);
    EXPECT_EQ(uneven.digitBits[2], 5U);
    EXPECT_EQ(uneven.digitShift[2], 12U);
    EXPECT_EQ(uneven.warps, 8U);

    EXPECT_EQ(planRadixSortPairsOneBlock(DataType::UINT16, 1, 16).warps, 1U);
    EXPECT_EQ(planRadixSortPairsOneBlock(DataType::UINT16, 128, 16).warps, 1U);
    EXPECT_EQ(planRadixSortPairsOneBlock(DataType::UINT16, 129, 16).warps, 2U);
    EXPECT_EQ(planRadixSortPairsOneBlock(DataType::UINT64, 2048, 64).warps, 16U);
    EXPECT_EQ(planRadixSortPairsOneBlock(DataType::UINT64, 2048, 64).passes, 8U);

    EXPECT_THROW((void)planRadixSortPairsOneBlock(DataType::INT32, 1000, 8), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairsOneBlock(DataType::UINT32, 0, 8), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairsOneBlock(DataType::UINT32, 4097, 8), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairsOneBlock(DataType::UINT64, 2049, 8), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairsOneBlock(DataType::UINT16, 1000, 17), std::invalid_argument);
    EXPECT_THROW((void)planRadixSortPairsOneBlock(DataType::UINT32, 1000, 0), std::invalid_argument);
}

TEST(RadixSortPairsOneBlock, RejectsMalformedLaunches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const RadixSortOneBlockPlan plan = planRadixSortPairsOneBlock(DataType::UINT32, 64, 8);
    Tensor keys(gpu, TensorDescriptor(DataType::UINT32, {64})), values(gpu, TensorDescriptor(DataType::UINT32, {64}));
    Tensor count(gpu, TensorDescriptor(DataType::UINT32, {1}));
    auto launch = [&](const RadixSortOneBlockPlan& p, void* k, uint32_t* v, const void* active, DataType activeType) {
        launchRadixSortPairsOneBlock(p, k, v, active, activeType, stream);
    };
    uint32_t* const valuesPointer = values.getMemPtr<uint32_t>();

    EXPECT_THROW(launch(plan, nullptr, valuesPointer, nullptr, DataType::UINT32), std::invalid_argument);
    EXPECT_THROW(launch(plan, keys.getMemPtr(), nullptr, nullptr, DataType::UINT32), std::invalid_argument);
    EXPECT_THROW(launch(plan, keys.getMemPtr(), valuesPointer, count.getMemPtr(), DataType::UINT16),
                 std::invalid_argument);

    RadixSortOneBlockPlan edited = plan;
    edited.warps = 2;
    EXPECT_THROW(launch(edited, keys.getMemPtr(), valuesPointer, nullptr, DataType::UINT32), std::invalid_argument);
    edited = plan;
    edited.capacity = 5000;
    EXPECT_THROW(launch(edited, keys.getMemPtr(), valuesPointer, nullptr, DataType::UINT32), std::invalid_argument);
    edited = plan;
    edited.digitBits[0] = 4;
    EXPECT_THROW(launch(edited, keys.getMemPtr(), valuesPointer, nullptr, DataType::UINT32), std::invalid_argument);
    EXPECT_THROW(launch(RadixSortOneBlockPlan{}, keys.getMemPtr(), valuesPointer, nullptr, DataType::UINT32),
                 std::invalid_argument);
    stream.synchronize();
}

TEST(RadixSortPairsOneBlock, SortsEveryKeyTypeDigitLayoutAndCapacity) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 1000;
    expectSortedForKeyType<uint16_t>({1, 9, 12, 16}, seed);
    expectSortedForKeyType<uint32_t>({3, 8, 17, 21, 32}, seed);
    expectSortedForKeyType<uint64_t>({21, 40, 64}, seed);
}

TEST(RadixSortPairsOneBlock, SortsOnlyTheDeviceItemCount) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 5000;
    for (const DataType activeType : {DataType::UINT32, DataType::UINT64}) {
        for (const uint64_t capacity : {uint64_t{1}, uint64_t{129}, uint64_t{1000}, uint64_t{2048}, uint64_t{4096}}) {
            for (const uint64_t active : {uint64_t{0},
                                          uint64_t{1},
                                          capacity / 3 + 7,
                                          capacity / 2,
                                          capacity - 1,
                                          capacity,
                                          capacity + 1000}) {
                expectSorted<uint32_t>(OneBlockCase{capacity, 21, active, activeType, Keys::Random}, ++seed);
                expectSorted<uint16_t>(OneBlockCase{capacity, 12, active, activeType, Keys::Skewed}, ++seed);
                if (capacity <= radixSortOneBlockMaxPairs(DataType::UINT64)) {
                    expectSorted<uint64_t>(OneBlockCase{capacity, 40, active, activeType, Keys::FewValues}, ++seed);
                }
            }
        }
    }
}
