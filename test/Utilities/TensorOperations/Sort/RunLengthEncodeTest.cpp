// Thor's run-length encode against a host encode: every key type, runs from single keys to runs across many tiles,
// counts around a tile's edges (2,048 keys) and item counts read on the device, 0 included. The outputs are checked
// past the number of runs: nothing there may change.

#include "Utilities/TensorOperations/Sort/RunLengthEncode.h"

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
            GTEST_SKIP() << "CUDA device is required for run-length encode tests.";               \
        }                                                                                         \
    } while (false)

const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
constexpr uint64_t kTile = 2048;

template <typename KeyT>
constexpr DataType keyDataTypeOf() {
    return sizeof(KeyT) == 2 ? DataType::UINT16 : sizeof(KeyT) == 4 ? DataType::UINT32 : DataType::UINT64;
}

enum class Runs { Singles, OneRun, Short, Long, Mixed };

struct EncodeCase {
    uint64_t capacity;
    std::optional<uint64_t> active;  // the device item count; none: the capacity, no device count
    DataType activeDataType = DataType::UINT32;
    Runs runs = Runs::Short;
    bool scratchAlreadyClear = false;  // the caller zeroes the scratch and the launch is told not to
};

template <typename KeyT>
void expectEncoded(const EncodeCase& c, uint64_t seed) {
    SCOPED_TRACE("capacity " + std::to_string(c.capacity) + " active " +
                 (c.active ? std::to_string(*c.active) : std::string("none")) + " runs " +
                 std::to_string(static_cast<int>(c.runs)));
    Stream stream(0);
    const RunLengthEncodePlan plan = planRunLengthEncode(keyDataTypeOf<KeyT>(), c.capacity);
    const uint64_t n = c.active ? std::min(*c.active, c.capacity) : c.capacity;

    const TensorDescriptor keysDescriptor(keyDataTypeOf<KeyT>(), {c.capacity});
    const TensorDescriptor wordsDescriptor(DataType::UINT32, {c.capacity});
    Tensor keysCpu(cpu, keysDescriptor), staleKeysCpu(cpu, keysDescriptor), staleWordsCpu(cpu, wordsDescriptor);
    std::mt19937_64 rng(seed);
    const KeyT staleKey = static_cast<KeyT>(0xA5A5A5A5A5A5A5A5ULL);
    KeyT key = static_cast<KeyT>(rng());
    for (uint64_t i = 0; i < c.capacity;) {
        uint64_t length = 1;
        switch (c.runs) {
            case Runs::Singles: length = 1; break;
            case Runs::OneRun: length = c.capacity; break;
            case Runs::Short: length = 1 + rng() % 5; break;
            case Runs::Long: length = 1 + rng() % (3 * kTile); break;
            case Runs::Mixed: length = rng() % 4 == 0 ? 1 + rng() % (5 * kTile) : 1; break;
        }
        KeyT next = key;
        while (next == key) next = static_cast<KeyT>(rng());
        key = next;
        for (uint64_t k = 0; k < length && i < c.capacity; ++k, ++i) keysCpu.getMemPtr<KeyT>()[i] = key;
    }
    for (uint64_t i = 0; i < c.capacity; ++i) {
        staleKeysCpu.getMemPtr<KeyT>()[i] = staleKey;
        staleWordsCpu.getMemPtr<uint32_t>()[i] = 0x5A5A5A5AU;
    }

    Tensor keys(gpu, keysDescriptor), uniqueKeys(gpu, keysDescriptor), runCounts(gpu, wordsDescriptor),
        runOffsets(gpu, wordsDescriptor), numRuns(gpu, TensorDescriptor(DataType::UINT32, {1}));
    Tensor scratch(gpu, TensorDescriptor(DataType::UINT64, {plan.scratchWords}));
    keys.copyFromAsync(keysCpu, stream);
    uniqueKeys.copyFromAsync(staleKeysCpu, stream);
    runCounts.copyFromAsync(staleWordsCpu, stream);
    runOffsets.copyFromAsync(staleWordsCpu, stream);
    numRuns.memsetAsync(stream, 0x77);
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

    launchRunLengthEncode(plan,
                          keys.getMemPtr(),
                          uniqueKeys.getMemPtr(),
                          runCounts.getMemPtr<uint32_t>(),
                          runOffsets.getMemPtr<uint32_t>(),
                          numRuns.getMemPtr<uint32_t>(),
                          scratch.getMemPtr<uint64_t>(),
                          activePointer,
                          c.activeDataType,
                          stream,
                          /*clearScratch=*/!c.scratchAlreadyClear);
    Tensor uniqueKeysCpu(cpu, keysDescriptor), runCountsCpu(cpu, wordsDescriptor), runOffsetsCpu(cpu, wordsDescriptor),
        numRunsCpu(cpu, TensorDescriptor(DataType::UINT32, {1}));
    uniqueKeysCpu.copyFromAsync(uniqueKeys, stream);
    runCountsCpu.copyFromAsync(runCounts, stream);
    runOffsetsCpu.copyFromAsync(runOffsets, stream);
    numRunsCpu.copyFromAsync(numRuns, stream);
    stream.synchronize();

    std::vector<KeyT> expectedKeys;
    std::vector<uint32_t> expectedCounts;
    std::vector<uint32_t> expectedOffsets;
    const KeyT* input = keysCpu.getMemPtr<KeyT>();
    for (uint64_t i = 0; i < n; ++i) {
        if (i == 0 || input[i] != input[i - 1]) {
            expectedKeys.push_back(input[i]);
            expectedOffsets.push_back(static_cast<uint32_t>(i));
            expectedCounts.push_back(0U);
        }
        ++expectedCounts.back();
    }
    const uint64_t runs = expectedKeys.size();
    ASSERT_EQ(numRunsCpu.getMemPtr<uint32_t>()[0], runs);
    uint64_t wrong = 0;
    uint64_t firstWrong = 0;
    for (uint64_t r = 0; r < runs; ++r) {
        if (uniqueKeysCpu.getMemPtr<KeyT>()[r] != expectedKeys[r] ||
            runCountsCpu.getMemPtr<uint32_t>()[r] != expectedCounts[r] ||
            runOffsetsCpu.getMemPtr<uint32_t>()[r] != expectedOffsets[r]) {
            if (wrong++ == 0) firstWrong = r;
        }
    }
    EXPECT_EQ(wrong, 0U) << "first wrong run " << firstWrong;
    uint64_t touched = 0;
    for (uint64_t r = runs; r < c.capacity; ++r) {
        if (uniqueKeysCpu.getMemPtr<KeyT>()[r] != staleKey || runCountsCpu.getMemPtr<uint32_t>()[r] != 0x5A5A5A5AU ||
            runOffsetsCpu.getMemPtr<uint32_t>()[r] != 0x5A5A5A5AU) {
            ++touched;
        }
    }
    EXPECT_EQ(touched, 0U) << "outputs past the number of runs were written";
}

template <typename KeyT>
void expectEveryLayout(uint64_t& seed) {
    for (const uint64_t capacity :
         {uint64_t{1}, uint64_t{2}, uint64_t{31}, kTile - 1, kTile, kTile + 1, 3 * kTile + 77, 9 * kTile}) {
        for (const Runs runs : {Runs::Singles, Runs::OneRun, Runs::Short, Runs::Long, Runs::Mixed}) {
            expectEncoded<KeyT>(EncodeCase{capacity, std::nullopt, DataType::UINT32, runs}, seed++);
        }
    }
}

}  // namespace

TEST(RunLengthEncode, PlansTilesAndScratch) {
    const RunLengthEncodePlan plan = planRunLengthEncode(DataType::UINT32, 100000);
    EXPECT_EQ(plan.tiles, 49U);
    EXPECT_EQ(plan.scratchWords, 49U);
    EXPECT_EQ(planRunLengthEncode(DataType::UINT16, 2048).tiles, 1U);
    EXPECT_EQ(planRunLengthEncode(DataType::UINT64, 2049).tiles, 2U);
    EXPECT_THROW((void)planRunLengthEncode(DataType::UINT32, 0), std::invalid_argument);
    EXPECT_THROW((void)planRunLengthEncode(DataType::UINT32, uint64_t{1} << 30), std::invalid_argument);
    EXPECT_THROW((void)planRunLengthEncode(DataType::UINT8, 100), std::invalid_argument);
    EXPECT_THROW((void)planRunLengthEncode(DataType::FP32, 100), std::invalid_argument);
}

TEST(RunLengthEncode, RejectsMalformedLaunches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const RunLengthEncodePlan plan = planRunLengthEncode(DataType::UINT32, 1000);
    Tensor keys(gpu, TensorDescriptor(DataType::UINT32, {1000})),
        unique(gpu, TensorDescriptor(DataType::UINT32, {1000})),
        counts(gpu, TensorDescriptor(DataType::UINT32, {1000})),
        offsets(gpu, TensorDescriptor(DataType::UINT32, {1000})), numRuns(gpu, TensorDescriptor(DataType::UINT32, {1})),
        scratch(gpu, TensorDescriptor(DataType::UINT64, {plan.scratchWords}));
    RunLengthEncodePlan edited = plan;
    edited.tiles += 1;
    EXPECT_THROW(launchRunLengthEncode(edited,
                                       keys.getMemPtr(),
                                       unique.getMemPtr(),
                                       counts.getMemPtr<uint32_t>(),
                                       offsets.getMemPtr<uint32_t>(),
                                       numRuns.getMemPtr<uint32_t>(),
                                       scratch.getMemPtr<uint64_t>(),
                                       nullptr,
                                       DataType::UINT32,
                                       stream),
                 std::invalid_argument);
    EXPECT_THROW(launchRunLengthEncode(plan,
                                       keys.getMemPtr(),
                                       keys.getMemPtr(),
                                       counts.getMemPtr<uint32_t>(),
                                       offsets.getMemPtr<uint32_t>(),
                                       numRuns.getMemPtr<uint32_t>(),
                                       scratch.getMemPtr<uint64_t>(),
                                       nullptr,
                                       DataType::UINT32,
                                       stream),
                 std::invalid_argument);
    EXPECT_THROW(launchRunLengthEncode(plan,
                                       keys.getMemPtr(),
                                       unique.getMemPtr(),
                                       counts.getMemPtr<uint32_t>(),
                                       offsets.getMemPtr<uint32_t>(),
                                       nullptr,
                                       scratch.getMemPtr<uint64_t>(),
                                       nullptr,
                                       DataType::UINT32,
                                       stream),
                 std::invalid_argument);
    EXPECT_THROW(launchRunLengthEncode(plan,
                                       keys.getMemPtr(),
                                       unique.getMemPtr(),
                                       counts.getMemPtr<uint32_t>(),
                                       offsets.getMemPtr<uint32_t>(),
                                       numRuns.getMemPtr<uint32_t>(),
                                       scratch.getMemPtr<uint64_t>(),
                                       numRuns.getMemPtr(),
                                       DataType::FP32,
                                       stream),
                 std::invalid_argument);
}

TEST(RunLengthEncode, EncodesEveryKeyTypeAndRunLayout) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 1;
    expectEveryLayout<uint16_t>(seed);
    expectEveryLayout<uint32_t>(seed);
    expectEveryLayout<uint64_t>(seed);
}

TEST(RunLengthEncode, EncodesOnlyTheDeviceItemCount) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 1000;
    for (const DataType countType : {DataType::UINT32, DataType::UINT64}) {
        for (const uint64_t active :
             {uint64_t{0}, uint64_t{1}, kTile - 1, kTile, kTile + 1, 4 * kTile + 9, 9 * kTile, 20 * kTile}) {
            for (const Runs runs : {Runs::Short, Runs::Long, Runs::Mixed}) {
                expectEncoded<uint32_t>(EncodeCase{9 * kTile + 1, active, countType, runs}, seed++);
                expectEncoded<uint16_t>(EncodeCase{5 * kTile, active, countType, runs}, seed++);
            }
        }
    }
}

// The scratch zeroed by the caller (in a graph, by an earlier kernel) and no memset by the launch.
TEST(RunLengthEncode, EncodesOnScratchTheCallerCleared) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 7000;
    for (const Runs runs : {Runs::Singles, Runs::Short, Runs::Long, Runs::Mixed}) {
        expectEncoded<uint32_t>(EncodeCase{9 * kTile + 1, std::nullopt, DataType::UINT32, runs, true}, seed++);
        expectEncoded<uint16_t>(EncodeCase{5 * kTile, uint64_t{2 * kTile + 3}, DataType::UINT64, runs, true}, seed++);
        expectEncoded<uint64_t>(EncodeCase{3 * kTile + 77, uint64_t{0}, DataType::UINT32, runs, true}, seed++);
    }
    expectEncoded<uint32_t>(EncodeCase{1048576, std::nullopt, DataType::UINT32, Runs::Mixed, true}, seed++);
}

TEST(RunLengthEncode, EncodesLargeInputs) {
    REQUIRE_CUDA_DEVICE();
    uint64_t seed = 5000;
    for (const Runs runs : {Runs::Singles, Runs::Short, Runs::Mixed}) {
        expectEncoded<uint32_t>(EncodeCase{3000001, std::nullopt, DataType::UINT32, runs}, seed++);
        expectEncoded<uint64_t>(EncodeCase{1048576, uint64_t{700001}, DataType::UINT64, runs}, seed++);
    }
}
