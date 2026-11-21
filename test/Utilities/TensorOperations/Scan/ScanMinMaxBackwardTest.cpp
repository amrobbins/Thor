// ScanMinMaxBackward against the backward's definition: each element's dy goes to the winner of its prefix, so dx[w]
// is the sum of dy over the elements whose winner is w. dy holds multiples of 1/4 with small sums, so every sum is
// exact in FP32 in any order, and dx is compared byte for byte with the FP32 sum converted by the host to dx's dtype.
//
// The large cases are 20M elements: ragged rows that end before the capacity, an empty row, a row of five starting
// with a NaN, long rows with runs over many supergroups, and dense rows of 5,000,001 (not dividing the tile). They run
// every op, mode and direction in FP32 twice on one workspace, and every combination of dtypes for x, dy and dx.

#include "Utilities/TensorOperations/Scan/ScanMinMaxBackward.h"

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int cuda_device_count_for_test = 0;                                                                            \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                      \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                  \
            GTEST_SKIP() << "CUDA device is required for scan min/max backward tests.";                                \
        }                                                                                                              \
    } while (false)

const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);

constexpr DataType kFloats[5] = {DataType::FP32, DataType::FP16, DataType::BF16, DataType::FP8_E4M3, DataType::FP8_E5M2};

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

// An FP32 value as one element of dtype, by the host's conversions: to nearest, FP8 saturating to its finite range.
void encodeFloat(DataType dtype, float value, uint8_t* at) {
    if (dtype == DataType::FP32) {
        std::memcpy(at, &value, 4);
    } else if (dtype == DataType::FP16) {
        const __half h = __float2half_rn(value);
        std::memcpy(at, &h, 2);
    } else if (dtype == DataType::BF16) {
        const __nv_bfloat16 b = __float2bfloat16_rn(value);
        std::memcpy(at, &b, 2);
    } else {
        *at = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, dtype == DataType::FP8_E5M2 ? __NV_E5M2 : __NV_E4M3);
    }
}

float storedValue(DataType dtype, const uint8_t* at) {
    if (dtype == DataType::FP32) {
        float f;
        std::memcpy(&f, at, 4);
        return f;
    }
    if (dtype == DataType::FP16) {
        __half h;
        std::memcpy(&h, at, 2);
        return __half2float(h);
    }
    if (dtype == DataType::BF16) {
        __nv_bfloat16 b;
        std::memcpy(&b, at, 2);
        return __bfloat162float(b);
    }
    if (dtype == DataType::FP8_E4M3) {
        __nv_fp8_e4m3 q;
        std::memcpy(&q, at, 1);
        return static_cast<float>(q);
    }
    __nv_fp8_e5m2 q;
    std::memcpy(&q, at, 1);
    return static_cast<float>(q);
}

// The backward by its definition over rows [bounds[r], bounds[r + 1]), walked in the scan's direction. The winner is
// replaced only by a strictly better element, so the first of equal elements keeps it; a NaN is below every number,
// so it never replaces a winner and any number replaces it. An exclusive scan's first element has no winner.
std::vector<float> referenceBackward(const std::vector<float>& x,
                                     const std::vector<float>& dy,
                                     const std::vector<uint64_t>& bounds,
                                     bool min,
                                     bool exclusive,
                                     bool reverse) {
    std::vector<float> dx(x.size(), 0.0f);
    for (uint64_t r = 0; r + 1 < bounds.size(); ++r) {
        const uint64_t begin = bounds[r];
        const uint64_t end = bounds[r + 1];
        bool has_winner = false;
        uint64_t winner = 0;
        for (uint64_t step = 0; step < end - begin; ++step) {
            const uint64_t i = reverse ? end - 1 - step : begin + step;
            if (exclusive && has_winner) dx[winner] += dy[i];
            const bool replaces =
                !std::isnan(x[i]) && (std::isnan(x[winner]) || (min ? x[i] < x[winner] : x[i] > x[winner]));
            if (!has_winner || replaces) {
                winner = i;
                has_winner = true;
            }
            if (!exclusive) dx[winner] += dy[i];
        }
    }
    return dx;
}

struct Mismatch {
    uint64_t wrong = 0;
    uint64_t first = 0;
};

// Elements [0, active) of `got` (dtype) against the FP32 sums `want` in dtype, byte for byte, and elements past them
// against `untouched`. An element that never wins is +0: all zero bytes in every dtype.
Mismatch compareBackward(DataType dtype, const std::vector<uint8_t>& got, const std::vector<float>& want, uint64_t active,
                         uint8_t untouched) {
    const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    Mismatch mismatch;
    for (uint64_t i = 0; i < active; ++i) {
        uint8_t expected[4] = {};
        uint32_t want_bits;
        std::memcpy(&want_bits, &want[i], 4);
        if (want_bits != 0U) encodeFloat(dtype, want[i], expected);
        if (std::memcmp(got.data() + i * element_bytes, expected, element_bytes) != 0 && mismatch.wrong++ == 0) {
            mismatch.first = i;
        }
    }
    for (uint64_t at = active * element_bytes; at < got.size(); ++at) {
        if (got[at] != untouched && mismatch.wrong++ == 0) mismatch.first = at / element_bytes;
    }
    return mismatch;
}

// On a Thor stream: its streams do not block, so the default stream's copies would not be ordered with them.
Tensor upload(DataType dtype, const std::vector<uint8_t>& bytes, Stream& stream) {
    const uint64_t element_bytes = static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dtype));
    Tensor tensor(gpu, TensorDescriptor(dtype, {bytes.size() / element_bytes}));
    EXPECT_EQ(cudaMemcpyAsync(tensor.getMemPtr<void>(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice, stream.getStream()),
              cudaSuccess);
    stream.synchronize();
    return tensor;
}

std::vector<uint8_t> download(const Tensor& tensor, Stream& stream) {
    std::vector<uint8_t> bytes(tensor.getArraySizeInBytes());
    EXPECT_EQ(cudaMemcpyAsync(bytes.data(), tensor.getMemPtr<void>(), bytes.size(), cudaMemcpyDeviceToHost, stream.getStream()),
              cudaSuccess);
    stream.synchronize();
    return bytes;
}

// The large inputs, shared by the tests that run on them. x and dy are drawn from small tables, so each dtype's
// tensor is a lookup: x from -4 to 4, -0, +0, NaN, 1000 and -1000 (E4M3 holds the last two as +-448, E5M2 as +-1024:
// still the row's extremes); dy the multiples of 1/4 in [-4, 4] (E5M2 rounds those above 2 to its step of 1/2).
struct LargeCase {
    static constexpr uint64_t n = 20000004;
    static constexpr uint64_t dense_cols = 5000001;
    static constexpr uint64_t end = n - 777;  // where the ragged rows end
    static constexpr uint32_t kNegativeZero = 9, kZero = 10, kNan = 11, kHigh = 12, kLow = 13, kXValues = 14;
    static constexpr uint32_t kDyValues = 33;

    std::vector<uint8_t> x_index = std::vector<uint8_t>(n);
    std::vector<uint8_t> dy_index = std::vector<uint8_t>(n);
    float x_table[5][kXValues];
    float dy_table[5][kDyValues];
    std::vector<Tensor> x;   // per dtype
    std::vector<Tensor> dy;  // per dtype
    std::vector<Tensor> dx;  // per dtype
    Tensor offsets32;
    Tensor offsets64;
    std::vector<uint64_t> ragged_bounds{0};
    std::vector<uint64_t> dense_bounds;

    LargeCase() {
        Stream stream(0);
        // Elements 0 to 2 are a row of their own (before offsets[0]); then a row of five, an empty row, a row of
        // 1.2M, one of 13M, one of one element, and one to the end; past it, 777 elements in no row.
        const std::vector<uint64_t> starts{3, 8, 8, 1200000, 14200000, 14200001, end};
        uint64_t state = 0x1497ULL;
        for (uint64_t i = 0; i < n; ++i) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            x_index[i] = (state >> 20) % 61 == 0 ? kNan : static_cast<uint8_t>(state % 9);
            if ((state >> 30) % 97 == 0) x_index[i] = (state & 1U) ? kNegativeZero : kZero;
            dy_index[i] = static_cast<uint8_t>((state >> 40) % 33ULL);
        }
        // Each long row's first two and last two elements: a max then a min at each end, so in either direction the
        // first element met wins the max scan for the whole row and the second the min scan: runs over supergroups.
        auto plant = [&](uint64_t begin, uint64_t stop) {
            x_index[begin] = kHigh;
            x_index[begin + 1] = kLow;
            x_index[stop - 1] = kHigh;
            x_index[stop - 2] = kLow;
        };
        plant(1200000, 14200000);
        for (uint64_t r = 1; r * dense_cols < n; ++r) plant(r * dense_cols, std::min(n, (r + 1) * dense_cols));
        x_index[3] = kNan;  // the row [3, 8) starts with a NaN in either direction: the first number met replaces it
        x_index[7] = kNan;

        float x_base[kXValues];
        for (uint32_t k = 0; k < 9; ++k) x_base[k] = static_cast<float>(static_cast<int>(k) - 4);
        x_base[kNegativeZero] = -0.0f;
        x_base[kZero] = 0.0f;
        x_base[kNan] = std::nanf("");
        x_base[kHigh] = 1000.0f;
        x_base[kLow] = -1000.0f;
        float dy_base[kDyValues];
        for (uint32_t k = 0; k < kDyValues; ++k) dy_base[k] = (static_cast<float>(k) - 16.0f) * 0.25f;

        for (uint32_t f = 0; f < 5; ++f) {
            const DataType dtype = kFloats[f];
            const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
            auto make = [&](const std::vector<uint8_t>& index, const float* base, uint32_t count, float* table) {
                std::vector<uint8_t> coded(count * element_bytes);
                for (uint32_t k = 0; k < count; ++k) {
                    encodeFloat(dtype, base[k], coded.data() + k * element_bytes);
                    table[k] = storedValue(dtype, coded.data() + k * element_bytes);
                }
                std::vector<uint8_t> bytes(n * element_bytes);
                for (uint64_t i = 0; i < n; ++i) {
                    std::memcpy(bytes.data() + i * element_bytes, coded.data() + index[i] * element_bytes, element_bytes);
                }
                return upload(dtype, bytes, stream);
            };
            x.push_back(make(x_index, x_base, kXValues, x_table[f]));
            dy.push_back(make(dy_index, dy_base, kDyValues, dy_table[f]));
            dx.emplace_back(gpu, TensorDescriptor(dtype, {n}));
        }

        std::vector<uint8_t> bytes32(starts.size() * 4);
        std::vector<uint8_t> bytes64(starts.size() * 8);
        for (uint64_t r = 0; r < starts.size(); ++r) {
            const uint32_t start32 = static_cast<uint32_t>(starts[r]);
            std::memcpy(bytes32.data() + r * 4, &start32, 4);
            std::memcpy(bytes64.data() + r * 8, &starts[r], 8);
            if (starts[r] > ragged_bounds.back()) ragged_bounds.push_back(starts[r]);
        }
        offsets32 = upload(DataType::UINT32, bytes32, stream);
        offsets64 = upload(DataType::UINT64, bytes64, stream);
        for (uint64_t at = 0; at < n; at += dense_cols) dense_bounds.push_back(at);
        dense_bounds.push_back(n);
    }

    // Bit 0: min; bit 1: exclusive; bit 2: reverse.
    std::vector<float> reference(uint32_t fx, uint32_t fdy, uint32_t combination, bool ragged) const {
        std::vector<float> x_values(n);
        std::vector<float> dy_values(n);
        for (uint64_t i = 0; i < n; ++i) {
            x_values[i] = x_table[fx][x_index[i]];
            dy_values[i] = dy_table[fdy][dy_index[i]];
        }
        return referenceBackward(x_values,
                                 dy_values,
                                 ragged ? ragged_bounds : dense_bounds,
                                 (combination & 1U) != 0,
                                 (combination & 2U) != 0,
                                 (combination & 4U) != 0);
    }

    // rows: 0 ragged UINT32, 1 ragged UINT64, 2 dense. Each launch on one backward (one workspace) is checked.
    void check(uint32_t fx, uint32_t fdy, uint32_t fdx, uint32_t combination, uint32_t rows, const std::vector<float>& want,
               uint32_t launches) {
        const bool min = (combination & 1U) != 0;
        const SegmentedScanMode mode = (combination & 2U) != 0 ? SegmentedScanMode::Exclusive : SegmentedScanMode::Inclusive;
        const SegmentedScanDirection direction =
            (combination & 4U) != 0 ? SegmentedScanDirection::Reverse : SegmentedScanDirection::Forward;
        const std::string label = "x " + dtypeName(kFloats[fx]) + ", dy " + dtypeName(kFloats[fdy]) + ", dx " +
                                  dtypeName(kFloats[fdx]) + "; " + (min ? "min" : "max") +
                                  ((combination & 2U) != 0 ? " exclusive" : " inclusive") +
                                  ((combination & 4U) != 0 ? " reverse" : " forward") +
                                  (rows == 2 ? " dense" : rows == 1 ? " ragged UINT64" : " ragged UINT32");
        SCOPED_TRACE(label);
        Tensor& out = dx[fdx];
        const std::unique_ptr<ScanMinMaxBackward> backward =
            rows == 2 ? std::make_unique<ScanMinMaxBackward>(min, mode, direction, x[fx], dy[fdy], out, dense_cols)
                      : std::make_unique<ScanMinMaxBackward>(
                            min, mode, direction, x[fx], dy[fdy], out, rows == 1 ? offsets64 : offsets32);
        Stream stream(0);
        for (uint32_t launch = 0; launch < launches; ++launch) {
            out.memsetAsync(stream, static_cast<int8_t>(0xA5));
            backward->runOn(stream);
            stream.synchronize();
            const std::vector<uint8_t> got = download(out, stream);
            const Mismatch mismatch = compareBackward(kFloats[fdx], got, want, rows == 2 ? n : end, 0xA5);
            if (mismatch.wrong != 0) {
                const uint64_t bytes = TensorDescriptor::getElementSizeInBytes(kFloats[fdx]);
                ADD_FAILURE() << "launch " << launch + 1 << ": " << mismatch.wrong << " wrong, the first at element "
                              << mismatch.first << " (got " << storedValue(kFloats[fdx], got.data() + mismatch.first * bytes)
                              << ", the FP32 sum is " << (mismatch.first < want.size() ? want[mismatch.first] : 0.0f) << ")";
            }
        }
    }
};

class ScanMinMaxBackwardLarge : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0) large = std::make_unique<LargeCase>();
    }
    static void TearDownTestSuite() { large.reset(); }
    void SetUp() override {
        if (!large) GTEST_SKIP() << "CUDA device is required for scan min/max backward tests.";
    }
    static std::unique_ptr<LargeCase> large;
};

std::unique_ptr<LargeCase> ScanMinMaxBackwardLarge::large;

}  // namespace

// Every op, mode and direction in FP32 on ragged rows (UINT32 and UINT64 offsets) and dense rows, two launches on one
// workspace each: the second starts from the words the first left.
TEST_F(ScanMinMaxBackwardLarge, Fp32EveryOpModeAndDirectionTwiceOnOneWorkspace) {
    for (uint32_t combination = 0; combination < 8; ++combination) {
        const std::vector<float> ragged = large->reference(0, 0, combination, true);
        large->check(0, 0, 0, combination, 0, ragged, 2);
        large->check(0, 0, 0, combination, 1, ragged, 2);
        large->check(0, 0, 0, combination, 2, large->reference(0, 0, combination, false), 2);
    }
}

// Each of the 25 pairs of dtypes for x and dy on its own kind of launch (7 and 24 share no factor, so the pairs take
// 24 of the 24 kinds), with dx in each of the 5 dtypes.
TEST_F(ScanMinMaxBackwardLarge, EveryCombinationOfDtypes) {
    for (uint32_t fx = 0; fx < 5; ++fx) {
        for (uint32_t fdy = 0; fdy < 5; ++fdy) {
            const uint32_t kind = (fx * 5 + fdy) * 7 % 24;
            const uint32_t combination = kind % 8;
            const uint32_t rows = kind / 8;
            const std::vector<float> want = large->reference(fx, fdy, combination, rows != 2);
            for (uint32_t fdx = 0; fdx < 5; ++fdx) large->check(fx, fdy, fdx, combination, rows, want, 1);
        }
    }
}

// Short dense rows: rows dividing the tile need no workspace; rows of 5 and 3,000 share and span tiles.
TEST(ScanMinMaxBackward, ShortDenseRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const uint64_t cols : {1ULL, 5ULL, 256ULL, 2048ULL, 3000ULL}) {
        const uint64_t n = cols * 12;
        std::vector<float> x(n);
        std::vector<float> dy(n);
        for (uint64_t i = 0; i < n; ++i) {
            x[i] = static_cast<float>((i * 7919) % 13) - 6.0f;
            dy[i] = static_cast<float>(static_cast<int>((i * 104729) % 33) - 16) * 0.25f;
        }
        std::vector<uint8_t> x_bytes(n * 4);
        std::vector<uint8_t> dy_bytes(n * 4);
        std::memcpy(x_bytes.data(), x.data(), n * 4);
        std::memcpy(dy_bytes.data(), dy.data(), n * 4);
        const Tensor gx = upload(DataType::FP32, x_bytes, stream);
        const Tensor gdy = upload(DataType::FP32, dy_bytes, stream);
        Tensor gdx(gpu, TensorDescriptor(DataType::FP32, {n}));
        std::vector<uint64_t> bounds;
        for (uint64_t at = 0; at <= n; at += cols) bounds.push_back(at);
        for (uint32_t combination = 0; combination < 8; ++combination) {
            const bool min = (combination & 1U) != 0;
            const bool exclusive = (combination & 2U) != 0;
            const bool reverse = (combination & 4U) != 0;
            SCOPED_TRACE("rows of " + std::to_string(cols) + ", combination " + std::to_string(combination));
            const ScanMinMaxBackward backward(min,
                                              exclusive ? SegmentedScanMode::Exclusive : SegmentedScanMode::Inclusive,
                                              reverse ? SegmentedScanDirection::Reverse : SegmentedScanDirection::Forward,
                                              gx, gdy, gdx, cols);
            EXPECT_EQ(backward.getWorkspaceSizeInBytes() == 0, backward.getTileElements() % cols == 0);
            backward.runOn(stream);
            stream.synchronize();
            const Mismatch mismatch =
                compareBackward(DataType::FP32, download(gdx, stream), referenceBackward(x, dy, bounds, min, exclusive, reverse), n, 0);
            EXPECT_EQ(mismatch.wrong, 0U) << "the first wrong element is " << mismatch.first;
        }
    }
}

TEST(ScanMinMaxBackward, RefusesWhatItDoesNotTake) {
    REQUIRE_CUDA_DEVICE();
    const Tensor x(gpu, TensorDescriptor(DataType::FP32, {12}));
    const Tensor dy(gpu, TensorDescriptor(DataType::FP16, {12}));
    const Tensor dx(gpu, TensorDescriptor(DataType::BF16, {12}));
    const Tensor ints(gpu, TensorDescriptor(DataType::UINT32, {12}));
    const Tensor short_dx(gpu, TensorDescriptor(DataType::FP32, {11}));
    const Tensor offsets(gpu, TensorDescriptor(DataType::UINT32, {3}));
    const Tensor float_offsets(gpu, TensorDescriptor(DataType::FP32, {3}));
    const auto inclusive = SegmentedScanMode::Inclusive;
    const auto forward = SegmentedScanDirection::Forward;

    EXPECT_TRUE(ScanMinMaxBackward::isSupported(DataType::FP8_E5M2, DataType::BF16, DataType::FP8_E4M3));
    EXPECT_FALSE(ScanMinMaxBackward::isSupported(DataType::UINT32, DataType::FP32, DataType::FP32));
    EXPECT_FALSE(ScanMinMaxBackward::isSupported(DataType::FP32, DataType::FP32, DataType::UINT32));
    EXPECT_NO_THROW(ScanMinMaxBackward(true, inclusive, forward, x, dy, dx, 4));
    EXPECT_NO_THROW(ScanMinMaxBackward(true, inclusive, forward, x, dy, dx, offsets));
    EXPECT_THROW(ScanMinMaxBackward(true, inclusive, forward, ints, dy, dx, 4), std::invalid_argument);
    EXPECT_THROW(ScanMinMaxBackward(true, inclusive, forward, x, ints, dx, 4), std::invalid_argument);
    EXPECT_THROW(ScanMinMaxBackward(true, inclusive, forward, x, dy, short_dx, 4), std::invalid_argument);
    EXPECT_THROW(ScanMinMaxBackward(true, inclusive, forward, x, dy, dx, 5), std::invalid_argument);
    EXPECT_THROW(ScanMinMaxBackward(true, inclusive, forward, x, dy, dx, float_offsets), std::invalid_argument);
}
