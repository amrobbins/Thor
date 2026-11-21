#include "Utilities/TensorOperations/Scan/SegmentedScan.h"

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int cuda_device_count_for_test = 0;                                                                            \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                      \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                  \
            GTEST_SKIP() << "CUDA device is required for segmented scan tests.";                                       \
        }                                                                                                              \
    } while (false)

const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);

const std::vector<DataType> kDTypes{
    DataType::FP32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP8_E4M3, DataType::FP8_E5M2};

std::string dtypeText(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

uint64_t bytesOf(DataType dtype) { return TensorDescriptor::getElementSizeInBytes(dtype); }

// An element of a float dtype as a float, and a float rounded into one the way the kernel writes it (to nearest; FP8
// saturating to its largest finite value).
float loadFloat(DataType dtype, const uint8_t* p) {
    switch (dtype) {
        case DataType::FP32: {
            float f;
            std::memcpy(&f, p, 4);
            return f;
        }
        case DataType::FP16: {
            __half h;
            std::memcpy(&h, p, 2);
            return __half2float(h);
        }
        case DataType::BF16: {
            __nv_bfloat16 b;
            std::memcpy(&b, p, 2);
            return __bfloat162float(b);
        }
        case DataType::FP8_E4M3: {
            __nv_fp8_e4m3 q;
            std::memcpy(&q, p, 1);
            return static_cast<float>(q);
        }
        default: {
            __nv_fp8_e5m2 q;
            std::memcpy(&q, p, 1);
            return static_cast<float>(q);
        }
    }
}

void storeFloat(DataType dtype, float value, uint8_t* p) {
    switch (dtype) {
        case DataType::FP32:
            std::memcpy(p, &value, 4);
            return;
        case DataType::FP16: {
            const __half h = __float2half_rn(value);
            std::memcpy(p, &h, 2);
            return;
        }
        case DataType::BF16: {
            const __nv_bfloat16 b = __float2bfloat16_rn(value);
            std::memcpy(p, &b, 2);
            return;
        }
        case DataType::FP8_E4M3: {
            const __nv_fp8_e4m3 q(value);
            std::memcpy(p, &q, 1);
            return;
        }
        default: {
            const __nv_fp8_e5m2 q(value);
            std::memcpy(p, &q, 1);
            return;
        }
    }
}

// Elements of `dtype`: small integers in [-3, 3] (UINT32: [0, 7]), so every sum in FP32 (or UINT32) is exact in any
// order and a sequential reference gives the kernel's bits. `nan_every`: a NaN about once in that many floats.
std::vector<uint8_t> makeInput(DataType dtype, uint64_t n, uint32_t seed, uint32_t nan_every = 0) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> bytes(n * bytesOf(dtype));
    for (uint64_t i = 0; i < n; ++i) {
        uint8_t* p = bytes.data() + i * bytesOf(dtype);
        if (dtype == DataType::UINT32) {
            const uint32_t u = rng() % 8;
            std::memcpy(p, &u, 4);
        } else {
            const bool nan = nan_every != 0 && rng() % nan_every == 0;
            const float value = static_cast<float>(static_cast<int>(rng() % 7) - 3);
            storeFloat(dtype, nan ? std::numeric_limits<float>::quiet_NaN() : value, p);
        }
    }
    return bytes;
}

// Elements for products: powers of two (0.5, 1 and 2, either sign) and, rarely, a zero (UINT32: odd values and, rarely,
// a 2 or a 0). The running exponent over the whole input stays within [-60, 60], so the product of any run of elements
// is within [2^-120, 2^120] or zero: every FP32 product is exact in any order, and a sequential reference gives the
// kernel's bits (UINT32 products wrap, which is exact in any order too).
std::vector<uint8_t> makeProductInput(DataType dtype, uint64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> bytes(n * bytesOf(dtype));
    int exponent = 0;
    for (uint64_t i = 0; i < n; ++i) {
        uint8_t* p = bytes.data() + i * bytesOf(dtype);
        const uint32_t r = rng();
        if (dtype == DataType::UINT32) {
            const uint32_t u = r % 65536 == 0 ? 0U : r % 1024 == 1 ? 2U : ((r >> 16) % 4) * 2 + 1;
            std::memcpy(p, &u, 4);
            continue;
        }
        float value = 0.0f;
        if (r % 262144 != 0) {
            const uint32_t pick = (r >> 18) % 8;
            int step = pick == 0 ? -1 : pick == 1 ? 1 : 0;
            if (exponent + step > 60 || exponent + step < -60) step = -step;
            exponent += step;
            value = std::ldexp(((r >> 21) & 1U) != 0U ? -1.0f : 1.0f, step);
        }
        storeFloat(dtype, value, p);
    }
    return bytes;
}

// Elements for the arg scans: small integers in [-3, 3] (many ties), -0 for about half the zeros, a NaN about once in
// 16 floats and an infinity of either sign about once in 64 (E4M3 has none: it saturates to 448) (UINT32: [0, 7],
// about one in 64 scaled to the top of the range).
std::vector<uint8_t> makeArgInput(DataType dtype, uint64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> bytes(n * bytesOf(dtype));
    for (uint64_t i = 0; i < n; ++i) {
        uint8_t* p = bytes.data() + i * bytesOf(dtype);
        const uint32_t r = rng();
        if (dtype == DataType::UINT32) {
            const uint32_t u = (r % 8) * ((r >> 12) % 64 == 0 ? 0x1FFFFFFFU : 1U);
            std::memcpy(p, &u, 4);
            continue;
        }
        float value = static_cast<float>(static_cast<int>(r % 7) - 3);
        if (value == 0.0f && ((r >> 8) & 1U) != 0U) value = -0.0f;
        if ((r >> 16) % 16 == 0) value = std::numeric_limits<float>::quiet_NaN();
        if ((r >> 20) % 64 == 1) value = ((r >> 26) & 1U) != 0U ? -INFINITY : INFINITY;
        storeFloat(dtype, value, p);
    }
    return bytes;
}

std::vector<uint8_t> makeInputFor(SegmentedScanOp op, DataType dtype, uint64_t n, uint32_t seed) {
    if (isArg(op)) return makeArgInput(dtype, n, seed);
    return op == SegmentedScanOp::Product ? makeProductInput(dtype, n, seed) : makeInput(dtype, n, seed);
}

template <typename V>
V applyReference(SegmentedScanOp op, V acc, V x) {
    switch (op) {
        case SegmentedScanOp::Sum:
            return acc + x;
        case SegmentedScanOp::Product:
            return acc * x;
        case SegmentedScanOp::Max:
            if constexpr (std::is_same_v<V, float>) return std::fmax(acc, x);
            return acc < x ? x : acc;
        case SegmentedScanOp::Min:
            if constexpr (std::is_same_v<V, float>) return std::fmin(acc, x);
            return x < acc ? x : acc;
        case SegmentedScanOp::ArgMax:
        case SegmentedScanOp::ArgMin:
            break;  // argReference
    }
    return acc;
}

// An arg scan, element by element, by comparisons: a later element wins only if it is strictly larger (argmax) or
// smaller (argmin), a NaN never does, and -0 equals +0 (as float compares say); a run restarts at every head.
std::vector<uint8_t> argReference(SegmentedScanOp op,
                                  DataType in,
                                  const std::vector<uint8_t>& input,
                                  const std::vector<char>& heads) {
    const uint64_t n = heads.size();
    std::vector<uint8_t> result(n * 4);
    const bool max = op == SegmentedScanOp::ArgMax;
    uint64_t best = 0;
    for (uint64_t i = 0; i < n; ++i) {
        if (heads[i]) {
            best = i;
        } else if (in == DataType::UINT32) {
            uint32_t x;
            uint32_t y;
            std::memcpy(&x, input.data() + 4 * i, 4);
            std::memcpy(&y, input.data() + 4 * best, 4);
            if (max ? x > y : x < y) best = i;
        } else {
            const float x = loadFloat(in, input.data() + i * bytesOf(in));
            const float y = loadFloat(in, input.data() + best * bytesOf(in));
            if (!std::isnan(x) && (std::isnan(y) || (max ? x > y : x < y))) best = i;
        }
        const uint32_t index = static_cast<uint32_t>(best);
        std::memcpy(result.data() + 4 * i, &index, 4);
    }
    return result;
}

// A scan's mode and direction.
struct Variant {
    SegmentedScanMode mode;
    SegmentedScanDirection direction;
};

constexpr SegmentedScanMode kInclusive = SegmentedScanMode::Inclusive;
constexpr SegmentedScanMode kExclusive = SegmentedScanMode::Exclusive;
constexpr SegmentedScanDirection kForward = SegmentedScanDirection::Forward;
constexpr SegmentedScanDirection kReverse = SegmentedScanDirection::Reverse;
constexpr Variant kInclusiveForward{kInclusive, kForward};
const std::vector<Variant> kVariants{
    {kInclusive, kForward}, {kExclusive, kForward}, {kInclusive, kReverse}, {kExclusive, kReverse}};

std::string variantText(const Variant& variant) {
    return std::string(variant.mode == kInclusive ? "inclusive" : "exclusive") +
           (variant.direction == kForward ? " forward" : " reverse");
}

// What an exclusive scan writes for a row's first element, in `dtype` (SegmentedScan.h): 0 for a sum, 1 for a product,
// +inf (min) and -inf (max), E4M3's +-448, UINT32's largest value (min) and 0 (max). An arg scan's value there is the
// min's (argmin) or the max's (argmax) in the input's dtype.
std::vector<uint8_t> identityBytes(SegmentedScanOp op, DataType dtype) {
    if (op == SegmentedScanOp::ArgMax) op = SegmentedScanOp::Max;
    if (op == SegmentedScanOp::ArgMin) op = SegmentedScanOp::Min;
    std::vector<uint8_t> bytes(bytesOf(dtype));
    const bool max = op == SegmentedScanOp::Max;
    if (op == SegmentedScanOp::Sum || op == SegmentedScanOp::Product) {
        const uint32_t value = op == SegmentedScanOp::Sum ? 0U : 1U;
        if (dtype == DataType::UINT32) {
            std::memcpy(bytes.data(), &value, 4);
        } else {
            storeFloat(dtype, static_cast<float>(value), bytes.data());
        }
        return bytes;
    }
    switch (dtype) {
        case DataType::UINT32: {
            const uint32_t bits = max ? 0U : 0xFFFFFFFFU;
            std::memcpy(bytes.data(), &bits, 4);
            break;
        }
        case DataType::FP32: {
            const uint32_t bits = max ? 0xFF800000U : 0x7F800000U;
            std::memcpy(bytes.data(), &bits, 4);
            break;
        }
        case DataType::FP16: {
            const uint16_t bits = max ? 0xFC00U : 0x7C00U;
            std::memcpy(bytes.data(), &bits, 2);
            break;
        }
        case DataType::BF16: {
            const uint16_t bits = max ? 0xFF80U : 0x7F80U;
            std::memcpy(bytes.data(), &bits, 2);
            break;
        }
        case DataType::FP8_E4M3:
            bytes[0] = max ? 0xFE : 0x7E;  // -448, 448
            break;
        default:
            bytes[0] = max ? 0xFC : 0x7C;  // E5M2: -inf, +inf
            break;
    }
    return bytes;
}

// An exclusive scan from the inclusive one: each element takes the element before's result, a head the identity (an
// arg scan: the index 0xFFFFFFFF).
std::vector<uint8_t> exclusiveOf(SegmentedScanOp op,
                                 DataType out,
                                 const std::vector<uint8_t>& inclusive,
                                 const std::vector<char>& heads) {
    const uint64_t b = bytesOf(out);
    const std::vector<uint8_t> identity =
        isArg(op) ? std::vector<uint8_t>{0xFF, 0xFF, 0xFF, 0xFF} : identityBytes(op, out);
    std::vector<uint8_t> result(inclusive.size());
    for (uint64_t i = 0; i < heads.size(); ++i) {
        const uint8_t* from = heads[i] ? identity.data() : inclusive.data() + (i - 1) * b;
        std::memcpy(result.data() + i * b, from, b);
    }
    return result;
}

std::vector<uint8_t> reference(SegmentedScanOp op,
                               const Variant& variant,
                               DataType in,
                               DataType out,
                               const std::vector<uint8_t>& input,
                               const std::vector<char>& heads);

// A reverse scan: the forward scan of the elements in reverse order, whose rows start at each row's last element (an
// element ends a row when the next starts one, and the last ends one), put back in order; an arg scan's indices map
// back, so of equal values it has the forward scan's first, the last in the row.
std::vector<uint8_t> reverseReference(SegmentedScanOp op,
                                      SegmentedScanMode mode,
                                      DataType in,
                                      DataType out,
                                      const std::vector<uint8_t>& input,
                                      const std::vector<char>& heads) {
    const uint64_t n = heads.size();
    const uint64_t in_bytes = bytesOf(in);
    const uint64_t out_bytes = isArg(op) ? 4 : bytesOf(out);
    std::vector<uint8_t> flipped(n * in_bytes);
    std::vector<char> ends(n);
    for (uint64_t i = 0; i < n; ++i) {
        std::memcpy(flipped.data() + (n - 1 - i) * in_bytes, input.data() + i * in_bytes, in_bytes);
        ends[n - 1 - i] = i + 1 == n || heads[i + 1];
    }
    const std::vector<uint8_t> scanned = reference(op, Variant{mode, kForward}, in, out, flipped, ends);
    std::vector<uint8_t> result(n * out_bytes);
    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t* from = scanned.data() + (n - 1 - i) * out_bytes;
        if (isArg(op)) {
            uint32_t index;
            std::memcpy(&index, from, 4);
            if (index != 0xFFFFFFFFU) index = static_cast<uint32_t>(n - 1 - index);
            std::memcpy(result.data() + i * 4, &index, 4);
        } else {
            std::memcpy(result.data() + i * out_bytes, from, out_bytes);
        }
    }
    return result;
}

// The scan, element by element: a run restarts at every head (heads[0] is always one).
std::vector<uint8_t> reference(SegmentedScanOp op,
                               const Variant& variant,
                               DataType in,
                               DataType out,
                               const std::vector<uint8_t>& input,
                               const std::vector<char>& heads) {
    if (variant.direction == kReverse) return reverseReference(op, variant.mode, in, out, input, heads);
    if (variant.mode == kExclusive) {
        return exclusiveOf(op, out, reference(op, kInclusiveForward, in, out, input, heads), heads);
    }
    if (isArg(op)) return argReference(op, in, input, heads);
    const uint64_t n = heads.size();
    std::vector<uint8_t> result(n * bytesOf(out));
    if (in == DataType::UINT32) {
        uint32_t acc = 0;
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t x;
            std::memcpy(&x, input.data() + 4 * i, 4);
            acc = heads[i] ? x : applyReference(op, acc, x);
            std::memcpy(result.data() + 4 * i, &acc, 4);
        }
        return result;
    }
    float acc = 0.0f;
    for (uint64_t i = 0; i < n; ++i) {
        const float x = loadFloat(in, input.data() + i * bytesOf(in));
        acc = heads[i] ? x : applyReference(op, acc, x);
        storeFloat(out, acc, result.data() + i * bytesOf(out));
    }
    return result;
}

std::vector<char> denseHeads(uint64_t n, uint64_t row_length) {
    std::vector<char> heads(n, 0);
    for (uint64_t i = 0; i < n; i += row_length) heads[i] = 1;
    return heads;
}

std::vector<char> raggedHeads(uint64_t n, const std::vector<uint64_t>& offsets) {
    std::vector<char> heads(n, 0);
    if (n > 0) heads[0] = 1;
    for (const uint64_t start : offsets) {
        if (start < n) heads[start] = 1;
    }
    return heads;
}

// Offsets with empty rows, short rows and rows longer than a tile; the last offset `tail` elements before n.
std::vector<uint64_t> makeOffsets(uint64_t n, uint64_t tile, uint32_t seed, uint64_t tail = 0) {
    std::mt19937 rng(seed);
    std::vector<uint64_t> offsets{0};
    const uint64_t end = n - tail;
    while (offsets.back() < end) {
        const uint32_t kind = rng() % 10;
        const uint64_t length = kind < 2   ? 0
                                : kind < 7 ? rng() % 40
                                : kind < 9 ? rng() % (tile / 2)
                                           : rng() % (3 * tile);
        offsets.push_back(std::min(end, offsets.back() + length));
    }
    return offsets;
}

Tensor upload(DataType dtype, const std::vector<uint8_t>& bytes, Stream& stream) {
    const uint64_t n = bytes.size() / bytesOf(dtype);
    Tensor host(cpu, TensorDescriptor(dtype, {n}));
    std::memcpy(host.getMemPtr<void>(), bytes.data(), bytes.size());
    Tensor device(gpu, TensorDescriptor(dtype, {n}));
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

std::vector<uint8_t> offsetBytes(DataType dtype, const std::vector<uint64_t>& offsets) {
    std::vector<uint8_t> bytes(offsets.size() * bytesOf(dtype));
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (dtype == DataType::UINT32) {
            const uint32_t v = static_cast<uint32_t>(offsets[i]);
            std::memcpy(bytes.data() + 4 * i, &v, 4);
        } else {
            std::memcpy(bytes.data() + 8 * i, &offsets[i], 8);
        }
    }
    return bytes;
}

Tensor uploadOffsets(DataType dtype, const std::vector<uint64_t>& offsets, Stream& stream) {
    return upload(dtype, offsetBytes(dtype, offsets), stream);
}

// New contents for a tensor already on the device.
void overwrite(Tensor& device, const std::vector<uint8_t>& bytes, Stream& stream) {
    Tensor host = device.clone(cpu);
    std::memcpy(host.getMemPtr<void>(), bytes.data(), bytes.size());
    device.copyFromAsync(host, stream);
    stream.synchronize();
}

std::vector<uint8_t> download(const Tensor& device, Stream& stream) {
    Tensor host = device.clone(cpu);
    host.copyFromAsync(device, stream);
    stream.synchronize();
    std::vector<uint8_t> bytes(host.getArraySizeInBytes());
    std::memcpy(bytes.data(), host.getMemPtr<void>(), bytes.size());
    return bytes;
}

// The first element whose bits differ (any two NaNs are equal: their payloads may differ), or -1.
int64_t firstDifference(DataType dtype, const std::vector<uint8_t>& got, const std::vector<uint8_t>& want) {
    const uint64_t b = bytesOf(dtype);
    for (uint64_t i = 0; i * b < want.size(); ++i) {
        if (std::memcmp(got.data() + i * b, want.data() + i * b, b) == 0) continue;
        if (dtype != DataType::UINT32 && std::isnan(loadFloat(dtype, got.data() + i * b)) &&
            std::isnan(loadFloat(dtype, want.data() + i * b))) {
            continue;
        }
        return static_cast<int64_t>(i);
    }
    return -1;
}

std::string describe(DataType dtype, const std::vector<uint8_t>& bytes, int64_t i) {
    if (dtype == DataType::UINT32) {
        uint32_t u;
        std::memcpy(&u, bytes.data() + 4 * i, 4);
        return std::to_string(u);
    }
    return std::to_string(loadFloat(dtype, bytes.data() + i * bytesOf(dtype)));
}

std::vector<uint32_t> indicesOf(const std::vector<uint8_t>& bytes) {
    std::vector<uint32_t> indices(bytes.size() / 4);
    std::memcpy(indices.data(), bytes.data(), bytes.size());
    return indices;
}

void expectEqual(DataType out, const std::vector<uint8_t>& got, const std::vector<uint8_t>& want) {
    ASSERT_EQ(got.size(), want.size());
    const int64_t i = firstDifference(out, got, want);
    EXPECT_EQ(i, -1) << "first difference at element " << i << ": got " << describe(out, got, i) << ", want "
                     << describe(out, want, i);
}

const std::vector<SegmentedScanOp> kOps{SegmentedScanOp::Sum,
                                        SegmentedScanOp::Product,
                                        SegmentedScanOp::Min,
                                        SegmentedScanOp::Max,
                                        SegmentedScanOp::ArgMax,
                                        SegmentedScanOp::ArgMin};

std::string opText(SegmentedScanOp op) {
    switch (op) {
        case SegmentedScanOp::Sum:
            return "sum";
        case SegmentedScanOp::Product:
            return "product";
        case SegmentedScanOp::Min:
            return "min";
        case SegmentedScanOp::Max:
            return "max";
        case SegmentedScanOp::ArgMax:
            return "argmax";
        case SegmentedScanOp::ArgMin:
            return "argmin";
    }
    return "unknown";
}

bool converts(SegmentedScanOp op) { return op == SegmentedScanOp::Sum || op == SegmentedScanOp::Product; }

struct Pair {
    SegmentedScanOp op;
    DataType in;
    DataType out;
};

std::vector<Pair> supportedPairs() {
    std::vector<Pair> pairs;
    for (const SegmentedScanOp op : kOps) {
        for (const DataType in : kDTypes) {
            for (const DataType out : kDTypes) {
                if (SegmentedScan::isSupported(op, in, out)) pairs.push_back({op, in, out});
            }
        }
    }
    return pairs;
}

std::string text(const Pair& pair) {
    return opText(pair.op) + " " + dtypeText(pair.in) + " -> " + dtypeText(pair.out);
}

uint64_t tileOf(const Pair& pair) {
    return SegmentedScanKernels::tileElements(static_cast<uint32_t>(bytesOf(pair.in)),
                                              static_cast<uint32_t>(bytesOf(pair.out)));
}

// Ragged rows end at the last offset, and a scan writes nothing from there on. An output that held kUntouched in every
// byte before the launch therefore holds the scan of the rows and then kUntouched (no scan of these inputs writes an
// element of that pattern).
constexpr uint8_t kUntouched = 0xA5;

std::vector<uint8_t> untouched(DataType dtype, uint64_t n) {
    return std::vector<uint8_t>(n * bytesOf(dtype), kUntouched);
}

std::vector<uint8_t> raggedReference(const Pair& pair,
                                     const Variant& variant,
                                     const std::vector<uint8_t>& input,
                                     uint64_t n,
                                     const std::vector<uint64_t>& offsets) {
    std::vector<char> heads = raggedHeads(n, offsets);
    heads.resize(std::min(n, offsets.back()));
    std::vector<uint8_t> want = reference(pair.op, variant, pair.in, pair.out, input, heads);
    want.resize(n * bytesOf(pair.out), kUntouched);
    return want;
}

// An arg scan's values: the input's element at each index, for the elements the indices cover (want_indices has the
// reference's indices for the first `active`), and kUntouched after them; at the index 0xFFFFFFFF (an exclusive scan's
// row start) the min's or the max's identity.
std::vector<uint8_t> argValuesReference(SegmentedScanOp op,
                                        DataType in,
                                        const std::vector<uint8_t>& input,
                                        const std::vector<uint8_t>& want_indices,
                                        uint64_t active,
                                        uint64_t n) {
    const uint64_t b = bytesOf(in);
    const std::vector<uint8_t> identity = identityBytes(op, in);
    std::vector<uint8_t> values(n * b, kUntouched);
    for (uint64_t i = 0; i < active; ++i) {
        uint32_t index;
        std::memcpy(&index, want_indices.data() + 4 * i, 4);
        const uint8_t* from = index == 0xFFFFFFFFU ? identity.data() : input.data() + index * b;
        std::memcpy(values.data() + i * b, from, b);
    }
    return values;
}

// Arg scans run with a values tensor (prefilled with kUntouched) and check it too: every bit of each winning element,
// -0 included (any NaN for a NaN).
void checkDense(const Pair& pair,
                const Variant& variant,
                uint64_t n,
                uint64_t row_length,
                uint32_t seed,
                Stream& stream) {
    SCOPED_TRACE(text(pair) + ", " + variantText(variant) + ", " + std::to_string(n) + " elements, rows of " +
                 std::to_string(row_length));
    const std::vector<uint8_t> host_input = makeInputFor(pair.op, pair.in, n, seed);
    Tensor input = upload(pair.in, host_input, stream);
    Tensor output(gpu, TensorDescriptor(pair.out, {n}));
    std::optional<Tensor> values;
    if (isArg(pair.op)) values = upload(pair.in, untouched(pair.in, n), stream);
    const SegmentedScan scan(pair.op, variant.mode, variant.direction, input, output, row_length, values);
    scan.runOn(stream);
    const std::vector<uint8_t> want =
        reference(pair.op, variant, pair.in, pair.out, host_input, denseHeads(n, row_length));
    expectEqual(pair.out, download(output, stream), want);
    if (values.has_value()) {
        expectEqual(pair.in,
                    download(values.value(), stream),
                    argValuesReference(pair.op, pair.in, host_input, want, n, n));
    }
}

void checkRagged(const Pair& pair,
                 const Variant& variant,
                 uint64_t n,
                 const std::vector<uint64_t>& offsets,
                 DataType offset_dtype,
                 uint32_t seed,
                 Stream& stream) {
    SCOPED_TRACE(text(pair) + ", " + variantText(variant) + ", " + std::to_string(n) + " elements, " +
                 std::to_string(offsets.size() - 1) + " " + dtypeText(offset_dtype) + " ragged rows");
    const std::vector<uint8_t> host_input = makeInputFor(pair.op, pair.in, n, seed);
    Tensor input = upload(pair.in, host_input, stream);
    Tensor output = upload(pair.out, untouched(pair.out, n), stream);
    std::optional<Tensor> values;
    if (isArg(pair.op)) values = upload(pair.in, untouched(pair.in, n), stream);
    const SegmentedScan scan(
        pair.op, variant.mode, variant.direction, input, output, uploadOffsets(offset_dtype, offsets, stream), values);
    scan.runOn(stream);
    const std::vector<uint8_t> want = raggedReference(pair, variant, host_input, n, offsets);
    expectEqual(pair.out, download(output, stream), want);
    if (values.has_value()) {
        const uint64_t active = std::min(n, offsets.back());
        const std::vector<uint8_t> want_values = argValuesReference(pair.op, pair.in, host_input, want, active, n);
        expectEqual(pair.in, download(values.value(), stream), want_values);
    }
}

}  // namespace

TEST(SegmentedScan, SupportsEveryDTypeOwnAndConvertsFloatSumsAndProducts) {
    // Own dtype for every op; a float sum or product to every other float.
    const std::vector<DataType> floats{
        DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16, DataType::BF16, DataType::FP32};
    for (const SegmentedScanOp op : kOps) {
        if (isArg(op)) continue;  // ArgScansWriteUint32IndicesFromEveryDType
        SCOPED_TRACE(opText(op));
        for (const DataType dtype : kDTypes) {
            EXPECT_TRUE(SegmentedScan::isSupported(op, dtype, dtype)) << dtypeText(dtype);
        }
        for (const DataType in : floats) {
            for (const DataType out : floats) {
                if (in == out) continue;
                EXPECT_EQ(SegmentedScan::isSupported(op, in, out), converts(op))
                    << dtypeText(in) << " -> " << dtypeText(out);
            }
            EXPECT_FALSE(SegmentedScan::isSupported(op, in, DataType::UINT32));
            EXPECT_FALSE(SegmentedScan::isSupported(op, DataType::UINT32, in));
        }
        EXPECT_FALSE(SegmentedScan::isSupported(op, DataType::INT32, DataType::INT32));
        EXPECT_FALSE(SegmentedScan::isSupported(op, DataType::UINT64, DataType::UINT64));
    }
    EXPECT_EQ(supportedPairs().size(), 76U);  // 24 own, 20 sums and 20 products to another float, 12 arg scans
    EXPECT_TRUE(SegmentedScan::isOffsetDataTypeSupported(DataType::UINT32));
    EXPECT_TRUE(SegmentedScan::isOffsetDataTypeSupported(DataType::UINT64));
    EXPECT_FALSE(SegmentedScan::isOffsetDataTypeSupported(DataType::INT32));
}

TEST(SegmentedScan, ArgScansWriteUint32IndicesFromEveryDType) {
    for (const SegmentedScanOp op : {SegmentedScanOp::ArgMax, SegmentedScanOp::ArgMin}) {
        SCOPED_TRACE(opText(op));
        for (const DataType in : kDTypes) {
            for (const DataType out : kDTypes) {
                EXPECT_EQ(SegmentedScan::isSupported(op, in, out), out == DataType::UINT32)
                    << dtypeText(in) << " -> " << dtypeText(out);
            }
        }
        EXPECT_FALSE(SegmentedScan::isSupported(op, DataType::INT32, DataType::UINT32));
        EXPECT_FALSE(SegmentedScan::isSupported(op, DataType::FP32, DataType::UINT64));
    }
}

// The arg scans' rules, written out: the first of equal elements, -0 equal to +0, NaN below every number (a row's
// first element while all so far are NaN); flat indices into the whole tensor, over dense and ragged rows; and the
// winning values, -0 kept, when a values tensor is given.
TEST(SegmentedScan, ArgScansTakeTheFirstOfEqualsAndSkipNaN) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> floats{nan, nan, 2.0f, -0.0f, 3.0f, 3.0f, 0.0f, nan, -1.0f};
    for (const DataType dtype : {DataType::FP32, DataType::FP16, DataType::BF16, DataType::FP8_E4M3,
                                 DataType::FP8_E5M2}) {
        SCOPED_TRACE(dtypeText(dtype));
        std::vector<uint8_t> bytes(floats.size() * bytesOf(dtype));
        for (size_t i = 0; i < floats.size(); ++i) storeFloat(dtype, floats[i], bytes.data() + i * bytesOf(dtype));
        Tensor input = upload(dtype, bytes, stream);
        Tensor output(gpu, TensorDescriptor(DataType::UINT32, {floats.size()}));
        Tensor offsets = uploadOffsets(DataType::UINT32, {0, 3, 9}, stream);
        Tensor values(gpu, TensorDescriptor(dtype, {floats.size()}));
        // The indices, and the values at them: written in the same pass, or (without a values tensor) not at all.
        auto run = [&](SegmentedScanOp op, bool ragged, bool with_values) {
            values.memsetAsync(stream, 0x5A);
            const std::optional<Tensor> arg_values = with_values ? std::optional<Tensor>(values) : std::nullopt;
            const std::unique_ptr<SegmentedScan> scan =
                ragged
                    ? std::make_unique<SegmentedScan>(op, kInclusive, kForward, input, output, offsets, arg_values)
                    : std::make_unique<SegmentedScan>(op, kInclusive, kForward, input, output, uint64_t{9}, arg_values);
            scan->runOn(stream);
            const std::vector<uint32_t> indices = indicesOf(download(output, stream));
            const std::vector<uint8_t> got = download(values, stream);
            if (with_values) {
                std::vector<uint8_t> want(bytes.size());
                for (size_t i = 0; i < indices.size(); ++i) {
                    std::memcpy(want.data() + i * bytesOf(dtype), bytes.data() + indices[i] * bytesOf(dtype),
                                bytesOf(dtype));
                }
                expectEqual(dtype, got, want);
            } else {
                EXPECT_EQ(got, std::vector<uint8_t>(bytes.size(), 0x5A));
            }
            return indices;
        };
        for (const bool with_values : {false, true}) {
            SCOPED_TRACE(with_values ? "with values" : "indices only");
            EXPECT_EQ(run(SegmentedScanOp::ArgMax, false, with_values),
                      (std::vector<uint32_t>{0, 0, 2, 2, 4, 4, 4, 4, 4}));
            EXPECT_EQ(run(SegmentedScanOp::ArgMin, false, with_values),
                      (std::vector<uint32_t>{0, 0, 2, 3, 3, 3, 3, 3, 8}));
            EXPECT_EQ(run(SegmentedScanOp::ArgMax, true, with_values),
                      (std::vector<uint32_t>{0, 0, 2, 3, 4, 4, 4, 4, 4}));
            EXPECT_EQ(run(SegmentedScanOp::ArgMin, true, with_values),
                      (std::vector<uint32_t>{0, 0, 2, 3, 3, 3, 3, 3, 8}));
        }
        // The -0 at index 3 is the argmin's winner from there to index 7: its value is -0, not +0.
        run(SegmentedScanOp::ArgMin, false, true);
        const std::vector<uint8_t> got = download(values, stream);
        EXPECT_TRUE(std::signbit(loadFloat(dtype, got.data() + 5 * bytesOf(dtype))));
    }
    const std::vector<uint32_t> uints{5U, 7U, 7U, 0U, 0xFFFFFFFFU, 0xFFFFFFFFU, 0U};
    std::vector<uint8_t> bytes(uints.size() * 4);
    std::memcpy(bytes.data(), uints.data(), bytes.size());
    Tensor input = upload(DataType::UINT32, bytes, stream);
    Tensor output(gpu, TensorDescriptor(DataType::UINT32, {uints.size()}));
    for (const SegmentedScanOp op : {SegmentedScanOp::ArgMax, SegmentedScanOp::ArgMin}) {
        const SegmentedScan scan(op, kInclusive, kForward, input, output, uint64_t{7});
        scan.runOn(stream);
        EXPECT_EQ(indicesOf(download(output, stream)),
                  op == SegmentedScanOp::ArgMax ? (std::vector<uint32_t>{0, 1, 1, 1, 4, 4, 4})
                                                : (std::vector<uint32_t>{0, 0, 0, 3, 3, 3, 3}))
            << opText(op);
    }
}

// Scans written out, over two rows of three (dense and ragged) in every dtype: {2, -1, 3 | 1, -2, 1} for the float
// inputs, {2, 1, 3 | 1, 0, 1} for UINT32's. Each op's expected outputs: `floats` for the float inputs, `uints` for
// UINT32's and an arg scan's indices (the same for both inputs). Where `identity_at` is set, the op's identity
// (identityBytes: 0, 1, +-inf, E4M3's +-448, UINT32's largest and 0; an arg scan the index 0xFFFFFFFF and, as its
// value, the min's or the max's identity).
struct WrittenOut {
    SegmentedScanOp op;
    std::vector<float> floats;
    std::vector<uint32_t> uints;
};

void checkWrittenOut(const Variant& variant, const std::vector<WrittenOut>& wants, const std::vector<char>& identity_at) {
    Stream stream(0);
    const std::vector<float> floats{2.0f, -1.0f, 3.0f, 1.0f, -2.0f, 1.0f};
    const std::vector<uint32_t> uints{2U, 1U, 3U, 1U, 0U, 1U};
    Tensor offsets = uploadOffsets(DataType::UINT32, {0, 3, 6}, stream);
    for (const DataType dtype : kDTypes) {
        const uint64_t b = bytesOf(dtype);
        std::vector<uint8_t> bytes(6 * b);
        for (size_t i = 0; i < 6; ++i) {
            if (dtype == DataType::UINT32) {
                std::memcpy(bytes.data() + 4 * i, &uints[i], 4);
            } else {
                storeFloat(dtype, floats[i], bytes.data() + i * b);
            }
        }
        Tensor input = upload(dtype, bytes, stream);
        for (const WrittenOut& want : wants) {
            SCOPED_TRACE(variantText(variant) + " " + opText(want.op) + " " + dtypeText(dtype));
            const bool arg = isArg(want.op);
            const DataType out = arg ? DataType::UINT32 : dtype;
            const uint64_t ob = bytesOf(out);
            const std::vector<uint8_t> identity = arg ? std::vector<uint8_t>(4, 0xFF) : identityBytes(want.op, out);
            std::vector<uint8_t> expected(6 * ob);
            for (size_t i = 0; i < 6; ++i) {
                uint8_t* p = expected.data() + i * ob;
                if (identity_at[i]) {
                    std::memcpy(p, identity.data(), ob);
                } else if (out == DataType::UINT32) {
                    std::memcpy(p, &want.uints[i], 4);
                } else {
                    storeFloat(out, want.floats[i], p);
                }
            }
            Tensor output(gpu, TensorDescriptor(out, {6}));
            const std::optional<Tensor> values =
                arg ? std::optional<Tensor>(Tensor(gpu, TensorDescriptor(dtype, {6}))) : std::nullopt;
            for (const bool ragged : {false, true}) {
                SCOPED_TRACE(ragged ? "ragged" : "dense");
                const SegmentedScanMode mode = variant.mode;
                const SegmentedScanDirection direction = variant.direction;
                const std::unique_ptr<SegmentedScan> scan =
                    ragged ? std::make_unique<SegmentedScan>(want.op, mode, direction, input, output, offsets, values)
                           : std::make_unique<SegmentedScan>(want.op, mode, direction, input, output, uint64_t{3}, values);
                scan->runOn(stream);
                expectEqual(out, download(output, stream), expected);
                if (arg) {
                    const std::vector<uint8_t> want_values = argValuesReference(want.op, dtype, bytes, expected, 6, 6);
                    expectEqual(dtype, download(values.value(), stream), want_values);
                }
            }
        }
    }
}

// Exclusive: each element takes the scan of the elements before it in its row, and a row's first element the identity.
TEST(SegmentedScan, ExclusiveScansStartEachRowWithTheIdentity) {
    REQUIRE_CUDA_DEVICE();
    checkWrittenOut({kExclusive, kForward},
                    {
                        {SegmentedScanOp::Sum, {0, 2, 1, 0, 1, -1}, {0, 2, 3, 0, 1, 1}},
                        {SegmentedScanOp::Product, {0, 2, -2, 0, 1, -2}, {0, 2, 2, 0, 1, 0}},
                        {SegmentedScanOp::Max, {0, 2, 2, 0, 1, 1}, {0, 2, 2, 0, 1, 1}},
                        {SegmentedScanOp::Min, {0, 2, -1, 0, 1, -2}, {0, 2, 1, 0, 1, 0}},
                        {SegmentedScanOp::ArgMax, {}, {0, 0, 0, 0, 3, 3}},  // ties keep the first
                        {SegmentedScanOp::ArgMin, {}, {0, 0, 1, 0, 3, 4}},
                    },
                    {1, 0, 0, 1, 0, 0});
}

// Reverse: each row from its last element to its first, ties keeping the first met (the last in the row: argmax of
// {1, -2, 1} is index 5); exclusive, each element takes the scan of the elements after it, and a row's last element
// the identity.
TEST(SegmentedScan, ReverseScansRunFromEachRowsLastElement) {
    REQUIRE_CUDA_DEVICE();
    checkWrittenOut({kInclusive, kReverse},
                    {
                        {SegmentedScanOp::Sum, {4, 2, 3, 0, -1, 1}, {6, 4, 3, 2, 1, 1}},
                        {SegmentedScanOp::Product, {-6, -3, 3, -2, -2, 1}, {6, 3, 3, 0, 0, 1}},
                        {SegmentedScanOp::Max, {3, 3, 3, 1, 1, 1}, {3, 3, 3, 1, 1, 1}},
                        {SegmentedScanOp::Min, {-1, -1, 3, -2, -2, 1}, {1, 1, 3, 0, 0, 1}},
                        {SegmentedScanOp::ArgMax, {}, {2, 2, 2, 5, 5, 5}},
                        {SegmentedScanOp::ArgMin, {}, {1, 1, 2, 4, 4, 5}},
                    },
                    {0, 0, 0, 0, 0, 0});
    checkWrittenOut({kExclusive, kReverse},
                    {
                        {SegmentedScanOp::Sum, {2, 3, 0, -1, 1, 0}, {4, 3, 0, 1, 1, 0}},
                        {SegmentedScanOp::Product, {-3, 3, 0, -2, 1, 0}, {3, 3, 0, 0, 1, 0}},
                        {SegmentedScanOp::Max, {3, 3, 0, 1, 1, 0}, {3, 3, 0, 1, 1, 0}},
                        {SegmentedScanOp::Min, {-1, 3, 0, -2, 1, 0}, {1, 3, 0, 0, 1, 0}},
                        {SegmentedScanOp::ArgMax, {}, {2, 2, 0, 5, 5, 0}},
                        {SegmentedScanOp::ArgMin, {}, {1, 2, 0, 4, 5, 0}},
                    },
                    {0, 0, 1, 0, 0, 1});
}

// Every pair in every mode and direction.
TEST(SegmentedScan, DenseRowsMatchTheReferenceForEveryPair) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    uint32_t seed = 1;
    for (const Pair& pair : supportedPairs()) {
        const uint64_t tile = tileOf(pair);
        for (const Variant& variant : kVariants) {
            // Row lengths that start a row in every tile (no tile waits) and that do not; a partial last tile and
            // chunk.
            for (const uint64_t row_length : {uint64_t{1}, uint64_t{7}, tile / 2, tile, 3 * tile + 5}) {
                const uint64_t rows = (40 * tile + 17 + row_length - 1) / row_length;
                checkDense(pair, variant, rows * row_length, row_length, seed++, stream);
            }
            checkDense(pair, variant, 40 * tile + 17, 40 * tile + 17, seed++, stream);  // one row
            checkDense(pair, variant, 5, 5, seed++, stream);                            // less than a chunk
        }
    }
}

TEST(SegmentedScan, RaggedRowsMatchTheReferenceForEveryPair) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    uint32_t seed = 100;
    for (const Pair& pair : supportedPairs()) {
        const uint64_t tile = tileOf(pair);
        const uint64_t n = 40 * tile + 17;
        for (const Variant& variant : kVariants) {
            checkRagged(pair, variant, n, makeOffsets(n, tile, seed, 0), DataType::UINT32, seed + 1, stream);
            seed += 2;
            // The rows end nine tiles and more before n (nothing is written from there on, and the tiles past the
            // end include the last of a group and of the supergroup); UINT64 offsets.
            checkRagged(pair, variant, n, makeOffsets(n, tile, seed, 9 * tile + 1000), DataType::UINT64, seed + 1, stream);
            seed += 2;
        }
    }
}

TEST(SegmentedScan, RaggedEdgeRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const Pair pair{SegmentedScanOp::Sum, DataType::FP32, DataType::FP32};
    const uint64_t tile = tileOf(pair);
    const uint64_t n = 5 * tile + 3;
    // Elements before the first offset; empty rows at the start, inside and at the end; one row over most tiles.
    for (const Variant& variant : kVariants) {
        checkRagged(pair, variant, n, {5, 5, 9, 9, 9, n - 2, n, n}, DataType::UINT32, 1, stream);
        checkRagged(pair, variant, n, {0, n}, DataType::UINT64, 2, stream);
        checkRagged(pair, variant, n, {0, 0, 0}, DataType::UINT32, 3, stream);  // no rows: nothing is written
        checkRagged(pair, variant, 1, {0, 1}, DataType::UINT32, 4, stream);
        // The rows end inside a chunk, at a tile's first element, and after one element.
        checkRagged(pair, variant, n, {0, 3, 2 * tile + 5}, DataType::UINT32, 5, stream);
        checkRagged(pair, variant, n, {0, 3, 2 * tile}, DataType::UINT64, 6, stream);
        checkRagged(pair, variant, n, {0, 1}, DataType::UINT32, 7, stream);
    }
}

// Two supergroups of tiles and part of a third: the chain from one supergroup's inclusive run to the next.
TEST(SegmentedScan, LongRowsAcrossSupergroups) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const Pair pair{SegmentedScanOp::Sum, DataType::FP32, DataType::FP32};
    const uint64_t tile = tileOf(pair);
    const uint64_t n = 2 * SegmentedScanKernels::kSuper * SegmentedScanKernels::kGroup * tile + 3 * tile + 1001;
    for (const Variant& variant : kVariants) {
        checkDense(pair, variant, n, n, 7, stream);
        checkRagged(pair, variant, n, {0, 3, 3, n - 5, n}, DataType::UINT32, 8, stream);
        checkRagged(pair, variant, n, makeOffsets(n, 64 * tile, 9), DataType::UINT64, 10, stream);
    }
}

TEST(SegmentedScan, MinAndMaxSkipNaN) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const SegmentedScanOp op : {SegmentedScanOp::Min, SegmentedScanOp::Max}) {
        for (const DataType dtype : {DataType::FP32, DataType::FP16, DataType::BF16}) {
            const Pair pair{op, dtype, dtype};
            SCOPED_TRACE(text(pair));
            const uint64_t tile = tileOf(pair);
            const uint64_t n = 1000 * ((9 * tile + 11) / 1000 + 1);  // rows of 1000 and a partial last tile
            std::vector<uint8_t> host_input = makeInput(dtype, n, 21, 5);
            for (uint64_t i = 0; i < 40; ++i) {  // a row of NaNs at the start: NaN until its first number
                storeFloat(dtype, std::numeric_limits<float>::quiet_NaN(), host_input.data() + i * bytesOf(dtype));
            }
            Tensor input = upload(dtype, host_input, stream);
            Tensor output(gpu, TensorDescriptor(dtype, {n}));
            const SegmentedScan scan(pair.op, kInclusive, kForward, input, output, uint64_t{1000});
            scan.runOn(stream);
            const std::vector<uint8_t> want =
                reference(pair.op, kInclusiveForward, dtype, dtype, host_input, denseHeads(n, 1000));
            expectEqual(dtype, download(output, stream), want);
        }
    }
}

// Real-valued sums: rounding depends on the association, which is fixed, so runs repeat bit for bit; and a scan run
// again on new input (its next epoch) is right too.
TEST(SegmentedScan, RepeatsBitForBitAndAcrossRuns) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    for (const Pair& pair : {Pair{SegmentedScanOp::Sum, DataType::FP32, DataType::FP32},
                             Pair{SegmentedScanOp::Sum, DataType::FP16, DataType::FP32}}) {
        SCOPED_TRACE(text(pair));
        const uint64_t tile = tileOf(pair);
        const uint64_t n = 300 * tile + 77;
        std::mt19937 rng(31);
        std::vector<uint8_t> host_input(n * bytesOf(pair.in));
        for (uint64_t i = 0; i < n; ++i) {
            const float value = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
            storeFloat(pair.in, value, host_input.data() + i * bytesOf(pair.in));
        }
        Tensor input = upload(pair.in, host_input, stream);
        Tensor output(gpu, TensorDescriptor(pair.out, {n}));
        const std::vector<uint64_t> offsets = makeOffsets(n, 64 * tile, 32);
        const SegmentedScan dense(pair.op, kInclusive, kForward, input, output, n);
        const SegmentedScan ragged(
            pair.op, kInclusive, kForward, input, output, uploadOffsets(DataType::UINT32, offsets, stream));
        for (const SegmentedScan* scan : {&dense, &ragged}) {
            scan->runOn(stream);
            const std::vector<uint8_t> first = download(output, stream);
            for (int run = 0; run < 3; ++run) {
                scan->runOn(stream);
                expectEqual(pair.out, download(output, stream), first);
            }
        }
        // New input into the same scans: their next launches publish a new epoch.
        const std::vector<uint8_t> integers = makeInput(pair.in, n, 33);
        Tensor fresh = upload(pair.in, integers, stream);
        input.copyFromAsync(fresh, stream);
        dense.runOn(stream);
        const std::vector<uint8_t> want_dense =
            reference(pair.op, kInclusiveForward, pair.in, pair.out, integers, denseHeads(n, n));
        expectEqual(pair.out, download(output, stream), want_dense);
        ragged.runOn(stream);
        const std::vector<uint8_t> want_ragged =
            reference(pair.op, kInclusiveForward, pair.in, pair.out, integers, raggedHeads(n, offsets));
        expectEqual(pair.out, download(output, stream), want_ragged);
    }
}

// A launch captured in a CUDA graph, as StampedConditional captures its stages, after launches outside it, then
// replayed on new input and mixed with launches outside it again: the epoch comes from the workspace, so every replay
// is a new launch and no tile takes a run an earlier launch left. Rows of three tiles (dense) and ragged rows, past one
// supergroup, so tiles wait at every level.
TEST(SegmentedScan, ReplaysInACapturedGraph) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const Pair pair{SegmentedScanOp::Sum, DataType::FP32, DataType::FP32};
    const uint64_t tile = tileOf(pair);
    const uint64_t n = 1101 * tile;
    Tensor input(gpu, TensorDescriptor(pair.in, {n}));
    Tensor output(gpu, TensorDescriptor(pair.out, {n}));
    const std::vector<uint64_t> offsets = makeOffsets(n, tile, 41);
    const SegmentedScan dense(pair.op, kInclusive, kForward, input, output, 3 * tile);
    const SegmentedScan ragged(
        pair.op, kInclusive, kForward, input, output, uploadOffsets(DataType::UINT32, offsets, stream));
    uint32_t seed = 50;
    for (const SegmentedScan* scan : {&dense, &ragged}) {
        const std::vector<char> heads = scan == &dense ? denseHeads(n, 3 * tile) : raggedHeads(n, offsets);
        auto check = [&](const char* what, bool replay, cudaGraphExec_t exec) {
            SCOPED_TRACE(std::string(scan == &dense ? "dense: " : "ragged: ") + what);
            const std::vector<uint8_t> host_input = makeInput(pair.in, n, ++seed);
            const Tensor fresh = upload(pair.in, host_input, stream);
            input.copyFromAsync(fresh, stream);
            output.memsetAsync(stream, 0);
            if (replay) {
                ASSERT_EQ(cudaGraphLaunch(exec, stream.getStream()), cudaSuccess);
            } else {
                scan->runOn(stream);
            }
            expectEqual(pair.out,
                        download(output, stream),
                        reference(pair.op, kInclusiveForward, pair.in, pair.out, host_input, heads));
        };
        check("before capture", false, nullptr);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        ASSERT_EQ(cudaStreamBeginCapture(stream.getStream(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
        scan->runOn(stream);
        ASSERT_EQ(cudaStreamEndCapture(stream.getStream(), &graph), cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
        for (int replay = 0; replay < 4; ++replay) check("replay", true, exec);
        check("between replays", false, nullptr);
        for (int replay = 0; replay < 2; ++replay) check("replay after", true, exec);
        EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
        EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    }
}

// One scan whose offsets are rewritten between launches, as a ragged tensor's rows change from step to step (under a
// captured graph too): the rows end early, cover everything in one long row, vanish, and come back. The tiles past the
// rows keep the workspace's epoch, so no launch takes or waits for a run of an earlier one. Past one supergroup.
TEST(SegmentedScan, RowsChangeBetweenLaunches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const Pair pair{SegmentedScanOp::Sum, DataType::FP32, DataType::FP32};
    const uint64_t tile = tileOf(pair);
    const uint64_t n = 1101 * tile + 9;
    const std::vector<uint64_t> few{0, 7, 7, 2 * tile + 5, 3 * tile + 9};
    const std::vector<uint64_t> all{0, 7, 7, 40 * tile, n};  // one row over most tiles
    const std::vector<uint64_t> none{0, 0, 0, 0, 0};
    const std::vector<uint64_t> half{0, 1, 500 * tile, 500 * tile, 600 * tile + 1};
    Tensor input(gpu, TensorDescriptor(pair.in, {n}));
    Tensor output(gpu, TensorDescriptor(pair.out, {n}));
    Tensor offsets = uploadOffsets(DataType::UINT32, few, stream);
    const SegmentedScan scan(pair.op, kInclusive, kForward, input, output, offsets);
    uint32_t seed = 70;
    for (const std::vector<uint64_t>* rows : {&few, &all, &none, &all, &half, &few, &all}) {
        SCOPED_TRACE("rows ending at " + std::to_string(rows->back()));
        const std::vector<uint8_t> host_input = makeInput(pair.in, n, ++seed);
        overwrite(input, host_input, stream);
        overwrite(output, untouched(pair.out, n), stream);
        overwrite(offsets, offsetBytes(DataType::UINT32, *rows), stream);
        scan.runOn(stream);
        expectEqual(pair.out, download(output, stream), raggedReference(pair, kInclusiveForward, host_input, n, *rows));
    }
}

TEST(SegmentedScan, WorkspaceOnlyWhenTilesWait) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t tile = SegmentedScanKernels::tileElements(4, 4);
    Tensor input(gpu, TensorDescriptor(DataType::FP32, {8 * tile}));
    Tensor output(gpu, TensorDescriptor(DataType::FP32, {8 * tile}));
    Tensor indices(gpu, TensorDescriptor(DataType::UINT32, {8 * tile}));
    auto bytes = [&](SegmentedScanOp op, const Variant& variant, const Tensor& out, uint64_t row_length) {
        return SegmentedScan(op, variant.mode, variant.direction, input, out, row_length).getWorkspaceSizeInBytes();
    };
    for (const Variant& variant : kVariants) {
        SCOPED_TRACE(variantText(variant));
        EXPECT_EQ(bytes(SegmentedScanOp::Sum, variant, output, tile / 4), 0U);
        EXPECT_GT(bytes(SegmentedScanOp::Sum, variant, output, 2 * tile), 0U);
        // An arg scan's runs are two words each (the same tiles: FP32 to UINT32 has FP32's tile).
        EXPECT_EQ(bytes(SegmentedScanOp::ArgMax, variant, indices, 2 * tile),
                  2 * bytes(SegmentedScanOp::Sum, variant, output, 2 * tile));
        EXPECT_EQ(bytes(SegmentedScanOp::ArgMin, variant, indices, tile / 4), 0U);
    }
    EXPECT_EQ(SegmentedScan(SegmentedScanOp::Sum, kInclusive, kForward, input, output, 8 * tile).getNumTiles(), 8U);
}

TEST(SegmentedScan, RejectsWhatItDoesNotTake) {
    REQUIRE_CUDA_DEVICE();
    using Op = SegmentedScanOp;
    Tensor fp32(gpu, TensorDescriptor(DataType::FP32, {1024}));
    Tensor fp16(gpu, TensorDescriptor(DataType::FP16, {1024}));
    Tensor other(gpu, TensorDescriptor(DataType::FP32, {1000}));
    Tensor host(cpu, TensorDescriptor(DataType::FP32, {1024}));
    Tensor int_offsets(gpu, TensorDescriptor(DataType::INT32, {3}));
    Tensor indices(gpu, TensorDescriptor(DataType::UINT32, {1024}));
    // Dense rows of row_length in every mode and direction (the same rules for all).
    for (const Variant& variant : kVariants) {
        SCOPED_TRACE(variantText(variant));
        auto dense = [&](Op op,
                         const Tensor& in,
                         const Tensor& out,
                         uint64_t row_length = 1024,
                         const std::optional<Tensor>& values = std::nullopt) {
            const SegmentedScan scan(op, variant.mode, variant.direction, in, out, row_length, values);
        };
        EXPECT_NO_THROW(dense(Op::Sum, fp32, fp16));                         // a float sum writes any float
        EXPECT_THROW(dense(Op::Sum, indices, fp32), std::invalid_argument);  // a UINT32 sum writes UINT32
        EXPECT_THROW(dense(Op::Max, fp16, fp32), std::invalid_argument);     // max writes its own dtype
        EXPECT_THROW(dense(Op::Min, fp32, fp16), std::invalid_argument);     // min writes its own dtype
        EXPECT_THROW(dense(Op::ArgMax, fp32, fp32), std::invalid_argument);  // values, not indices
        EXPECT_NO_THROW(dense(Op::ArgMax, fp32, indices, 1024, fp32));
        EXPECT_THROW(dense(Op::ArgMax, fp32, indices, 1024, fp16), std::invalid_argument);
        EXPECT_THROW(dense(Op::Max, fp32, fp32, 1024, fp32), std::invalid_argument);
        EXPECT_THROW(dense(Op::Sum, fp32, other, 1000), std::invalid_argument);
        EXPECT_THROW(dense(Op::Sum, host, host), std::invalid_argument);
        EXPECT_THROW(dense(Op::Sum, fp32, fp32, 1000), std::invalid_argument);  // not 1024's
        EXPECT_THROW(SegmentedScan(Op::Sum, variant.mode, variant.direction, fp32, fp32, int_offsets),
                     std::invalid_argument);
    }
    EXPECT_TRUE(SegmentedScan::isValueOutputSupported(Op::ArgMin, fp32, fp32));
    EXPECT_FALSE(SegmentedScan::isValueOutputSupported(Op::ArgMin, fp32, other));
    EXPECT_FALSE(SegmentedScan::isValueOutputSupported(Op::Min, fp32, fp32));
    EXPECT_FALSE(SegmentedScan::isSupported(Op::Sum, fp32, other));
    EXPECT_FALSE(SegmentedScan::isSupported(Op::Sum, host, host));
    EXPECT_TRUE(SegmentedScan::isSupported(Op::Sum, fp16, fp32));
    EXPECT_TRUE(SegmentedScan::isSupported(Op::Product, fp32, fp16));
}
