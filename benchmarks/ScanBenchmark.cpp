// Times Thor's CUB scan wrappers in TB/s on the paths StampedScan routes a final-axis scan to, at a fixed element
// count over a range of row counts:
//   flat     one row: DeviceScan over the row (StampedScan's path when the scan has one row);
//   uniform  rows of equal length: the uniform-segmented scan (its path for more than one row);
//   offsets  the offset-segmented scan (its path for a ragged scan), timed here over the same equal-length rows.
// Each table's "flat" line is Thor's DeviceScan over every element as one row: the same bytes in one pass. For forward
// sum, max and argmax the benchmark also times CUB used directly (cub, bykey, flags; see CubWay below), the only
// paths for FP8, which Thor's scans do not take. Each cell is TB/s and its percentage of the input dtype's
// device-to-device copy. Every timed run starts with L2 scrubbed. TB/s counts the bytes a scan must move: the input
// and the output (an arg scan's output is its UINT32 indices).
//
// Each cell's output is checked once, outside the timed runs: sums against a double prefix (FP32 output within 1e-3
// of the row's largest |prefix|; UINT32 exact; 16- and 8-bit outputs report the error only), min / max exactly, arg
// scans by the value at the returned index.

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitives.h"

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda/std/bit>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr double FP32_SUM_TOLERANCE = 1.0e-3;

enum class OpKind { Sum, Min, Max, ArgMin, ArgMax };

struct OpSpec {
    std::string name;
    OpKind kind = OpKind::Sum;
    bool arg = false;
    CubScanOp scan_op = CubScanOp::Sum;
    CubArgScanOp arg_op = CubArgScanOp::ArgMin;
};

struct Options {
    int gpu = 0;
    int warmup = 3;
    int iterations = 12;
    uint64_t elements = 64ULL * MIB;
    std::vector<uint64_t> rows{1, 2, 8, 64, 512, 4096, 32768, 262144, 1048576};
    std::vector<DataType> dtypes{DataType::FP32};
    std::vector<OpSpec> ops;
    std::vector<CubScanDirection> directions{CubScanDirection::Forward};
    bool verify = true;
    std::string csv;
};

std::vector<std::string> splitList(std::string_view text) {
    std::vector<std::string> items;
    std::stringstream stream{std::string(text)};
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

uint64_t parseUnsigned(std::string_view text, std::string_view flag) {
    size_t used = 0;
    const std::string value(text);
    const unsigned long long parsed = std::stoull(value, &used, 10);
    if (used != value.size()) {
        throw std::invalid_argument(std::string(flag) + " expects an unsigned integer, got '" + value + "'.");
    }
    return static_cast<uint64_t>(parsed);
}

DataType parseDType(const std::string& name) {
    if (name == "fp32") return DataType::FP32;
    if (name == "fp16") return DataType::FP16;
    if (name == "bf16") return DataType::BF16;
    if (name == "uint32") return DataType::UINT32;
    if (name == "fp8_e4m3") return DataType::FP8_E4M3;
    if (name == "fp8_e5m2") return DataType::FP8_E5M2;
    throw std::invalid_argument("--dtypes: unknown dtype '" + name +
                                "' (fp32, fp16, bf16, uint32, fp8_e4m3, fp8_e5m2).");
}

const char* dtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return "fp32";
        case DataType::FP16:
            return "fp16";
        case DataType::BF16:
            return "bf16";
        case DataType::UINT32:
            return "uint32";
        case DataType::FP8_E4M3:
            return "fp8_e4m3";
        case DataType::FP8_E5M2:
            return "fp8_e5m2";
        default:
            return "unknown";
    }
}

OpSpec parseOp(const std::string& name) {
    if (name == "sum") return {name, OpKind::Sum, false, CubScanOp::Sum, CubArgScanOp::ArgMin};
    if (name == "min") return {name, OpKind::Min, false, CubScanOp::Min, CubArgScanOp::ArgMin};
    if (name == "max") return {name, OpKind::Max, false, CubScanOp::Max, CubArgScanOp::ArgMin};
    if (name == "argmin") return {name, OpKind::ArgMin, true, CubScanOp::Sum, CubArgScanOp::ArgMin};
    if (name == "argmax") return {name, OpKind::ArgMax, true, CubScanOp::Sum, CubArgScanOp::ArgMax};
    throw std::invalid_argument("--ops: unknown op '" + name + "' (sum, min, max, argmin, argmax).");
}

CubScanDirection parseDirection(const std::string& name) {
    if (name == "forward") return CubScanDirection::Forward;
    if (name == "reverse") return CubScanDirection::Reverse;
    throw std::invalid_argument("--directions: unknown direction '" + name + "' (forward, reverse).");
}

const char* directionName(CubScanDirection direction) {
    return direction == CubScanDirection::Forward ? "forward" : "reverse";
}

void printHelp() {
    std::cout << "thor_scan_benchmark: Thor's CUB scan wrappers in TB/s by row count (inclusive, final axis)\n"
              << "  --gpu N              GPU number (default 0)\n"
              << "  --elements N         elements per tensor (default 67108864)\n"
              << "  --rows A,B,...       row counts; each must divide --elements\n"
              << "                       (default 1,2,8,64,512,4096,32768,262144,1048576)\n"
              << "  --dtypes A,B,...     fp32, fp16, bf16, uint32, fp8_e4m3, fp8_e5m2 (default fp32); FP8 runs\n"
              << "                       the CUB paths only (Thor's scans do not take FP8)\n"
              << "  --ops A,B,...        sum, min, max, argmin, argmax (default sum,max,argmax); forward sum, max\n"
              << "                       and argmax also time CUB used directly (cub, bykey, flags; argmax packed);\n"
              << "                       16-bit sums also write FP32, FP8 sums BF16 and FP32\n"
              << "  --directions A,B     forward, reverse (default forward)\n"
              << "  --warmup N           untimed runs per cell (default 3)\n"
              << "  --iterations N       timed runs per cell; the median is reported (default 12)\n"
              << "  --no-verify          skip the output checks\n"
              << "  --csv PATH           also write one line per cell and path\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    std::vector<std::string> ops{"sum", "max", "argmax"};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto value = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(arg) + " expects a value.");
            }
            return argv[++i];
        };
        if (arg == "--gpu") {
            options.gpu = static_cast<int>(parseUnsigned(value(), arg));
        } else if (arg == "--elements") {
            options.elements = parseUnsigned(value(), arg);
        } else if (arg == "--rows") {
            options.rows.clear();
            for (const std::string& item : splitList(value())) options.rows.push_back(parseUnsigned(item, arg));
        } else if (arg == "--dtypes") {
            options.dtypes.clear();
            for (const std::string& item : splitList(value())) options.dtypes.push_back(parseDType(item));
        } else if (arg == "--ops") {
            ops = splitList(value());
        } else if (arg == "--directions") {
            options.directions.clear();
            for (const std::string& item : splitList(value())) options.directions.push_back(parseDirection(item));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<int>(parseUnsigned(value(), arg));
        } else if (arg == "--iterations") {
            options.iterations = static_cast<int>(parseUnsigned(value(), arg));
        } else if (arg == "--no-verify") {
            options.verify = false;
        } else if (arg == "--csv") {
            options.csv = std::string(value());
        } else if (arg == "--help" || arg == "-h") {
            printHelp();
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option '" + std::string(arg) + "' (--help lists them).");
        }
    }
    for (const std::string& name : ops) options.ops.push_back(parseOp(name));
    if (options.elements == 0 || options.elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--elements must be in [1, 2^31 - 1] (CUB's item-count limit).");
    }
    for (const uint64_t rows : options.rows) {
        if (rows == 0 || options.elements % rows != 0) {
            throw std::invalid_argument("--rows: " + std::to_string(rows) + " does not divide --elements.");
        }
    }
    if (options.rows.empty() || options.dtypes.empty() || options.ops.empty() || options.directions.empty()) {
        throw std::invalid_argument("--rows, --dtypes, --ops and --directions must not be empty.");
    }
    if (options.iterations <= 0) throw std::invalid_argument("--iterations must be positive.");
    return options;
}

// Deterministic input: floats uniform in [-1, 1) rounded to the dtype (zero mean, so long-row sums stay in FP16
// range; FP8 E4M3 sums still pass its 448 maximum), UINT32 in [0, 7]. `values` holds each element exactly as stored.
struct HostInput {
    std::vector<float> values;
    std::vector<uint32_t> uints;
    std::vector<uint8_t> bytes;
};

HostInput makeInput(DataType dtype, uint64_t elements) {
    HostInput input;
    uint64_t state = 0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(dtype) * 0x1446ULL);
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    input.bytes.resize(elements * element_bytes);
    if (dtype == DataType::UINT32) {
        input.uints.resize(elements);
        for (uint64_t i = 0; i < elements; ++i) input.uints[i] = static_cast<uint32_t>(next() & 7ULL);
        std::memcpy(input.bytes.data(), input.uints.data(), input.bytes.size());
        return input;
    }
    input.values.resize(elements);
    for (uint64_t i = 0; i < elements; ++i) {
        const float u = static_cast<float>((next() >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
        if (dtype == DataType::FP32) {
            input.values[i] = u;
            std::memcpy(input.bytes.data() + i * 4, &u, 4);
        } else if (dtype == DataType::FP16) {
            const __half h = __float2half(u);
            input.values[i] = __half2float(h);
            std::memcpy(input.bytes.data() + i * 2, &h, 2);
        } else if (dtype == DataType::BF16) {
            const __nv_bfloat16 b = __float2bfloat16(u);
            input.values[i] = __bfloat162float(b);
            std::memcpy(input.bytes.data() + i * 2, &b, 2);
        } else if (dtype == DataType::FP8_E4M3) {
            const __nv_fp8_e4m3 q(u);
            input.values[i] = static_cast<float>(q);
            std::memcpy(input.bytes.data() + i, &q, 1);
        } else {
            const __nv_fp8_e5m2 q(u);
            input.values[i] = static_cast<float>(q);
            std::memcpy(input.bytes.data() + i, &q, 1);
        }
    }
    return input;
}

float storedValue(DataType dtype, const std::vector<uint8_t>& bytes, uint64_t i) {
    if (dtype == DataType::FP32) {
        float f;
        std::memcpy(&f, bytes.data() + i * 4, 4);
        return f;
    }
    if (dtype == DataType::FP16) {
        __half h;
        std::memcpy(&h, bytes.data() + i * 2, 2);
        return __half2float(h);
    }
    if (dtype == DataType::FP8_E4M3) {
        __nv_fp8_e4m3 q;
        std::memcpy(&q, bytes.data() + i, 1);
        return static_cast<float>(q);
    }
    if (dtype == DataType::FP8_E5M2) {
        __nv_fp8_e5m2 q;
        std::memcpy(&q, bytes.data() + i, 1);
        return static_cast<float>(q);
    }
    __nv_bfloat16 b;
    std::memcpy(&b, bytes.data() + i * 2, 2);
    return __bfloat162float(b);
}

struct Check {
    bool judged = true;  // false: the error is reported, not judged (sums written in 16 or 8 bits)
    bool ok = true;
    double max_error = 0.0;
};

// Walks each row in the scan's direction. Sums: the largest |out - prefix| over the row's largest |prefix| (at least
// 1); judged only for an FP32 output (UINT32: exact). Min / max: exact. Arg scans: the index lies in the row, at or
// before the position (in the scan's direction), and holds the running extreme.
Check checkOutput(DataType dtype,
                  DataType out_dtype,
                  const OpSpec& op,
                  CubScanDirection direction,
                  uint64_t rows,
                  uint64_t cols,
                  const HostInput& input,
                  const std::vector<uint8_t>& output) {
    Check check;
    const bool reverse = direction == CubScanDirection::Reverse;
    auto value = [&](uint64_t i) -> double {
        return dtype == DataType::UINT32 ? static_cast<double>(input.uints[i]) : static_cast<double>(input.values[i]);
    };
    for (uint64_t row = 0; row < rows && check.ok; ++row) {
        const uint64_t begin = row * cols;
        double prefix = 0.0;
        uint32_t uprefix = 0;
        double scale = 1.0;
        double row_error = 0.0;
        double extreme = 0.0;
        for (uint64_t step = 0; step < cols; ++step) {
            const uint64_t i = begin + (reverse ? cols - 1 - step : step);
            const double v = value(i);
            if (op.arg) {
                uint32_t index;
                std::memcpy(&index, output.data() + i * 4, 4);
                if (step == 0 || (op.kind == OpKind::ArgMax ? v > extreme : v < extreme)) extreme = v;
                const bool inside = index >= begin && index < begin + cols;
                const bool behind = reverse ? index >= i : index <= i;
                if (!inside || !behind || value(index) != extreme) {
                    check.ok = false;
                    break;
                }
                continue;
            }
            if (op.kind == OpKind::Sum) {
                if (dtype == DataType::UINT32) {
                    uprefix += input.uints[i];
                    uint32_t got;
                    std::memcpy(&got, output.data() + i * 4, 4);
                    if (got != uprefix) {
                        check.ok = false;
                        break;
                    }
                    continue;
                }
                prefix += v;
                scale = std::max(scale, std::fabs(prefix));
                const double got = static_cast<double>(storedValue(out_dtype, output, i));
                row_error = std::max(row_error, std::fabs(got - prefix));
                continue;
            }
            if (step == 0 || (op.kind == OpKind::Max ? v > extreme : v < extreme)) extreme = v;
            double got;
            if (dtype == DataType::UINT32) {
                uint32_t u;
                std::memcpy(&u, output.data() + i * 4, 4);
                got = static_cast<double>(u);
            } else {
                got = static_cast<double>(storedValue(out_dtype, output, i));
            }
            if (got != extreme) {
                check.ok = false;
                break;
            }
        }
        if (op.kind == OpKind::Sum && dtype != DataType::UINT32) {
            check.max_error = std::max(check.max_error, row_error / scale);
        }
    }
    if (op.kind == OpKind::Sum && dtype != DataType::UINT32) {
        if (out_dtype == DataType::FP32) {
            check.ok = check.max_error <= FP32_SUM_TOLERANCE;
        } else {
            check.judged = false;
        }
    }
    return check;
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0U) return values[middle];
    return 0.5 * (static_cast<double>(values[middle - 1]) + static_cast<double>(values[middle]));
}

struct Timer {
    Stream& stream;
    void* scrub = nullptr;
    uint64_t scrub_bytes = 0;
    int warmup = 0;
    int iterations = 0;

    double medianMs(const std::function<void()>& run) const {
        for (int i = 0; i < warmup; ++i) {
            CUDA_CHECK(cudaMemsetAsync(scrub, 0x31 + (i & 0x3f), scrub_bytes, stream.getStream()));
            run();
        }
        stream.synchronize();
        Event start(stream.getGpuNum(), /*enableTiming=*/true);
        Event stop(stream.getGpuNum(), /*enableTiming=*/true);
        std::vector<float> ms;
        ms.reserve(iterations);
        for (int i = 0; i < iterations; ++i) {
            // The scrub is ordered before the start event, outside the timed interval.
            CUDA_CHECK(cudaMemsetAsync(scrub, 0x71 + (i & 0x3f), scrub_bytes, stream.getStream()));
            start.record(stream);
            run();
            stop.record(stream);
            ms.push_back(stop.synchronizeAndReportElapsedTimeInMilliseconds(start));
        }
        return median(std::move(ms));
    }
};

double terabytesPerSecond(uint64_t bytes, double ms) {
    return ms > 0.0 ? static_cast<double>(bytes) / (ms * 1.0e9) : 0.0;
}

struct Path {
    std::string name;
    size_t temp_bytes = 0;
    std::function<void(const Tensor& temp, Stream& stream)> run;
};

// The plan for one path; `rows` is ignored by "flat" (DeviceScan over every element).
Path makePath(const std::string& name,
              const OpSpec& op,
              CubScanDirection direction,
              const Tensor& input,
              Tensor& output,
              const Tensor& offsets,
              uint64_t elements,
              uint64_t rows) {
    const uint64_t cols = elements / rows;
    const CubScanMode mode = CubScanMode::Inclusive;
    Path path;
    path.name = name;
    if (name == "flat") {
        if (op.arg) {
            const CubDeviceArgScanPlan plan =
                prepareCubDeviceArgScan(input, output, elements, op.arg_op, mode, direction);
            path.temp_bytes = plan.temp_storage_bytes;
            path.run = [plan, &input, &output](const Tensor& temp, Stream& stream) {
                cubDeviceArgScan(plan, temp, input, output, stream);
            };
        } else {
            const CubDeviceScanPlan plan = prepareCubDeviceScan(input, output, elements, op.scan_op, mode, direction);
            path.temp_bytes = plan.temp_storage_bytes;
            path.run = [plan, &input, &output](const Tensor& temp, Stream& stream) {
                cubDeviceScan(plan, temp, input, output, stream);
            };
        }
    } else if (name == "uniform") {
        if (op.arg) {
            const CubDeviceSegmentedUniformArgScanPlan plan =
                prepareCubDeviceSegmentedUniformArgScan(
                    input, output, elements, rows, cols, op.arg_op, mode, direction);
            path.temp_bytes = plan.temp_storage_bytes;
            path.run = [plan, &input, &output](const Tensor& temp, Stream& stream) {
                cubDeviceSegmentedUniformArgScan(plan, temp, input, output, stream);
            };
        } else {
            const CubDeviceSegmentedUniformScanPlan plan =
                prepareCubDeviceSegmentedUniformScan(input, output, elements, rows, cols, op.scan_op, mode, direction);
            path.temp_bytes = plan.temp_storage_bytes;
            path.run = [plan, &input, &output](const Tensor& temp, Stream& stream) {
                cubDeviceSegmentedUniformScan(plan, temp, input, output, stream);
            };
        }
    } else {
        if (op.arg) {
            const CubDeviceSegmentedArgScanPlan plan =
                prepareCubDeviceSegmentedArgScan(input, output, offsets, elements, rows, op.arg_op, mode, direction);
            path.temp_bytes = plan.temp_storage_bytes;
            path.run = [plan, &input, &output, &offsets](const Tensor& temp, Stream& stream) {
                cubDeviceSegmentedArgScan(plan, temp, input, output, offsets, stream);
            };
        } else {
            const CubDeviceSegmentedScanPlan plan =
                prepareCubDeviceSegmentedScan(input, output, offsets, elements, rows, op.scan_op, mode, direction);
            path.temp_bytes = plan.temp_storage_bytes;
            path.run = [plan, &input, &output, &offsets](const Tensor& temp, Stream& stream) {
                cubDeviceSegmentedScan(plan, temp, input, output, offsets, stream);
            };
        }
    }
    return path;
}

std::string checkText(const Check& check, bool verified) {
    if (!verified) return "-";
    std::ostringstream text;
    if (!check.judged) {
        text << "err " << std::scientific << std::setprecision(1) << check.max_error;
    } else if (!check.ok) {
        text << "FAILED";
        if (check.max_error > 0.0) {
            text << " (err " << std::scientific << std::setprecision(1) << check.max_error << ")";
        }
    } else if (check.max_error > 0.0) {
        text << "ok " << std::scientific << std::setprecision(1) << check.max_error;
    } else {
        text << "ok";
    }
    return text.str();
}


// CUB used directly, the way the scans could use it: forward sum, max and argmax.
//   cub     DeviceScan over every element as one row.
//   bykey   DeviceScan::InclusiveScanByKey keyed by the row id i / cols (dense rows).
//   flags   ScanByKey keyed by i, with a row-start bitmap (one bit per element, built from the offsets inside the
//           timed run) deciding whether two neighbours share a row: a ragged scan without a per-element search.
// Values accumulate in FP32 (UINT32 in UINT32): 32- and 16-bit types read and write their own type through raw
// pointers (implicit conversions), FP8 through transform iterators (its conversions are explicit). A sum may write a
// wider type than it reads. Arg scans pack (value, index) into one uint64 (the value's order-preserving bits high,
// 0xFFFFFFFF - index low) and run a plain uint64 max scan, the first of equal values winning.
enum class CubWay { Flat, ByKey, Flags };

template <typename T>
constexpr bool kIsFp8 = std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;

template <typename T>
using AccT = std::conditional_t<std::is_same_v<T, uint32_t>, uint32_t, float>;

template <typename T>
struct ToAccumulator {
    __host__ __device__ AccT<T> operator()(T value) const { return static_cast<AccT<T>>(value); }
};

template <typename O>
struct FromAccumulator {
    __host__ __device__ O operator()(float value) const { return O(value); }
};

template <typename A>
struct SumOp {
    __host__ __device__ A operator()(const A& a, const A& b) const { return a + b; }
};

template <typename A>
struct MaxOp {
    __host__ __device__ A operator()(const A& a, const A& b) const { return a < b ? b : a; }
};

// Bits that order as the values do (floats: the sign flipped in, the magnitude flipped for negatives).
__host__ __device__ inline uint32_t orderedBits(float value) {
    const uint32_t bits = cuda::std::bit_cast<uint32_t>(value);
    return (bits & 0x80000000U) != 0U ? ~bits : (bits | 0x80000000U);
}
__host__ __device__ inline uint32_t orderedBits(uint32_t value) { return value; }

template <typename T>
struct MakePackedArg {
    const T* input;
    __host__ __device__ uint64_t operator()(uint32_t i) const {
        const uint32_t value = orderedBits(ToAccumulator<T>{}(input[i]));
        return (static_cast<uint64_t>(value) << 32) | static_cast<uint64_t>(0xFFFFFFFFU - i);
    }
};

struct TakePackedIndex {
    __host__ __device__ uint32_t operator()(uint64_t packed) const {
        return 0xFFFFFFFFU - static_cast<uint32_t>(packed);
    }
};

struct PackedMaxOp {
    __host__ __device__ uint64_t operator()(uint64_t a, uint64_t b) const { return a < b ? b : a; }
};

struct UniformRowOf {
    uint32_t cols;
    __host__ __device__ uint32_t operator()(uint32_t i) const { return i / cols; }
};

struct KeyEqual {
    __host__ __device__ bool operator()(uint32_t a, uint32_t b) const { return a == b; }
};

// Keys are element indices; CUB compares neighbours (i - 1, i), which share a row unless i starts one.
struct SameRowByStartBits {
    const uint32_t* row_starts;
    __host__ __device__ bool operator()(uint32_t a, uint32_t b) const {
        const uint32_t later = a < b ? b : a;
        return ((row_starts[later >> 5] >> (later & 31U)) & 1U) == 0U;
    }
};

__global__ void markRowStartsKernel(const uint32_t* offsets, uint32_t rows, uint32_t n, uint32_t* row_starts) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        const uint32_t start = offsets[row];
        if (start < n) atomicOr(row_starts + (start >> 5), 1U << (start & 31U));
    }
}

struct CubContext {
    uint32_t n = 0;
    uint32_t rows = 1;
    const uint32_t* offsets = nullptr;
    uint32_t* row_starts = nullptr;  // n / 32 words, rebuilt by every Flags run
};

// Queries the temporary bytes when temp is null (CUB's convention), runs otherwise.
template <typename ValuesIt, typename OutputIt, typename Op>
size_t cubScan(CubWay way,
               void* temp,
               size_t temp_bytes,
               ValuesIt values,
               OutputIt output,
               Op op,
               const CubContext& context,
               cudaStream_t stream) {
    const int items = static_cast<int>(context.n);
    if (way == CubWay::Flat) {
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(temp, temp_bytes, values, output, op, items, stream));
    } else if (way == CubWay::ByKey) {
        const UniformRowOf row_of{context.n / context.rows};
        auto keys = thrust::make_transform_iterator(thrust::counting_iterator<uint32_t>(0), row_of);
        CUDA_CHECK(
            cub::DeviceScan::InclusiveScanByKey(temp, temp_bytes, keys, values, output, op, items, KeyEqual{}, stream));
    } else {
        if (temp != nullptr) {
            const size_t words = (static_cast<size_t>(context.n) + 31) / 32;
            CUDA_CHECK(cudaMemsetAsync(context.row_starts, 0, words * sizeof(uint32_t), stream));
            const uint32_t blocks = (context.rows + 255) / 256;
            markRowStartsKernel<<<blocks, 256, 0, stream>>>(
                context.offsets, context.rows, context.n, context.row_starts);
            CUDA_CHECK(cudaGetLastError());
        }
        const thrust::counting_iterator<uint32_t> keys(0);
        CUDA_CHECK(cub::DeviceScan::InclusiveScanByKey(
            temp, temp_bytes, keys, values, output, op, items, SameRowByStartBits{context.row_starts}, stream));
    }
    return temp_bytes;
}

// A sum or max reading T and writing O.
template <typename T, typename O>
size_t cubValueScan(CubWay way,
                    OpKind kind,
                    void* temp,
                    size_t temp_bytes,
                    const Tensor& input,
                    Tensor& output,
                    const CubContext& context,
                    cudaStream_t stream) {
    using A = AccT<T>;
    const T* in = input.getMemPtr<T>();
    O* out_ptr = output.getMemPtr<O>();
    auto run = [&](auto values, auto out) {
        return kind == OpKind::Sum ? cubScan(way, temp, temp_bytes, values, out, SumOp<A>{}, context, stream)
                                   : cubScan(way, temp, temp_bytes, values, out, MaxOp<A>{}, context, stream);
    };
    auto values = [&]() {
        if constexpr (kIsFp8<T>) {
            return thrust::make_transform_iterator(in, ToAccumulator<T>{});
        } else {
            return in;
        }
    }();
    if constexpr (kIsFp8<O>) {
        return run(values, thrust::make_transform_output_iterator(out_ptr, FromAccumulator<O>{}));
    } else {
        return run(values, out_ptr);
    }
}

template <typename T>
size_t cubArgScan(CubWay way,
                  void* temp,
                  size_t temp_bytes,
                  const Tensor& input,
                  Tensor& output,
                  const CubContext& context,
                  cudaStream_t stream) {
    const MakePackedArg<T> make_packed{input.getMemPtr<T>()};
    auto values = thrust::make_transform_iterator(thrust::counting_iterator<uint32_t>(0), make_packed);
    auto indices = thrust::make_transform_output_iterator(output.getMemPtr<uint32_t>(), TakePackedIndex{});
    return cubScan(way, temp, temp_bytes, values, indices, PackedMaxOp{}, context, stream);
}

bool hasCubWay(const OpSpec& op, CubScanDirection direction) {
    return direction == CubScanDirection::Forward &&
           (op.kind == OpKind::Sum || op.kind == OpKind::Max || op.kind == OpKind::ArgMax);
}

// The output dtypes timed for an op on a dtype: an arg scan writes UINT32 indices; a sum may also write FP32 (16-bit
// and FP8 inputs) and BF16 (FP8 inputs); a max writes its own type.
std::vector<DataType> outputDTypes(const OpSpec& op, DataType dtype, bool cub_ways) {
    if (op.arg) return {DataType::UINT32};
    std::vector<DataType> outs{dtype};
    if (cub_ways && op.kind == OpKind::Sum) {
        const bool fp8 = dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;
        if (fp8) outs.push_back(DataType::BF16);
        if (fp8 || dtype == DataType::FP16 || dtype == DataType::BF16) outs.push_back(DataType::FP32);
    }
    return outs;
}

Path makeCubPath(const std::string& name,
                 CubWay way,
                 const OpSpec& op,
                 DataType dtype,
                 DataType out_dtype,
                 const Tensor& input,
                 Tensor& output,
                 const CubContext& context) {
    const OpKind kind = op.kind;
    auto call = [=, &input, &output](void* temp, size_t bytes, cudaStream_t stream) -> size_t {
        auto value = [&](auto in_tag, auto out_tag) {
            using T = decltype(in_tag);
            using O = decltype(out_tag);
            return cubValueScan<T, O>(way, kind, temp, bytes, input, output, context, stream);
        };
        auto arg = [&](auto in_tag) {
            return cubArgScan<decltype(in_tag)>(way, temp, bytes, input, output, context, stream);
        };
        const bool argmax = kind == OpKind::ArgMax;
        switch (dtype) {
            case DataType::FP32:
                if (argmax) return arg(float{});
                return value(float{}, float{});
            case DataType::UINT32:
                if (argmax) return arg(uint32_t{});
                return value(uint32_t{}, uint32_t{});
            case DataType::FP16:
                if (argmax) return arg(__half{});
                if (out_dtype == DataType::FP32) return value(__half{}, float{});
                return value(__half{}, __half{});
            case DataType::BF16:
                if (argmax) return arg(__nv_bfloat16{});
                if (out_dtype == DataType::FP32) return value(__nv_bfloat16{}, float{});
                return value(__nv_bfloat16{}, __nv_bfloat16{});
            case DataType::FP8_E4M3:
                if (argmax) return arg(__nv_fp8_e4m3{});
                if (out_dtype == DataType::FP32) return value(__nv_fp8_e4m3{}, float{});
                if (out_dtype == DataType::BF16) return value(__nv_fp8_e4m3{}, __nv_bfloat16{});
                return value(__nv_fp8_e4m3{}, __nv_fp8_e4m3{});
            case DataType::FP8_E5M2:
                if (argmax) return arg(__nv_fp8_e5m2{});
                if (out_dtype == DataType::FP32) return value(__nv_fp8_e5m2{}, float{});
                if (out_dtype == DataType::BF16) return value(__nv_fp8_e5m2{}, __nv_bfloat16{});
                return value(__nv_fp8_e5m2{}, __nv_fp8_e5m2{});
            default:
                throw std::invalid_argument("Unsupported dtype for the CUB paths.");
        }
    };
    Path path;
    path.name = name;
    path.temp_bytes = call(nullptr, 0, nullptr);
    path.run = [call](const Tensor& temp, Stream& stream) {
        void* temp_ptr = const_cast<void*>(temp.getMemPtr<void>());
        static_cast<void>(call(temp_ptr, temp.getArraySizeInBytes(), stream.getStream()));
    };
    return path;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        ScopedGpu scoped_gpu(options.gpu);
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.gpu));
        const uint64_t l2_bytes = static_cast<uint64_t>(properties.l2CacheSize);
        const uint64_t scrub_bytes = std::max<uint64_t>(2 * l2_bytes, 64 * MIB);

        void* scrub = nullptr;
        CUDA_CHECK(cudaMalloc(&scrub, scrub_bytes));
        struct ScrubGuard {
            void* ptr;
            ~ScrubGuard() { (void)cudaFree(ptr); }
        } scrub_guard{scrub};

        Stream stream(options.gpu);
        const Timer timer{stream, scrub, scrub_bytes, options.warmup, options.iterations};
        const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, options.gpu);
        const uint64_t n = options.elements;
        Tensor row_starts(gpu, TensorDescriptor(DataType::UINT32, {(n + 31) / 32}));

        std::ofstream csv;
        if (!options.csv.empty()) {
            csv.open(options.csv);
            if (!csv) throw std::runtime_error("Cannot open --csv file '" + options.csv + "'.");
            csv << "dtype,out_dtype,op,direction,rows,cols,path,production,p50_ms,tbps,pct_of_copy,check,max_error\n";
        }
#ifdef THOR_DEBUG
        std::cout << "WARNING: Thor was built with THOR_DEBUG. Use a Release or RelWithDebInfo build for timings.\n";
#endif
        std::cout << "Thor scan benchmark (inclusive, final axis)\n"
                  << "GPU: " << properties.name << " (device " << options.gpu << ", " << properties.multiProcessorCount
                  << " SMs, L2 " << std::fixed << std::setprecision(1) << static_cast<double>(l2_bytes) / MIB
                  << " MiB)\nElements per tensor: " << n << "; L2 scrubbed before every timed run; median of "
                  << options.iterations << " runs.\nEach cell: TB/s and % of the input dtype's device-to-device copy. "
                  << "Checks: per path under each table.\n";

        bool any_failed = false;
        std::vector<uint8_t> host_output;
        for (const DataType dtype : options.dtypes) {
            const bool thor_dtype = isCubScanDTypeSupported(dtype);
            const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
            const HostInput host = makeInput(dtype, n);
            Tensor input(gpu, TensorDescriptor(dtype, {n}));
            Tensor copy_out(gpu, TensorDescriptor(dtype, {n}));
            Tensor indices_out(gpu, TensorDescriptor(DataType::UINT32, {n}));
            CUDA_CHECK(
                cudaMemcpy(input.getMemPtr<void>(), host.bytes.data(), host.bytes.size(), cudaMemcpyHostToDevice));

            const uint64_t copy_bytes = 2 * n * element_bytes;
            const double copy_ms = timer.medianMs([&]() {
                CUDA_CHECK(cudaMemcpyAsync(copy_out.getMemPtr<void>(),
                                           input.getMemPtr<void>(),
                                           n * element_bytes,
                                           cudaMemcpyDeviceToDevice,
                                           stream.getStream()));
            });
            const double copy_tbps = terabytesPerSecond(copy_bytes, copy_ms);
            std::cout << "\n" << dtypeName(dtype) << ": copy (read + write " << std::fixed << std::setprecision(0)
                      << static_cast<double>(copy_bytes) / MIB << " MiB) " << std::setprecision(3) << copy_tbps
                      << " TB/s" << (thor_dtype ? "" : "; Thor's scans do not take this dtype (CUB paths only)")
                      << '\n';

            for (const OpSpec& op : options.ops) {
                for (const CubScanDirection direction : options.directions) {
                    const bool cub_ways = hasCubWay(op, direction);
                    if (!thor_dtype && !cub_ways) {
                        std::cout << "\n" << dtypeName(dtype) << ' ' << op.name << ' ' << directionName(direction)
                                  << ": no path (Thor's scans do not take this dtype; no CUB path for this op)\n";
                        continue;
                    }
                    for (const DataType out_dtype : outputDTypes(op, dtype, cub_ways)) {
                        const bool thor_paths = thor_dtype && (op.arg || out_dtype == dtype);
                        const uint64_t out_bytes = TensorDescriptor::getElementSizeInBytes(out_dtype);
                        Tensor values_out(gpu, TensorDescriptor(op.arg ? dtype : out_dtype, {n}));
                        Tensor& output = op.arg ? indices_out : values_out;
                        const uint64_t moved = n * element_bytes + n * out_bytes;

                        // Per path over the table: the largest error and any failed check.
                        struct Summary {
                            std::string name;
                            bool judged = true;
                            bool failed = false;
                            double max_error = 0.0;
                        };
                        std::vector<Summary> summaries;
                        auto summarize = [&](const std::string& name, const Check& check) {
                            auto it = std::find_if(
                                summaries.begin(), summaries.end(), [&](const Summary& s) { return s.name == name; });
                            if (it == summaries.end()) {
                                summaries.push_back(Summary{name});
                                it = summaries.end() - 1;
                            }
                            it->judged = check.judged;
                            it->failed = it->failed || (check.judged && !check.ok);
                            it->max_error = std::max(it->max_error, check.max_error);
                        };
                        auto measure = [&](Path& path, const Tensor& temp, uint64_t rows, bool production) {
                            const double ms = timer.medianMs([&]() { path.run(temp, stream); });
                            Check check;
                            if (options.verify) {
                                host_output.resize(n * out_bytes);
                                CUDA_CHECK(cudaMemcpy(host_output.data(),
                                                      output.getMemPtr<void>(),
                                                      host_output.size(),
                                                      cudaMemcpyDeviceToHost));
                                check = checkOutput(dtype, out_dtype, op, direction, rows, n / rows, host, host_output);
                                any_failed = any_failed || (check.judged && !check.ok);
                                summarize(path.name, check);
                            }
                            const double tbps = terabytesPerSecond(moved, ms);
                            const double pct = copy_tbps > 0.0 ? 100.0 * tbps / copy_tbps : 0.0;
                            if (csv) {
                                csv << dtypeName(dtype) << ',' << dtypeName(out_dtype) << ',' << op.name << ','
                                    << directionName(direction) << ',' << rows << ',' << n / rows << ',' << path.name
                                    << ',' << (production ? 1 : 0) << ',' << ms << ',' << tbps << ',' << pct << ','
                                    << checkText(check, options.verify) << ',' << check.max_error << '\n';
                            }
                            return std::pair<double, double>{tbps, pct};
                        };
                        auto cell = [](std::pair<double, double> result) {
                            std::ostringstream text;
                            text << std::fixed << std::setprecision(3) << std::setw(7) << result.first
                                 << std::setprecision(0) << std::setw(5) << result.second << '%';
                            return text.str();
                        };

                        CubContext flat_context;
                        flat_context.n = static_cast<uint32_t>(n);
                        const Tensor no_offsets;
                        std::cout << "\n" << dtypeName(dtype) << ' ' << op.name;
                        if (!op.arg && out_dtype != dtype) std::cout << " -> " << dtypeName(out_dtype);
                        std::cout << ' ' << directionName(direction) << " (TB/s of " << std::setprecision(0)
                                  << static_cast<double>(moved) / MIB << " MiB moved), one row of " << n
                                  << " elements:\n";
                        std::vector<Path> heads;
                        if (thor_paths) {
                            heads.push_back(makePath("flat", op, direction, input, output, no_offsets, n, 1));
                        }
                        if (cub_ways) {
                            heads.push_back(
                                makeCubPath("cub", CubWay::Flat, op, dtype, out_dtype, input, output, flat_context));
                        }
                        for (Path& head : heads) {
                            Tensor head_temp = allocateCubTemporaryStorage(gpu, head.temp_bytes);
                            std::cout << "  " << std::left << std::setw(6) << head.name << std::right
                                      << cell(measure(head, head_temp, 1, false)) << '\n';
                        }

                        std::vector<std::string> columns;
                        if (thor_paths) {
                            columns.push_back("thor");
                            columns.push_back("offsets");
                        }
                        if (cub_ways) {
                            columns.push_back("bykey");
                            columns.push_back("flags");
                        }
                        std::cout << std::right << std::setw(9) << "rows" << std::setw(10) << "cols";
                        if (thor_paths) std::cout << std::setw(9) << "thor is";
                        for (const std::string& column : columns) std::cout << std::setw(14) << column;
                        std::cout << '\n';

                        for (const uint64_t rows : options.rows) {
                            const uint64_t cols = n / rows;
                            // StampedScan: one row runs DeviceScan over it; more rows run the uniform-segmented scan.
                            const std::string production_name = rows == 1 ? "flat" : "uniform";
                            std::vector<uint32_t> host_offsets(rows + 1);
                            for (uint64_t r = 0; r <= rows; ++r) host_offsets[r] = static_cast<uint32_t>(r * cols);
                            Tensor offsets(gpu, TensorDescriptor(DataType::UINT32, {rows + 1}));
                            CUDA_CHECK(cudaMemcpy(offsets.getMemPtr<void>(),
                                                  host_offsets.data(),
                                                  host_offsets.size() * sizeof(uint32_t),
                                                  cudaMemcpyHostToDevice));
                            CubContext context;
                            context.n = static_cast<uint32_t>(n);
                            context.rows = static_cast<uint32_t>(rows);
                            context.offsets = offsets.getMemPtr<uint32_t>();
                            context.row_starts = row_starts.getMemPtr<uint32_t>();

                            std::vector<Path> paths;
                            if (thor_paths) {
                                paths.push_back(
                                    makePath(production_name, op, direction, input, output, offsets, n, rows));
                                paths.back().name = "thor";
                                paths.push_back(makePath("offsets", op, direction, input, output, offsets, n, rows));
                            }
                            if (cub_ways) {
                                paths.push_back(makeCubPath(
                                    "bykey", CubWay::ByKey, op, dtype, out_dtype, input, output, context));
                                paths.push_back(makeCubPath(
                                    "flags", CubWay::Flags, op, dtype, out_dtype, input, output, context));
                            }
                            size_t temp_bytes = 0;
                            for (const Path& path : paths) temp_bytes = std::max(temp_bytes, path.temp_bytes);
                            Tensor temp = allocateCubTemporaryStorage(gpu, temp_bytes);

                            std::cout << std::right << std::setw(9) << rows << std::setw(10) << cols;
                            if (thor_paths) std::cout << std::setw(9) << production_name;
                            for (size_t i = 0; i < paths.size(); ++i) {
                                std::cout << "  " << cell(measure(paths[i], temp, rows, thor_paths && i == 0));
                            }
                            std::cout << '\n';
                        }
                        if (options.verify) {
                            std::cout << "  checks:";
                            for (const Summary& s : summaries) {
                                std::cout << ' ' << s.name << ' ';
                                if (s.failed) {
                                    std::cout << "FAILED";
                                } else if (!s.judged || s.max_error > 0.0) {
                                    std::cout << (s.judged ? "ok " : "err ") << std::scientific
                                              << std::setprecision(1) << s.max_error << std::fixed;
                                } else {
                                    std::cout << "ok";
                                }
                                std::cout << ';';
                            }
                            std::cout << '\n';
                        }
                    }
                }
            }
        }
        if (any_failed) {
            std::cout << "\nSome outputs FAILED their check.\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "thor_scan_benchmark: " << error.what() << '\n';
        return 2;
    }
}
