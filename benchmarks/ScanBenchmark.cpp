// Times Thor's scan kernel (SegmentedScan) in TB/s at a fixed element count over a range of row counts, inclusive or
// exclusive (--modes), forward or reverse (--directions):
//   tk       over dense rows: StampedScan's path for every scan (the benchmark has no product);
//   tkr      over ragged rows (UINT32 offsets), timed here over the same equal-length rows;
//   tkv      (arg scans) tk writing the winning values too, in the same pass: StampedScan's path for a paired arg scan.
// Each table's line above the rows is tk over every element as one row. Float sums also write every other float.
//
// --ops maxbwd / minbwd time the backward of a max or min scan (ScanMinMaxBackward, StampedScanMinMaxBackward's path):
// fu over dense rows and fur over ragged rows (UINT32 offsets, the same equal-length rows). It takes x, dy and dx each
// in any float dtype: --dtypes gives x's, --grads the pairs of dy's and dx's (one table per pair). Its test on other
// rows and in every combination of dtypes is ScanMinMaxBackwardTest.
// TB/s counts what the backward must move at least: x, dy and dx (12 B per element in FP32). Each cell's dx is checked
// exactly against the backward's definition on the host, converted to dx's format (dy is a multiple of 1/4, so every
// sum is exact in any order), then run once more and compared byte for byte. (The backward Thor ran before 1505, an arg
// scan and a sort-based scatter-add, was timed beside it until 1500: 5.2 to 5.7 ms at 64M FP32 against 0.51 to 0.56.)
// Each cell is TB/s and its percentage of the input dtype's device-to-device copy. Every timed run starts with L2
// scrubbed. TB/s counts the bytes a scan must move: the input and the output (an arg scan's output is its UINT32
// indices; tkv also counts the values it writes, once).
//
// Each cell's output is checked once, outside the timed runs: sums against a double prefix (FP32 output within 1e-3
// of the row's largest |prefix|; UINT32 exact; 16- and 8-bit outputs report the error only), min / max exactly, arg
// scans by the value at the returned index (tkv: and its values, against the input's bytes at the index); an
// exclusive scan against the prefix before each element, a row's first element against the op's identity (an arg
// scan's: the index 0xFFFFFFFF, and the min's or max's identity for its value). Then the cell runs once more and the
// two outputs are compared byte for byte (reproducibility).

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Scan/ScanMinMaxBackward.h"
#include "Utilities/TensorOperations/Scan/SegmentedScan.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr double FP32_SUM_TOLERANCE = 1.0e-3;

enum class OpKind { Sum, Min, Max, ArgMin, ArgMax, MaxBackward, MinBackward };

struct OpSpec {
    std::string name;
    OpKind kind = OpKind::Sum;
    bool arg = false;
    bool backward = false;  // maxbwd / minbwd: the scan's backward (the fused kernel's prototype)
};

// The formats of dy and dx for the fused backward kernel; no value: x's dtype.
struct GradientDTypes {
    std::optional<DataType> dy;
    std::optional<DataType> dx;
};

struct Options {
    int gpu = 0;
    int warmup = 3;
    int iterations = 12;
    uint64_t elements = 64ULL * MIB;
    std::vector<uint64_t> rows{1, 2, 8, 64, 512, 4096, 32768, 262144, 1048576};
    std::vector<DataType> dtypes{DataType::FP32};
    std::vector<OpSpec> ops;
    std::vector<SegmentedScanDirection> directions{SegmentedScanDirection::Forward};
    std::vector<SegmentedScanMode> modes{SegmentedScanMode::Inclusive};
    bool verify = true;
    std::vector<GradientDTypes> grads{GradientDTypes{DataType::FP32, DataType::FP32}};  // maxbwd / minbwd
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
    if (name == "sum") return {name, OpKind::Sum, false};
    if (name == "min") return {name, OpKind::Min, false};
    if (name == "max") return {name, OpKind::Max, false};
    if (name == "argmin") return {name, OpKind::ArgMin, true};
    if (name == "argmax") return {name, OpKind::ArgMax, true};
    if (name == "maxbwd") return {name, OpKind::MaxBackward, false, true};
    if (name == "minbwd") return {name, OpKind::MinBackward, false, true};
    throw std::invalid_argument("--ops: unknown op '" + name + "' (sum, min, max, argmin, argmax, maxbwd, minbwd).");
}

SegmentedScanDirection parseDirection(const std::string& name) {
    if (name == "forward") return SegmentedScanDirection::Forward;
    if (name == "reverse") return SegmentedScanDirection::Reverse;
    throw std::invalid_argument("--directions: unknown direction '" + name + "' (forward, reverse).");
}

const char* directionName(SegmentedScanDirection direction) {
    return direction == SegmentedScanDirection::Forward ? "forward" : "reverse";
}

SegmentedScanMode parseMode(const std::string& name) {
    if (name == "inclusive") return SegmentedScanMode::Inclusive;
    if (name == "exclusive") return SegmentedScanMode::Exclusive;
    throw std::invalid_argument("--modes: unknown mode '" + name + "' (inclusive, exclusive).");
}

const char* modeName(SegmentedScanMode mode) {
    return mode == SegmentedScanMode::Inclusive ? "inclusive" : "exclusive";
}

bool isExclusive(SegmentedScanMode mode) { return mode == SegmentedScanMode::Exclusive; }

// "DY:DX", each a float dtype or x.
GradientDTypes parseGradientDTypes(const std::string& text) {
    const size_t colon = text.find(':');
    if (colon == std::string::npos) throw std::invalid_argument("--grads: '" + text + "' is not DY:DX.");
    auto one = [&](const std::string& name) -> std::optional<DataType> {
        if (name == "x") return std::nullopt;
        const DataType dtype = parseDType(name);
        if (dtype == DataType::UINT32) throw std::invalid_argument("--grads: gradients are floats.");
        return dtype;
    };
    return GradientDTypes{one(text.substr(0, colon)), one(text.substr(colon + 1))};
}

void printHelp() {
    std::cout << "thor_scan_benchmark: Thor's scans in TB/s by row count (final axis)\n"
              << "  --gpu N              GPU number (default 0)\n"
              << "  --elements N         elements per tensor (default 67108864)\n"
              << "  --rows A,B,...       row counts; each must divide --elements\n"
              << "                       (default 1,2,8,64,512,4096,32768,262144,1048576)\n"
              << "  --dtypes A,B,...     fp32, fp16, bf16, uint32, fp8_e4m3, fp8_e5m2 (default fp32)\n"
              << "  --ops A,B,...        sum, min, max, argmin, argmax (default sum,max,argmax); float sums also\n"
              << "                       write every other float. maxbwd, minbwd: the backward of a max or min\n"
              << "                       scan, the fused kernel's prototype (x in each float --dtypes)\n"
              << "  --grads DY:DX,...    maxbwd / minbwd: the dtypes of dy and dx for the fused kernel, a float dtype\n"
              << "                       or x for x's dtype; one table per pair (default fp32:fp32)\n"
              << "  --directions A,B     forward, reverse (default forward)\n"
              << "  --modes A,B          inclusive, exclusive (default inclusive)\n"
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
        } else if (arg == "--modes") {
            options.modes.clear();
            for (const std::string& item : splitList(value())) options.modes.push_back(parseMode(item));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<int>(parseUnsigned(value(), arg));
        } else if (arg == "--iterations") {
            options.iterations = static_cast<int>(parseUnsigned(value(), arg));
        } else if (arg == "--no-verify") {
            options.verify = false;
        } else if (arg == "--grads") {
            options.grads.clear();
            for (const std::string& item : splitList(value())) options.grads.push_back(parseGradientDTypes(item));
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
    if (options.elements == 0 || options.elements > (1ULL << 31)) {
        throw std::invalid_argument("--elements must be in [1, 2^31] (the arg scans' limit).");
    }
    for (const uint64_t rows : options.rows) {
        if (rows == 0 || options.elements % rows != 0) {
            throw std::invalid_argument("--rows: " + std::to_string(rows) + " does not divide --elements.");
        }
    }
    if (options.rows.empty() || options.dtypes.empty() || options.ops.empty() || options.directions.empty() ||
        options.modes.empty() || options.grads.empty()) {
        throw std::invalid_argument("--rows, --dtypes, --ops, --directions, --modes and --grads must not be empty.");
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

// What an exclusive min or max writes for a row's first element: +inf / -inf, E4M3's +-448 (it has no infinity), and
// UINT32's largest value / 0 (SegmentedScan.h). Arg scans' values: the min's (argmin) or the max's (argmax).
double identityValue(DataType out_dtype, bool max) {
    if (out_dtype == DataType::UINT32) return max ? 0.0 : 4294967295.0;
    if (out_dtype == DataType::FP8_E4M3) return max ? -448.0 : 448.0;
    return max ? -INFINITY : INFINITY;
}

// The bytes of that identity in a dtype, as an exclusive arg scan writes them as the value of a row's first element.
std::vector<uint8_t> identityBytes(DataType dtype, bool max) {
    switch (dtype) {
        case DataType::FP32: {
            const uint32_t bits = max ? 0xFF800000U : 0x7F800000U;
            std::vector<uint8_t> bytes(4);
            std::memcpy(bytes.data(), &bits, 4);
            return bytes;
        }
        case DataType::UINT32: {
            const uint32_t bits = max ? 0U : 0xFFFFFFFFU;
            std::vector<uint8_t> bytes(4);
            std::memcpy(bytes.data(), &bits, 4);
            return bytes;
        }
        case DataType::FP16:
            return max ? std::vector<uint8_t>{0x00, 0xFC} : std::vector<uint8_t>{0x00, 0x7C};
        case DataType::BF16:
            return max ? std::vector<uint8_t>{0x80, 0xFF} : std::vector<uint8_t>{0x80, 0x7F};
        case DataType::FP8_E4M3:
            return {static_cast<uint8_t>(max ? 0xFE : 0x7E)};
        default:
            return {static_cast<uint8_t>(max ? 0xFC : 0x7C)};
    }
}

// Walks each row in the scan's direction. Sums: the largest |out - prefix| over the row's largest |prefix| (at least
// 1); judged only for an FP32 output (UINT32: exact). Min / max: exact. Arg scans: the index lies in the row, at or
// before the position (in the scan's direction), and holds the running extreme. Exclusive: each element against the
// elements before it (an arg scan's index strictly before), a row's first element against the identity (an arg
// scan's: the index 0xFFFFFFFF).
Check checkOutput(DataType dtype,
                  DataType out_dtype,
                  const OpSpec& op,
                  SegmentedScanDirection direction,
                  SegmentedScanMode mode,
                  uint64_t rows,
                  uint64_t cols,
                  const HostInput& input,
                  const std::vector<uint8_t>& output) {
    Check check;
    const bool reverse = direction == SegmentedScanDirection::Reverse;
    const bool exclusive = isExclusive(mode);
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
            // Folds element i into the running state; an inclusive scan's output includes it, an exclusive one's not.
            auto fold = [&]() {
                if (op.kind == OpKind::Sum) {
                    prefix += v;
                    uprefix += dtype == DataType::UINT32 ? input.uints[i] : 0U;
                } else {
                    const bool max = op.kind == OpKind::Max || op.kind == OpKind::ArgMax;
                    if (step == 0 || (max ? v > extreme : v < extreme)) extreme = v;
                }
            };
            if (!exclusive) fold();
            if (op.arg) {
                uint32_t index;
                std::memcpy(&index, output.data() + i * 4, 4);
                bool good;
                if (exclusive && step == 0) {
                    good = index == 0xFFFFFFFFU;
                } else {
                    const bool inside = index >= begin && index < begin + cols;
                    const bool behind =
                        reverse ? (exclusive ? index > i : index >= i) : (exclusive ? index < i : index <= i);
                    good = inside && behind && value(index) == extreme;
                }
                if (!good) {
                    check.ok = false;
                    break;
                }
            } else if (op.kind == OpKind::Sum) {
                if (dtype == DataType::UINT32) {
                    uint32_t got;
                    std::memcpy(&got, output.data() + i * 4, 4);
                    if (got != uprefix) {
                        check.ok = false;
                        break;
                    }
                } else {
                    scale = std::max(scale, std::fabs(prefix));
                    const double got = static_cast<double>(storedValue(out_dtype, output, i));
                    row_error = std::max(row_error, std::fabs(got - prefix));
                }
            } else {
                double got;
                if (dtype == DataType::UINT32) {
                    uint32_t u;
                    std::memcpy(&u, output.data() + i * 4, 4);
                    got = static_cast<double>(u);
                } else {
                    got = static_cast<double>(storedValue(out_dtype, output, i));
                }
                const double want = exclusive && step == 0 ? identityValue(out_dtype, op.kind == OpKind::Max) : extreme;
                if (got != want) {
                    check.ok = false;
                    break;
                }
            }
            if (exclusive) fold();
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
    std::function<void(Stream& stream)> run;
    const Tensor* arg_values = nullptr;  // tkv: the values written beside the indices
    uint64_t extra_bytes = 0;            // the bytes of those values
};

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

// The output dtypes timed for an op on a dtype: an arg scan writes UINT32 indices; a float sum also writes every other
// float (SegmentedScan.h); a min or max writes its own type.
std::vector<DataType> outputDTypes(const OpSpec& op, DataType dtype) {
    if (op.arg) return {DataType::UINT32};
    std::vector<DataType> outs{dtype};
    if (op.kind == OpKind::Sum) {
        for (const DataType out : {DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16, DataType::BF16, DataType::FP32}) {
            if (out != dtype && SegmentedScan::isSupported(SegmentedScanOp::Sum, dtype, out)) outs.push_back(out);
        }
    }
    return outs;
}

SegmentedScanOp segmentedScanOp(const OpSpec& op) {
    switch (op.kind) {
        case OpKind::Sum:
            return SegmentedScanOp::Sum;
        case OpKind::Min:
            return SegmentedScanOp::Min;
        case OpKind::Max:
            return SegmentedScanOp::Max;
        case OpKind::ArgMin:
            return SegmentedScanOp::ArgMin;
        case OpKind::ArgMax:
            return SegmentedScanOp::ArgMax;
        case OpKind::MaxBackward:
        case OpKind::MinBackward:
            break;
    }
    throw std::invalid_argument("Unknown op.");
}

// Thor's scan kernel over dense rows of n / rows elements, or ragged rows by the offsets; an arg scan given
// arg_values also writes its values there.
Path makeSegmentedScanPath(const std::string& name,
                           bool ragged,
                           const OpSpec& op,
                           SegmentedScanMode mode,
                           SegmentedScanDirection direction,
                           const Tensor& input,
                           Tensor& output,
                           const Tensor& offsets,
                           uint64_t n,
                           uint64_t rows,
                           const Tensor* arg_values = nullptr) {
    const std::optional<Tensor> values = arg_values != nullptr ? std::optional<Tensor>(*arg_values) : std::nullopt;
    const SegmentedScanOp scan_op = segmentedScanOp(op);
    std::shared_ptr<const SegmentedScan> scan =
        ragged ? std::make_shared<const SegmentedScan>(scan_op, mode, direction, input, output, offsets, values)
               : std::make_shared<const SegmentedScan>(scan_op, mode, direction, input, output, n / rows, values);
    Path path;
    path.name = name;
    path.run = [scan](Stream& stream) { scan->runOn(stream); };
    if (arg_values != nullptr) {
        path.arg_values = arg_values;
        path.extra_bytes = n * TensorDescriptor::getElementSizeInBytes(input.getDataType());
    }
    return path;
}

// The backward of a max or min scan by its definition: each element's dy goes to the winner of its prefix, walked in
// the scan's direction over rows [bounds[r], bounds[r + 1]). The winner is the arg scan's: replaced only by a strictly
// better element, so the first of equal elements keeps it; a NaN is below every number for a max and for a min, so
// it never replaces a winner and any number replaces it. An exclusive scan's first element in a row has no winner,
// and its dy goes nowhere.
std::vector<float> referenceBackward(const std::vector<float>& x,
                                     const std::vector<float>& dy,
                                     const std::vector<uint64_t>& bounds,
                                     bool max,
                                     bool inclusive,
                                     bool reverse) {
    std::vector<double> dx(x.size(), 0.0);
    for (uint64_t r = 0; r + 1 < bounds.size(); ++r) {
        const uint64_t begin = bounds[r];
        const uint64_t end = bounds[r + 1];
        bool has_winner = false;
        uint64_t winner = 0;
        for (uint64_t step = 0; step < end - begin; ++step) {
            const uint64_t i = reverse ? end - 1 - step : begin + step;
            if (!inclusive && has_winner) dx[winner] += dy[i];
            const bool replaces =
                !std::isnan(x[i]) && (std::isnan(x[winner]) || (max ? x[i] > x[winner] : x[i] < x[winner]));
            if (!has_winner || replaces) {
                winner = i;
                has_winner = true;
            }
            if (inclusive) dx[winner] += dy[i];
        }
    }
    return std::vector<float>(dx.begin(), dx.end());
}

// An FP32 value as one element of dtype, by the host's conversions (CUDA's headers): to nearest, FP8 saturating to
// its finite range. What the fused kernel must store for an FP32 sum.
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

// dy in dtype: multiples of 1/4 in [-4, 4] rounded to the dtype (only E5M2 moves any: its step is 1/2 above 2), with
// zero mean, so every partial sum of a 64M-element row stays far below 2^22 and is exact in FP32 in any order.
// `values` holds each element exactly as stored.
HostInput makeGradient(DataType dtype, uint64_t elements) {
    const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    float value_of[33];
    uint8_t bytes_of[33][4] = {};
    for (uint32_t k = 0; k < 33; ++k) {
        encodeFloat(dtype, (static_cast<float>(k) - 16.0f) * 0.25f, bytes_of[k]);
        value_of[k] = storedValue(dtype, std::vector<uint8_t>(bytes_of[k], bytes_of[k] + element_bytes), 0);
    }
    HostInput dy;
    dy.values.resize(elements);
    dy.bytes.resize(elements * element_bytes);
    uint64_t state = 0xD1B54A32D192ED03ULL;
    for (uint64_t i = 0; i < elements; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const uint32_t k = static_cast<uint32_t>((state >> 40) % 33ULL);
        dy.values[i] = value_of[k];
        std::memcpy(dy.bytes.data() + i * element_bytes, bytes_of[k], element_bytes);
    }
    return dy;
}

struct BackwardContext {
    const Options& options;
    const Timer& timer;
    Stream& stream;
    std::ofstream& csv;
    double copy_tbps = 0.0;
    uint64_t n = 0;
    const HostInput& host;
    const Tensor& input;  // x, [n]
    const Tensor& dy;     // [n]
    const HostInput& host_dy;
    Tensor& fused_dx;  // [n]: the fused kernel's output, in the dtype asked for
};

// One backward cell's result: its time, and its dx against the definition and against a second run.
struct BackwardCell {
    double ms = 0.0;
    bool ok = true;
    bool reproducible = true;
    uint64_t wrong = 0;
    uint64_t first_wrong = 0;
    double max_error = 0.0;
};

// What a path's cells came to over a table.
struct BackwardSummary {
    std::string name;
    uint64_t cells = 0;
    uint64_t failed = 0;
    uint64_t not_reproducible = 0;
    uint64_t wrong = 0;        // in the first failing cell
    uint64_t first_wrong = 0;  // and its first wrong element
};

// Elements [0, count) of `got` (dtype) against the FP32 sums `want` converted to dtype, byte for byte. Most sums are
// +0 (an element that never wins), which is all zero bytes in every format.
void compareBackward(DataType dtype,
                     const std::vector<uint8_t>& got,
                     const std::vector<float>& want,
                     uint64_t count,
                     uint64_t& wrong,
                     uint64_t& first_wrong,
                     double& max_error) {
    const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    const uint8_t zero[4] = {};
    for (uint64_t i = 0; i < count; ++i) {
        uint8_t expected[4] = {};
        uint32_t want_bits;
        std::memcpy(&want_bits, &want[i], 4);
        if (want_bits != 0U) encodeFloat(dtype, want[i], expected);
        if (std::memcmp(got.data() + i * element_bytes, want_bits != 0U ? expected : zero, element_bytes) != 0) {
            if (wrong++ == 0) first_wrong = i;
            const double stored = storedValue(dtype, std::vector<uint8_t>(expected, expected + element_bytes), 0);
            max_error = std::max(max_error, std::fabs(static_cast<double>(storedValue(dtype, got, i)) - stored));
        }
    }
}

void checkBackward(const BackwardContext& context,
                   const Tensor& dx,
                   const std::vector<float>& want,
                   const std::function<void()>& run_again,
                   BackwardCell& cell) {
    const uint64_t n = context.n;
    const DataType dtype = dx.getDataType();
    const uint64_t bytes = n * TensorDescriptor::getElementSizeInBytes(dtype);
    std::vector<uint8_t> got(bytes);
    CUDA_CHECK(cudaMemcpy(got.data(), dx.getMemPtr<void>(), bytes, cudaMemcpyDeviceToHost));
    compareBackward(dtype, got, want, n, cell.wrong, cell.first_wrong, cell.max_error);
    cell.ok = cell.wrong == 0;
    run_again();
    context.stream.synchronize();
    std::vector<uint8_t> again(bytes);
    CUDA_CHECK(cudaMemcpy(again.data(), dx.getMemPtr<void>(), bytes, cudaMemcpyDeviceToHost));
    cell.reproducible = std::memcmp(again.data(), got.data(), bytes) == 0;
}

// The offsets of `rows` equal rows of n elements, on the GPU.
Tensor makeEqualOffsets(const TensorPlacement& gpu, uint64_t n, uint64_t rows) {
    std::vector<uint32_t> host_offsets(rows + 1);
    for (uint64_t r = 0; r <= rows; ++r) host_offsets[r] = static_cast<uint32_t>(r * (n / rows));
    Tensor offsets(gpu, TensorDescriptor(DataType::UINT32, {rows + 1}));
    CUDA_CHECK(cudaMemcpy(
        offsets.getMemPtr<void>(), host_offsets.data(), host_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return offsets;
}

// The backward on these rows (fu dense, fur ragged).
BackwardCell measureFused(const BackwardContext& context,
                          const OpSpec& op,
                          SegmentedScanMode mode,
                          SegmentedScanDirection direction,
                          uint64_t rows,
                          bool ragged,
                          const std::vector<float>& want) {
    const uint64_t n = context.n;
    const TensorPlacement gpu = context.input.getPlacement();
    Tensor offsets;
    if (ragged) offsets = makeEqualOffsets(gpu, n, rows);
    const bool min = op.kind == OpKind::MinBackward;
    const std::unique_ptr<const ScanMinMaxBackward> backward =
        ragged ? std::make_unique<const ScanMinMaxBackward>(
                     min, mode, direction, context.input, context.dy, context.fused_dx, offsets)
               : std::make_unique<const ScanMinMaxBackward>(
                     min, mode, direction, context.input, context.dy, context.fused_dx, n / rows);
    BackwardCell cell;
    cell.ms = context.timer.medianMs([&]() { backward->runOn(context.stream); });
    if (context.options.verify) checkBackward(context, context.fused_dx, want, [&]() { backward->runOn(context.stream); }, cell);
    return cell;
}

// The backward tables of one op for the context's dtypes of x, dy and dx, in every mode and direction: one row of n,
// then every row count, on the fused kernel (fu dense, fur ragged). Returns false if a check failed.
bool runBackwardTables(const BackwardContext& context,
                       const OpSpec& op,
                       const std::vector<std::pair<SegmentedScanMode, SegmentedScanDirection>>& variants) {
    const uint64_t n = context.n;
    const DataType x_dtype = context.input.getDataType();
    const DataType dy_dtype = context.dy.getDataType();
    const DataType dx_dtype = context.fused_dx.getDataType();
    const uint64_t moved = n * (TensorDescriptor::getElementSizeInBytes(x_dtype) + TensorDescriptor::getElementSizeInBytes(dy_dtype) +
                                TensorDescriptor::getElementSizeInBytes(dx_dtype));
    bool all_ok = true;
    for (const auto& [mode, direction] : variants) {
        const bool max = op.kind == OpKind::MaxBackward;
        const bool inclusive = !isExclusive(mode);
        const bool reverse = direction == SegmentedScanDirection::Reverse;
        std::vector<BackwardSummary> summaries{{"fu"}, {"fur"}};
        auto record = [&](size_t path, const BackwardCell& cell, uint64_t rows) {
            BackwardSummary& summary = summaries[path];
            ++summary.cells;
            if (!cell.ok && summary.failed++ == 0) {
                summary.wrong = cell.wrong;
                summary.first_wrong = cell.first_wrong;
            }
            if (!cell.reproducible) ++summary.not_reproducible;
            const double tbps = terabytesPerSecond(moved, cell.ms);
            const double pct = context.copy_tbps > 0.0 ? 100.0 * tbps / context.copy_tbps : 0.0;
            if (context.csv) {
                std::string check = "-";
                if (context.options.verify) {
                    check = cell.ok ? "ok"
                                    : "FAILED (" + std::to_string(cell.wrong) + " wrong, first at " +
                                          std::to_string(cell.first_wrong) + ")";
                    if (!cell.reproducible) check += " NOT reproducible";
                }
                context.csv << dtypeName(x_dtype) << ',' << dtypeName(dx_dtype) << ',' << op.name << ',' << modeName(mode) << ','
                            << directionName(direction) << ',' << rows << ',' << n / rows << ',' << summary.name << ','
                            << 0 << ',' << cell.ms << ',' << tbps << ',' << pct << ',' << check << ','
                            << cell.max_error << ',' << dtypeName(dy_dtype) << '\n';
            }
            std::ostringstream text;
            text << std::fixed << std::setprecision(3) << std::setw(8) << tbps << std::setprecision(0) << std::setw(5)
                 << pct << '%' << std::setprecision(3) << std::setw(8) << cell.ms << " ms";
            return text.str();
        };
        auto row = [&](uint64_t rows, bool print_header) {
            std::vector<uint64_t> bounds(rows + 1);
            for (uint64_t r = 0; r <= rows; ++r) bounds[r] = r * (n / rows);
            std::vector<float> want;
            if (context.options.verify) {
                want = referenceBackward(context.host.values, context.host_dy.values, bounds, max, inclusive, reverse);
            }
            std::vector<BackwardCell> cells;
            cells.push_back(measureFused(context, op, mode, direction, rows, false, want));
            cells.push_back(measureFused(context, op, mode, direction, rows, true, want));
            if (print_header) {
                std::cout << std::setw(9) << "rows" << std::setw(10) << "cols";
                for (const BackwardSummary& summary : summaries) std::cout << std::setw(25) << summary.name;
                std::cout << '\n';
            }
            std::cout << std::setw(9) << rows << std::setw(10) << n / rows;
            for (size_t path = 0; path < cells.size(); ++path) std::cout << record(path, cells[path], rows);
            std::cout << '\n';
        };
        std::cout << "\n" << dtypeName(x_dtype) << ' ' << op.name << (isExclusive(mode) ? " exclusive" : "") << ' '
                  << directionName(direction) << ", dy " << dtypeName(dy_dtype) << ", dx " << dtypeName(dx_dtype) << " (TB/s of "
                  << std::fixed << std::setprecision(0) << static_cast<double>(moved) / MIB << " MiB: x, dy and dx; % of copy; ms). "
                  << "fu, fur: the fused kernel, dense and ragged rows:\n";
        bool first = true;
        for (const uint64_t rows : context.options.rows) {
            row(rows, first);
            first = false;
        }
        if (context.options.verify) {
            std::cout << "  checks:";
            for (const BackwardSummary& summary : summaries) {
                std::cout << ' ' << summary.name << ' ';
                if (summary.failed == 0) {
                    std::cout << "ok";
                } else {
                    std::cout << "FAILED in " << summary.failed << " of " << summary.cells << " cells (first: "
                              << summary.wrong << " wrong, the first at element " << summary.first_wrong << ")";
                    all_ok = false;
                }
                if (summary.not_reproducible != 0) {
                    std::cout << ", NOT reproducible in " << summary.not_reproducible;
                    all_ok = false;
                }
                std::cout << ';';
            }
            std::cout << '\n';
        }
    }
    return all_ok;
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

        std::ofstream csv;
        if (!options.csv.empty()) {
            csv.open(options.csv);
            if (!csv) throw std::runtime_error("Cannot open --csv file '" + options.csv + "'.");
            csv << "dtype,out_dtype,op,mode,direction,rows,cols,path,production,p50_ms,tbps,pct_of_copy,check,"
                   "max_error,dy_dtype\n";
        }
#ifdef THOR_DEBUG
        std::cout << "WARNING: Thor was built with THOR_DEBUG. Use a Release or RelWithDebInfo build for timings.\n";
#endif
        std::cout << "Thor scan benchmark (final axis)\n"
                  << "GPU: " << properties.name << " (device " << options.gpu << ", " << properties.multiProcessorCount
                  << " SMs, L2 " << std::fixed << std::setprecision(1) << static_cast<double>(l2_bytes) / MIB
                  << " MiB)\nElements per tensor: " << n << "; L2 scrubbed before every timed run; median of "
                  << options.iterations << " runs.\nEach cell: TB/s and % of the input dtype's device-to-device copy. "
                  << "Checks: per path under each table.\n";

        // Each op runs in every mode and direction asked for.
        std::vector<std::pair<SegmentedScanMode, SegmentedScanDirection>> variants;
        for (const SegmentedScanMode mode : options.modes) {
            for (const SegmentedScanDirection direction : options.directions) variants.emplace_back(mode, direction);
        }
        bool any_failed = false;
        std::vector<uint8_t> host_output;
        std::vector<uint8_t> second_output;
        for (const DataType dtype : options.dtypes) {
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
                      << " TB/s\n";

            for (const OpSpec& op : options.ops) {
                if (op.backward) {
                    if (dtype == DataType::UINT32) {
                        std::cout << "\n" << dtypeName(dtype) << ' ' << op.name << ": skipped (the backward takes float x)\n";
                        continue;
                    }
                    std::vector<std::pair<DataType, DataType>> done;  // pairs with x name the same dtypes for some x
                    for (const GradientDTypes& grads : options.grads) {
                        const DataType dy_dtype = grads.dy.value_or(dtype);
                        const DataType dx_dtype = grads.dx.value_or(dtype);
                        if (std::find(done.begin(), done.end(), std::pair{dy_dtype, dx_dtype}) != done.end()) continue;
                        done.emplace_back(dy_dtype, dx_dtype);
                        const HostInput host_dy = makeGradient(dy_dtype, n);
                        Tensor dy(gpu, TensorDescriptor(dy_dtype, {n}));
                        CUDA_CHECK(cudaMemcpy(dy.getMemPtr<void>(), host_dy.bytes.data(), host_dy.bytes.size(), cudaMemcpyHostToDevice));
                        Tensor fused_dx(gpu, TensorDescriptor(dx_dtype, {n}));
                        const BackwardContext context{options, timer, stream, csv, copy_tbps, n, host, input, dy, host_dy, fused_dx};
                        if (!runBackwardTables(context, op, variants)) any_failed = true;
                    }
                    continue;
                }
                for (const std::pair<SegmentedScanMode, SegmentedScanDirection>& variant : variants) {
                    const SegmentedScanMode mode = variant.first;
                    const SegmentedScanDirection direction = variant.second;
                    // The tables' titles name the mode only for exclusive scans, so inclusive ones read as before.
                    const std::string mode_text = isExclusive(mode) ? " exclusive" : "";
                    for (const DataType out_dtype : outputDTypes(op, dtype)) {
                        const uint64_t out_bytes = TensorDescriptor::getElementSizeInBytes(out_dtype);
                        Tensor values_out(gpu, TensorDescriptor(op.arg ? dtype : out_dtype, {n}));
                        Tensor& output = op.arg ? indices_out : values_out;
                        Tensor arg_values_out = op.arg ? Tensor(gpu, TensorDescriptor(dtype, {n})) : Tensor();  // tkv
                        const uint64_t moved = n * element_bytes + n * out_bytes;
                        // Thor's kernel for this cell's op, mode and direction (makeSegmentedScanPath).
                        auto thorPath = [&](const std::string& name,
                                            bool ragged,
                                            const Tensor& offsets,
                                            uint64_t rows,
                                            const Tensor* arg_values = nullptr) {
                            return makeSegmentedScanPath(
                                name, ragged, op, mode, direction, input, output, offsets, n, rows, arg_values);
                        };

                        // Per path over the table: the largest error and any failed check.
                        struct Summary {
                            std::string name;
                            bool judged = true;
                            bool failed = false;
                            double max_error = 0.0;
                            bool not_reproducible = false;
                        };
                        std::vector<Summary> summaries;
                        auto summarize = [&](const std::string& name, const Check& check, bool reproducible) {
                            auto it = std::find_if(
                                summaries.begin(), summaries.end(), [&](const Summary& s) { return s.name == name; });
                            if (it == summaries.end()) {
                                summaries.push_back(Summary{name});
                                it = summaries.end() - 1;
                            }
                            it->judged = check.judged;
                            it->failed = it->failed || (check.judged && !check.ok);
                            it->max_error = std::max(it->max_error, check.max_error);
                            it->not_reproducible = it->not_reproducible || !reproducible;
                        };
                        // production: StampedScan's path for these dense rows (tk).
                        auto measure = [&](Path& path, uint64_t rows, bool production) {
                            const double ms = timer.medianMs([&]() { path.run(stream); });
                            Check check;
                            if (options.verify) {
                                host_output.resize(n * out_bytes);
                                CUDA_CHECK(cudaMemcpy(host_output.data(),
                                                      output.getMemPtr<void>(),
                                                      host_output.size(),
                                                      cudaMemcpyDeviceToHost));
                                check = checkOutput(
                                    dtype, out_dtype, op, direction, mode, rows, n / rows, host, host_output);
                                if (path.arg_values != nullptr && check.ok) {  // the values at the checked indices
                                    std::vector<uint8_t> got(n * element_bytes);
                                    CUDA_CHECK(cudaMemcpy(got.data(),
                                                          path.arg_values->getMemPtr<void>(),
                                                          got.size(),
                                                          cudaMemcpyDeviceToHost));
                                    const std::vector<uint8_t> identity =
                                        identityBytes(dtype, op.kind == OpKind::ArgMax);
                                    for (uint64_t i = 0; i < n && check.ok; ++i) {
                                        uint32_t index;
                                        std::memcpy(&index, host_output.data() + 4 * i, 4);
                                        const uint8_t* want = index == 0xFFFFFFFFU
                                                                  ? identity.data()
                                                                  : host.bytes.data() + index * element_bytes;
                                        check.ok =
                                            std::memcmp(got.data() + i * element_bytes, want, element_bytes) == 0;
                                    }
                                }
                                any_failed = any_failed || (check.judged && !check.ok);
                                // Reproducibility: one more run, compared byte for byte.
                                path.run(stream);
                                stream.synchronize();
                                second_output.resize(host_output.size());
                                CUDA_CHECK(cudaMemcpy(second_output.data(),
                                                      output.getMemPtr<void>(),
                                                      second_output.size(),
                                                      cudaMemcpyDeviceToHost));
                                const bool reproducible =
                                    std::memcmp(second_output.data(), host_output.data(), host_output.size()) == 0;
                                summarize(path.name, check, reproducible);
                            }
                            const double tbps = terabytesPerSecond(moved + path.extra_bytes, ms);
                            const double pct = copy_tbps > 0.0 ? 100.0 * tbps / copy_tbps : 0.0;
                            if (csv) {
                                csv << dtypeName(dtype) << ',' << dtypeName(out_dtype) << ',' << op.name << ','
                                    << modeName(mode) << ',' << directionName(direction) << ',' << rows << ','
                                    << n / rows << ',' << path.name << ',' << (production ? 1 : 0) << ',' << ms << ','
                                    << tbps << ',' << pct << ',' << checkText(check, options.verify) << ','
                                    << check.max_error << ",-\n";
                            }
                            return std::pair<double, double>{tbps, pct};
                        };
                        auto cell = [](std::pair<double, double> result) {
                            std::ostringstream text;
                            text << std::fixed << std::setprecision(3) << std::setw(7) << result.first
                                 << std::setprecision(0) << std::setw(5) << result.second << '%';
                            return text.str();
                        };

                        const Tensor no_offsets;
                        std::cout << "\n" << dtypeName(dtype) << ' ' << op.name;
                        if (!op.arg && out_dtype != dtype) std::cout << " -> " << dtypeName(out_dtype);
                        std::cout << mode_text << ' ' << directionName(direction) << " (TB/s of "
                                  << std::setprecision(0) << static_cast<double>(moved) / MIB
                                  << " MiB moved), one row of " << n << " elements:\n";
                        Path head = thorPath("tk", false, no_offsets, 1);
                        std::cout << "  " << std::left << std::setw(6) << head.name << std::right
                                  << cell(measure(head, 1, false)) << '\n';

                        std::vector<std::string> columns{"tk", "tkr"};
                        if (op.arg) columns.push_back("tkv");
                        std::cout << std::right << std::setw(9) << "rows" << std::setw(10) << "cols";
                        for (const std::string& column : columns) std::cout << std::setw(14) << column;
                        std::cout << '\n';

                        for (const uint64_t rows : options.rows) {
                            const uint64_t cols = n / rows;
                            std::vector<uint32_t> host_offsets(rows + 1);
                            for (uint64_t r = 0; r <= rows; ++r) host_offsets[r] = static_cast<uint32_t>(r * cols);
                            Tensor offsets(gpu, TensorDescriptor(DataType::UINT32, {rows + 1}));
                            CUDA_CHECK(cudaMemcpy(offsets.getMemPtr<void>(),
                                                  host_offsets.data(),
                                                  host_offsets.size() * sizeof(uint32_t),
                                                  cudaMemcpyHostToDevice));

                            std::vector<Path> paths;
                            paths.push_back(thorPath("tk", false, offsets, rows));
                            paths.push_back(thorPath("tkr", true, offsets, rows));
                            if (op.arg) paths.push_back(thorPath("tkv", false, offsets, rows, &arg_values_out));

                            std::cout << std::right << std::setw(9) << rows << std::setw(10) << cols;
                            for (size_t i = 0; i < paths.size(); ++i) {
                                std::cout << "  " << cell(measure(paths[i], rows, i == 0));
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
                                if (s.not_reproducible) std::cout << " NOT reproducible";
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
