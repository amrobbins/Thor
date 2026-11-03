// 1168: randomized end-to-end tests of multi-step original-FP8 reductions (plans of two or more passes).
//
// Every run covers every public VALUE operation (Sum, Mean, Product, Min, Max, L1Norm, L2Norm, SumSquares) and both
// ARG operations (argmin, argmax), each on THOR_FP8_MULTISTEP_CASES_PER_OP (default 2) random cases. A case draws:
//   - a shape whose canonical problem has at least two reduced runs (so the plan has at least two passes), from four
//     families: a random rank-3..5 mask (singletons included), deep-R small-K with a second reduced axis (where staged
//     RK passes are chosen), few long rows with a reduced outer axis, and a reduced outer axis with a wide inner run;
//   - the FP8 format (E4M3 / E5M2), the stamp's tuning depth (Quick / Regular / Ultra) and, for ARG, the outputs
//     (value + UINT32 index, value + UINT64 index, index only, value only);
//   - FP8 data for the operation (finite values for VALUE; for ARG, ties, +-0 and, in some cases, a few NaNs).
// Results are checked against a host reference on the same decoded values: exact for Product (powers of two only),
// Min, Max and ARG (values and indices, NaN and -0/+0 rules of Fp8ArgReference), and within an FP32 summation bound for
// the additive operations.
//
// Coverage grows run to run: the seed is random unless THOR_FP8_MULTISTEP_SEED is set, and it is printed, with every
// failure carrying `THOR_FP8_MULTISTEP_SEED=<seed> THOR_FP8_MULTISTEP_CASE=<case>`, which reruns exactly that case.
// Each run prints a coverage line (cases with a staged pass, depths, pass counts).

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Reference = ThorImplementation::Fp8ArgReference;
namespace Internal = ThorImplementation::CubReductionInternal;

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

constexpr uint64_t MaxElements = uint64_t{1} << 21;  // 2 MiB of FP8 per case keeps the whole test to a few seconds

std::optional<uint64_t> envU64(const char* name) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') return std::nullopt;
    return std::stoull(text);
}

uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t logUniform(std::mt19937_64& rng, uint64_t low, uint64_t high) {
    if (high <= low) return low;
    const double l = std::log2(static_cast<double>(low)), h = std::log2(static_cast<double>(high));
    const double v = l + (h - l) * std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    return std::clamp<uint64_t>(static_cast<uint64_t>(std::llround(std::exp2(v))), low, high);
}

uint64_t product(const std::vector<uint64_t>& dims) {
    uint64_t p = 1;
    for (uint64_t d : dims) p *= d;
    return p;
}

size_t reducedRuns(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes) {
    const auto problem = Internal::makeInitialDenseReductionProblem(dims, axes, Internal::DenseReductionAggregateKind::Value,
                                                                    Internal::DenseReductionOutputSpec{});
    return static_cast<size_t>(std::count_if(problem.runs.begin(), problem.runs.end(), [](const auto& run) {
        return run.kind == Internal::DenseReductionRunKind::OriginalReduction;
    }));
}

struct Shape {
    std::vector<uint64_t> dims;
    std::vector<uint32_t> axes;
    const char* family = "";
};

// A shape with at least two reduced runs after canonicalization and at most MaxElements elements.
Shape drawShape(std::mt19937_64& rng) {
    const uint64_t family = rng() % 4;  // drawn once, so a family that often needs redraws keeps its share
    for (;;) {
        Shape s;
        switch (family) {
            case 0: {  // random mask
                s.family = "mask";
                const size_t rank = 3 + rng() % 3;
                for (size_t d = 0; d < rank; ++d) s.dims.push_back(rng() % 4 == 0 ? 1 : logUniform(rng, 2, 96));
                for (uint32_t d = 0; d < rank; ++d)
                    if (rng() % 2) s.axes.push_back(d);
                break;
            }
            case 1: {  // deep R, small K, and a second reduced axis: [O, R, K, T], axes {1, 3}
                s.family = "deep_r";
                const uint64_t o = 1 + rng() % 4, k = 2 + rng() % 7, t = 2 + rng() % 15;
                const uint64_t r = logUniform(rng, 1024, std::max<uint64_t>(1024, MaxElements / (o * k * t)));
                s.dims = {o, r, k, t};
                s.axes = {1, 3};
                break;
            }
            case 2: {  // few long rows and a reduced outer axis: [A, B, C], axes {0, 2}
                s.family = "long_rows";
                const uint64_t a = 2 + rng() % 31, b = 2 + rng() % 3;
                const uint64_t c = logUniform(rng, 4096, std::max<uint64_t>(4096, MaxElements / (a * b)));
                s.dims = {a, b, c};
                s.axes = {0, 2};
                break;
            }
            default: {  // reduced outer axis over a wide inner run: [R, K, C], axes {0, 2}
                s.family = "wide_inner";
                const uint64_t k = 2 + rng() % 63, c = logUniform(rng, 2, 256);
                const uint64_t r = logUniform(rng, 256, std::max<uint64_t>(256, MaxElements / (k * c)));
                s.dims = {r, k, c};
                s.axes = {0, 2};
                break;
            }
        }
        if (s.axes.empty()) continue;
        const uint64_t elements = product(s.dims);
        if (elements < 16 || elements > MaxElements) continue;
        if (reducedRuns(s.dims, s.axes) >= 2) return s;
    }
}

std::string describe(const Shape& s) {
    std::ostringstream out;
    out << s.family << " dims=[";
    for (size_t i = 0; i < s.dims.size(); ++i) out << (i ? "," : "") << s.dims[i];
    out << "] axes={";
    for (size_t i = 0; i < s.axes.size(); ++i) out << (i ? "," : "") << s.axes[i];
    out << '}';
    return out.str();
}

// Row-major mapping of every input element to its output and its flattened index over the reduced axes.
struct Mapping {
    std::vector<uint32_t> output;  // per input element
    std::vector<uint64_t> index;   // per input element (ARG)
    uint64_t outputs = 1, reduced = 1;
};

Mapping mapping(const Shape& s) {
    const size_t rank = s.dims.size();
    std::vector<bool> reduced(rank, false);
    for (uint32_t axis : s.axes) reduced[axis] = true;
    Mapping m;
    for (size_t d = 0; d < rank; ++d) (reduced[d] ? m.reduced : m.outputs) *= s.dims[d];
    const uint64_t elements = product(s.dims);
    m.output.resize(elements);
    m.index.resize(elements);
    std::vector<uint64_t> coordinate(rank, 0);
    for (uint64_t linear = 0; linear < elements; ++linear) {
        uint64_t output = 0, index = 0;
        for (size_t d = 0; d < rank; ++d) {
            if (reduced[d]) index = index * s.dims[d] + coordinate[d];
            else output = output * s.dims[d] + coordinate[d];
        }
        m.output[linear] = static_cast<uint32_t>(output);
        m.index[linear] = index;
        for (size_t d = rank; d-- > 0;) {
            if (++coordinate[d] < s.dims[d]) break;
            coordinate[d] = 0;
        }
    }
    return m;
}

enum class Data { Additive, Product, Extremum, Arg };

// FP8 bytes for one operation's data, decoded to floats (exactly representable, so the FP8 tensor holds them).
std::vector<float> drawValues(std::mt19937_64& rng, uint64_t count, bool e5m2, Data data, uint64_t reduced) {
    std::array<float, 256> table{};
    for (uint32_t b = 0; b < 256; ++b) table[b] = Reference::decode(b, e5m2);
    std::vector<uint8_t> additive, extremum;
    std::array<int, 6> powers{{-1, -1, -1, -1, -1, -1}};  // bytes of +1, -1, +2, -2, +0.5, -0.5
    const float power_values[6] = {1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f};
    for (uint32_t b = 0; b < 256; ++b) {
        const float v = table[b];
        if (!std::isfinite(v)) continue;
        if (std::fabs(v) <= 4.0f) additive.push_back(static_cast<uint8_t>(b));
        if (std::fabs(v) <= 8.0f) extremum.push_back(static_cast<uint8_t>(b));
        for (int p = 0; p < 6; ++p)
            if (v == power_values[p] && powers[p] < 0) powers[p] = static_cast<int>(b);
    }
    std::vector<float> values(count);
    const double rare = std::min(1.0, 2.0 / static_cast<double>(reduced));  // ~2 non-unit factors per domain
    const bool with_nans = data == Data::Arg && rng() % 4 == 0;
    for (uint64_t i = 0; i < count; ++i) {
        switch (data) {
            case Data::Additive: {
                // Mostly positive (a nonzero mean makes a lost or doubled partial visible), zeros included.
                uint8_t byte = additive[rng() % additive.size()];
                if (table[byte] < 0.0f && rng() % 3 != 0) byte = static_cast<uint8_t>(byte ^ 0x80);
                values[i] = table[byte];
                break;
            }
            case Data::Product: {
                const bool non_unit = std::uniform_real_distribution<double>(0.0, 1.0)(rng) < rare;
                values[i] = table[powers[(non_unit ? 2 + rng() % 4 : rng() % 2)]];
                break;
            }
            case Data::Extremum:
            case Data::Arg:
                values[i] = table[extremum[rng() % extremum.size()]];
                break;
        }
    }
    if (with_nans) {
        constexpr uint8_t NanByte = 0x7F;  // a NaN in both formats (E4M3 S.1111.111, E5M2 S.11111.11)
        for (int n = 0; n < 3; ++n) values[rng() % count] = table[NanByte];
    }
    return values;
}

const char* depthName(CubReductionTuningDepth depth) {
    return depth == CubReductionTuningDepth::Quick ? "quick" : depth == CubReductionTuningDepth::Ultra ? "ultra" : "regular";
}

const char* valueOpName(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum: return "sum";
        case CubReductionOp::Mean: return "mean";
        case CubReductionOp::Product: return "product";
        case CubReductionOp::Min: return "min";
        case CubReductionOp::Max: return "max";
        case CubReductionOp::L1Norm: return "l1_norm";
        case CubReductionOp::L2Norm: return "l2_norm";
        case CubReductionOp::SumSquares: return "sum_squares";
    }
    return "?";
}

struct Coverage {
    uint64_t cases = 0, staged = 0;
    std::map<std::string, uint64_t> depths, families, passes;
};

// Host reference for one VALUE operation: the result and, for additive operations, the allowed error.
struct ValueReference {
    std::vector<double> value;
    std::vector<double> tolerance;  // 0: exact
};

ValueReference valueReference(CubReductionOp op, const std::vector<float>& x, const Mapping& m) {
    ValueReference ref;
    ref.value.assign(m.outputs, 0.0);
    ref.tolerance.assign(m.outputs, 0.0);
    std::vector<double> magnitude(m.outputs, 0.0);
    if (op == CubReductionOp::Product) std::fill(ref.value.begin(), ref.value.end(), 1.0);
    if (op == CubReductionOp::Min) std::fill(ref.value.begin(), ref.value.end(), std::numeric_limits<double>::infinity());
    if (op == CubReductionOp::Max) std::fill(ref.value.begin(), ref.value.end(), -std::numeric_limits<double>::infinity());
    for (uint64_t i = 0; i < x.size(); ++i) {
        const double v = x[i];
        double& r = ref.value[m.output[i]];
        switch (op) {
            case CubReductionOp::Sum:
            case CubReductionOp::Mean: r += v; magnitude[m.output[i]] += std::fabs(v); break;
            case CubReductionOp::L1Norm: r += std::fabs(v); magnitude[m.output[i]] += std::fabs(v); break;
            case CubReductionOp::L2Norm:
            case CubReductionOp::SumSquares: r += v * v; magnitude[m.output[i]] += v * v; break;
            case CubReductionOp::Product: r *= v; break;
            case CubReductionOp::Min: r = std::min(r, v); break;
            case CubReductionOp::Max: r = std::max(r, v); break;
        }
    }
    // FP32 accumulation in any order: a bound of a few ulps of the summed magnitude per tree level, generously.
    const double n = static_cast<double>(m.reduced);
    const double bound = 64.0 * std::numeric_limits<float>::epsilon() * std::max(1.0, std::log2(n));
    for (uint64_t o = 0; o < m.outputs; ++o) {
        switch (op) {
            case CubReductionOp::Sum:
            case CubReductionOp::L1Norm:
            case CubReductionOp::SumSquares:
                ref.tolerance[o] = bound * magnitude[o] + 1e-6;
                break;
            case CubReductionOp::Mean:
                ref.value[o] /= n;
                ref.tolerance[o] = (bound * magnitude[o] + 1e-6) / n;
                break;
            case CubReductionOp::L2Norm:
                ref.tolerance[o] = bound * magnitude[o] / (2.0 * std::max(std::sqrt(ref.value[o]), 1e-3)) + 1e-6;
                ref.value[o] = std::sqrt(ref.value[o]);
                break;
            default:
                break;  // exact: products of powers of two, and order-free min / max
        }
    }
    return ref;
}

}  // namespace

TEST(CubReductionFp8MultiStep, RandomShapesEveryOperationAndDepth) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint64_t seed = envU64("THOR_FP8_MULTISTEP_SEED").value_or(
        (static_cast<uint64_t>(std::random_device{}()) << 32) ^ std::random_device{}());
    const uint64_t per_op = envU64("THOR_FP8_MULTISTEP_CASES_PER_OP").value_or(2);
    const std::optional<uint64_t> only_case = envU64("THOR_FP8_MULTISTEP_CASE");
    std::cout << "[ fp8 multi-step ] THOR_FP8_MULTISTEP_SEED=" << seed << " (" << per_op
              << " cases per operation; set the seed to reproduce this run)" << std::endl;

    const std::array<CubReductionOp, 8> value_ops{{CubReductionOp::Sum, CubReductionOp::Mean, CubReductionOp::Product,
                                                   CubReductionOp::Min, CubReductionOp::Max, CubReductionOp::L1Norm,
                                                   CubReductionOp::L2Norm, CubReductionOp::SumSquares}};
    constexpr size_t OpCount = 10;  // 8 VALUE + argmin + argmax
    Coverage coverage;

    for (uint64_t case_index = 0; case_index < OpCount * per_op; ++case_index) {
        if (only_case && *only_case != case_index) continue;
        std::mt19937_64 rng(mix(seed ^ mix(case_index)));
        const size_t op_slot = case_index % OpCount;
        const bool is_arg = op_slot >= value_ops.size();
        const Shape shape = drawShape(rng);
        const bool e5m2 = rng() % 2;
        const DataType dtype = e5m2 ? DataType::FP8_E5M2 : DataType::FP8_E4M3;
        const CubReductionTuningDepth depth = std::array<CubReductionTuningDepth, 3>{
            CubReductionTuningDepth::Quick, CubReductionTuningDepth::Regular, CubReductionTuningDepth::Ultra}[rng() % 3];
        const Mapping m = mapping(shape);
        std::ostringstream what;
        what << "THOR_FP8_MULTISTEP_SEED=" << seed << " THOR_FP8_MULTISTEP_CASE=" << case_index << ": "
             << (is_arg ? (op_slot == 8 ? "argmin" : "argmax") : valueOpName(value_ops[op_slot])) << ' '
             << (e5m2 ? "e5m2" : "e4m3") << ' ' << depthName(depth) << ' ' << describe(shape);

        const Data data = is_arg ? Data::Arg
            : value_ops[op_slot] == CubReductionOp::Product ? Data::Product
            : (value_ops[op_slot] == CubReductionOp::Min || value_ops[op_slot] == CubReductionOp::Max) ? Data::Extremum
            : Data::Additive;
        const std::vector<float> generated = drawValues(rng, product(shape.dims), e5m2, data, m.reduced);
        Tensor input = makeGpuTensor(generated, shape.dims, stream, dtype);
        // The reference consumes exactly what the GPU holds.
        const std::vector<float> x = copyGpuTensorAsFloat(input, stream);
        size_t passes = 0;

        if (!is_arg) {
            const CubReductionOp op = value_ops[op_slot];
            auto stamped = CubReduction(op, shape.axes, DataType::FP32, 1.0f, depth).stamp(input, stream);
            passes = stamped->usesDenseExecutablePlan() ? stamped->getDensePhysicalPassDiagnostics().size() : 0;
            what << " passes=" << passes;
            SCOPED_TRACE(what.str());
            stamped->run();
            stream.synchronize();
            const std::vector<float> got = copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
            const ValueReference ref = valueReference(op, x, m);
            ASSERT_EQ(got.size(), ref.value.size());
            for (uint64_t o = 0; o < got.size(); ++o) {
                if (ref.tolerance[o] == 0.0) {
                    ASSERT_EQ(static_cast<double>(got[o]), ref.value[o]) << "output " << o;
                } else {
                    ASSERT_NEAR(static_cast<double>(got[o]), ref.value[o], ref.tolerance[o]) << "output " << o;
                }
            }
        } else {
            const bool maximum = op_slot == 9;
            CubArgReductionOutputOptions outputs;
            switch (rng() % 4) {
                case 0: outputs.value_output_dtype = DataType::FP32; outputs.index_output_dtype = DataType::UINT32; break;
                case 1: outputs.value_output_dtype = DataType::FP32; outputs.index_output_dtype = DataType::UINT64; break;
                case 2: outputs.produce_value = false; outputs.index_output_dtype = DataType::UINT32; break;
                default: outputs.produce_index = false; outputs.value_output_dtype = DataType::FP32; break;
            }
            auto stamped = CubArgReduction(maximum ? CubArgReductionOp::ArgMax : CubArgReductionOp::ArgMin, shape.axes,
                                           outputs, depth).stamp(input, stream);
            passes = stamped->getPlannedPassNames().size();
            what << " outputs=" << (outputs.produce_value ? "value" : "") << (outputs.produce_index ? "+index" : "")
                 << " passes=" << passes;
            SCOPED_TRACE(what.str());
            stamped->run();
            stream.synchronize();
            std::vector<Reference::Candidate> best(m.outputs);
            std::vector<bool> seen(m.outputs, false);
            for (uint64_t i = 0; i < x.size(); ++i) {
                const Reference::Candidate c{m.index[i], x[i]};
                best[m.output[i]] = seen[m.output[i]] ? Reference::combine(best[m.output[i]], c, maximum) : c;
                seen[m.output[i]] = true;
            }
            if (outputs.produce_value) {
                ASSERT_TRUE(stamped->getValueOutputTensor().has_value());
                const std::vector<float> got = copyGpuTensorAsFloat(stamped->getValueOutputTensor().value(), stream);
                ASSERT_EQ(got.size(), best.size());
                for (uint64_t o = 0; o < got.size(); ++o)
                    ASSERT_TRUE(Reference::sameValue(got[o], best[o].value))
                        << "output " << o << ": " << got[o] << " vs " << best[o].value;
            }
            if (outputs.produce_index) {
                ASSERT_TRUE(stamped->getIndexOutputTensor().has_value());
                const std::vector<uint64_t> got = copyGpuTensorAsUnsigned(stamped->getIndexOutputTensor().value(), stream);
                ASSERT_EQ(got.size(), best.size());
                for (uint64_t o = 0; o < got.size(); ++o) ASSERT_EQ(got[o], best[o].index) << "output " << o;
            }
        }
        // Every case is multi-step by construction; a plan with more passes than reduced runs had a staged pass.
        EXPECT_GE(passes, 2U) << what.str();
        ++coverage.cases;
        coverage.staged += passes > reducedRuns(shape.dims, shape.axes);
        ++coverage.depths[depthName(depth)];
        ++coverage.families[shape.family];
        ++coverage.passes[std::to_string(passes)];
    }

    std::ostringstream summary;
    summary << "[ fp8 multi-step ] seed=" << seed << " cases=" << coverage.cases << " with_staged_pass=" << coverage.staged;
    for (const auto& [name, count] : coverage.families) summary << ' ' << name << '=' << count;
    for (const auto& [name, count] : coverage.depths) summary << ' ' << name << '=' << count;
    summary << " passes:";
    for (const auto& [name, count] : coverage.passes) summary << ' ' << name << 'x' << count;
    std::cout << summary.str() << std::endl;
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES
