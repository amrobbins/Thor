// 1227 (phase 9): the routing is unconditional; the switch and RoutingIsOnByDefault are gone.
// 1222: and misaligned FP32 input views (MisalignedInputsUseTolerantPasses).
// 1221: and ARG (RoutedArgPlansUseRetainedFp32Passes, RandomMultiPassArgPlans): every FP32 ARG site, original or a
// carried-index intermediate of FP32 and FP8 plans, runs a retained FP32 ARG pass in the planned ARG executor.
// 1220: the retained FP32 routing (plans/Thor_FP32_Dense_Reduction_Plan, phase 8), VALUE: every FP32 VALUE site
// (original FP32 input, and the FP32 intermediates of FP32 and FP8 plans) is selected by the retained FP32 selectors
// and runs their passes.
//
//   - RoutedPlansUseRetainedFp32Passes: fixed FP32 sites of every family (RK, KR, pure R; Complete and staged first
//     stages) plan only retained FP32 passes, and compute the reduction.
//   - RandomMultiPassPlans: randomized multi-pass reductions (the FP8 multi-step test's shape families, plus
//     single-axis shapes) on FP32 and FP8 inputs, every public VALUE operation, a random tuning depth: every pass that
//     reads FP32 is a retained FP32 pass, every pass that reads FP8 a retained FP8 pass, and the result matches a host
//     reference (exact for Product, Min and Max; within an FP32 summation bound otherwise). The seed is printed;
//     THOR_FP32_ROUTING_SEED=<seed> THOR_FP32_ROUTING_CASE=<case> reruns one case (THOR_FP32_ROUTING_CASES_PER_OP,
//     default 2).

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

namespace {

constexpr uint64_t MaxElements = uint64_t{1} << 21;

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
    const auto problem = Internal::makeInitialDenseReductionProblem(
        dims, axes, Internal::DenseReductionAggregateKind::Value, Internal::DenseReductionOutputSpec{});
    return static_cast<size_t>(std::count_if(problem.runs.begin(), problem.runs.end(), [](const auto& run) {
        return run.kind == Internal::DenseReductionRunKind::OriginalReduction;
    }));
}

struct Shape {
    std::vector<uint64_t> dims;
    std::vector<uint32_t> axes;
    const char* family = "";
};

// The FP8 multi-step test's four multi-pass families, plus single-axis RK / KR / R shapes (one pass, or a staged
// first stage and its continuation).
Shape drawShape(std::mt19937_64& rng) {
    const uint64_t family = rng() % 5;
    for (;;) {
        Shape s;
        switch (family) {
            case 0: {
                s.family = "mask";
                const size_t rank = 3 + rng() % 3;
                for (size_t d = 0; d < rank; ++d) s.dims.push_back(rng() % 4 == 0 ? 1 : logUniform(rng, 2, 96));
                for (uint32_t d = 0; d < rank; ++d)
                    if (rng() % 2) s.axes.push_back(d);
                break;
            }
            case 1: {
                s.family = "deep_r";
                const uint64_t o = 1 + rng() % 4, k = 2 + rng() % 7, t = 2 + rng() % 15;
                const uint64_t r = logUniform(rng, 1024, std::max<uint64_t>(1024, MaxElements / (o * k * t)));
                s.dims = {o, r, k, t};
                s.axes = {1, 3};
                break;
            }
            case 2: {
                s.family = "long_rows";
                const uint64_t a = 2 + rng() % 31, b = 2 + rng() % 3;
                const uint64_t c = logUniform(rng, 4096, std::max<uint64_t>(4096, MaxElements / (a * b)));
                s.dims = {a, b, c};
                s.axes = {0, 2};
                break;
            }
            case 3: {
                s.family = "wide_inner";
                const uint64_t k = 2 + rng() % 63, c = logUniform(rng, 2, 256);
                const uint64_t r = logUniform(rng, 256, std::max<uint64_t>(256, MaxElements / (k * c)));
                s.dims = {r, k, c};
                s.axes = {0, 2};
                break;
            }
            default: {
                s.family = "single_axis";
                const uint64_t a = logUniform(rng, 1, 512), r = logUniform(rng, 2, 1 << 16);
                const uint64_t k = logUniform(rng, 1, 512);
                s.dims = {a, r, k};
                s.axes = {1};
                break;
            }
        }
        if (s.axes.empty()) continue;
        const uint64_t elements = product(s.dims);
        if (elements < 16 || elements > MaxElements) continue;
        const size_t runs = reducedRuns(s.dims, s.axes);
        if (runs >= 2 || (family == 4 && runs == 1)) return s;
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

// Row-major mapping of every input element to its output (and its flattened index over the reduced axes).
struct Mapping {
    std::vector<uint32_t> output;
    std::vector<uint64_t> index;  // per input element (ARG)
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

// Values exactly representable in FP8 (so an FP8 tensor holds them; an FP32 tensor holds them too).
std::vector<float> drawValues(std::mt19937_64& rng, uint64_t count, bool e5m2, Data data, uint64_t reduced) {
    std::array<float, 256> table{};
    for (uint32_t b = 0; b < 256; ++b) table[b] = Reference::decode(b, e5m2);
    std::vector<uint8_t> additive, extremum;
    std::array<int, 6> powers{{-1, -1, -1, -1, -1, -1}};
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
    const double rare = std::min(1.0, 2.0 / static_cast<double>(reduced));
    for (uint64_t i = 0; i < count; ++i) {
        switch (data) {
            case Data::Additive: {
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
    if (data == Data::Arg && rng() % 4 == 0) {
        constexpr uint8_t NanByte = 0x7F;  // a NaN in both formats
        for (int n = 0; n < 3; ++n) values[rng() % count] = table[NanByte];
    }
    return values;
}

const char* depthName(CubReductionTuningDepth depth) {
    return depth == CubReductionTuningDepth::Quick ? "quick"
           : depth == CubReductionTuningDepth::Ultra ? "ultra" : "regular";
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
    if (op == CubReductionOp::Min)
        std::fill(ref.value.begin(), ref.value.end(), std::numeric_limits<double>::infinity());
    if (op == CubReductionOp::Max)
        std::fill(ref.value.begin(), ref.value.end(), -std::numeric_limits<double>::infinity());
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
                break;
        }
    }
    return ref;
}

void expectMatches(const std::vector<float>& got, const ValueReference& ref) {
    ASSERT_EQ(got.size(), ref.value.size());
    for (uint64_t o = 0; o < got.size(); ++o) {
        if (ref.tolerance[o] == 0.0) {
            ASSERT_EQ(static_cast<double>(got[o]), ref.value[o]) << "output " << o;
        } else {
            ASSERT_NEAR(static_cast<double>(got[o]), ref.value[o], ref.tolerance[o]) << "output " << o;
        }
    }
}

bool startsWith(const std::string& text, const char* prefix) { return text.rfind(prefix, 0) == 0; }

// Every pass reading FP32 is a retained FP32 pass and every pass reading FP8 a retained FP8 pass.
void expectRetainedPasses(const std::vector<DensePhysicalPassDiagnostic>& passes) {
    for (size_t i = 0; i < passes.size(); ++i) {
        const DensePhysicalPassDiagnostic& pass = passes[i];
        if (pass.input_dtype == DataType::FP32) {
            EXPECT_TRUE(startsWith(pass.family, "retained_fp32_")) << "pass " << i << " family " << pass.family;
        } else {
            EXPECT_TRUE(startsWith(pass.family, "retained_fp8_")) << "pass " << i << " family " << pass.family;
        }
    }
}

// 1221: an ARG stamp's passes (getPlannedPassNames without the FP8 packed-continuation entries): the first reads the
// input, the rest read FP32 values and a carried index.
std::vector<std::string> argPasses(const StampedCubArgReduction& stamped) {
    std::vector<std::string> passes;
    for (const std::string& name : stamped.getPlannedPassNames())
        if (name != "fp8_arg_packed_continuation") passes.push_back(name);
    return passes;
}

void expectRetainedArgPasses(const std::vector<std::string>& passes, bool fp8_input) {
    ASSERT_FALSE(passes.empty());
    for (size_t i = 0; i < passes.size(); ++i) {
        const char* want = i != 0 ? "fp32_carried_arg_" : fp8_input ? "fp8_arg_" : "fp32_arg_";
        EXPECT_TRUE(startsWith(passes[i], want)) << "pass " << i << " " << passes[i] << " (want " << want << ")";
    }
}

struct ArgOutputs {
    CubArgReductionOutputOptions options;
    const char* name;
};

ArgOutputs drawArgOutputs(std::mt19937_64& rng) {
    ArgOutputs outputs;
    switch (rng() % 4) {
        case 0:
            outputs.options.value_output_dtype = DataType::FP32;
            outputs.options.index_output_dtype = DataType::UINT32;
            outputs.name = "value+u32";
            break;
        case 1:
            outputs.options.value_output_dtype = DataType::FP32;
            outputs.options.index_output_dtype = DataType::UINT64;
            outputs.name = "value+u64";
            break;
        case 2:
            outputs.options.produce_value = false;
            outputs.options.index_output_dtype = DataType::UINT32;
            outputs.name = "u32";
            break;
        default:
            outputs.options.produce_index = false;
            outputs.options.value_output_dtype = DataType::FP32;
            outputs.name = "value";
            break;
    }
    return outputs;
}

// Runs the stamp and checks the value bits and the flattened index against the ARG contract.
void expectArgMatches(StampedCubArgReduction& stamped, const CubArgReductionOutputOptions& outputs,
                      const std::vector<float>& x, const Mapping& m, bool maximum, Stream& stream) {
    stamped.run();
    stream.synchronize();
    std::vector<Reference::Candidate> best(m.outputs);
    std::vector<bool> seen(m.outputs, false);
    for (uint64_t i = 0; i < x.size(); ++i) {
        const Reference::Candidate c{m.index[i], x[i]};
        best[m.output[i]] = seen[m.output[i]] ? Reference::combine(best[m.output[i]], c, maximum) : c;
        seen[m.output[i]] = true;
    }
    if (outputs.produce_value) {
        ASSERT_TRUE(stamped.getValueOutputTensor().has_value());
        const std::vector<float> got = copyGpuTensorAsFloat(stamped.getValueOutputTensor().value(), stream);
        ASSERT_EQ(got.size(), best.size());
        for (uint64_t o = 0; o < got.size(); ++o)
            ASSERT_TRUE(Reference::sameValue(got[o], best[o].value))
                << "output " << o << ": " << got[o] << " vs " << best[o].value;
    }
    if (outputs.produce_index) {
        ASSERT_TRUE(stamped.getIndexOutputTensor().has_value());
        const std::vector<uint64_t> got = copyGpuTensorAsUnsigned(stamped.getIndexOutputTensor().value(), stream);
        ASSERT_EQ(got.size(), best.size());
        for (uint64_t o = 0; o < got.size(); ++o) ASSERT_EQ(got[o], best[o].index) << "output " << o;
    }
}

}  // namespace

// Fixed FP32 sites of every family plan only retained FP32 passes and compute the reduction.
TEST(CubReductionFp32Routing, RoutedPlansUseRetainedFp32Passes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Internal::denseReductionSelectionCache().clear();
    struct Case {
        Shape shape;
        CubReductionOp op;
    };
    const std::vector<Case> cases{
        {{{16, 3000, 64}, {1}, "rk"}, CubReductionOp::Sum},                 // RK, staged or Complete
        {{{4, 200000, 3}, {1}, "deep_rk"}, CubReductionOp::Mean},           // deep RK: staged first stage
        {{{5000, 300}, {1}, "kr"}, CubReductionOp::Max},                    // KR
        {{{1 << 20}, {0}, "pure_r"}, CubReductionOp::L2Norm},               // pure R (Split)
        {{{8, 600, 30, 50}, {1, 3}, "two_axes"}, CubReductionOp::L1Norm},  // KR then RK (an FP32 intermediate)
        {{{300, 7, 4096}, {0, 2}, "outer_and_row"}, CubReductionOp::Min}};
    std::mt19937_64 rng(1220);
    for (const Case& c : cases) {
        SCOPED_TRACE(describe(c.shape) + " " + valueOpName(c.op));
        const Mapping m = mapping(c.shape);
        const Data data = c.op == CubReductionOp::Min || c.op == CubReductionOp::Max ? Data::Extremum : Data::Additive;
        const std::vector<float> values = drawValues(rng, product(c.shape.dims), false, data, m.reduced);
        const Tensor input = makeGpuTensor(values, c.shape.dims, stream, DataType::FP32);
        auto stamped = CubReduction(c.op, c.shape.axes, DataType::FP32).stamp(input, stream);
        ASSERT_TRUE(stamped->usesDenseExecutablePlan());
        const auto passes = stamped->getDensePhysicalPassDiagnostics();
        ASSERT_FALSE(passes.empty());
        expectRetainedPasses(passes);
        stamped->run();
        stream.synchronize();
        expectMatches(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), valueReference(c.op, values, m));
    }
}

TEST(CubReductionFp32Routing, RandomMultiPassPlans) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Internal::denseReductionSelectionCache().clear();
    const uint64_t seed = envU64("THOR_FP32_ROUTING_SEED").value_or(
        (static_cast<uint64_t>(std::random_device{}()) << 32) ^ std::random_device{}());
    const uint64_t per_op = envU64("THOR_FP32_ROUTING_CASES_PER_OP").value_or(2);
    const std::optional<uint64_t> only_case = envU64("THOR_FP32_ROUTING_CASE");
    std::cout << "[ fp32 routing ] THOR_FP32_ROUTING_SEED=" << seed << " (" << per_op
              << " cases per operation and input dtype; set the seed to reproduce this run)" << std::endl;

    const std::array<CubReductionOp, 8> ops{{CubReductionOp::Sum, CubReductionOp::Mean, CubReductionOp::Product,
                                             CubReductionOp::Min, CubReductionOp::Max, CubReductionOp::L1Norm,
                                             CubReductionOp::L2Norm, CubReductionOp::SumSquares}};
    constexpr size_t InputDtypes = 2;  // FP32, then FP8 (E4M3 / E5M2 drawn)
    std::map<std::string, uint64_t> families, pass_counts;
    uint64_t cases = 0, fp32_passes = 0;
    for (uint64_t case_index = 0; case_index < ops.size() * InputDtypes * per_op; ++case_index) {
        if (only_case && *only_case != case_index) continue;
        std::mt19937_64 rng(mix(seed ^ mix(case_index)));
        const CubReductionOp op = ops[case_index % ops.size()];
        const bool fp8 = (case_index / ops.size()) % InputDtypes == 1;
#if !THOR_CUB_ENABLE_FP8_TYPES
        if (fp8) continue;
#endif
        const Shape shape = drawShape(rng);
        const bool e5m2 = rng() % 2;
        const DataType dtype = !fp8 ? DataType::FP32 : e5m2 ? DataType::FP8_E5M2 : DataType::FP8_E4M3;
        const CubReductionTuningDepth depth = std::array<CubReductionTuningDepth, 3>{
            CubReductionTuningDepth::Quick, CubReductionTuningDepth::Regular,
            CubReductionTuningDepth::Ultra}[rng() % 3];
        const Mapping m = mapping(shape);
        const Data data = op == CubReductionOp::Product ? Data::Product
                          : (op == CubReductionOp::Min || op == CubReductionOp::Max) ? Data::Extremum
                          : Data::Additive;
        const std::vector<float> generated = drawValues(rng, product(shape.dims), e5m2, data, m.reduced);
        const Tensor input = makeGpuTensor(generated, shape.dims, stream, dtype);
        const std::vector<float> x = copyGpuTensorAsFloat(input, stream);
        std::ostringstream what;
        what << "THOR_FP32_ROUTING_SEED=" << seed << " THOR_FP32_ROUTING_CASE=" << case_index << ": "
             << valueOpName(op) << ' ' << (!fp8 ? "fp32" : e5m2 ? "e5m2" : "e4m3") << ' ' << depthName(depth) << ' '
             << describe(shape);
        auto stamped = CubReduction(op, shape.axes, DataType::FP32, 1.0f, depth).stamp(input, stream);
        ASSERT_TRUE(stamped->usesDenseExecutablePlan()) << what.str();
        const auto passes = stamped->getDensePhysicalPassDiagnostics();
        what << " passes=" << passes.size();
        for (const auto& pass : passes) what << ' ' << pass.family;
        SCOPED_TRACE(what.str());
        expectRetainedPasses(passes);
        stamped->run();
        stream.synchronize();
        expectMatches(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), valueReference(op, x, m));
        ++cases;
        ++families[shape.family];
        ++pass_counts[std::to_string(passes.size())];
        for (const auto& pass : passes) fp32_passes += pass.input_dtype == DataType::FP32;
    }
    std::ostringstream summary;
    summary << "[ fp32 routing ] seed=" << seed << " cases=" << cases << " retained_fp32_passes=" << fp32_passes;
    for (const auto& [name, count] : families) summary << ' ' << name << '=' << count;
    summary << " passes:";
    for (const auto& [name, count] : pass_counts) summary << ' ' << name << 'x' << count;
    std::cout << summary.str() << std::endl;
}

// 1221: fixed FP32 ARG sites of every family (original RK, KR and pure R; carried RK and KR after a first pass in both
// orders, over an original and an intermediate run) plan only retained FP32 ARG passes and compute the reduction.
TEST(CubReductionFp32Routing, RoutedArgPlansUseRetainedFp32Passes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Internal::denseReductionSelectionCache().clear();
    const std::vector<Shape> shapes{{{16, 3000, 64}, {1}, "rk"},
                                    {{4, 200000, 3}, {1}, "deep_rk"},
                                    {{5000, 300}, {1}, "kr"},
                                    {{1 << 20}, {0}, "pure_r"},
                                    {{8, 600, 30, 50}, {1, 3}, "two_axes"},
                                    {{300, 7, 4096}, {0, 2}, "outer_and_row"},
                                    {{40, 3, 50, 2, 70}, {0, 2, 4}, "three_axes"}};
    std::mt19937_64 rng(1221);
    for (const Shape& shape : shapes) {
        for (const bool maximum : {false, true}) {
            const ArgOutputs outputs = drawArgOutputs(rng);
            SCOPED_TRACE(describe(shape) + (maximum ? " argmax " : " argmin ") + outputs.name);
            const Mapping m = mapping(shape);
            const std::vector<float> values = drawValues(rng, product(shape.dims), false, Data::Arg, m.reduced);
            const Tensor input = makeGpuTensor(values, shape.dims, stream, DataType::FP32);
            auto stamped = CubArgReduction(maximum ? CubArgReductionOp::ArgMax : CubArgReductionOp::ArgMin,
                                           shape.axes, outputs.options)
                               .stamp(input, stream);
            const std::vector<std::string> passes = argPasses(*stamped);
            EXPECT_EQ(reducedRuns(shape.dims, shape.axes), passes.size());
            expectRetainedArgPasses(passes, false);
            expectArgMatches(*stamped, outputs.options, values, m, maximum, stream);
        }
    }
}

// 1221: randomized multi-pass ARG reductions on FP32 and FP8 inputs (argmin / argmax, every output combination, a
// random tuning depth): the first pass is a retained FP32 (or FP8) ARG pass, every later one a retained FP32 carried
// ARG pass, and values and indices match the ARG contract (NaNs, -0 / +0, ties). Seed and case variables as above
// (THOR_FP32_ROUTING_ARG_CASES, default 8 per input dtype).
TEST(CubReductionFp32Routing, RandomMultiPassArgPlans) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Internal::denseReductionSelectionCache().clear();
    const uint64_t seed = envU64("THOR_FP32_ROUTING_SEED").value_or(
        (static_cast<uint64_t>(std::random_device{}()) << 32) ^ std::random_device{}());
    const uint64_t per_dtype = envU64("THOR_FP32_ROUTING_ARG_CASES").value_or(8);
    const std::optional<uint64_t> only_case = envU64("THOR_FP32_ROUTING_CASE");
    std::cout << "[ fp32 routing arg ] THOR_FP32_ROUTING_SEED=" << seed << " (" << per_dtype
              << " cases per input dtype; set the seed to reproduce this run)" << std::endl;
    std::map<std::string, uint64_t> families, pass_counts;
    uint64_t cases = 0;
    for (uint64_t case_index = 0; case_index < 2 * per_dtype; ++case_index) {
        if (only_case && *only_case != case_index) continue;
        std::mt19937_64 rng(mix(seed ^ mix(0x41524700ULL + case_index)));
        const bool fp8 = case_index % 2 == 1;
#if !THOR_CUB_ENABLE_FP8_TYPES
        if (fp8) continue;
#endif
        const bool maximum = rng() % 2;
        const Shape shape = drawShape(rng);
        const bool e5m2 = rng() % 2;
        const DataType dtype = !fp8 ? DataType::FP32 : e5m2 ? DataType::FP8_E5M2 : DataType::FP8_E4M3;
        const CubReductionTuningDepth depth = std::array<CubReductionTuningDepth, 3>{
            CubReductionTuningDepth::Quick, CubReductionTuningDepth::Regular,
            CubReductionTuningDepth::Ultra}[rng() % 3];
        const ArgOutputs outputs = drawArgOutputs(rng);
        const Mapping m = mapping(shape);
        const Tensor input =
            makeGpuTensor(drawValues(rng, product(shape.dims), e5m2, Data::Arg, m.reduced), shape.dims, stream, dtype);
        const std::vector<float> x = copyGpuTensorAsFloat(input, stream);
        std::ostringstream what;
        what << "THOR_FP32_ROUTING_SEED=" << seed << " THOR_FP32_ROUTING_CASE=" << case_index << ": "
             << (maximum ? "argmax " : "argmin ") << (!fp8 ? "fp32" : e5m2 ? "e5m2" : "e4m3") << ' '
             << depthName(depth) << ' ' << outputs.name << ' ' << describe(shape);
        auto stamped = CubArgReduction(maximum ? CubArgReductionOp::ArgMax : CubArgReductionOp::ArgMin, shape.axes,
                                       outputs.options, depth)
                           .stamp(input, stream);
        const std::vector<std::string> passes = argPasses(*stamped);
        what << " passes=" << passes.size();
        for (const std::string& pass : passes) what << ' ' << pass;
        SCOPED_TRACE(what.str());
        expectRetainedArgPasses(passes, fp8);
        expectArgMatches(*stamped, outputs.options, x, m, maximum, stream);
        ++cases;
        ++families[shape.family];
        ++pass_counts[std::to_string(passes.size())];
    }
    std::ostringstream summary;
    summary << "[ fp32 routing arg ] seed=" << seed << " cases=" << cases;
    for (const auto& [name, count] : families) summary << ' ' << name << '=' << count;
    summary << " passes:";
    for (const auto& [name, count] : pass_counts) summary << ' ' << name << 'x' << count;
    std::cout << summary.str() << std::endl;
}

// 1222: an FP32 input view off a 16-byte boundary (4 and 8 bytes) still reduces: its RK passes are the
// alignment-tolerant ones (the rotated RuntimeK pass; AlignedK at a packet no wider than the alignment), its KR passes
// any shape, and VALUE and ARG results are exact (a one-time warning names the slower path).
TEST(CubReductionFp32Routing, MisalignedInputsUseTolerantPasses) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Internal::denseReductionSelectionCache().clear();
    std::mt19937_64 rng(1222);
    const std::vector<Shape> shapes{{{16, 3000, 64}, {1}, "rk"}, {{2, 5000, 6}, {1}, "deep_rk"},
                                    {{5000, 300}, {1}, "kr"}, {{8, 600, 30, 50}, {1, 3}, "two_axes"}};
    for (const uint64_t offset_bytes : {uint64_t{4}, uint64_t{8}, uint64_t{0}}) {
        for (const Shape& shape : shapes) {
            SCOPED_TRACE(describe(shape) + " offset " + std::to_string(offset_bytes));
            const Mapping m = mapping(shape);
            const uint64_t elements = product(shape.dims);
            // A flat UINT8 storage holding the FP32 bytes at the offset (aliasByteStorageAs views byte storage), kept
            // alive beside its view.
            std::vector<Tensor> storages;
            const auto misaligned = [&](const std::vector<float>& values) {
                std::vector<uint64_t> bytes(offset_bytes + 4 * elements + 16, 0);
                for (uint64_t i = 0; i < elements; ++i) {
                    uint8_t element[4];
                    std::memcpy(element, &values[i], 4);
                    for (uint64_t b = 0; b < 4; ++b) bytes[offset_bytes + 4 * i + b] = element[b];
                }
                storages.push_back(makeGpuUnsignedTensor(bytes, {bytes.size()}, stream, DataType::UINT8));
                return storages.back().aliasByteStorageAs(DataType::FP32, shape.dims, offset_bytes);
            };
            // VALUE data without NaNs (MAX propagates a NaN; the host reference does not), ARG data with them.
            const std::vector<float> values = drawValues(rng, elements, false, Data::Extremum, m.reduced);
            const std::vector<float> arg_values = drawValues(rng, elements, false, Data::Arg, m.reduced);
            const Tensor input = misaligned(values);
            const Tensor arg_input = misaligned(arg_values);
            ASSERT_EQ(offset_bytes == 0 ? 16u : offset_bytes, input.alignmentBytes(16));
            ASSERT_EQ(offset_bytes == 0, input.isAligned(16));

            // VALUE: Max is exact.
            auto value = CubReduction(CubReductionOp::Max, shape.axes, DataType::FP32).stamp(input, stream);
            const auto passes = value->getDensePhysicalPassDiagnostics();
            expectRetainedPasses(passes);
            // The first pass reads the misaligned input (later passes read fresh, aligned intermediates).
            ASSERT_FALSE(passes.empty());
            const DensePhysicalPassDiagnostic& first = passes.front();
            if (offset_bytes != 0 && startsWith(first.family, "retained_fp32_rk")) {
                const bool tolerant = first.family == "retained_fp32_rk:runtime_k" ||
                                      (first.family == "retained_fp32_rk:aligned_k" && first.packet_bytes &&
                                       *first.packet_bytes <= offset_bytes);
                EXPECT_TRUE(tolerant) << first.family;
            }
            value->run();
            stream.synchronize();
            expectMatches(copyGpuTensorAsFloat(value->getOutputTensor(), stream),
                          valueReference(CubReductionOp::Max, values, m));

            // ARG: argmin, value and index.
            CubArgReductionOutputOptions outputs;
            outputs.value_output_dtype = DataType::FP32;
            outputs.index_output_dtype = DataType::UINT64;
            auto arg = CubArgReduction(CubArgReductionOp::ArgMin, shape.axes, outputs).stamp(arg_input, stream);
            const std::vector<std::string> arg_passes = argPasses(*arg);
            expectRetainedArgPasses(arg_passes, false);
            if (offset_bytes != 0 && startsWith(arg_passes.front(), "fp32_arg_rk:")) {
                EXPECT_TRUE(startsWith(arg_passes.front(), "fp32_arg_rk:runtime_k") ||
                            startsWith(arg_passes.front(), offset_bytes >= 8 ? "fp32_arg_rk:aligned_k_a8"
                                                                             : "fp32_arg_rk:runtime_k"))
                    << arg_passes.front();
            }
            expectArgMatches(*arg, outputs, arg_values, m, false, stream);
        }
    }
}
