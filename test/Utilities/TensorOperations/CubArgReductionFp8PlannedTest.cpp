// ARG-P1: original FP8 dense ARG runs the planDenseReduction() pass list. Checks exact public indices and values
// against an FP32 reference over every reduction mask of several ranks (singleton-heavy shapes included), the
// interval executor on composed geometries, the public path, and workspace accounting.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Reference = ThorImplementation::Fp8ArgReference;

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

struct ArgResult {
    std::vector<float> values;
    std::vector<uint64_t> indices;
};

// FP8 bytes with NaNs, +-0, extremes and (for small pools) many ties, decoded to FP32.
std::vector<float> fp8Values(uint64_t count, bool e5m2, uint32_t seed, bool tie_heavy) {
    static constexpr std::array<uint8_t, 10> SPECIALS{{0x00, 0x80, 0x7F, 0xFF, 0x7E, 0xFE, 0x7B, 0xFB, 0x01, 0x81}};
    static constexpr std::array<uint8_t, 4> TIES{{0x38, 0xB8, 0x00, 0x80}};
    std::mt19937 rng(seed);
    std::vector<float> values(count);
    for (float& value : values) {
        const uint32_t pick = rng() % 8u;
        const uint8_t byte = tie_heavy ? TIES[rng() % TIES.size()]
                             : pick == 0 ? SPECIALS[rng() % SPECIALS.size()]
                                         : static_cast<uint8_t>(rng());
        value = Reference::decode(byte, e5m2);
    }
    return values;
}

// Row-major reference: the index is the flattened position over the reduced axes in increasing axis order.
ArgResult reference(const std::vector<float>& input,
                    const std::vector<uint64_t>& dimensions,
                    const std::vector<uint32_t>& axes,
                    bool maximum) {
    const size_t rank = dimensions.size();
    std::vector<bool> reduced(rank, false);
    for (uint32_t axis : axes) reduced[axis] = true;
    uint64_t outputs = 1;
    for (size_t d = 0; d < rank; ++d)
        if (!reduced[d]) outputs *= dimensions[d];
    ArgResult result;
    std::vector<bool> seen(outputs, false);
    std::vector<Reference::Candidate> best(outputs);
    std::vector<uint64_t> coordinate(rank, 0);
    for (uint64_t linear = 0; linear < input.size(); ++linear) {
        uint64_t remainder = linear;
        for (size_t d = rank; d-- > 0;) {
            coordinate[d] = remainder % dimensions[d];
            remainder /= dimensions[d];
        }
        uint64_t output = 0, index = 0;
        for (size_t d = 0; d < rank; ++d) {
            if (reduced[d]) index = index * dimensions[d] + coordinate[d];
            else output = output * dimensions[d] + coordinate[d];
        }
        const Reference::Candidate candidate{index, input[linear]};
        best[output] = seen[output] ? Reference::combine(best[output], candidate, maximum) : candidate;
        seen[output] = true;
    }
    for (const Reference::Candidate& candidate : best) {
        result.values.push_back(candidate.value);
        result.indices.push_back(candidate.index);
    }
    return result;
}

void expectSameValues(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
        ASSERT_TRUE(Reference::sameValue(actual[i], expected[i]))
            << "output " << i << ": " << actual[i] << " vs " << expected[i];
}

std::vector<std::vector<uint32_t>> everyMask(size_t rank) {
    std::vector<std::vector<uint32_t>> masks;
    for (uint32_t bits = 1; bits < (1u << rank); ++bits) {
        std::vector<uint32_t> axes;
        for (uint32_t axis = 0; axis < rank; ++axis)
            if (bits & (1u << axis)) axes.push_back(axis);
        masks.push_back(axes);
    }
    return masks;
}

CubArgReductionOutputOptions valueAndIndex(DataType index_dtype = DataType::UINT32) {
    CubArgReductionOutputOptions outputs;
    outputs.value_output_dtype = DataType::FP32;
    outputs.index_output_dtype = index_dtype;
    return outputs;
}

// Stamps the production reduction, checks path, workspace and outputs against the reference.
void checkProduction(const Tensor& input,
                     const std::vector<float>& input_values,
                     const std::vector<uint64_t>& dimensions,
                     const std::vector<uint32_t>& axes,
                     CubArgReductionOp op,
                     const CubArgReductionOutputOptions& outputs,
                     Stream& stream) {
    const CubArgReduction reduction(op, axes, outputs);
    auto stamped = reduction.stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), CubArgReduction::analyzeDenseGeometry(dimensions, axes).path);
    EXPECT_EQ(reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream), stamped->getWorkspaceSizeInBytes());
    stamped->run();
    stream.synchronize();
    const ArgResult expected = reference(input_values, dimensions, axes, op == CubArgReductionOp::ArgMax);
    if (outputs.produce_value) {
        ASSERT_TRUE(stamped->getValueOutputTensor().has_value());
        expectSameValues(copyGpuTensorAsFloat(stamped->getValueOutputTensor().value(), stream), expected.values);
    }
    if (outputs.produce_index) {
        ASSERT_TRUE(stamped->getIndexOutputTensor().has_value());
        EXPECT_EQ(copyGpuTensorAsUnsigned(stamped->getIndexOutputTensor().value(), stream), expected.indices);
    }
}

}  // namespace

TEST(CubArgReductionFp8Planned, EveryMaskMatchesReferenceAcrossRanksAndSingletons) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<std::vector<uint64_t>> shapes = {
        {7}, {3, 5}, {5, 3}, {2, 3, 4}, {4, 1, 3}, {1, 6, 1}, {2, 1, 3, 1, 5}, {3, 4, 5, 2}, {2, 3, 2, 3, 2}};
    uint32_t seed = 1106;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const auto& dimensions : shapes) {
            uint64_t elements = 1;
            for (uint64_t d : dimensions) elements *= d;
            for (const bool tie_heavy : {false, true}) {
                const std::vector<float> generated = fp8Values(elements, dtype == DataType::FP8_E5M2, ++seed, tie_heavy);
                Tensor input = makeGpuTensor(generated, dimensions, stream, dtype);
                // The reference consumes exactly what the GPU holds (storage conversion may saturate).
                const std::vector<float> values = copyGpuTensorAsFloat(input, stream);
                for (const auto& axes : everyMask(dimensions.size())) {
                    for (const CubArgReductionOp op : {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax}) {
                        SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dtype) << " rank="
                                                          << dimensions.size() << " mask_first=" << axes.front()
                                                          << " mask_size=" << axes.size() << " ties=" << tie_heavy
                                                          << " max=" << (op == CubArgReductionOp::ArgMax));
                        checkProduction(input, values, dimensions, axes, op, valueAndIndex(), stream);
                    }
                }
            }
        }
    }
}

TEST(CubArgReductionFp8Planned, RKShapesOutputModesAndUint64Indices) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    struct Case {
        std::vector<uint64_t> dimensions;
        std::vector<uint32_t> axes;
    };
    const std::vector<Case> cases = {
        {{3, 37, 129}, {1}}, {{2, 300, 33}, {1}}, {{1, 1000, 7}, {1}}, {{64, 4, 2}, {1}}, {{5, 17, 512}, {1}},
        {{1000, 7}, {1}},    {{4, 33, 5, 17}, {1, 3}}, {{4, 33, 5, 17}, {0, 2}}, {{6, 9, 10, 3}, {0, 1, 3}}};
    CubArgReductionOutputOptions value_only;
    value_only.produce_index = false;
    value_only.value_output_dtype = DataType::FP32;
    CubArgReductionOutputOptions index_only;
    index_only.produce_value = false;
    index_only.index_output_dtype = DataType::UINT32;
    uint32_t seed = 77;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const Case& test_case : cases) {
            uint64_t elements = 1;
            for (uint64_t d : test_case.dimensions) elements *= d;
            Tensor input = makeGpuTensor(fp8Values(elements, dtype == DataType::FP8_E5M2, ++seed, false),
                                         test_case.dimensions, stream, dtype);
            const std::vector<float> values = copyGpuTensorAsFloat(input, stream);
            for (const CubArgReductionOp op : {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax}) {
                SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dtype) << " elements=" << elements
                                                  << " max=" << (op == CubArgReductionOp::ArgMax));
                for (const auto& outputs : {valueAndIndex(), valueAndIndex(DataType::UINT64), value_only, index_only})
                    checkProduction(input, values, test_case.dimensions, test_case.axes, op, outputs, stream);
            }
        }
    }
}

// Composed FP8 geometries: the planned executor and the forced interval executor return identical public results
// (the topology may differ; the ARG contract fixes the answer).
TEST(CubArgReductionFp8Planned, ComposedGeometriesMatchReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<std::pair<std::vector<uint64_t>, std::vector<uint32_t>>> cases = {
        {{4, 33, 5, 17}, {1, 3}}, {{4, 33, 5, 17}, {0, 2}}, {{3, 4, 5, 6, 7}, {0, 2, 4}}, {{8, 3, 9, 2}, {0, 3}}};
    uint32_t seed = 5;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        for (const auto& [dimensions, axes] : cases) {
            uint64_t elements = 1;
            for (uint64_t d : dimensions) elements *= d;
            Tensor input = makeGpuTensor(fp8Values(elements, dtype == DataType::FP8_E5M2, ++seed, true), dimensions,
                                         stream, dtype);
            ASSERT_EQ(CubArgReduction::analyzeDenseGeometry(dimensions, axes).path, CubReductionPath::ComposedDense);
            for (const CubArgReductionOp op : {CubArgReductionOp::ArgMin, CubArgReductionOp::ArgMax}) {
                const CubArgReduction reduction(op, axes, valueAndIndex());
                auto planned = reduction.stamp(input, stream);
                // A5 (1119): the direct interval executor no longer compiles FP8 tiled stages; check against the
                // host reference instead.
                EXPECT_THROW(static_cast<void>(reduction.stampComposedDense(input, stream)), std::invalid_argument);
                planned->run();
                stream.synchronize();
                const ArgResult expected = reference(copyGpuTensorAsFloat(input, stream), dimensions, axes,
                                                     op == CubArgReductionOp::ArgMax);
                expectSameValues(copyGpuTensorAsFloat(planned->getValueOutputTensor().value(), stream),
                                 expected.values);
                EXPECT_EQ(copyGpuTensorAsUnsigned(planned->getIndexOutputTensor().value(), stream), expected.indices);
            }
        }
    }
}

// Rerunning a stamped planned reduction after the input changes reuses its stamp-owned intermediates.
TEST(CubArgReductionFp8Planned, RerunAfterInputChangeUsesSameStamp) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<uint64_t> dimensions{4, 33, 5, 17};
    const std::vector<uint32_t> axes{1, 3};
    uint64_t elements = 1;
    for (uint64_t d : dimensions) elements *= d;
    Tensor input = makeGpuTensor(fp8Values(elements, false, 9, false), dimensions, stream, DataType::FP8_E4M3);
    const CubArgReduction reduction(CubArgReductionOp::ArgMax, axes, valueAndIndex());
    auto stamped = reduction.stamp(input, stream);
    const size_t workspace = stamped->getWorkspaceSizeInBytes();
    for (uint32_t round = 0; round < 3; ++round) {
        Tensor replacement =
            makeGpuTensor(fp8Values(elements, false, 100 + round, round == 1), dimensions, stream, DataType::FP8_E4M3);
        input.copyFromAsync(replacement, stream);
        stamped->run();
        stream.synchronize();
        const ArgResult expected = reference(copyGpuTensorAsFloat(input, stream), dimensions, axes, true);
        expectSameValues(copyGpuTensorAsFloat(stamped->getValueOutputTensor().value(), stream), expected.values);
        EXPECT_EQ(copyGpuTensorAsUnsigned(stamped->getIndexOutputTensor().value(), stream), expected.indices);
        EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), workspace);
    }
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES
