#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

const NamedOutput& namedOutput(const PhysicalOutputs& outputs, const std::string& name) {
    for (const NamedOutput& output : outputs.outputs) {
        if (output.name == name) {
            return output;
        }
    }
    throw std::runtime_error("Missing named output in CastAutoDiffTest: " + name);
}

std::vector<DataType> inputDTypes(const PhysicalOutputs& outputs,
                                  const std::unordered_map<std::string, DataType>& dtypeByName) {
    if (!outputs.expr) {
        throw std::runtime_error("CastAutoDiffTest received null expression.");
    }

    std::vector<DataType> dtypes(outputs.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : outputs.expr->inputs) {
        auto it = dtypeByName.find(input.name);
        if (it == dtypeByName.end()) {
            throw std::runtime_error("Missing input dtype in CastAutoDiffTest for: " + input.name);
        }
        dtypes.at(input.slot) = it->second;
    }
    return dtypes;
}

PhysicalOutputs buildCastBackward(const Expression& result,
                                  DataType sourceDtype,
                                  DataType outputDtype) {
    PhysicalOutputs forward = Expression::outputs({{"y", result}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {sourceDtype});

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", outputDtype}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 3}}});
    resolveOutputsDTypesInPlace(backward,
                                inputDTypes(backward,
                                            {
                                                {"x", sourceDtype},
                                                {"dy", outputDtype},
                                            }));
    return backward;
}

std::vector<uint32_t> castNodesWithOutputDType(const PhysicalOutputs& outputs, DataType dtype) {
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < outputs.expr->nodes.size(); ++i) {
        const ExprNode& node = outputs.expr->nodes.at(i);
        if (node.op == ExprOp::CAST && node.output_dtype.has_value() && node.output_dtype.value() == dtype) {
            result.push_back(i);
        }
    }
    return result;
}

}  // namespace

TEST(ExpressionCastAutoDiff, Fp32ToBf16CastsIncomingGradientBackToFp32) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    PhysicalOutputs backward = buildCastBackward(x.cast(DataType::BF16), DataType::FP32, DataType::BF16);

    const std::vector<uint32_t> fp32Casts = castNodesWithOutputDType(backward, DataType::FP32);
    ASSERT_EQ(fp32Casts.size(), 1u);

    const ExprNode& gradientOutput = backward.expr->nodes.at(namedOutput(backward, "x_grad").node_idx);
    ASSERT_TRUE(gradientOutput.output_dtype.has_value());
    EXPECT_EQ(gradientOutput.output_dtype.value(), DataType::FP32);
}

TEST(ExpressionCastAutoDiff, Bf16ToFp32CastsIncomingGradientBackToBf16) {
    const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
    PhysicalOutputs backward = buildCastBackward(x.cast(DataType::FP32), DataType::BF16, DataType::FP32);

    const std::vector<uint32_t> bf16Casts = castNodesWithOutputDType(backward, DataType::BF16);
    ASSERT_EQ(bf16Casts.size(), 1u);

    const ExprNode& gradientOutput = backward.expr->nodes.at(namedOutput(backward, "x_grad").node_idx);
    ASSERT_TRUE(gradientOutput.output_dtype.has_value());
    EXPECT_EQ(gradientOutput.output_dtype.value(), DataType::BF16);
}

TEST(ExpressionCastAutoDiff, ChainedCastsReverseTheDtypeConversions) {
    const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
    const Expression y = x.cast(DataType::FP32).cast(DataType::FP16);
    PhysicalOutputs backward = buildCastBackward(y, DataType::BF16, DataType::FP16);

    EXPECT_EQ(castNodesWithOutputDType(backward, DataType::FP32).size(), 1u);
    EXPECT_EQ(castNodesWithOutputDType(backward, DataType::BF16).size(), 1u);

    const ExprNode& gradientOutput = backward.expr->nodes.at(namedOutput(backward, "x_grad").node_idx);
    ASSERT_TRUE(gradientOutput.output_dtype.has_value());
    EXPECT_EQ(gradientOutput.output_dtype.value(), DataType::BF16);
}

TEST(ExpressionCastAutoDiff, RequiresResolvedSourceValueDtype) {
    const Expression x = Expression::input("x");
    const PhysicalOutputs forward = Expression::outputs({{"y", x.cast(DataType::FP32)}}).physicalOutputs();

    EXPECT_THROW(
        (void)buildBackwardOutputs(
            forward,
            {"x"},
            std::unordered_map<std::string, std::string>{{"y", "dy"}},
            std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
            std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 3}}}),
        std::runtime_error);
}
