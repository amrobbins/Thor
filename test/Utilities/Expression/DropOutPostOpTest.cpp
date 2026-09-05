#include "Utilities/Expression/DropOutPostOp.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <string>

using namespace ThorImplementation;

namespace {

bool hasTensorInputNamed(const CudaKernelExpression& kernel, const std::string& name) {
    return std::any_of(kernel.inputs().begin(), kernel.inputs().end(), [&](const CudaKernelExpression::TensorParamSpec& input) {
        return input.name == name;
    });
}

}  // namespace

TEST(DropOutPostOp, RaggedManagedActiveCountUsesScalarCarrierInForwardAndBackward) {
    const CudaKernelExpression kernel = makeDropOutPostOpKernel(DataType::FP32,
                                                                0.25f,
                                                                false,
                                                                true,
                                                                DataType::UINT32,
                                                                3,
                                                                5,
                                                                "RP6B9 active count",
                                                                RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);

    EXPECT_TRUE(hasTensorInputNamed(kernel, "active_count"));
    EXPECT_FALSE(hasTensorInputNamed(kernel, "offsets"));
    EXPECT_NE(kernel.source().find("active_count[0]"), std::string::npos);
    EXPECT_EQ(kernel.source().find("offsets[batch]"), std::string::npos);

    ASSERT_EQ(kernel.backwardSpecs().size(), 1u);
    ASSERT_NE(kernel.backwardSpecs().front().kernel, nullptr);
    const CudaKernelExpression& backward = *kernel.backwardSpecs().front().kernel;
    EXPECT_TRUE(hasTensorInputNamed(backward, "active_count"));
    EXPECT_FALSE(hasTensorInputNamed(backward, "offsets"));
    EXPECT_NE(backward.source().find("active_count[0]"), std::string::npos);
    EXPECT_EQ(backward.source().find("offsets[batch]"), std::string::npos);
}

TEST(DropOutPostOp, RaggedLegacyOffsetsModeRemainsAvailableForRowIndexedCallers) {
    const CudaKernelExpression kernel = makeDropOutPostOpKernel(DataType::FP16,
                                                                0.1f,
                                                                false,
                                                                true,
                                                                DataType::UINT64,
                                                                4,
                                                                7,
                                                                "legacy offsets");

    EXPECT_TRUE(hasTensorInputNamed(kernel, "offsets"));
    EXPECT_FALSE(hasTensorInputNamed(kernel, "active_count"));
    EXPECT_NE(kernel.source().find("offsets[batch]"), std::string::npos);
    EXPECT_EQ(kernel.source().find("active_count[0]"), std::string::npos);
}
