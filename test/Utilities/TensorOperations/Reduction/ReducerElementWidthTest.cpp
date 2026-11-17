// An RK physical identity carries its input element width, so kernels of the same geometry at different widths never
// compare equal, and each identity validates against its own width's inventory (FP8, FP16 / BF16 and FP32); the
// selection cache's FP32 kinds keep their entries apart from FP8's. Host-only.

#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalIdentity.h"
#include "Utilities/TensorOperations/Reduction/RK/RKSelection.h"

#include <gtest/gtest.h>

#include <string>

using namespace ThorImplementation::ReductionInternal;
using RK::PhysicalFamily;
using RK::PhysicalKernel;

namespace {

// A legal FP8 DeepRSmallK identity at [16, 2048, 32] (depth 128, 16 shards of FP32 partials).
PhysicalKernel deepRSmallK() {
    PhysicalKernel kernel{PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 16 * 16, 16 * 16 * 32 * 4};
    return kernel;
}

}  // namespace

TEST(ReducerElementWidth, IdentitiesDefaultToOneByteAndDifferByWidth) {
    const PhysicalKernel fp8 = deepRSmallK();
    EXPECT_EQ(fp8.element_bytes, 1U);
    PhysicalKernel fp32 = fp8;
    fp32.element_bytes = 4;
    EXPECT_FALSE(fp8 == fp32);
    fp32.element_bytes = 1;
    EXPECT_TRUE(fp8 == fp32);
}

// FP32 has an inventory too (its DeepRSmallK runs K <= 125). And FP16 / BF16 (2-byte DeepRSmallK runs K <= 249); other
// widths validate nothing.
TEST(ReducerElementWidth, OnlyTheFp8Fp16AndFp32InventoriesValidate) {
    const PhysicalKernel fp8 = deepRSmallK();
    EXPECT_TRUE(RK::structurallyValid(fp8, 16, 2048, 32));
    PhysicalKernel fp32 = fp8;
    fp32.element_bytes = 4;
    EXPECT_TRUE(RK::structurallyValid(fp32, 16, 2048, 32));
    PhysicalKernel f16 = fp8;
    f16.element_bytes = 2;
    EXPECT_TRUE(RK::structurallyValid(f16, 16, 2048, 32));
    for (const uint8_t width : {uint8_t{3}, uint8_t{8}}) {
        PhysicalKernel other = fp8;
        other.element_bytes = width;
        EXPECT_FALSE(RK::structurallyValid(other, 16, 2048, 32)) << "width " << int(width);
    }
    // K = 200: FP8's DeepRSmallK (K <= 256) and FP16 / BF16's (K <= 249), not FP32's.
    PhysicalKernel wide{PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 16 * 16, 16 * 16 * 200 * 4};
    EXPECT_TRUE(RK::structurallyValid(wide, 16, 2048, 200));
    wide.element_bytes = 2;
    EXPECT_TRUE(RK::structurallyValid(wide, 16, 2048, 200));
    wide.element_bytes = 4;
    EXPECT_FALSE(RK::structurallyValid(wide, 16, 2048, 200));
}

TEST(ReducerElementWidth, Fp32SelectionKindsAreNamedAndKeptApart) {
    EXPECT_EQ(std::string(denseReductionSelectionKindName(DenseReductionSelectionKind::Fp32RK)), "fp32_rk");
    EXPECT_EQ(std::string(denseReductionSelectionKindName(DenseReductionSelectionKind::Fp32KR)), "fp32_kr");
    DenseReductionSelectionCache cache;
    DenseReductionSelectionKey fp8;
    fp8.kind = DenseReductionSelectionKind::Fp8RK;
    fp8.site = {1, 2, 3};
    DenseReductionSelectionKey fp32 = fp8;
    fp32.kind = DenseReductionSelectionKind::Fp32RK;
    EXPECT_FALSE(fp8 == fp32);
    RK::RKSelection recipe;
    recipe.winner_name = "fp8";
    (void)cache.getOrTune<RK::RKSelection>(fp8, [&] { return recipe; });
    EXPECT_FALSE(cache.find<RK::RKSelection>(fp32).has_value());
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp8RK), 1U);
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK), 0U);
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR), 0U);
}
