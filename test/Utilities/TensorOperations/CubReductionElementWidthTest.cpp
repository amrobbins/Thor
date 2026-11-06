// 1178: FP32 groundwork. A retained RK physical identity carries its input element width, so an FP8 and an FP32 kernel
// of the same geometry never compare equal and no FP32 identity validates before FP32 has an inventory; the selection
// cache has FP32 kinds whose entries stay apart from FP8's. Host-only. 1192: FP32 identities validate against FP32's
// inventory.

#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelection.h"

#include <gtest/gtest.h>

#include <string>

using namespace ThorImplementation::CubReductionInternal;
using Fp8RK::PhysicalFamily;
using Fp8RK::PhysicalKernel;

namespace {

// A legal FP8 DeepRSmallK identity at [16, 2048, 32] (depth 128, 16 shards of FP32 partials).
PhysicalKernel deepRSmallK() {
    PhysicalKernel kernel{PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 16 * 16, 16 * 16 * 32 * 4};
    return kernel;
}

}  // namespace

TEST(CubReductionElementWidth, IdentitiesDefaultToOneByteAndDifferByWidth) {
    const PhysicalKernel fp8 = deepRSmallK();
    EXPECT_EQ(fp8.element_bytes, 1U);
    PhysicalKernel fp32 = fp8;
    fp32.element_bytes = 4;
    EXPECT_FALSE(fp8 == fp32);
    fp32.element_bytes = 1;
    EXPECT_TRUE(fp8 == fp32);
}

// 1192: FP32 has an inventory too (its DeepRSmallK runs K <= 125). 1246: and FP16 / BF16 (2-byte DeepRSmallK runs
// K <= 249); other widths still validate nothing. (1256: renamed from OnlyTheFp8AndFp32InventoriesValidate.)
TEST(CubReductionElementWidth, OnlyTheFp8Fp16AndFp32InventoriesValidate) {
    const PhysicalKernel fp8 = deepRSmallK();
    EXPECT_TRUE(Fp8RK::structurallyValid(fp8, 16, 2048, 32));
    PhysicalKernel fp32 = fp8;
    fp32.element_bytes = 4;
    EXPECT_TRUE(Fp8RK::structurallyValid(fp32, 16, 2048, 32));
    PhysicalKernel f16 = fp8;
    f16.element_bytes = 2;
    EXPECT_TRUE(Fp8RK::structurallyValid(f16, 16, 2048, 32));
    for (const uint8_t width : {uint8_t{3}, uint8_t{8}}) {
        PhysicalKernel other = fp8;
        other.element_bytes = width;
        EXPECT_FALSE(Fp8RK::structurallyValid(other, 16, 2048, 32)) << "width " << int(width);
    }
    // K = 200: FP8's DeepRSmallK (K <= 256) and FP16 / BF16's (K <= 249), not FP32's.
    PhysicalKernel wide{PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 16 * 16, 16 * 16 * 200 * 4};
    EXPECT_TRUE(Fp8RK::structurallyValid(wide, 16, 2048, 200));
    wide.element_bytes = 2;
    EXPECT_TRUE(Fp8RK::structurallyValid(wide, 16, 2048, 200));
    wide.element_bytes = 4;
    EXPECT_FALSE(Fp8RK::structurallyValid(wide, 16, 2048, 200));
}

TEST(CubReductionElementWidth, Fp32SelectionKindsAreNamedAndKeptApart) {
    EXPECT_EQ(std::string(denseReductionSelectionKindName(DenseReductionSelectionKind::Fp32RK)), "fp32_rk");
    EXPECT_EQ(std::string(denseReductionSelectionKindName(DenseReductionSelectionKind::Fp32KR)), "fp32_kr");
    DenseReductionSelectionCache cache;
    DenseReductionSelectionKey fp8;
    fp8.kind = DenseReductionSelectionKind::Fp8RK;
    fp8.site = {1, 2, 3};
    DenseReductionSelectionKey fp32 = fp8;
    fp32.kind = DenseReductionSelectionKind::Fp32RK;
    EXPECT_FALSE(fp8 == fp32);
    Fp8RK::Fp8RKSelection recipe;
    recipe.winner_name = "fp8";
    (void)cache.getOrTune<Fp8RK::Fp8RKSelection>(fp8, [&] { return recipe; });
    EXPECT_FALSE(cache.find<Fp8RK::Fp8RKSelection>(fp32).has_value());
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp8RK), 1U);
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK), 0U);
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR), 0U);
}
