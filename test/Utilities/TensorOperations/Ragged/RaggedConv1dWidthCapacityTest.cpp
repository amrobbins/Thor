#include "Utilities/TensorOperations/Ragged/RaggedConv1dWidthCapacity.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

TEST(RaggedConv1dWidthCapacity, BuildsDeterministicFiniteQuarterOctaveFamily) {
    EXPECT_EQ(makeRaggedConv1dWidthCapacities(5), (std::vector<uint64_t>{5}));
    EXPECT_EQ(makeRaggedConv1dWidthCapacities(9), (std::vector<uint64_t>{8, 9}));
    EXPECT_EQ(makeRaggedConv1dWidthCapacities(64), (std::vector<uint64_t>{8, 16, 32, 64}));
    EXPECT_EQ(makeRaggedConv1dWidthCapacities(371), (std::vector<uint64_t>{8, 16, 32, 64, 128, 160, 192, 224, 256, 320, 371}));
    EXPECT_EQ(makeRaggedConv1dWidthCapacities(818),
              (std::vector<uint64_t>{8, 16, 32, 64, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 818}));
}

TEST(RaggedConv1dWidthCapacity, EveryTransitionSelectsSmallestPlacementDefinedCapacity) {
    const std::vector<uint64_t> capacities = makeRaggedConv1dWidthCapacities(818);
    ASSERT_FALSE(capacities.empty());

    EXPECT_EQ(chooseRaggedConv1dWidthCapacity(0, capacities), 0u);
    EXPECT_EQ(chooseRaggedConv1dWidthCapacity(371, capacities), 384u);
    uint64_t previous = 0;
    for (const uint64_t capacity : capacities) {
        if (previous + 1 <= capacity) {
            EXPECT_EQ(chooseRaggedConv1dWidthCapacity(previous + 1, capacities), capacity) << "lower boundary for capacity " << capacity;
        }
        EXPECT_EQ(chooseRaggedConv1dWidthCapacity(capacity, capacities), capacity) << "exact boundary for capacity " << capacity;
        previous = capacity;
    }
}

TEST(RaggedConv1dWidthCapacity, RuntimeSelectionCanNeverIntroduceANewPhysicalWidth) {
    const uint64_t maxValuesPerRow = 818;
    const std::vector<uint64_t> capacities = makeRaggedConv1dWidthCapacities(maxValuesPerRow);

    for (uint64_t activeWidth = 1; activeWidth <= maxValuesPerRow; ++activeWidth) {
        const uint64_t selected = chooseRaggedConv1dWidthCapacity(activeWidth, capacities);
        EXPECT_TRUE(std::binary_search(capacities.begin(), capacities.end(), selected))
            << "active width " << activeWidth << " selected undeclared width " << selected;
        EXPECT_GE(selected, activeWidth);
    }

    EXPECT_THROW((void)chooseRaggedConv1dWidthCapacity(maxValuesPerRow + 1, capacities), std::invalid_argument);
}

TEST(RaggedConv1dWidthCapacity, RejectsInvalidStructuralCapacity) {
    EXPECT_THROW((void)makeRaggedConv1dWidthCapacities(0), std::invalid_argument);
    EXPECT_THROW((void)chooseRaggedConv1dWidthCapacity(1, std::vector<uint64_t>{}), std::invalid_argument);
}
