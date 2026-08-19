#include "Utilities/Common/GpuMemoryDiagnostics.h"

#include <gtest/gtest.h>

using namespace ThorImplementation;

TEST(GpuMemoryDiagnostics, FormatsByteCountsForPlacementLogs) {
    EXPECT_EQ(formatGpuMemoryBytes(0), "0 B (0.00 B)");
    EXPECT_EQ(formatGpuMemoryBytes(1024), "1024 B (1.00 KiB)");
    EXPECT_EQ(formatGpuMemoryBytes(3ULL * 1024ULL * 1024ULL), "3145728 B (3.00 MiB)");
    EXPECT_EQ(formatGpuMemoryBytes(2ULL * 1024ULL * 1024ULL * 1024ULL), "2147483648 B (2.00 GiB)");
}

TEST(GpuMemoryDiagnostics, AllocationContextNestsAndRestoresWithoutSentinelState) {
    EXPECT_TRUE(currentGpuAllocationContext().empty());
    {
        ScopedGpuAllocationContext outer("attention backward workspace");
        EXPECT_EQ(currentGpuAllocationContext(), "attention backward workspace");
        {
            ScopedGpuAllocationContext inner("nested allocation");
            EXPECT_EQ(currentGpuAllocationContext(), "nested allocation");
        }
        EXPECT_EQ(currentGpuAllocationContext(), "attention backward workspace");
    }
    EXPECT_TRUE(currentGpuAllocationContext().empty());
}
