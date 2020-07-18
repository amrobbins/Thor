#include "MLDev.h"

#include <string>

#include "cuda.h"
#include "cuda_runtime.h"

#include "gtest/gtest.h"

TEST(MachineEvaluator, FoundAllGpus) {
    unsigned int numGpus;
    cudaError_t cudaStatus;
    int iNumGpus;
    cudaStatus = cudaGetDeviceCount(&iNumGpus);
    numGpus = iNumGpus;
    ASSERT_EQ(cudaStatus, cudaSuccess);

    ASSERT_EQ(MachineEvaluator::instance().getNumGpus(), numGpus);
}

TEST(MachineEvaluator, WhenPeerToPeerIsAvailableConnectionSpeedIsRanked) {
    if (MachineEvaluator::instance().getNumGpus() > 1 && MachineEvaluator::instance().isPeerToPeerAvailable(0, 1)) {
        ASSERT_GE(MachineEvaluator::instance().getConnectionSpeedRanking(0, 1), 0);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
