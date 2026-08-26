#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdlib>

// The T10 CMake production gate deliberately runs this disabled preflight with
// --gtest_also_run_disabled_tests. The constituent CUDA tests use GTEST_SKIP on
// GPU-less machines for ordinary developer convenience; the production gate
// itself must never succeed vacuously that way.
TEST(RetainedRaggedTrainingProductionGate, DISABLED_RequiresCudaDevice) {
    if (std::getenv("THOR_T10_RETAINED_RAGGED_TRAINING_GATE") == nullptr) {
        GTEST_SKIP() << "T10 retained-ragged training preflight only runs through "
                        "check-retained-ragged-training-production-gate";
    }

    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    ASSERT_GT(device_count, 0) << "T10 retained-ragged training production qualification requires a CUDA device.";
}
