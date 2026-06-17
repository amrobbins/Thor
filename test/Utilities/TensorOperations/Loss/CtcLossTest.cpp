#include "Utilities/TensorOperations/Loss/CtcLoss.h"

#include "gtest/gtest.h"

#include <stdexcept>

using namespace ThorImplementation;

namespace {

CudnnCtcLossConfig validConfig() {
    CudnnCtcLossConfig config;
    config.maxTimeSteps = 16;
    config.batchSize = 4;
    config.numClasses = 8;
    config.maxLabelLength = 15;
    config.dataType = DataType::FP32;
    return config;
}

}  // namespace

TEST(CudnnCtcLossPlan, AcceptsNarrowDeterministicFp32Config) {
    CudnnCtcLossPlan::validateConfig(validConfig());
}

TEST(CudnnCtcLossPlan, RejectsNonFp32DataType) {
    CudnnCtcLossConfig config = validConfig();
    config.dataType = DataType::FP16;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);
}

TEST(CudnnCtcLossPlan, RejectsDegenerateShape) {
    CudnnCtcLossConfig config = validConfig();
    config.maxTimeSteps = 0;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);

    config = validConfig();
    config.batchSize = 0;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);

    config = validConfig();
    config.numClasses = 1;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);
}

TEST(CudnnCtcLossPlan, RejectsDeterministicMaxLabelLengthAtCudnnLimit) {
    CudnnCtcLossConfig config = validConfig();
    config.maxLabelLength = 256;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);
}
