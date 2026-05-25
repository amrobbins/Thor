#include "Utilities/TensorOperations/DeepLearning/CudnnAdaptiveLayerNorm.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

namespace {

CudnnAdaptiveLayerNormDescriptor makeDescriptor() {
    CudnnAdaptiveLayerNormDescriptor descriptor;
    descriptor.batchSize = 2;
    descriptor.leadingFeatureCount = 4;
    descriptor.normalizedFeatureCount = 32;
    descriptor.inputDataType = DataType::FP16;
    descriptor.outputDataType = DataType::FP16;
    descriptor.scaleBiasDataType = DataType::FP32;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    return descriptor;
}

}  // namespace

TEST(CudnnAdaptiveLayerNormDescriptor, AcceptsFp16Bf16AndFp32IoWithFp32ScaleBias) {
    for (DataType dtype : {DataType::FP16,
                                             DataType::BF16,
                                             DataType::FP32}) {
        CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
        descriptor.inputDataType = dtype;
        descriptor.outputDataType = dtype;
        EXPECT_NO_THROW(descriptor.validateForward());
        EXPECT_NO_THROW(descriptor.validateBackward());
    }
}

TEST(CudnnAdaptiveLayerNormDescriptor, RejectsEmptyBatchLeadingOrFeatureCount) {
    CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.batchSize = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.leadingFeatureCount = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.normalizedFeatureCount = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAdaptiveLayerNormDescriptor, RejectsFp32NormalizedFeatureCountsUnsupportedByPrimaryEngines) {
    CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.normalizedFeatureCount = 16;
    descriptor.inputDataType = DataType::FP32;
    descriptor.outputDataType = DataType::FP32;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor.normalizedFeatureCount = 32;
    EXPECT_NO_THROW(descriptor.validateForward());
}

TEST(CudnnAdaptiveLayerNormDescriptor, RejectsUnsupportedIoDtype) {
    CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.inputDataType = DataType::FP8_E4M3;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.outputDataType = DataType::INT32;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAdaptiveLayerNormDescriptor, RejectsNonFp32ScaleBiasOrCompute) {
    CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.scaleBiasDataType = DataType::FP16;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.computeDataType = DataType::FP16;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAdaptiveLayerNormDescriptor, RejectsNonPositiveEpsilon) {
    CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.epsilon = 0.0f;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.epsilon = -1.0e-5f;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}
