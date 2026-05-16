#include "Utilities/TensorOperations/DeepLearning/CudnnAdaptiveLayerNorm.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

namespace {

CudnnAdaptiveLayerNormDescriptor makeDescriptor() {
    CudnnAdaptiveLayerNormDescriptor descriptor;
    descriptor.outerSize = 8;
    descriptor.normalizedFeatureCount = 16;
    descriptor.inputDataType = TensorDescriptor::DataType::FP16;
    descriptor.outputDataType = TensorDescriptor::DataType::FP16;
    descriptor.scaleBiasDataType = TensorDescriptor::DataType::FP32;
    descriptor.computeDataType = TensorDescriptor::DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    return descriptor;
}

}  // namespace

TEST(CudnnAdaptiveLayerNormDescriptor, AcceptsFp16Bf16AndFp32IoWithFp32ScaleBias) {
    for (TensorDescriptor::DataType dtype : {TensorDescriptor::DataType::FP16,
                                             TensorDescriptor::DataType::BF16,
                                             TensorDescriptor::DataType::FP32}) {
        CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
        descriptor.inputDataType = dtype;
        descriptor.outputDataType = dtype;
        EXPECT_NO_THROW(descriptor.validateForward());
        EXPECT_NO_THROW(descriptor.validateBackward());
    }
}

TEST(CudnnAdaptiveLayerNormDescriptor, RejectsEmptyOuterOrFeatureCount) {
    CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.outerSize = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.normalizedFeatureCount = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAdaptiveLayerNormDescriptor, RejectsUnsupportedIoDtype) {
    CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.inputDataType = TensorDescriptor::DataType::FP8_E4M3;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.outputDataType = TensorDescriptor::DataType::INT32;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAdaptiveLayerNormDescriptor, RejectsNonFp32ScaleBiasOrCompute) {
    CudnnAdaptiveLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.scaleBiasDataType = TensorDescriptor::DataType::FP16;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.computeDataType = TensorDescriptor::DataType::FP16;
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
