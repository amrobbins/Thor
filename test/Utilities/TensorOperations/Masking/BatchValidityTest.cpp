#include "Utilities/TensorOperations/Masking/BatchValidity.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include "gtest/gtest.h"

#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

TEST(BatchValidity, ZeroInvalidBatchTailPreservesValidPrefix) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor values = makeGpuTensor({1.0f, 2.0f, 3.0f,
                                   4.0f, 5.0f, 6.0f,
                                   7.0f, 8.0f, 9.0f,
                                   10.0f, 11.0f, 12.0f},
                                  {4, 3},
                                  stream);

    zeroInvalidBatchTail(values, 2, stream);
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(values, stream),
                          {1.0f, 2.0f, 3.0f,
                           4.0f, 5.0f, 6.0f,
                           0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f});
}

TEST(BatchValidity, WritesDenseValidityMaskForEveryElementInEachRow) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor mask = makeGpuTensor(std::vector<float>(4, -1.0f), {4, 1}, stream, DataType::FP32);

    writeBatchValidityMask(mask, 3, stream);
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(mask, stream), {1.0f, 1.0f, 1.0f, 0.0f});
}

TEST(BatchValidity, FullBatchRewriteClearsPriorTailMasking) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor mask = makeGpuTensor(std::vector<float>(4, -1.0f), {4, 1}, stream, DataType::FP32);

    writeBatchValidityMask(mask, 2, stream);
    writeBatchValidityMask(mask, 4, stream);
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(mask, stream), {1.0f, 1.0f, 1.0f, 1.0f});
}

TEST(BatchValidity, RejectsInvalidCardinality) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor values = makeGpuTensor(std::vector<float>(8, 1.0f), {4, 2}, stream);

    EXPECT_THROW(zeroInvalidBatchTail(values, 0, stream), std::logic_error);
    EXPECT_THROW(zeroInvalidBatchTail(values, 5, stream), std::logic_error);
    EXPECT_THROW(writeBatchValidityMask(values, 0, stream), std::logic_error);
    EXPECT_THROW(writeBatchValidityMask(values, 5, stream), std::logic_error);
}
