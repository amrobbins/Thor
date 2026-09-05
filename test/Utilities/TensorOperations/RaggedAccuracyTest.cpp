#include "Utilities/TensorOperations/Ragged/RaggedAccuracy.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

TEST(RaggedAccuracy, BinaryUsesOnlyActiveTokensAndHonorsOffsetWidthsAndPartialRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t maxTotalValues = 9;
    const std::vector<float> predictionsHost{
        0.9f, 0.2f,  // only these two belong to valid rows
        0.8f, 0.1f, 0.7f, 0.9f, 0.9f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    const std::vector<uint64_t> labelsHost{1, 1, 1, 0, 0, 1, 1, 1, 1};

    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Tensor predictions = makeGpuTensor(predictionsHost, {maxTotalValues, 1}, stream, DataType::FP16);
        Tensor labels = makeGpuUnsignedTensor(labelsHost, {maxTotalValues, 1}, stream, DataType::UINT32);
        Tensor offsets = makeGpuUnsignedTensor({0, 2, 2, 5, 7}, {5}, stream, offsetsDType);
        Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
        Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

        raggedBinaryAccuracyStatistics(predictions, labels, offsets, correct, tokens, 2, maxTotalValues, stream);
        stream.synchronize();

        EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(correct, stream).front(), 1.0f);
        EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(tokens, stream).front(), 2.0f);
    }
}

TEST(RaggedAccuracy, CategoricalIndexArgmaxIsPerTokenAndIgnoresInactiveCapacity) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t maxTotalValues = 8;
    constexpr uint64_t numClasses = 3;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor predictions = makeGpuTensor(
        {
            9.0f, 1.0f, 0.0f,  // class 0, correct
            0.0f, 2.0f, 8.0f,  // class 2, wrong
            0.0f, 7.0f, 1.0f,  // class 1, correct
            6.0f, 5.0f, 4.0f,  // class 0, correct
            nan, nan, nan,
            nan, nan, nan,
            nan, nan, nan,
            nan, nan, nan,
        },
        {maxTotalValues, numClasses},
        stream,
        DataType::FP32);
    Tensor labels = makeGpuUnsignedTensor({0, 1, 1, 0, 99, 99, 99, 99}, {maxTotalValues, 1}, stream, DataType::UINT32);

    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Tensor offsets = makeGpuUnsignedTensor({0, 1, 1, 4, 4}, {5}, stream, offsetsDType);
        Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
        Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
        raggedCategoricalAccuracyStatistics(predictions,
                                            labels,
                                            offsets,
                                            correct,
                                            tokens,
                                            4,
                                            maxTotalValues,
                                            numClasses,
                                            RaggedCategoricalLabelFormat::CLASS_INDEX,
                                            stream);
        stream.synchronize();
        EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(correct, stream).front(), 3.0f);
        EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(tokens, stream).front(), 4.0f);
    }
}

TEST(RaggedAccuracy, CategoricalPerClassLabelsArgmaxWithinEachToken) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t maxTotalValues = 6;
    constexpr uint64_t numClasses = 3;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor predictions = makeGpuTensor(
        {
            0.9f, 0.1f, 0.0f,
            0.1f, 0.2f, 0.7f,
            0.2f, 0.6f, 0.2f,
            nan, nan, nan,
            nan, nan, nan,
            nan, nan, nan,
        },
        {maxTotalValues, numClasses},
        stream,
        DataType::FP32);
    Tensor labels = makeGpuTensor(
        {
            1.0f, 0.0f, 0.0f,  // correct
            0.0f, 1.0f, 0.0f,  // wrong
            0.0f, 0.9f, 0.1f,  // correct
            nan, nan, nan,
            nan, nan, nan,
            nan, nan, nan,
        },
        {maxTotalValues, numClasses},
        stream,
        DataType::FP16);
    Tensor offsets = makeGpuUnsignedTensor({0, 1, 1, 3}, {4}, stream, DataType::UINT32);
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    raggedCategoricalAccuracyStatistics(predictions,
                                        labels,
                                        offsets,
                                        correct,
                                        tokens,
                                        3,
                                        maxTotalValues,
                                        numClasses,
                                        RaggedCategoricalLabelFormat::PER_CLASS,
                                        stream);
    stream.synchronize();
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(correct, stream).front(), 2.0f);
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(tokens, stream).front(), 3.0f);
}
