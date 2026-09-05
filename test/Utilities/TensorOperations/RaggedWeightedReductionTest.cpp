#include "Utilities/TensorOperations/Ragged/RaggedWeightedReduction.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

struct DTypePair {
    DataType values;
    DataType weights;
};

}  // namespace

TEST(RaggedWeightedReduction, ActivePrefixHonorsValidRowsPoisonedCapacityAndOffsetWidths) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::array<DTypePair, 5> dtypePairs{{
        {DataType::FP32, DataType::FP32},
        {DataType::FP16, DataType::BF16},
        {DataType::BF16, DataType::FP16},
        {DataType::FP8_E4M3, DataType::FP16},
        {DataType::FP8_E5M2, DataType::BF16},
    }};

    constexpr uint64_t batchSize = 4;
    constexpr uint64_t validRows = 2;
    constexpr uint64_t maxTotalValues = 9;
    constexpr uint64_t elementsPerValue = 2;
    const std::vector<float> hostValues{
        1.0f, 10.0f, 2.0f, 20.0f,
        1000.0f, 2000.0f, 3000.0f, 4000.0f, 5000.0f, 6000.0f,
        7000.0f, 8000.0f, 9000.0f, 10000.0f,
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
    };
    const std::vector<float> hostWeights{
        1.0f, 2.0f, 3.0f, 4.0f,
        50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f,
        110.0f, 120.0f, 130.0f, 140.0f,
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
    };

    for (DataType offsetsDType : {DataType::UINT32, DataType::UINT64}) {
        Tensor offsets = makeGpuUnsignedTensor({0, 2, 2, 5, 7}, {batchSize + 1}, stream, offsetsDType);
        for (const DTypePair dtypes : dtypePairs) {
            if (!CubSegmentedReduction::isInputDataTypeSupported(dtypes.values) ||
                !CubSegmentedReduction::isInputDataTypeSupported(dtypes.weights)) {
                continue;
            }
            Tensor values = makeGpuTensor(hostValues, {maxTotalValues, elementsPerValue}, stream, dtypes.values);
            Tensor weights = makeGpuTensor(hostWeights, {maxTotalValues, elementsPerValue}, stream, dtypes.weights);
            Tensor numerator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
            Tensor denominator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

            raggedWeightedMeanStatistics(values,
                                         weights,
                                         offsets,
                                         numerator,
                                         denominator,
                                         validRows,
                                         maxTotalValues,
                                         elementsPerValue,
                                         stream);
            stream.synchronize();

            // Compute the reference from the actually quantized storage so FP8,
            // FP16, and BF16 conversion differences do not weaken the test.
            const std::vector<float> storedValues = copyGpuTensorAsFloat(values, stream);
            const std::vector<float> storedWeights = copyGpuTensorAsFloat(weights, stream);
            float expectedNumerator = 0.0f;
            float expectedDenominator = 0.0f;
            for (size_t i = 0; i < 4; ++i) {
                expectedNumerator += storedValues[i] * storedWeights[i];
                expectedDenominator += storedWeights[i];
            }

            const std::vector<float> actualNumerator = copyGpuTensorAsFloat(numerator, stream);
            const std::vector<float> actualDenominator = copyGpuTensorAsFloat(denominator, stream);
            ASSERT_EQ(actualNumerator.size(), 1U);
            ASSERT_EQ(actualDenominator.size(), 1U);
            EXPECT_NEAR(actualNumerator.front(), expectedNumerator, 1.0e-4f);
            EXPECT_NEAR(actualDenominator.front(), expectedDenominator, 1.0e-4f);
            EXPECT_TRUE(std::isfinite(actualNumerator.front()));
            EXPECT_TRUE(std::isfinite(actualDenominator.front()));
        }
    }
}

TEST(RaggedWeightedReduction, ZeroWeightCanonicalizesNumeratorEvenWhenIgnoredValuesAreNan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor values = makeGpuTensor(
        {std::numeric_limits<float>::quiet_NaN(),
         std::numeric_limits<float>::quiet_NaN(),
         3.0f,
         4.0f,
         100.0f,
         200.0f},
        {3, 2},
        stream,
        DataType::FP32);
    Tensor weights =
        makeGpuTensor({0.0f, 0.0f, 0.0f, 0.0f, 9.0f, 9.0f}, {3, 2}, stream, DataType::FP32);
    Tensor offsets = makeGpuUnsignedTensor({0, 2, 3}, {3}, stream, DataType::UINT32);
    Tensor numerator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor denominator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    // Only row 0 is valid: its active weights are all zero and its values may
    // contain NaN because the batch has no WeightedMean contribution.
    raggedWeightedMeanStatistics(values, weights, offsets, numerator, denominator, 1, 3, 2, stream);
    stream.synchronize();

    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(numerator, stream).front(), 0.0f);
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(denominator, stream).front(), 0.0f);
}
