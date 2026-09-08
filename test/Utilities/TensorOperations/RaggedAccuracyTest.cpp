#include "Utilities/TensorOperations/Ragged/RaggedAccuracy.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

TEST(RaggedAccuracy, BinaryUsesOnlyActivePackedPrefix) {
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

    Tensor predictions = makeGpuTensor(predictionsHost, {maxTotalValues, 1}, stream, DataType::FP16);
    Tensor labels = makeGpuUnsignedTensor(labelsHost, {maxTotalValues, 1}, stream, DataType::UINT32);
    Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues));
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    raggedBinaryAccuracyStatistics(predictions, labels, partials, correct, tokens, 2, maxTotalValues, stream);
    stream.synchronize();

    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(correct, stream).front(), 1.0f);
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(tokens, stream).front(), 2.0f);
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

    Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues, numClasses));
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    raggedCategoricalAccuracyStatistics(predictions,
                                        labels,
                                        partials,
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
    Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues, numClasses));
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    raggedCategoricalAccuracyStatistics(predictions,
                                        labels,
                                        partials,
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

TEST(RaggedAccuracy, CategoricalLaneGroupBoundariesPreserveFirstOccurrenceAndNanSemantics) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t activeValues = 5;
    constexpr uint64_t maxTotalValues = 7;
    const float nan = std::numeric_limits<float>::quiet_NaN();

    for (uint64_t numClasses : {2ULL, 3ULL, 4ULL, 5ULL, 8ULL, 9ULL, 16ULL, 17ULL, 32ULL, 33ULL, 65ULL}) {
        std::vector<float> predictions(maxTotalValues * numClasses, nan);
        std::vector<uint64_t> labels(activeValues, 0);

        // Token 0: class 0 ties the final class. First occurrence must win.
        predictions[0] = 7.0f;
        predictions[numClasses - 1] = 7.0f;
        labels[0] = 0;

        // Token 1: serial argmax initialized from class 0 keeps class 0 when
        // that initial value is NaN, regardless of later finite values.
        predictions[numClasses] = nan;
        predictions[numClasses + numClasses - 1] = 100.0f;
        labels[1] = 0;

        // Token 2: all -inf ties also choose the first class.
        const uint64_t token2Base = 2 * numClasses;
        for (uint64_t c = 0; c < numClasses; ++c)
            predictions[token2Base + c] = -std::numeric_limits<float>::infinity();
        labels[2] = 0;

        // Token 3: exercise the final class, including the >32 class loop.
        const uint64_t token3Base = 3 * numClasses;
        for (uint64_t c = 0; c < numClasses; ++c)
            predictions[token3Base + c] = static_cast<float>(c);
        labels[3] = numClasses - 1;

        // Token 4: an interior unique maximum.
        const uint64_t token4Base = 4 * numClasses;
        const uint64_t middle = numClasses / 2;
        for (uint64_t c = 0; c < numClasses; ++c)
            predictions[token4Base + c] = c == middle ? 50.0f : -static_cast<float>(c + 1);
        labels[4] = middle;

        Tensor predictionsGpu = makeGpuTensor(
            predictions, {maxTotalValues, numClasses}, stream, DataType::FP32);
        // Class-index label storage is capacity-shaped; poison inactive entries.
        Tensor labelsCapacity = makeGpuUnsignedTensor(
            {labels[0], labels[1], labels[2], labels[3], labels[4], 999, 999},
            {maxTotalValues, 1}, stream, DataType::UINT32);
        Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues, numClasses));
        Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
        Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

        raggedCategoricalAccuracyStatistics(predictionsGpu,
                                            labelsCapacity,
                                            partials,
                                            correct,
                                            tokens,
                                            activeValues,
                                            maxTotalValues,
                                            numClasses,
                                            RaggedCategoricalLabelFormat::CLASS_INDEX,
                                            stream);
        stream.synchronize();
        EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(correct, stream).front(), static_cast<float>(activeValues))
            << "numClasses=" << numClasses;
        EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(tokens, stream).front(), static_cast<float>(activeValues))
            << "numClasses=" << numClasses;
    }
}

TEST(RaggedAccuracy, CategoricalPerClassWideArgmaxUsesCooperativeClassReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t activeValues = 4;
    constexpr uint64_t maxTotalValues = 6;
    constexpr uint64_t numClasses = 33;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> predictions(maxTotalValues * numClasses, nan);
    std::vector<float> labels(maxTotalValues * numClasses, nan);

    for (uint64_t token = 0; token < activeValues; ++token) {
        const uint64_t predictedClass = (token * 11 + 7) % numClasses;
        const uint64_t labelClass = token == 2 ? (predictedClass + 1) % numClasses : predictedClass;
        for (uint64_t c = 0; c < numClasses; ++c) {
            predictions[token * numClasses + c] = c == predictedClass ? 10.0f : static_cast<float>(c) * 0.01f;
            labels[token * numClasses + c] = c == labelClass ? 20.0f : static_cast<float>(c) * 0.02f;
        }
    }

    Tensor predictionsGpu = makeGpuTensor(
        predictions, {maxTotalValues, numClasses}, stream, DataType::FP16);
    Tensor labelsGpu = makeGpuTensor(labels, {maxTotalValues, numClasses}, stream, DataType::FP32);
    Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues, numClasses));
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    raggedCategoricalAccuracyStatistics(predictionsGpu,
                                        labelsGpu,
                                        partials,
                                        correct,
                                        tokens,
                                        activeValues,
                                        maxTotalValues,
                                        numClasses,
                                        RaggedCategoricalLabelFormat::PER_CLASS,
                                        stream);
    stream.synchronize();
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(correct, stream).front(), 3.0f);
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(tokens, stream).front(), 4.0f);
}

TEST(RaggedAccuracy, MultiCtaBinaryReductionUsesUniquePartialsAndFinalReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t activeValues = 5000;
    constexpr uint64_t maxTotalValues = 5003;
    std::vector<float> predictions(maxTotalValues, 0.9f);
    std::vector<uint64_t> labels(maxTotalValues, 1);
    // Inactive capacity must not participate.
    predictions[activeValues] = std::numeric_limits<float>::quiet_NaN();
    labels[activeValues] = 0;

    Tensor predictionsGpu = makeGpuTensor(predictions, {maxTotalValues, 1}, stream, DataType::FP32);
    Tensor labelsGpu = makeGpuUnsignedTensor(labels, {maxTotalValues, 1}, stream, DataType::UINT8);
    Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues));
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    raggedBinaryAccuracyStatistics(predictionsGpu,
                                   labelsGpu,
                                   partials,
                                   correct,
                                   tokens,
                                   activeValues,
                                   maxTotalValues,
                                   stream);
    stream.synchronize();
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(correct, stream).front(), static_cast<float>(activeValues));
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(tokens, stream).front(), static_cast<float>(activeValues));
}

TEST(RaggedAccuracy, EmptyActivePrefixProducesZeroStatisticsWithoutReadingCapacity) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t maxTotalValues = 4;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor predictions = makeGpuTensor({nan, nan, nan, nan}, {maxTotalValues, 1}, stream, DataType::FP32);
    Tensor labels = makeGpuUnsignedTensor({255, 255, 255, 255}, {maxTotalValues, 1}, stream, DataType::UINT8);
    Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues));
    Tensor correct = makeGpuTensor({7.0f}, {1}, stream, DataType::FP32);
    Tensor tokens = makeGpuTensor({9.0f}, {1}, stream, DataType::FP32);

    raggedBinaryAccuracyStatistics(predictions, labels, partials, correct, tokens, 0, maxTotalValues, stream);
    stream.synchronize();
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(correct, stream).front(), 0.0f);
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(tokens, stream).front(), 0.0f);
}

TEST(RaggedAccuracy, MultiCtaCategoricalReductionHandlesWideClasses) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t activeValues = 100;
    constexpr uint64_t maxTotalValues = 103;
    constexpr uint64_t numClasses = 65;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> predictions(maxTotalValues * numClasses, nan);
    std::vector<uint64_t> labels(maxTotalValues, 999);

    for (uint64_t token = 0; token < activeValues; ++token) {
        const uint64_t target = (token * 17 + 3) % numClasses;
        labels[token] = target;
        for (uint64_t c = 0; c < numClasses; ++c)
            predictions[token * numClasses + c] = c == target ? 100.0f : static_cast<float>(c) * 0.001f;
    }

    Tensor predictionsGpu = makeGpuTensor(
        predictions, {maxTotalValues, numClasses}, stream, DataType::FP32);
    Tensor labelsGpu = makeGpuUnsignedTensor(labels, {maxTotalValues, 1}, stream, DataType::UINT32);
    Tensor partials(gpuPlacement, raggedAccuracyStatisticsWorkspaceDescriptor(maxTotalValues, numClasses));
    Tensor correct(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor tokens(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    raggedCategoricalAccuracyStatistics(predictionsGpu,
                                        labelsGpu,
                                        partials,
                                        correct,
                                        tokens,
                                        activeValues,
                                        maxTotalValues,
                                        numClasses,
                                        RaggedCategoricalLabelFormat::CLASS_INDEX,
                                        stream);
    stream.synchronize();
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(correct, stream).front(), static_cast<float>(activeValues));
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(tokens, stream).front(), static_cast<float>(activeValues));
}
