#include "DeepLearning/Implementation/Layers/Utility/RaggedGather.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedSequenceConcatenate.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedSequenceSlice.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace {

using namespace ThorImplementation;

const TensorPlacement kCpuPlacement(TensorPlacement::MemDevices::CPU);

Tensor makeTensor(const TensorDescriptor& descriptor) {
    return Tensor(kCpuPlacement, descriptor);
}

void publishPartition(const Tensor& carrier,
                      const RaggedTensorDescriptor& descriptor,
                      RowPartitionId id,
                      const std::vector<uint64_t>& offsets) {
    RowPartitionRuntime::publishHostState(carrier, descriptor.getRowPartition(), id, offsets);
}

class TestRaggedGather : public RaggedGather {
   public:
    using RaggedGather::RaggedGather;

    void setLogicalWorkTensors(Tensor source,
                               Tensor indices,
                               Tensor sourceOffsets,
                               Tensor indicesOffsets,
                               Tensor output,
                               std::optional<Tensor> upstream,
                               std::optional<Tensor> sourceGradient) {
        featureInputs[0] = std::move(source);
        featureInputs[1] = std::move(indices);
        featureInputs[2] = std::move(sourceOffsets);
        if (featureInputs.size() == 4) featureInputs[3] = std::move(indicesOffsets);
        featureOutputs[0] = std::move(output);
        errorInputs[0] = std::move(upstream);
        errorOutputs[0] = std::move(sourceGradient);
    }
};

class TestRaggedSequenceSlice : public RaggedSequenceSlice {
   public:
    using RaggedSequenceSlice::RaggedSequenceSlice;

    void setLogicalWorkTensors(Tensor input,
                               Tensor inputOffsets,
                               Tensor outputOffsets,
                               Tensor output,
                               std::optional<Tensor> upstream,
                               std::optional<Tensor> inputGradient) {
        featureInputs[0] = std::move(input);
        featureInputs[1] = std::move(inputOffsets);
        featureInputs[2] = std::move(outputOffsets);
        featureOutputs[0] = std::move(output);
        errorInputs[0] = std::move(upstream);
        errorOutputs[0] = std::move(inputGradient);
    }
};

class TestRaggedSequenceConcatenate : public RaggedSequenceConcatenate {
   public:
    using RaggedSequenceConcatenate::RaggedSequenceConcatenate;

    void setLogicalWorkTensors(std::vector<Tensor> values,
                               std::vector<Tensor> partitionCarriers,
                               Tensor output,
                               std::optional<Tensor> upstream,
                               std::vector<std::optional<Tensor>> gradients) {
        ASSERT_EQ(values.size() + partitionCarriers.size(), featureInputs.size());
        ASSERT_EQ(gradients.size(), values.size());
        for (uint32_t i = 0; i < values.size(); ++i) featureInputs[i] = std::move(values[i]);
        for (uint32_t i = 0; i < partitionCarriers.size(); ++i) {
            featureInputs[values.size() + i] = std::move(partitionCarriers[i]);
        }
        featureOutputs[0] = std::move(output);
        errorInputs[0] = std::move(upstream);
        for (uint32_t i = 0; i < gradients.size(); ++i) errorOutputs[i] = std::move(gradients[i]);
    }
};

TEST(RaggedLogicalWork, Lwa4e2GatherUsesSourceAndIndexPrefixesWithoutCountingOffsets) {
    constexpr uint64_t batchSize = 3;
    const auto logicalWork = [&](uint64_t sourceCapacity, uint64_t indexCapacity) {
        const RaggedTensorDescriptor sourceDescriptor(DataType::FP32, {2}, batchSize, sourceCapacity, DataType::UINT32);
        const RaggedTensorDescriptor indicesDescriptor(DataType::UINT32, {}, batchSize, indexCapacity, DataType::UINT32);
        const RaggedTensorDescriptor outputDescriptor(DataType::FP32, {2}, batchSize, indexCapacity, DataType::UINT32);

        Tensor source = makeTensor(sourceDescriptor.getValuesDescriptor());
        Tensor indices = makeTensor(indicesDescriptor.getValuesDescriptor());
        Tensor sourceOffsets = makeTensor(sourceDescriptor.getOffsetsDescriptor());
        Tensor indicesOffsets = makeTensor(indicesDescriptor.getOffsetsDescriptor());
        Tensor output = makeTensor(outputDescriptor.getValuesDescriptor());
        Tensor upstream = makeTensor(outputDescriptor.getValuesDescriptor());
        Tensor sourceGradient = makeTensor(sourceDescriptor.getValuesDescriptor());

        publishPartition(source, sourceDescriptor, 101, {0, 3, 5, 8});
        publishPartition(indices, indicesDescriptor, 102, {0, 2, 5, 6});
        // Deliberately leave the physical offsets tensors unpublished. Logical
        // work must use authoritative host state on values carriers only.

        TestRaggedGather layer(sourceDescriptor, indicesDescriptor, outputDescriptor, false);
        layer.setLogicalWorkTensors(source, indices, sourceOffsets, indicesOffsets, output, upstream, sourceGradient);

        std::vector<uint64_t> work{layer.logicalByteCountForward(0),
                                   layer.logicalByteCountBackward(0),
                                   layer.logicalByteCountForward(2),
                                   layer.logicalByteCountBackward(2)};
        publishPartition(indices, indicesDescriptor, 102, {0, 0, 0, 0});
        work.push_back(layer.logicalByteCountForward(0));
        work.push_back(layer.logicalByteCountBackward(0));
        return work;
    };

    const std::vector<uint64_t> expected{120, 136, 100, 100, 0, 64};
    EXPECT_EQ(logicalWork(10, 8), expected);
    EXPECT_EQ(logicalWork(20, 16), expected);
}

TEST(RaggedLogicalWork, Lwa4e2SequenceSliceUsesDistinctInputAndOutputPrefixes) {
    constexpr uint64_t batchSize = 3;
    const auto logicalWork = [&](uint64_t inputCapacity, uint64_t outputCapacity) {
        const RaggedTensorDescriptor inputDescriptor(DataType::FP32, {2}, batchSize, inputCapacity, DataType::UINT32);
        const RaggedTensorDescriptor outputDescriptor(DataType::FP32, {2}, batchSize, outputCapacity, DataType::UINT32);

        Tensor input = makeTensor(inputDescriptor.getValuesDescriptor());
        Tensor inputOffsets = makeTensor(inputDescriptor.getOffsetsDescriptor());
        Tensor outputOffsets = makeTensor(outputDescriptor.getOffsetsDescriptor());
        Tensor output = makeTensor(outputDescriptor.getValuesDescriptor());
        Tensor upstream = makeTensor(outputDescriptor.getValuesDescriptor());
        Tensor inputGradient = makeTensor(inputDescriptor.getValuesDescriptor());

        publishPartition(input, inputDescriptor, 201, {0, 4, 7, 9});
        publishPartition(output, outputDescriptor, 202, {0, 2, 4, 5});

        TestRaggedSequenceSlice layer(1, 2, inputDescriptor, outputDescriptor);
        layer.setLogicalWorkTensors(input, inputOffsets, outputOffsets, output, upstream, inputGradient);

        std::vector<uint64_t> work{layer.logicalByteCountForward(0),
                                   layer.logicalByteCountBackward(0),
                                   layer.logicalByteCountForward(2),
                                   layer.logicalByteCountBackward(2)};
        publishPartition(output, outputDescriptor, 202, {0, 0, 0, 0});
        work.push_back(layer.logicalByteCountForward(0));
        work.push_back(layer.logicalByteCountBackward(0));
        return work;
    };

    const std::vector<uint64_t> expected{80, 112, 64, 88, 0, 72};
    EXPECT_EQ(logicalWork(10, 6), expected);
    EXPECT_EQ(logicalWork(20, 12), expected);
}

TEST(RaggedLogicalWork, Lwa4e2SequenceConcatenateCountsOnlyAuthoredGradientRoutes) {
    constexpr uint64_t batchSize = 3;
    const auto logicalWork = [&](uint64_t firstCapacity, uint64_t secondCapacity) {
        const RaggedTensorDescriptor firstDescriptor(DataType::FP32, {2}, batchSize, firstCapacity, DataType::UINT32);
        const RaggedTensorDescriptor secondDescriptor(DataType::FP32, {2}, batchSize, secondCapacity, DataType::UINT32);
        const RaggedTensorDescriptor outputDescriptor(
            DataType::FP32, {2}, batchSize, firstCapacity + secondCapacity, DataType::UINT32);

        Tensor first = makeTensor(firstDescriptor.getValuesDescriptor());
        Tensor second = makeTensor(secondDescriptor.getValuesDescriptor());
        Tensor firstOffsets = makeTensor(firstDescriptor.getOffsetsDescriptor());
        Tensor secondOffsets = makeTensor(secondDescriptor.getOffsetsDescriptor());
        Tensor output = makeTensor(outputDescriptor.getValuesDescriptor());
        Tensor upstream = makeTensor(outputDescriptor.getValuesDescriptor());
        Tensor firstGradient = makeTensor(firstDescriptor.getValuesDescriptor());
        Tensor secondGradient = makeTensor(secondDescriptor.getValuesDescriptor());

        publishPartition(first, firstDescriptor, 301, {0, 2, 5, 6});
        publishPartition(second, secondDescriptor, 302, {0, 1, 1, 4});
        publishPartition(output, outputDescriptor, 303, {0, 3, 6, 10});

        TestRaggedSequenceConcatenate allGradients(2, 2, outputDescriptor);
        allGradients.setLogicalWorkTensors({first, second}, {firstOffsets, secondOffsets}, output, upstream,
                                           {firstGradient, secondGradient});

        TestRaggedSequenceConcatenate firstGradientOnly(2, 2, outputDescriptor);
        firstGradientOnly.setLogicalWorkTensors({first, second}, {firstOffsets, secondOffsets}, output, upstream,
                                                {firstGradient, std::nullopt});

        std::vector<uint64_t> work{allGradients.logicalByteCountForward(0),
                                   allGradients.logicalByteCountBackward(0),
                                   allGradients.logicalByteCountForward(2),
                                   allGradients.logicalByteCountBackward(2),
                                   firstGradientOnly.logicalByteCountBackward(0),
                                   firstGradientOnly.logicalByteCountBackward(2)};
        publishPartition(first, firstDescriptor, 301, {0, 0, 0, 0});
        publishPartition(second, secondDescriptor, 302, {0, 0, 0, 0});
        publishPartition(output, outputDescriptor, 303, {0, 0, 0, 0});
        work.push_back(allGradients.logicalByteCountForward(0));
        work.push_back(allGradients.logicalByteCountBackward(0));
        return work;
    };

    const std::vector<uint64_t> expected{160, 160, 96, 96, 96, 80, 0, 0};
    EXPECT_EQ(logicalWork(8, 6), expected);
    EXPECT_EQ(logicalWork(16, 12), expected);
}

}  // namespace
