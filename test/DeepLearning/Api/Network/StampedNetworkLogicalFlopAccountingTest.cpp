#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/Layer.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace {

class StampedNetworkForLogicalFlopAccountingTest final : public ThorImplementation::StampedNetwork {
   public:
    void setPhysicalBatchCapacityForTest(uint64_t physicalBatchCapacity) { batchSize = physicalBatchCapacity; }
    void addOtherLayerForTest(ThorImplementation::Layer* layer) { otherLayers.push_back(layer); }
};

class FixedPerExampleFlopLayer final : public ThorImplementation::Layer {
   public:
    FixedPerExampleFlopLayer(uint64_t forwardFlopsPerExample,
                             uint64_t backwardFlopsPerExample)
        : forwardFlopsPerExample(forwardFlopsPerExample),
          backwardFlopsPerExample(backwardFlopsPerExample) {}

    uint64_t floatingPointOperationsPerExampleForward() override {
        return forwardFlopsPerExample;
    }

    uint64_t floatingPointOperationsPerExampleBackward() override {
        return backwardFlopsPerExample;
    }

   protected:
    void infer(std::optional<ThorImplementation::Tensor>,
               std::optional<ThorImplementation::Tensor>,
               Stream) override {}

    void backProp(std::optional<ThorImplementation::Tensor>,
                  std::optional<ThorImplementation::Tensor>,
                  std::optional<ThorImplementation::Tensor>,
                  Stream) override {}

   private:
    uint64_t forwardFlopsPerExample;
    uint64_t backwardFlopsPerExample;
};

class CountingLogicalWorkLayer final : public ThorImplementation::Layer {
   public:
    uint64_t floatingPointOperationsPerExampleForward() override {
        forwardFlopQueries += 1;
        return 7;
    }

    uint64_t floatingPointOperationsPerExampleBackward() override {
        backwardFlopQueries += 1;
        return 11;
    }

    uint64_t logicalByteCountForward(uint64_t validExampleCount) override {
        forwardByteQueries += 1;
        return 13 * validExampleCount;
    }

    uint64_t logicalByteCountBackward(uint64_t validExampleCount) override {
        backwardByteQueries += 1;
        return 17 * validExampleCount;
    }

    uint64_t forwardFlopQueries = 0;
    uint64_t backwardFlopQueries = 0;
    uint64_t forwardByteQueries = 0;
    uint64_t backwardByteQueries = 0;

   protected:
    void infer(std::optional<ThorImplementation::Tensor>,
               std::optional<ThorImplementation::Tensor>,
               Stream) override {}

    void backProp(std::optional<ThorImplementation::Tensor>,
                  std::optional<ThorImplementation::Tensor>,
                  std::optional<ThorImplementation::Tensor>,
                  Stream) override {}
};


class BoundaryLogicalByteLayer final : public ThorImplementation::Layer {
   public:
    BoundaryLogicalByteLayer(ThorImplementation::Tensor featureInput,
                             ThorImplementation::Tensor featureOutput,
                             ThorImplementation::Tensor errorInput,
                             ThorImplementation::Tensor errorOutput) {
        this->featureInput = std::move(featureInput);
        this->featureOutput = std::move(featureOutput);
        this->errorInput = std::move(errorInput);
        this->errorOutput = std::move(errorOutput);
    }

   protected:
    void infer(std::optional<ThorImplementation::Tensor>,
               std::optional<ThorImplementation::Tensor>,
               Stream) override {}

    void backProp(std::optional<ThorImplementation::Tensor>,
                  std::optional<ThorImplementation::Tensor>,
                  std::optional<ThorImplementation::Tensor>,
                  Stream) override {}
};

ThorImplementation::Tensor makeCpuTensor(ThorImplementation::DataType dataType,
                                         std::vector<uint64_t> dimensions) {
    ThorImplementation::TensorPlacement cpu(ThorImplementation::TensorPlacement::MemDevices::CPU);
    return ThorImplementation::Tensor(
        cpu, ThorImplementation::TensorDescriptor(dataType, std::move(dimensions)));
}

}  // namespace

TEST(StampedNetworkLogicalFlopAccounting, Lwa2dFixedPerExampleLayersUseValidExamplesNotPhysicalBatchCapacity) {
    StampedNetworkForLogicalFlopAccountingTest stamped;
    stamped.setPhysicalBatchCapacityForTest(8);

    FixedPerExampleFlopLayer first(/*forwardFlopsPerExample=*/7,
                                   /*backwardFlopsPerExample=*/11);
    FixedPerExampleFlopLayer second(/*forwardFlopsPerExample=*/3,
                                    /*backwardFlopsPerExample=*/5);
    stamped.addOtherLayerForTest(&first);
    stamped.addOtherLayerForTest(&second);

    EXPECT_EQ(stamped.getFloatingPointOperationsCurrentBatchForward(8), 80u);
    EXPECT_EQ(stamped.getFloatingPointOperationsCurrentBatchBackward(8), 128u);
    EXPECT_EQ(stamped.getFloatingPointOperationsCurrentBatchTraining(8), 208u);

    // The physical stamp still has capacity for eight examples, but only three
    // examples belong to this submitted semantic workload. Fixed per-example
    // logical work must therefore scale by three, not by padded capacity eight.
    EXPECT_EQ(stamped.getFloatingPointOperationsCurrentBatchForward(3), 30u);
    EXPECT_EQ(stamped.getFloatingPointOperationsCurrentBatchBackward(3), 48u);
    EXPECT_EQ(stamped.getFloatingPointOperationsCurrentBatchTraining(3), 78u);
}

TEST(StampedNetworkLogicalWorkAccounting, Lwa5aCombinedTrainingQueryCollectsFlopsAndBytesTogether) {
    StampedNetworkForLogicalFlopAccountingTest stamped;
    CountingLogicalWorkLayer layer;
    stamped.addOtherLayerForTest(&layer);

    const ThorImplementation::LogicalWorkCount work =
        stamped.getLogicalWorkCurrentBatchTraining(/*validExampleCount=*/3);

    EXPECT_EQ(work.floatingPointOperations, (7u + 11u) * 3u);
    EXPECT_EQ(work.bytes, (13u + 17u) * 3u);
    EXPECT_EQ(layer.forwardFlopQueries, 1u);
    EXPECT_EQ(layer.backwardFlopQueries, 1u);
    EXPECT_EQ(layer.forwardByteQueries, 1u);
    EXPECT_EQ(layer.backwardByteQueries, 1u);
}

TEST(StampedNetworkLogicalWorkAccounting, Lwa5aCombinedForwardQueryExcludesBackwardWork) {
    StampedNetworkForLogicalFlopAccountingTest stamped;
    CountingLogicalWorkLayer layer;
    stamped.addOtherLayerForTest(&layer);

    const ThorImplementation::LogicalWorkCount work =
        stamped.getLogicalWorkCurrentBatchForward(/*validExampleCount=*/3);

    EXPECT_EQ(work.floatingPointOperations, 7u * 3u);
    EXPECT_EQ(work.bytes, 13u * 3u);
    EXPECT_EQ(layer.forwardFlopQueries, 1u);
    EXPECT_EQ(layer.backwardFlopQueries, 0u);
    EXPECT_EQ(layer.forwardByteQueries, 1u);
    EXPECT_EQ(layer.backwardByteQueries, 0u);
}


TEST(StampedNetworkLogicalByteAccounting, Lwa3dOrdinaryLayerBoundaryUsesValidExamples) {
    StampedNetworkForLogicalFlopAccountingTest stamped;

    BoundaryLogicalByteLayer layer(
        makeCpuTensor(ThorImplementation::DataType::FP32, {8, 4}),
        makeCpuTensor(ThorImplementation::DataType::FP32, {8, 4}),
        makeCpuTensor(ThorImplementation::DataType::FP32, {8, 4}),
        makeCpuTensor(ThorImplementation::DataType::FP32, {8, 4}));
    stamped.addOtherLayerForTest(&layer);

    // 8 * 4 * 4 = 128 bytes per tensor. Forward and backward each have one
    // logical read and one logical write.
    EXPECT_EQ(stamped.getLogicalBytesCurrentBatchForward(8), 256u);
    EXPECT_EQ(stamped.getLogicalBytesCurrentBatchBackward(8), 256u);
    EXPECT_EQ(stamped.getLogicalBytesCurrentBatchTraining(8), 512u);

    // Three semantic examples use 3/8 of each batch-shaped tensor.
    EXPECT_EQ(stamped.getLogicalBytesCurrentBatchForward(3), 96u);
    EXPECT_EQ(stamped.getLogicalBytesCurrentBatchBackward(3), 96u);
    EXPECT_EQ(stamped.getLogicalBytesCurrentBatchTraining(3), 192u);
}

TEST(StampedNetworkLogicalByteAccounting, Lwa3dFixedSizeResultCountsOncePerBatch) {
    StampedNetworkForLogicalFlopAccountingTest stamped;

    BoundaryLogicalByteLayer layer(
        makeCpuTensor(ThorImplementation::DataType::FP32, {8, 4}),
        makeCpuTensor(ThorImplementation::DataType::FP32, {1}),
        makeCpuTensor(ThorImplementation::DataType::FP32, {8, 4}),
        makeCpuTensor(ThorImplementation::DataType::FP32, {8, 4}));
    stamped.addOtherLayerForTest(&layer);

    // Three active input examples are 48 bytes. The scalar result is one
    // logical 4-byte result for the batch, not 4 bytes times three examples.
    EXPECT_EQ(stamped.getLogicalBytesCurrentBatchForward(3), 52u);
}
