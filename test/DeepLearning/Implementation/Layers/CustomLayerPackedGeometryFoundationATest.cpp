#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/Expression/DynamicExpression.h"

#include "gtest/gtest.h"
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_FOR_GEOMETRY()                                                                            \
    do {                                                                                                       \
        int deviceCount = 0;                                                                                   \
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {                             \
            GTEST_SKIP() << "CUDA device required for physical CustomLayer geometry tests";                    \
        }                                                                                                      \
    } while (false)

const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

class PassiveEndpoint final : public Layer {
   private:
    void infer(std::optional<Tensor>, std::optional<Tensor>, Stream) override {}
    void backProp(std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>, Stream) override {}
};

// The geometry is evaluated during graph connection. No equation must be built,
// stamped, launched or allocated during these placement-only tests.
DynamicExpression neverExecutedExpression() {
    return DynamicExpression([](const DynamicExpression::TensorMap&,
                                const DynamicExpression::TensorMap&,
                                Stream&) -> DynamicExpressionBuild {
        throw std::runtime_error("Foundation-A geometry test must not stamp an expression");
    });
}

using Input = CustomLayer::PackedRaggedInputGeometry;
using Geometry = CustomLayer::PackedRaggedOutputGeometry;
using Output = CustomLayer::DeclaredOutputDescriptor;

Output packedOutput(uint64_t mQ = 3, uint64_t mKv = 5) {
    return Output{DataType::FP32,
                  {mQ, 2, 8},
                  false,
                  Geometry{3, mQ, {Input{0, 3, mQ}, Input{1, 4, mKv}, Input{2, 4, mKv}}}};
}

std::unique_ptr<CustomLayer> makeLayer(Output output = packedOutput(),
                                       std::vector<bool> fullDimensions = {true, true, true, true, true},
                                       std::vector<std::string> names = {"q", "k", "v", "q_offsets", "kv_offsets"}) {
    return std::make_unique<CustomLayer>(neverExecutedExpression(),
                                          std::move(names),
                                          std::vector<std::string>{"out"},
                                          gpuPlacement,
                                          std::vector<std::shared_ptr<PhysicalParameter>>{},
                                          true,
                                          -1,
                                          std::vector<Output>{std::move(output)},
                                          false,
                                          false,
                                          std::move(fullDimensions));
}

std::vector<Tensor> tensors(uint64_t batch,
                            uint64_t mQ = 3,
                            uint64_t mKv = 5,
                            DataType qOffsetType = DataType::UINT32,
                            uint64_t kvBatch = 0,
                            uint64_t qCapacityOverride = 0) {
    if (kvBatch == 0) kvBatch = batch;
    return {Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {qCapacityOverride == 0 ? batch * mQ : qCapacityOverride, 2, 8})),
            Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {batch * mKv, 2, 8})),
            Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {batch * mKv, 2, 8})),
            Tensor(gpuPlacement, TensorDescriptor(qOffsetType, {batch + 1})),
            Tensor(gpuPlacement, TensorDescriptor(DataType::UINT64, {kvBatch + 1}))};
}

std::vector<uint64_t> connectAndReturnOutput(CustomLayer& layer, const std::vector<Tensor>& input,
                                             const std::vector<uint32_t>& order) {
    Stream stream(0);
    std::vector<std::unique_ptr<PassiveEndpoint>> sources;
    for (uint32_t port : order) {
        sources.push_back(std::make_unique<PassiveEndpoint>());
        EXPECT_FALSE(layer.connectToPreviousLayer(sources.back().get(), input.at(port), stream, false,
                                                 static_cast<int>(port)).has_value());
    }
    PassiveEndpoint sink;
    layer.connectToNextLayer(&sink, 0, 0);
    const std::vector<std::optional<Tensor>> outputs = layer.getFeatureOutputs();
    EXPECT_EQ(outputs.size(), 1u);
    EXPECT_TRUE(outputs.at(0).has_value());
    return outputs.at(0).value().getDimensions();
}

TEST(CustomLayerPackedGeometryFoundationA, DifferentCanonicalPartitionsAndValuesOrOffsetsFirstB2B7) {
    REQUIRE_CUDA_FOR_GEOMETRY();
    for (uint64_t batch : {2ull, 7ull}) {
        for (const auto& order : {std::vector<uint32_t>{0, 1, 2, 3, 4},
                                  std::vector<uint32_t>{4, 3, 2, 1, 0}}) {
            auto layer = makeLayer();
            EXPECT_EQ(connectAndReturnOutput(*layer, tensors(batch), order),
                      (std::vector<uint64_t>{batch * 3, 2, 8}));
        }
    }
}

TEST(CustomLayerPackedGeometryFoundationA, RejectMismatchedBatchAndValueCapacities) {
    REQUIRE_CUDA_FOR_GEOMETRY();
    const std::vector<uint32_t> order{0, 1, 2, 3, 4};
    {
        auto layer = makeLayer();
        EXPECT_THROW(connectAndReturnOutput(*layer, tensors(2, 3, 5, DataType::UINT32, 7), order), std::runtime_error);
    }
    {
        auto layer = makeLayer();
        EXPECT_THROW(connectAndReturnOutput(*layer, tensors(7, 3, 5, DataType::UINT32, 0, 22), order), std::runtime_error);
    }
    {
        auto layer = makeLayer();
        EXPECT_THROW(connectAndReturnOutput(*layer, tensors(2, 3, 5, DataType::FP32), order), std::runtime_error);
    }
    {
        auto layer = makeLayer();
        auto inputs = tensors(2);
        inputs[3] = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT32, {2, 1}));
        EXPECT_THROW(connectAndReturnOutput(*layer, inputs, order), std::runtime_error);
    }
}

TEST(CustomLayerPackedGeometryFoundationA, RejectOverflowAndMisdeclaredSemanticOutput) {
    REQUIRE_CUDA_FOR_GEOMETRY();
    auto inputs = tensors(2);
    {
        Output huge = packedOutput();
        huge.featureDimensions.front() = std::numeric_limits<uint64_t>::max();
        huge.packedRaggedByPartition->maxValuesPerRow = huge.featureDimensions.front();
        // Keep Q's declared semantic M bound to the output partition;
        // the invalid geometry under test is physical B*M overflow.
        huge.packedRaggedByPartition->raggedInputs.front().maxValuesPerRow = huge.featureDimensions.front();
        auto layer = makeLayer(huge);
        EXPECT_THROW(connectAndReturnOutput(*layer, inputs, {0, 1, 2, 3, 4}), std::runtime_error);
    }
    {
        Output wrong = packedOutput();
        wrong.featureDimensions.front() = 4;
        EXPECT_THROW(makeLayer(wrong), std::runtime_error);
    }
    {
        Output wrong = packedOutput();
        wrong.dimensionsIncludeBatch = true;
        EXPECT_THROW(makeLayer(wrong), std::runtime_error);
    }
    {
        Output wrong = packedOutput();
        // Selecting KV offsets would incorrectly claim that Q's packed
        // output shares KV's partition. Reject at declaration time.
        wrong.packedRaggedByPartition->offsetsPort = 4;
        EXPECT_THROW(makeLayer(wrong), std::runtime_error);
    }
}

TEST(CustomLayerPackedGeometryFoundationA, MixedDenseInputMustAgreeWithOffsetsBatch) {
    REQUIRE_CUDA_FOR_GEOMETRY();
    Output output = packedOutput();
    const std::vector<bool> flags{true, true, true, true, true, false};
    const std::vector<std::string> names{"q", "k", "v", "q_offsets", "kv_offsets", "dense"};
    for (uint64_t denseBatch : {2ull, 7ull}) {
        auto inputs = tensors(2);
        inputs.emplace_back(gpuPlacement, TensorDescriptor(DataType::FP32, {denseBatch, 2, 8}));
        auto layer = makeLayer(output, flags, names);
        if (denseBatch == 2) {
            EXPECT_EQ(connectAndReturnOutput(*layer, inputs, {5, 4, 3, 2, 1, 0}),
                      (std::vector<uint64_t>{6, 2, 8}));
        } else {
            EXPECT_THROW(connectAndReturnOutput(*layer, inputs, {5, 4, 3, 2, 1, 0}), std::runtime_error);
        }
    }
}

TEST(CustomLayerPackedGeometryFoundationA, LegacyFullPhysicalAndImplicitBatchOutputBehaviorsUnchanged) {
    REQUIRE_CUDA_FOR_GEOMETRY();
    const auto inputs = tensors(2);
    {
        auto layer = makeLayer(Output{DataType::FP32, {11, 2}, true});
        EXPECT_EQ(connectAndReturnOutput(*layer, inputs, {0, 1, 2, 3, 4}),
                  (std::vector<uint64_t>{11, 2}));
    }
    {
        auto layer = makeLayer(Output{DataType::FP32, {9}, false},
                               {false, true, true, true, true});
        // Ordinarily batch-leading features must be B, not packed B*M.
        auto dense = tensors(2);
        dense[0] = Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2, 8}));
        EXPECT_EQ(connectAndReturnOutput(*layer, dense, {0, 1, 2, 3, 4}),
                  (std::vector<uint64_t>{2, 9}));
    }
}

}  // namespace
