#include "DeepLearning/Api/Layers/Utility/Einsum.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Utility/EinsumLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;
using Impl::DataType;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t numel(const Impl::Tensor& tensor) {
    uint64_t result = 1;
    for (uint64_t dimension : tensor.getDimensions())
        result *= dimension;
    return result;
}

void writeCpuTensor(Impl::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(numel(tensor), values.size());
    auto* data = static_cast<float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        data[i] = values[i];
}

vector<float> readCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);
    vector<float> values(numel(tensor));
    const auto* data = static_cast<const float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = data[i];
    return values;
}

Impl::Tensor copyToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor host = tensor.clone(cpuPlacement);
    host.copyFromAsync(tensor, stream);
    stream.synchronize();
    return host;
}

void synchronize(vector<Event>& events) {
    for (Event& event : events)
        event.synchronize();
    events.clear();
}

void expectAllClose(const vector<float>& actual, const vector<float>& expected, float tolerance = 2.0e-3f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
}

vector<float> batchedMatmul(const vector<float>& lhs,
                            const vector<float>& rhs,
                            uint32_t batch,
                            uint32_t m,
                            uint32_t k,
                            uint32_t n) {
    vector<float> result(static_cast<size_t>(batch) * m * n, 0.0f);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (uint32_t x = 0; x < k; ++x)
                    sum += lhs[(b * m + i) * k + x] * rhs[(b * k + x) * n + j];
                result[(b * m + i) * n + j] = sum;
            }
        }
    }
    return result;
}

vector<float> lhsGradientReference(const vector<float>& upstream,
                                   const vector<float>& rhs,
                                   uint32_t batch,
                                   uint32_t m,
                                   uint32_t k,
                                   uint32_t n) {
    vector<float> result(static_cast<size_t>(batch) * m * k, 0.0f);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t x = 0; x < k; ++x) {
                float sum = 0.0f;
                for (uint32_t j = 0; j < n; ++j)
                    sum += upstream[(b * m + i) * n + j] * rhs[(b * k + x) * n + j];
                result[(b * m + i) * k + x] = sum;
            }
        }
    }
    return result;
}

vector<float> rhsGradientReference(const vector<float>& lhs,
                                   const vector<float>& upstream,
                                   uint32_t batch,
                                   uint32_t m,
                                   uint32_t k,
                                   uint32_t n) {
    vector<float> result(static_cast<size_t>(batch) * k * n, 0.0f);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t x = 0; x < k; ++x) {
            for (uint32_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (uint32_t i = 0; i < m; ++i)
                    sum += lhs[(b * m + i) * k + x] * upstream[(b * m + i) * n + j];
                result[(b * k + x) * n + j] = sum;
            }
        }
    }
    return result;
}

}  // namespace

TEST(EinsumApi, BuilderInfersOutputShapeAndClonePreservesEquationAndOperandOrder) {
    Api::Network network("einsum_build");
    Api::Tensor lhs(DataType::FP32, {2, 3});
    Api::Tensor rhs(DataType::FP32, {3, 4});

    Api::Einsum einsum = Api::Einsum::Builder()
                             .network(network)
                             .equation("ik,kj->ij")
                             .featureInput(lhs)
                             .featureInput(rhs)
                             .build();

    ASSERT_TRUE(einsum.isInitialized());
    EXPECT_EQ(einsum.getEquation(), "ik,kj->ij");
    ASSERT_EQ(einsum.getFeatureInputs().size(), 2u);
    EXPECT_EQ(einsum.getFeatureInputs()[0], lhs);
    EXPECT_EQ(einsum.getFeatureInputs()[1], rhs);
    ASSERT_TRUE(einsum.getFeatureOutput().has_value());
    EXPECT_EQ(einsum.getFeatureOutput()->getDimensions(), (vector<uint64_t>{2, 4}));
    EXPECT_EQ(einsum.getFeatureOutput()->getDataType(), DataType::FP32);

    shared_ptr<Api::Layer> cloneLayer = einsum.clone();
    auto clone = std::dynamic_pointer_cast<Api::Einsum>(cloneLayer);
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getId(), einsum.getId());
    EXPECT_EQ(clone->getEquation(), einsum.getEquation());
    EXPECT_EQ(clone->getFeatureInputs(), einsum.getFeatureInputs());
    EXPECT_EQ(clone->getFeatureOutputs(), einsum.getFeatureOutputs());
}

TEST(EinsumApi, BuilderResolvesFeatureEllipsisAndBroadcastWithoutExposingRuntimeBatch) {
    Api::Network network("einsum_ellipsis");
    Api::Tensor lhs(DataType::BF16, {2, 5, 3});
    Api::Tensor rhs(DataType::BF16, {3, 4});

    Api::Einsum einsum = Api::Einsum::Builder()
                             .network(network)
                             .equation("i...j,jk->i...k")
                             .featureInputs({lhs, rhs})
                             .build();

    ASSERT_TRUE(einsum.getFeatureOutput().has_value());
    EXPECT_EQ(einsum.getFeatureOutput()->getDimensions(), (vector<uint64_t>{2, 5, 4}));
    EXPECT_EQ(einsum.getFeatureOutput()->getDataType(), DataType::BF16);
}

TEST(EinsumApi, BuilderRejectsOperandCountDtypeAndShapeContractViolations) {
    Api::Network network("einsum_validation");

    EXPECT_THROW((void)Api::Einsum::Builder()
                     .network(network)
                     .equation("ik,kj->ij")
                     .featureInput(Api::Tensor(DataType::FP32, {2, 3}))
                     .build(),
                 std::invalid_argument);

    EXPECT_THROW((void)Api::Einsum::Builder()
                     .network(network)
                     .equation("ik,kj->ij")
                     .featureInput(Api::Tensor(DataType::FP32, {2, 3}))
                     .featureInput(Api::Tensor(DataType::BF16, {3, 4}))
                     .build(),
                 std::invalid_argument);

    EXPECT_THROW((void)Api::Einsum::Builder()
                     .network(network)
                     .equation("ik,kj->ij")
                     .featureInput(Api::Tensor(DataType::INT32, {2, 3}))
                     .featureInput(Api::Tensor(DataType::INT32, {3, 4}))
                     .build(),
                 std::invalid_argument);

    EXPECT_THROW((void)Api::Einsum::Builder()
                     .network(network)
                     .equation("ii->i")
                     .featureInput(Api::Tensor(DataType::FP32, {2, 3}))
                     .build(),
                 std::invalid_argument);
}

TEST(EinsumApi, DuplicateSymbolicInputRotatesPhysicalOperandBindingsDeterministically) {
    Api::Network network("einsum_duplicate_connections");
    Api::Tensor input(DataType::FP32, {3, 3});

    Api::Einsum einsum = Api::Einsum::Builder()
                             .network(network)
                             .equation("ij,ij->ij")
                             .featureInput(input)
                             .featureInput(input)
                             .build();

    EXPECT_EQ(einsum.getConnectionType(input), 0);
    EXPECT_EQ(einsum.getConnectionType(input), 1);
    EXPECT_EQ(einsum.getConnectionType(input), 0);
    einsum.resetGraphTraversalState();
    EXPECT_EQ(einsum.getConnectionType(input), 0);
    EXPECT_EQ(einsum.getConnectionType(input), 1);

    einsum.resetGraphTraversalState();
    einsum.informThatInputConnectionMade(input);
    ASSERT_EQ(einsum.getOutputsFromInput(input).size(), 1u);
    EXPECT_TRUE(einsum.getOutputsFromInput(input).empty());
}

TEST(EinsumApi, ArchitectureRoundTripPreservesEquationAndDuplicateOperandOccurrences) {
    Api::Network original("einsum_serialization_original");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(original)
                                  .name("x")
                                  .dimensions({3, 3})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::Tensor x = input.getFeatureOutput().value();
    Api::Einsum einsum = Api::Einsum::Builder()
                             .network(original)
                             .equation("ij,ij->ij")
                             .featureInput(x)
                             .featureInput(x)
                             .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(original)
                                    .name("y")
                                    .inputTensor(einsum.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    const nlohmann::json inputJson = input.architectureJson();
    const nlohmann::json einsumJson = einsum.architectureJson();
    const nlohmann::json outputJson = output.architectureJson();
    EXPECT_EQ(einsumJson.at("layer_type"), "einsum");
    EXPECT_EQ(einsumJson.at("equation"), "ij,ij->ij");
    ASSERT_EQ(einsumJson.at("inputs").size(), 2u);
    EXPECT_EQ(einsumJson.at("inputs")[0].at("id"), einsumJson.at("inputs")[1].at("id"));

    Api::Network restored("einsum_serialization_restored");
    Api::NetworkInput::deserialize(inputJson, &restored);
    std::shared_ptr<thor_file::TarReader> unusedArchiveReader;
    Api::Layer::deserialize(unusedArchiveReader, einsumJson, &restored);
    Api::NetworkOutput::deserialize(outputJson, &restored);

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = restored.place(2, initDoneEvents, /*inferenceOnly=*/true);
    synchronize(initDoneEvents);
    ASSERT_NE(placed, nullptr);

    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    shared_ptr<Impl::EinsumLayer> physicalEinsum;
    for (const shared_ptr<Impl::Layer>& layer : stamped.getOtherLayers()) {
        auto candidate = std::dynamic_pointer_cast<Impl::EinsumLayer>(layer);
        if (candidate != nullptr) {
            ASSERT_EQ(physicalEinsum, nullptr);
            physicalEinsum = candidate;
        }
    }
    ASSERT_NE(physicalEinsum, nullptr);
    EXPECT_EQ(physicalEinsum->getEquation(), "ij,ij->ij");
    EXPECT_EQ(physicalEinsum->getExpectedNumInputs(), 2u);
    ASSERT_EQ(physicalEinsum->getFeatureInputs().size(), 2u);
    ASSERT_TRUE(physicalEinsum->getFeatureInputs()[0].has_value());
    ASSERT_TRUE(physicalEinsum->getFeatureInputs()[1].has_value());
}

TEST(EinsumApi, PlacedLayerRunsBatchedForwardAndBackwardThroughRegularGraphConnections) {
    constexpr uint32_t batch = 2;
    constexpr uint32_t m = 2;
    constexpr uint32_t k = 3;
    constexpr uint32_t n = 2;

    Api::Network network("einsum_forward_backward");
    Api::NetworkInput lhsInput = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("lhs")
                                     .dimensions({m, k})
                                     .dataType(DataType::FP32)
                                     .build();
    Api::NetworkInput rhsInput = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("rhs")
                                     .dimensions({k, n})
                                     .dataType(DataType::FP32)
                                     .build();
    Api::GradientRivet lhsRivet =
        Api::GradientRivet::Builder().network(network).tensor(lhsInput.getFeatureOutput().value()).build();
    Api::GradientRivet rhsRivet =
        Api::GradientRivet::Builder().network(network).tensor(rhsInput.getFeatureOutput().value()).build();
    Api::Einsum einsum = Api::Einsum::Builder()
                             .network(network)
                             .equation("ik,kj->ij")
                             .featureInput(lhsRivet.getFeatureOutput().value())
                             .featureInput(rhsRivet.getFeatureOutput().value())
                             .build();
    Api::GradientRivet outputRivet =
        Api::GradientRivet::Builder().network(network).tensor(einsum.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = network.place(batch, initDoneEvents, /*inferenceOnly=*/false);
    synchronize(initDoneEvents);
    ASSERT_NE(placed, nullptr);

    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    auto physicalLhsInput = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(lhsInput.getId()));
    auto physicalRhsInput = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(rhsInput.getId()));
    auto physicalOutput = std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(output.getId()));
    auto physicalEinsum = std::dynamic_pointer_cast<Impl::EinsumLayer>(stamped.getPhysicalLayerFromApiLayer(einsum.getId()));
    ASSERT_NE(physicalLhsInput, nullptr);
    ASSERT_NE(physicalRhsInput, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(physicalEinsum, nullptr);

    const vector<float> lhs = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        2.0f, 0.0f, 1.0f,
        1.0f, 3.0f, 2.0f,
    };
    const vector<float> rhs = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 2.0f,
        2.0f, 1.0f,
        0.0f, 1.0f,
    };
    Impl::Tensor lhsHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batch, m, k}));
    Impl::Tensor rhsHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batch, k, n}));
    writeCpuTensor(lhsHost, lhs);
    writeCpuTensor(rhsHost, rhs);

    physicalLhsInput->forward(lhsHost, false, batch);
    physicalRhsInput->forward(rhsHost, false, batch);
    physicalOutput->getOutputReadyEvent().synchronize();

    Stream forwardStream = physicalEinsum->getStream();
    ASSERT_EQ(physicalEinsum->getFeatureOutputs().size(), 1u);
    ASSERT_TRUE(physicalEinsum->getFeatureOutputs()[0].has_value());
    const vector<float> actualForward =
        readCpuTensor(copyToCpu(physicalEinsum->getFeatureOutputs()[0].value(), forwardStream));
    expectAllClose(actualForward, batchedMatmul(lhs, rhs, batch, m, k, n));

    ASSERT_EQ(physicalEinsum->getErrorInputs().size(), 1u);
    ASSERT_TRUE(physicalEinsum->getErrorInputs()[0].has_value());
    ASSERT_EQ(physicalEinsum->getErrorOutputs().size(), 2u);
    ASSERT_TRUE(physicalEinsum->getErrorOutputs()[0].has_value());
    ASSERT_TRUE(physicalEinsum->getErrorOutputs()[1].has_value());

    const vector<float> upstream = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        -1.0f, 0.5f,
        2.0f, -0.25f,
    };
    Impl::Tensor errorInput = physicalEinsum->getErrorInputs()[0].value();
    Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    writeCpuTensor(errorInputHost, upstream);
    errorInput.copyFromAsync(errorInputHost, forwardStream);
    physicalEinsum->backward(errorInput, batch);

    vector<Stream> backwardStreams = physicalEinsum->getStreams();
    ASSERT_EQ(backwardStreams.size(), 2u);
    const vector<float> actualLhsGradient =
        readCpuTensor(copyToCpu(physicalEinsum->getErrorOutputs()[0].value(), backwardStreams[0]));
    const vector<float> actualRhsGradient =
        readCpuTensor(copyToCpu(physicalEinsum->getErrorOutputs()[1].value(), backwardStreams[1]));

    expectAllClose(actualLhsGradient, lhsGradientReference(upstream, rhs, batch, m, k, n));
    expectAllClose(actualRhsGradient, rhsGradientReference(lhs, upstream, batch, m, k, n));
}

TEST(EinsumApi, DuplicateSymbolicOperandPlacesAsTwoPhysicalOperandsWithLiveBackwardTerms) {
    constexpr uint32_t batch = 2;
    Api::Network network("einsum_duplicate_placement");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("x")
                                  .dimensions({2, 2})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::GradientRivet inputRivet =
        Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::Tensor x = inputRivet.getFeatureOutput().value();
    Api::Einsum einsum = Api::Einsum::Builder()
                             .network(network)
                             .equation("ij,ij->ij")
                             .featureInput(x)
                             .featureInput(x)
                             .build();
    Api::GradientRivet outputRivet =
        Api::GradientRivet::Builder().network(network).tensor(einsum.getFeatureOutput().value()).build();
    (void)Api::NetworkOutput::Builder()
        .network(network)
        .name("y")
        .inputTensor(outputRivet.getFeatureOutput().value())
        .dataType(DataType::FP32)
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = network.place(batch, initDoneEvents, /*inferenceOnly=*/false);
    synchronize(initDoneEvents);
    ASSERT_NE(placed, nullptr);

    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    auto physicalEinsum = std::dynamic_pointer_cast<Impl::EinsumLayer>(stamped.getPhysicalLayerFromApiLayer(einsum.getId()));
    ASSERT_NE(physicalEinsum, nullptr);
    ASSERT_EQ(physicalEinsum->getFeatureInputs().size(), 2u);
    ASSERT_TRUE(physicalEinsum->getFeatureInputs()[0].has_value());
    ASSERT_TRUE(physicalEinsum->getFeatureInputs()[1].has_value());
    EXPECT_NE(physicalEinsum->getStreams()[0].getId(), physicalEinsum->getStreams()[1].getId());

    ASSERT_EQ(physicalEinsum->getErrorOutputs().size(), 2u);
    EXPECT_TRUE(physicalEinsum->getErrorOutputs()[0].has_value());
    EXPECT_TRUE(physicalEinsum->getErrorOutputs()[1].has_value());
}

TEST(EinsumApi, ThreeStageForwardThenBackwardPropagatesThroughEveryEinsum) {
    constexpr uint32_t batch = 2;
    constexpr uint32_t m = 2;
    constexpr uint32_t k1 = 3;
    constexpr uint32_t k2 = 4;
    constexpr uint32_t k3 = 3;
    constexpr uint32_t n = 2;

    Api::Network network("einsum_three_stage_forward_backward");
    Api::NetworkInput xInput = Api::NetworkInput::Builder()
                                   .network(network)
                                   .name("x")
                                   .dimensions({m, k1})
                                   .dataType(DataType::FP32)
                                   .build();
    Api::NetworkInput w1Input = Api::NetworkInput::Builder()
                                    .network(network)
                                    .name("w1")
                                    .dimensions({k1, k2})
                                    .dataType(DataType::FP32)
                                    .build();
    Api::NetworkInput w2Input = Api::NetworkInput::Builder()
                                    .network(network)
                                    .name("w2")
                                    .dimensions({k2, k3})
                                    .dataType(DataType::FP32)
                                    .build();
    Api::NetworkInput w3Input = Api::NetworkInput::Builder()
                                    .network(network)
                                    .name("w3")
                                    .dimensions({k3, n})
                                    .dataType(DataType::FP32)
                                    .build();

    Api::GradientRivet xRivet =
        Api::GradientRivet::Builder().network(network).tensor(xInput.getFeatureOutput().value()).build();
    Api::GradientRivet w1Rivet =
        Api::GradientRivet::Builder().network(network).tensor(w1Input.getFeatureOutput().value()).build();
    Api::GradientRivet w2Rivet =
        Api::GradientRivet::Builder().network(network).tensor(w2Input.getFeatureOutput().value()).build();
    Api::GradientRivet w3Rivet =
        Api::GradientRivet::Builder().network(network).tensor(w3Input.getFeatureOutput().value()).build();

    Api::Einsum stage1 = Api::Einsum::Builder()
                             .network(network)
                             .equation("ik,kj->ij")
                             .featureInput(xRivet.getFeatureOutput().value())
                             .featureInput(w1Rivet.getFeatureOutput().value())
                             .build();
    Api::Einsum stage2 = Api::Einsum::Builder()
                             .network(network)
                             .equation("ij,jk->ik")
                             .featureInput(stage1.getFeatureOutput().value())
                             .featureInput(w2Rivet.getFeatureOutput().value())
                             .build();
    Api::Einsum stage3 = Api::Einsum::Builder()
                             .network(network)
                             .equation("ij,jk->ik")
                             .featureInput(stage2.getFeatureOutput().value())
                             .featureInput(w3Rivet.getFeatureOutput().value())
                             .build();
    Api::GradientRivet outputRivet =
        Api::GradientRivet::Builder().network(network).tensor(stage3.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed = network.place(batch, initDoneEvents, /*inferenceOnly=*/false);
    synchronize(initDoneEvents);
    ASSERT_NE(placed, nullptr);

    Impl::StampedNetwork& stamped = placed->getStampedNetwork(0);
    auto physicalX = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(xInput.getId()));
    auto physicalW1 = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(w1Input.getId()));
    auto physicalW2 = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(w2Input.getId()));
    auto physicalW3 = std::dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(w3Input.getId()));
    auto physicalOutput =
        std::dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(output.getId()));
    auto physicalStage1 =
        std::dynamic_pointer_cast<Impl::EinsumLayer>(stamped.getPhysicalLayerFromApiLayer(stage1.getId()));
    auto physicalStage2 =
        std::dynamic_pointer_cast<Impl::EinsumLayer>(stamped.getPhysicalLayerFromApiLayer(stage2.getId()));
    auto physicalStage3 =
        std::dynamic_pointer_cast<Impl::EinsumLayer>(stamped.getPhysicalLayerFromApiLayer(stage3.getId()));
    ASSERT_NE(physicalX, nullptr);
    ASSERT_NE(physicalW1, nullptr);
    ASSERT_NE(physicalW2, nullptr);
    ASSERT_NE(physicalW3, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(physicalStage1, nullptr);
    ASSERT_NE(physicalStage2, nullptr);
    ASSERT_NE(physicalStage3, nullptr);

    const auto deterministic = [](size_t count, int offset) {
        vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            const int centered = static_cast<int>((i + static_cast<size_t>(offset)) % 11) - 5;
            values[i] = static_cast<float>(centered) * 0.125f;
        }
        return values;
    };

    const vector<float> x = deterministic(batch * m * k1, 1);
    const vector<float> w1 = deterministic(batch * k1 * k2, 3);
    const vector<float> w2 = deterministic(batch * k2 * k3, 5);
    const vector<float> w3 = deterministic(batch * k3 * n, 7);
    const vector<float> upstream = deterministic(batch * m * n, 9);

    Impl::Tensor xHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batch, m, k1}));
    Impl::Tensor w1Host(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batch, k1, k2}));
    Impl::Tensor w2Host(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batch, k2, k3}));
    Impl::Tensor w3Host(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batch, k3, n}));
    writeCpuTensor(xHost, x);
    writeCpuTensor(w1Host, w1);
    writeCpuTensor(w2Host, w2);
    writeCpuTensor(w3Host, w3);

    // Deliberately submit the independent inputs in a non-topological order.
    // Each EinsumLayer must rendezvous on its own operands before forwarding,
    // and the three layers must then execute as a regular graph chain.
    physicalW3->forward(w3Host, false, batch);
    physicalW2->forward(w2Host, false, batch);
    physicalX->forward(xHost, false, batch);
    physicalW1->forward(w1Host, false, batch);
    physicalOutput->getOutputReadyEvent().synchronize();

    const vector<float> y1 = batchedMatmul(x, w1, batch, m, k1, k2);
    const vector<float> y2 = batchedMatmul(y1, w2, batch, m, k2, k3);
    const vector<float> y3 = batchedMatmul(y2, w3, batch, m, k3, n);
    Stream stage3Stream = physicalStage3->getStream();
    ASSERT_EQ(physicalStage3->getFeatureOutputs().size(), 1u);
    ASSERT_TRUE(physicalStage3->getFeatureOutputs()[0].has_value());
    expectAllClose(readCpuTensor(copyToCpu(physicalStage3->getFeatureOutputs()[0].value(), stage3Stream)), y3, 3.0e-3f);

    ASSERT_EQ(physicalStage3->getErrorInputs().size(), 1u);
    ASSERT_TRUE(physicalStage3->getErrorInputs()[0].has_value());
    Impl::Tensor finalError = physicalStage3->getErrorInputs()[0].value();
    Impl::Tensor finalErrorHost = finalError.clone(cpuPlacement);
    writeCpuTensor(finalErrorHost, upstream);
    finalError.copyFromAsync(finalErrorHost, stage3Stream);

    // One backward call at the tail must recursively traverse stage3 -> stage2
    // -> stage1 and populate all four source gradients.
    physicalStage3->backward(finalError, batch);

    const vector<float> dY2 = lhsGradientReference(upstream, w3, batch, m, k3, n);
    const vector<float> dW3 = rhsGradientReference(y2, upstream, batch, m, k3, n);
    const vector<float> dY1 = lhsGradientReference(dY2, w2, batch, m, k2, k3);
    const vector<float> dW2 = rhsGradientReference(y1, dY2, batch, m, k2, k3);
    const vector<float> dX = lhsGradientReference(dY1, w1, batch, m, k1, k2);
    const vector<float> dW1 = rhsGradientReference(x, dY1, batch, m, k1, k2);

    ASSERT_EQ(physicalStage1->getErrorOutputs().size(), 2u);
    ASSERT_EQ(physicalStage2->getErrorOutputs().size(), 2u);
    ASSERT_EQ(physicalStage3->getErrorOutputs().size(), 2u);
    for (const auto& error : physicalStage1->getErrorOutputs()) ASSERT_TRUE(error.has_value());
    for (const auto& error : physicalStage2->getErrorOutputs()) ASSERT_TRUE(error.has_value());
    for (const auto& error : physicalStage3->getErrorOutputs()) ASSERT_TRUE(error.has_value());

    vector<Stream> stage1Streams = physicalStage1->getStreams();
    vector<Stream> stage2Streams = physicalStage2->getStreams();
    vector<Stream> stage3Streams = physicalStage3->getStreams();
    expectAllClose(readCpuTensor(copyToCpu(physicalStage1->getErrorOutputs()[0].value(), stage1Streams[0])), dX, 4.0e-3f);
    expectAllClose(readCpuTensor(copyToCpu(physicalStage1->getErrorOutputs()[1].value(), stage1Streams[1])), dW1, 4.0e-3f);
    expectAllClose(readCpuTensor(copyToCpu(physicalStage2->getErrorOutputs()[1].value(), stage2Streams[1])), dW2, 4.0e-3f);
    expectAllClose(readCpuTensor(copyToCpu(physicalStage3->getErrorOutputs()[1].value(), stage3Streams[1])), dW3, 4.0e-3f);
}
