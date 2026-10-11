#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Implementation/Layers/Activation/Softmax.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"
#include "cuda_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
namespace Impl = ThorImplementation;
using json = nlohmann::json;

namespace {
constexpr uint64_t kM = 4;
constexpr uint64_t kChannels = 3;
constexpr const char* kNetworkName = "standalone_softmax_s2_training";

Impl::DynamicExpression squaredDifference(bool gradient) {
    const Impl::Expression p = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    const Impl::Expression t = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    const Impl::Expression delta = p - t;
    Impl::Expression result = gradient
        ? delta * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor())
        : delta * delta;
    result = result.withOutputDType(DataType::FP32);
    const auto definition = Impl::ExpressionDefinition::fromOutputs(
        Impl::Expression::outputs({{gradient ? "predictions_grad" : "loss", result}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

shared_ptr<Softmax> findSoftmax(Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto layer = dynamic_pointer_cast<Softmax>(network.getLayer(i));
        if (layer) return layer;
    }
    return nullptr;
}

shared_ptr<RaggedCustomLoss> findRawLoss(Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto layer = dynamic_pointer_cast<RaggedCustomLoss>(network.getLayer(i));
        if (layer) return layer;
    }
    return nullptr;
}

vector<float> downloadActive(const Impl::Tensor& source, uint64_t activeTokens) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = source.clone(cpu);
    Stream stream = Stream::getNextDownloadStream(source.getPlacement().getDeviceNum());
    // Copy only the defined packed prefix; inactive gradient bytes are neither
    // transferred nor inspected. The full host descriptor is storage only.
    const uint64_t activeElements = activeTokens * kChannels;
    Impl::Tensor sourceHandle = source;
    host.downloadSection(sourceHandle, stream, 0, 0, activeElements * sizeof(float));
    stream.synchronize();
    const float* p = host.getMemPtr<float>();
    return vector<float>(p, p + activeElements);
}

void exerciseSavedTrainingGraph(const filesystem::path& archive, uint32_t b,
                                uint64_t originalInputId, uint64_t originalOutputId,
                                uint64_t originalPartitionId) {
    SCOPED_TRACE(::testing::Message() << "placement batch " << b);
    Network loaded(kNetworkName);
    ASSERT_NO_THROW(loaded.load(archive.string()));
    const auto external = loaded.getExternalRaggedNetworkInputs();
    ASSERT_EQ(external.size(), 2u);
    const auto tokenIt = find_if(external.begin(), external.end(), [](const auto& in) { return in.name == "tokens"; });
    const auto targetIt = find_if(external.begin(), external.end(), [](const auto& in) { return in.name == "targets"; });
    ASSERT_NE(tokenIt, external.end());
    ASSERT_NE(targetIt, external.end());
    const RaggedTensor tokens = tokenIt->raggedTensor;
    const RaggedTensor targets = targetIt->raggedTensor;
    EXPECT_TRUE(tokens.sharesPartitionWith(targets));
    EXPECT_EQ(tokens.getValuesDimensions(), (vector<uint64_t>{kM, kChannels}));
    EXPECT_EQ(tokens.getValues().getOriginalId(), originalInputId);
    EXPECT_EQ(tokens.getRowPartitionToken().getOriginalId(), originalPartitionId);

    const auto apiSoftmax = findSoftmax(loaded);
    const auto apiLoss = findRawLoss(loaded);
    ASSERT_NE(apiSoftmax, nullptr);
    ASSERT_NE(apiLoss, nullptr);
    ASSERT_TRUE(apiSoftmax->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(apiSoftmax->getRaggedFeatureOutput()->getValues().getOriginalId(), originalOutputId);
    ASSERT_TRUE(apiSoftmax->getRaggedFeatureInput().has_value());
    EXPECT_TRUE(apiSoftmax->getRaggedFeatureInput()->sharesPartitionWith(tokens));
    EXPECT_EQ(apiSoftmax->getRaggedFeatureInput()->getValuesDimensions(),
              (vector<uint64_t>{kM, kChannels}));
    EXPECT_EQ(apiLoss->getRaggedPredictions().getValues(), apiSoftmax->getFeatureOutput().value());
    EXPECT_EQ(apiLoss->getRaggedPredictions().getRowPartitionToken(), tokens.getRowPartitionToken());
    EXPECT_EQ(apiSoftmax->architectureJson().at("version"), "1.1.0");
    EXPECT_EQ(apiLoss->architectureJson().at("version"), "1.1.0");
    for (const auto& layer : {apiSoftmax->architectureJson(), apiLoss->architectureJson()}) {
        const string encoded = layer.dump();
        EXPECT_EQ(encoded.find("\"batch_size\""), string::npos);
        EXPECT_EQ(encoded.find("\"max_total_values\""), string::npos);
    }

    const json architectureBefore = loaded.architectureJson();
    vector<Event> init;
    auto placed = loaded.place(b, init, false); // false: actual training/backward, not inference.
    ASSERT_NE(placed, nullptr);
    for (Event& e : init) e.synchronize();
    auto& stamp = placed->getStampedNetwork(0);
    const uint64_t capacity = static_cast<uint64_t>(b) * kM;

    auto physicalSoftmax = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
        stamp.getPhysicalLayerFromApiLayer(apiSoftmax->getId()));
    auto physicalLoss = dynamic_pointer_cast<Impl::RaggedCustomLoss>(
        stamp.getPhysicalLayerFromApiLayer(apiLoss->getId()));
    ASSERT_NE(physicalSoftmax, nullptr) << "standalone ragged Softmax must use RaggedCustomLayer";
    ASSERT_NE(physicalLoss, nullptr);
    EXPECT_EQ(physicalLoss->getLogicalBatchSize(), b);
    EXPECT_EQ(physicalLoss->getMaxTotalValues(), capacity);
    EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(tokens.getRowPartitionId()), nullptr);
    const auto activeCarrier = stamp.getManagedPartitionActiveCountInputForTest(tokens.getRowPartitionId());
    ASSERT_NE(activeCarrier, nullptr);
    ASSERT_TRUE(activeCarrier->getFeatureOutput().has_value());
    EXPECT_EQ(activeCarrier->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
    ASSERT_TRUE(physicalLoss->getOffsetsInput().has_value());
    EXPECT_EQ(physicalLoss->getOffsetsInput()->getDimensions(), (vector<uint64_t>{1}));
    const auto physInput = stamp.getNamedInput("tokens.values");
    ASSERT_NE(physInput, nullptr);
    ASSERT_TRUE(physInput->getFeatureOutput().has_value());
    EXPECT_EQ(physInput->getFeatureOutput()->getDimensions(), (vector<uint64_t>{capacity, kChannels}));
    const auto featureOutputs = physicalSoftmax->getFeatureOutputs();
    ASSERT_EQ(featureOutputs.size(), 1u);
    ASSERT_TRUE(featureOutputs[0].has_value());
    EXPECT_EQ(featureOutputs[0]->getDimensions(), (vector<uint64_t>{capacity, kChannels}));
    EXPECT_EQ(loaded.architectureJson(), architectureBefore);

    // First row is empty. Some later rows are nonempty, with gaps between rows;
    // the packed active prefix is smaller than B*M for both B=2 and B=7.
    vector<uint64_t> offsets(b + 1, 0);
    for (uint32_t row = 0; row < b; ++row)
        offsets[row + 1] = offsets[row] + (row == 0 || row % 3 == 0 ? 0 : (row % 2 == 0 ? 1 : 2));
    const uint64_t active = offsets.back();
    ASSERT_GT(active, 0u);
    ASSERT_LT(active, capacity);

    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    const auto desc = tokens.getDescriptorForPlacement(b);
    Impl::Tensor x(cpu, desc.getValuesDescriptor());
    Impl::Tensor t(cpu, desc.getValuesDescriptor());
    Impl::Tensor off(cpu, desc.getOffsetsDescriptor());
    float* xValues = x.getMemPtr<float>();
    float* tValues = t.getMemPtr<float>();
    for (uint64_t token = 0; token < capacity; ++token) {
        for (uint64_t c = 0; c < kChannels; ++c) {
            const uint64_t i = token * kChannels + c;
            xValues[i] = token < active ? (0.19f * static_cast<float>(token) + 0.41f * static_cast<float>(c)
                                              - 0.17f * static_cast<float>((token + c) % 3))
                                         : numeric_limits<float>::quiet_NaN();
            tValues[i] = token < active ? (0.08f + 0.13f * static_cast<float>((2 * token + c) % 5)
                                               - 0.07f * static_cast<float>(c))
                                         : numeric_limits<float>::quiet_NaN();
        }
    }
    for (uint32_t i = 0; i <= b; ++i)
        off.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(offsets[i]);
    Impl::RowPartitionRuntime partition(off, desc.getRowPartition());
    partition.setHostOffsets(offsets);
    Batch batch;
    batch.insert("tokens", Impl::RaggedTensor(x, partition));
    batch.insert("targets", t);
    batch.setValidExampleCount(b);

    map<string, Impl::Tensor> results;
    map<string, Event> outputReady;
    Event done = placed->submitBatch(0, batch, results, outputReady, false);
    done.synchronize();
    ASSERT_TRUE(outputReady.contains("raw_loss"));
    outputReady.at("raw_loss").synchronize();
    placed->synchronize();
    ASSERT_TRUE(results.contains("raw_loss"));
    EXPECT_EQ(results.at("raw_loss").getDimensions(), (vector<uint64_t>{capacity, kChannels}));

    const auto softmaxInputGradients = physicalSoftmax->getErrorOutputs();
    ASSERT_GE(softmaxInputGradients.size(), 1u);
    ASSERT_TRUE(softmaxInputGradients[0].has_value())
        << "the terminal RAW loss must backpropagate through standalone Softmax";
    EXPECT_EQ(softmaxInputGradients[0]->getDataType(), DataType::FP32);
    EXPECT_EQ(softmaxInputGradients[0]->getDimensions(), (vector<uint64_t>{capacity, kChannels}));
    ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
    const vector<float> y = downloadActive(featureOutputs[0].value(), active);
    const vector<float> raw = downloadActive(results.at("raw_loss"), active);
    const vector<float> dy = downloadActive(physicalLoss->getErrorOutput().value(), active);
    const vector<float> dx = downloadActive(softmaxInputGradients[0].value(), active);
    ASSERT_EQ(y.size(), active * kChannels);
    ASSERT_EQ(raw.size(), active * kChannels);
    ASSERT_EQ(dy.size(), active * kChannels);
    ASSERT_EQ(dx.size(), active * kChannels);

    bool observedNonzero = false;
    for (uint64_t token = 0; token < active; ++token) {
        const uint64_t base = token * kChannels;
        const float maximum = *max_element(xValues + base, xValues + base + kChannels);
        float denominator = 0.0f;
        for (uint64_t c = 0; c < kChannels; ++c)
            denominator += exp(xValues[base + c] - maximum);
        float expectedY[kChannels];
        float expectedDY[kChannels];
        float contraction = 0.0f;
        for (uint64_t c = 0; c < kChannels; ++c) {
            const uint64_t i = base + c;
            expectedY[c] = exp(xValues[i] - maximum) / denominator;
            expectedDY[c] = 2.0f * (expectedY[c] - tValues[i]) * Impl::Loss::getLossScalingFactor();
            contraction += expectedDY[c] * expectedY[c];
            EXPECT_NEAR(y[i], expectedY[c], 6e-5f) << "Softmax output " << i;
            EXPECT_NEAR(raw[i], (expectedY[c] - tValues[i]) * (expectedY[c] - tValues[i]), 6e-5f)
                << "RAW loss " << i;
            EXPECT_NEAR(dy[i], expectedDY[c], 5e-3f) << "loss upstream gradient " << i;
        }
        for (uint64_t c = 0; c < kChannels; ++c) {
            const uint64_t i = base + c;
            const float expectedDX = expectedY[c] * (expectedDY[c] - contraction);
            observedNonzero = observedNonzero || abs(expectedDX) > 1e-4f;
            EXPECT_NEAR(dx[i], expectedDX, 6e-3f) << "standalone Softmax backward JVP " << i;
        }
    }
    EXPECT_TRUE(observedNonzero) << "the selected nonconstant upstream gradient must exercise backward";
    // Inactive packed capacity is deliberately unspecified; never read or assert its values.
    EXPECT_EQ(loaded.architectureJson(), architectureBefore);
}
} // namespace

TEST(SoftmaxNeutralS2, OneSavedStandaloneSoftmaxAndRawLossTrainAtTwoAndSeven) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const auto archive = filesystem::temp_directory_path() / ("thor_s2_" + to_string(nonce));
    filesystem::remove_all(archive);
    uint64_t inputId = 0, softmaxOutputId = 0, partitionId = 0;
    {
        Network network(kNetworkName);
        const RaggedTensor tokens = RaggedNetworkInput::Builder().network(network).name("tokens")
            .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
            .trailingDimensions({kChannels}).maxValuesPerRow(kM).build();
        const RaggedTensor targets = RaggedNetworkInput::Builder().network(network).name("targets")
            .valuesDataType(DataType::FP32).trailingDimensions({kChannels}).partition(tokens).build();
        // An external NetworkInput deliberately discards upstream errors. The
        // established test-only GradientRivet is a transparent identity edge
        // that requests dInput so the *standalone* Softmax JVP is observable.
        // It preserves the canonical partition and semantic M on both sides.
        const auto rivet = GradientRivet::Builder().network(network).tensor(tokens.getValues()).build();
        // Architecture-only Network::save() calls architectureJson(), not the
        // obsolete serialize(storageDir, stream) helper signature.  A null
        // rivet layer would make Network::load() throw before placement.
        const json rivetJson = rivet.architectureJson();
        ASSERT_TRUE(rivetJson.is_object());
        EXPECT_EQ(rivetJson.at("layer_type"), "gradient_rivet");
        EXPECT_EQ(rivetJson.at("input_tensor").at("id"), tokens.getValues().getId());
        EXPECT_EQ(rivetJson.at("output_tensor").at("id"), rivet.getFeatureOutput()->getId());
        const RaggedTensor observedTokens = tokens.withValues(rivet.getFeatureOutput().value());
        const auto softmax = dynamic_pointer_cast<Softmax>(
            Softmax::Builder().network(network).featureInput(observedTokens).build());
        ASSERT_NE(softmax, nullptr);
        ASSERT_TRUE(softmax->getRaggedFeatureOutput().has_value());
        const RaggedTensor activated = softmax->getRaggedFeatureOutput().value();
        auto raw = RaggedCustomLoss::Builder().network(network)
            .predictions(activated).labels(targets)
            .lossExpression(squaredDifference(false)).gradientExpression(squaredDifference(true)).build();
        EXPECT_TRUE(raw.getRaggedRawLoss().sharesPartitionWith(tokens));
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        inputId = tokens.getValues().getOriginalId();
        partitionId = tokens.getRowPartitionToken().getOriginalId();
        softmaxOutputId = activated.getValues().getOriginalId();
        EXPECT_EQ(softmax->architectureJson().at("version"), "1.1.0");
        EXPECT_EQ(raw.architectureJson().at("version"), "1.1.0");
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u})
        exerciseSavedTrainingGraph(archive, b, inputId, softmaxOutputId, partitionId);
    filesystem::remove_all(archive);
}

// CCE owns a *different* backward contract from an ordinary standalone Softmax.
// Its dense support-layer is a physical Softmax with loss-owned backward; ragged
// CCE feeds logits directly into a RAW loss without an API Softmax support layer.
TEST(SoftmaxNeutralS2, DenseCceExternalBackwardAndRaggedCceLogitsNativeStaySeparate) {
    Network dense("s2_dense_cce_policy");
    const Tensor logits = NetworkInput::Builder().network(dense).name("logits")
        .dimensions({kChannels}).dataType(DataType::FP32).build().getFeatureOutput().value();
    const Tensor labels = NetworkInput::Builder().network(dense).name("labels")
        .dimensions({kChannels}).dataType(DataType::FP32).build().getFeatureOutput().value();
    const auto denseLoss = CategoricalCrossEntropy::Builder().network(dense)
        .predictions(logits).labels(labels).reportsRawLoss().build();
    (void)NetworkOutput::Builder().network(dense).name("dense_loss")
        .inputTensor(denseLoss.getLoss()).dataType(DataType::FP32).build();
    const auto internalSoftmax = findSoftmax(dense);
    ASSERT_NE(internalSoftmax, nullptr);
    EXPECT_EQ(internalSoftmax->architectureJson().at("version"), "1.0.0");
    int cudaDevices = 0;
    if (cudaGetDeviceCount(&cudaDevices) == cudaSuccess && cudaDevices > 0) {
        vector<Event> init;
        const auto placed = dense.place(2, init, false);
        ASSERT_NE(placed, nullptr);
        for (Event& e : init) e.synchronize();
        const auto physical = dynamic_pointer_cast<Impl::Softmax>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(internalSoftmax->getId()));
        ASSERT_NE(physical, nullptr);
        EXPECT_TRUE(physical->isBackwardComputedExternally());
    }

    Network ragged("s2_ragged_cce_policy");
    const RaggedTensor raggedLogits = RaggedNetworkInput::Builder().network(ragged).name("logits")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({kChannels}).maxValuesPerRow(kM).build();
    const RaggedTensor raggedLabels = RaggedNetworkInput::Builder().network(ragged).name("labels")
        .valuesDataType(DataType::FP32).trailingDimensions({kChannels}).partition(raggedLogits).build();
    const auto raggedLoss = CategoricalCrossEntropy::Builder().network(ragged)
        .predictions(raggedLogits).labels(raggedLabels).reportsRawLoss().build();
    EXPECT_TRUE(raggedLoss.getRaggedLoss().sharesPartitionWith(raggedLogits));
    uint32_t apiSoftmaxCount = 0;
    uint32_t rawLossCount = 0;
    for (uint32_t i = 0; i < ragged.getNumLayers(); ++i) {
        apiSoftmaxCount += dynamic_pointer_cast<Softmax>(ragged.getLayer(i)) != nullptr;
        rawLossCount += dynamic_pointer_cast<RaggedCustomLoss>(ragged.getLayer(i)) != nullptr;
    }
    EXPECT_EQ(apiSoftmaxCount, 0u);
    EXPECT_EQ(rawLossCount, 1u);
}
