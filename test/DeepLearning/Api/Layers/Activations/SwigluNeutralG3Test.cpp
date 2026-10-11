#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Activations/Swiglu.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
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
constexpr uint64_t kInputWidth = 4;
constexpr uint64_t kOutputWidth = 2;
constexpr const char* kName = "neutral_swiglu_g3_training";

Impl::DynamicExpression squaredError(bool gradient) {
    const Impl::Expression prediction = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    const Impl::Expression label = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    const Impl::Expression delta = prediction - label;
    Impl::Expression out = gradient ? delta * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor())
                                    : delta * delta;
    out = out.withOutputDType(DataType::FP32);
    const auto definition = Impl::ExpressionDefinition::fromOutputs(
        Impl::Expression::outputs({{gradient ? "predictions_grad" : "loss", out}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

shared_ptr<Swiglu> findSwiglu(Network& network) {
    for (uint32_t index = 0; index < network.getNumLayers(); ++index) {
        auto activation = dynamic_pointer_cast<Swiglu>(network.getLayer(index));
        if (activation) return activation;
    }
    return nullptr;
}

shared_ptr<RaggedCustomLoss> findRawLoss(Network& network) {
    for (uint32_t index = 0; index < network.getNumLayers(); ++index) {
        auto loss = dynamic_pointer_cast<RaggedCustomLoss>(network.getLayer(index));
        if (loss) return loss;
    }
    return nullptr;
}

vector<float> downloadActive(const Impl::Tensor& tensor, uint64_t activeElements) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = tensor.clone(cpu);
    Impl::Tensor source = tensor;
    Stream stream = Stream::getNextDownloadStream(tensor.getPlacement().getDeviceNum());
    // Inactive packed gradient storage is unspecified. Do not transfer or inspect it.
    host.downloadSection(source, stream, 0, 0, activeElements * sizeof(float));
    stream.synchronize();
    const float* p = host.getMemPtr<float>();
    return vector<float>(p, p + activeElements);
}

float inputAt(uint64_t token, uint32_t channel) {
    // Alternating gate signs plus different value channels ensure both halves of
    // the Jacobian are exercised; an identity/no-gradient implementation fails.
    switch (channel) {
        case 0: return 0.63f + 0.17f * static_cast<float>(token);
        case 1: return -0.82f + 0.12f * static_cast<float>(token);
        case 2: return -1.14f + 0.19f * static_cast<float>(token);
        default: return 0.77f - 0.14f * static_cast<float>(token);
    }
}

float targetAt(uint64_t token, uint32_t channel) {
    return 0.11f + 0.09f * static_cast<float>((3 * token + channel) % 4)
                 - 0.15f * static_cast<float>(channel);
}

void runSavedTraining(const filesystem::path& archive, uint32_t b, uint64_t inputId,
                      uint64_t outputId, uint64_t partitionId) {
    SCOPED_TRACE(::testing::Message() << "placement B=" << b);
    Network loaded(kName);
    ASSERT_NO_THROW(loaded.load(archive.string()));
    auto inputs = loaded.getExternalRaggedNetworkInputs();
    ASSERT_EQ(inputs.size(), 2u);
    auto inputIt = find_if(inputs.begin(), inputs.end(), [](const auto& in) { return in.name == "tokens"; });
    auto labelIt = find_if(inputs.begin(), inputs.end(), [](const auto& in) { return in.name == "targets"; });
    ASSERT_NE(inputIt, inputs.end());
    ASSERT_NE(labelIt, inputs.end());
    const RaggedTensor tokens = inputIt->raggedTensor;
    const RaggedTensor targets = labelIt->raggedTensor;
    EXPECT_TRUE(tokens.sharesPartitionWith(targets));
    EXPECT_EQ(tokens.getValuesDimensions(), (vector<uint64_t>{kM, kInputWidth}));
    EXPECT_EQ(targets.getValuesDimensions(), (vector<uint64_t>{kM, kOutputWidth}));
    EXPECT_EQ(tokens.getValues().getOriginalId(), inputId);
    EXPECT_EQ(tokens.getRowPartitionToken().getOriginalId(), partitionId);
    const auto swiglu = findSwiglu(loaded);
    const auto loss = findRawLoss(loaded);
    ASSERT_NE(swiglu, nullptr);
    ASSERT_NE(loss, nullptr);
    ASSERT_TRUE(swiglu->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(swiglu->getFeatureOutput()->getOriginalId(), outputId);
    EXPECT_EQ(swiglu->getRaggedFeatureOutput()->getValuesDimensions(),
              (vector<uint64_t>{kM, kOutputWidth}));
    EXPECT_TRUE(swiglu->getRaggedFeatureOutput()->sharesPartitionWith(tokens));
    EXPECT_EQ(loss->getRaggedPredictions().getValues(), swiglu->getFeatureOutput().value());
    EXPECT_EQ(swiglu->architectureJson().at("version"), "1.1.0");
    EXPECT_EQ(loss->architectureJson().at("version"), "1.1.0");
    for (const auto& layer : {swiglu->architectureJson(), loss->architectureJson()}) {
        EXPECT_EQ(layer.dump().find("\"batch_size\""), string::npos);
        EXPECT_EQ(layer.dump().find("\"max_total_values\""), string::npos);
    }
    const json before = loaded.architectureJson();
    vector<Event> events;
    auto placed = loaded.place(b, events, /*inferenceOnly=*/false);
    ASSERT_NE(placed, nullptr);
    for (auto& event : events) event.synchronize();
    auto& stamped = placed->getStampedNetwork(0);
    const uint64_t capacity = b * kM;
    auto physical = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
        stamped.getPhysicalLayerFromApiLayer(swiglu->getId()));
    auto physicalLoss = dynamic_pointer_cast<Impl::RaggedCustomLoss>(
        stamped.getPhysicalLayerFromApiLayer(loss->getId()));
    ASSERT_NE(physical, nullptr) << "standalone ragged Swiglu must remain one expression-backed layer";
    ASSERT_NE(physicalLoss, nullptr);
    EXPECT_EQ(physicalLoss->getLogicalBatchSize(), b);
    EXPECT_EQ(physicalLoss->getMaxTotalValues(), capacity);
    auto managedCount = stamped.getManagedPartitionActiveCountInputForTest(tokens.getRowPartitionId());
    ASSERT_NE(managedCount, nullptr);
    ASSERT_TRUE(managedCount->getFeatureOutput().has_value());
    EXPECT_EQ(managedCount->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(stamped.getManagedPartitionOffsetsInputForTest(tokens.getRowPartitionId()), nullptr);
    const auto namedInput = stamped.getNamedInput("tokens.values");
    ASSERT_NE(namedInput, nullptr);
    ASSERT_TRUE(namedInput->getFeatureOutput().has_value());
    EXPECT_EQ(namedInput->getFeatureOutput()->getDimensions(), (vector<uint64_t>{capacity, kInputWidth}));
    const auto outputs = physical->getFeatureOutputs();
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].has_value());
    EXPECT_EQ(outputs[0]->getDimensions(), (vector<uint64_t>{capacity, kOutputWidth}));
    EXPECT_EQ(loaded.architectureJson(), before);

    vector<uint64_t> offsets(b + 1, 0);
    for (uint32_t row = 0; row < b; ++row)
        offsets[row + 1] = offsets[row] + (row == 0 || row % 3 == 0 ? 0 : (row % 2 == 0 ? 1 : 2));
    const uint64_t active = offsets.back();
    ASSERT_GT(active, 0u);
    ASSERT_LT(active, capacity);
    const auto desc = tokens.getDescriptorForPlacement(b);
    const auto labelDesc = targets.getDescriptorForPlacement(b);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor x(cpu, desc.getValuesDescriptor());
    Impl::Tensor t(cpu, labelDesc.getValuesDescriptor());
    Impl::Tensor off(cpu, desc.getOffsetsDescriptor());
    for (uint64_t token = 0; token < capacity; ++token) {
        for (uint32_t c = 0; c < kInputWidth; ++c)
            x.getMemPtr<float>()[token * kInputWidth + c] =
                token < active ? inputAt(token, c) : numeric_limits<float>::quiet_NaN();
        for (uint32_t c = 0; c < kOutputWidth; ++c)
            t.getMemPtr<float>()[token * kOutputWidth + c] =
                token < active ? targetAt(token, c) : numeric_limits<float>::quiet_NaN();
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
    map<string, Event> ready;
    auto done = placed->submitBatch(0, batch, results, ready, false);
    done.synchronize();
    ASSERT_TRUE(ready.contains("raw_loss"));
    ready.at("raw_loss").synchronize();
    placed->synchronize();
    ASSERT_TRUE(results.contains("raw_loss"));
    EXPECT_EQ(results.at("raw_loss").getDimensions(), (vector<uint64_t>{capacity, kOutputWidth}));
    const auto errors = physical->getErrorOutputs();
    // RaggedCustomLayer has two physical input ports: packed feature values at
    // index 0 and the DEVICE_ACTIVE_COUNT [1] partition carrier at index 1.
    // Only the packed-values port carries the Swiglu input gradient.
    ASSERT_EQ(errors.size(), 2u);
    ASSERT_TRUE(errors[0].has_value()) << "RAW terminal loss must differentiate through standalone Swiglu";
    EXPECT_EQ(errors[0]->getDataType(), DataType::FP32);
    EXPECT_EQ(errors[0]->getDimensions(), (vector<uint64_t>{capacity, kInputWidth}));
    ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
    const auto y = downloadActive(outputs[0].value(), active * kOutputWidth);
    const auto raw = downloadActive(results.at("raw_loss"), active * kOutputWidth);
    const auto dy = downloadActive(physicalLoss->getErrorOutput().value(), active * kOutputWidth);
    const auto dx = downloadActive(errors[0].value(), active * kInputWidth);
    ASSERT_EQ(y.size(), active * kOutputWidth);
    ASSERT_EQ(raw.size(), active * kOutputWidth);
    ASSERT_EQ(dy.size(), active * kOutputWidth);
    ASSERT_EQ(dx.size(), active * kInputWidth);
    bool nonzeroValueGradient = false, nonzeroGateGradient = false;
    for (uint64_t token = 0; token < active; ++token) {
        for (uint32_t c = 0; c < kOutputWidth; ++c) {
            const float value = inputAt(token, c);
            const float gate = inputAt(token, c + kOutputWidth);
            const float sigmoid = 1.0f / (1.0f + exp(-gate));
            const float swish = gate * sigmoid;
            const float expectedY = value * swish;
            const float delta = expectedY - targetAt(token, c);
            const float expectedDY = 2.0f * delta * Impl::Loss::getLossScalingFactor();
            const float expectedDV = expectedDY * swish;
            const float expectedDG = expectedDY * value * (sigmoid + gate * sigmoid * (1.0f - sigmoid));
            const uint64_t out = token * kOutputWidth + c;
            const uint64_t v = token * kInputWidth + c;
            const uint64_t g = v + kOutputWidth;
            EXPECT_NEAR(y[out], expectedY, 1e-4f) << "Swiglu forward " << out;
            EXPECT_NEAR(raw[out], delta * delta, 1e-4f) << "RAW loss " << out;
            EXPECT_NEAR(dy[out], expectedDY, 5e-3f) << "upstream derivative " << out;
            EXPECT_NEAR(dx[v], expectedDV, 7e-3f) << "value-half gradient " << v;
            EXPECT_NEAR(dx[g], expectedDG, 7e-3f) << "gate-half gradient " << g;
            nonzeroValueGradient |= abs(expectedDV) > 1e-4f;
            nonzeroGateGradient |= abs(expectedDG) > 1e-4f;
        }
    }
    EXPECT_TRUE(nonzeroValueGradient);
    EXPECT_TRUE(nonzeroGateGradient);
    // Packed capacity after the active prefix is deliberately never examined.
    EXPECT_EQ(loaded.architectureJson(), before);
}
} // namespace

TEST(SwigluNeutralG3, SameSavedStandaloneSwigluAndRawLossBackpropagateAtTwoAndSeven) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const auto archive = filesystem::temp_directory_path() /
        (string("thor_swiglu_g3_") + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    uint64_t inputId = 0, outputId = 0, partitionId = 0;
    {
        Network network(kName);
        const RaggedTensor tokens = RaggedNetworkInput::Builder().network(network).name("tokens")
            .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
            .trailingDimensions({kInputWidth}).maxValuesPerRow(kM).build();
        const RaggedTensor targets = RaggedNetworkInput::Builder().network(network).name("targets")
            .valuesDataType(DataType::FP32).trailingDimensions({kOutputWidth})
            .partition(tokens).build();
        // Test-only identity layer exposes dInput despite external NetworkInput
        // intentionally pruning its own error output. It preserves partition.
        const auto rivet = GradientRivet::Builder().network(network).tensor(tokens.getValues()).build();
        ASSERT_TRUE(rivet.architectureJson().is_object());
        const RaggedTensor observed = tokens.withValues(rivet.getFeatureOutput().value());
        auto swiglu = dynamic_pointer_cast<Swiglu>(
            Swiglu::Builder().network(network).featureInput(observed).build());
        ASSERT_NE(swiglu, nullptr);
        ASSERT_TRUE(swiglu->getRaggedFeatureOutput().has_value());
        const auto activated = swiglu->getRaggedFeatureOutput().value();
        const auto raw = RaggedCustomLoss::Builder().network(network)
            .predictions(activated).labels(targets)
            .lossExpression(squaredError(false)).gradientExpression(squaredError(true)).build();
        EXPECT_TRUE(raw.getRaggedRawLoss().sharesPartitionWith(tokens));
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        inputId = tokens.getValues().getOriginalId();
        outputId = activated.getValues().getOriginalId();
        partitionId = tokens.getRowPartitionToken().getOriginalId();
        EXPECT_EQ(swiglu->architectureJson().at("version"), "1.1.0");
        EXPECT_EQ(raw.architectureJson().at("version"), "1.1.0");
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u})
        runSavedTraining(archive, b, inputId, outputId, partitionId);
    filesystem::remove_all(archive);
}
