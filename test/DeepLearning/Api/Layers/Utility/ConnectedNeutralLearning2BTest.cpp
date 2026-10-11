// SERIALIZATION-OWNERSHIP-2B: one saved, genuinely connected neutral ragged
// learning graph; all gradient-bearing operators here have accepted backward.
// The original 2B graph remains unchanged; LN-H adds a separate trainable LayerNorm graph.
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Layers/Utility/LayerNorm.h"
#include "DeepLearning/Api/Layers/Utility/Add.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedRMSNorm.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Stream.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>
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
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
constexpr uint64_t M2B = 4, H2B = 3, W2B = 2 * H2B;

Impl::DynamicExpression squaredError2B(bool derivative) {
    auto p = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    auto y = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    auto delta = p - y;
    auto result = derivative
        ? delta * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor())
        : delta * delta;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            derivative ? "predictions_grad" : "loss",
            result.withOutputDType(DataType::FP32)}})));
}

struct Stack2B {
    RaggedTensor tokens, aux, labels;
    FullyConnected fc;
    RMSNorm rms;
    Add add;
    Concatenate joined;
    RaggedCustomLoss raw;
    RaggedLossShaper reporting;
};

Stack2B buildStack2B(Network& n) {
    // The two input values and their label share exactly one canonical partition.
    auto x = RaggedNetworkInput::Builder().network(n).name("tokens")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({H2B}).maxValuesPerRow(M2B).build();
    auto a = RaggedNetworkInput::Builder().network(n).name("aux")
        .valuesDataType(DataType::FP32).trailingDimensions({H2B}).partition(x).build();
    auto labels = RaggedNetworkInput::Builder().network(n).name("labels")
        .valuesDataType(DataType::FP32).trailingDimensions({W2B}).partition(x).build();

    // Rivets are test-only backward sinks: the external data inputs otherwise
    // legitimately prune gradients before an integration test can inspect them.
    auto xRivet = GradientRivet::Builder().network(n).tensor(x.getValues()).build();
    auto aRivet = GradientRivet::Builder().network(n).tensor(a.getValues()).build();
    auto xBackprop = x.withValues(xRivet.getFeatureOutput().value());
    auto aBackprop = a.withValues(aRivet.getFeatureOutput().value());

    auto fc = FullyConnected::Builder().network(n).featureInput(xBackprop)
        .numOutputFeatures(H2B).hasBias(true).weightsDataType(DataType::FP32)
        .computeDataType(DataType::FP32).outputDataType(DataType::FP32)
        .noActivation().build();
    auto projection = fc.getRaggedFeatureOutput().value();
    auto rms = RMSNorm::Builder().network(n).featureInput(projection)
        .normalizedShape({H2B}).epsilon(1e-5f).parameterDataType(DataType::FP32)
        .epilogueInput("auxiliary", aBackprop)
        .epilogue(RMSNorm::epilogueInput() +
                  RMSNorm::epilogueAuxInput("auxiliary") * Impl::Expression(0.5))
        .build();
    auto add = Add::Builder().network(n).left(rms.getRaggedFeatureOutput().value())
        .right(aBackprop).build();
    auto joined = Concatenate::Builder().network(n)
        .featureInput(add.getRaggedFeatureOutput().value())
        .featureInput(projection).concatenationAxis(0).build();
    auto raw = RaggedCustomLoss::Builder().network(n)
        .predictions(joined.getRaggedFeatureOutput().value()).labels(labels)
        .lossExpression(squaredError2B(false)).gradientExpression(squaredError2B(true))
        .build();
    auto reporting = RaggedLossShaper::Builder().network(n)
        .lossInput(raw.getRaggedRawLoss()).reportsPerExampleLoss().build();
    // BOTH forward consumers must remain; the raw loss must still own its
    // training root after the physical TensorFanout is introduced.
    (void)NetworkOutput::Builder().network(n).name("raw_loss")
        .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
    (void)NetworkOutput::Builder().network(n).name("report")
        .inputTensor(reporting.getLossOutput()).dataType(DataType::FP32).build();
    return {x, a, labels, fc, rms, add, joined, raw, reporting};
}

template <typename T> shared_ptr<T> find2B(Network& n) {
    for (uint32_t i = 0; i < n.getNumLayers(); ++i)
        if (auto layer = dynamic_pointer_cast<T>(n.getLayer(i))) return layer;
    return {};
}

vector<uint64_t> rowOffsets2B(uint32_t b) {
    vector<uint64_t> offsets(b + 1, 0);
    for (uint32_t row = 0; row < b; ++row)
        offsets[row + 1] = offsets[row] + (row % 3 == 0 ? 0U : row % 3 == 1 ? 2U : 1U);
    return offsets;
}

Batch makeBatch2B(const RaggedTensor& x, uint32_t b) {
    const auto d = x.getDescriptorForPlacement(b);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor tokens(cpu, d.getValuesDescriptor());
    Impl::Tensor aux(cpu, d.getValuesDescriptor());
    Impl::Tensor labels(cpu, Impl::TensorDescriptor(DataType::FP32, {b * M2B, W2B}));
    Impl::Tensor ends(cpu, d.getOffsetsDescriptor());
    const auto offsets = rowOffsets2B(b);
    for (uint64_t i = 0; i < b * M2B; ++i) {
        for (uint32_t f = 0; f < H2B; ++f) {
            tokens.getMemPtr<float>()[i * H2B + f] = i < offsets.back()
                ? 0.3f + 0.14f * static_cast<float>(i) - 0.09f * static_cast<float>(f)
                : numeric_limits<float>::quiet_NaN();
            aux.getMemPtr<float>()[i * H2B + f] = i < offsets.back()
                ? 0.2f - 0.04f * static_cast<float>(i) + 0.07f * static_cast<float>(f)
                : numeric_limits<float>::quiet_NaN();
        }
        for (uint32_t f = 0; f < W2B; ++f)
            labels.getMemPtr<float>()[i * W2B + f] = i < offsets.back()
                ? 0.05f * static_cast<float>(f % 3)
                : numeric_limits<float>::quiet_NaN();
    }
    for (uint32_t i = 0; i <= b; ++i) ends.getMemPtr<uint64_t>()[i] = offsets[i];
    Impl::RowPartitionRuntime partition(ends, d.getRowPartition());
    partition.setHostOffsets(offsets);
    Batch data;
    data.insert("tokens", Impl::RaggedTensor(tokens, partition));
    data.insert("aux", Impl::RaggedTensor(aux, partition));
    data.insert("labels", Impl::RaggedTensor(labels, partition));
    data.setValidExampleCount(b);
    return data;
}

vector<float> readActive2B(const Impl::Tensor& device, uint64_t count) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = device.clone(cpu);
    Impl::Tensor source = device;
    Stream stream = Stream::getNextDownloadStream(device.getPlacement().getDeviceNum());
    host.downloadSection(source, stream, 0, 0, count * sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + count);
}

bool anyDifferent2B(const vector<float>& a, const vector<float>& b, float tolerance = 1e-8f) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); ++i)
        if (isfinite(a[i]) && isfinite(b[i]) && fabsf(a[i] - b[i]) > tolerance) return true;
    return false;
}
} // namespace

TEST(ConnectedNeutralLearning2B, GraphUsesOneCanonicalPartitionAndStrictNeutralArtifacts) {
    Network network("connected_2b_semantic");
    auto stack = buildStack2B(network);
    EXPECT_TRUE(stack.aux.sharesPartitionWith(stack.tokens));
    EXPECT_TRUE(stack.labels.sharesPartitionWith(stack.tokens));
    for (const auto& result : {stack.fc.getRaggedFeatureOutput().value(),
                               stack.rms.getRaggedFeatureOutput().value(),
                               stack.add.getRaggedFeatureOutput().value(),
                               stack.joined.getRaggedFeatureOutput().value(),
                               stack.raw.getRaggedRawLoss()}) {
        EXPECT_TRUE(result.sharesPartitionWith(stack.tokens));
        EXPECT_EQ(result.getRowPartitionToken(), stack.tokens.getRowPartitionToken());
    }
    EXPECT_EQ(stack.joined.getRaggedFeatureOutput()->getValuesDimensions(),
              (vector<uint64_t>{M2B, W2B}));
    EXPECT_EQ(stack.fc.architectureJson().at("version"), "1.3.0");
    EXPECT_EQ(stack.rms.architectureJson().at("version"), "1.1.0");
    EXPECT_EQ(stack.add.architectureJson().at("version"), "1.1.0");
    EXPECT_EQ(stack.joined.architectureJson().at("version"), "1.1.0");
    EXPECT_EQ(stack.raw.architectureJson().at("version"), "1.1.0");
    EXPECT_EQ(stack.reporting.architectureJson().at("version"), "1.1.0");
    for (uint32_t b : {2U, 7U}) {
        EXPECT_EQ(stack.joined.getRaggedFeatureOutput()->getDescriptorForPlacement(b).getMaxTotalValues(), b * M2B);
        EXPECT_EQ(stack.raw.getRaggedRawLoss().getDescriptorForPlacement(b).getMaxTotalValues(), b * M2B);
    }
}

TEST(ConnectedNeutralLearning2B, SavedB2AndB7TrainWithRawReportingFanoutAndActiveGradients) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const string name = "connected_neutral_training_2b";
    const auto archive = filesystem::temp_directory_path() /
        (name + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    {
        Network network(name);
        (void)Sgd::Builder().network(network).initialLearningRate(0.001f)
            .decay(0.f).momentum(0.f).build();
        (void)buildStack2B(network);
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network network(name);
        ASSERT_NO_THROW(network.load(archive.string()));
        const json before = network.architectureJson();
        auto fc = find2B<FullyConnected>(network);
        auto rms = find2B<RMSNorm>(network);
        auto loss = find2B<RaggedCustomLoss>(network);
        ASSERT_NE(fc, nullptr);
        ASSERT_NE(rms, nullptr);
        ASSERT_NE(loss, nullptr);
        auto inputs = network.getExternalRaggedNetworkInputs();
        auto token = find_if(inputs.begin(), inputs.end(), [](const auto& entry) {
            return entry.name == "tokens";
        });
        ASSERT_NE(token, inputs.end());
        vector<Event> events;
        auto placed = network.place(b, events, false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto& stamped = placed->getStampedNetwork(0);
        auto fcPhysical = dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            stamped.getPhysicalLayerFromApiLayer(fc->getId()));
        auto rmsPhysical = dynamic_pointer_cast<Impl::RaggedRMSNorm>(
            stamped.getPhysicalLayerFromApiLayer(rms->getId()));
        auto rawPhysical = dynamic_pointer_cast<Impl::RaggedCustomLoss>(
            stamped.getPhysicalLayerFromApiLayer(loss->getId()));
        ASSERT_NE(fcPhysical, nullptr);
        ASSERT_NE(rmsPhysical, nullptr);
        ASSERT_NE(rawPhysical, nullptr);
        EXPECT_TRUE(rawPhysical->isTrainingActive());
        // Reporting requires genuine per-row boundaries, not just a scalar
        // active-count carrier. The RAW terminal loss itself needs only the
        // active count and must retain its semantic training root.
        EXPECT_NE(stamped.getManagedPartitionOffsetsInputForTest(
            token->raggedTensor.getRowPartitionId()), nullptr);
        auto next = rawPhysical->getNextLayer();
        ASSERT_TRUE(next.has_value());
        EXPECT_NE(dynamic_cast<Impl::TensorFanout*>(next.value()), nullptr);
        ASSERT_TRUE(rawPhysical->getErrorOutput().has_value());
        ASSERT_TRUE(fcPhysical->getFeatureOutput().has_value());
        ASSERT_TRUE(rmsPhysical->getFeatureOutput().has_value());
        EXPECT_EQ(fcPhysical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b * M2B, H2B}));
        EXPECT_EQ(rmsPhysical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b * M2B, H2B}));
        const auto oldFc = readActive2B(fcPhysical->getParameterStorage("weights"), H2B * H2B);
        const auto oldBias = readActive2B(fcPhysical->getParameterStorage("biases"), H2B);
        const auto oldRms = readActive2B(rmsPhysical->getParameterStorage("weights"), H2B);
        map<string, Impl::Tensor> outputs;
        map<string, Event> ready;
        auto done = placed->submitBatch(0, makeBatch2B(token->raggedTensor, b), outputs, ready, false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        ASSERT_TRUE(outputs.contains("report"));
        ASSERT_TRUE(rawPhysical->getErrorOutput().has_value());
        const auto offsets = rowOffsets2B(b);
        const uint64_t active = offsets.back();
        ASSERT_GT(active, 0U);
        ASSERT_LT(active, b * M2B);
        auto losses = readActive2B(outputs.at("raw_loss"), active * W2B);
        auto grads = readActive2B(rawPhysical->getErrorOutput().value(), active * W2B);
        auto reports = readActive2B(outputs.at("report"), b);
        // RAW loss and its active gradients are checked independently of the
        // reporting branch. Neither a NaN tail nor a pruned RAW root may pass.
        bool sawNonZeroGradient = false;
        for (uint64_t t = 0; t < active; ++t)
            for (uint32_t f = 0; f < W2B; ++f) {
                const size_t idx = t * W2B + f;
                ASSERT_TRUE(isfinite(losses[idx])) << idx;
                ASSERT_TRUE(isfinite(grads[idx])) << idx;
                EXPECT_GE(losses[idx], -1e-5f);
                EXPECT_NEAR(fabs(grads[idx]),
                    2.f * sqrtf(max(0.f, losses[idx])) * Impl::Loss::getLossScalingFactor(), 6e-2f);
                sawNonZeroGradient |= fabsf(grads[idx]) > 1e-5f;
            }
        EXPECT_TRUE(sawNonZeroGradient);
        ASSERT_EQ(reports.size(), b);
        for (uint32_t row = 0; row < b; ++row) {
            float expected = 0;
            for (uint64_t t = offsets[row]; t < offsets[row + 1]; ++t)
                for (uint32_t f = 0; f < W2B; ++f) expected += losses[t * W2B + f];
            EXPECT_TRUE(isfinite(reports[row]));
            EXPECT_NEAR(reports[row], expected, 5e-2f);
            if (offsets[row] == offsets[row + 1]) EXPECT_NEAR(reports[row], 0.f, 1e-5f);
        }
        const auto fcErrors = fcPhysical->getErrorOutputs();
        const auto rmsErrors = rmsPhysical->getErrorOutputs();
        ASSERT_FALSE(fcErrors.empty());
        ASSERT_GE(rmsErrors.size(), 3U);
        ASSERT_TRUE(fcErrors[0].has_value());
        ASSERT_TRUE(rmsErrors[0].has_value());
        ASSERT_TRUE(rmsErrors[2].has_value());
        auto dx = readActive2B(fcErrors[0].value(), active * H2B);
        auto dRms = readActive2B(rmsErrors[0].value(), active * H2B);
        auto dAux = readActive2B(rmsErrors[2].value(), active * H2B);
        for (float g : dx) EXPECT_TRUE(isfinite(g));
        for (float g : dRms) EXPECT_TRUE(isfinite(g));
        bool sawAuxGradient = false;
        for (float g : dAux) {
            EXPECT_TRUE(isfinite(g));
            sawAuxGradient |= isfinite(g) && fabsf(g) > 1e-6f;
        }
        EXPECT_TRUE(sawAuxGradient);
        const auto newFc = readActive2B(fcPhysical->getParameterStorage("weights"), H2B * H2B);
        const auto newBias = readActive2B(fcPhysical->getParameterStorage("biases"), H2B);
        const auto newRms = readActive2B(rmsPhysical->getParameterStorage("weights"), H2B);
        EXPECT_TRUE(anyDifferent2B(oldFc, newFc));
        EXPECT_TRUE(anyDifferent2B(oldBias, newBias));
        EXPECT_TRUE(anyDifferent2B(oldRms, newRms));
        EXPECT_EQ(network.architectureJson(), before);
    }
    filesystem::remove_all(archive);
}


// LN-H: a second connected model inserts trainable neutral LayerNorm between
// FC and Add, while preserving the RAW loss as training root and a separate
// loss-shaping reporting consumer. The original 2B network above is unchanged.
namespace {
struct LayerNormStackH1 {
    RaggedTensor tokens;
    FullyConnected fc;
    LayerNorm norm;
    RaggedCustomLoss raw;
};

LayerNormStackH1 buildLayerNormStackH1(Network& n) {
    auto x = RaggedNetworkInput::Builder().network(n).name("tokens")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({H2B}).maxValuesPerRow(M2B).build();
    auto aux = RaggedNetworkInput::Builder().network(n).name("aux")
        .valuesDataType(DataType::FP32).trailingDimensions({H2B}).partition(x).build();
    auto labels = RaggedNetworkInput::Builder().network(n).name("labels")
        .valuesDataType(DataType::FP32).trailingDimensions({W2B}).partition(x).build();

    // FC is trainable and must consume the gradient propagated through LN.
    auto fc = FullyConnected::Builder().network(n).featureInput(x)
        .numOutputFeatures(H2B).hasBias(true).weightsDataType(DataType::FP32)
        .computeDataType(DataType::FP32).outputDataType(DataType::FP32)
        .noActivation().build();
    const auto projection = fc.getRaggedFeatureOutput().value();
    auto norm = LayerNorm::Builder().network(n).featureInput(projection).build();
    auto add = Add::Builder().network(n).left(norm.getRaggedFeatureOutput().value())
        .right(aux).build();
    auto joined = Concatenate::Builder().network(n)
        .featureInput(add.getRaggedFeatureOutput().value())
        .featureInput(projection).concatenationAxis(0).build();
    auto raw = RaggedCustomLoss::Builder().network(n)
        .predictions(joined.getRaggedFeatureOutput().value()).labels(labels)
        .lossExpression(squaredError2B(false)).gradientExpression(squaredError2B(true))
        .build();
    auto reporting = RaggedLossShaper::Builder().network(n)
        .lossInput(raw.getRaggedRawLoss()).reportsPerExampleLoss().build();
    (void)NetworkOutput::Builder().network(n).name("raw_loss")
        .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
    (void)NetworkOutput::Builder().network(n).name("report")
        .inputTensor(reporting.getLossOutput()).dataType(DataType::FP32).build();
    return {x, fc, norm, raw};
}
} // namespace

TEST(ConnectedLayerNormTrainingH1, NeutralGraphUsesOneCanonicalPartitionWithoutRuntimeGeometryInArtifact) {
    Network network("connected_layernorm_h1_schema");
    auto stack = buildLayerNormStackH1(network);
    const auto output = stack.norm.getRaggedFeatureOutput();
    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(output->sharesPartitionWith(stack.tokens));
    EXPECT_TRUE(stack.raw.getRaggedRawLoss().sharesPartitionWith(stack.tokens));
    const auto artifact = stack.norm.architectureJson();
    EXPECT_EQ(artifact.at("version"), "1.1.0");
    for (const char* key : {"ragged_inputs", "ragged_outputs"}) {
        const auto& desc = artifact.at(key).at(0);
        EXPECT_FALSE(desc.contains("geometry_mode"));
        EXPECT_FALSE(desc.contains("batch_size"));
        EXPECT_FALSE(desc.contains("max_total_values"));
    }
    for (uint32_t b : {2U, 7U}) {
        EXPECT_EQ(output->getDescriptorForPlacement(b).getMaxTotalValues(), b * M2B);
        EXPECT_EQ(stack.raw.getRaggedRawLoss().getDescriptorForPlacement(b).getMaxTotalValues(), b * M2B);
    }
}

TEST(ConnectedLayerNormTrainingH1, SavedB2B7BackpropagatesThroughLayerNormAndPreservesRawReportingFanout) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const string name = "connected_layernorm_training_h1";
    const auto archive = filesystem::temp_directory_path() /
        (name + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    {
        Network network(name);
        (void)Sgd::Builder().network(network).initialLearningRate(0.001f)
            .decay(0.f).momentum(0.f).build();
        (void)buildLayerNormStackH1(network);
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network network(name);
        ASSERT_NO_THROW(network.load(archive.string()));
        const json before = network.architectureJson();
        auto fc = find2B<FullyConnected>(network);
        auto norm = find2B<LayerNorm>(network);
        auto loss = find2B<RaggedCustomLoss>(network);
        ASSERT_NE(fc, nullptr);
        ASSERT_NE(norm, nullptr);
        ASSERT_NE(loss, nullptr);
        const auto inputs = network.getExternalRaggedNetworkInputs();
        auto tokens = find_if(inputs.begin(), inputs.end(), [](const auto& item) {
            return item.name == "tokens";
        });
        ASSERT_NE(tokens, inputs.end());
        vector<Event> events;
        auto placed = network.place(b, events, false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto& stamped = placed->getStampedNetwork(0);
        auto physicalFc = dynamic_pointer_cast<Impl::RaggedFullyConnected>(
            stamped.getPhysicalLayerFromApiLayer(fc->getId()));
        auto physicalLn = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
            stamped.getPhysicalLayerFromApiLayer(norm->getId()));
        auto physicalLoss = dynamic_pointer_cast<Impl::RaggedCustomLoss>(
            stamped.getPhysicalLayerFromApiLayer(loss->getId()));
        ASSERT_NE(physicalFc, nullptr);
        ASSERT_NE(physicalLn, nullptr);
        ASSERT_NE(physicalLoss, nullptr);
        EXPECT_TRUE(physicalLoss->isTrainingActive());
        ASSERT_TRUE(physicalLn->getFeatureOutput().has_value());
        EXPECT_EQ(physicalLn->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{b * M2B, H2B}));
        EXPECT_NE(stamped.getManagedPartitionOffsetsInputForTest(
            tokens->raggedTensor.getRowPartitionId()), nullptr)
            << "reporting branch needs real row offsets";
        const auto next = physicalLoss->getNextLayer();
        ASSERT_TRUE(next.has_value());
        EXPECT_NE(dynamic_cast<Impl::TensorFanout*>(next.value()), nullptr)
            << "reporting must not displace RAW training root";
        const auto oldFc = readActive2B(physicalFc->getParameterStorage("weights"), H2B * H2B);
        const auto oldGamma = readActive2B(physicalLn->getParameterStorage("weights"), H2B);
        const auto oldBeta = readActive2B(physicalLn->getParameterStorage("biases"), H2B);
        map<string, Impl::Tensor> outputs;
        map<string, Event> ready;
        auto finished = placed->submitBatch(0, makeBatch2B(tokens->raggedTensor, b),
                                           outputs, ready, false);
        finished.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        ASSERT_TRUE(outputs.contains("report"));
        ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
        const auto offsets = rowOffsets2B(b);
        const uint64_t active = offsets.back();
        ASSERT_GT(active, 0U);
        ASSERT_LT(active, b * M2B);
        const auto losses = readActive2B(outputs.at("raw_loss"), active * W2B);
        const auto gradients = readActive2B(physicalLoss->getErrorOutput().value(), active * W2B);
        const auto reports = readActive2B(outputs.at("report"), b);
        bool nonzero = false;
        for (size_t i = 0; i < losses.size(); ++i) {
            EXPECT_TRUE(isfinite(losses[i])) << i;
            EXPECT_TRUE(isfinite(gradients[i])) << i;
            nonzero |= isfinite(gradients[i]) && fabsf(gradients[i]) > 1e-6f;
            EXPECT_NEAR(fabsf(gradients[i]),
                        2.f * sqrtf(max(0.f, losses[i])) * Impl::Loss::getLossScalingFactor(), 6e-2f);
        }
        EXPECT_TRUE(nonzero);
        ASSERT_EQ(reports.size(), b);
        for (uint32_t row = 0; row < b; ++row) {
            float sum = 0;
            for (uint64_t t = offsets[row]; t < offsets[row + 1]; ++t)
                for (uint32_t f = 0; f < W2B; ++f) sum += losses[t * W2B + f];
            EXPECT_TRUE(isfinite(reports[row]));
            EXPECT_NEAR(reports[row], sum, 5e-2f);
            if (offsets[row] == offsets[row + 1]) EXPECT_NEAR(reports[row], 0.f, 1e-5f);
        }
        const auto lnErrors = physicalLn->getErrorOutputs();
        ASSERT_EQ(lnErrors.size(), 2U);
        ASSERT_TRUE(lnErrors[0].has_value()) << "LN input gradient did not reach the FC";
        EXPECT_FALSE(lnErrors[1].has_value()) << "structural HOST_EXTENT must not receive gradients";
        const auto dx = readActive2B(lnErrors[0].value(), active * H2B);
        for (float value : dx) EXPECT_TRUE(isfinite(value));
        const auto newFc = readActive2B(physicalFc->getParameterStorage("weights"), H2B * H2B);
        const auto newGamma = readActive2B(physicalLn->getParameterStorage("weights"), H2B);
        const auto newBeta = readActive2B(physicalLn->getParameterStorage("biases"), H2B);
        EXPECT_TRUE(anyDifferent2B(oldFc, newFc)) << "connected FC weights must update";
        EXPECT_TRUE(anyDifferent2B(oldGamma, newGamma)) << "LN gamma must update";
        EXPECT_TRUE(anyDifferent2B(oldBeta, newBeta)) << "LN beta must update";
        for (float v : newFc) EXPECT_TRUE(isfinite(v));
        for (float v : newGamma) EXPECT_TRUE(isfinite(v));
        for (float v : newBeta) EXPECT_TRUE(isfinite(v));
        EXPECT_EQ(network.architectureJson(), before)
            << "placement or training must not serialize runtime B/active extent/plans";
    }
    filesystem::remove_all(archive);
}
