#include "DeepLearning/Api/Layers/Utility/RaggedGather.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedGather.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Event.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
RaggedTensor source1Q(Network& n, const string& name = "source") {
    return RaggedNetworkInput::Builder().network(n).name(name)
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2, 2}).maxValuesPerRow(4).build();
}
RaggedTensor indices1Q(Network& n, const RaggedTensor& source, bool shared,
                       const string& name = "indices") {
    RaggedNetworkInput::Builder builder;
    builder.network(n).name(name)
        .valuesDataType(DataType::UINT32).trailingDimensions({});
    if (shared) builder.partition(source);
    else builder.offsetsDataType(DataType::UINT32).maxValuesPerRow(3);
    return builder.build();
}
Impl::DynamicExpression squared1Q(bool derivative) {
    auto p = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    auto l = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    auto delta = p - l;
    auto e = derivative ? delta * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()) : delta * delta;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            derivative ? "predictions_grad" : "loss", e.withOutputDType(DataType::FP32)}})));
}
float sourceValue1Q(uint64_t t, uint32_t f) {
    return 0.17f * static_cast<float>(t) - 0.13f * static_cast<float>(f) + 0.31f;
}
float upstream1Q(uint64_t t, uint32_t f) {
    return ((t + f) % 2 ? -1.0f : 1.0f) * (0.14f + 0.045f * static_cast<float>((t + 3 * f) % 5));
}
vector<float> download1Q(const Impl::Tensor& device, uint64_t elements) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = device.clone(cpu);
    Impl::Tensor handle = device;
    Stream stream = Stream::getNextDownloadStream(device.getPlacement().getDeviceNum());
    host.downloadSection(handle, stream, 0, 0, elements * sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + elements);
}
class Stamp1Q : public RaggedGather {
public:
    explicit Stamp1Q(const RaggedGather& g) : RaggedGather(g) {}
    using RaggedGather::stamp;
};
void reject1Q(Network& n, const json& j) { EXPECT_ANY_THROW(RaggedGather::deserialize(j, &n)); }

void savedGather1Q(bool shared) {
    const string name = shared ? "gather_1q_shared" : "gather_1q_distinct";
    const auto archive = filesystem::temp_directory_path() /
        (name + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    uint64_t outputId = 0, qTokenId = 0;
    {
        Network n(name);
        const auto src = source1Q(n);
        const auto ids = indices1Q(n, src, shared);
        const auto targets = RaggedNetworkInput::Builder().network(n).name("targets")
            .partition(ids).valuesDataType(DataType::FP32).trailingDimensions({2, 2}).build();
        const auto sourceRivet = GradientRivet::Builder().network(n).tensor(src.getValues()).build();
        const auto gather = RaggedGather::Builder().network(n)
            .sourceInput(src.withValues(sourceRivet.getFeatureOutput().value()))
            .indicesInput(ids).build();
        const auto result = gather.getRaggedFeatureOutput();
        ASSERT_TRUE(result.sharesPartitionWith(ids));
        EXPECT_EQ(result.getRowPartitionToken(), ids.getRowPartitionToken());
        EXPECT_EQ(gather.architectureJson().at("version"), "1.1.0");
        outputId = result.getValues().getOriginalId();
        qTokenId = ids.getRowPartitionToken().getOriginalId();
        const auto loss = RaggedCustomLoss::Builder().network(n).predictions(result).labels(targets)
            .lossExpression(squared1Q(false)).gradientExpression(squared1Q(true)).build();
        (void)NetworkOutput::Builder().network(n).name("raw_loss")
            .inputTensor(loss.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        n.save(archive.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "shared=" << shared << " B=" << b);
        Network n(name);
        ASSERT_NO_THROW(n.load(archive.string()));
        RaggedTensor src, ids, targets;
        for (const auto& entry : n.getExternalRaggedNetworkInputs()) {
            if (entry.name == "source") src = entry.raggedTensor;
            else if (entry.name == "indices") ids = entry.raggedTensor;
            else if (entry.name == "targets") targets = entry.raggedTensor;
        }
        ASSERT_TRUE(src.isInitialized());
        ASSERT_TRUE(ids.isInitialized());
        ASSERT_TRUE(targets.isInitialized());
        EXPECT_EQ(src.sharesPartitionWith(ids), shared);
        EXPECT_TRUE(ids.sharesPartitionWith(targets));
        EXPECT_EQ(ids.getRowPartitionToken().getOriginalId(), qTokenId);
        shared_ptr<RaggedGather> gather;
        for (uint32_t i = 0; i < n.getNumLayers(); ++i)
            if (auto candidate = dynamic_pointer_cast<RaggedGather>(n.getLayer(i))) gather = candidate;
        ASSERT_NE(gather, nullptr);
        const auto out = gather->getRaggedFeatureOutput();
        EXPECT_EQ(out.getValues().getOriginalId(), outputId);
        EXPECT_EQ(out.getRowPartitionToken(), ids.getRowPartitionToken());
        const json before = n.architectureJson();
        vector<Event> init;
        auto placed = n.place(b, init, false);
        ASSERT_NE(placed, nullptr);
        for (auto& e : init) e.synchronize();
        auto& stamped = placed->getStampedNetwork(0);
        auto physical = dynamic_pointer_cast<Impl::RaggedGather>(
            stamped.getPhysicalLayerFromApiLayer(gather->getId()));
        ASSERT_NE(physical, nullptr);
        const auto ports = physical->getFeatureInputs();
        ASSERT_EQ(ports.size(), shared ? 3U : 4U);
        for (const auto& port : ports) ASSERT_TRUE(port.has_value());
        EXPECT_EQ(ports[0]->getDimensions(), (vector<uint64_t>{b * 4U, 2, 2}));
        EXPECT_EQ(ports[1]->getDimensions(), (vector<uint64_t>{b * (shared ? 4U : 3U)}));
        EXPECT_EQ(ports[2]->getDimensions(), (vector<uint64_t>{b + 1U}));
        if (!shared) EXPECT_EQ(ports[3]->getDimensions(), (vector<uint64_t>{b + 1U}));
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{b * (shared ? 4U : 3U), 2, 2}));
        EXPECT_NE(stamped.getManagedPartitionOffsetsInputForTest(src.getRowPartitionId()), nullptr);
        if (!shared) EXPECT_NE(stamped.getManagedPartitionOffsetsInputForTest(ids.getRowPartitionId()), nullptr);
        EXPECT_EQ(n.architectureJson(), before);

        vector<uint64_t> p(b + 1, 0), q(b + 1, 0);
        for (uint32_t row = 0; row < b; ++row) {
            const uint32_t length = row % 3 == 0 ? 0U : row % 3 == 1 ? 3U : 4U;
            p[row + 1] = p[row] + length;
            q[row + 1] = q[row] + (shared ? length : length == 4 ? 2U : length);
        }
        const uint64_t activeP = p.back(), activeQ = q.back();
        ASSERT_GT(activeQ, 0U);
        ASSERT_LT(activeP, 4ULL * b);
        ASSERT_LT(activeQ, (shared ? 4ULL : 3ULL) * b);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        const auto pDesc = src.getDescriptorForPlacement(b);
        const auto qDesc = ids.getDescriptorForPlacement(b);
        Impl::Tensor pOff(cpu, pDesc.getOffsetsDescriptor());
        Impl::Tensor qOff(cpu, qDesc.getOffsetsDescriptor());
        for (uint32_t row = 0; row <= b; ++row) {
            pOff.getMemPtr<uint32_t>()[row] = static_cast<uint32_t>(p[row]);
            qOff.getMemPtr<uint32_t>()[row] = static_cast<uint32_t>(q[row]);
        }
        Impl::RowPartitionRuntime pRuntime(pOff, pDesc.getRowPartition());
        pRuntime.setHostOffsets(p);
        Impl::RowPartitionRuntime qRuntime(shared ? pOff : qOff, qDesc.getRowPartition());
        qRuntime.setHostOffsets(q);
        Impl::Tensor sourceData(cpu, pDesc.getValuesDescriptor());
        Impl::Tensor indexData(cpu, qDesc.getValuesDescriptor());
        Impl::Tensor labelData(cpu, targets.getDescriptorForPlacement(b).getValuesDescriptor());
        for (uint64_t t = 0; t < 4ULL * b; ++t)
            for (uint32_t f = 0; f < 4; ++f)
                sourceData.getMemPtr<float>()[t * 4 + f] = t < activeP ? sourceValue1Q(t, f) :
                    numeric_limits<float>::quiet_NaN();
        for (uint64_t t = 0; t < static_cast<uint64_t>(b) * (shared ? 4U : 3U); ++t) {
            indexData.getMemPtr<uint32_t>()[t] = 0U;
            for (uint32_t f = 0; f < 4; ++f)
                labelData.getMemPtr<float>()[t * 4 + f] = numeric_limits<float>::quiet_NaN();
        }
        vector<float> expectedGrad(activeP * 4, 0.0f);
        for (uint32_t row = 0; row < b; ++row) {
            const uint32_t lengthP = static_cast<uint32_t>(p[row + 1] - p[row]);
            const uint32_t lengthQ = static_cast<uint32_t>(q[row + 1] - q[row]);
            for (uint32_t j = 0; j < lengthQ; ++j) {
                const uint32_t selected = lengthP == 3 ? (j < 2 ? 1 : 2) :
                                          lengthP == 4 ? (j % 2 ? 3 : 0) : 0;
                ASSERT_LT(selected, lengthP);
                const uint64_t qt = q[row] + j, pt = p[row] + selected;
                indexData.getMemPtr<uint32_t>()[qt] = selected;
                for (uint32_t f = 0; f < 4; ++f) {
                    const float upstream = upstream1Q(qt, f);
                    labelData.getMemPtr<float>()[qt * 4 + f] = sourceValue1Q(pt, f) - upstream;
                    expectedGrad[pt * 4 + f] += 2.0f * upstream * Impl::Loss::getLossScalingFactor();
                }
            }
        }
        Batch batch;
        batch.insert("source", Impl::RaggedTensor(sourceData, pRuntime));
        batch.insert("indices", Impl::RaggedTensor(indexData, shared ? pRuntime : qRuntime));
        batch.insert("targets", labelData);
        batch.setValidExampleCount(b);
        map<string, Impl::Tensor> outputs;
        map<string, Event> ready;
        auto done = placed->submitBatch(0, batch, outputs, ready, false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        const auto actual = download1Q(physical->getFeatureOutput().value(), activeQ * 4);
        const auto gradients = physical->getErrorOutputs();
        ASSERT_EQ(gradients.size(), shared ? 3U : 4U);
        ASSERT_TRUE(gradients[0].has_value()) << "source values gradient was pruned";
        EXPECT_FALSE(gradients[1].has_value()) << "indices must not receive value gradients";
        EXPECT_FALSE(gradients[2].has_value()) << "P offsets are structural";
        if (!shared) EXPECT_FALSE(gradients[3].has_value()) << "Q offsets are structural";
        const auto grad = download1Q(gradients[0].value(), activeP * 4);
        for (uint32_t row = 0; row < b; ++row)
            for (uint64_t t = q[row]; t < q[row + 1]; ++t)
                for (uint32_t f = 0; f < 4; ++f) {
                    const uint64_t local = t - q[row];
                    const uint32_t lp = static_cast<uint32_t>(p[row + 1] - p[row]);
                    const uint32_t selected = lp == 3 ? (local < 2 ? 1 : 2) :
                                              lp == 4 ? (local % 2 ? 3 : 0) : 0;
                    EXPECT_NEAR(actual[t * 4 + f], sourceValue1Q(p[row] + selected, f), 5e-4f);
                }
        for (uint64_t i = 0; i < activeP * 4; ++i)
            EXPECT_NEAR(grad[i], expectedGrad[i], 8e-3f) << "source gradient component " << i;
        EXPECT_EQ(n.architectureJson(), before);
    }
    filesystem::remove_all(archive);
}
}  // namespace

// Simulate historical packed dimensions and deliberately wrong obsolete
// physical capacity metadata without changing semantic tensor identities.
static void toHistoricalPackedReference(nlohmann::json& ref) {
    const uint64_t m = ref.at("max_values_per_row").get<uint64_t>();
    ref.erase("layer_ragged_reference_version");
    ref.erase("geometry_mode");
    ref.erase("max_values_per_row");
    ref["batch_size"] = 999U;
    ref["max_total_values"] = 123456U;
    ref["values"]["dimensions"][0] = m * 3U;
    ref["offsets"]["dimensions"][0] = 4U;
}

TEST(RaggedGatherNeutral1Q, TwoPartitionsAndMemory) {
    Network n("gather_1q_builder");
    auto src = source1Q(n);
    auto ids = indices1Q(n, src, false);
    auto g = RaggedGather::Builder().network(n).sourceInput(src).indicesInput(ids).build();
    const auto out = g.getRaggedFeatureOutput();
    EXPECT_TRUE(out.sharesPartitionWith(ids));
    EXPECT_FALSE(out.sharesPartitionWith(src));
    EXPECT_EQ(out.getRowPartitionToken(), ids.getRowPartitionToken());
    EXPECT_EQ(out.getValuesDimensions(), (vector<uint64_t>{3, 2, 2}));
    ASSERT_EQ(g.getFeatureInputs().size(), 4U);
    EXPECT_EQ(g.getFeatureInputs()[0], src.getValues());
    EXPECT_EQ(g.getFeatureInputs()[1], ids.getValues());
    EXPECT_EQ(g.getFeatureInputs()[2], src.getOffsets());
    EXPECT_EQ(g.getFeatureInputs()[3], ids.getOffsets());
    EXPECT_EQ(g.getRaggedPartitionRequirementForInput(src.getOffsets()),
              Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(g.getRaggedPartitionRequirementForInput(ids.getOffsets()),
              Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(g.architectureJson().at("version"), "1.1.0");
    EXPECT_EQ(g.architectureJson().at("ragged_source").at("max_values_per_row"), 4);
    EXPECT_EQ(g.architectureJson().at("ragged_indices").at("max_values_per_row"), 3);
    for (const auto& ref : {g.architectureJson().at("ragged_source"),
                            g.architectureJson().at("ragged_indices"), g.architectureJson().at("ragged_output")}) {
        EXPECT_FALSE(ref.contains("batch_size"));
        EXPECT_FALSE(ref.contains("max_total_values"));
    }
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2U, 7U}) {
        EXPECT_EQ(g.getOutputTensorBytes(b), 3ULL * 4 * sizeof(float) * b);
        EXPECT_EQ(g.getFirstInstanceMemRequirementInBytes(b, cpu), 3ULL * 4 * sizeof(float) * b);
    }
    EXPECT_ANY_THROW((void)g.getOutputTensorBytes(0));
    EXPECT_NO_THROW(RaggedGather::deserialize(g.architectureJson(), &n));

    auto shared = indices1Q(n, src, true, "shared_indices");
    auto h = RaggedGather::Builder().network(n).sourceInput(src).indicesInput(shared).build();
    EXPECT_EQ(h.getFeatureInputs().size(), 3U);
    EXPECT_TRUE(h.getRaggedFeatureOutput().sharesPartitionWith(src));
    EXPECT_EQ(h.getRaggedFeatureOutput().getValuesDimensions(), (vector<uint64_t>{4, 2, 2}));
    EXPECT_NO_THROW(RaggedGather::deserialize(h.architectureJson(), &n));
}

TEST(RaggedGatherNeutral1Q, StrictReferencesAndLegacyFixed) {
    Network n("gather_1q_bad");
    auto src = source1Q(n);
    auto ids = indices1Q(n, src, false);
    auto g = RaggedGather::Builder().network(n).sourceInput(src).indicesInput(ids).build();
    const json j = g.architectureJson();
    json bad = j;
    bad["version"] = "2.0.0"; reject1Q(n, bad);
    bad = j; bad["batch_size"] = 2; reject1Q(n, bad);
    bad = j; bad["ragged_source"]["max_total_values"] = 8; reject1Q(n, bad);
    bad = j; bad["ragged_source"]["values"]["dimensions"] = vector<uint64_t>{3, 2, 2}; reject1Q(n, bad);
    bad = j; bad["ragged_source"]["offsets"]["data_type"] = DataType::UINT64; reject1Q(n, bad);
    bad = j; bad["ragged_source"]["values"]["id"] = ids.getValues().getOriginalId(); reject1Q(n, bad);
    bad = j; bad["ragged_indices"]["values"]["data_type"] = DataType::UINT64; reject1Q(n, bad);
    bad = j; bad["ragged_indices"]["offsets"]["id"] = src.getOffsets().getOriginalId(); reject1Q(n, bad);
    bad = j; bad["ragged_indices"]["max_values_per_row"] = 4; reject1Q(n, bad);
    bad = j; bad["ragged_indices"]["geometry_mode"] = "fixed"; reject1Q(n, bad);
    bad = j; bad["ragged_output"]["offsets"]["id"] = src.getOffsets().getOriginalId(); reject1Q(n, bad);
    bad = j; bad["ragged_output"]["offsets"]["dimensions"] = vector<uint64_t>{4}; reject1Q(n, bad);
    bad = j; bad["ragged_output"]["values"]["dimensions"] = vector<uint64_t>{4, 2, 2}; reject1Q(n, bad);
    bad = j; bad["ragged_output"]["values"]["id"] = src.getValues().getOriginalId(); reject1Q(n, bad);
    bad = j; bad["ragged_output"]["max_values_per_row"] = 4; reject1Q(n, bad);
    bad = j; bad["ragged_output"]["extra"] = 1; reject1Q(n, bad);
    bad = j; bad["factory"] = "broken"; reject1Q(n, bad);

    Network old("gather_1q_fixed");
    auto oldP = RaggedNetworkInput::Builder().network(old).name("source")
        .valuesDataType(DataType::FP32).trailingDimensions({2}).maxValuesPerRow(4).build();
    auto oldQ = RaggedNetworkInput::Builder().network(old).name("indices")
        .valuesDataType(DataType::UINT32).trailingDimensions({}).maxValuesPerRow(3).build();
    auto canonical = RaggedGather::Builder().network(old).sourceInput(oldP).indicesInput(oldQ).build();
    const json current = canonical.architectureJson();
    EXPECT_EQ(current.at("version"), "1.1.0");
    EXPECT_EQ(canonical.getRaggedFeatureOutput().getValuesDimensions(), (vector<uint64_t>{3, 2}));
    EXPECT_EQ(canonical.getOutputTensorBytes(7), 7ULL * 3 * 2 * sizeof(float));
    EXPECT_NO_THROW(RaggedGather::deserialize(current, &old));
    // Historical packed records have their own dedicated test below.
    // Index values must be integer scalars, independent of any placement B.
    EXPECT_THROW((void)RaggedGather::Builder().network(n).sourceInput(src).indicesInput(
        ids.withValues(Tensor(DataType::FP32, {3}))).build(), invalid_argument);
    EXPECT_ANY_THROW((void)RaggedGather::Builder().network(n).sourceInput(src).indicesInput(
        src.withValues(Thor::Tensor(DataType::UINT32, {4, 2, 2}))).build());
}

TEST(RaggedGatherNeutral1Q, AnyValuesOrOffsetsMayDriveStamp) {
    Network n("gather_1q_stamp");
    auto src = source1Q(n);
    auto ids = indices1Q(n, src, false);
    auto g = RaggedGather::Builder().network(n).sourceInput(src).indicesInput(ids).build();
    Stamp1Q probe(g);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2U, 7U}) {
        const vector<pair<Thor::Tensor, Impl::TensorDescriptor>> ports = {
            {src.getValues(), Impl::TensorDescriptor(DataType::FP32, {b * 4U, 2, 2})},
            {ids.getValues(), Impl::TensorDescriptor(DataType::UINT32, {b * 3U})},
            {src.getOffsets(), Impl::TensorDescriptor(DataType::UINT32, {b + 1U})},
            {ids.getOffsets(), Impl::TensorDescriptor(DataType::UINT32, {b + 1U})}};
        for (const auto& [logical, desc] : ports) {
            auto driver = make_shared<Impl::NetworkInput>(cpu, desc.getDataType(), desc.getDimensions());
            EXPECT_NE(dynamic_pointer_cast<Impl::RaggedGather>(
                probe.stamp(cpu, driver, nullptr, logical, true)), nullptr);
        }
        auto wrong = make_shared<Impl::NetworkInput>(cpu, DataType::UINT32, vector<uint64_t>{1});
        EXPECT_ANY_THROW((void)probe.stamp(cpu, wrong, nullptr, src.getOffsets(), true));
        auto badValues = make_shared<Impl::NetworkInput>(cpu, DataType::FP32, vector<uint64_t>{b * 5U, 2, 2});
        EXPECT_ANY_THROW((void)probe.stamp(cpu, badValues, nullptr, src.getValues(), true));
    }
    // The shared P=Q form has one deduplicated structural offset port.
    auto sharedIds = indices1Q(n, src, true, "shared_indices");
    auto sharedLayer = RaggedGather::Builder().network(n).sourceInput(src).indicesInput(sharedIds).build();
    Stamp1Q sharedProbe(sharedLayer);
    for (uint32_t b : {2U, 7U}) {
        const vector<pair<Thor::Tensor, Impl::TensorDescriptor>> sharedPorts = {
            {src.getValues(), Impl::TensorDescriptor(DataType::FP32, {b * 4U, 2, 2})},
            {sharedIds.getValues(), Impl::TensorDescriptor(DataType::UINT32, {b * 4U})},
            {src.getOffsets(), Impl::TensorDescriptor(DataType::UINT32, {b + 1U})}};
        for (const auto& [logical, desc] : sharedPorts) {
            auto driver = make_shared<Impl::NetworkInput>(cpu, desc.getDataType(), desc.getDimensions());
            EXPECT_NE(dynamic_pointer_cast<Impl::RaggedGather>(
                sharedProbe.stamp(cpu, driver, nullptr, logical, true)), nullptr);
        }
    }
}

TEST(RaggedGatherNeutral1Q, SavedDistinctPartitionsForwardBackwardAtTwoAndSeven) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    savedGather1Q(false);
}
TEST(RaggedGatherNeutral1Q, SavedSharedPartitionForwardBackwardAtTwoAndSeven) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    savedGather1Q(true);
}

TEST(RaggedGatherNeutral1Q, NeutralCloneMapsSourceAndDestinationPartitions) {
    Network n("gather_1q_clone_original");
    auto src = source1Q(n), ids = indices1Q(n, src, false);
    auto gather = RaggedGather::Builder().network(n).sourceInput(src).indicesInput(ids).build();
    (void)RaggedNetworkOutput::Builder().network(n).name("gathered")
        .inputTensor(gather.getRaggedFeatureOutput()).build();
    Network dest("gather_1q_clone_dest");
    auto dsrc = source1Q(dest), dids = indices1Q(dest, dsrc, false);
    ApiTensorRemap remap;
    remap.map(src.getValues(), dsrc.getValues());
    remap.map(src.getRowPartitionToken(), dsrc.getRowPartitionToken());
    remap.map(ids.getValues(), dids.getValues());
    remap.map(ids.getRowPartitionToken(), dids.getRowPartitionToken());
    ApiSubgraphCloneOptions opt;
    opt.inferenceOnly = true;
    const auto clone = dest.cloneSubgraphInto(n, {"__thor_ragged_output.gathered.values"}, remap, opt);
    ASSERT_EQ(clone.outputTensorsByName.size(), 1U);
    EXPECT_EQ(clone.outputTensorsByName.at("__thor_ragged_output.gathered.values").getDimensions(),
              (vector<uint64_t>{3, 2, 2}));
    EXPECT_EQ(clone.clonedTensorBySourceOriginalId.at(
        gather.getRaggedFeatureOutput().getRowPartitionToken().getOriginalId()),
        dids.getRowPartitionToken());
}

TEST(RaggedGatherNeutral1Q, GatherHistoricalPackedGrammarKeepsIndependentPartitions) {
    Network n("gather_old_reader");
    const auto p = source1Q(n);
    const auto q = indices1Q(n, p, false);
    const auto op = RaggedGather::Builder().network(n).sourceInput(p).indicesInput(q).build();
    json old = op.architectureJson();
    old["version"] = "1.0.0";
    for (const char* ref : {"ragged_source", "ragged_indices", "ragged_output"})
        toHistoricalPackedReference(old[ref]);
    ASSERT_NO_THROW(RaggedGather::deserialize(old, &n));
    const auto result = op.getRaggedFeatureOutput();
    EXPECT_TRUE(result.sharesPartitionWith(q));
    EXPECT_FALSE(result.sharesPartitionWith(p));
    for (uint64_t b : {2U, 7U}) {
        const auto descriptor = result.getDescriptorForPlacement(b);
        EXPECT_EQ(descriptor.getValuesDimensions(), (vector<uint64_t>{b * 3U, 2U, 2U}));
        EXPECT_EQ(descriptor.getOffsetsDescriptor().getDimensions(), (vector<uint64_t>{b + 1U}));
    }
    old["ragged_output"]["offsets"]["id"] = p.getRowPartitionToken().getOriginalId();
    EXPECT_ANY_THROW(RaggedGather::deserialize(old, &n));
}
