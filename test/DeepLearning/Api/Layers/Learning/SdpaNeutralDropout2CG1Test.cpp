#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/ScaledDotProductAttention.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

constexpr uint64_t MQG1 = 4;
constexpr uint64_t MKG1 = 6;
constexpr uint64_t HQG1 = 4;
constexpr uint64_t HKG1 = 2;
constexpr uint64_t DG1 = 32;
constexpr float PG1 = 0.5f;
constexpr int64_t SeedG1 = 371223;
constexpr int64_t OffsetG1 = 29;

std::filesystem::path archiveG1() {
    return std::filesystem::temp_directory_path() /
        ("thor_sdpa_dropout_g1_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
}

std::vector<uint64_t> offsetsG1(uint32_t b, bool query) {
    std::vector<uint64_t> result(b + 1, 0);
    for (uint32_t row = 0; row < b; ++row) {
        const uint64_t n = query ? (2 + row % 2) : (3 + row % 3);
        result[row + 1] = result[row] + n;
    }
    return result;
}

Impl::Tensor offsetsTensorG1(const std::vector<uint64_t>& offsets, DataType type) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor result(cpu, Impl::TensorDescriptor(type, {static_cast<uint64_t>(offsets.size())}));
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (type == DataType::UINT32) result.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(offsets[i]);
        else result.getMemPtr<uint64_t>()[i] = offsets[i];
    }
    return result;
}

Impl::Tensor valuesG1(uint32_t b, uint64_t m, uint64_t heads,
                      const std::vector<uint64_t>& offsets, int which) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor result(cpu, Impl::TensorDescriptor(DataType::FP16, {b * m, heads, DG1}));
    half* ptr = result.getMemPtr<half>();
    for (uint64_t token = 0; token < b * m; ++token)
        for (uint64_t head = 0; head < heads; ++head)
            for (uint64_t dim = 0; dim < DG1; ++dim) {
                const uint64_t i = (token * heads + head) * DG1 + dim;
                const float sample = .023f * static_cast<float>((token * 13 + head * 7 + dim * 5 + which * 3) % 37)
                    + .017f * static_cast<float>((token * 3 + dim + which) % 9) - .29f;
                ptr[i] = token < offsets.back() ? __float2half(sample)
                    : __float2half(std::numeric_limits<float>::quiet_NaN());
            }
    return result;
}

Batch makeBatchG1(uint32_t b, const Api::RaggedTensor& q,
                  const Api::RaggedTensor& kv, const Api::RaggedTensor& v) {
    const auto qo = offsetsG1(b, true);
    const auto ko = offsetsG1(b, false);
    Impl::RowPartitionRuntime qp(offsetsTensorG1(qo, DataType::UINT32),
                                 q.getDescriptorForPlacement(b).getRowPartition());
    Impl::RowPartitionRuntime kp(offsetsTensorG1(ko, DataType::UINT64),
                                 kv.getDescriptorForPlacement(b).getRowPartition());
    qp.setHostOffsets(qo);
    kp.setHostOffsets(ko);
    Batch batch;
    batch.insert("q", Impl::RaggedTensor(valuesG1(b, MQG1, HQG1, qo, 0), qp));
    batch.insert("kv", Impl::RaggedTensor(valuesG1(b, MKG1, HKG1, ko, 1), kp));
    batch.insert("v", Impl::RaggedTensor(valuesG1(b, MKG1, HKG1, ko, 2), kp));
    batch.setValidExampleCount(b);
    return batch;
}

std::shared_ptr<Api::ScaledDotProductAttention> findG1(Api::Network& net) {
    for (uint32_t i = 0; i < net.getNumLayers(); ++i)
        if (auto found = std::dynamic_pointer_cast<Api::ScaledDotProductAttention>(net.getLayer(i)))
            return found;
    return {};
}

void saveG1(const std::filesystem::path& path) {
    Api::Network net("sdpa_neutral_dropout_2cg1");
    const auto q = Api::RaggedNetworkInput::Builder().network(net).name("q")
        .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
        .trailingDimensions({HQG1, DG1}).maxValuesPerRow(MQG1).build();
    const auto kv = Api::RaggedNetworkInput::Builder().network(net).name("kv")
        .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
        .trailingDimensions({HKG1, DG1}).maxValuesPerRow(MKG1).build();
    const auto v = Api::RaggedNetworkInput::Builder().network(net).name("v")
        .valuesDataType(DataType::FP16).trailingDimensions({HKG1, DG1}).partition(kv).build();
    auto attention = Api::ScaledDotProductAttention::Builder().network(net)
        .queryInput(q).keyInput(kv).valueInput(v)
        .computeDataType(DataType::FP32).dropout(PG1, SeedG1, OffsetG1).build();
    ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
    ASSERT_TRUE(attention.getRaggedFeatureOutput()->sharesPartitionWith(q));
    (void)Api::RaggedNetworkOutput::Builder().network(net).name("out")
        .inputTensor(*attention.getRaggedFeatureOutput()).build();
    net.save(path.string(), true);
}

struct InstanceG1 {
    std::unique_ptr<Api::Network> network;
    std::shared_ptr<Api::PlacedNetwork> placed;
    std::shared_ptr<Api::ScaledDotProductAttention> attention;
    nlohmann::json originalArchitecture;
};

InstanceG1 loadG1(const std::filesystem::path& archive, uint32_t b) {
    InstanceG1 result;
    result.network = std::make_unique<Api::Network>("sdpa_neutral_dropout_2cg1");
    result.network->load(archive.string());
    result.attention = findG1(*result.network);
    if (!result.attention) throw std::runtime_error("Missing reloaded dropout SDPA");
    result.originalArchitecture = result.network->architectureJson();
    std::vector<Event> events;
    result.placed = result.network->place(b, events, true);
    for (auto& event : events) event.synchronize();
    return result;
}

std::vector<float> forwardG1(InstanceG1& instance, uint32_t b) {
    const auto& sdpa = *instance.attention;
    auto outputs = instance.placed->inferLogical(makeBatchG1(
        b, *sdpa.getQueryRaggedInput(), *sdpa.getKeyRaggedInput(), *sdpa.getValueRaggedInput()));
    if (!outputs.contains("out") || !std::holds_alternative<Impl::RaggedTensor>(outputs.at("out")))
        throw std::runtime_error("Expected ragged SDPA output");
    const Impl::RaggedTensor& output = std::get<Impl::RaggedTensor>(outputs.at("out"));
    if (output.getValues().getDimensions() != std::vector<uint64_t>{b * MQG1, HQG1, DG1})
        throw std::runtime_error("Incorrect packed output capacity");
    if (!output.getHostOffsetsIfAvailable().has_value() ||
        output.getHostOffsetsIfAvailable().value() != offsetsG1(b, true))
        throw std::runtime_error("Incorrect output Q partition");
    const uint64_t active = offsetsG1(b, true).back() * HQG1 * DG1;
    const half* ptr = output.getValues().getMemPtr<half>();
    std::vector<float> result(active);
    for (uint64_t i = 0; i < active; ++i) {
        result[i] = __half2float(ptr[i]);
        if (!std::isfinite(result[i])) throw std::runtime_error("Nonfinite active dropout output");
    }
    return result;
}

} // namespace

TEST(SdpaNeutralDropout2CG1, SavedNeutralConfigurationContainsOnlyInitialSeedOffsetNotRuntimePlacement) {
    const auto path = archiveG1();
    std::filesystem::remove_all(path);
    saveG1(path);
    Api::Network loaded("sdpa_neutral_dropout_2cg1");
    loaded.load(path.string());
    auto sdpa = findG1(loaded);
    ASSERT_NE(sdpa, nullptr);
    const auto arch = sdpa->architectureJson();
    EXPECT_EQ(arch.at("version"), "2.2.0");
    EXPECT_FLOAT_EQ(arch.at("dropout_probability").get<float>(), PG1);
    EXPECT_EQ(arch.at("dropout_seed").get<int64_t>(), SeedG1);
    EXPECT_EQ(arch.at("dropout_offset").get<int64_t>(), OffsetG1);
    for (const char* forbidden : {"batch_size", "max_total_values", "physical_offsets",
                                  "runtime_dropout_offset", "dropout_counter", "dropout_state"})
        EXPECT_FALSE(arch.contains(forbidden)) << forbidden;
    ASSERT_TRUE(sdpa->getQueryRaggedInput().has_value());
    ASSERT_TRUE(sdpa->getKeyRaggedInput().has_value());
    EXPECT_FALSE(sdpa->getQueryRaggedInput()->sharesPartitionWith(*sdpa->getKeyRaggedInput()));
    std::filesystem::remove_all(path);
}

TEST(SdpaNeutralDropout2CG1, SavedB2B7FreshInstancesReplayAndAdvanceIndependently) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device required for real SDPA dropout replay";
    const auto archive = archiveG1();
    std::filesystem::remove_all(archive);
    saveG1(archive);
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        auto first = loadG1(archive, b);
        const auto a0 = forwardG1(first, b);
        const auto a1 = forwardG1(first, b);
        // A fresh, independently stamped network must start at the *saved*
        // initial dropout offset, unaffected by another instance's forwards.
        auto second = loadG1(archive, b);
        const auto b0 = forwardG1(second, b);
        const auto b1 = forwardG1(second, b);
        ASSERT_EQ(a0.size(), b0.size());
        ASSERT_EQ(a1.size(), b1.size());
        EXPECT_EQ(a0, b0) << "fresh instance must replay configured seed/offset";
        EXPECT_EQ(a1, b1) << "independent instances must advance identically";
        // With many nonempty query/key rows and p=0.5, the first and second
        // masks must not be accidentally the same due to a frozen offset.
        EXPECT_NE(a0, a1) << "dropout offset must advance on each actual forward";
        EXPECT_EQ(first.network->architectureJson(), first.originalArchitecture);
        EXPECT_EQ(second.network->architectureJson(), second.originalArchitecture);
        auto record = first.attention->architectureJson();
        EXPECT_EQ(record.at("dropout_offset").get<int64_t>(), OffsetG1)
            << "runtime advance must not rewrite the saved artifact";
    }
    std::filesystem::remove_all(archive);
}
