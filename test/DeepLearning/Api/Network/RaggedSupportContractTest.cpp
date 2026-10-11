#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "test/DeepLearning/RaggedTestUtils.h"
#include "Utilities/Common/Event.h"
#include "Utilities/TarFile/TarWriter.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;
using Api::DataType;

namespace {

constexpr uint32_t kBatchSize = 2;
constexpr uint64_t kMaxTotalValues = 6;
constexpr uint64_t kWidth = 2;

// Old model grammars must be tested as archived input, not by constructing a
// fictitious fixed-batch API tensor in the current graph. Build a valid graph,
// then write a test-only archive with the historical packed source dimensions
// and 1.1 boundary field names. This is not a production JSON rewrite path.
Api::RaggedTensor makeHistoricalPhysicalBoundary(Api::Network& network,
                                                 const std::string& name,
                                                 DataType offsetsType,
                                                 std::optional<uint64_t> semanticM) {
    (void)semanticM;  // Missing-M is represented in the archived grammar below.
    return Api::RaggedNetworkInput::Builder()
        .network(network).name(name)
        .valuesDataType(DataType::FP32).offsetsDataType(offsetsType)
        .trailingDimensions({kWidth}).maxValuesPerRow(3).build();
}

void saveHistoricalBoundaryFixture(Api::Network& network,
                                   const std::filesystem::path& archiveDir,
                                   bool omitSemanticM = false) {
    nlohmann::json archive = network.architectureJson();
    for (auto& layer : archive.at("layers")) {
        if (layer.at("layer_type") != "network_input") continue;
        const std::string name = layer.at("name").get<std::string>();
        std::vector<uint64_t> physical;
        if (name == "__thor_row_partition.tokens") physical = {kBatchSize + 1};
        else if (name == "tokens.values" || name == "labels.values") physical = {kMaxTotalValues, kWidth};
        else continue;
        layer["dimensions"] = physical;
        layer.at("feature_input")["dimensions"] = physical;
        layer.at("feature_output")["dimensions"] = physical;
    }
    for (auto& entry : archive.at("ragged_network_inputs")) {
        entry["version"] = "1.1.0";  // Historical offsets_tensor_id grammar.
        entry["batch_size"] = kBatchSize;
        entry["max_total_values"] = kMaxTotalValues;
        if (entry.contains("row_partition_token_tensor_id")) {
            entry["offsets_tensor_id"] = entry.at("row_partition_token_tensor_id");
            entry.erase("row_partition_token_tensor_id");
        }
        if (omitSemanticM && !entry.contains("partition_input_name"))
            entry.erase("max_values_per_row");
    }
    const std::string jsonDump = archive.dump(4);
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor bytes(cpu, Impl::TensorDescriptor(DataType::UINT8, {jsonDump.size()}));
    std::memcpy(bytes.getMemPtr<void>(), jsonDump.data(), jsonDump.size());
    // TarReader identifies the JSON member using the archive's model name.
    const auto fileName = network.getNetworkName() + ".thor.json";
    thor_file::TarWriter writer(network.getNetworkName());
    writer.addArchiveFile(fileName, bytes);
    writer.createArchive(archiveDir, /*overwriteIfExists=*/true);
}

bool cudaAvailable() {
    int deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

std::shared_ptr<Api::PlacedNetwork> placeIdentityRaggedNetwork(Api::Network& network) {
    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placed = network.place(kBatchSize, initDoneEvents, /*inferenceOnly=*/true);
    for (Event& event : initDoneEvents) event.synchronize();
    return placed;
}

Api::RaggedTensor buildIdentityRaggedNetwork(Api::Network& network, DataType offsetsDataType) {
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(offsetsDataType)
                                  .trailingDimensions({kWidth})
                                  .maxValuesPerRow(3)
                                  .build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("tokens_out").inputTensor(input).build();
    return input;
}

void writeOffsets(Impl::Tensor& tensor, DataType dataType, const std::vector<uint64_t>& offsets,
                  uint32_t batchSize = kBatchSize) {
    ASSERT_EQ(offsets.size(), batchSize + 1);
    if (dataType == DataType::UINT32) {
        uint32_t* out = tensor.getMemPtr<uint32_t>();
        for (size_t i = 0; i < offsets.size(); ++i) {
            ASSERT_LE(offsets[i], static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
            out[i] = static_cast<uint32_t>(offsets[i]);
        }
        return;
    }
    ASSERT_EQ(dataType, DataType::UINT64);
    uint64_t* out = tensor.getMemPtr<uint64_t>();
    for (size_t i = 0; i < offsets.size(); ++i) out[i] = offsets[i];
}

std::vector<uint64_t> readOffsets(const Impl::Tensor& tensor, uint32_t batchSize = kBatchSize) {
    std::vector<uint64_t> offsets(batchSize + 1);
    if (tensor.getDataType() == DataType::UINT32) {
        const uint32_t* in = tensor.getMemPtr<uint32_t>();
        for (size_t i = 0; i < offsets.size(); ++i) offsets[i] = in[i];
        return offsets;
    }
    EXPECT_EQ(tensor.getDataType(), DataType::UINT64);
    const uint64_t* in = tensor.getMemPtr<uint64_t>();
    for (size_t i = 0; i < offsets.size(); ++i) offsets[i] = in[i];
    return offsets;
}

void runIdentityCase(Api::PlacedNetwork& placed,
                     DataType offsetsDataType,
                     const std::vector<uint64_t>& offsetsVector,
                     float activeBase,
                     uint32_t batchSize = kBatchSize,
                     uint64_t maxTotalValues = kMaxTotalValues) {
    ASSERT_EQ(offsetsVector.size(), batchSize + 1);
    ASSERT_EQ(offsetsVector.front(), 0u);
    for (size_t i = 1; i < offsetsVector.size(); ++i) ASSERT_LE(offsetsVector[i - 1], offsetsVector[i]);
    const uint64_t activeValues = offsetsVector.back();
    ASSERT_LE(activeValues, maxTotalValues);

    const Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor values(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {maxTotalValues, kWidth}));
    Impl::Tensor offsets(cpuPlacement, Impl::TensorDescriptor(offsetsDataType, {batchSize + 1}));

    float* valuesPtr = values.getMemPtr<float>();
    for (uint64_t i = 0; i < activeValues * kWidth; ++i) {
        valuesPtr[i] = activeBase + static_cast<float>(i) * 0.25f;
    }
    ThorTest::poisonInactiveElements(valuesPtr,
                                     activeValues * kWidth,
                                     maxTotalValues * kWidth,
                                     ThorTest::RaggedInactivePoison::NaN);
    writeOffsets(offsets, offsetsDataType, offsetsVector, batchSize);

    Batch batch;
    batch.insert("tokens", Impl::RaggedTensor(values, offsets, 3));
    std::map<std::string, Api::InferenceOutputValue> outputs = placed.inferLogical(batch);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs.contains("tokens_out"));
    ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("tokens_out")));

    const Impl::RaggedTensor result = std::get<Impl::RaggedTensor>(outputs.at("tokens_out"));
    EXPECT_EQ(result.getOffsets().getDataType(), offsetsDataType);
    EXPECT_EQ(readOffsets(result.getOffsets(), batchSize), offsetsVector);
    EXPECT_EQ(result.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(activeValues));

    const float* resultValues = result.getValues().getMemPtr<float>();
    for (uint64_t i = 0; i < activeValues * kWidth; ++i) {
        EXPECT_EQ(resultValues[i], activeBase + static_cast<float>(i) * 0.25f);
    }

    // Deliberately do not inspect output storage beyond the active prefix. The
    // contract says that capacity is undefined, not zero-padding.
}

std::filesystem::path uniqueArchiveDir(const std::string& stem) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "_" + std::to_string(nonce));
}

}  // namespace

// The repository-level CMake qualification target deliberately runs this
// disabled preflight with --gtest_also_run_disabled_tests. Individual
// CUDA-backed tests may skip during ordinary development; the aggregate R11D
// contract gate must not pass vacuously without a CUDA device.
TEST(RaggedSupportContract, DISABLED_RequiresCudaDevice) {
    if (std::getenv("THOR_RAGGED_SUPPORT_GATE") == nullptr) {
        GTEST_SKIP() << "Ragged support preflight only runs through check-ragged-support-contract.";
    }

    int deviceCount = 0;
    const cudaError_t status = cudaGetDeviceCount(&deviceCount);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    ASSERT_GT(deviceCount, 0) << "Rank-1 ragged support qualification requires a CUDA device.";
}

TEST(RaggedSupportContract, CanonicalBoundaryCoversOffsetWidthsPoisonAllEmptyAndShortLongShortReuse) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for ragged support contract runtime coverage.";

    for (const DataType offsetsDataType : {DataType::UINT32, DataType::UINT64}) {
        SCOPED_TRACE(offsetsDataType == DataType::UINT32 ? "UINT32 offsets" : "UINT64 offsets");
        Api::Network network(offsetsDataType == DataType::UINT32 ? "ragged_support_contract_u32"
                                                                 : "ragged_support_contract_u64");
        (void)buildIdentityRaggedNetwork(network, offsetsDataType);
        std::shared_ptr<Api::PlacedNetwork> placed = placeIdentityRaggedNetwork(network);
        ASSERT_NE(placed, nullptr);

        // One executable must derive its runtime extent from each newly supplied
        // canonical partition. The repeated short case catches stale extent
        // state after the longer execution.
        runIdentityCase(*placed, offsetsDataType, {0, 1, 2}, 10.0f);
        runIdentityCase(*placed, offsetsDataType, {0, 3, 6}, 20.0f);
        runIdentityCase(*placed, offsetsDataType, {0, 1, 2}, 30.0f);

        // An all-empty partition is valid even though the entire values capacity
        // is poisoned and semantically inactive.
        runIdentityCase(*placed, offsetsDataType, {0, 0, 0}, 40.0f);
    }
}


TEST(RaggedSupportContract, PartitionOwningGpuSubmissionRequiresAuthoritativeHostPartition) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for ragged support contract runtime coverage.";

    Api::Network network("ragged_support_contract_gpu_host_partition_required");
    (void)buildIdentityRaggedNetwork(network, DataType::UINT32);
    std::shared_ptr<Api::PlacedNetwork> placed = placeIdentityRaggedNetwork(network);
    ASSERT_NE(placed, nullptr);

    const Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);
    Impl::Tensor values(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {kMaxTotalValues, kWidth}));
    Impl::Tensor offsets(gpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {kBatchSize + 1}));

    Batch batch;
    batch.insert("tokens", Impl::RaggedTensor(values, offsets, 3));
    try {
        (void)placed->inferLogical(batch);
        FAIL() << "Expected a GPU-only ragged submission without authoritative host offsets to be rejected.";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("no authoritative host row partition"), std::string::npos);
        EXPECT_NE(message.find("device-to-host"), std::string::npos);
    }
}

TEST(RaggedSupportContract, SaveLoadDoesNotPersistRuntimeExtentAndLoadedModelAcceptsDifferentPartition) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for ragged support contract runtime coverage.";

    const std::string networkName = "ragged_support_contract_save_load";
    const std::filesystem::path archiveDir = uniqueArchiveDir(networkName);
    std::filesystem::remove_all(archiveDir);

    try {
        Api::Network source(networkName);
        (void)buildIdentityRaggedNetwork(source, DataType::UINT64);
        std::shared_ptr<Api::PlacedNetwork> sourcePlaced = placeIdentityRaggedNetwork(source);
        ASSERT_NE(sourcePlaced, nullptr);

        // Publish a short runtime host partition before saving. Runtime partition
        // state must not become serialized model state.
        runIdentityCase(*sourcePlaced, DataType::UINT64, {0, 1, 2}, 50.0f);
        sourcePlaced->save(archiveDir.string(), /*overwrite=*/true, /*saveOptimizerState=*/false);

        Api::Network loaded(networkName);
        loaded.load(archiveDir.string());
        const std::vector<Api::RaggedNetworkInputReference> loadedInputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(loadedInputs.size(), 1u);
        EXPECT_EQ(loadedInputs.front().raggedTensor.getOffsetsDataType(), DataType::UINT64);
        EXPECT_EQ(loadedInputs.front().raggedTensor.getMaxValuesPerRow(), 3u);
        EXPECT_EQ(loadedInputs.front().raggedTensor.getValuesDimensions(),
                  (std::vector<uint64_t>{3, kWidth}));

        std::shared_ptr<Api::PlacedNetwork> loadedPlaced = placeIdentityRaggedNetwork(loaded);
        ASSERT_NE(loadedPlaced, nullptr);

        // A newly placed loaded model must derive state from the newly submitted
        // partition, not from the short host partition that preceded save().
        runIdentityCase(*loadedPlaced, DataType::UINT64, {0, 3, 6}, 60.0f);
        runIdentityCase(*loadedPlaced, DataType::UINT64, {0, 1, 2}, 70.0f);
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}


// A test archive carrying the old fixed-boundary grammar, with an independently
// serialized semantic M. Loading must ignore old B/total, recover [M,width] and
// [1], and place independently at B=2 and B=7. No production JSON
// rewriting or layer-specific compatibility mode is involved.
TEST(RaggedSupportContract, RecoverableLegacyBoundaryImportsLogicalSourcesBeforeLayerLoad) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required for the cross-placement legacy boundary gate.";

    const std::string networkName = "ragged_legacy_logical_boundary_u1";
    const auto archiveDir = uniqueArchiveDir(networkName);
    std::filesystem::remove_all(archiveDir);
    try {
        Api::Network historical(networkName);
        const auto legacyInput = makeHistoricalPhysicalBoundary(
            historical, "tokens", DataType::UINT64, /*semanticM=*/3);
        (void)Api::RaggedNetworkOutput::Builder()
            .network(historical).name("tokens_out").inputTensor(legacyInput).build();
        saveHistoricalBoundaryFixture(historical, archiveDir);

        for (const uint32_t B : {2u, 7u}) {
            Api::Network restored(networkName);
            ASSERT_NO_THROW(restored.load(archiveDir.string()));
            const auto boundary = restored.getExternalRaggedNetworkInputs();
            ASSERT_EQ(boundary.size(), 1u);
            EXPECT_EQ(boundary.front().raggedTensor.getValuesDimensions(),
                      (std::vector<uint64_t>{3, kWidth}));
            EXPECT_EQ(boundary.front().raggedTensor.getOffsetsDimensions(),
                      (std::vector<uint64_t>{1}));
            EXPECT_EQ(boundary.front().raggedTensor.getDescriptorForPlacement(B).getValuesDimensions(),
                      (std::vector<uint64_t>{B * 3u, kWidth}));
            const auto outputs = restored.getExternalRaggedNetworkOutputs();
            ASSERT_EQ(outputs.size(), 1u);
            EXPECT_TRUE(boundary.front().raggedTensor.sharesPartitionWith(outputs.front().raggedTensor));

            std::vector<Event> initDoneEvents;
            auto placed = restored.place(B, initDoneEvents, /*inferenceOnly=*/true);
            ASSERT_NE(placed, nullptr);
            for (Event& event : initDoneEvents) event.synchronize();
            if (B == 2) runIdentityCase(*placed, DataType::UINT64, {0, 3, 6}, 100.0f);
            else runIdentityCase(*placed, DataType::UINT64,
                                 {0, 1, 1, 3, 3, 4, 5, 7}, 120.0f, B, B * 3u);
        }
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(RaggedSupportContract, HistoricalBoundaryWithoutIndependentSemanticMFailsExplicitly) {
    const std::string networkName = "ragged_legacy_missing_m_u1";
    const auto archiveDir = uniqueArchiveDir(networkName);
    std::filesystem::remove_all(archiveDir);
    try {
        Api::Network historical(networkName);
        (void)makeHistoricalPhysicalBoundary(
            historical, "tokens", DataType::UINT32, std::nullopt);
        saveHistoricalBoundaryFixture(historical, archiveDir, /*omitSemanticM=*/true);

        Api::Network restored(networkName);
        try {
            restored.load(archiveDir.string());
            FAIL() << "Historical physical total must not be divided by old B to guess M";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("Unrecoverable historical ragged boundary"),
                      std::string::npos) << error.what();
        }
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(RaggedSupportContract, LegacySharedBoundaryUsesOneLogicalPartitionAfterLoad) {
    const std::string networkName = "ragged_legacy_shared_boundary_u1";
    const auto archiveDir = uniqueArchiveDir(networkName);
    std::filesystem::remove_all(archiveDir);
    try {
        Api::Network historical(networkName);
        const auto source = makeHistoricalPhysicalBoundary(
            historical, "tokens", DataType::UINT32, /*semanticM=*/3);
        (void)Api::RaggedNetworkInput::Builder()
            .network(historical).name("labels")
            .valuesDataType(DataType::FP32)
            .trailingDimensions({kWidth}).partition(source).build();
        saveHistoricalBoundaryFixture(historical, archiveDir);

        Api::Network restored(networkName);
        restored.load(archiveDir.string());
        const auto inputs = restored.getExternalRaggedNetworkInputs();
        ASSERT_EQ(inputs.size(), 2u);
        EXPECT_TRUE(inputs[0].raggedTensor.sharesPartitionWith(inputs[1].raggedTensor));
        EXPECT_EQ(inputs[0].raggedTensor.getValuesDimensions(), (std::vector<uint64_t>{3, kWidth}));
        EXPECT_EQ(inputs[1].raggedTensor.getValuesDimensions(), (std::vector<uint64_t>{3, kWidth}));
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}
