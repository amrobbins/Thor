#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "test/DeepLearning/RaggedTestUtils.h"
#include "Utilities/Common/Event.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <chrono>
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
                                  .batchSize(kBatchSize)
                                  .maxTotalValues(kMaxTotalValues)
                                  .build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("tokens_out").inputTensor(input).build();
    return input;
}

void writeOffsets(Impl::Tensor& tensor, DataType dataType, const std::vector<uint64_t>& offsets) {
    ASSERT_EQ(offsets.size(), kBatchSize + 1);
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

std::vector<uint64_t> readOffsets(const Impl::Tensor& tensor) {
    std::vector<uint64_t> offsets(kBatchSize + 1);
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
                     float activeBase) {
    ASSERT_EQ(offsetsVector.size(), kBatchSize + 1);
    ASSERT_EQ(offsetsVector.front(), 0u);
    for (size_t i = 1; i < offsetsVector.size(); ++i) ASSERT_LE(offsetsVector[i - 1], offsetsVector[i]);
    const uint64_t activeValues = offsetsVector.back();
    ASSERT_LE(activeValues, kMaxTotalValues);

    const Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor values(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {kMaxTotalValues, kWidth}));
    Impl::Tensor offsets(cpuPlacement, Impl::TensorDescriptor(offsetsDataType, {kBatchSize + 1}));

    float* valuesPtr = values.getMemPtr<float>();
    for (uint64_t i = 0; i < activeValues * kWidth; ++i) {
        valuesPtr[i] = activeBase + static_cast<float>(i) * 0.25f;
    }
    ThorTest::poisonInactiveElements(valuesPtr,
                                     activeValues * kWidth,
                                     kMaxTotalValues * kWidth,
                                     ThorTest::RaggedInactivePoison::NaN);
    writeOffsets(offsets, offsetsDataType, offsetsVector);

    Batch batch;
    batch.insert("tokens", Impl::RaggedTensor(values, offsets));
    std::map<std::string, Api::InferenceOutputValue> outputs = placed.inferLogical(batch);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs.contains("tokens_out"));
    ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(outputs.at("tokens_out")));

    const Impl::RaggedTensor result = std::get<Impl::RaggedTensor>(outputs.at("tokens_out"));
    EXPECT_EQ(result.getOffsets().getDataType(), offsetsDataType);
    EXPECT_EQ(readOffsets(result.getOffsets()), offsetsVector);
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

// The R1 CMake qualification target deliberately runs this disabled preflight
// with --gtest_also_run_disabled_tests. Individual CUDA-backed tests may skip
// during ordinary development; the aggregate contract gate must not pass
// vacuously without a CUDA device.
TEST(RaggedSupportContract, DISABLED_RequiresCudaDevice) {
    if (std::getenv("THOR_R1_RAGGED_SUPPORT_GATE") == nullptr) {
        GTEST_SKIP() << "R1 ragged support preflight only runs through check-ragged-support-contract.";
    }

    int deviceCount = 0;
    const cudaError_t status = cudaGetDeviceCount(&deviceCount);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    ASSERT_GT(deviceCount, 0) << "R1 ragged support qualification requires a CUDA device.";
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

        // Populate the runtime cache with a short partition before saving. That
        // payload-derived extent must not become serialized model state.
        runIdentityCase(*sourcePlaced, DataType::UINT64, {0, 1, 2}, 50.0f);
        sourcePlaced->save(archiveDir.string(), /*overwrite=*/true, /*saveOptimizerState=*/false);

        Api::Network loaded(networkName);
        loaded.load(archiveDir.string());
        const std::vector<Api::RaggedNetworkInputReference> loadedInputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(loadedInputs.size(), 1u);
        EXPECT_EQ(loadedInputs.front().raggedTensor.getOffsetsDataType(), DataType::UINT64);
        EXPECT_EQ(loadedInputs.front().raggedTensor.getMaxTotalValues(), kMaxTotalValues);

        std::shared_ptr<Api::PlacedNetwork> loadedPlaced = placeIdentityRaggedNetwork(loaded);
        ASSERT_NE(loadedPlaced, nullptr);

        // A newly placed loaded model must derive state from the newly submitted
        // partition, not from the short active count that preceded save().
        runIdentityCase(*loadedPlaced, DataType::UINT64, {0, 3, 6}, 60.0f);
        runIdentityCase(*loadedPlaced, DataType::UINT64, {0, 1, 2}, 70.0f);
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}
