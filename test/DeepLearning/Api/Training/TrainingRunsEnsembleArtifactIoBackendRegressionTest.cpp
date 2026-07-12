#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Training/TrainingRuns.h"
#include "Utilities/Common/Event.h"
#include "Utilities/TarFile/UringDirect.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace Thor;

namespace {

struct BackendCase {
    const char* envValue;
    const char* displayName;
};

class ScopedEnvVar {
   public:
    ScopedEnvVar(std::string name, std::string value) : name_(std::move(name)) {
        const char* previous = std::getenv(name_.c_str());
        if (previous != nullptr) {
            previous_ = std::string(previous);
        }
        setenv(name_.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (previous_.has_value()) {
            setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

   private:
    std::string name_;
    std::optional<std::string> previous_;
};

class ScopedIoUringCompletionByteLimit {
   public:
    explicit ScopedIoUringCompletionByteLimit(std::optional<std::uint32_t> limitBytes,
                                              UringDirect::TestIoOperation operation = UringDirect::TestIoOperation::Any,
                                              std::uint64_t matchingOperationsToSkip = 0) {
        UringDirect::testSetNextIoUringCompletionByteLimitForOperation(limitBytes, operation, matchingOperationsToSkip);
    }

    ~ScopedIoUringCompletionByteLimit() { UringDirect::testResetIoUringShortIoHooks(); }

    std::uint64_t hitCount() const { return UringDirect::testGetIoUringCompletionByteLimitHitCount(); }

    ScopedIoUringCompletionByteLimit(const ScopedIoUringCompletionByteLimit&) = delete;
    ScopedIoUringCompletionByteLimit& operator=(const ScopedIoUringCompletionByteLimit&) = delete;
};

std::filesystem::path uniqueTempPath(const std::string& prefix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path path = std::filesystem::temp_directory_path() / (prefix + "_" + std::to_string(now));
    std::filesystem::remove_all(path);
    return path;
}

void synchronizeEvents(std::vector<Event>& events) {
    for (Event& event : events) {
        event.synchronize();
    }
    events.clear();
}

bool isUnavailableExplicitUringDirect(const BackendCase& backend, const std::string& message) {
    if (std::string(backend.envValue) != "uring_direct") {
        return false;
    }
    return message.find("io_uring_queue_init") != std::string::npos ||
           message.find("uring_direct is unavailable") != std::string::npos ||
           message.find("explicit uring_direct") != std::string::npos;
}

std::string outputName(uint32_t index) { return "prediction_" + std::to_string(index); }

std::shared_ptr<Network> buildManyOutputMemberNetwork(const std::string& networkName,
                                                      uint32_t inputFeatureCount,
                                                      uint32_t outputCount) {
    auto network = std::make_shared<Network>(networkName);
    NetworkInput input = NetworkInput::Builder()
                             .network(*network)
                             .name("features")
                             .dimensions({inputFeatureCount})
                             .dataType(DataType::FP32)
                             .build();

    for (uint32_t i = 0; i < outputCount; ++i) {
        FullyConnected fc = FullyConnected::Builder()
                                .network(*network)
                                .featureInput(input.getFeatureOutput().value())
                                .numOutputFeatures(1)
                                .hasBias(true)
                                .weightsDataType(DataType::FP32)
                                .computeDataType(DataType::FP32)
                                .outputDataType(DataType::FP32)
                                .weightsInitializer(UniformRandom::Builder().minValue(-0.01f).maxValue(0.01f).build())
                                .biasInitializer(UniformRandom::Builder().minValue(-0.01f).maxValue(0.01f).build())
                                .noActivation()
                                .build();
        NetworkOutput::Builder()
            .network(*network)
            .name(outputName(i))
            .inputTensor(fc.getFeatureOutput().value())
            .dataType(DataType::FP32)
            .build();
    }

    return network;
}

TrainingRunsResult makeSavedManyOutputEnsembleMembers(const std::filesystem::path& root,
                                                       uint32_t memberCount,
                                                       uint32_t inputFeatureCount,
                                                       uint32_t outputCount) {
    std::vector<TrainingRunResult> runResults;
    std::vector<TrainingEnsembleMemberResult> members;
    runResults.reserve(memberCount);
    members.reserve(memberCount);

    for (uint32_t memberIndex = 0; memberIndex < memberCount; ++memberIndex) {
        const std::string runName = "fold_" + std::to_string(memberIndex);
        const std::string networkName = "ensemble_crc_member_" + std::to_string(memberIndex);
        const std::filesystem::path memberDir = root / runName;
        std::filesystem::create_directories(memberDir);
        buildManyOutputMemberNetwork(networkName, inputFeatureCount, outputCount)->save(memberDir.string(), /*overwrite=*/true);
        std::ofstream(memberDir / "training_selection_metadata.json") << "{}\n";

        TrainingRunResult result = TrainingRunResult::completedResult(
            runName, {}, {}, {}, TrainingRunCompletionReason::COMPLETED, 1, 1, 1.0, memberDir.string());
        result.ensembleGroup = "demand";
        result.savedModelNetworkName = networkName;
        runResults.push_back(std::move(result));

        members.push_back(TrainingEnsembleMemberResult{runName, 1.0, TrainingRunStatus::COMPLETED});
    }

    TrainingEnsembleResult ensemble;
    ensemble.ensembleGroup = "demand";
    ensemble.minSuccessfulModels = memberCount;
    ensemble.members = std::move(members);
    ensemble.inputSignature = {TrainingRunInputSignature{"features", {0, inputFeatureCount}, "FP32", true}};
    for (uint32_t i = 0; i < outputCount; ++i) {
        ensemble.outputSignature.push_back(TrainingRunOutputSignature{outputName(i), {0, 1}, "FP32"});
    }

    return TrainingRunsResult(std::move(runResults), {std::move(ensemble)});
}


void loadAndPlaceSavedEnsemble(const std::filesystem::path& ensembleDir) {
    Network loaded("ensemble_demand");
    loaded.load(ensembleDir.string());
    std::vector<Event> initDoneEvents;
    std::shared_ptr<PlacedNetwork> placed = loaded.place(/*batchSize=*/1,
                                                         initDoneEvents,
                                                         /*inferenceOnly=*/true,
                                                         /*forcedDevices=*/std::vector<int32_t>{0},
                                                         /*forcedNumStampsPerGpu=*/1,
                                                         /*networkOutputsOnGpu=*/false);
    EXPECT_NE(placed, nullptr);
    synchronizeEvents(initDoneEvents);
}

class TrainingRunsEnsembleArtifactIoBackendRegression : public ::testing::TestWithParam<BackendCase> {};

// Disabled by default because this is a slow, GPU-backed end-to-end artifact
// regression. It is valuable as a manual production-path probe, but it should
// not be part of the default test set.
// Run explicitly with:
//   --gtest_also_run_disabled_tests
//   --gtest_filter=ExplicitBackends/TrainingRunsEnsembleArtifactIoBackendRegression.DISABLED_*
TEST_P(TrainingRunsEnsembleArtifactIoBackendRegression, DISABLED_SaveEnsembleThenLoadAndPlacePreservesGpuBackedParameterPayloadCrcs) {
    const BackendCase backend = GetParam();
    ScopedEnvVar scopedBackend("THOR_IO_BACKEND", backend.envValue);

    // This intentionally exercises the higher-level path that failed in the SKU forecaster:
    // save_ensemble builds a composed inference ensemble, places it, saves GPU-backed parameter
    // tensors through TarWriter, then an immediate Network::load()->place() scans the archive
    // and reloads those parameter tensors through TarReader.
    constexpr uint32_t kMemberCount = 3;
    constexpr uint32_t kInputFeatureCount = 8;
    constexpr uint32_t kOutputCount = 192;
    const std::filesystem::path root = uniqueTempPath(std::string("thor-training-runs-ensemble-artifact-crc-") + backend.displayName);
    const std::filesystem::path ensembleDir = root / "ensemble";

    try {
        TrainingRunsResult results =
            makeSavedManyOutputEnsembleMembers(root, kMemberCount, kInputFeatureCount, kOutputCount);
        const std::string savedPath = results.saveEnsemble("demand", ensembleDir.string(), "mean", /*overwrite=*/true);
        EXPECT_EQ(savedPath, ensembleDir.string());

        loadAndPlaceSavedEnsemble(ensembleDir);
    } catch (const std::runtime_error& e) {
        std::filesystem::remove_all(root);
        if (isUnavailableExplicitUringDirect(backend, e.what())) {
            GTEST_SKIP() << "Explicit uring_direct backend is unavailable in this runtime: " << e.what();
        }
        throw;
    } catch (...) {
        std::filesystem::remove_all(root);
        throw;
    }

    std::filesystem::remove_all(root);
}


// Disabled by default because this is a targeted, slow, fault-injected production-path
// regression. Run explicitly with:
//   --gtest_also_run_disabled_tests
//   --gtest_filter=TrainingRunsEnsembleArtifactShortIoRegression.DISABLED_*
TEST(TrainingRunsEnsembleArtifactShortIoRegression, DISABLED_ShortUringWriteDuringSaveEnsembleStillLoadsAndPlaces) {
    // This is the closest regression to the production SKU failure.  Member artifacts
    // and the verification read use pread_direct so the only fault-injected phase is
    // the final ensemble archive write.  The 4096-feature FC weights are 16 KiB each,
    // guaranteeing that a 4 KiB completion limit represents a real positive short CQE.
    constexpr uint32_t kMemberCount = 3;
    constexpr uint32_t kInputFeatureCount = 4096;
    constexpr uint32_t kOutputCount = 16;
    const std::filesystem::path root = uniqueTempPath("thor-training-runs-ensemble-artifact-short-write");
    const std::filesystem::path ensembleDir = root / "ensemble";

    try {
        ScopedEnvVar memberBackend("THOR_IO_BACKEND", "pread_direct");
        TrainingRunsResult results =
            makeSavedManyOutputEnsembleMembers(root, kMemberCount, kInputFeatureCount, kOutputCount);

        {
            ScopedEnvVar writeBackend("THOR_IO_BACKEND", "uring_direct");
            ScopedIoUringCompletionByteLimit shortIoUringWrite(/*limitBytes=*/4096,
                                                               UringDirect::TestIoOperation::Write);
            const std::string savedPath = results.saveEnsemble("demand", ensembleDir.string(), "mean", /*overwrite=*/true);
            EXPECT_EQ(savedPath, ensembleDir.string());
            ASSERT_EQ(shortIoUringWrite.hitCount(), 1u)
                << "The intended final ensemble archive short-write completion was not exercised.";
        }

        ScopedEnvVar verificationBackend("THOR_IO_BACKEND", "pread_direct");
        loadAndPlaceSavedEnsemble(ensembleDir);
    } catch (const std::runtime_error& e) {
        std::filesystem::remove_all(root);
        if (isUnavailableExplicitUringDirect(BackendCase{"uring_direct", "uring_direct"}, e.what())) {
            GTEST_SKIP() << "Explicit uring_direct backend is unavailable in this runtime: " << e.what();
        }
        throw;
    } catch (...) {
        std::filesystem::remove_all(root);
        throw;
    }

    std::filesystem::remove_all(root);
}

TEST(TrainingRunsEnsembleArtifactShortIoRegression, DISABLED_ShortUringReadDuringImmediateLoadAndPlaceStillSucceeds) {
    // This is the complementary reader isolation test.  The complete ensemble artifact
    // is produced with pread_direct, then only Network::load()/place() uses uring_direct.
    // A 16 KiB parameter payload guarantees that the injected 4 KiB CQE is truly short.
    constexpr uint32_t kMemberCount = 3;
    constexpr uint32_t kInputFeatureCount = 4096;
    constexpr uint32_t kOutputCount = 16;
    const std::filesystem::path root = uniqueTempPath("thor-training-runs-ensemble-artifact-short-read");
    const std::filesystem::path ensembleDir = root / "ensemble";

    try {
        {
            ScopedEnvVar writeBackend("THOR_IO_BACKEND", "pread_direct");
            TrainingRunsResult results =
                makeSavedManyOutputEnsembleMembers(root, kMemberCount, kInputFeatureCount, kOutputCount);
            const std::string savedPath = results.saveEnsemble("demand", ensembleDir.string(), "mean", /*overwrite=*/true);
            EXPECT_EQ(savedPath, ensembleDir.string());
        }

        ScopedEnvVar readBackend("THOR_IO_BACKEND", "uring_direct");
        ScopedIoUringCompletionByteLimit shortIoUringRead(/*limitBytes=*/4096, UringDirect::TestIoOperation::Read);
        loadAndPlaceSavedEnsemble(ensembleDir);
        ASSERT_EQ(shortIoUringRead.hitCount(), 1u)
            << "The intended immediate ensemble archive short-read completion was not exercised.";
    } catch (const std::runtime_error& e) {
        std::filesystem::remove_all(root);
        if (isUnavailableExplicitUringDirect(BackendCase{"uring_direct", "uring_direct"}, e.what())) {
            GTEST_SKIP() << "Explicit uring_direct backend is unavailable in this runtime: " << e.what();
        }
        throw;
    } catch (...) {
        std::filesystem::remove_all(root);
        throw;
    }

    std::filesystem::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(ExplicitBackends,
                         TrainingRunsEnsembleArtifactIoBackendRegression,
                         ::testing::Values(BackendCase{"uring_direct", "uring_direct"},
                                           BackendCase{"pread_direct", "pread_direct"},
                                           BackendCase{"pread_buffered", "pread_buffered"}),
                         [](const ::testing::TestParamInfo<BackendCase>& info) { return info.param.displayName; });

}  // namespace
