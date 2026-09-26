#include "DeepLearning/Implementation/Training/TrainingCheckpointRepository.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <unistd.h>

namespace {

class TemporaryDirectory {
   public:
    explicit TemporaryDirectory(const std::string& stem) {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               (stem + "_" + std::to_string(::getpid()) + "_" + std::to_string(now));
        std::filesystem::create_directories(path);
    }
    ~TemporaryDirectory() {
        std::error_code errorCode;
        std::filesystem::remove_all(path, errorCode);
    }

    std::filesystem::path path;
};

void writeFakeArtifact(const std::filesystem::path& directory, const std::string& value) {
    std::filesystem::create_directories(directory / "nested");
    std::ofstream(directory / "weights.bin", std::ios::binary | std::ios::trunc) << value;
    std::ofstream(directory / "nested" / "optimizer.bin", std::ios::binary | std::ios::trunc) << "optimizer-" << value;
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void writeCheckpointState(const std::filesystem::path& directory,
                          uint64_t completedEpoch,
                          uint64_t phaseStartCompletedEpoch,
                          uint32_t requestedPhaseEpochs,
                          std::optional<uint64_t> bestEpoch = std::nullopt,
                          std::optional<double> bestScore = std::nullopt) {
    std::ofstream out(directory / "checkpoint_state.json", std::ios::binary | std::ios::trunc);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"checkpoint_reason\": \"test\",\n";
    out << "  \"completed_epoch\": " << completedEpoch << ",\n";
    out << "  \"phase_start_completed_epoch\": " << phaseStartCompletedEpoch << ",\n";
    out << "  \"requested_phase_epochs\": " << requestedPhaseEpochs << ",\n";
    out << "  \"phase_target_completed_epoch\": "
        << (phaseStartCompletedEpoch + requestedPhaseEpochs) << ",\n";
    out << "  \"best_epoch\": ";
    if (bestEpoch.has_value()) out << bestEpoch.value(); else out << "null";
    out << ",\n  \"best_score\": ";
    if (bestScore.has_value()) out << bestScore.value(); else out << "null";
    out << ",\n";
    out << "  \"check_best_model_every_epochs\": 1,\n";
    out << "  \"first_model_selection_epoch\": 1,\n";
    out << "  \"checkpoint_every_epochs\": 2,\n";
    out << "  \"device_dataset_storage\": {\n";
    out << "    \"requested\": \"off\",\n";
    out << "    \"attempted\": false,\n";
    out << "    \"used\": false,\n";
    out << "    \"reason\": \"\",\n";
    out << "    \"examples\": 0,\n";
    out << "    \"required_bytes\": 0,\n";
    out << "    \"available_bytes_after_placement\": 0,\n";
    out << "    \"resident_bytes\": 0,\n";
    out << "    \"resident_cache_hit\": false,\n";
    out << "    \"resident_construction_joined\": false,\n";
    out << "    \"resident_construction_started\": false,\n";
    out << "    \"materialization_seconds\": 0.0,\n";
    out << "    \"windowed_device_cache\": {\n";
    out << "      \"requested\": \"auto\",\n";
    out << "      \"attempted\": false,\n";
    out << "      \"used\": false,\n";
    out << "      \"reason\": \"\",\n";
    out << "      \"eligible_sources\": 0,\n";
    out << "      \"active_sources\": 0,\n";
    out << "      \"eligible_source_bytes\": 0,\n";
    out << "      \"budget_bytes\": 0,\n";
    out << "      \"max_access_policy_window_bytes\": 0,\n";
    out << "      \"active_unique_bytes\": 0,\n";
    out << "      \"hit_ratio\": 0.0\n";
    out << "    }\n";
    out << "  }\n";
    out << "}\n";
}

TEST(TrainingCheckpointRepository, CreateBeforeDestroyKeepsAReferencedGenerationRecoverable) {
    TemporaryDirectory temp("thor_training_checkpoint_repository");
    ThorImplementation::TrainingCheckpointRepository repository(temp.path / "repo");

    auto first = repository.prepareGeneration(10);
    writeFakeArtifact(first.temporaryDirectory, "first");
    repository.publishGeneration(first);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, first.name);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST, first.name);

    ASSERT_TRUE(std::filesystem::is_directory(first.directory));
    EXPECT_EQ(repository.readPointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT), first.name);
    EXPECT_EQ(repository.readPointer(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST), first.name);

    auto second = repository.prepareGeneration(20);
    writeFakeArtifact(second.temporaryDirectory, "second");
    repository.publishGeneration(second);

    // Publishing a complete generation does not consume the old generation or
    // change either pointer. A failure here therefore recovers the first one.
    EXPECT_TRUE(std::filesystem::is_directory(first.directory));
    EXPECT_TRUE(std::filesystem::is_directory(second.directory));
    EXPECT_EQ(repository.readPointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT), first.name);
    EXPECT_EQ(repository.readPointer(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST), first.name);

    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, second.name);

    // A failure between CURRENT and BEST publication still leaves both named
    // generations intact. In particular, retiring first is not yet permitted.
    repository.retireGenerationIfUnreferenced(first.name);
    EXPECT_TRUE(std::filesystem::is_directory(first.directory));
    EXPECT_EQ(repository.readPointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT), second.name);
    EXPECT_EQ(repository.readPointer(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST), first.name);

    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST, second.name);
    repository.retireGenerationIfUnreferenced(first.name);

    EXPECT_FALSE(std::filesystem::exists(first.directory));
    EXPECT_TRUE(std::filesystem::is_directory(second.directory));
    EXPECT_EQ(repository.readPointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT), second.name);
    EXPECT_EQ(repository.readPointer(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST), second.name);
}

TEST(TrainingCheckpointRepository, TemporaryGenerationIsNotVisibleUntilDurablyPublished) {
    TemporaryDirectory temp("thor_training_checkpoint_repository_tmp");
    ThorImplementation::TrainingCheckpointRepository repository(temp.path / "repo");

    auto generation = repository.prepareGeneration(7);
    writeFakeArtifact(generation.temporaryDirectory, "partial-or-complete-but-not-published");

    EXPECT_TRUE(std::filesystem::is_directory(generation.temporaryDirectory));
    EXPECT_FALSE(std::filesystem::exists(generation.directory));
    EXPECT_FALSE(repository.readPointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT).has_value());
    EXPECT_THROW(repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, generation.name),
                 std::runtime_error);

    repository.publishGeneration(generation);
    EXPECT_FALSE(std::filesystem::exists(generation.temporaryDirectory));
    EXPECT_TRUE(std::filesystem::is_directory(generation.directory));
}

TEST(TrainingCheckpointRepository, FinalArtifactCopyDoesNotConsumeRecoverableGeneration) {
    TemporaryDirectory temp("thor_training_checkpoint_repository_copy");
    ThorImplementation::TrainingCheckpointRepository repository(temp.path / "repo");

    auto generation = repository.prepareGeneration(31);
    writeFakeArtifact(generation.temporaryDirectory, "selected");
    repository.publishGeneration(generation);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, generation.name);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST, generation.name);

    const std::filesystem::path finalBest = temp.path / "final.tmp" / "best";
    repository.copyGenerationDurably(generation.name, finalBest);

    EXPECT_TRUE(std::filesystem::is_directory(generation.directory));
    EXPECT_EQ(readFile(generation.directory / "weights.bin"), "selected");
    EXPECT_EQ(readFile(finalBest / "weights.bin"), "selected");
    EXPECT_EQ(readFile(finalBest / "nested" / "optimizer.bin"), "optimizer-selected");
}

TEST(TrainingCheckpointRepository, PointerRemovalCannotDestroyAStillReferencedGeneration) {
    TemporaryDirectory temp("thor_training_checkpoint_repository_remove");
    ThorImplementation::TrainingCheckpointRepository repository(temp.path / "repo");

    auto generation = repository.prepareGeneration(5);
    writeFakeArtifact(generation.temporaryDirectory, "candidate");
    repository.publishGeneration(generation);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, generation.name);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST, generation.name);

    repository.removePointerIfMatches(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, generation.name);
    repository.retireGenerationIfUnreferenced(generation.name);
    EXPECT_TRUE(std::filesystem::is_directory(generation.directory));

    repository.removePointerIfMatches(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST, generation.name);
    repository.retireGenerationIfUnreferenced(generation.name);
    EXPECT_FALSE(std::filesystem::exists(generation.directory));
}

TEST(TrainingCheckpointRepository, RecoveryStateUsesCurrentAndIndependentlyCommittedBest) {
    TemporaryDirectory temp("thor_training_checkpoint_repository_recovery");
    ThorImplementation::TrainingCheckpointRepository repository(temp.path / "repo");

    auto best = repository.prepareGeneration(3);
    writeFakeArtifact(best.temporaryDirectory, "best");
    writeCheckpointState(best.temporaryDirectory, 3, 0, 10, 3, 0.25);
    repository.publishGeneration(best);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, best.name);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::BEST, best.name);

    auto current = repository.prepareGeneration(6);
    writeFakeArtifact(current.temporaryDirectory, "current");
    // CURRENT can carry newer in-memory selection metadata, but recovery BEST
    // is determined by the independently committed BEST pointer.
    writeCheckpointState(current.temporaryDirectory, 6, 0, 10, 6, 0.10);
    repository.publishGeneration(current);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, current.name);

    const auto recovery = repository.readRecoveryState();
    ASSERT_TRUE(recovery.has_value());
    EXPECT_EQ(recovery->currentGeneration, current.name);
    EXPECT_EQ(recovery->current.completedEpoch, 6u);
    EXPECT_EQ(recovery->current.phaseTargetCompletedEpoch, 10u);
    ASSERT_TRUE(recovery->bestGeneration.has_value());
    EXPECT_EQ(recovery->bestGeneration.value(), best.name);
    ASSERT_TRUE(recovery->best.has_value());
    ASSERT_TRUE(recovery->best->bestEpoch.has_value());
    ASSERT_TRUE(recovery->best->bestScore.has_value());
    EXPECT_EQ(recovery->best->bestEpoch.value(), 3u);
    EXPECT_DOUBLE_EQ(recovery->best->bestScore.value(), 0.25);
}


TEST(TrainingCheckpointRepository, CompletedStateProtectsSelectedGenerationAndCompactsUnselectedHistory) {
    TemporaryDirectory temp("thor_training_checkpoint_repository_completed");
    ThorImplementation::TrainingCheckpointRepository repository(temp.path / "repo");

    auto selected = repository.prepareGeneration(3);
    writeFakeArtifact(selected.temporaryDirectory, "selected");
    writeCheckpointState(selected.temporaryDirectory, 3, 0, 5);
    repository.publishGeneration(selected);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, selected.name);

    auto obsolete = repository.prepareGeneration(5);
    writeFakeArtifact(obsolete.temporaryDirectory, "obsolete");
    writeCheckpointState(obsolete.temporaryDirectory, 5, 0, 5);
    repository.publishGeneration(obsolete);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, obsolete.name);

    ThorImplementation::TrainingCheckpointRepository::CompletionState completion;
    completion.schemaVersion = 1;
    completion.selectedGeneration = selected.name;
    completion.selectedArtifactKind = "best";
    completion.bestGeneration = selected.name;
    completion.networkName = "network";
    completion.phaseStartCompletedEpoch = 0;
    completion.requestedPhaseEpochs = 5;
    completion.phaseTargetCompletedEpoch = 5;
    completion.completedEpoch = 5;
    completion.selectedEpoch = 3;
    completion.completionReason = "completed";
    completion.bestEpoch = 3;
    completion.bestScore = 0.25;
    completion.completedTrainingElapsedSeconds = 12.5;
    completion.deviceDatasetStorageReport.requested = Thor::DeviceDatasetStorage::STRICT;
    completion.deviceDatasetStorageReport.attempted = true;
    completion.deviceDatasetStorageReport.used = true;
    completion.deviceDatasetStorageReport.reason = "resident";
    completion.deviceDatasetStorageReport.examples = 42;
    completion.deviceDatasetStorageReport.requiredBytes = 4096;
    completion.deviceDatasetStorageReport.residentBytes = 4096;
    completion.deviceDatasetStorageReport.windowedDeviceCache.requested = Thor::WindowedDeviceCache::REQUIRED;
    completion.deviceDatasetStorageReport.windowedDeviceCache.attempted = true;
    completion.deviceDatasetStorageReport.windowedDeviceCache.used = true;
    completion.deviceDatasetStorageReport.windowedDeviceCache.reason = "active";
    completion.deviceDatasetStorageReport.windowedDeviceCache.hitRatio = 0.75f;
    completion.checkBestModelEveryEpochs = 0;
    completion.firstModelSelectionEpoch = 0;
    repository.publishCompleted(completion);

    const auto persisted = repository.readCompleted();
    ASSERT_TRUE(persisted.has_value());
    EXPECT_EQ(persisted->selectedGeneration, selected.name);
    EXPECT_EQ(persisted->completedEpoch, 5u);
    EXPECT_EQ(persisted->selectedEpoch, 3u);
    EXPECT_DOUBLE_EQ(persisted->completedTrainingElapsedSeconds, 12.5);
    EXPECT_EQ(persisted->deviceDatasetStorageReport.requested, Thor::DeviceDatasetStorage::STRICT);
    EXPECT_TRUE(persisted->deviceDatasetStorageReport.attempted);
    EXPECT_TRUE(persisted->deviceDatasetStorageReport.used);
    EXPECT_EQ(persisted->deviceDatasetStorageReport.reason, "resident");
    EXPECT_EQ(persisted->deviceDatasetStorageReport.examples, 42u);
    EXPECT_EQ(persisted->deviceDatasetStorageReport.requiredBytes, 4096u);
    EXPECT_EQ(persisted->deviceDatasetStorageReport.residentBytes, 4096u);
    EXPECT_EQ(persisted->deviceDatasetStorageReport.windowedDeviceCache.requested,
              Thor::WindowedDeviceCache::REQUIRED);
    EXPECT_TRUE(persisted->deviceDatasetStorageReport.windowedDeviceCache.used);
    EXPECT_FLOAT_EQ(persisted->deviceDatasetStorageReport.windowedDeviceCache.hitRatio, 0.75f);

    // COMPLETED protects the selected generation even before CURRENT/BEST are
    // retired. Compaction can then safely remove every other generation.
    repository.retireGenerationIfUnreferenced(selected.name);
    EXPECT_TRUE(std::filesystem::is_directory(selected.directory));
    repository.compactCompletedRepository();
    EXPECT_TRUE(std::filesystem::is_directory(selected.directory));
    EXPECT_FALSE(std::filesystem::exists(obsolete.directory));
    EXPECT_FALSE(repository.readPointer(
        ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT).has_value());
    EXPECT_FALSE(repository.readPointer(
        ThorImplementation::TrainingCheckpointRepository::Pointer::BEST).has_value());
    ASSERT_TRUE(repository.readCompleted().has_value());
}

TEST(TrainingCheckpointRepository, CompletedStateRejectsSelectedGenerationFromWrongEpoch) {
    TemporaryDirectory temp("thor_training_checkpoint_repository_bad_completed");
    ThorImplementation::TrainingCheckpointRepository repository(temp.path / "repo");

    auto selected = repository.prepareGeneration(2);
    writeFakeArtifact(selected.temporaryDirectory, "selected");
    writeCheckpointState(selected.temporaryDirectory, 2, 0, 5);
    repository.publishGeneration(selected);

    ThorImplementation::TrainingCheckpointRepository::CompletionState completion;
    completion.schemaVersion = 1;
    completion.selectedGeneration = selected.name;
    completion.selectedArtifactKind = "latest";
    completion.latestGeneration = selected.name;
    completion.networkName = "network";
    completion.phaseStartCompletedEpoch = 0;
    completion.requestedPhaseEpochs = 5;
    completion.phaseTargetCompletedEpoch = 5;
    completion.completedEpoch = 5;
    completion.selectedEpoch = 3;  // generation is epoch 2
    completion.completionReason = "completed";

    EXPECT_THROW(repository.publishCompleted(completion), std::invalid_argument);
    EXPECT_FALSE(repository.readCompleted().has_value());
}

TEST(TrainingCheckpointRepository, RecoveryRejectsInconsistentEpochBounds) {
    TemporaryDirectory temp("thor_training_checkpoint_repository_bad_recovery");
    ThorImplementation::TrainingCheckpointRepository repository(temp.path / "repo");

    auto current = repository.prepareGeneration(11);
    writeFakeArtifact(current.temporaryDirectory, "bad");
    std::ofstream out(current.temporaryDirectory / "checkpoint_state.json", std::ios::binary | std::ios::trunc);
    out << R"({
  "schema_version": 1,
  "checkpoint_reason": "test",
  "completed_epoch": 11,
  "phase_start_completed_epoch": 0,
  "requested_phase_epochs": 10,
  "phase_target_completed_epoch": 10,
  "best_epoch": null,
  "best_score": null,
  "check_best_model_every_epochs": 0,
  "first_model_selection_epoch": 0,
  "checkpoint_every_epochs": 1,
  "device_dataset_storage": {
    "requested": "off",
    "attempted": false,
    "used": false,
    "reason": "",
    "examples": 0,
    "required_bytes": 0,
    "available_bytes_after_placement": 0,
    "resident_bytes": 0,
    "resident_cache_hit": false,
    "resident_construction_joined": false,
    "resident_construction_started": false,
    "materialization_seconds": 0.0,
    "windowed_device_cache": {
      "requested": "auto",
      "attempted": false,
      "used": false,
      "reason": "",
      "eligible_sources": 0,
      "active_sources": 0,
      "eligible_source_bytes": 0,
      "budget_bytes": 0,
      "max_access_policy_window_bytes": 0,
      "active_unique_bytes": 0,
      "hit_ratio": 0.0
    }
  }
})";
    out.close();
    repository.publishGeneration(current);
    repository.updatePointer(ThorImplementation::TrainingCheckpointRepository::Pointer::CURRENT, current.name);

    EXPECT_THROW((void)repository.readRecoveryState(), std::runtime_error);
}

}  // namespace
