#pragma once

#include "DeepLearning/Api/Training/DeviceDatasetStorage.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace ThorImplementation {

// Filesystem transaction primitive used by training checkpointing. Generations
// are immutable once published. CURRENT/BEST are small atomically-replaced text
// pointers to published generation names. COMPLETING/COMPLETED carry the
// terminal stage transaction: COMPLETING makes finished training durable before
// final artifact publication, and COMPLETED makes later fit() calls idempotent.
// Resume readers consume only published generations; temporary generations are
// never visible.
class TrainingCheckpointRepository {
   public:
    enum class Pointer { CURRENT, BEST };

    struct PreparedGeneration {
        std::string name;
        std::filesystem::path temporaryDirectory;
        std::filesystem::path directory;
    };

    struct CheckpointState {
        uint32_t schemaVersion = 0;
        std::string checkpointReason{};
        uint64_t completedEpoch = 0;
        uint64_t phaseStartCompletedEpoch = 0;
        uint32_t requestedPhaseEpochs = 0;
        uint64_t phaseTargetCompletedEpoch = 0;
        std::optional<uint64_t> bestEpoch{};
        std::optional<double> bestScore{};
        uint32_t checkBestModelEveryEpochs = 0;
        uint64_t firstModelSelectionEpoch = 0;
        uint32_t checkpointEveryEpochs = 0;
        Thor::DeviceDatasetStorageReport deviceDatasetStorageReport{};
    };


    struct CompletionState {
        uint32_t schemaVersion = 0;
        std::string selectedGeneration{};
        std::string selectedArtifactKind{};
        std::optional<std::string> latestGeneration{};
        std::optional<std::string> bestGeneration{};
        std::string networkName{};
        uint64_t phaseStartCompletedEpoch = 0;
        uint32_t requestedPhaseEpochs = 0;
        uint64_t phaseTargetCompletedEpoch = 0;
        uint64_t completedEpoch = 0;
        uint64_t selectedEpoch = 0;
        std::string completionReason{};
        std::optional<uint64_t> bestEpoch{};
        std::optional<double> bestScore{};
        std::optional<double> latestScore{};
        std::optional<double> latestTrainingLoss{};
        std::optional<double> latestValidationLoss{};
        double completedTrainingElapsedSeconds = 0.0;
        uint32_t checkBestModelEveryEpochs = 0;
        uint64_t firstModelSelectionEpoch = 0;
        Thor::DeviceDatasetStorageReport deviceDatasetStorageReport{};
    };

    struct RecoveryState {
        std::string currentGeneration{};
        std::filesystem::path currentDirectory{};
        CheckpointState current{};
        std::optional<std::string> bestGeneration{};
        std::optional<std::filesystem::path> bestDirectory{};
        std::optional<CheckpointState> best{};
    };

    explicit TrainingCheckpointRepository(std::filesystem::path rootDirectory);

    [[nodiscard]] static std::filesystem::path repositoryDirectoryForModelArtifact(
        const std::filesystem::path& modelArtifactDirectory);

    [[nodiscard]] const std::filesystem::path& getRootDirectory() const { return rootDirectory; }
    [[nodiscard]] std::filesystem::path getGenerationsDirectory() const;
    [[nodiscard]] std::filesystem::path getPointerPath(Pointer pointer) const;
    [[nodiscard]] std::filesystem::path getCompletingPath() const { return rootDirectory / "COMPLETING"; }
    [[nodiscard]] std::filesystem::path getCompletedPath() const { return rootDirectory / "COMPLETED"; }
    [[nodiscard]] std::filesystem::path getGenerationDirectory(const std::string& generationName) const;

    // Reserve unique names for a generation. The caller writes the complete
    // artifact into temporaryDirectory and then calls publishGeneration().
    // Neither path is visible through CURRENT/BEST until updatePointer().
    PreparedGeneration prepareGeneration(uint64_t epoch);

    // Commit a completely-written temporary generation under its immutable
    // final name. The durable rename is the generation publication point.
    void publishGeneration(const PreparedGeneration& generation);

    // Atomically replace CURRENT or BEST with a pointer to an already-published
    // generation. The pointer file contents and its directory entry are made
    // durable before this method returns.
    void updatePointer(Pointer pointer, const std::string& generationName);

    [[nodiscard]] std::optional<std::string> readPointer(Pointer pointer) const;

    // Parse and validate the checkpoint metadata associated with one immutable
    // generation. readRecoveryState() treats CURRENT as the authoritative
    // crash-resume state and BEST as the independently committed model-selection
    // incumbent. A missing CURRENT means there is no resumable work.
    [[nodiscard]] CheckpointState readCheckpointState(const std::string& generationName) const;
    [[nodiscard]] std::optional<RecoveryState> readRecoveryState() const;

    // COMPLETING records that optimizer work is terminal before final artifact
    // publication begins. Once save_model_dir is durable, commitCompletion()
    // atomically renames it to COMPLETED. A later fit() can therefore either
    // finish an interrupted finalization or return a completed result without
    // scheduling training.
    void prepareCompletion(const CompletionState& completion);
    void commitCompletion();
    void publishCompleted(const CompletionState& completion);
    [[nodiscard]] std::optional<CompletionState> readCompleting() const;
    [[nodiscard]] std::optional<CompletionState> readCompleted() const;

    // After COMPLETED is durable the selected generation is authoritative.
    // Under the default replacement policy CURRENT/BEST and all unselected
    // generations may be retired. A crash during this cleanup remains safe
    // because COMPLETED always continues to protect its selected generation.
    void compactCompletedRepository();

    // Remove a pointer only when it still names generationName. These helpers
    // are for retiring an in-memory candidate after a startup retry; they never
    // destroy a generation while CURRENT or BEST still references it.
    void removePointerIfMatches(Pointer pointer, const std::string& generationName);
    void retireGenerationIfUnreferenced(const std::string& generationName);

    // Copy an immutable generation into an independent destination while
    // preserving the source generation. This is used by the existing final
    // model artifact handoff so a crash during finalization cannot consume the
    // only recoverable checkpoint.
    void copyGenerationDurably(const std::string& generationName,
                               const std::filesystem::path& destination) const;

    // Safe only after another durable artifact has become authoritative. Used
    // after successful legacy finalization; failures may be ignored by callers
    // because retaining a repository is always safer than deleting it early.
    void removeRepositoryDurably();

   private:
    [[nodiscard]] std::string nextGenerationName(uint64_t epoch);
    [[nodiscard]] std::filesystem::path uniquePointerTemporaryPath(Pointer pointer);
    void validateCompletionState(const CompletionState& completion) const;
    void writeCompletionStateAtomically(const std::filesystem::path& destination,
                                        const CompletionState& completion);
    [[nodiscard]] std::optional<CompletionState> readCompletionStateFile(
        const std::filesystem::path& path) const;
    static const char* pointerName(Pointer pointer);
    static void validateGenerationName(const std::string& generationName);
    static void removePathIfExists(const std::filesystem::path& path);
    static void syncCopiedTree(const std::filesystem::path& root);

    std::filesystem::path rootDirectory;
    uint64_t localSequence = 0;
};

}  // namespace ThorImplementation
