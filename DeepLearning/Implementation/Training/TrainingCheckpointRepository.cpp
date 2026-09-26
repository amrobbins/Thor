#include "DeepLearning/Implementation/Training/TrainingCheckpointRepository.h"

#include "Utilities/Common/FilesystemDurability.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace ThorImplementation {
namespace {

[[nodiscard]] size_t pathDepth(const std::filesystem::path& path) {
    return static_cast<size_t>(std::distance(path.begin(), path.end()));
}

void validateDeviceDatasetStorageReport(const Thor::DeviceDatasetStorageReport& report,
                                        const std::string& context) {
    if (!std::isfinite(report.materializationSeconds)) {
        throw std::invalid_argument(context + " device_dataset_storage.materialization_seconds must be finite.");
    }
    if (!std::isfinite(report.windowedDeviceCache.hitRatio)) {
        throw std::invalid_argument(context + " device_dataset_storage.windowed_device_cache.hit_ratio must be finite.");
    }
}

nlohmann::json deviceDatasetStorageReportToJson(const Thor::DeviceDatasetStorageReport& report) {
    validateDeviceDatasetStorageReport(report, "Training repository state");
    nlohmann::json windowed;
    windowed["requested"] = Thor::windowedDeviceCacheName(report.windowedDeviceCache.requested);
    windowed["attempted"] = report.windowedDeviceCache.attempted;
    windowed["used"] = report.windowedDeviceCache.used;
    windowed["reason"] = report.windowedDeviceCache.reason;
    windowed["eligible_sources"] = report.windowedDeviceCache.eligibleSources;
    windowed["active_sources"] = report.windowedDeviceCache.activeSources;
    windowed["eligible_source_bytes"] = report.windowedDeviceCache.eligibleSourceBytes;
    windowed["budget_bytes"] = report.windowedDeviceCache.budgetBytes;
    windowed["max_access_policy_window_bytes"] = report.windowedDeviceCache.maxAccessPolicyWindowBytes;
    windowed["active_unique_bytes"] = report.windowedDeviceCache.activeUniqueBytes;
    windowed["hit_ratio"] = report.windowedDeviceCache.hitRatio;

    nlohmann::json storage;
    storage["requested"] = Thor::deviceDatasetStorageName(report.requested);
    storage["attempted"] = report.attempted;
    storage["used"] = report.used;
    storage["reason"] = report.reason;
    storage["examples"] = report.examples;
    storage["required_bytes"] = report.requiredBytes;
    storage["available_bytes_after_placement"] = report.availableBytesAfterPlacement;
    storage["resident_bytes"] = report.residentBytes;
    storage["resident_cache_hit"] = report.residentCacheHit;
    storage["resident_construction_joined"] = report.residentConstructionJoined;
    storage["resident_construction_started"] = report.residentConstructionStarted;
    storage["materialization_seconds"] = report.materializationSeconds;
    storage["windowed_device_cache"] = std::move(windowed);
    return storage;
}

Thor::DeviceDatasetStorageReport deviceDatasetStorageReportFromJson(
    const nlohmann::json& metadata,
    const std::string& context) {
    if (!metadata.is_object()) {
        throw std::runtime_error(context + " requires object field 'device_dataset_storage'.");
    }
    auto requiredString = [&](const nlohmann::json& object, const char* key, const std::string& fieldContext) {
        if (!object.contains(key) || !object.at(key).is_string()) {
            throw std::runtime_error(fieldContext + " requires string field '" + key + "'.");
        }
        return object.at(key).get<std::string>();
    };
    auto requiredBool = [&](const nlohmann::json& object, const char* key, const std::string& fieldContext) {
        if (!object.contains(key) || !object.at(key).is_boolean()) {
            throw std::runtime_error(fieldContext + " requires boolean field '" + key + "'.");
        }
        return object.at(key).get<bool>();
    };
    auto requiredUint64 = [&](const nlohmann::json& object, const char* key, const std::string& fieldContext) {
        if (!object.contains(key) || !object.at(key).is_number_unsigned()) {
            throw std::runtime_error(fieldContext + " requires unsigned integer field '" + key + "'.");
        }
        return object.at(key).get<uint64_t>();
    };
    auto requiredDouble = [&](const nlohmann::json& object, const char* key, const std::string& fieldContext) {
        if (!object.contains(key) || !object.at(key).is_number()) {
            throw std::runtime_error(fieldContext + " requires numeric field '" + key + "'.");
        }
        const double value = object.at(key).get<double>();
        if (!std::isfinite(value)) {
            throw std::runtime_error(fieldContext + " requires finite numeric field '" + key + "'.");
        }
        return value;
    };

    Thor::DeviceDatasetStorageReport report;
    const std::string storageContext = context + " device_dataset_storage";
    try {
        report.requested = Thor::deviceDatasetStorageFromName(
            requiredString(metadata, "requested", storageContext));
    } catch (const std::exception& e) {
        throw std::runtime_error(storageContext + " has invalid requested policy: " + e.what());
    }
    report.attempted = requiredBool(metadata, "attempted", storageContext);
    report.used = requiredBool(metadata, "used", storageContext);
    report.reason = requiredString(metadata, "reason", storageContext);
    report.examples = requiredUint64(metadata, "examples", storageContext);
    report.requiredBytes = requiredUint64(metadata, "required_bytes", storageContext);
    report.availableBytesAfterPlacement = requiredUint64(
        metadata, "available_bytes_after_placement", storageContext);
    report.residentBytes = requiredUint64(metadata, "resident_bytes", storageContext);
    report.residentCacheHit = requiredBool(metadata, "resident_cache_hit", storageContext);
    report.residentConstructionJoined = requiredBool(
        metadata, "resident_construction_joined", storageContext);
    report.residentConstructionStarted = requiredBool(
        metadata, "resident_construction_started", storageContext);
    report.materializationSeconds = requiredDouble(
        metadata, "materialization_seconds", storageContext);

    if (!metadata.contains("windowed_device_cache") ||
        !metadata.at("windowed_device_cache").is_object()) {
        throw std::runtime_error(storageContext + " requires object field 'windowed_device_cache'.");
    }
    const nlohmann::json& windowed = metadata.at("windowed_device_cache");
    const std::string windowedContext = storageContext + ".windowed_device_cache";
    try {
        report.windowedDeviceCache.requested = Thor::windowedDeviceCacheFromName(
            requiredString(windowed, "requested", windowedContext));
    } catch (const std::exception& e) {
        throw std::runtime_error(windowedContext + " has invalid requested policy: " + e.what());
    }
    report.windowedDeviceCache.attempted = requiredBool(windowed, "attempted", windowedContext);
    report.windowedDeviceCache.used = requiredBool(windowed, "used", windowedContext);
    report.windowedDeviceCache.reason = requiredString(windowed, "reason", windowedContext);
    report.windowedDeviceCache.eligibleSources = requiredUint64(windowed, "eligible_sources", windowedContext);
    report.windowedDeviceCache.activeSources = requiredUint64(windowed, "active_sources", windowedContext);
    report.windowedDeviceCache.eligibleSourceBytes = requiredUint64(
        windowed, "eligible_source_bytes", windowedContext);
    report.windowedDeviceCache.budgetBytes = requiredUint64(windowed, "budget_bytes", windowedContext);
    report.windowedDeviceCache.maxAccessPolicyWindowBytes = requiredUint64(
        windowed, "max_access_policy_window_bytes", windowedContext);
    report.windowedDeviceCache.activeUniqueBytes = requiredUint64(
        windowed, "active_unique_bytes", windowedContext);
    report.windowedDeviceCache.hitRatio = static_cast<float>(requiredDouble(
        windowed, "hit_ratio", windowedContext));
    validateDeviceDatasetStorageReport(report, context);
    return report;
}

}  // namespace

TrainingCheckpointRepository::TrainingCheckpointRepository(std::filesystem::path rootDirectory)
    : rootDirectory(std::move(rootDirectory)) {
    if (this->rootDirectory.empty()) {
        throw std::invalid_argument("Training checkpoint repository directory must not be empty.");
    }
}

std::filesystem::path TrainingCheckpointRepository::repositoryDirectoryForModelArtifact(
    const std::filesystem::path& modelArtifactDirectory) {
    std::filesystem::path parent = modelArtifactDirectory.parent_path();
    std::string filename = modelArtifactDirectory.filename().string();
    if (filename.empty()) {
        filename = "model";
    }
    return parent / ("." + filename + ".training_repository");
}

std::filesystem::path TrainingCheckpointRepository::getGenerationsDirectory() const {
    return rootDirectory / "generations";
}

const char* TrainingCheckpointRepository::pointerName(Pointer pointer) {
    switch (pointer) {
        case Pointer::CURRENT:
            return "CURRENT";
        case Pointer::BEST:
            return "BEST";
    }
    throw std::logic_error("Unknown training checkpoint repository pointer.");
}

std::filesystem::path TrainingCheckpointRepository::getPointerPath(Pointer pointer) const {
    return rootDirectory / pointerName(pointer);
}

void TrainingCheckpointRepository::validateGenerationName(const std::string& generationName) {
    if (generationName.empty() || generationName == "." || generationName == ".." ||
        generationName.find('/') != std::string::npos || generationName.find('\\') != std::string::npos) {
        throw std::invalid_argument("Invalid training checkpoint generation name: '" + generationName + "'.");
    }
}

std::filesystem::path TrainingCheckpointRepository::getGenerationDirectory(const std::string& generationName) const {
    validateGenerationName(generationName);
    return getGenerationsDirectory() / generationName;
}

std::string TrainingCheckpointRepository::nextGenerationName(uint64_t epoch) {
    std::ostringstream out;
    out << "epoch_" << std::setw(20) << std::setfill('0') << epoch << "_pid_" << ::getpid() << "_seq_" << std::setw(8)
        << std::setfill('0') << localSequence++;
    return out.str();
}

TrainingCheckpointRepository::PreparedGeneration TrainingCheckpointRepository::prepareGeneration(uint64_t epoch) {
    Thor::FilesystemDurability::createDirectoriesDurably(getGenerationsDirectory());

    for (;;) {
        PreparedGeneration generation;
        generation.name = nextGenerationName(epoch);
        generation.directory = getGenerationDirectory(generation.name);
        generation.temporaryDirectory = getGenerationsDirectory() / ("." + generation.name + ".tmp");

        std::error_code errorCode;
        const bool finalExists = std::filesystem::exists(generation.directory, errorCode);
        if (errorCode) {
            throw std::runtime_error("Failed to inspect training checkpoint generation '" + generation.directory.string() +
                                     "': " + errorCode.message());
        }
        errorCode.clear();
        const bool temporaryExists = std::filesystem::exists(generation.temporaryDirectory, errorCode);
        if (errorCode) {
            throw std::runtime_error("Failed to inspect temporary training checkpoint generation '" +
                                     generation.temporaryDirectory.string() + "': " + errorCode.message());
        }
        if (!finalExists && !temporaryExists) {
            return generation;
        }
    }
}

void TrainingCheckpointRepository::publishGeneration(const PreparedGeneration& generation) {
    validateGenerationName(generation.name);
    if (generation.directory != getGenerationDirectory(generation.name)) {
        throw std::invalid_argument("Training checkpoint generation destination does not belong to this repository.");
    }
    if (generation.temporaryDirectory.parent_path().lexically_normal() != getGenerationsDirectory().lexically_normal()) {
        throw std::invalid_argument("Training checkpoint temporary generation does not belong to this repository.");
    }

    std::error_code errorCode;
    if (!std::filesystem::is_directory(generation.temporaryDirectory, errorCode) || errorCode) {
        throw std::runtime_error("Training checkpoint temporary generation is missing or is not a directory: '" +
                                 generation.temporaryDirectory.string() + "'.");
    }
    errorCode.clear();
    if (std::filesystem::exists(generation.directory, errorCode) || errorCode) {
        if (errorCode) {
            throw std::runtime_error("Failed to inspect training checkpoint generation destination '" +
                                     generation.directory.string() + "': " + errorCode.message());
        }
        throw std::runtime_error("Training checkpoint generation already exists and is immutable: '" +
                                 generation.directory.string() + "'.");
    }

    // The caller is responsible for making the artifact contents durable before
    // this point (PlacedNetwork::save() has that contract). This rename then
    // publishes the complete immutable generation and fsyncs its directory.
    Thor::FilesystemDurability::durableRename(generation.temporaryDirectory, generation.directory);
}

std::filesystem::path TrainingCheckpointRepository::uniquePointerTemporaryPath(Pointer pointer) {
    Thor::FilesystemDurability::createDirectoriesDurably(rootDirectory);
    std::ostringstream name;
    name << "." << pointerName(pointer) << ".tmp." << ::getpid() << "." << localSequence++;
    return rootDirectory / name.str();
}

void TrainingCheckpointRepository::updatePointer(Pointer pointer, const std::string& generationName) {
    validateGenerationName(generationName);
    const std::filesystem::path generationDirectory = getGenerationDirectory(generationName);
    std::error_code errorCode;
    if (!std::filesystem::is_directory(generationDirectory, errorCode) || errorCode) {
        throw std::runtime_error("Cannot point " + std::string(pointerName(pointer)) +
                                 " at missing training checkpoint generation '" + generationDirectory.string() + "'.");
    }

    const std::filesystem::path pointerPath = getPointerPath(pointer);
    const std::filesystem::path temporaryPath = uniquePointerTemporaryPath(pointer);
    removePathIfExists(temporaryPath);
    try {
        {
            std::ofstream out(temporaryPath, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("Unable to open temporary training checkpoint pointer for writing: '" +
                                         temporaryPath.string() + "'.");
            }
            out << generationName << '\n';
            out.flush();
            if (!out) {
                throw std::runtime_error("Failed while writing temporary training checkpoint pointer: '" +
                                         temporaryPath.string() + "'.");
            }
        }
        Thor::FilesystemDurability::syncFile(temporaryPath);
        // POSIX rename atomically replaces an existing regular pointer file. The
        // durableRename fsync makes that replacement survive a machine failure.
        Thor::FilesystemDurability::durableRename(temporaryPath, pointerPath);
    } catch (...) {
        removePathIfExists(temporaryPath);
        throw;
    }
}

std::optional<std::string> TrainingCheckpointRepository::readPointer(Pointer pointer) const {
    const std::filesystem::path pointerPath = getPointerPath(pointer);
    std::error_code errorCode;
    const bool exists = std::filesystem::exists(pointerPath, errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to inspect training checkpoint pointer '" + pointerPath.string() + "': " +
                                 errorCode.message());
    }
    if (!exists) {
        return std::nullopt;
    }

    std::ifstream in(pointerPath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Unable to open training checkpoint pointer: '" + pointerPath.string() + "'.");
    }
    std::string generationName;
    std::getline(in, generationName);
    if (!in && !in.eof()) {
        throw std::runtime_error("Failed while reading training checkpoint pointer: '" + pointerPath.string() + "'.");
    }
    validateGenerationName(generationName);
    return generationName;
}


TrainingCheckpointRepository::CheckpointState TrainingCheckpointRepository::readCheckpointState(
    const std::string& generationName) const {
    validateGenerationName(generationName);
    const std::filesystem::path generationDirectory = getGenerationDirectory(generationName);
    const std::filesystem::path metadataPath = generationDirectory / "checkpoint_state.json";

    std::error_code errorCode;
    if (!std::filesystem::is_directory(generationDirectory, errorCode) || errorCode) {
        throw std::runtime_error("Training checkpoint pointer names a missing generation '" +
                                 generationDirectory.string() + "'.");
    }

    std::ifstream in(metadataPath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Training checkpoint generation is missing checkpoint_state.json: '" +
                                 metadataPath.string() + "'.");
    }

    nlohmann::json metadata;
    try {
        in >> metadata;
    } catch (const std::exception& e) {
        throw std::runtime_error("Unable to parse training checkpoint metadata '" + metadataPath.string() +
                                 "': " + e.what());
    }

    auto requiredUint64 = [&](const char* key) -> uint64_t {
        if (!metadata.contains(key) || !metadata.at(key).is_number_unsigned()) {
            throw std::runtime_error("Training checkpoint metadata '" + metadataPath.string() +
                                     "' requires unsigned integer field '" + key + "'.");
        }
        return metadata.at(key).get<uint64_t>();
    };
    auto requiredUint32 = [&](const char* key) -> uint32_t {
        const uint64_t value = requiredUint64(key);
        if (value > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Training checkpoint metadata field '" + std::string(key) +
                                     "' exceeds uint32 range in '" + metadataPath.string() + "'.");
        }
        return static_cast<uint32_t>(value);
    };
    auto optionalUint64 = [&](const char* key) -> std::optional<uint64_t> {
        if (!metadata.contains(key) || metadata.at(key).is_null()) {
            return std::nullopt;
        }
        if (!metadata.at(key).is_number_unsigned()) {
            throw std::runtime_error("Training checkpoint metadata field '" + std::string(key) +
                                     "' must be an unsigned integer or null in '" + metadataPath.string() + "'.");
        }
        return metadata.at(key).get<uint64_t>();
    };
    auto optionalDouble = [&](const char* key) -> std::optional<double> {
        if (!metadata.contains(key) || metadata.at(key).is_null()) {
            return std::nullopt;
        }
        if (!metadata.at(key).is_number()) {
            throw std::runtime_error("Training checkpoint metadata field '" + std::string(key) +
                                     "' must be numeric or null in '" + metadataPath.string() + "'.");
        }
        const double value = metadata.at(key).get<double>();
        if (!std::isfinite(value)) {
            throw std::runtime_error("Training checkpoint metadata field '" + std::string(key) +
                                     "' must be finite in '" + metadataPath.string() + "'.");
        }
        return value;
    };

    CheckpointState state;
    state.schemaVersion = requiredUint32("schema_version");
    if (state.schemaVersion != 1) {
        throw std::runtime_error("Unsupported training checkpoint metadata schema version " +
                                 std::to_string(state.schemaVersion) + " in '" + metadataPath.string() + "'.");
    }
    if (!metadata.contains("checkpoint_reason") || !metadata.at("checkpoint_reason").is_string()) {
        throw std::runtime_error("Training checkpoint metadata '" + metadataPath.string() +
                                 "' requires string field 'checkpoint_reason'.");
    }
    state.checkpointReason = metadata.at("checkpoint_reason").get<std::string>();
    state.completedEpoch = requiredUint64("completed_epoch");
    state.phaseStartCompletedEpoch = requiredUint64("phase_start_completed_epoch");
    state.requestedPhaseEpochs = requiredUint32("requested_phase_epochs");
    state.phaseTargetCompletedEpoch = requiredUint64("phase_target_completed_epoch");
    state.bestEpoch = optionalUint64("best_epoch");
    state.bestScore = optionalDouble("best_score");
    state.checkBestModelEveryEpochs = requiredUint32("check_best_model_every_epochs");
    state.firstModelSelectionEpoch = requiredUint64("first_model_selection_epoch");
    state.checkpointEveryEpochs = requiredUint32("checkpoint_every_epochs");
    if (!metadata.contains("device_dataset_storage")) {
        throw std::runtime_error("Training checkpoint metadata '" + metadataPath.string() +
                                 "' requires object field 'device_dataset_storage'.");
    }
    state.deviceDatasetStorageReport = deviceDatasetStorageReportFromJson(
        metadata.at("device_dataset_storage"),
        "Training checkpoint metadata '" + metadataPath.string() + "'");

    if (state.requestedPhaseEpochs == 0 ||
        state.phaseStartCompletedEpoch >
            std::numeric_limits<uint64_t>::max() - state.requestedPhaseEpochs ||
        state.phaseTargetCompletedEpoch != state.phaseStartCompletedEpoch + state.requestedPhaseEpochs ||
        state.completedEpoch < state.phaseStartCompletedEpoch ||
        state.completedEpoch > state.phaseTargetCompletedEpoch) {
        throw std::runtime_error("Training checkpoint metadata has inconsistent phase epoch bounds in '" +
                                 metadataPath.string() + "'.");
    }
    if (state.bestEpoch.has_value() != state.bestScore.has_value()) {
        throw std::runtime_error("Training checkpoint metadata must contain best_epoch and best_score together in '" +
                                 metadataPath.string() + "'.");
    }
    if (state.bestEpoch.has_value() &&
        (state.bestEpoch.value() < state.phaseStartCompletedEpoch ||
         state.bestEpoch.value() > state.completedEpoch)) {
        throw std::runtime_error("Training checkpoint metadata best_epoch is outside the completed phase range in '" +
                                 metadataPath.string() + "'.");
    }
    return state;
}

std::optional<TrainingCheckpointRepository::RecoveryState> TrainingCheckpointRepository::readRecoveryState() const {
    const std::optional<std::string> currentGeneration = readPointer(Pointer::CURRENT);
    if (!currentGeneration.has_value()) {
        return std::nullopt;
    }

    RecoveryState recovery;
    recovery.currentGeneration = currentGeneration.value();
    recovery.currentDirectory = getGenerationDirectory(recovery.currentGeneration);
    recovery.current = readCheckpointState(recovery.currentGeneration);

    recovery.bestGeneration = readPointer(Pointer::BEST);
    if (recovery.bestGeneration.has_value()) {
        recovery.bestDirectory = getGenerationDirectory(recovery.bestGeneration.value());
        recovery.best = readCheckpointState(recovery.bestGeneration.value());
        if (!recovery.best->bestEpoch.has_value() || !recovery.best->bestScore.has_value()) {
            throw std::runtime_error("BEST training checkpoint generation does not contain committed best-model state: '" +
                                     recovery.bestDirectory->string() + "'.");
        }
        if (recovery.best->phaseStartCompletedEpoch != recovery.current.phaseStartCompletedEpoch ||
            recovery.best->phaseTargetCompletedEpoch != recovery.current.phaseTargetCompletedEpoch) {
            throw std::runtime_error("CURRENT and BEST training checkpoint generations belong to different phase ranges.");
        }
        if (recovery.best->bestEpoch.value() > recovery.current.completedEpoch) {
            throw std::runtime_error("BEST training checkpoint epoch is newer than CURRENT completed epoch.");
        }
    }
    return recovery;
}


void TrainingCheckpointRepository::validateCompletionState(const CompletionState& completion) const {
    if (completion.schemaVersion != 1) {
        throw std::invalid_argument("Training completion state schema_version must be 1.");
    }
    validateGenerationName(completion.selectedGeneration);
    if (completion.selectedArtifactKind != "best" && completion.selectedArtifactKind != "latest") {
        throw std::invalid_argument("Training completion selected_artifact_kind must be 'best' or 'latest'.");
    }
    auto validateOptionalGeneration = [&](const std::optional<std::string>& generation, const char* field) {
        if (!generation.has_value()) return;
        try {
            validateGenerationName(generation.value());
        } catch (const std::exception&) {
            throw std::invalid_argument("Training completion " + std::string(field) + " is invalid.");
        }
        std::error_code errorCode;
        if (!std::filesystem::is_directory(getGenerationDirectory(generation.value()), errorCode) || errorCode) {
            throw std::invalid_argument("Training completion " + std::string(field) +
                                        " names a missing checkpoint generation.");
        }
    };
    validateOptionalGeneration(completion.latestGeneration, "latest_generation");
    validateOptionalGeneration(completion.bestGeneration, "best_generation");
    if (completion.selectedArtifactKind == "latest") {
        if (!completion.latestGeneration.has_value() ||
            completion.latestGeneration.value() != completion.selectedGeneration) {
            throw std::invalid_argument("Training completion selected latest artifact must match latest_generation.");
        }
    } else {
        if (!completion.bestGeneration.has_value() ||
            completion.bestGeneration.value() != completion.selectedGeneration) {
            throw std::invalid_argument("Training completion selected best artifact must match best_generation.");
        }
    }
    if (completion.networkName.empty()) {
        throw std::invalid_argument("Training completion network_name must not be empty.");
    }
    if (completion.requestedPhaseEpochs == 0 ||
        completion.phaseStartCompletedEpoch >
            std::numeric_limits<uint64_t>::max() - completion.requestedPhaseEpochs ||
        completion.phaseTargetCompletedEpoch !=
            completion.phaseStartCompletedEpoch + completion.requestedPhaseEpochs ||
        completion.completedEpoch < completion.phaseStartCompletedEpoch ||
        completion.completedEpoch > completion.phaseTargetCompletedEpoch ||
        completion.selectedEpoch < completion.phaseStartCompletedEpoch ||
        completion.selectedEpoch > completion.completedEpoch) {
        throw std::invalid_argument("Training completion state has inconsistent phase epoch bounds.");
    }
    if (completion.completionReason != "completed" && completion.completionReason != "early_completed") {
        throw std::invalid_argument("Training completion reason must be 'completed' or 'early_completed'.");
    }
    if (completion.completionReason == "completed" &&
        completion.completedEpoch != completion.phaseTargetCompletedEpoch) {
        throw std::invalid_argument("Normally completed training must reach the persisted phase target epoch.");
    }
    if (completion.bestEpoch.has_value() != completion.bestScore.has_value()) {
        throw std::invalid_argument("Training completion best_epoch and best_score must be present together.");
    }
    if (completion.bestEpoch.has_value() &&
        (completion.bestEpoch.value() < completion.phaseStartCompletedEpoch ||
         completion.bestEpoch.value() > completion.completedEpoch)) {
        throw std::invalid_argument("Training completion best_epoch is outside the completed phase range.");
    }
    if (completion.selectedArtifactKind == "best" &&
        (!completion.bestEpoch.has_value() || completion.selectedEpoch != completion.bestEpoch.value())) {
        throw std::invalid_argument("Training completion selected best artifact must match best_epoch.");
    }
    auto requireFiniteOptional = [](const std::optional<double>& value, const char* field) {
        if (value.has_value() && !std::isfinite(value.value())) {
            throw std::invalid_argument("Training completion " + std::string(field) + " must be finite when present.");
        }
    };
    requireFiniteOptional(completion.latestScore, "latest_score");
    requireFiniteOptional(completion.latestTrainingLoss, "latest_training_loss");
    requireFiniteOptional(completion.latestValidationLoss, "latest_validation_loss");
    if (!std::isfinite(completion.completedTrainingElapsedSeconds) ||
        completion.completedTrainingElapsedSeconds < 0.0) {
        throw std::invalid_argument("Training completion elapsed seconds must be finite and non-negative.");
    }

    const CheckpointState selectedCheckpoint = readCheckpointState(completion.selectedGeneration);
    if (selectedCheckpoint.completedEpoch != completion.selectedEpoch ||
        selectedCheckpoint.phaseStartCompletedEpoch != completion.phaseStartCompletedEpoch ||
        selectedCheckpoint.phaseTargetCompletedEpoch != completion.phaseTargetCompletedEpoch) {
        throw std::invalid_argument(
            "Training completion selected generation does not match the completion phase/selected epoch.");
    }
    if (completion.latestGeneration.has_value()) {
        const CheckpointState latest = readCheckpointState(completion.latestGeneration.value());
        if (latest.completedEpoch != completion.completedEpoch ||
            latest.phaseStartCompletedEpoch != completion.phaseStartCompletedEpoch ||
            latest.phaseTargetCompletedEpoch != completion.phaseTargetCompletedEpoch) {
            throw std::invalid_argument("Training completion latest_generation does not match completed_epoch.");
        }
    }
    validateDeviceDatasetStorageReport(
        completion.deviceDatasetStorageReport, "Training completion state");
    if (completion.bestGeneration.has_value()) {
        if (!completion.bestEpoch.has_value()) {
            throw std::invalid_argument("Training completion best_generation requires best_epoch/best_score.");
        }
        const CheckpointState best = readCheckpointState(completion.bestGeneration.value());
        if (best.completedEpoch != completion.bestEpoch.value() ||
            best.phaseStartCompletedEpoch != completion.phaseStartCompletedEpoch ||
            best.phaseTargetCompletedEpoch != completion.phaseTargetCompletedEpoch) {
            throw std::invalid_argument("Training completion best_generation does not match best_epoch.");
        }
    }
}

void TrainingCheckpointRepository::writeCompletionStateAtomically(
    const std::filesystem::path& destination,
    const CompletionState& completion) {
    validateCompletionState(completion);
    Thor::FilesystemDurability::createDirectoriesDurably(rootDirectory);

    std::ostringstream temporaryName;
    temporaryName << "." << destination.filename().string() << ".tmp." << ::getpid() << "." << localSequence++;
    const std::filesystem::path temporaryPath = rootDirectory / temporaryName.str();
    removePathIfExists(temporaryPath);

    nlohmann::json metadata;
    metadata["schema_version"] = completion.schemaVersion;
    metadata["selected_generation"] = completion.selectedGeneration;
    metadata["selected_artifact_kind"] = completion.selectedArtifactKind;
    metadata["latest_generation"] = completion.latestGeneration.has_value()
        ? nlohmann::json(completion.latestGeneration.value()) : nlohmann::json(nullptr);
    metadata["best_generation"] = completion.bestGeneration.has_value()
        ? nlohmann::json(completion.bestGeneration.value()) : nlohmann::json(nullptr);
    metadata["network_name"] = completion.networkName;
    metadata["phase_start_completed_epoch"] = completion.phaseStartCompletedEpoch;
    metadata["requested_phase_epochs"] = completion.requestedPhaseEpochs;
    metadata["phase_target_completed_epoch"] = completion.phaseTargetCompletedEpoch;
    metadata["completed_epoch"] = completion.completedEpoch;
    metadata["selected_epoch"] = completion.selectedEpoch;
    metadata["completion_reason"] = completion.completionReason;
    metadata["best_epoch"] = completion.bestEpoch.has_value()
        ? nlohmann::json(completion.bestEpoch.value()) : nlohmann::json(nullptr);
    metadata["best_score"] = completion.bestScore.has_value()
        ? nlohmann::json(completion.bestScore.value()) : nlohmann::json(nullptr);
    metadata["latest_score"] = completion.latestScore.has_value()
        ? nlohmann::json(completion.latestScore.value()) : nlohmann::json(nullptr);
    metadata["latest_training_loss"] = completion.latestTrainingLoss.has_value()
        ? nlohmann::json(completion.latestTrainingLoss.value()) : nlohmann::json(nullptr);
    metadata["latest_validation_loss"] = completion.latestValidationLoss.has_value()
        ? nlohmann::json(completion.latestValidationLoss.value()) : nlohmann::json(nullptr);
    metadata["completed_training_elapsed_seconds"] = completion.completedTrainingElapsedSeconds;
    metadata["check_best_model_every_epochs"] = completion.checkBestModelEveryEpochs;
    metadata["first_model_selection_epoch"] = completion.firstModelSelectionEpoch;
    metadata["device_dataset_storage"] =
        deviceDatasetStorageReportToJson(completion.deviceDatasetStorageReport);

    try {
        {
            std::ofstream out(temporaryPath, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("Unable to open temporary training completion state for writing: '" +
                                         temporaryPath.string() + "'.");
            }
            out << metadata.dump(2) << '\n';
            out.flush();
            if (!out) {
                throw std::runtime_error("Failed while writing temporary training completion state: '" +
                                         temporaryPath.string() + "'.");
            }
        }
        Thor::FilesystemDurability::syncFile(temporaryPath);
        Thor::FilesystemDurability::durableRename(temporaryPath, destination);
    } catch (...) {
        removePathIfExists(temporaryPath);
        throw;
    }
}

std::optional<TrainingCheckpointRepository::CompletionState>
TrainingCheckpointRepository::readCompletionStateFile(const std::filesystem::path& path) const {
    std::error_code errorCode;
    const bool exists = std::filesystem::exists(path, errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to inspect training completion state '" + path.string() + "': " +
                                 errorCode.message());
    }
    if (!exists) return std::nullopt;

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Unable to open training completion state: '" + path.string() + "'.");
    }
    nlohmann::json metadata;
    try {
        in >> metadata;
    } catch (const std::exception& e) {
        throw std::runtime_error("Unable to parse training completion state '" + path.string() + "': " + e.what());
    }

    auto requiredUint64 = [&](const char* key) -> uint64_t {
        if (!metadata.contains(key) || !metadata.at(key).is_number_unsigned()) {
            throw std::runtime_error("Training completion state '" + path.string() +
                                     "' requires unsigned integer field '" + key + "'.");
        }
        return metadata.at(key).get<uint64_t>();
    };
    auto requiredUint32 = [&](const char* key) -> uint32_t {
        const uint64_t value = requiredUint64(key);
        if (value > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Training completion field '" + std::string(key) +
                                     "' exceeds uint32 range in '" + path.string() + "'.");
        }
        return static_cast<uint32_t>(value);
    };
    auto requiredString = [&](const char* key) -> std::string {
        if (!metadata.contains(key) || !metadata.at(key).is_string()) {
            throw std::runtime_error("Training completion state '" + path.string() +
                                     "' requires string field '" + key + "'.");
        }
        return metadata.at(key).get<std::string>();
    };
    auto optionalString = [&](const char* key) -> std::optional<std::string> {
        if (!metadata.contains(key) || metadata.at(key).is_null()) return std::nullopt;
        if (!metadata.at(key).is_string()) {
            throw std::runtime_error("Training completion field '" + std::string(key) +
                                     "' must be a string or null in '" + path.string() + "'.");
        }
        return metadata.at(key).get<std::string>();
    };
    auto optionalUint64 = [&](const char* key) -> std::optional<uint64_t> {
        if (!metadata.contains(key) || metadata.at(key).is_null()) return std::nullopt;
        if (!metadata.at(key).is_number_unsigned()) {
            throw std::runtime_error("Training completion field '" + std::string(key) +
                                     "' must be an unsigned integer or null in '" + path.string() + "'.");
        }
        return metadata.at(key).get<uint64_t>();
    };
    auto optionalDouble = [&](const char* key) -> std::optional<double> {
        if (!metadata.contains(key) || metadata.at(key).is_null()) return std::nullopt;
        if (!metadata.at(key).is_number()) {
            throw std::runtime_error("Training completion field '" + std::string(key) +
                                     "' must be numeric or null in '" + path.string() + "'.");
        }
        const double value = metadata.at(key).get<double>();
        if (!std::isfinite(value)) {
            throw std::runtime_error("Training completion field '" + std::string(key) +
                                     "' must be finite in '" + path.string() + "'.");
        }
        return value;
    };

    CompletionState completion;
    completion.schemaVersion = requiredUint32("schema_version");
    completion.selectedGeneration = requiredString("selected_generation");
    completion.selectedArtifactKind = requiredString("selected_artifact_kind");
    completion.latestGeneration = optionalString("latest_generation");
    completion.bestGeneration = optionalString("best_generation");
    completion.networkName = requiredString("network_name");
    completion.phaseStartCompletedEpoch = requiredUint64("phase_start_completed_epoch");
    completion.requestedPhaseEpochs = requiredUint32("requested_phase_epochs");
    completion.phaseTargetCompletedEpoch = requiredUint64("phase_target_completed_epoch");
    completion.completedEpoch = requiredUint64("completed_epoch");
    completion.selectedEpoch = requiredUint64("selected_epoch");
    completion.completionReason = requiredString("completion_reason");
    completion.bestEpoch = optionalUint64("best_epoch");
    completion.bestScore = optionalDouble("best_score");
    completion.latestScore = optionalDouble("latest_score");
    completion.latestTrainingLoss = optionalDouble("latest_training_loss");
    completion.latestValidationLoss = optionalDouble("latest_validation_loss");
    if (!metadata.contains("completed_training_elapsed_seconds") ||
        !metadata.at("completed_training_elapsed_seconds").is_number()) {
        throw std::runtime_error("Training completion state '" + path.string() +
                                 "' requires numeric field 'completed_training_elapsed_seconds'.");
    }
    completion.completedTrainingElapsedSeconds =
        metadata.at("completed_training_elapsed_seconds").get<double>();
    completion.checkBestModelEveryEpochs = requiredUint32("check_best_model_every_epochs");
    completion.firstModelSelectionEpoch = requiredUint64("first_model_selection_epoch");
    if (!metadata.contains("device_dataset_storage")) {
        throw std::runtime_error("Training completion state '" + path.string() +
                                 "' requires object field 'device_dataset_storage'.");
    }
    completion.deviceDatasetStorageReport = deviceDatasetStorageReportFromJson(
        metadata.at("device_dataset_storage"),
        "Training completion state '" + path.string() + "'");

    try {
        validateCompletionState(completion);
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid training completion state '" + path.string() + "': " + e.what());
    }
    return completion;
}

void TrainingCheckpointRepository::prepareCompletion(const CompletionState& completion) {
    writeCompletionStateAtomically(getCompletingPath(), completion);
}

void TrainingCheckpointRepository::commitCompletion() {
    const std::optional<CompletionState> completing = readCompleting();
    if (!completing.has_value()) {
        throw std::runtime_error("Cannot commit training completion without durable COMPLETING state.");
    }
    Thor::FilesystemDurability::durableRename(getCompletingPath(), getCompletedPath());
}

void TrainingCheckpointRepository::publishCompleted(const CompletionState& completion) {
    prepareCompletion(completion);
    commitCompletion();
}

std::optional<TrainingCheckpointRepository::CompletionState> TrainingCheckpointRepository::readCompleting() const {
    return readCompletionStateFile(getCompletingPath());
}

std::optional<TrainingCheckpointRepository::CompletionState> TrainingCheckpointRepository::readCompleted() const {
    return readCompletionStateFile(getCompletedPath());
}

void TrainingCheckpointRepository::compactCompletedRepository() {
    const std::optional<CompletionState> completion = readCompleted();
    if (!completion.has_value()) {
        throw std::runtime_error("Cannot compact a training checkpoint repository before COMPLETED is durable.");
    }

    // The final save_model_dir already contains any non-selected latest/best
    // copy needed for inspection. Shrink the terminal repository record to the
    // one generation required for idempotent stage handoff before deleting any
    // auxiliary generation it previously named.
    CompletionState compacted = completion.value();
    if (compacted.selectedArtifactKind == "best") {
        compacted.latestGeneration.reset();
        compacted.bestGeneration = compacted.selectedGeneration;
    } else {
        compacted.latestGeneration = compacted.selectedGeneration;
        compacted.bestGeneration.reset();
    }
    writeCompletionStateAtomically(getCompletedPath(), compacted);

    Thor::FilesystemDurability::durableRemoveAll(getPointerPath(Pointer::CURRENT));
    Thor::FilesystemDurability::durableRemoveAll(getPointerPath(Pointer::BEST));

    const std::filesystem::path generations = getGenerationsDirectory();
    std::error_code errorCode;
    if (!std::filesystem::exists(generations, errorCode)) {
        if (errorCode) {
            throw std::runtime_error("Failed to inspect training checkpoint generations during completion compaction: " +
                                     errorCode.message());
        }
        return;
    }

    for (const auto& entry : std::filesystem::directory_iterator(generations)) {
        if (entry.path().filename() == completion->selectedGeneration) {
            continue;
        }
        Thor::FilesystemDurability::durableRemoveAll(entry.path());
    }
}

void TrainingCheckpointRepository::removePointerIfMatches(Pointer pointer, const std::string& generationName) {
    const std::optional<std::string> value = readPointer(pointer);
    if (!value.has_value() || value.value() != generationName) {
        return;
    }
    Thor::FilesystemDurability::durableRemoveAll(getPointerPath(pointer));
}

void TrainingCheckpointRepository::retireGenerationIfUnreferenced(const std::string& generationName) {
    validateGenerationName(generationName);
    const std::optional<std::string> current = readPointer(Pointer::CURRENT);
    if (current.has_value() && current.value() == generationName) {
        return;
    }
    const std::optional<std::string> best = readPointer(Pointer::BEST);
    if (best.has_value() && best.value() == generationName) {
        return;
    }
    const std::optional<CompletionState> completing = readCompleting();
    if (completing.has_value() && completing->selectedGeneration == generationName) {
        return;
    }
    const std::optional<CompletionState> completed = readCompleted();
    if (completed.has_value() && completed->selectedGeneration == generationName) {
        return;
    }
    Thor::FilesystemDurability::durableRemoveAll(getGenerationDirectory(generationName));
}

void TrainingCheckpointRepository::syncCopiedTree(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> directories;
    directories.push_back(root);

    std::error_code errorCode;
    std::filesystem::recursive_directory_iterator iterator(root, errorCode);
    const std::filesystem::recursive_directory_iterator end;
    while (!errorCode && iterator != end) {
        const std::filesystem::directory_entry& entry = *iterator;
        std::error_code entryError;
        if (entry.is_regular_file(entryError) && !entryError) {
            Thor::FilesystemDurability::syncFile(entry.path());
        } else if (entry.is_directory(entryError) && !entryError) {
            directories.push_back(entry.path());
        } else if (entryError) {
            throw std::runtime_error("Failed to inspect copied checkpoint path '" + entry.path().string() + "': " +
                                     entryError.message());
        }
        iterator.increment(errorCode);
    }
    if (errorCode) {
        throw std::runtime_error("Failed while traversing copied training checkpoint tree '" + root.string() + "': " +
                                 errorCode.message());
    }

    std::sort(directories.begin(), directories.end(), [](const auto& lhs, const auto& rhs) {
        return pathDepth(lhs) > pathDepth(rhs);
    });
    for (const std::filesystem::path& directory : directories) {
        Thor::FilesystemDurability::syncDirectory(directory);
    }
    Thor::FilesystemDurability::syncParentDirectory(root);
}

void TrainingCheckpointRepository::copyGenerationDurably(const std::string& generationName,
                                                          const std::filesystem::path& destination) const {
    const std::filesystem::path source = getGenerationDirectory(generationName);
    std::error_code errorCode;
    if (!std::filesystem::is_directory(source, errorCode) || errorCode) {
        throw std::runtime_error("Cannot copy missing training checkpoint generation '" + source.string() + "'.");
    }
    if (std::filesystem::exists(destination, errorCode) || errorCode) {
        if (errorCode) {
            throw std::runtime_error("Failed to inspect training checkpoint copy destination '" + destination.string() +
                                     "': " + errorCode.message());
        }
        throw std::runtime_error("Training checkpoint copy destination already exists: '" + destination.string() + "'.");
    }

    Thor::FilesystemDurability::createDirectoriesDurably(destination.parent_path());
    try {
        std::filesystem::copy(source, destination, std::filesystem::copy_options::recursive, errorCode);
        if (errorCode) {
            throw std::runtime_error("Failed to copy training checkpoint generation from '" + source.string() + "' to '" +
                                     destination.string() + "': " + errorCode.message());
        }
        syncCopiedTree(destination);
    } catch (...) {
        std::error_code cleanupError;
        std::filesystem::remove_all(destination, cleanupError);
        throw;
    }
}

void TrainingCheckpointRepository::removeRepositoryDurably() {
    Thor::FilesystemDurability::durableRemoveAll(rootDirectory);
}

void TrainingCheckpointRepository::removePathIfExists(const std::filesystem::path& path) {
    std::error_code errorCode;
    const bool exists = std::filesystem::exists(path, errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to inspect path '" + path.string() + "': " + errorCode.message());
    }
    if (!exists) {
        return;
    }
    std::filesystem::remove_all(path, errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to remove path '" + path.string() + "': " + errorCode.message());
    }
}

}  // namespace ThorImplementation
