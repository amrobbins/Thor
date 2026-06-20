#pragma once

#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"
#include "DeepLearning/Api/Training/Events/TrainingStatsSnapshot.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <new>
#include <optional>
#include <map>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace Thor {

enum class TrainingRunStatus { NOT_STARTED, RUNNING, COMPLETED, FAILED, CANCELLED, INTERRUPTED, OUT_OF_MEMORY };

[[nodiscard]] inline const char* trainingRunStatusName(TrainingRunStatus status) {
    switch (status) {
        case TrainingRunStatus::NOT_STARTED:
            return "not_started";
        case TrainingRunStatus::RUNNING:
            return "running";
        case TrainingRunStatus::COMPLETED:
            return "completed";
        case TrainingRunStatus::FAILED:
            return "failed";
        case TrainingRunStatus::CANCELLED:
            return "cancelled";
        case TrainingRunStatus::INTERRUPTED:
            return "interrupted";
        case TrainingRunStatus::OUT_OF_MEMORY:
            return "oom";
        default:
            return "unknown";
    }
}

struct TrainingRunExceptionSummary {
    std::string type{};
    std::string message{};
};

namespace training_run_result_detail {

[[nodiscard]] inline std::string lowercaseAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

[[nodiscard]] inline bool messageLooksLikeOutOfMemory(const std::string& message) {
    const std::string lower = lowercaseAscii(message);
    return lower.find("out of memory") != std::string::npos || lower.find("gpu out of memory") != std::string::npos ||
           lower.find("cudaerrormemoryallocation") != std::string::npos || lower.find("cuda_error_out_of_memory") != std::string::npos ||
           lower.find("cublas_status_alloc_failed") != std::string::npos || lower.find("cusparse_status_alloc_failed") != std::string::npos ||
           lower.find("cudnn_status_alloc_failed") != std::string::npos;
}

}  // namespace training_run_result_detail

[[nodiscard]] inline TrainingRunExceptionSummary summarizeTrainingRunException(std::exception_ptr exception) {
    if (exception == nullptr) {
        return {};
    }

    try {
        std::rethrow_exception(exception);
    } catch (const TrainingCancelled& e) {
        return {"TrainingCancelled", e.what()};
    } catch (const TrainingInterrupted& e) {
        return {"TrainingInterrupted", e.what()};
    } catch (const std::bad_alloc& e) {
        return {"std::bad_alloc", e.what()};
    } catch (const std::exception& e) {
        return {typeid(e).name(), e.what()};
    } catch (...) {
        return {"unknown", "Non-standard exception."};
    }
}

[[nodiscard]] inline TrainingRunStatus classifyTrainingRunException(std::exception_ptr exception) {
    if (exception == nullptr) {
        return TrainingRunStatus::COMPLETED;
    }

    try {
        std::rethrow_exception(exception);
    } catch (const TrainingCancelled&) {
        return TrainingRunStatus::CANCELLED;
    } catch (const TrainingInterrupted&) {
        return TrainingRunStatus::INTERRUPTED;
    } catch (const std::bad_alloc&) {
        return TrainingRunStatus::OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return training_run_result_detail::messageLooksLikeOutOfMemory(e.what()) ? TrainingRunStatus::OUT_OF_MEMORY : TrainingRunStatus::FAILED;
    } catch (...) {
        return TrainingRunStatus::FAILED;
    }
}

struct TrainingRunResult {
    std::string runName{};
    std::optional<std::string> ensembleGroup{};
    double ensembleWeight = 1.0;
    TrainingRunStatus status = TrainingRunStatus::NOT_STARTED;
    std::optional<TrainingStatsSnapshot> finalTrainingStats{};
    std::optional<TrainingStatsSnapshot> finalValidationStats{};
    std::optional<TrainingStatsSnapshot> finalTestStats{};
    TrainingRunExceptionSummary exception{};

    [[nodiscard]] bool completed() const { return status == TrainingRunStatus::COMPLETED; }
    [[nodiscard]] bool failed() const {
        return status == TrainingRunStatus::FAILED || status == TrainingRunStatus::OUT_OF_MEMORY || status == TrainingRunStatus::INTERRUPTED;
    }
    [[nodiscard]] bool cancelled() const { return status == TrainingRunStatus::CANCELLED; }

    [[nodiscard]] const std::optional<TrainingStatsSnapshot>& finalStatsForPhase(TrainingEventPhase phase) const {
        switch (phase) {
            case TrainingEventPhase::TRAIN:
                return finalTrainingStats;
            case TrainingEventPhase::VALIDATE:
                return finalValidationStats;
            case TrainingEventPhase::TEST:
                return finalTestStats;
            case TrainingEventPhase::UNKNOWN:
            default:
                return finalTrainingStats;
        }
    }

    [[nodiscard]] std::optional<double> finalLossForPhase(TrainingEventPhase phase) const {
        const std::optional<TrainingStatsSnapshot>& stats = finalStatsForPhase(phase);
        if (!stats.has_value()) {
            return std::nullopt;
        }
        return stats->loss;
    }

    [[nodiscard]] std::optional<double> finalAccuracyForPhase(TrainingEventPhase phase) const {
        const std::optional<TrainingStatsSnapshot>& stats = finalStatsForPhase(phase);
        if (!stats.has_value()) {
            return std::nullopt;
        }
        return stats->accuracy;
    }

    [[nodiscard]] static TrainingRunResult completedResult(std::string runName,
                                                           std::optional<TrainingStatsSnapshot> finalTrainingStats = {},
                                                           std::optional<TrainingStatsSnapshot> finalValidationStats = {},
                                                           std::optional<TrainingStatsSnapshot> finalTestStats = {}) {
        TrainingRunResult result;
        result.runName = std::move(runName);
        result.status = TrainingRunStatus::COMPLETED;
        result.finalTrainingStats = std::move(finalTrainingStats);
        result.finalValidationStats = std::move(finalValidationStats);
        result.finalTestStats = std::move(finalTestStats);
        return result;
    }

    [[nodiscard]] static TrainingRunResult fromException(std::string runName,
                                                         std::exception_ptr exception,
                                                         std::optional<TrainingStatsSnapshot> finalTrainingStats = {},
                                                         std::optional<TrainingStatsSnapshot> finalValidationStats = {},
                                                         std::optional<TrainingStatsSnapshot> finalTestStats = {}) {
        TrainingRunResult result;
        result.runName = std::move(runName);
        result.status = classifyTrainingRunException(exception);
        result.finalTrainingStats = std::move(finalTrainingStats);
        result.finalValidationStats = std::move(finalValidationStats);
        result.finalTestStats = std::move(finalTestStats);
        result.exception = summarizeTrainingRunException(exception);
        return result;
    }
};

struct TrainingRunInputSignature {
    std::string inputName{};
    std::vector<uint64_t> dimensions{};
    std::string dataType{};
    bool dimensionsIncludeBatch = false;

    [[nodiscard]] bool compatibleWith(const TrainingRunInputSignature& other) const {
        return inputName == other.inputName && dimensions == other.dimensions && dataType == other.dataType &&
               dimensionsIncludeBatch == other.dimensionsIncludeBatch;
    }
};

struct TrainingRunOutputSignature {
    std::string outputName{};
    std::vector<uint64_t> dimensions{};
    std::string dataType{};

    [[nodiscard]] bool compatibleWith(const TrainingRunOutputSignature& other) const {
        return outputName == other.outputName && dimensions == other.dimensions && dataType == other.dataType;
    }
};

struct TrainingEnsembleMemberResult {
    std::string runName{};
    double weight = 1.0;
    TrainingRunStatus status = TrainingRunStatus::NOT_STARTED;
    std::optional<double> finalTrainingLoss{};
    std::optional<double> finalValidationLoss{};
    std::optional<double> finalTestLoss{};
    std::optional<double> finalTestAccuracy{};
};

struct TrainingEnsembleResult {
    std::string ensembleGroup{};
    std::vector<TrainingEnsembleMemberResult> members{};
    std::vector<TrainingRunInputSignature> inputSignature{};
    std::vector<TrainingRunOutputSignature> outputSignature{};
    std::optional<double> ensembleTrainingLoss{};
    std::optional<double> ensembleTestLoss{};
    std::optional<double> ensembleTestAccuracy{};

    [[nodiscard]] size_t size() const { return members.size(); }
    [[nodiscard]] bool empty() const { return members.empty(); }
    [[nodiscard]] bool allCompleted() const;
    [[nodiscard]] bool anyFailed() const;
    [[nodiscard]] double totalWeight() const;
    [[nodiscard]] std::map<std::string, size_t> statusCounts() const;
    [[nodiscard]] std::optional<double> ensembleFinalTrainingLoss() const { return ensembleTrainingLoss; }
    [[nodiscard]] std::optional<double> ensembleFinalTestLoss() const { return ensembleTestLoss; }
    [[nodiscard]] std::optional<double> ensembleFinalTestAccuracy() const { return ensembleTestAccuracy; }
};

}  // namespace Thor
