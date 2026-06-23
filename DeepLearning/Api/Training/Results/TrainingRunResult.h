#pragma once

#include "DeepLearning/Api/Training/Cancellation/TrainingCancellation.h"
#include "DeepLearning/Api/Training/Events/TrainingStatsSnapshot.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
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

enum class TrainingRunCompletionReason { COMPLETED, EARLY_COMPLETED };

[[nodiscard]] inline const char* trainingRunCompletionReasonName(TrainingRunCompletionReason reason) {
    switch (reason) {
        case TrainingRunCompletionReason::COMPLETED:
            return "completed";
        case TrainingRunCompletionReason::EARLY_COMPLETED:
            return "early_completed";
        default:
            return "unknown";
    }
}

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
    TrainingRunCompletionReason completionReason = TrainingRunCompletionReason::COMPLETED;
    std::optional<uint64_t> completedEpoch{};
    std::optional<uint64_t> bestEpoch{};
    std::optional<double> bestScore{};
    std::optional<std::string> savedModelDirectory{};
    std::optional<TrainingStatsSnapshot> finalTrainingStats{};
    std::optional<TrainingStatsSnapshot> finalValidationStats{};
    std::optional<TrainingStatsSnapshot> finalTestStats{};
    TrainingRunExceptionSummary exception{};

    [[nodiscard]] bool completed() const { return status == TrainingRunStatus::COMPLETED; }
    [[nodiscard]] bool earlyCompleted() const {
        return status == TrainingRunStatus::COMPLETED && completionReason == TrainingRunCompletionReason::EARLY_COMPLETED;
    }
    [[nodiscard]] const char* resultName() const {
        if (status == TrainingRunStatus::COMPLETED) {
            return trainingRunCompletionReasonName(completionReason);
        }
        return trainingRunStatusName(status);
    }
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
                                                           std::optional<TrainingStatsSnapshot> finalTestStats = {},
                                                           TrainingRunCompletionReason completionReason = TrainingRunCompletionReason::COMPLETED,
                                                           std::optional<uint64_t> completedEpoch = {},
                                                           std::optional<uint64_t> bestEpoch = {},
                                                           std::optional<double> bestScore = {},
                                                           std::optional<std::string> savedModelDirectory = {}) {
        TrainingRunResult result;
        result.runName = std::move(runName);
        result.status = TrainingRunStatus::COMPLETED;
        result.completionReason = completionReason;
        result.completedEpoch = completedEpoch;
        result.bestEpoch = bestEpoch;
        result.bestScore = bestScore;
        result.savedModelDirectory = std::move(savedModelDirectory);
        result.finalTrainingStats = std::move(finalTrainingStats);
        result.finalValidationStats = std::move(finalValidationStats);
        result.finalTestStats = std::move(finalTestStats);
        return result;
    }

    [[nodiscard]] static TrainingRunResult fromException(std::string runName,
                                                         std::exception_ptr exception,
                                                         std::optional<TrainingStatsSnapshot> finalTrainingStats = {},
                                                         std::optional<TrainingStatsSnapshot> finalValidationStats = {},
                                                         std::optional<TrainingStatsSnapshot> finalTestStats = {},
                                                         std::optional<std::string> savedModelDirectory = {}) {
        TrainingRunResult result;
        result.runName = std::move(runName);
        result.status = classifyTrainingRunException(exception);
        result.finalTrainingStats = std::move(finalTrainingStats);
        result.finalValidationStats = std::move(finalValidationStats);
        result.finalTestStats = std::move(finalTestStats);
        result.savedModelDirectory = std::move(savedModelDirectory);
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

struct TrainingNamedMetricResult {
    std::string name{};
    std::string outputName{};
    std::string targetInputName{};
    double overallWeight = 1.0;
    std::string overallWeightSource{};
    std::optional<double> trainValue{};
    std::optional<double> testValue{};

    [[nodiscard]] bool hasValue() const { return trainValue.has_value() || testValue.has_value(); }
};

struct TrainingEnsembleResult {
    std::string ensembleGroup{};
    std::vector<TrainingEnsembleMemberResult> members{};
    std::vector<TrainingRunInputSignature> inputSignature{};
    std::vector<TrainingRunOutputSignature> outputSignature{};
    std::optional<double> ensembleTrainingLoss{};
    std::optional<double> ensembleTestLoss{};
    std::optional<double> ensembleTestAccuracy{};
    std::vector<TrainingNamedMetricResult> namedMetrics{};
    size_t minSuccessfulModels = 0;

    [[nodiscard]] size_t size() const { return members.size(); }
    [[nodiscard]] bool empty() const { return members.empty(); }
    [[nodiscard]] bool allCompleted() const;
    [[nodiscard]] bool anyFailed() const;
    [[nodiscard]] size_t successfulModels() const;
    [[nodiscard]] size_t requiredSuccessfulModels() const { return minSuccessfulModels == 0 ? members.size() : minSuccessfulModels; }
    [[nodiscard]] bool hasEnoughSuccessfulModels() const;
    [[nodiscard]] double totalWeight() const;
    [[nodiscard]] std::map<std::string, size_t> statusCounts() const;
    [[nodiscard]] bool hasNamedMetricValues() const {
        return std::any_of(namedMetrics.begin(), namedMetrics.end(), [](const TrainingNamedMetricResult& metricResult) {
            return metricResult.hasValue();
        });
    }
    [[nodiscard]] bool hasEnsembleEvaluationMetrics() const {
        return ensembleTrainingLoss.has_value() || ensembleTestLoss.has_value() || ensembleTestAccuracy.has_value() || hasNamedMetricValues();
    }
    [[nodiscard]] std::optional<double> ensembleFinalTrainingLoss() const { return ensembleTrainingLoss; }
    [[nodiscard]] std::optional<double> ensembleFinalTestLoss() const { return ensembleTestLoss; }
    [[nodiscard]] std::optional<double> ensembleFinalTestAccuracy() const { return ensembleTestAccuracy; }
};

}  // namespace Thor
