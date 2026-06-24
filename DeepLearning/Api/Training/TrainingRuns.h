#pragma once

#include "DeepLearning/Api/Training/Results/TrainingRunResult.h"
#include "DeepLearning/Api/Training/Trainer.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Thor {

enum class TrainingRunsFailurePolicy { CONTINUE, CANCEL_SIBLINGS };

[[nodiscard]] const char* trainingRunsFailurePolicyName(TrainingRunsFailurePolicy policy);


struct TrainingRunsEvaluationOptions {
    // Ordinary held-out test loader for post-fit evaluation. For grouped runs,
    // callers are expected to supply the same logical test set they would attach
    // to every member; grouped evaluation treats that test population as shared.
    std::shared_ptr<Loader> testLoader = nullptr;

    // Diagnostic post-fit pass over each member's validation population. This
    // is reported as the grouped training-population loss, not as held-out test
    // performance.
    bool evaluateTrainingPopulation = true;
};

struct TrainingRunsSpec {
    std::string runName{};
    std::shared_ptr<Trainer> trainer = nullptr;
    std::optional<std::string> ensembleGroup{};
    double ensembleWeight = 1.0;

    TrainingRunsSpec() = default;
    TrainingRunsSpec(std::string runName, std::shared_ptr<Trainer> trainer)
        : runName(std::move(runName)), trainer(std::move(trainer)) {}
    TrainingRunsSpec(std::string runName, std::shared_ptr<Trainer> trainer, std::string ensembleGroup, double ensembleWeight = 1.0)
        : runName(std::move(runName)),
          trainer(std::move(trainer)),
          ensembleGroup(std::move(ensembleGroup)),
          ensembleWeight(ensembleWeight) {}
};

class TrainingRunsResult {
   public:
    TrainingRunsResult() = default;
    explicit TrainingRunsResult(std::vector<TrainingRunResult> results, std::vector<TrainingEnsembleResult> ensembles = {})
        : results(std::move(results)), ensembles_(std::move(ensembles)) {}

    [[nodiscard]] const std::vector<TrainingRunResult>& runs() const { return results; }
    [[nodiscard]] const std::vector<TrainingEnsembleResult>& ensembles() const { return ensembles_; }
    [[nodiscard]] bool hasEnsembles() const { return !ensembles_.empty(); }
    [[nodiscard]] size_t size() const { return results.size(); }
    [[nodiscard]] bool empty() const { return results.empty(); }

    [[nodiscard]] bool allCompleted() const;
    [[nodiscard]] bool anyFailed() const;
    [[nodiscard]] bool anyCancelled() const;
    [[nodiscard]] std::map<std::string, size_t> statusCounts() const;

    [[nodiscard]] const TrainingRunResult& at(size_t index) const;
    [[nodiscard]] const TrainingRunResult& at(std::string_view runName) const;
    [[nodiscard]] const TrainingEnsembleResult& ensemble(std::string_view ensembleGroup) const;
    [[nodiscard]] std::string saveEnsemble(std::string_view ensembleGroup,
                                           const std::string& directory,
                                           std::string aggregation = "auto",
                                           bool overwrite = false) const;

    [[nodiscard]] const TrainingRunResult& operator[](size_t index) const { return at(index); }
    [[nodiscard]] const TrainingRunResult& operator[](std::string_view runName) const { return at(runName); }

   private:
    std::vector<TrainingRunResult> results{};
    std::vector<TrainingEnsembleResult> ensembles_{};
};

class TrainingRuns {
   public:
    explicit TrainingRuns(std::vector<TrainingRunsSpec> runs,
                          TrainingRunsFailurePolicy failurePolicy = TrainingRunsFailurePolicy::CANCEL_SIBLINGS,
                          double maxSummaryLogsPerSecond = 2.0,
                          std::optional<size_t> maxParallelRuns = std::nullopt,
                          std::vector<TrainingRunsRestartPolicy> restartConditions = {},
                          std::vector<TrainingRunsEarlyCompletionRule> earlyCompletionRules = {},
                          std::map<std::string, size_t> minSuccessfulModels = {},
                          std::map<std::string, std::vector<std::string>> reportedLosses = {},
                          std::map<std::string, std::vector<std::string>> reportedMetrics = {});

    [[nodiscard]] TrainingRunsResult fit(uint32_t epochs);
    [[nodiscard]] TrainingRunsResult fit(uint32_t epochs, std::shared_ptr<Loader> testLoader);
    [[nodiscard]] TrainingRunsResult fit(const TrainerFitOptions& options);
    [[nodiscard]] TrainingRunsResult fit(const TrainerFitOptions& options, const TrainingRunsEvaluationOptions& evaluationOptions);

    [[nodiscard]] const std::vector<TrainingRunsSpec>& getRuns() const { return runs; }
    [[nodiscard]] TrainingRunsFailurePolicy getFailurePolicy() const { return failurePolicy; }
    [[nodiscard]] double getMaxSummaryLogsPerSecond() const { return maxSummaryLogsPerSecond; }
    [[nodiscard]] std::optional<size_t> getMaxParallelRuns() const { return maxParallelRuns; }
    [[nodiscard]] const std::map<std::string, size_t>& getMinSuccessfulModels() const { return minSuccessfulModels; }
    [[nodiscard]] const std::vector<TrainingRunsRestartPolicy>& getRestartConditions() const { return restartConditions; }
    [[nodiscard]] const std::vector<TrainingRunsEarlyCompletionRule>& getEarlyCompletionRules() const { return earlyCompletionRules; }
    [[nodiscard]] const std::map<std::string, std::vector<std::string>>& getReportedLosses() const { return reportedLosses; }
    [[nodiscard]] const std::map<std::string, std::vector<std::string>>& getReportedMetrics() const { return reportedMetrics; }
    [[nodiscard]] size_t getEffectiveMaxParallelRuns() const;

   private:
    void validateRunSpecs() const;
    void validateMinSuccessfulModels() const;
    [[nodiscard]] bool failedRunShouldTriggerCancellation(size_t runIndex, const std::vector<TrainingRunResult>& results) const;
    [[nodiscard]] size_t minSuccessfulModelsForGroup(std::string_view ensembleGroup, size_t defaultValue) const;
    void validateRestartConditions() const;
    void validateEarlyCompletionRules() const;
    void validateReportedLosses() const;
    void validateReportedMetrics() const;
    [[nodiscard]] std::vector<TrainingRestartCondition> restartConditionsForRun(const TrainingRunsSpec& run) const;
    [[nodiscard]] std::vector<TrainingEarlyCompletionPolicy> earlyCompletionPoliciesForRun(const TrainingRunsSpec& run) const;
    [[nodiscard]] std::vector<TrainingNamedMetricResult> namedMetricResultsForGroup(std::string_view ensembleGroup) const;
    [[nodiscard]] std::vector<TrainingNamedMetricResult> namedGraphMetricResultsForGroup(std::string_view ensembleGroup) const;
    [[nodiscard]] std::vector<std::string> reportedMetricNamesForSpec(const TrainingRunsSpec& spec) const;
    [[nodiscard]] bool hasEnsembleGroups() const;
    void validateEnsembleArtifactsForFit(const TrainingRunsEvaluationOptions& evaluationOptions) const;
    void validateFitOptions(const TrainerFitOptions& options) const;
    void validateTestLoader(Loader& loader) const;
    void evaluateEnsembles(std::vector<TrainingRunResult>& results, std::map<std::string, TrainingEnsembleResult>& ensembleResultsByGroup) const;
    void evaluateEnsemblesOnTestLoader(std::vector<TrainingRunResult>& results,
                                       std::map<std::string, TrainingEnsembleResult>& ensembleResultsByGroup,
                                       std::shared_ptr<Loader> testLoader) const;
    [[nodiscard]] std::vector<TrainingEnsembleResult> buildEnsembleResults(const std::vector<TrainingRunResult>& results) const;
    [[nodiscard]] std::map<std::string, TrainingEnsembleResult> buildEnsembleResultsByGroup(const std::vector<TrainingRunResult>& results) const;

    std::vector<TrainingRunsSpec> runs{};
    TrainingRunsFailurePolicy failurePolicy = TrainingRunsFailurePolicy::CANCEL_SIBLINGS;
    double maxSummaryLogsPerSecond = 2.0;
    std::optional<size_t> maxParallelRuns{};
    std::map<std::string, size_t> minSuccessfulModels{};
    std::vector<TrainingRunsRestartPolicy> restartConditions{};
    std::vector<TrainingRunsEarlyCompletionRule> earlyCompletionRules{};
    std::map<std::string, std::vector<std::string>> reportedLosses{};
    std::map<std::string, std::vector<std::string>> reportedMetrics{};
};

}  // namespace Thor
