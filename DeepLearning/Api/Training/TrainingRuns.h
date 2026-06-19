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
                          std::optional<size_t> maxParallelRuns = std::nullopt);

    [[nodiscard]] TrainingRunsResult fit(uint32_t epochs);
    [[nodiscard]] TrainingRunsResult fit(const TrainerFitOptions& options);

    [[nodiscard]] const std::vector<TrainingRunsSpec>& getRuns() const { return runs; }
    [[nodiscard]] TrainingRunsFailurePolicy getFailurePolicy() const { return failurePolicy; }
    [[nodiscard]] double getMaxSummaryLogsPerSecond() const { return maxSummaryLogsPerSecond; }
    [[nodiscard]] std::optional<size_t> getMaxParallelRuns() const { return maxParallelRuns; }
    [[nodiscard]] size_t getEffectiveMaxParallelRuns() const;

   private:
    void validateRunSpecs() const;
    void validateFitOptions(const TrainerFitOptions& options) const;
    [[nodiscard]] std::vector<TrainingEnsembleResult> buildEnsembleResults(const std::vector<TrainingRunResult>& results) const;

    std::vector<TrainingRunsSpec> runs{};
    TrainingRunsFailurePolicy failurePolicy = TrainingRunsFailurePolicy::CANCEL_SIBLINGS;
    double maxSummaryLogsPerSecond = 2.0;
    std::optional<size_t> maxParallelRuns{};
};

}  // namespace Thor
