#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace Thor {

struct TrainingEarlyCompletionPolicy {
    using CompletionCondition = std::function<bool(double currentScore, double bestScore, uint64_t currentEpoch, uint64_t bestEpoch)>;

    CompletionCondition completionCondition{};

    TrainingEarlyCompletionPolicy() = default;
    explicit TrainingEarlyCompletionPolicy(CompletionCondition completionCondition)
        : completionCondition(std::move(completionCondition)) {}

    [[nodiscard]] bool shouldComplete(double currentScore, double bestScore, uint64_t currentEpoch, uint64_t bestEpoch) const {
        if (!completionCondition) {
            return false;
        }
        return completionCondition(currentScore, bestScore, currentEpoch, bestEpoch);
    }
};

struct TrainingRunsEarlyCompletionRule : public TrainingEarlyCompletionPolicy {
    std::optional<std::string> runName{};
    std::optional<std::string> ensembleGroup{};

    TrainingRunsEarlyCompletionRule() = default;
    explicit TrainingRunsEarlyCompletionRule(CompletionCondition completionCondition)
        : TrainingEarlyCompletionPolicy(std::move(completionCondition)) {}

    static TrainingRunsEarlyCompletionRule forRun(std::string runName, CompletionCondition completionCondition) {
        TrainingRunsEarlyCompletionRule rule(std::move(completionCondition));
        rule.runName = std::move(runName);
        return rule;
    }

    static TrainingRunsEarlyCompletionRule forEnsembleGroup(std::string ensembleGroup, CompletionCondition completionCondition) {
        TrainingRunsEarlyCompletionRule rule(std::move(completionCondition));
        rule.ensembleGroup = std::move(ensembleGroup);
        return rule;
    }

    [[nodiscard]] TrainingEarlyCompletionPolicy toEarlyCompletionPolicy() const {
        return TrainingEarlyCompletionPolicy{completionCondition};
    }
};

using TrainingRunsEarlyCompletionPolicy = TrainingRunsEarlyCompletionRule;

}  // namespace Thor
