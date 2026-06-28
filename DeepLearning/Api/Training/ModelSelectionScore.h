#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace Thor {

struct TrainingModelSelectionPhaseStats {
    std::optional<double> loss{};
    std::unordered_map<std::string, double> losses{};
    std::unordered_map<std::string, double> metrics{};
};

struct TrainingModelSelectionContext {
    uint64_t epoch = 0;
    TrainingModelSelectionPhaseStats train{};
    TrainingModelSelectionPhaseStats validate{};
    TrainingModelSelectionPhaseStats test{};

    [[nodiscard]] std::optional<double> trainingLoss() const { return train.loss; }
    [[nodiscard]] std::optional<double> validationLoss() const { return validate.loss; }
};

struct TrainingModelSelectionScore {
    using ContextScoreFunction = std::function<std::optional<double>(const TrainingModelSelectionContext& context)>;
    using LegacyScoreFunction = std::function<std::optional<double>(std::optional<double> validationLoss,
                                                                   std::optional<double> trainingLoss,
                                                                   uint64_t epoch)>;

    ContextScoreFunction scoreFunction{};

    TrainingModelSelectionScore() = default;
    explicit TrainingModelSelectionScore(ContextScoreFunction scoreFunction) : scoreFunction(std::move(scoreFunction)) {}
    explicit TrainingModelSelectionScore(LegacyScoreFunction scoreFunction) {
        this->scoreFunction = [scoreFunction = std::move(scoreFunction)](const TrainingModelSelectionContext& context) {
            return scoreFunction(context.validationLoss(), context.trainingLoss(), context.epoch);
        };
    }

    [[nodiscard]] static std::optional<double> defaultScore(const TrainingModelSelectionContext& context) {
        if (context.validate.loss.has_value()) {
            return context.validate.loss;
        }
        return context.train.loss;
    }

    [[nodiscard]] static std::optional<double> defaultScore(std::optional<double> validationLoss,
                                                            std::optional<double> trainingLoss,
                                                            uint64_t epoch) {
        TrainingModelSelectionContext context;
        context.epoch = epoch;
        context.validate.loss = validationLoss;
        context.train.loss = trainingLoss;
        return defaultScore(context);
    }

    [[nodiscard]] std::optional<double> evaluate(const TrainingModelSelectionContext& context) const {
        if (!scoreFunction) {
            return defaultScore(context);
        }
        return scoreFunction(context);
    }

    [[nodiscard]] std::optional<double> evaluate(std::optional<double> validationLoss,
                                                 std::optional<double> trainingLoss,
                                                 uint64_t epoch) const {
        TrainingModelSelectionContext context;
        context.epoch = epoch;
        context.validate.loss = validationLoss;
        context.train.loss = trainingLoss;
        return evaluate(context);
    }

    [[nodiscard]] bool isCustom() const { return static_cast<bool>(scoreFunction); }
};

}  // namespace Thor
