#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

namespace Thor {

struct TrainingModelSelectionScore {
    using ScoreFunction = std::function<std::optional<double>(std::optional<double> validationLoss,
                                                              std::optional<double> trainingLoss,
                                                              uint64_t epoch)>;

    ScoreFunction scoreFunction{};

    TrainingModelSelectionScore() = default;
    explicit TrainingModelSelectionScore(ScoreFunction scoreFunction) : scoreFunction(std::move(scoreFunction)) {}

    [[nodiscard]] static std::optional<double> defaultScore(std::optional<double> validationLoss,
                                                            std::optional<double> trainingLoss,
                                                            uint64_t epoch) {
        (void)epoch;
        if (validationLoss.has_value()) {
            return validationLoss;
        }
        return trainingLoss;
    }

    [[nodiscard]] std::optional<double> evaluate(std::optional<double> validationLoss,
                                                 std::optional<double> trainingLoss,
                                                 uint64_t epoch) const {
        if (!scoreFunction) {
            return defaultScore(validationLoss, trainingLoss, epoch);
        }
        return scoreFunction(validationLoss, trainingLoss, epoch);
    }

    [[nodiscard]] bool isCustom() const { return static_cast<bool>(scoreFunction); }
};

}  // namespace Thor
