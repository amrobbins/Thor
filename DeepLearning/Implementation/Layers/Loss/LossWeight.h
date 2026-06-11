#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace ThorImplementation {

inline void validateLossWeight(float lossWeight) {
    if (!std::isfinite(lossWeight)) {
        throw std::invalid_argument("Loss weight must be finite.");
    }
}

inline std::optional<float> normalizeLossWeight(std::optional<float> lossWeight) {
    if (!lossWeight.has_value()) {
        return std::nullopt;
    }
    validateLossWeight(lossWeight.value());
    if (lossWeight.value() == 1.0f) {
        return std::nullopt;
    }
    return lossWeight;
}

inline std::optional<float> normalizeLossWeight(float lossWeight) { return normalizeLossWeight(std::optional<float>(lossWeight)); }

inline float materializeLossWeight(std::optional<float> lossWeight) { return lossWeight.value_or(1.0f); }

template <typename Json>
inline std::optional<float> lossWeightFromJson(const Json& j) {
    if (!j.contains("loss_weight") || j.at("loss_weight").is_null()) {
        return std::nullopt;
    }
    return normalizeLossWeight(j.at("loss_weight").template get<float>());
}

template <typename Json>
inline void addLossWeightToJson(Json& j, std::optional<float> lossWeight) {
    std::optional<float> normalized = normalizeLossWeight(lossWeight);
    if (normalized.has_value()) {
        j["loss_weight"] = normalized.value();
    }
}

}  // namespace ThorImplementation
