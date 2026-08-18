#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ThorTest {

enum class RaggedInactivePoison {
    PositiveFinite,
    NegativeFinite,
    NaN,
};

inline float raggedInactivePoisonValue(RaggedInactivePoison poison) {
    switch (poison) {
        case RaggedInactivePoison::PositiveFinite:
            return 4096.0f;
        case RaggedInactivePoison::NegativeFinite:
            return -4096.0f;
        case RaggedInactivePoison::NaN:
            return std::numeric_limits<float>::quiet_NaN();
    }
    throw std::invalid_argument("Unknown RaggedInactivePoison value.");
}

inline void poisonInactiveElements(float* values,
                                   uint64_t activeElements,
                                   uint64_t totalElements,
                                   RaggedInactivePoison poison) {
    if (activeElements > totalElements) {
        throw std::invalid_argument("Ragged inactive poison active extent exceeds capacity.");
    }
    std::fill(values + activeElements, values + totalElements, raggedInactivePoisonValue(poison));
}

inline void poisonInactiveElements(std::vector<float>& values,
                                   uint64_t activeElements,
                                   RaggedInactivePoison poison) {
    poisonInactiveElements(values.data(), activeElements, values.size(), poison);
}

inline void poisonInactiveRows(std::vector<float>& values,
                               uint64_t activeRows,
                               uint64_t elementsPerValue,
                               RaggedInactivePoison poison) {
    if (elementsPerValue == 0) {
        throw std::invalid_argument("Ragged inactive poison requires a nonzero row width.");
    }
    if (values.size() % elementsPerValue != 0) {
        throw std::invalid_argument("Ragged inactive poison values size is not an integral number of rows.");
    }
    poisonInactiveElements(values, activeRows * elementsPerValue, poison);
}

inline std::vector<float> logicalActivePrefix(const std::vector<float>& values,
                                              uint64_t activeRows,
                                              uint64_t elementsPerValue) {
    const uint64_t activeElements = activeRows * elementsPerValue;
    if (activeElements > values.size()) {
        throw std::invalid_argument("Ragged active prefix exceeds physical capacity.");
    }
    return std::vector<float>(values.begin(), values.begin() + activeElements);
}

}  // namespace ThorTest
