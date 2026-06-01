#pragma once

#include "Utilities/Expression/Expression.h"

#include <cstdint>
#include <optional>

namespace ThorImplementation {

struct NewtonSchulzOrthogonalizationOptions {
    uint32_t numIterations = 5;
    double coefficientA = 3.4445;
    double coefficientB = -4.775;
    double coefficientC = 2.0315;
    double epsilon = 1.0e-8;
    bool transposeTallMatrices = true;
    std::optional<DataType> computeDType = DataType::FP32;
    std::optional<DataType> outputDType = std::nullopt;
};

[[nodiscard]] Expression newtonSchulzOrthogonalize(const Expression& input,
                                                   uint64_t numRows,
                                                   uint64_t numCols,
                                                   NewtonSchulzOrthogonalizationOptions options = {});

[[nodiscard]] Expression newtonSchulzOrthogonalization(const Expression& input,
                                                       uint64_t numRows,
                                                       uint64_t numCols,
                                                       NewtonSchulzOrthogonalizationOptions options = {});

}  // namespace ThorImplementation
