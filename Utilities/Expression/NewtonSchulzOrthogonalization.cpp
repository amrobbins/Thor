#include "Utilities/Expression/NewtonSchulzOrthogonalization.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ThorImplementation {

namespace {

void validateOptions(uint64_t numRows, uint64_t numCols, const NewtonSchulzOrthogonalizationOptions& options) {
    THOR_THROW_IF_FALSE(numRows > 0);
    THOR_THROW_IF_FALSE(numCols > 0);
    THOR_THROW_IF_FALSE(std::isfinite(options.coefficientA));
    THOR_THROW_IF_FALSE(std::isfinite(options.coefficientB));
    THOR_THROW_IF_FALSE(std::isfinite(options.coefficientC));
    THOR_THROW_IF_FALSE(std::isfinite(options.epsilon));
    THOR_THROW_IF_FALSE(options.epsilon > 0.0);
}

Expression normalizeByFrobeniusNorm(const Expression& input, const NewtonSchulzOrthogonalizationOptions& options) {
    auto frobeniusNorm = (input * input).reduce_sum(/*reduction_axes=*/{}, /*squeeze_axes=*/{UINT64_MAX}, options.computeDType).sqrt();
    return input / (frobeniusNorm + Expression::constantScalar(options.epsilon));
}

Expression leftNewtonSchulzStep(const Expression& x, const NewtonSchulzOrthogonalizationOptions& options) {
    auto xxT = Expression::matmul(x,
                                  x,
                                  /*transpose_lhs=*/false,
                                  /*transpose_rhs=*/true,
                                  options.computeDType,
                                  options.computeDType);
    auto xxTSquared = Expression::matmul(xxT,
                                         xxT,
                                         /*transpose_lhs=*/false,
                                         /*transpose_rhs=*/false,
                                         options.computeDType,
                                         options.computeDType);
    auto polynomial = Expression::constantScalar(options.coefficientB) * xxT + Expression::constantScalar(options.coefficientC) * xxTSquared;
    return Expression::constantScalar(options.coefficientA) * x +
           Expression::matmul(polynomial,
                              x,
                              /*transpose_lhs=*/false,
                              /*transpose_rhs=*/false,
                              options.computeDType,
                              options.computeDType);
}

Expression rightNewtonSchulzStep(const Expression& x, const NewtonSchulzOrthogonalizationOptions& options) {
    auto xTx = Expression::matmul(x,
                                  x,
                                  /*transpose_lhs=*/true,
                                  /*transpose_rhs=*/false,
                                  options.computeDType,
                                  options.computeDType);
    auto xTxSquared = Expression::matmul(xTx,
                                         xTx,
                                         /*transpose_lhs=*/false,
                                         /*transpose_rhs=*/false,
                                         options.computeDType,
                                         options.computeDType);
    auto polynomial = Expression::constantScalar(options.coefficientB) * xTx + Expression::constantScalar(options.coefficientC) * xTxSquared;
    return Expression::constantScalar(options.coefficientA) * x +
           Expression::matmul(x,
                              polynomial,
                              /*transpose_lhs=*/false,
                              /*transpose_rhs=*/false,
                              options.computeDType,
                              options.computeDType);
}

}  // namespace

Expression newtonSchulzOrthogonalize(const Expression& input,
                                     uint64_t numRows,
                                     uint64_t numCols,
                                     NewtonSchulzOrthogonalizationOptions options) {
    validateOptions(numRows, numCols, options);

    const bool shouldTransposeTallMatrix = options.transposeTallMatrices && numRows > numCols;
    Expression x = shouldTransposeTallMatrix ? input.transpose() : input;
    x = normalizeByFrobeniusNorm(x, options);

    const uint64_t workingRows = shouldTransposeTallMatrix ? numCols : numRows;
    const uint64_t workingCols = shouldTransposeTallMatrix ? numRows : numCols;
    const bool useLeftPolynomial = workingRows <= workingCols;

    for (uint32_t i = 0; i < options.numIterations; ++i) {
        x = useLeftPolynomial ? leftNewtonSchulzStep(x, options) : rightNewtonSchulzStep(x, options);
    }

    if (shouldTransposeTallMatrix) {
        x = x.transpose();
    }
    if (options.outputDType.has_value()) {
        x = x.withOutputDType(options.outputDType.value());
    }
    return x;
}

Expression newtonSchulzOrthogonalization(const Expression& input,
                                         uint64_t numRows,
                                         uint64_t numCols,
                                         NewtonSchulzOrthogonalizationOptions options) {
    return newtonSchulzOrthogonalize(input, numRows, numCols, std::move(options));
}

}  // namespace ThorImplementation
