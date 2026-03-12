#include "Utilities/TensorMathFusion/StampedEquation.h"
#include "Utilities/TensorMathFusion/EquationRunner.h"

#include <stdexcept>
#include <vector>

namespace ThorImplementation {

void StampedEquation::run() { EquationRunner::run(compiledEquation, inputs, output, stream); }

}  // namespace ThorImplementation
