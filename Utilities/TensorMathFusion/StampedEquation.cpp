#include "Utilities/TensorMathFusion/StampedEquation.h"
#include "Utilities/TensorMathFusion/EquationRunner.h"

#include <stdexcept>
#include <vector>

namespace ThorImplementation {

void StampedEquation::run() {
    if (deviceBroadcastInfo.isPresent())
        EquationRunner::run(compiledEquation, inputs, output, stream, deviceBroadcastInfo);
    else
        EquationRunner::run(compiledEquation, inputs, output, stream);
}

}  // namespace ThorImplementation
