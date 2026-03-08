#pragma once

#include "Utilities/TensorMathFusion/EquationInstance.h"

namespace ThorImplementation {
class FusedEquation {
   public:
    static FusedEquation compile(const PhysicalExpression& expr, TensorDescriptor::DataType dtype, int device_num);

    EquationInstance instantiate(Tensor output, Stream stream) const;
};

}  // namespace ThorImplementation
