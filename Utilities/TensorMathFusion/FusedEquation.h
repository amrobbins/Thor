#pragma once

#include "Utilities/TensorMathFusion/Equation.h"
#include "Utilities/TensorMathFusion/Expression.h"

namespace ThorImplementation {
class FusedEquation {
   public:
    static FusedEquation compile(const PhysicalExpression& expr,
                                 const std::vector<Tensor>& inputs,
                                 TensorDescriptor::DataType dtype,
                                 int device_num);

    Equation instantiate(Stream stream) const;

   private:
    explicit FusedEquation(std::shared_ptr<CompiledEquation> compiledEquation) : compiledEquation(std::move(compiledEquation)) {}

    std::shared_ptr<CompiledEquation> compiledEquation;

    friend class EquationCompiler;
};

}  // namespace ThorImplementation
