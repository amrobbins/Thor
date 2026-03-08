#pragma once

#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {
class FusedEquation {
   public:
    static FusedEquation compile(const PhysicalExpression& expr, TensorDescriptor::DataType dtype, int device_num);

    StampedEquation stamp(const std::vector<Tensor>& inputs, Stream stream) const;
    void run(const std::vector<Tensor>& inputs, Tensor output, Stream stream) const;

   private:
    explicit FusedEquation(std::shared_ptr<CompiledEquation> compiledEquation) : compiledEquation(std::move(compiledEquation)) {}

    std::shared_ptr<CompiledEquation> compiledEquation;

    friend class EquationCompiler;
};

}  // namespace ThorImplementation
