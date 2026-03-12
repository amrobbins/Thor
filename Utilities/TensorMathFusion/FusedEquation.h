#pragma once

#include "Utilities/TensorMathFusion/EquationRunner.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {
class FusedEquation {
   public:
    static FusedEquation compile(const PhysicalExpression& expr,
                                 TensorDescriptor::DataType dtype,
                                 int device_num,
                                 bool use_fast_math = false);

    [[nodiscard]] StampedEquation stamp(std::vector<Tensor> inputs,
                                        const Stream& stream,
                                        const std::vector<uint64_t>& requestedOutputShape = {}) const;
    void run(std::vector<Tensor> inputs, Tensor output, Stream stream) const;

   private:
    explicit FusedEquation(std::shared_ptr<CompiledEquation> compiledEquation) : compiledEquation(std::move(compiledEquation)) {}

    static std::vector<uint64_t> resolveLayout(std::vector<Tensor>& inputs);

    std::shared_ptr<CompiledEquation> compiledEquation;

    friend class EquationCompiler;
};

}  // namespace ThorImplementation
