#pragma once

#include "Utilities/TensorMathFusion/BroadcastStructs.h"
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
    explicit FusedEquation(std::shared_ptr<CompiledEquation> flatEquation, std::shared_ptr<CompiledEquation> broadcastEquation)
        : compiledFlatEquation(std::move(flatEquation)), compiledBroadcastEquation(std::move(broadcastEquation)) {}

    static bool resolveLayout(std::vector<Tensor>& inputs, std::vector<uint64_t>& outputDimensions);
    static Tensor createDeviceBroadcastInfo(const std::vector<Tensor>& inputs,
                                            const std::vector<uint64_t>& outputDimensions,
                                            Stream stream);

    std::shared_ptr<CompiledEquation> compiledFlatEquation;
    std::shared_ptr<CompiledEquation> compiledBroadcastEquation;
};

}  // namespace ThorImplementation
