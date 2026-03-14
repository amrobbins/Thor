#pragma once

#include <cstdint>
#include <string>

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ThorImplementation {

class CudaSourceEmitter {
   public:
    static std::string emit(const PhysicalExpression& expr,
                            TensorDescriptor::DataType dtype,
                            const std::string& kernel_name,
                            const bool broadcast_support);

    static std::string emitVector2Flat(const PhysicalExpression& expr, TensorDescriptor::DataType dtype, const std::string& kernel_name);
    static std::string emitVector2Broadcast(const PhysicalExpression& expr,
                                            TensorDescriptor::DataType dtype,
                                            const std::string& kernel_name);

   private:
    static std::string ref(uint32_t idx);
};

}  // namespace ThorImplementation
