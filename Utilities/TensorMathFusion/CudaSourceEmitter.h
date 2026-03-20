#pragma once

#include <cstdint>
#include <string>

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"
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
    static std::string emit(const PhysicalExecutionStage& stage,
                            TensorDescriptor::DataType dtype,
                            const std::string& kernel_name,
                            bool broadcast_support);

    static std::string emitVector2Flat(const PhysicalExpression& expr, TensorDescriptor::DataType dtype, const std::string& kernel_name);
    static std::string emitVector2Broadcast(const PhysicalExpression& expr,
                                            TensorDescriptor::DataType dtype,
                                            const std::string& kernel_name);

    static std::string emitGroupedBroadcast(const CompiledExecutionStage& stage,
                                            const std::vector<std::vector<uint32_t>>& output_groups,
                                            TensorDescriptor::DataType dtype,
                                            const std::string& kernel_name);

   private:
    static std::string ref(uint32_t idx);

    static void emitScalarNode(std::ostringstream& ss,
                               const PhysicalExpression& expr,
                               uint32_t node_idx,
                               TensorDescriptor::DataType dtype,
                               const std::string& compute_type,
                               bool broadcast_support);
};

}  // namespace ThorImplementation
