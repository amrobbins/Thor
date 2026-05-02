#pragma once

#include <cstdint>
#include <string>

#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ThorImplementation {

class CudaSourceEmitter {
   public:
    static std::string emitFlat(const PhysicalExpression& expr, const std::string& kernel_name, bool use_uint32_index_math = true);
    static std::string emitFlat(const PhysicalExecutionStage& stage, const std::string& kernel_name, bool use_uint32_index_math = true);

    static std::string emitSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                const std::vector<SpecializedBroadcastGroup>& groups,
                                                const std::string& kernel_name);

    static std::string emitTranspose(const PhysicalExecutionStage& stage, const std::string& kernel_name);

    static Optional<TensorDescriptor::DataType> getVectorizedStageStorageDType(const PhysicalExecutionStage& stage);
    static Optional<TensorDescriptor::DataType> getVectorizedStageStorageDType(const CompiledExecutionStage& stage);
    static uint32_t flatElementsPerThread(const PhysicalExecutionStage& stage);
    static uint32_t tiledTransposePackScalars(const PhysicalExecutionStage& stage);

    static std::string ref(uint32_t idx);
};

}  // namespace ThorImplementation
