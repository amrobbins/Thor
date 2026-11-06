#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"

#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/Common/Stream.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace ThorImplementation::CubReductionInternal {

DenseRKValueOperation denseRKValueOperation(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum: return DenseRKValueOperation::Sum;
        case CubReductionOp::Mean: return DenseRKValueOperation::Mean;
        case CubReductionOp::L1Norm: return DenseRKValueOperation::L1;
        case CubReductionOp::L2Norm: return DenseRKValueOperation::L2;
        case CubReductionOp::SumSquares: return DenseRKValueOperation::SumSquares;
        case CubReductionOp::Product: return DenseRKValueOperation::Product;
        case CubReductionOp::Min: return DenseRKValueOperation::Minimum;
        case CubReductionOp::Max: return DenseRKValueOperation::Maximum;
    }
    throw std::logic_error("Invalid DenseRK VALUE operation");
}

DenseProductionPhysicalSelectionContext::DenseProductionPhysicalSelectionContext(
    DataType original_input_dtype, DenseRKValueOperation operation, const Stream& stream,
    uint64_t original_reduction_size, float output_scale,
    std::optional<CubReductionOp> original_value_operation, uint64_t workspace_budget_bytes,
    CubReductionTuningDepth tuning_depth, uint64_t original_input_alignment_bytes) {
    int multiprocessors = 0;
    const cudaError_t status =
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, static_cast<int>(stream.getGpuNum()));
    if (status != cudaSuccess || multiprocessors <= 0) {
        throw std::runtime_error(std::string("Failed to query CUDA device geometry for dense planning: ")
                                 + cudaGetErrorString(status));
    }

    context_.tuning_stream = &stream;
    context_.gpu_ordinal = stream.getGpuNum();
    context_.original_value_operation = original_value_operation;
    context_.original_reduction_size = original_reduction_size;
    context_.output_scale = output_scale;
    context_.max_workspace_bytes = workspace_budget_bytes;
    context_.tuning_depth = tuning_depth;
    context_.original_input_alignment_bytes = original_input_alignment_bytes;
    context_.rk.original_input_dtype = original_input_dtype;
    context_.rk.value_operation = operation;
    context_.rk.multiprocessors = static_cast<uint32_t>(multiprocessors);
}

}  // namespace ThorImplementation::CubReductionInternal
