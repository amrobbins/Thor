// The front end's shared argument checks and workspace helpers (FrontEnd.h).

#include "Utilities/TensorOperations/Reduction/FrontEnd.h"

#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation {

namespace ReductionFrontEnd {

using namespace CubDevicePrimitiveSupport;
using namespace ReductionInternal;

[[nodiscard]] bool isSupportedFloatingStorageDType(DataType dtype) {
    switch (dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
#endif
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:  // no FP64 reductions
            return false;
    }
}

void requireSupportedFloatingStorageDType(DataType dtype, const char* role) {
    if (!isSupportedFloatingStorageDType(dtype)) {
        throw std::invalid_argument(std::string("Tensor reduction does not support ") + role + " dtype " +
                                    dtypeName(dtype) +
                                    ". Supported storage dtypes are FP8 (per THOR_CUB_ENABLE_FP8_TYPES), FP16, BF16 "
                                    "and FP32; TF32 is not a storage dtype.");
    }
}

[[nodiscard]] bool isSupportedArgIndexDType(DataType dtype) {
    return dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

void requireSupportedArgIndexDType(DataType dtype) {
    if (!isSupportedArgIndexDType(dtype)) {
        throw std::invalid_argument("Arg reduction index output dtype must be UINT32 or UINT64.");
    }
}

[[nodiscard]] std::vector<uint64_t> stripSingletonDimensions(const std::vector<uint64_t>& dimensions) {
    std::vector<uint64_t> stripped;
    stripped.reserve(dimensions.size());
    for (uint64_t dimension : dimensions) {
        if (dimension != 1) {
            stripped.push_back(dimension);
        }
    }
    if (stripped.empty()) {
        stripped.push_back(1);
    }
    return stripped;
}

[[nodiscard]] ReductionInternal::ReductionStageRole physicalStageRole(bool apply_input_transform,
                                                                             bool apply_output_finalize) {
    using ReductionInternal::ReductionStageRole;
    if (apply_input_transform && apply_output_finalize) {
        return ReductionStageRole::Complete;
    }
    if (apply_input_transform) {
        return ReductionStageRole::First;
    }
    if (apply_output_finalize) {
        return ReductionStageRole::Final;
    }
    return ReductionStageRole::Intermediate;
}

void requireDenseContiguousReductionOutput(const Tensor& output, const char* role) {
    if (!output.isInitialized()) {
        throw std::invalid_argument(std::string(role) + " must be initialized.");
    }
    if (output.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string(role) + " must be a GPU tensor.");
    }
    // Reduction outputs only require a dense contiguous address range beginning at getMemPtr().
    if (!output.isDenseContiguous()) {
        throw std::invalid_argument(std::string(role) + " must be dense contiguous.");
    }
}

void requireSupportedArgOperation(ArgReductionOp op) {
    switch (op) {
        case ArgReductionOp::ArgMin:
        case ArgReductionOp::ArgMax:
            return;
    }
    throw std::invalid_argument("Unsupported arg reduction operation.");
}

void requireSupportedOperation(ReductionOp op) {
    switch (op) {
        case ReductionOp::Sum:
        case ReductionOp::Min:
        case ReductionOp::Max:
        case ReductionOp::Product:
        case ReductionOp::Mean:
        case ReductionOp::L1Norm:
        case ReductionOp::L2Norm:
        case ReductionOp::SumSquares:
            return;
    }
    throw std::invalid_argument("Unsupported tensor reduction operation.");
}

[[nodiscard]] size_t checkedWorkspaceAdd(size_t lhs, size_t rhs, const char* message) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) throw std::overflow_error(message);
    return lhs + rhs;
}

}  // namespace ReductionFrontEnd

namespace {

using namespace CubDevicePrimitiveSupport;
using namespace ReductionInternal;
using namespace ReductionFrontEnd;

}  // namespace

}  // namespace ThorImplementation
