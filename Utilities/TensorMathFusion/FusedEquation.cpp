#include "Utilities/TensorMathFusion/FusedEquation.h"

#include "Utilities/TensorMathFusion/Equation.h"
#include "Utilities/TensorMathFusion/EquationCompiler.h"
#include "Utilities/TensorMathFusion/Expression.h"

#include <cuda_runtime.h>

#include <stdexcept>

using namespace ThorImplementation;

namespace {
cudaDataType_t toCudaDataType(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP32:
            return CUDA_R_32F;
        default:
            throw std::runtime_error("Unsupported tensor data type for fused equation.");
    }
}
}  // namespace

FusedEquation FusedEquation::compile(const PhysicalExpression& expr, TensorDescriptor::DataType dtype, int device_num) {
    if (dtype != TensorDescriptor::DataType::FP32) {
        throw std::runtime_error("FusedEquation::compile V1 currently only supports FP32.");
    }

    if (device_num < 0) {
        throw std::runtime_error("FusedEquation::compile requires device_num >= 0.");
    }

    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(cuda_status));
    }

    if (device_num >= device_count) {
        throw std::runtime_error("FusedEquation::compile device_num is out of range.");
    }

    cudaDeviceProp prop{};
    cuda_status = cudaGetDeviceProperties(&prop, device_num);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(cuda_status));
    }

    // FIXME when you add it:
    // PhysicalExpressionValidator::validate(expr);

    EquationSignature sig{};
    sig.rank = 1;  // V1 assumes flattened contiguous tensors
    sig.num_inputs = expr.num_inputs;
    sig.dtype = toCudaDataType(dtype);
    sig.contiguous = true;
    sig.sm_major = prop.major;
    sig.sm_minor = prop.minor;
    sig.device_num = device_num;

    EquationCompiler compiler;
    std::shared_ptr<CompiledEquation> compiledEquation = compiler.compile(expr, sig);

    return FusedEquation(std::move(compiledEquation));
}

Equation FusedEquation::instantiate(Tensor output, Stream stream) const {
    if (!compiledEquation) {
        throw std::runtime_error("Cannot instantiate an empty FusedEquation.");
    }

    if (!output.isInitialized()) {
        throw std::runtime_error("FusedEquation::instantiate requires an initialized output tensor.");
    }

    if (output.getDescriptor().getDataType() != compiledEquation->dtype) {
        throw std::runtime_error("Output tensor data type does not match fused equation data type.");
    }

    if (output.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
        throw std::runtime_error("Output tensor GPU does not match compiled fused equation device.");
    }

    return Equation(compiledEquation, output, stream);
}
