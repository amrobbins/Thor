#include "Utilities/TensorMathFusion/FusedEquation.h"

#include "Utilities/TensorMathFusion/EquationCompiler.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

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

FusedEquation FusedEquation::compile(const PhysicalExpression& expr, TensorDescriptor::DataType dtype, int device_num, bool use_fast_math) {
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

    // FIXME add:
    // PhysicalExpressionValidator::validate(expr);

    EquationSignature sig{};
    sig.rank = 1;  // V1 assumes flattened contiguous tensors
    sig.num_inputs = expr.num_inputs;
    sig.dtype = toCudaDataType(dtype);
    sig.contiguous = true;
    sig.sm_major = prop.major;
    sig.sm_minor = prop.minor;
    sig.device_num = device_num;
    sig.use_fast_math = use_fast_math;

    EquationCompiler compiler;
    std::shared_ptr<CompiledEquation> compiledEquation = compiler.compile(expr, sig);

    return FusedEquation(std::move(compiledEquation));
}

StampedEquation FusedEquation::stamp(const std::vector<Tensor>& inputs, const Stream& stream) const {
    if (!compiledEquation) {
        throw std::runtime_error("Cannot stamp an empty FusedEquation.");
    }

    if (inputs.empty()) {
        throw std::runtime_error("FusedEquation::stamp requires at least one input tensor.");
    }

    if (inputs.size() != compiledEquation->num_inputs) {
        throw std::runtime_error("Wrong number of inputs passed to FusedEquation::stamp.");
    }

    const Tensor& firstInput = inputs[0];
    if (!firstInput.isInitialized()) {
        throw std::runtime_error("First input tensor is not initialized.");
    }

    if (firstInput.getDescriptor().getDataType() != compiledEquation->dtype) {
        throw std::runtime_error("Input tensor data type does not match compiled equation data type.");
    }

    if (firstInput.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
        throw std::runtime_error("Input tensor GPU does not match compiled equation device.");
    }

    for (uint64_t i = 1; i < inputs.size(); ++i) {
        if (!inputs[i].isInitialized()) {
            throw std::runtime_error("Input tensor is not initialized.");
        }
        if (inputs[i].getDescriptor().getDataType() != compiledEquation->dtype) {
            throw std::runtime_error("Input tensor data type mismatch.");
        }
        if (inputs[i].getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Input tensor GPU mismatch.");
        }
        if (inputs[i].getDescriptor().getTotalNumElements() != firstInput.getDescriptor().getTotalNumElements()) {
            throw std::runtime_error("All input tensors must have the same number of elements in V1.");
        }
    }

    TensorDescriptor outputDescriptor = firstInput.getDescriptor();
    Tensor output(firstInput.getPlacement(), outputDescriptor);

    if (!output.isInitialized()) {
        throw std::runtime_error("Failed to allocate output tensor during FusedEquation::stamp.");
    }

    return StampedEquation(compiledEquation, inputs, output, stream);
}

void FusedEquation::run(const std::vector<Tensor>& inputs, Tensor output, Stream stream) const {
    EquationRunner::run(compiledEquation, inputs, output, stream);
}
