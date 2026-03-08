#include "Utilities/TensorMathFusion/Equation.h"

#include <stdexcept>
#include <vector>

namespace ThorImplementation {

void Equation::run(const std::vector<Tensor>& inputs) {
    if (!compiledEquation) {
        throw std::runtime_error("EquationInstance has no compiled equation.");
    }

    if (inputs.size() != compiledEquation->num_inputs) {
        throw std::runtime_error("Wrong number of inputs");
    }

    uint64_t numel = output.getDescriptor().getTotalNumElements();

    for (const auto& t : inputs) {
        if (!t.isInitialized()) {
            throw std::runtime_error("All input tensors must be initialized.");
        }

        if (t.getDescriptor().getDataType() != TensorDescriptor::DataType::FP32) {
            throw std::runtime_error("V1 fused equations require FP32 tensors");
        }

        if (t.getDescriptor().getTotalNumElements() != numel) {
            throw std::runtime_error("Input numel mismatch");
        }

        if (t.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Input tensor GPU does not match compiled fused equation device.");
        }
    }

    if (output.getDataType() != TensorDescriptor::DataType::FP32) {
        throw std::runtime_error("Output must be FP32");
    }

    if (output.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
        throw std::runtime_error("Output tensor GPU does not match compiled fused equation device.");
    }

    std::vector<const float*> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto& t : inputs) {
        input_ptrs.push_back(t.getMemPtr<float>());
    }

    float* out_ptr = output.getMemPtr<float>();

    std::vector<void*> args;
    args.reserve(inputs.size() + 2);

    for (size_t i = 0; i < input_ptrs.size(); ++i) {
        args.push_back((void*)&input_ptrs[i]);
    }
    args.push_back((void*)&out_ptr);
    args.push_back((void*)&numel);

    uint32_t block = 256;
    uint32_t grid = static_cast<uint32_t>((numel + block - 1) / block);

    CU_CHECK(cuLaunchKernel(compiledEquation->kernel, grid, 1, 1, block, 1, 1, 0, stream, args.data(), nullptr));
}

void hashCombine(std::size_t& seed, std::size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

}  // namespace ThorImplementation
