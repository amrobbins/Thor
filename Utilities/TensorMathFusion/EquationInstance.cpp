#include "Utilities/TensorMathFusion/EquationInstance.h"

namespace ThorImplementation {

void EquationInstance::run(const std::vector<Tensor>& inputs) {
    if (inputs.size() != compiled->num_inputs)
        throw std::runtime_error("Wrong number of inputs");

    for (const auto& t : inputs) {
        if (t.getDataType() != TensorDescriptor::DataType::FP32)
            throw std::runtime_error("V1 fused equations require FP32 tensors");
        if (t.getDescriptor().getTotalNumElements() != numel)
            throw std::runtime_error("Input numel mismatch");
    }

    if (output.getDataType() != TensorDescriptor::DataType::FP32)
        throw std::runtime_error("Output must be FP32");
    if (output.getDescriptor().getTotalNumElements() != numel)
        throw std::runtime_error("Output numel mismatch");

    std::vector<const float*> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto& t : inputs)
        input_ptrs.push_back(t.getMemPtr<float>());

    float* out_ptr = reinterpret_cast<float*>(output.getMemPtr());

    std::vector<void*> args;
    for (auto p : input_ptrs)
        args.push_back((void*)&p);
    args.push_back((void*)&out_ptr);
    args.push_back((void*)&numel);

    uint32_t block = 256;
    uint32_t grid = (numel + block - 1) / block;

    CU_CHECK(cuLaunchKernel(compiled->kernel, grid, 1, 1, block, 1, 1, 0, stream, args.data(), nullptr));
}

}  // namespace ThorImplementation
