#include "Utilities/TensorMathFusion/EquationRunner.h"

namespace ThorImplementation {

void EquationRunner::run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                         const std::vector<Tensor>& inputs,
                         const std::vector<Tensor>& outputs,
                         Stream& stream) {
    if (!compiledEquation) {
        throw std::runtime_error("EquationInstance has no compiled equation.");
    }

    if (inputs.size() != compiledEquation->numInputs()) {
        std::string error_message = "Wrong number of inputs actual " + std::to_string(inputs.size()) + " vs expected " +
                                    std::to_string(compiledEquation->numInputs()) + "\n";
        throw std::runtime_error(error_message.c_str());
    }
    if (outputs.size() != compiledEquation->numOutputs()) {
        throw std::runtime_error("Wrong number of outputs passed to EquationRunner::run.");
    }

    uint64_t max_numel = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const Tensor& output = outputs[i];
        if (output.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::runtime_error("Output tensor is not located on a GPU.");
        }
        uint64_t numel = output.getTotalNumElements();
        if (numel > max_numel)
            max_numel = numel;

        if (output.getDataType() != compiledEquation->output_dtypes[i]) {
            throw std::runtime_error("Output type mismatch. Got: " + TensorDescriptor::getElementTypeName(output.getDataType()) +
                                     " Expected: " + TensorDescriptor::getElementTypeName(compiledEquation->output_dtypes[i]));
        }
        if (output.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::runtime_error("Output tensor is not located on a GPU.");
        }
        if (output.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Output tensor GPU does not match compiled fused equation device.");
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        const Tensor& t = inputs[i];
        if (!t.isInitialized()) {
            throw std::runtime_error("All input tensors must be initialized.");
        }

        if (t.getDescriptor().getDataType() != compiledEquation->input_dtypes[i]) {
            throw std::runtime_error("Fused equation input dtype mismatch");
        }

        if (t.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::runtime_error("Input tensor is not located on a GPU.");
        }

        if (t.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Input tensor GPU does not match compiled fused equation device.");
        }
    }

    assert(max_numel != 0);
    std::vector<const void*> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto& t : inputs) {
        input_ptrs.push_back(t.getMemPtr());
    }

    std::vector<const void*> output_ptrs;
    output_ptrs.reserve(outputs.size());
    for (const auto& t : outputs) {
        output_ptrs.push_back(t.getMemPtr());
    }

    std::vector<void*> args;
    args.reserve(inputs.size() + outputs.size() + 1);

    for (size_t i = 0; i < input_ptrs.size(); ++i) {
        args.push_back((void*)&input_ptrs[i]);
    }
    for (size_t i = 0; i < output_ptrs.size(); ++i) {
        args.push_back((void*)&output_ptrs[i]);
    }
    args.push_back((void*)&max_numel);

    const uint64_t launch_numel = (max_numel + static_cast<uint64_t>(compiledEquation->elements_per_thread) - 1ULL) /
                                  static_cast<uint64_t>(compiledEquation->elements_per_thread);
    uint32_t block = static_cast<uint32_t>(std::min<uint64_t>(launch_numel, 256ULL));
    uint32_t grid = static_cast<uint32_t>((launch_numel + block - 1ULL) / block);

    CU_CHECK(cuLaunchKernel(compiledEquation->kernel, grid, 1, 1, block, 1, 1, 0, stream, args.data(), nullptr));
}

}  // namespace ThorImplementation
