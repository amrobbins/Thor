#include "Utilities/TensorMathFusion/EquationRunner.h"
#include <limits>

namespace ThorImplementation {

static size_t runtimeInputScalarSizeBytes(TensorDescriptor::DataType dtype) {
    switch (dtype) {
        case TensorDescriptor::DataType::FP32:
            return 4;
        case TensorDescriptor::DataType::FP16:
            return 2;
        case TensorDescriptor::DataType::BF16:
            return 2;
        case TensorDescriptor::DataType::FP8_E4M3:
            return 1;
        case TensorDescriptor::DataType::FP8_E5M2:
            return 1;
        case TensorDescriptor::DataType::UINT8:
            return 1;
        case TensorDescriptor::DataType::UINT16:
            return 2;
        case TensorDescriptor::DataType::UINT32:
            return 4;
        case TensorDescriptor::DataType::INT32:
            return 4;
        default:
            throw std::runtime_error("Unsupported dtype in runtimeInputScalarSizeBytes.");
    }
}

void EquationRunner::run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                         const std::vector<RuntimeInputValue>& inputs,
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
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::Tensor) {
            if (!std::holds_alternative<Tensor>(inputs[i])) {
                throw std::runtime_error("Fused equation expected a tensor input at slot " + std::to_string(i) + ".");
            }

            const Tensor& t = std::get<Tensor>(inputs[i]);
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
        } else if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            if (!std::holds_alternative<float>(inputs[i])) {
                throw std::runtime_error("Fused equation expected a runtime scalar input at slot " + std::to_string(i) + ".");
            }
            if (compiledEquation->input_dtypes[i] != TensorDescriptor::DataType::FP32) {
                throw std::runtime_error("Runtime scalar inputs currently require FP32 compiled input dtype.");
            }
        } else {
            if (!std::holds_alternative<TensorScalarBinding>(inputs[i])) {
                throw std::runtime_error("Fused equation expected a tensor-backed runtime scalar input at slot " + std::to_string(i) + ".");
            }
            const TensorScalarBinding& binding = std::get<TensorScalarBinding>(inputs[i]);
            if (!binding.buffer.isInitialized()) {
                throw std::runtime_error("Tensor-backed runtime scalar buffer is not initialized.");
            }
            if (binding.buffer.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::runtime_error("Tensor-backed runtime scalar buffer is not located on a GPU.");
            }
            if (binding.buffer.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
                throw std::runtime_error("Tensor-backed runtime scalar GPU does not match compiled fused equation device.");
            }
            if (binding.sourceDType != compiledEquation->input_dtypes[i]) {
                throw std::runtime_error("Tensor-backed runtime scalar dtype mismatch.");
            }
            if (binding.byteOffset + runtimeInputScalarSizeBytes(binding.sourceDType) > binding.buffer.getArraySizeInBytes()) {
                throw std::runtime_error("Tensor-backed runtime scalar binding exceeds backing buffer size.");
            }
        }
    }

    assert(max_numel != 0);
    std::vector<const void*> input_ptrs;
    input_ptrs.reserve(inputs.size());
    std::vector<float> scalar_inputs_fp32;
    scalar_inputs_fp32.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::Tensor) {
            input_ptrs.push_back(std::get<Tensor>(inputs[i]).getMemPtr());
        } else if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            scalar_inputs_fp32.push_back(std::get<float>(inputs[i]));
            input_ptrs.push_back(&scalar_inputs_fp32.back());
        } else {
            const TensorScalarBinding& binding = std::get<TensorScalarBinding>(inputs[i]);
            const uint8_t* base = static_cast<const uint8_t*>(binding.buffer.getMemPtr());
            input_ptrs.push_back(base + binding.byteOffset);
        }
    }

    std::vector<const void*> output_ptrs;
    output_ptrs.reserve(outputs.size());
    for (const auto& t : outputs) {
        output_ptrs.push_back(t.getMemPtr());
    }

    std::vector<void*> args;
    args.reserve(inputs.size() + outputs.size() + 1);

    size_t next_scalar_index = 0;
    for (size_t i = 0; i < input_ptrs.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            args.push_back((void*)&scalar_inputs_fp32[next_scalar_index++]);
        } else {
            args.push_back((void*)&input_ptrs[i]);
        }
    }
    for (size_t i = 0; i < output_ptrs.size(); ++i) {
        args.push_back((void*)&output_ptrs[i]);
    }

    int32_t max_numel_i32 = 0;
    int64_t max_numel_i64 = 0;
    if (compiledEquation->uses_int32_numel_arg) {
        if (max_numel > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("Flat kernel compiled for int32_t numel was launched with numel exceeding int32_t.");
        }
        max_numel_i32 = static_cast<int32_t>(max_numel);
        args.push_back((void*)&max_numel_i32);
    } else {
        if (max_numel > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("Flat kernel compiled for int64_t numel was launched with numel exceeding int64_t.");
        }
        max_numel_i64 = static_cast<int64_t>(max_numel);
        args.push_back((void*)&max_numel_i64);
    }

    const uint64_t launch_numel = (max_numel + static_cast<uint64_t>(compiledEquation->elements_per_thread) - 1ULL) /
                                  static_cast<uint64_t>(compiledEquation->elements_per_thread);
    uint32_t block = static_cast<uint32_t>(std::min<uint64_t>(launch_numel, 256ULL));
    uint32_t grid = static_cast<uint32_t>((launch_numel + block - 1ULL) / block);

    CU_CHECK(cuLaunchKernel(compiledEquation->kernel, grid, 1, 1, block, 1, 1, 0, stream, args.data(), nullptr));
}

}  // namespace ThorImplementation
