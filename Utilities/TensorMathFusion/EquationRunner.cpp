#include "Utilities/TensorMathFusion/EquationRunner.h"

namespace ThorImplementation {
void EquationRunner::run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                         const std::vector<Tensor>& inputs,
                         Tensor& output,
                         Stream& stream) {
    if (!compiledEquation) {
        throw std::runtime_error("EquationInstance has no compiled equation.");
    }

    if (inputs.size() != compiledEquation->num_inputs) {
        std::string error_message = "Wrong number of inputs actual " + std::to_string(inputs.size()) + " vs expected " +
                                    std::to_string(compiledEquation->num_inputs) + "\n";
        throw std::runtime_error(error_message.c_str());
    }

    const TensorDescriptor& outputDescriptor = output.getDescriptor();
    if (output.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error("Output tensor is not located on a GPU.");
    }

    uint64_t numel = outputDescriptor.getTotalNumElements();

    for (const auto& t : inputs) {
        if (!t.isInitialized()) {
            throw std::runtime_error("All input tensors must be initialized.");
        }

        if (t.getDescriptor().getDataType() != compiledEquation->dtype) {
            throw std::runtime_error("V1 fused equations require FP32 tensors");
        }

        if (t.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::runtime_error("Input tensor is not located on a GPU.");
        }

        if (t.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Input tensor GPU does not match compiled fused equation device.");
        }

        if (t.getDescriptor() != outputDescriptor) {
            throw std::runtime_error("All input tensor descriptors must exactly match the output tensor descriptor in V1.");
        }
    }

    if (output.getDataType() != compiledEquation->dtype) {
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

    uint32_t block = std::min(numel, 256UL);
    uint32_t grid = static_cast<uint32_t>((numel + block - 1) / block);

    CU_CHECK(cuLaunchKernel(compiledEquation->kernel, grid, 1, 1, block, 1, 1, 0, stream, args.data(), nullptr));
}

void EquationRunner::run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                         const std::vector<Tensor>& inputs,
                         Tensor& output,
                         Stream stream,
                         Tensor deviceBroadcastInfo) {
    if (!compiledEquation) {
        throw std::runtime_error("EquationRunner::run has no compiled broadcast equation.");
    }

    if (inputs.size() != compiledEquation->num_inputs) {
        throw std::runtime_error("Wrong number of inputs actual " + std::to_string(inputs.size()) + " vs expected " +
                                 std::to_string(compiledEquation->num_inputs));
    }

    const TensorDescriptor& outputDescriptor = output.getDescriptor();

    if (output.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error("Output tensor is not located on a GPU.");
    }

    if (output.getDataType() != compiledEquation->dtype) {
        throw std::runtime_error("Output must match compiled fused equation dtype.");
    }

    if (output.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
        throw std::runtime_error("Output tensor GPU does not match compiled fused equation device.");
    }

    if (!deviceBroadcastInfo.isInitialized()) {
        throw std::runtime_error("Device broadcast info tensor is not initialized.");
    }

    if (deviceBroadcastInfo.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error("Device broadcast info tensor is not on GPU.");
    }

    if (deviceBroadcastInfo.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
        throw std::runtime_error("Device broadcast info tensor GPU does not match compiled fused equation device.");
    }

    if (deviceBroadcastInfo.getDescriptor().getDataType() != TensorDescriptor::DataType::UINT8 ||
        deviceBroadcastInfo.getDescriptor().getTotalNumElements() != sizeof(BroadcastInfo)) {
        throw std::runtime_error("Device broadcast info tensor has wrong descriptor.");
    }

    const uint64_t numel = outputDescriptor.getTotalNumElements();
    if (numel == 0) {
        return;
    }

    std::vector<const float*> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto& t : inputs) {
        if (!t.isInitialized()) {
            throw std::runtime_error("All input tensors must be initialized.");
        }

        if (t.getDescriptor().getDataType() != compiledEquation->dtype) {
            throw std::runtime_error("Input tensor dtype does not match compiled fused equation dtype.");
        }

        if (t.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::runtime_error("Input tensor is not located on a GPU.");
        }

        if (t.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Input tensor GPU does not match compiled fused equation device.");
        }

        input_ptrs.push_back(t.getMemPtr<float>());
    }

    float* out_ptr = output.getMemPtr<float>();
    BroadcastInfo* deviceBroadcastInfoPtr = reinterpret_cast<BroadcastInfo*>(deviceBroadcastInfo.getMemPtr());

    std::vector<void*> args;
    args.reserve(inputs.size() + 2);

    for (size_t i = 0; i < input_ptrs.size(); ++i) {
        args.push_back((void*)&input_ptrs[i]);
    }
    args.push_back((void*)&out_ptr);
    args.push_back((void*)&deviceBroadcastInfoPtr);

    uint32_t block = static_cast<uint32_t>(std::min<uint64_t>(numel, 256ULL));
    uint32_t grid = static_cast<uint32_t>((numel + block - 1) / block);

    CU_CHECK(cuLaunchKernel(compiledEquation->kernel, grid, 1, 1, block, 1, 1, 0, stream, args.data(), nullptr));
}

}  // namespace ThorImplementation
