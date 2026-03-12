#pragma once

#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorMathFusion/CompiledEquation.h"
#include "Utilities/TensorMathFusion/CudaHelpers.h"

namespace ThorImplementation {

constexpr uint32_t MAX_BROADCAST_DIMS = 10;
constexpr uint32_t MAX_FUSED_INPUTS = 64;

struct BroadcastInputInfo {
    unsigned long long strides[MAX_BROADCAST_DIMS];
};

struct BroadcastInfo {
    unsigned int rank;
    unsigned int num_inputs;
    unsigned long long numel;
    unsigned long long output_strides[MAX_BROADCAST_DIMS];
    // I only expose the actual number of inputs in cuda
    BroadcastInputInfo inputs[MAX_FUSED_INPUTS];
};

class EquationRunner {
   public:
    // FIXME: Eventually: This is hard coded to FP32
    static void run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                    const std::vector<Tensor>& inputs,
                    Tensor& output,
                    Stream& stream);
    static void run(const std::shared_ptr<CompiledEquation>& compiledEquation,
                    const std::vector<Tensor>& inputs,
                    Tensor& output,
                    Stream stream,
                    Tensor deviceBroadcastInfo);
};

}  // namespace ThorImplementation
