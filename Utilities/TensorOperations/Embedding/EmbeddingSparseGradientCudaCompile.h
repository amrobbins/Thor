#pragma once

#include <cuda.h>

#include <string>

namespace ThorImplementation {

struct CompiledEmbeddingSparseGradientCudaKernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

[[nodiscard]] CompiledEmbeddingSparseGradientCudaKernel compileEmbeddingSparseGradientCudaKernel(const std::string& source,
                                                                                                 const std::string& kernelName,
                                                                                                 int deviceNum);

}  // namespace ThorImplementation
