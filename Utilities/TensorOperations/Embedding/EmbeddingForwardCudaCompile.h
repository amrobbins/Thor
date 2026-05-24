#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ThorImplementation {

std::vector<char> compileEmbeddingForwardCudaKernelToCubin(const std::string& source,
                                                           const std::string& kernelName,
                                                           int deviceNum,
                                                           uint32_t numKernelInputs);

}  // namespace ThorImplementation
