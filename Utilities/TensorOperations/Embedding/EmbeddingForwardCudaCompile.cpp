#include "Utilities/TensorOperations/Embedding/EmbeddingForwardCudaCompile.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/FusedEquation.h"

#include <vector>

namespace ThorImplementation {

std::vector<char> compileEmbeddingForwardCudaKernelToCubin(const std::string& source,
                                                           const std::string& kernelName,
                                                           int deviceNum,
                                                           uint32_t numKernelInputs) {
    ScopedGpu scopedGpu(deviceNum);
    EquationSignature signature = FusedEquation::buildSignature(numKernelInputs, deviceNum, /*useFastMath=*/false);
    std::vector<char> lto = EquationCompiler::compileToLtoIr(source, kernelName, signature);
    return EquationCompiler::linkToCubin(lto, signature);
}

}  // namespace ThorImplementation
