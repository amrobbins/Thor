#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradientCudaCompile.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/FusedEquation.h"

#include <vector>

namespace ThorImplementation {

CompiledEmbeddingSparseGradientCudaKernel compileEmbeddingSparseGradientCudaKernel(const std::string& source,
                                                                                   const std::string& kernelName,
                                                                                   int deviceNum) {
    ScopedGpu scopedGpu(deviceNum);
    EquationSignature signature = FusedEquation::buildSignature(0, deviceNum, /*useFastMath=*/false);
    std::vector<char> lto = EquationCompiler::compileToLtoIr(source, kernelName, signature);
    std::vector<char> cubin = EquationCompiler::linkToCubin(lto, signature);

    CompiledEmbeddingSparseGradientCudaKernel compiled;
    CU_CHECK(cuModuleLoadData(&compiled.module, cubin.data()));
    CU_CHECK(cuModuleGetFunction(&compiled.function, compiled.module, kernelName.c_str()));
    return compiled;
}

}  // namespace ThorImplementation
