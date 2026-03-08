#pragma once

#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#include <nvJitLink.h>
#include <nvrtc.h>

#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {
class EquationCompiler {
   public:
    std::shared_ptr<CompiledEquation> compile(const PhysicalExpression& expr, const EquationSignature& sig);
    std::vector<char> compileToLtoIr(const std::string& src, const std::string& kernel_name, const EquationSignature& sig);
    std::vector<char> linkToCubin(const std::vector<char>& ltoir, const EquationSignature& sig);
    std::shared_ptr<CompiledEquation> loadCubin(const EquationCacheKey& key,
                                                const std::vector<char>& cubin,
                                                const std::string& kernel_name,
                                                uint32_t num_inputs,
                                                TensorDescriptor::DataType dtype,
                                                int device_num);
};

}  // namespace ThorImplementation
