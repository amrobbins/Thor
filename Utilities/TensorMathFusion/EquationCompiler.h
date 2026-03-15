#pragma once

#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#include <nvJitLink.h>
#include <nvrtc.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/TensorMathFusion/CudaHelpers.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {

struct PhysicalExecutionStage {
    enum class Kind { FusedKernel, Reduction };

    Kind kind;
    PhysicalExpression expr;
    std::vector<uint32_t> input_value_ids;
    uint32_t output_value_id;
};

class EquationCompiler {
   public:
    static std::shared_ptr<CompiledEquation> compile(const PhysicalExpression& expr,
                                                     const EquationSignature& sig,
                                                     const bool broadcast_support);
    static std::vector<char> compileToLtoIr(const std::string& src, const std::string& kernel_name, const EquationSignature& sig);
    static std::vector<char> linkToCubin(const std::vector<char>& ltoir, const EquationSignature& sig);
    static std::shared_ptr<CompiledEquation> loadCubin(const EquationCacheKey& key,
                                                       const std::vector<char>& cubin,
                                                       const std::string& kernel_name,
                                                       const std::vector<std::string>& input_names,
                                                       TensorDescriptor::DataType dtype,
                                                       int device_num);

    static std::vector<PhysicalExecutionStage> splitAtReductionBoundaries(const PhysicalExpression& expr);

    static std::shared_ptr<CompiledReduction> compileReduction(const PhysicalExpression& expr, TensorDescriptor::DataType inout_dtype);
};

}  // namespace ThorImplementation
