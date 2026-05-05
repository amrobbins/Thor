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

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

namespace ThorImplementation {

struct PhysicalExecutionStage {
    enum class Kind { FusedKernel, Reduction, ArgMinMax, Softmax, Matmul, Convolution, ConvolutionBackward, ReduceMinMaxBackward };

    Kind kind;
    PhysicalExpression expr;
    std::vector<uint32_t> input_value_ids;
    std::vector<CompiledStageOutput> outputs;
    std::vector<ParameterFanOverride> parameter_fan_overrides;
};

class EquationCompiler {
   public:
    static std::shared_ptr<CompiledOutputs> compile(const PhysicalOutputs& outputs, const EquationSignature& sig, bool broadcast_support);
    static std::vector<char> compileToLtoIr(const std::string& src, const std::string& kernel_name, const EquationSignature& sig);
    static std::vector<char> linkToCubin(const std::vector<char>& ltoir, const EquationSignature& sig);
    static std::shared_ptr<CompiledEquation> loadCubin(const EquationCacheKey& key,
                                                       const std::vector<char>& cubin,
                                                       const std::string& kernel_name,
                                                       const std::vector<std::string>& input_names,
                                                       const std::vector<NamedInput::Kind>& input_kinds,
                                                       const std::vector<TensorDescriptor::DataType>& input_dtypes,
                                                       const std::vector<TensorDescriptor::DataType>& output_dtypes,
                                                       int device_num);

    static std::shared_ptr<CompiledEquation> compileFusedStage(const PhysicalExecutionStage& stage,
                                                               const EquationSignature& sig,
                                                               bool use_uint32_index_math = true);
    static std::vector<PhysicalExecutionStage> splitAtReductionBoundaries(const PhysicalOutputs& outputs);

    static std::shared_ptr<CompiledReduction> compileReduction(const PhysicalExpression& expr);
    static std::shared_ptr<CompiledArgMinMax> compileArgMinMax(const PhysicalExpression& expr);
    static std::shared_ptr<CompiledSoftmax> compileSoftmax(const PhysicalExpression& expr);
    static std::shared_ptr<CompiledMatmul> compileMatmul(const PhysicalExpression& expr);
    static std::shared_ptr<CompiledConvolution> compileConvolution(const PhysicalExpression& expr);
    static std::shared_ptr<CompiledConvolutionBackward> compileConvolutionBackward(const PhysicalExpression& expr);
    static std::shared_ptr<CompiledReduceMinMaxBackward> compileReduceMinMaxBackward(const PhysicalExpression& expr);
    static std::shared_ptr<CompiledEquation> compileSpecializedBroadcastStage(const CompiledExecutionStage& stage,
                                                                              const EquationSignature& sig,
                                                                              const std::vector<SpecializedBroadcastGroup>& groups);
    static std::string getCudaIncludeDir();
};

}  // namespace ThorImplementation
