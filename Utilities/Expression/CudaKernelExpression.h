#pragma once

#include <cuda.h>
#include <vector_types.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/StampedEquation.h"

namespace ThorImplementation {

class CudaKernelExpression {
   public:
    class DimExpr {
       public:
        enum class Kind : uint8_t { Constant, TensorDim, TensorNumel };

        [[nodiscard]] static DimExpr constant(uint64_t value) { return DimExpr(Kind::Constant, "", 0, value); }
        [[nodiscard]] static DimExpr dim(std::string tensor_name, uint32_t axis) {
            return DimExpr(Kind::TensorDim, std::move(tensor_name), axis, 0);
        }
        [[nodiscard]] static DimExpr numel(std::string tensor_name) { return DimExpr(Kind::TensorNumel, std::move(tensor_name), 0, 0); }

        [[nodiscard]] uint64_t resolve(const std::unordered_map<std::string, Tensor>& tensors) const;
        [[nodiscard]] uint64_t resolve(const std::unordered_map<std::string, std::vector<uint64_t>>& tensor_shapes) const;
        [[nodiscard]] std::string describe() const;

       private:
        DimExpr(Kind kind, std::string tensor_name, uint32_t axis, uint64_t value)
            : kind_(kind), tensor_name_(std::move(tensor_name)), axis_(axis), value_(value) {}

        Kind kind_ = Kind::Constant;
        std::string tensor_name_;
        uint32_t axis_ = 0;
        uint64_t value_ = 0;
    };

    struct TensorParamSpec {
        enum class Kind : uint8_t { Tensor, TensorRuntimeScalar, HostRuntimeScalar };

        std::string name;
        TensorDescriptor::DataType dtype = TensorDescriptor::DataType::FP32;
        Kind kind = Kind::Tensor;
    };

    struct OutputParamSpec {
        std::string name;
        TensorDescriptor::DataType dtype = TensorDescriptor::DataType::FP32;
        std::vector<DimExpr> shape;
        std::string like_input_name;
    };

    struct ScalarParamSpec {
        std::string name;
        TensorDescriptor::DataType type = TensorDescriptor::DataType::INT64;
        std::variant<int32_t, uint32_t, int64_t, uint64_t, float, double, DimExpr> value = int64_t{0};
    };

    struct LaunchContext {
        const std::unordered_map<std::string, Tensor>& inputs;
        const std::unordered_map<std::string, Tensor>& outputs;
        int device_num = 0;

        [[nodiscard]] const Tensor& input(const std::string& name) const;
        [[nodiscard]] const Tensor& output(const std::string& name) const;
        [[nodiscard]] uint64_t dim(const std::string& tensor_name, uint32_t axis) const;
        [[nodiscard]] uint64_t numel(const std::string& tensor_name) const;
        [[nodiscard]] TensorDescriptor::DataType dtype(const std::string& tensor_name) const;
    };

    using LaunchFn = std::function<CudaKernelLaunchConfig(const LaunchContext&)>;

    class Builder {
       public:
        explicit Builder(std::string name);

        Builder& source(std::string cuda_source);
        Builder& entry(std::string entrypoint);
        Builder& input(std::string name, TensorDescriptor::DataType dtype);
        Builder& tensorRuntimeScalarInput(std::string name, TensorDescriptor::DataType dtype);
        Builder& hostRuntimeScalarInput(std::string name, TensorDescriptor::DataType dtype);
        Builder& output(std::string name, TensorDescriptor::DataType dtype, std::vector<DimExpr> shape);
        Builder& outputLike(std::string name, TensorDescriptor::DataType dtype, const std::string& input_name);
        Builder& scalar(std::string name, TensorDescriptor::DataType type, std::variant<int32_t, uint32_t, int64_t, uint64_t, float, double, DimExpr> value);
        Builder& launch(LaunchFn launch_fn);
        Builder& useFastMath(bool enabled = true);

        [[nodiscard]] CudaKernelExpression build() const;

       private:
        std::string name_;
        std::string source_;
        std::string entry_;
        std::vector<TensorParamSpec> inputs_;
        std::vector<OutputParamSpec> outputs_;
        std::vector<ScalarParamSpec> scalars_;
        LaunchFn launch_fn_;
        bool use_fast_math_ = false;
    };

    [[nodiscard]] static Builder builder(std::string name) { return Builder(std::move(name)); }

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] const std::vector<TensorParamSpec>& inputs() const { return inputs_; }
    [[nodiscard]] const std::vector<OutputParamSpec>& outputs() const { return outputs_; }
    [[nodiscard]] const std::vector<ScalarParamSpec>& scalars() const { return scalars_; }
    [[nodiscard]] std::string cacheSignature() const;
    [[nodiscard]] std::vector<std::vector<uint64_t>> inferOutputShapesFromInputShapes(
        const std::unordered_map<std::string, std::vector<uint64_t>>& input_shapes) const;

    [[nodiscard]] Outputs apply(const std::unordered_map<std::string, Expression>& inputs) const;
    [[nodiscard]] DynamicExpression asDynamicExpression() const;
    [[nodiscard]] StampedExecutionPlan stamp(
        const std::unordered_map<std::string, Tensor>& inputs,
        const std::unordered_map<std::string, Tensor>& preallocated_outputs,
        Stream& stream,
        const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs = {}) const;

    [[nodiscard]] std::shared_ptr<CompiledCudaKernel> compile(int device_num) const;
    [[nodiscard]] std::shared_ptr<StampedCudaKernel> stampCompiled(
        const std::shared_ptr<CompiledCudaKernel>& compiled,
        const std::unordered_map<std::string, Tensor>& inputs,
        const std::unordered_map<std::string, Tensor>& preallocated_outputs,
        const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes,
        const Stream& stream,
        std::unordered_map<std::string, Tensor>& resolved_outputs,
        const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs = {}) const;

   private:
    CudaKernelExpression(std::string name,
                         std::string source,
                         std::string entry,
                         std::vector<TensorParamSpec> inputs,
                         std::vector<OutputParamSpec> outputs,
                         std::vector<ScalarParamSpec> scalars,
                         LaunchFn launch_fn,
                         bool use_fast_math);

    [[nodiscard]] std::unordered_map<std::string, Tensor> allocateAndValidateOutputs(
        const std::unordered_map<std::string, Tensor>& inputs,
        const std::unordered_map<std::string, Tensor>& preallocated_outputs,
        const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes,
        const TensorPlacement& placement) const;

    [[nodiscard]] CudaKernelScalarValue resolveScalar(const ScalarParamSpec& scalar,
                                                      const std::unordered_map<std::string, Tensor>& tensors) const;

    std::string name_;
    std::string source_;
    std::string entry_;
    std::vector<TensorParamSpec> inputs_;
    std::vector<OutputParamSpec> outputs_;
    std::vector<ScalarParamSpec> scalars_;
    LaunchFn launch_fn_;
    bool use_fast_math_ = false;
};

}  // namespace ThorImplementation
