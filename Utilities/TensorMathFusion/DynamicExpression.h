#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Utilities/TensorMathFusion/CompiledEquation.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {

struct DynamicExpressionBuild {
    std::shared_ptr<FusedEquation> equation;
    std::unordered_map<std::string, Tensor> stamp_inputs;
    std::unordered_map<std::string, TensorScalarBinding> tensor_scalar_inputs;
    std::unordered_map<std::string, Tensor> preallocated_outputs;
    std::unordered_map<std::string, std::vector<uint64_t>> requested_output_shapes;
};

class PreparedDynamicExpression {
   public:
    using TensorMap = std::unordered_map<std::string, Tensor>;
    using TensorScalarMap = std::unordered_map<std::string, TensorScalarBinding>;
    using ShapeMap = std::unordered_map<std::string, std::vector<uint64_t>>;

    PreparedDynamicExpression(DynamicExpressionBuild build, const Stream& stream) : build_(std::move(build)), stream_(stream) {
        if (!build_.equation) {
            throw std::invalid_argument("PreparedDynamicExpression requires a non-null equation.");
        }

        validateTensorMap(build_.stamp_inputs, stream_, true, "stamp input");
        validateTensorScalarMap(build_.tensor_scalar_inputs, stream_, "tensor scalar input");
        validateTensorMap(build_.preallocated_outputs, stream_, false, "preallocated output");
    }

    [[nodiscard]] StampedExecutionPlan stamp() const { return stamp({}, {}); }

    [[nodiscard]] StampedExecutionPlan stamp(const TensorMap& preallocated_outputs_override,
                                             const ShapeMap& requested_output_shapes_override = {}) const {
        validateTensorMap(preallocated_outputs_override, stream_, false, "preallocated output override");

        TensorMap final_preallocated_outputs = build_.preallocated_outputs;
        for (const auto& [name, tensor] : preallocated_outputs_override) {
            final_preallocated_outputs[name] = tensor;
        }

        ShapeMap final_requested_output_shapes = build_.requested_output_shapes;
        for (const auto& [name, shape] : requested_output_shapes_override) {
            final_requested_output_shapes[name] = shape;
        }

        return build_.equation->stamp(
            build_.stamp_inputs, stream_, build_.tensor_scalar_inputs, final_preallocated_outputs, final_requested_output_shapes);
    }

    [[nodiscard]] PreparedDynamicExpression compileBackward(const std::vector<std::string>& wrt_names = {},
                                                            const std::optional<std::string>& upstream_input_name = std::nullopt,
                                                            bool accumulate_grad_outputs = false,
                                                            const TensorMap& additional_inputs = {},
                                                            const TensorScalarMap& additional_tensor_scalar_inputs = {},
                                                            const TensorMap& preallocated_grad_outputs = {},
                                                            const ShapeMap& requested_grad_output_shapes = {}) const {
        auto backward_equation =
            std::make_shared<FusedEquation>(build_.equation->compileBackward(wrt_names, upstream_input_name, accumulate_grad_outputs));

        return PreparedDynamicExpression(
            DynamicExpressionBuild{
                backward_equation,
                mergeTensorMaps(build_.stamp_inputs, additional_inputs, "backward additional input"),
                mergeTensorScalarMaps(
                    build_.tensor_scalar_inputs, additional_tensor_scalar_inputs, "backward additional tensor scalar input"),
                preallocated_grad_outputs,
                requested_grad_output_shapes,
            },
            stream_);
    }

    [[nodiscard]] PreparedDynamicExpression compileBackward(
        const std::vector<std::string>& wrt_names,
        const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
        bool accumulate_grad_outputs = false,
        const TensorMap& additional_inputs = {},
        const TensorScalarMap& additional_tensor_scalar_inputs = {},
        const TensorMap& preallocated_grad_outputs = {},
        const ShapeMap& requested_grad_output_shapes = {}) const {
        auto backward_equation = std::make_shared<FusedEquation>(
            build_.equation->compileBackward(wrt_names, upstream_input_names_by_output, accumulate_grad_outputs));

        return PreparedDynamicExpression(
            DynamicExpressionBuild{
                backward_equation,
                mergeTensorMaps(build_.stamp_inputs, additional_inputs, "backward additional input"),
                mergeTensorScalarMaps(
                    build_.tensor_scalar_inputs, additional_tensor_scalar_inputs, "backward additional tensor scalar input"),
                preallocated_grad_outputs,
                requested_grad_output_shapes,
            },
            stream_);
    }

    [[nodiscard]] StampedExecutionPlan stampBackward(const std::vector<std::string>& wrt_names = {},
                                                     const std::optional<std::string>& upstream_input_name = std::nullopt,
                                                     bool accumulate_grad_outputs = false,
                                                     const TensorMap& additional_inputs = {},
                                                     const TensorScalarMap& additional_tensor_scalar_inputs = {},
                                                     const TensorMap& preallocated_grad_outputs = {},
                                                     const ShapeMap& requested_grad_output_shapes = {}) const {
        return compileBackward(wrt_names,
                               upstream_input_name,
                               accumulate_grad_outputs,
                               additional_inputs,
                               additional_tensor_scalar_inputs,
                               preallocated_grad_outputs,
                               requested_grad_output_shapes)
            .stamp();
    }

    [[nodiscard]] StampedExecutionPlan stampBackward(const std::vector<std::string>& wrt_names,
                                                     const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
                                                     bool accumulate_grad_outputs = false,
                                                     const TensorMap& additional_inputs = {},
                                                     const TensorScalarMap& additional_tensor_scalar_inputs = {},
                                                     const TensorMap& preallocated_grad_outputs = {},
                                                     const ShapeMap& requested_grad_output_shapes = {}) const {
        return compileBackward(wrt_names,
                               upstream_input_names_by_output,
                               accumulate_grad_outputs,
                               additional_inputs,
                               additional_tensor_scalar_inputs,
                               preallocated_grad_outputs,
                               requested_grad_output_shapes)
            .stamp();
    }

    [[nodiscard]] const FusedEquation& equation() const { return *build_.equation; }

    [[nodiscard]] const TensorMap& stampInputs() const { return build_.stamp_inputs; }

    [[nodiscard]] const TensorScalarMap& tensorScalarInputs() const { return build_.tensor_scalar_inputs; }

    [[nodiscard]] const TensorMap& preallocatedOutputs() const { return build_.preallocated_outputs; }

    [[nodiscard]] const ShapeMap& requestedOutputShapes() const { return build_.requested_output_shapes; }

   private:
    static void validateTensorMap(const TensorMap& tensors, const Stream& stream, bool require_non_empty, const std::string& what) {
        if (require_non_empty && tensors.empty()) {
            throw std::invalid_argument("PreparedDynamicExpression requires at least one " + what + ".");
        }

        for (const auto& [name, tensor] : tensors) {
            if (!tensor.isInitialized()) {
                throw std::invalid_argument("PreparedDynamicExpression " + what + " '" + name + "' is not initialized.");
            }

            const auto placement = tensor.getPlacement();
            if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::invalid_argument("PreparedDynamicExpression " + what + " '" + name + "' is not on GPU.");
            }
            if (placement.getDeviceNum() != stream.getGpuNum()) {
                throw std::invalid_argument("PreparedDynamicExpression " + what + " '" + name + "' is on a different GPU than the stream.");
            }
        }
    }

    static void validateTensorScalarMap(const TensorScalarMap& bindings, const Stream& stream, const std::string& what) {
        for (const auto& [name, binding] : bindings) {
            if (!binding.buffer.isInitialized()) {
                throw std::invalid_argument("PreparedDynamicExpression " + what + " '" + name + "' buffer is not initialized.");
            }

            const auto placement = binding.buffer.getPlacement();
            if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::invalid_argument("PreparedDynamicExpression " + what + " '" + name + "' buffer is not on GPU.");
            }
            if (placement.getDeviceNum() != stream.getGpuNum()) {
                throw std::invalid_argument("PreparedDynamicExpression " + what + " '" + name +
                                            "' buffer is on a different GPU than the stream.");
            }
        }
    }

    static TensorMap mergeTensorMaps(const TensorMap& base, const TensorMap& extra, const std::string& extra_what) {
        TensorMap merged = base;
        for (const auto& [name, tensor] : extra) {
            auto [it, inserted] = merged.emplace(name, tensor);
            if (!inserted) {
                throw std::invalid_argument("PreparedDynamicExpression duplicate name '" + name + "' in " + extra_what + ".");
            }
        }
        return merged;
    }

    static TensorScalarMap mergeTensorScalarMaps(const TensorScalarMap& base, const TensorScalarMap& extra, const std::string& extra_what) {
        TensorScalarMap merged = base;
        for (const auto& [name, binding] : extra) {
            auto [it, inserted] = merged.emplace(name, binding);
            if (!inserted) {
                throw std::invalid_argument("PreparedDynamicExpression duplicate name '" + name + "' in " + extra_what + ".");
            }
        }
        return merged;
    }

    DynamicExpressionBuild build_;
    Stream stream_;
};

class DynamicExpression {
   public:
    using TensorMap = std::unordered_map<std::string, Tensor>;
    using BuilderFn = std::function<DynamicExpressionBuild(const TensorMap& inputs, Stream& stream)>;

    explicit DynamicExpression(BuilderFn builder) : builder_(std::move(builder)) {
        if (!builder_) {
            throw std::invalid_argument("DynamicExpression requires a non-empty builder.");
        }
    }

    [[nodiscard]] PreparedDynamicExpression prepare(const TensorMap& inputs, Stream& stream) const {
        validateInputs(inputs, stream);
        return PreparedDynamicExpression(builder_(inputs, stream), stream);
    }

    [[nodiscard]] StampedExecutionPlan stamp(const TensorMap& inputs, Stream& stream) const { return prepare(inputs, stream).stamp(); }

    [[nodiscard]] StampedExecutionPlan stamp(
        const TensorMap& inputs,
        Stream& stream,
        const TensorMap& preallocated_outputs,
        const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes = {}) const {
        return prepare(inputs, stream).stamp(preallocated_outputs, requested_output_shapes);
    }

    [[nodiscard]] StampedExecutionPlan stampBackward(
        const TensorMap& inputs,
        Stream& stream,
        const std::vector<std::string>& wrt_names = {},
        const std::optional<std::string>& upstream_input_name = std::nullopt,
        bool accumulate_grad_outputs = false,
        const TensorMap& additional_inputs = {},
        const std::unordered_map<std::string, TensorScalarBinding>& additional_tensor_scalar_inputs = {},
        const TensorMap& preallocated_grad_outputs = {},
        const std::unordered_map<std::string, std::vector<uint64_t>>& requested_grad_output_shapes = {}) const {
        return prepare(inputs, stream)
            .stampBackward(wrt_names,
                           upstream_input_name,
                           accumulate_grad_outputs,
                           additional_inputs,
                           additional_tensor_scalar_inputs,
                           preallocated_grad_outputs,
                           requested_grad_output_shapes);
    }

    [[nodiscard]] StampedExecutionPlan stampBackward(
        const TensorMap& inputs,
        Stream& stream,
        const std::vector<std::string>& wrt_names,
        const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
        bool accumulate_grad_outputs = false,
        const TensorMap& additional_inputs = {},
        const std::unordered_map<std::string, TensorScalarBinding>& additional_tensor_scalar_inputs = {},
        const TensorMap& preallocated_grad_outputs = {},
        const std::unordered_map<std::string, std::vector<uint64_t>>& requested_grad_output_shapes = {}) const {
        return prepare(inputs, stream)
            .stampBackward(wrt_names,
                           upstream_input_names_by_output,
                           accumulate_grad_outputs,
                           additional_inputs,
                           additional_tensor_scalar_inputs,
                           preallocated_grad_outputs,
                           requested_grad_output_shapes);
    }

   private:
    static void validateInputs(const TensorMap& inputs, Stream& stream) {
        if (inputs.empty()) {
            throw std::invalid_argument("DynamicExpression requires at least one input tensor.");
        }

        const auto firstPlacement = inputs.begin()->second.getPlacement();
        const bool onGpu = firstPlacement.getMemDevice() == TensorPlacement::MemDevices::GPU;
        if (!onGpu) {
            throw std::invalid_argument("DynamicExpression currently requires GPU input tensors.");
        }

        const int32_t gpuNum = firstPlacement.getDeviceNum();

        for (const auto& [name, tensor] : inputs) {
            if (!tensor.isInitialized()) {
                throw std::invalid_argument("DynamicExpression input tensor '" + name + "' is not initialized.");
            }

            const auto placement = tensor.getPlacement();
            if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::invalid_argument("DynamicExpression input tensor '" + name + "' is not on GPU.");
            }
            if (placement.getDeviceNum() != gpuNum) {
                throw std::invalid_argument("DynamicExpression input tensor '" + name + "' is on a different GPU than the other inputs.");
            }
        }

        if (stream.getGpuNum() != gpuNum) {
            throw std::runtime_error("DynamicExpression stream GPU does not match input tensor GPU.");
        }
    }

    BuilderFn builder_;
};

}  // namespace ThorImplementation
