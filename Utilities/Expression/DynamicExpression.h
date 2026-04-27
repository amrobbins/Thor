#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Utilities/Expression/CompiledEquation.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

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

    [[nodiscard]] FusedEquation::ParameterFanOverrideMap getParameterFanOverrides(
        const std::unordered_set<std::string>& parameter_names) const {
        return build_.equation->getParameterFanOverrides(
            build_.stamp_inputs, parameter_names, build_.tensor_scalar_inputs, build_.requested_output_shapes);
    }

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
    using BuilderFn = std::function<DynamicExpressionBuild(const TensorMap& inputs, const TensorMap& outputs, Stream& stream)>;

    explicit DynamicExpression(BuilderFn builder) : builder_(std::move(builder)) {
        if (!builder_) {
            throw std::invalid_argument("DynamicExpression requires a non-empty builder.");
        }
    }

    DynamicExpression(std::vector<std::string> expected_input_names, std::vector<std::string> expected_output_names, BuilderFn builder)
        : expected_input_names_(std::move(expected_input_names)),
          expected_output_names_(std::move(expected_output_names)),
          builder_(std::move(builder)) {
        if (!builder_) {
            throw std::invalid_argument("DynamicExpression requires a non-empty builder.");
        }
        validateExpectedNames(expected_input_names_, "input");
        validateExpectedNames(expected_output_names_, "output");
    }

    [[nodiscard]] const std::vector<std::string>& getExpectedInputNames() const { return expected_input_names_; }
    [[nodiscard]] const std::vector<std::string>& getExpectedOutputNames() const { return expected_output_names_; }

    [[nodiscard]] DynamicExpressionBuild build(const TensorMap& inputs, const TensorMap& outputs, Stream& stream) const {
        validateExpectedTensorNames(inputs, expected_input_names_, "input");
        if (!outputs.empty()) {
            validateExpectedTensorNames(outputs, expected_output_names_, "output");
        }

        DynamicExpressionBuild build = builder_(inputs, outputs, stream);
        validateBuild(build, outputs);
        return build;
    }

    [[nodiscard]] PreparedDynamicExpression prepare(const TensorMap& inputs, const TensorMap& outputs, Stream& stream) const {
        validateExpectedTensorNames(inputs, expected_input_names_, "input");
        if (!outputs.empty()) {
            validateExpectedTensorNames(outputs, expected_output_names_, "output");
        }
        validateInputs(inputs, stream);
        validateOutputs(outputs, stream);

        DynamicExpressionBuild build = builder_(inputs, outputs, stream);
        validateBuild(build, outputs);
        return PreparedDynamicExpression(std::move(build), stream);
    }

    [[nodiscard]] StampedExecutionPlan stamp(const TensorMap& inputs, const TensorMap& outputs, Stream& stream) const {
        return prepare(inputs, outputs, stream).stamp();
    }

    [[nodiscard]] StampedExecutionPlan stamp(
        const TensorMap& inputs,
        const TensorMap& outputs,
        Stream& stream,
        const TensorMap& preallocated_outputs,
        const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes = {}) const {
        return prepare(inputs, outputs, stream).stamp(preallocated_outputs, requested_output_shapes);
    }

    [[nodiscard]] StampedExecutionPlan stampBackward(
        const TensorMap& inputs,
        const TensorMap& outputs,
        Stream& stream,
        const std::vector<std::string>& wrt_names = {},
        const std::optional<std::string>& upstream_input_name = std::nullopt,
        bool accumulate_grad_outputs = false,
        const TensorMap& additional_inputs = {},
        const std::unordered_map<std::string, TensorScalarBinding>& additional_tensor_scalar_inputs = {},
        const TensorMap& preallocated_grad_outputs = {},
        const std::unordered_map<std::string, std::vector<uint64_t>>& requested_grad_output_shapes = {}) const {
        return prepare(inputs, outputs, stream)
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
        const TensorMap& outputs,
        Stream& stream,
        const std::vector<std::string>& wrt_names,
        const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
        bool accumulate_grad_outputs = false,
        const TensorMap& additional_inputs = {},
        const std::unordered_map<std::string, TensorScalarBinding>& additional_tensor_scalar_inputs = {},
        const TensorMap& preallocated_grad_outputs = {},
        const std::unordered_map<std::string, std::vector<uint64_t>>& requested_grad_output_shapes = {}) const {
        return prepare(inputs, outputs, stream)
            .stampBackward(wrt_names,
                           upstream_input_names_by_output,
                           accumulate_grad_outputs,
                           additional_inputs,
                           additional_tensor_scalar_inputs,
                           preallocated_grad_outputs,
                           requested_grad_output_shapes);
    }

   private:
    static void validateInputs(const TensorMap& inputs, const Stream& stream) {
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

    static void validateOutputs(const TensorMap& outputs, const Stream& stream) {
        for (const auto& [name, tensor] : outputs) {
            if (!tensor.isInitialized()) {
                throw std::invalid_argument("DynamicExpression output tensor '" + name + "' is not initialized.");
            }

            const auto placement = tensor.getPlacement();
            if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::invalid_argument("DynamicExpression output tensor '" + name + "' is not on GPU.");
            }
            if (placement.getDeviceNum() != stream.getGpuNum()) {
                throw std::invalid_argument("DynamicExpression output tensor '" + name + "' is on a different GPU than the stream.");
            }
        }
    }

    static std::set<std::string> tensorMapKeys(const TensorMap& tensors) {
        std::set<std::string> names;
        for (const auto& [name, _] : tensors) {
            names.insert(name);
        }
        return names;
    }

    static std::set<std::string> shapeMapKeys(const std::unordered_map<std::string, std::vector<uint64_t>>& shapes) {
        std::set<std::string> names;
        for (const auto& [name, _] : shapes) {
            names.insert(name);
        }
        return names;
    }

    static std::set<std::string> vectorToSet(const std::vector<std::string>& names) {
        return std::set<std::string>(names.begin(), names.end());
    }

    static std::string joinNames(const std::set<std::string>& names) {
        if (names.empty()) {
            return "<none>";
        }

        std::ostringstream oss;
        bool first = true;
        for (const auto& name : names) {
            if (!first) {
                oss << ", ";
            }
            oss << name;
            first = false;
        }
        return oss.str();
    }

    static std::set<std::string> setDifference(const std::set<std::string>& lhs, const std::set<std::string>& rhs) {
        std::set<std::string> difference;
        for (const auto& name : lhs) {
            if (!rhs.contains(name)) {
                difference.insert(name);
            }
        }
        return difference;
    }

    static void validateBuild(const DynamicExpressionBuild& build, const TensorMap& requested_outputs) {
        if (!build.equation) {
            throw std::invalid_argument("DynamicExpression builder returned a null equation.");
        }

        const std::set<std::string> built_output_names = vectorToSet(build.equation->getOutputNames());
        const std::set<std::string> requested_output_names = tensorMapKeys(requested_outputs);
        const std::set<std::string> preallocated_output_names = tensorMapKeys(build.preallocated_outputs);
        const std::set<std::string> requested_output_shape_names = shapeMapKeys(build.requested_output_shapes);

        if (!requested_output_names.empty() && built_output_names != requested_output_names) {
            throw std::invalid_argument("DynamicExpression builder returned equation outputs {" + joinNames(built_output_names) +
                                        "} but caller requested outputs {" + joinNames(requested_output_names) +
                                        "}. Prune the equation outputs in the builder or provide matching outputs.");
        }

        const std::set<std::string> unknown_preallocated_output_names = setDifference(preallocated_output_names, built_output_names);
        if (!unknown_preallocated_output_names.empty()) {
            throw std::invalid_argument("DynamicExpression builder returned preallocated outputs {" + joinNames(preallocated_output_names) +
                                        "} but the equation only produces {" + joinNames(built_output_names) +
                                        "}. Unknown preallocated outputs: {" + joinNames(unknown_preallocated_output_names) + "}.");
        }

        const std::set<std::string> unknown_requested_output_shape_names = setDifference(requested_output_shape_names, built_output_names);
        if (!unknown_requested_output_shape_names.empty()) {
            throw std::invalid_argument("DynamicExpression builder returned requested output shapes for names {" +
                                        joinNames(requested_output_shape_names) + "} but the equation only produces {" +
                                        joinNames(built_output_names) + "}. Unknown requested output shape names: {" +
                                        joinNames(unknown_requested_output_shape_names) + "}.");
        }

        if (!requested_output_names.empty()) {
            if (preallocated_output_names != requested_output_names) {
                throw std::invalid_argument(
                    "DynamicExpression builder must return preallocated outputs exactly matching the caller-requested outputs. "
                    "Requested outputs: {" +
                    joinNames(requested_output_names) +
                    "} "
                    "Builder returned preallocated outputs: {" +
                    joinNames(preallocated_output_names) + "}.");
            }

            for (const auto& [name, requested_tensor] : requested_outputs) {
                const auto it = build.preallocated_outputs.find(name);
                if (it == build.preallocated_outputs.end()) {
                    throw std::invalid_argument("DynamicExpression builder did not return caller-requested output tensor '" + name + "'.");
                }
                if (!(it->second == requested_tensor)) {
                    throw std::invalid_argument("DynamicExpression builder returned a different tensor for caller-requested output '" +
                                                name + "'. Reuse the caller-provided tensor or prune the output in the builder.");
                }
            }
        }
    }

    static void validateExpectedNames(const std::vector<std::string>& names, const std::string& what) {
        std::set<std::string> seen;
        for (const auto& name : names) {
            if (name.empty()) {
                throw std::invalid_argument("DynamicExpression expected " + what + " name cannot be empty.");
            }
            if (!seen.insert(name).second) {
                throw std::invalid_argument("DynamicExpression duplicate expected " + what + " name: " + name);
            }
        }
    }

    static void validateExpectedTensorNames(const TensorMap& tensors,
                                            const std::vector<std::string>& expected_names,
                                            const std::string& what) {
        if (expected_names.empty()) {
            return;
        }

        const std::set<std::string> actual = tensorMapKeys(tensors);
        const std::set<std::string> expected = vectorToSet(expected_names);
        if (actual != expected) {
            throw std::invalid_argument("DynamicExpression " + what + " name mismatch. Expected {" + joinNames(expected) + "}, got {" +
                                        joinNames(actual) + "}.");
        }
    }

    std::vector<std::string> expected_input_names_;
    std::vector<std::string> expected_output_names_;
    BuilderFn builder_;
};

}  // namespace ThorImplementation
