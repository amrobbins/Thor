#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/Cache/LruCache.h"
#include "Utilities/TensorMathFusion/EquationRunner.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/SqueezeAxes.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {

struct CompiledStageOutput {
    std::string name;
    uint32_t local_node_idx = UINT32_MAX;
    uint32_t value_id = UINT32_MAX;
};

struct CompiledExecutionStage {
    enum class Kind { FusedKernel, Reduction, ArgMinMax, ReduceMinMaxBackward };
    static std::string kindToString(const Kind kind) {
        switch (kind) {
            case Kind::FusedKernel:
                return "FusedKernel";
            case Kind::Reduction:
                return "Reduction";
            case Kind::ArgMinMax:
                return "ArgMinMax";
            case Kind::ReduceMinMaxBackward:
                return "ReduceMinMaxBackward";
        }
        return "<unknown>";
    }

    const Kind kind;

    PhysicalExpression expr;

    const std::shared_ptr<CompiledEquation> flat = nullptr;
    const std::shared_ptr<CompiledReduction> reduction = nullptr;
    const std::shared_ptr<CompiledArgMinMax> arg_minmax = nullptr;
    const std::shared_ptr<CompiledReduceMinMaxBackward> reduce_minmax_backward = nullptr;

    const std::vector<uint32_t> input_value_ids;
    const std::vector<CompiledStageOutput> outputs;

    CompiledExecutionStage(const PhysicalExpression& expr,
                           const std::shared_ptr<CompiledEquation>& flat,
                           std::vector<uint32_t> input_value_ids,
                           std::vector<CompiledStageOutput> outputs)
        : kind(Kind::FusedKernel), expr(expr), flat(flat), input_value_ids(std::move(input_value_ids)), outputs(std::move(outputs)) {}

    CompiledExecutionStage(const std::shared_ptr<CompiledReduction>& reduction,
                           std::vector<uint32_t> input_value_ids,
                           std::vector<CompiledStageOutput> outputs)
        : kind(Kind::Reduction), reduction(reduction), input_value_ids(std::move(input_value_ids)), outputs(std::move(outputs)) {}

    CompiledExecutionStage(const std::shared_ptr<CompiledArgMinMax>& arg_minmax,
                           std::vector<uint32_t> input_value_ids,
                           std::vector<CompiledStageOutput> outputs)
        : kind(Kind::ArgMinMax), arg_minmax(arg_minmax), input_value_ids(std::move(input_value_ids)), outputs(std::move(outputs)) {}

    CompiledExecutionStage(const std::shared_ptr<CompiledReduceMinMaxBackward>& reduce_minmax_backward,
                           std::vector<uint32_t> input_value_ids,
                           std::vector<CompiledStageOutput> outputs)
        : kind(Kind::ReduceMinMaxBackward),
          reduce_minmax_backward(reduce_minmax_backward),
          input_value_ids(std::move(input_value_ids)),
          outputs(std::move(outputs)) {}
};

struct CompiledOutputs {
    EquationSignature signature;
    bool broadcast_support = false;

    std::vector<CompiledExecutionStage> stages;
    std::vector<CompiledStageOutput> final_outputs;
};

struct RuntimeDTypeKey {
    std::vector<TensorDescriptor::DataType> root_input_dtypes;

    bool operator==(const RuntimeDTypeKey& other) const = default;
};

struct RuntimeDTypeKeyHash {
    size_t operator()(const RuntimeDTypeKey& k) const noexcept {
        size_t h = 0;

        auto combine = [&](size_t value) noexcept { h ^= value + 0x9e3779b9 + (h << 6) + (h >> 2); };

        combine(std::hash<size_t>{}(k.root_input_dtypes.size()));
        for (TensorDescriptor::DataType dtype : k.root_input_dtypes) {
            combine(std::hash<int>{}(static_cast<int>(dtype)));
        }

        return h;
    }
};

struct RuntimeShapeKey {
    RuntimeDTypeKey dtype_key;
    std::vector<std::vector<uint64_t>> root_input_dims;

    bool operator==(const RuntimeShapeKey& other) const = default;
};

struct RuntimeShapeKeyHash {
    size_t operator()(const RuntimeShapeKey& k) const noexcept {
        size_t h = RuntimeDTypeKeyHash{}(k.dtype_key);

        auto combine = [&](size_t value) noexcept { h ^= value + 0x9e3779b9 + (h << 6) + (h >> 2); };

        combine(std::hash<size_t>{}(k.root_input_dims.size()));
        for (const std::vector<uint64_t>& dims : k.root_input_dims) {
            combine(std::hash<size_t>{}(dims.size()));
            for (uint64_t d : dims) {
                combine(std::hash<uint64_t>{}(d));
            }
        }

        return h;
    }
};

struct PreparedConvenienceRunStage {
    std::shared_ptr<CompiledEquation> compiled_equation;
    std::vector<std::vector<uint64_t>> expected_output_dims;  // aligned with stage.outputs
};

struct PreparedConvenienceRunPlan {
    std::shared_ptr<CompiledOutputs> compiled_outputs;
    std::vector<PreparedConvenienceRunStage> stages;          // aligned with compiled_outputs->stages
    std::vector<std::string> expected_output_names_in_order;  // stable convenience-run validation order
};

struct BackwardEquationConfig {
    PhysicalOutputs forward_outputs_template;
    std::vector<std::string> wrt_names;
    std::optional<std::unordered_map<std::string, std::string>> upstream_input_names_by_output;
    bool accumulate_grad_outputs = false;
};

class FusedEquation {
   public:
    static FusedEquation compile(const PhysicalExpression& expr, int device_num, bool use_fast_math = false);

    static FusedEquation compile(const PhysicalOutputs& outputs, int device_num, bool use_fast_math = false);

    [[nodiscard]] FusedEquation compileBackward(const std::vector<std::string>& wrt_names = {},
                                                const std::optional<std::string>& upstream_input_name = std::nullopt,
                                                bool accumulate_grad_outputs = false) const;

    [[nodiscard]] FusedEquation compileBackward(const std::vector<std::string>& wrt_names,
                                                const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
                                                bool accumulate_grad_outputs = false) const;

    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const Stream& stream,
                                             const std::vector<uint64_t>& requestedOutputShape = {}) const;
    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
                                             const Stream& stream,
                                             const std::vector<uint64_t>& requestedOutputShape = {}) const;

    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const Stream& stream,
                                             const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const;
    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
                                             const Stream& stream,
                                             const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const;

    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const std::unordered_map<std::string, Tensor>& outputs,
                                             const Stream& stream,
                                             const std::vector<uint64_t>& requestedOutputShape = {}) const;
    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
                                             const std::unordered_map<std::string, Tensor>& outputs,
                                             const Stream& stream,
                                             const std::vector<uint64_t>& requestedOutputShape = {}) const;

    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const std::unordered_map<std::string, Tensor>& preallocated_outputs,
                                             const Stream& stream,
                                             const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const;
    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
                                             const std::unordered_map<std::string, Tensor>& preallocated_outputs,
                                             const Stream& stream,
                                             const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const;

    void run(const Tensor& input, Tensor& output, Stream& stream) const;
    void run(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const;
    void run(const std::unordered_map<std::string, Tensor>& inputs,
             const std::unordered_map<std::string, float>& scalar_inputs,
             Tensor& output,
             Stream& stream) const;
    void run(const Tensor& input, std::unordered_map<std::string, Tensor>& outputs, Stream& stream) const;
    void run(const std::unordered_map<std::string, Tensor>& inputs, std::unordered_map<std::string, Tensor>& outputs, Stream& stream) const;
    void run(const std::unordered_map<std::string, Tensor>& inputs,
             const std::unordered_map<std::string, float>& scalar_inputs,
             std::unordered_map<std::string, Tensor>& outputs,
             Stream& stream) const;

    std::vector<std::string> getOutputNames() const;

    std::vector<uint64_t> getOutputShape(const Tensor& input) const;
    std::vector<uint64_t> getOutputShape(const std::unordered_map<std::string, Tensor>& inputs) const;
    std::unordered_map<std::string, std::vector<uint64_t>> getOutputShapes(const Tensor& input) const;
    std::unordered_map<std::string, std::vector<uint64_t>> getOutputShapes(const std::unordered_map<std::string, Tensor>& inputs) const;

    static bool resolveLayout(std::vector<Tensor>& inputs, std::vector<uint64_t>& outputDimensions);

   private:
    explicit FusedEquation(PhysicalOutputs outputs_template,
                           int device_num,
                           bool use_fast_math,
                           EquationSignature base_signature,
                           std::optional<BackwardEquationConfig> backward_config = std::nullopt)
        : outputs_template(std::move(outputs_template)),
          root_inputs(this->outputs_template.expr ? this->outputs_template.expr->inputs : std::vector<NamedInput>{}),
          device_num(device_num),
          use_fast_math(use_fast_math),
          base_signature(std::move(base_signature)),
          backward_config(std::move(backward_config)),
          compiled_outputs_runtime_cache(
              std::make_shared<LruCacheThreadSafe<RuntimeDTypeKey, std::shared_ptr<CompiledOutputs>, RuntimeDTypeKeyHash>>(128)),
          compiled_outputs_shape_cache(
              std::make_shared<LruCacheThreadSafe<RuntimeShapeKey, std::shared_ptr<CompiledOutputs>, RuntimeShapeKeyHash>>(128)),
          convenience_run_plan_cache(
              std::make_shared<LruCacheThreadSafe<RuntimeShapeKey, std::shared_ptr<PreparedConvenienceRunPlan>, RuntimeShapeKeyHash>>(
                  128)) {}

    [[nodiscard]] std::shared_ptr<StampedEquation> stampEquation(const std::shared_ptr<CompiledEquation>& compiledEquation,
                                                                 std::vector<RuntimeInputValue>& inputs,
                                                                 std::vector<Tensor>& outputs,
                                                                 const Stream& stream) const;

    std::shared_ptr<StampedReduction> stampReduction(const std::shared_ptr<CompiledReduction>& compiledReduction,
                                                     Tensor& input,
                                                     const Stream& stream,
                                                     const std::vector<uint64_t>& requested_output_shape) const;

    [[nodiscard]] std::shared_ptr<StampedReduction> stampReduction(const std::shared_ptr<CompiledReduction>& compiledReduction,
                                                                   Tensor& input,
                                                                   const Optional<Tensor>& preallocatedOutput,
                                                                   const Stream& stream,
                                                                   const std::vector<uint64_t>& requested_output_shape) const;

    [[nodiscard]] std::shared_ptr<StampedArgMinMax> stampArgMinMax(const std::shared_ptr<CompiledArgMinMax>& compiledStage,
                                                                   Tensor& input,
                                                                   const Stream& stream,
                                                                   const std::vector<uint64_t>& requested_output_shape) const;

    [[nodiscard]] std::shared_ptr<StampedArgMinMax> stampArgMinMax(const std::shared_ptr<CompiledArgMinMax>& compiledStage,
                                                                   Tensor& input,
                                                                   const Optional<Tensor>& preallocatedOutput,
                                                                   const Stream& stream,
                                                                   const std::vector<uint64_t>& requested_output_shape) const;

    [[nodiscard]] std::shared_ptr<StampedReduceMinMaxBackward> stampReduceMinMaxBackward(
        const std::shared_ptr<CompiledReduceMinMaxBackward>& compiledStage,
        Tensor& input,
        Tensor& grad_output,
        const Optional<Tensor>& preallocatedOutput,
        const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedReduceMinMaxBackward> stampReduceMinMaxBackward(
        const std::shared_ptr<CompiledReduceMinMaxBackward>& compiledStage, Tensor& input, Tensor& grad_output, const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<CompiledOutputs> compileForInputs(const std::unordered_map<std::string, Tensor>& namedInputs,
                                                                    const std::unordered_map<std::string, float>& scalarInputs = {}) const;
    [[nodiscard]] std::shared_ptr<CompiledOutputs> compileForRootValues(
        const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) const;
    [[nodiscard]] PhysicalOutputs buildShapeSpecializedOutputs(const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) const;
    [[nodiscard]] std::shared_ptr<PreparedConvenienceRunPlan> prepareConvenienceRunPlan(
        const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) const;

    [[nodiscard]] static EquationSignature buildSignature(uint32_t num_inputs, int device_num, bool use_fast_math);

    [[nodiscard]] std::unordered_map<uint32_t, RuntimeInputValue> bindRootInputs(
        const std::unordered_map<std::string, Tensor>& namedInputs,
        const std::unordered_map<std::string, float>& scalar_inputs = {},
        const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs = {},
        const std::unordered_map<std::string, Tensor>* namedOutputs = nullptr) const;
    [[nodiscard]] std::unordered_map<uint32_t, RuntimeInputValue> bindRootInputsForCompilation(
        const std::unordered_map<std::string, Tensor>& namedInputs,
        const std::unordered_map<std::string, float>& scalar_inputs = {},
        const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs = {},
        const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes = {}) const;

    [[nodiscard]] std::unordered_map<std::string, std::vector<uint64_t>> makeSingleOutputRequestedShapeMap(
        const std::vector<uint64_t>& requestedOutputShape) const;

    const PhysicalOutputs outputs_template;
    const std::vector<NamedInput> root_inputs;
    const int device_num;
    const bool use_fast_math;
    const EquationSignature base_signature;
    const std::optional<BackwardEquationConfig> backward_config;

    // Forward equations compile per runtime dtype only. Shape-specialized backward equations
    // compile against runtime shapes so they use a separate cache.
    std::shared_ptr<LruCacheThreadSafe<RuntimeDTypeKey, std::shared_ptr<CompiledOutputs>, RuntimeDTypeKeyHash>>
        compiled_outputs_runtime_cache;
    std::shared_ptr<LruCacheThreadSafe<RuntimeShapeKey, std::shared_ptr<CompiledOutputs>, RuntimeShapeKeyHash>>
        compiled_outputs_shape_cache;
    std::shared_ptr<LruCacheThreadSafe<RuntimeShapeKey, std::shared_ptr<PreparedConvenienceRunPlan>, RuntimeShapeKeyHash>>
        convenience_run_plan_cache;
};

struct SpecializedBroadcastAxis {
    uint64_t dim = 1;
    uint64_t output_stride = 1;
    std::vector<uint64_t> input_strides;  // same order as used_input_slots
};

enum class SpecializedInputLoadKind { ScalarPack, NativeVector };

struct SpecializedBroadcastGroup {
    uint64_t numel = 0;
    std::vector<uint64_t> output_dims;
    std::vector<uint32_t> output_indices;
    std::vector<uint32_t> used_input_slots;  // sorted, local stage input slots
    std::vector<SpecializedInputLoadKind> used_input_load_kinds;
    std::vector<SpecializedBroadcastAxis> active_axes;
};

}  // namespace ThorImplementation
