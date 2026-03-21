#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/TensorMathFusion/BroadcastStructs.h"
#include "Utilities/TensorMathFusion/EquationRunner.h"
#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {

struct CompiledStageOutput {
    std::string name;
    uint32_t local_node_idx = UINT32_MAX;
    uint32_t value_id = UINT32_MAX;
};

struct CompiledExecutionStage {
    enum class Kind { FusedKernel, Reduction };
    static std::string kindToString(const Kind kind) {
        switch (kind) {
            case Kind::FusedKernel:
                return "FusedKernel";
            case Kind::Reduction:
                return "Reduction";
        }
        return "<unknown>";
    }

    const Kind kind;

    PhysicalExpression expr;

    const std::shared_ptr<CompiledEquation> flat = nullptr;
    const std::shared_ptr<CompiledReduction> reduction = nullptr;

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
};

struct CompiledOutputs {
    EquationSignature signature;
    bool broadcast_support = false;

    std::vector<CompiledExecutionStage> stages;
    std::vector<CompiledStageOutput> final_outputs;
};

class FusedEquation {
   public:
    static FusedEquation compile(const PhysicalExpression& expr,
                                 TensorDescriptor::DataType dtype,
                                 int device_num,
                                 bool use_fast_math = false);

    static FusedEquation compile(const PhysicalOutputs& outputs,
                                 TensorDescriptor::DataType dtype,
                                 int device_num,
                                 bool use_fast_math = false);

    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const Stream& stream,
                                             const std::vector<uint64_t>& requestedOutputShape = {}) const;

    [[nodiscard]] StampedExecutionPlan stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                             const Stream& stream,
                                             const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const;

    void run(const Tensor& input, Tensor& output, Stream& stream) const;
    void run(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const;
    void run(const Tensor& input, std::unordered_map<std::string, Tensor>& outputs, Stream& stream) const;
    void run(const std::unordered_map<std::string, Tensor>& inputs, std::unordered_map<std::string, Tensor>& outputs, Stream& stream) const;

    static Tensor createDeviceBroadcastInfo(const std::vector<Tensor>& inputs,
                                            const std::vector<uint64_t>& outputDimensions,
                                            Stream stream);

   private:
    explicit FusedEquation(std::shared_ptr<CompiledOutputs> compiled_outputs, std::vector<NamedInput> root_inputs)
        : compiled_outputs(std::move(compiled_outputs)), root_inputs(std::move(root_inputs)) {}

    [[nodiscard]] std::shared_ptr<StampedEquation> stampEquation(const std::shared_ptr<CompiledEquation>& compiledEquation,
                                                                 std::vector<Tensor>& inputs,
                                                                 std::vector<Tensor>& outputs,
                                                                 const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedReduction> stampReduction(const std::shared_ptr<CompiledReduction>& compiledReduction,
                                                                   Tensor& input,
                                                                   const Stream& stream) const;

    static bool resolveLayout(std::vector<Tensor>& inputs, std::vector<uint64_t>& outputDimensions);

    [[nodiscard]] std::unordered_map<uint32_t, Tensor> bindRootInputs(const std::unordered_map<std::string, Tensor>& namedInputs) const;
    const std::shared_ptr<CompiledOutputs> compiled_outputs;
    const std::vector<NamedInput> root_inputs;
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
