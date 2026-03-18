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
    Kind kind;

    std::shared_ptr<CompiledEquation> flat = nullptr;
    std::shared_ptr<CompiledEquation> broadcast = nullptr;
    std::shared_ptr<CompiledReduction> reduction = nullptr;

    std::vector<uint32_t> input_value_ids;
    std::vector<CompiledStageOutput> outputs;

    explicit CompiledExecutionStage(std::shared_ptr<CompiledEquation> flat,
                                    std::shared_ptr<CompiledEquation> broadcast,
                                    std::vector<uint32_t> input_value_ids,
                                    std::vector<CompiledStageOutput> outputs)
        : kind(Kind::FusedKernel),
          flat(std::move(flat)),
          broadcast(std::move(broadcast)),
          input_value_ids(std::move(input_value_ids)),
          outputs(std::move(outputs)) {}

    explicit CompiledExecutionStage(std::shared_ptr<CompiledReduction> reduction,
                                    std::vector<uint32_t> input_value_ids,
                                    std::vector<CompiledStageOutput> outputs)
        : kind(Kind::Reduction),
          reduction(std::move(reduction)),
          input_value_ids(std::move(input_value_ids)),
          outputs(std::move(outputs)) {}

    CompiledExecutionStage() = default;
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

    void run(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const;

   private:
    explicit FusedEquation(std::shared_ptr<CompiledOutputs> compiled_outputs, std::vector<NamedInput> root_inputs)
        : compiled_outputs(std::move(compiled_outputs)), root_inputs(std::move(root_inputs)) {}

    [[nodiscard]] std::shared_ptr<StampedEquation> stampEquation(const std::shared_ptr<CompiledEquation>& compiledEquation,
                                                                 std::vector<Tensor>& inputs,
                                                                 std::vector<Tensor>& outputs,
                                                                 const Stream& stream) const;

    [[nodiscard]] std::shared_ptr<StampedEquation> stampEquation(const std::shared_ptr<CompiledEquation>& compiledEquation,
                                                                 std::vector<Tensor>& inputs,
                                                                 std::vector<Tensor>& outputs,
                                                                 const Stream& stream,
                                                                 const Tensor& deviceBroadcastInfo) const;

    [[nodiscard]] std::shared_ptr<StampedReduction> stampReduction(const std::shared_ptr<CompiledReduction>& compiledReduction,
                                                                   Tensor& input,
                                                                   const Stream& stream) const;

    static bool resolveLayout(std::vector<Tensor>& inputs, std::vector<uint64_t>& outputDimensions);
    static Tensor createDeviceBroadcastInfo(const std::vector<Tensor>& inputs,
                                            const std::vector<uint64_t>& outputDimensions,
                                            Stream stream);

    [[nodiscard]] std::unordered_map<uint32_t, Tensor> bindRootInputs(const std::unordered_map<std::string, Tensor>& namedInputs) const;

    const std::shared_ptr<CompiledOutputs> compiled_outputs;
    const std::vector<NamedInput> root_inputs;
};

}  // namespace ThorImplementation
