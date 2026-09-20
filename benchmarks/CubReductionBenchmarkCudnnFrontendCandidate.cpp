#include "benchmarks/CubReductionBenchmarkCandidate.h"
#include "benchmarks/CubReductionBenchmarkCudnnFrontendLayout.h"
#include "Utilities/Common/CudnnFrontendPlan.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"

#include <cudnn_frontend.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionBenchmarking {
namespace {

namespace fe = cudnn_frontend;

constexpr int64_t UID_X = 1;
constexpr int64_t UID_Y = 2;

[[nodiscard]] bool isSupportedIoDtype(DataType dtype) {
    return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

[[nodiscard]] fe::DataType_t toFrontendDataType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return fe::DataType_t::HALF;
        case DataType::BF16:
            return fe::DataType_t::BFLOAT16;
        case DataType::FP32:
            return fe::DataType_t::FLOAT;
        default:
            throw std::invalid_argument("Unsupported cuDNN Frontend reduction dtype: "
                                        + TensorDescriptor::getElementTypeName(dtype));
    }
}

[[nodiscard]] std::vector<int64_t> checkedI64Vector(const std::vector<uint64_t>& values,
                                                     std::string_view what) {
    std::vector<int64_t> result;
    result.reserve(values.size());
    for (uint64_t value : values) {
        if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument("cuDNN Frontend reduction " + std::string(what)
                                        + " contains a value outside positive int64 range.");
        }
        result.push_back(static_cast<int64_t>(value));
    }
    return result;
}

void checkFrontendStatus(fe::error_t status, const std::string& message) {
    if (!status.is_good()) {
        throw std::runtime_error(message + ": " + status.get_message());
    }
}

[[nodiscard]] std::vector<int64_t> nhwcPackedStrides(const std::vector<int64_t>& dimensions) {
    if (dimensions.size() != 4) {
        throw std::invalid_argument("cuDNN Frontend reduction benchmark NHWC layout requires rank-4 tensors.");
    }
    const int64_t n = dimensions[0];
    const int64_t c = dimensions[1];
    const int64_t h = dimensions[2];
    const int64_t w = dimensions[3];
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
        throw std::invalid_argument("cuDNN Frontend reduction benchmark NHWC dimensions must be positive.");
    }
    if (c > std::numeric_limits<int64_t>::max() / w
        || c * w > std::numeric_limits<int64_t>::max() / h) {
        throw std::overflow_error("cuDNN Frontend reduction benchmark NHWC stride overflow.");
    }
    // cuDNN dimensions remain [N,C,H,W], while packed NHWC storage has C as
    // the unit-stride dimension: [H*W*C, 1, W*C, C].
    return {h * w * c, 1, w * c, c};
}

[[nodiscard]] CudnnFrontendGraphFactory makeReductionGraphFactory(DataType input_dtype,
                                                                   DataType output_dtype,
                                                                   std::vector<int64_t> input_dims,
                                                                   std::vector<int64_t> input_strides,
                                                                   std::vector<int64_t> output_dims,
                                                                   std::vector<int64_t> output_strides) {
    return [input_dtype,
            output_dtype,
            input_dims = std::move(input_dims),
            input_strides = std::move(input_strides),
            output_dims = std::move(output_dims),
            output_strides = std::move(output_strides)]() {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(toFrontendDataType(input_dtype))
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        auto x = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("reduction_x")
                                   .set_uid(UID_X)
                                   .set_dim(input_dims)
                                   .set_stride(input_strides)
                                   .set_data_type(toFrontendDataType(input_dtype)));

        auto reduced = graph->reduction(x,
                                        fe::graph::Reduction_attributes()
                                            .set_name("sum")
                                            .set_mode(fe::ReductionMode_t::ADD)
                                            .set_compute_data_type(fe::DataType_t::FLOAT));
        reduced->set_output(true)
            .set_uid(UID_Y)
            .set_dim(output_dims)
            .set_stride(output_strides)
            .set_data_type(toFrontendDataType(output_dtype));
        return graph;
    };
}

[[nodiscard]] CudnnFrontendPlanSelection selectAutotunedReductionPlan(const CudnnFrontendGraphFactory& graph_factory,
                                                                       Tensor& input,
                                                                       Tensor& output,
                                                                       const Stream& stream,
                                                                       std::string_view operation_name) {
    ScopedGpu scoped_gpu(stream.getGpuNum());
    std::shared_ptr<fe::graph::Graph> graph = graph_factory();
    if (!graph || graph.use_count() != 1) {
        throw std::runtime_error("cuDNN Frontend reduction selection requires a pristine operation-local graph.");
    }

    const std::string operation(operation_name);
    checkFrontendStatus(graph->validate(), "Failed to validate " + operation + " graph");
    checkFrontendStatus(graph->build_operation_graph(stream.getCudnnHandle()),
                        "Failed to build " + operation + " operation graph");
    checkFrontendStatus(graph->create_execution_plans({fe::HeurMode_t::A,
                                                        fe::HeurMode_t::B,
                                                        fe::HeurMode_t::FALLBACK}),
                        "Failed to enumerate " + operation + " execution plans");
    // Thor reductions require FP32 accumulation for these storage types. cuDNN's FLOAT compute type alone does not
    // exclude engines that advertise reduced-precision reduction, so filter those plans before support/build/autotune.
    graph->deselect_numeric_notes({fe::NumericalNote_t::REDUCED_PRECISION_REDUCTION});
    checkFrontendStatus(graph->check_support(stream.getCudnnHandle()),
                        "Failed to check support for " + operation + " execution plans");

    const int64_t plan_count = graph->get_execution_plan_count();
    uint64_t max_workspace_bytes = 0;
    int64_t built_plan_count = 0;
    std::string last_build_failure;
    for (int64_t plan_index = 0; plan_index < plan_count; ++plan_index) {
        auto status = graph->build_plan_at_index(stream.getCudnnHandle(), plan_index);
        if (!status.is_good()) {
            last_build_failure = status.get_message();
            continue;
        }
        const int64_t workspace_bytes = graph->get_workspace_size_plan_at_index(plan_index);
        if (workspace_bytes < 0) {
            throw std::runtime_error(operation + " returned a negative candidate workspace size.");
        }
        max_workspace_bytes = std::max(max_workspace_bytes, static_cast<uint64_t>(workspace_bytes));
        ++built_plan_count;
    }
    if (built_plan_count == 0) {
        std::string message = operation + " produced no buildable execution plan.";
        if (!last_build_failure.empty()) {
            message += " Last build failure: " + last_build_failure;
        }
        throw std::runtime_error(message);
    }

    Tensor autotune_workspace;
    void* workspace_ptr = nullptr;
    if (max_workspace_bytes > 0) {
        autotune_workspace = Tensor(input.getPlacement(),
                                    TensorDescriptor(DataType::UINT8, {max_workspace_bytes}),
                                    256);
        workspace_ptr = autotune_workspace.getMemPtr<void>();
    }

    std::unordered_map<int64_t, void*> tensor_pack;
    tensor_pack.emplace(UID_X, input.getMemPtr<void>());
    tensor_pack.emplace(UID_Y, output.getMemPtr<void>());
    checkFrontendStatus(graph->autotune(stream.getCudnnHandle(), tensor_pack, workspace_ptr),
                        "Failed to autotune " + operation + " execution plans");
    stream.synchronize();

    if (graph->get_execution_plan_count() <= 0) {
        throw std::runtime_error(operation + " autotune produced no successfully timed execution plan.");
    }
    return cudnnFrontendSelectedSerializedPlanSelection(*graph, operation);
}

class StampedCudnnFrontendReductionCandidate final : public StampedReductionCandidate {
   public:
    StampedCudnnFrontendReductionCandidate(Tensor input,
                                           const std::vector<uint32_t>& axes,
                                           const Stream& stream,
                                           bool include_input_reformat)
        : input_(std::move(input)), include_input_reformat_(include_input_reformat) {
        if (!input_.isDenseContiguous()) {
            throw std::invalid_argument("cuDNN Frontend reduction benchmark candidate requires dense contiguous input.");
        }
        if (input_.getDimensions().size() != 4 || axes != std::vector<uint32_t>({0, 2, 3})) {
            throw std::invalid_argument("cuDNN Frontend reduction benchmark candidate currently supports [N,C,H,W] -> [C] only.");
        }

        const CubReductionGeometry geometry =
            CubReduction::analyzeGeometry(input_.getDimensions(), input_.getStridesElements(), axes);
        output_ = Tensor(input_.getPlacement(), TensorDescriptor(input_.getDataType(), geometry.output_dimensions));

        const std::vector<int64_t> input_dims = checkedI64Vector(input_.getDimensions(), "input dimensions");
        const std::vector<int64_t> input_strides = nhwcPackedStrides(input_dims);

        // The runtime cuDNN Frontend reduction engine available on SM120 requires fully packed NHWC input. Thor's
        // census tensor is packed NCHW, so keep an operation-local NHWC buffer. The native-NHWC candidate populates
        // it once outside timing to expose cuDNN's reducer cost; the end-to-end candidate repeats this reformat in
        // runOn() so the reported latency is a fair replacement cost for Thor's actual NCHW input.
        nhwc_input_ = Tensor(input_.getPlacement(), input_.getDescriptor());
        launchInputReformat(stream);
        stream.synchronize();

        // cuDNN infers reduction axes by comparing input/output dimensions. Keep the documented rank-4 form and
        // replace N/H/W with singleton dimensions. For NHWC, [1,C,1,1] is still a contiguous C-element output.
        std::vector<int64_t> cudnn_output_dims = input_dims;
        for (uint32_t axis : axes) {
            cudnn_output_dims.at(axis) = 1;
        }
        const std::vector<int64_t> cudnn_output_strides = nhwcPackedStrides(cudnn_output_dims);

        if (input_.getDataType() == DataType::FP32) {
            graph_factory_ = makeReductionGraphFactory(input_.getDataType(),
                                                       DataType::FP32,
                                                       input_dims,
                                                       input_strides,
                                                       cudnn_output_dims,
                                                       cudnn_output_strides);
            const CudnnFrontendPlanSelection selection = selectAutotunedReductionPlan(
                graph_factory_, nhwc_input_, output_, stream, "cuDNN Frontend benchmark NHWC reduction");
            executable_ = std::make_unique<CudnnFrontendExecutablePlan>(
                replayCudnnFrontendExecutablePlan(graph_factory_,
                                                  selection,
                                                  stream.getCudnnHandle(),
                                                  "benchmark NHWC reduction"));
            strategy_ = include_input_reformat_ ? "nchw_to_nhwc_tiled_transpose_plus_autotuned_fp32_output"
                                                : "native_nhwc_autotuned_fp32_output";
        } else {
            // cuDNN's runtime reduction support requires FLOAT Y and FLOAT reduction compute. Preserve Thor's public
            // FP16/BF16 output contract with the ordinary Thor GPU type conversion. The output cast stays in timing.
            fp32_reduction_output_ = Tensor(
                input_.getPlacement(), TensorDescriptor(DataType::FP32, geometry.output_dimensions));
            graph_factory_ = makeReductionGraphFactory(input_.getDataType(),
                                                       DataType::FP32,
                                                       input_dims,
                                                       input_strides,
                                                       cudnn_output_dims,
                                                       cudnn_output_strides);
            const CudnnFrontendPlanSelection selection = selectAutotunedReductionPlan(
                graph_factory_, nhwc_input_, fp32_reduction_output_, stream, "cuDNN Frontend benchmark NHWC FP32-output reduction");
            executable_ = std::make_unique<CudnnFrontendExecutablePlan>(
                replayCudnnFrontendExecutablePlan(graph_factory_,
                                                  selection,
                                                  stream.getCudnnHandle(),
                                                  "benchmark NHWC FP32-output reduction"));
            needs_output_cast_ = true;
            strategy_ = include_input_reformat_
                            ? "nchw_to_nhwc_tiled_transpose_plus_autotuned_fp32_output_plus_thor_cast"
                            : "native_nhwc_autotuned_fp32_output_plus_thor_cast";
        }

        if (!executable_) {
            throw std::logic_error("cuDNN Frontend reduction candidate did not prepare an executable plan.");
        }
        workspace_bytes_ = executable_->workspaceBytes();
        if (workspace_bytes_ > 0) {
            workspace_ = Tensor(input_.getPlacement(),
                                TensorDescriptor(DataType::UINT8, {workspace_bytes_}),
                                256);
        }

        tensor_pack_.emplace(UID_X, nhwc_input_.getMemPtr<void>());
        tensor_pack_.emplace(UID_Y,
                             needs_output_cast_ ? fp32_reduction_output_.getMemPtr<void>() : output_.getMemPtr<void>());
    }

    void runOn(Stream& stream) const override {
        if (include_input_reformat_) {
            launchInputReformat(stream);
        }
        void* workspace_ptr =
            workspace_bytes_ == 0 ? nullptr : const_cast<void*>(workspace_.getMemPtr<void>());
        executable_->execute(stream.getCudnnHandle(), tensor_pack_, workspace_ptr);
        if (needs_output_cast_) {
            TypeConverter::convertType(const_cast<void*>(fp32_reduction_output_.getMemPtr<void>()),
                                       const_cast<void*>(output_.getMemPtr<void>()),
                                       DataType::FP32,
                                       output_.getDataType(),
                                       static_cast<long>(output_.getTotalNumElements()),
                                       stream,
                                       stream.getGpuNum());
        }
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }
    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override {
        return workspace_bytes_ + nhwc_input_.getArraySizeInBytes()
               + (needs_output_cast_ ? fp32_reduction_output_.getArraySizeInBytes() : 0);
    }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        return CandidateLaunchMetadata{
            .implementation = "cudnn_frontend_reduce",
            .strategy = strategy_,
        };
    }

   private:
    void launchInputReformat(const Stream& stream) const {
        const std::vector<uint64_t>& dims = input_.getDimensions();
        launchPackedNchwToNhwc(input_.getMemPtr<void>(),
                               const_cast<void*>(nhwc_input_.getMemPtr<void>()),
                               static_cast<uint32_t>(TensorDescriptor::getElementSizeInBytes(input_.getDataType())),
                               dims[0],
                               dims[1],
                               dims[2],
                               dims[3],
                               stream.getStream());
    }

    Tensor input_;
    Tensor output_;
    Tensor nhwc_input_;
    Tensor fp32_reduction_output_;
    Tensor workspace_;
    size_t workspace_bytes_ = 0;
    bool needs_output_cast_ = false;
    bool include_input_reformat_ = false;
    CudnnFrontendGraphFactory graph_factory_;
    std::unique_ptr<CudnnFrontendExecutablePlan> executable_;
    std::string strategy_;
    mutable std::unordered_map<int64_t, void*> tensor_pack_;
};

class CudnnFrontendReductionCandidateBase : public ReductionCandidate {
   public:
    explicit CudnnFrontendReductionCandidateBase(bool include_input_reformat)
        : include_input_reformat_(include_input_reformat) {}

    [[nodiscard]] bool supports(CubReductionOp op,
                                const TensorDescriptor& input_descriptor,
                                const std::vector<uint32_t>& axes) const override {
        return op == CubReductionOp::Sum && isSupportedIoDtype(input_descriptor.getDataType())
               && input_descriptor.getDimensions().size() == 4 && axes == std::vector<uint32_t>({0, 2, 3});
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                   const Tensor& input,
                                                                   const std::vector<uint32_t>& axes,
                                                                   const Stream& stream) const override {
        if (!supports(op, input.getDescriptor(), axes)) {
            throw std::invalid_argument("Unsupported cuDNN Frontend benchmark reduction candidate geometry.");
        }
        return std::make_unique<StampedCudnnFrontendReductionCandidate>(
            input, axes, stream, include_input_reformat_);
    }

   private:
    bool include_input_reformat_;
};

class CudnnFrontendReductionCandidate final : public CudnnFrontendReductionCandidateBase {
   public:
    CudnnFrontendReductionCandidate() : CudnnFrontendReductionCandidateBase(true) {}
    [[nodiscard]] std::string_view getName() const override { return "cudnn_frontend_reduce"; }
};

class CudnnFrontendNativeNhwcReductionCandidate final : public CudnnFrontendReductionCandidateBase {
   public:
    CudnnFrontendNativeNhwcReductionCandidate() : CudnnFrontendReductionCandidateBase(false) {}
    [[nodiscard]] std::string_view getName() const override { return "cudnn_frontend_reduce_native_nhwc"; }
};

const ReductionCandidateRegistrar<CudnnFrontendReductionCandidate> cudnn_frontend_reduce_registrar;
const ReductionCandidateRegistrar<CudnnFrontendNativeNhwcReductionCandidate> cudnn_frontend_reduce_native_nhwc_registrar;

}  // namespace
}  // namespace ThorImplementation::CubReductionBenchmarking
