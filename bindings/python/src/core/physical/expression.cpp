#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>
#include <variant>

#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/CudaKernelSecurity.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

namespace nb = nanobind;
using namespace nb::literals;

using Expression = ThorImplementation::Expression;
using FusedEquation = ThorImplementation::FusedEquation;
using StampedEquation = ThorImplementation::StampedEquation;
using DataType = ThorImplementation::TensorDescriptor::DataType;
using Tensor = ThorImplementation::Tensor;
using StampedExecutionPlan = ThorImplementation::StampedExecutionPlan;
using Outputs = ThorImplementation::Outputs;
using ExpressionDefinition = ThorImplementation::ExpressionDefinition;
using NamedOutput = ThorImplementation::NamedOutput;
using DynamicExpression = ThorImplementation::DynamicExpression;
using TensorScalarBinding = ThorImplementation::TensorScalarBinding;
using AttentionTensorLayout = ThorImplementation::AttentionTensorLayout;
using AttentionMaskKind = ThorImplementation::AttentionMaskKind;
using AttentionOptions = ThorImplementation::AttentionOptions;
using RotaryScalingKind = ThorImplementation::RotaryScalingKind;
using RotaryPositionEmbeddingOptions = ThorImplementation::RotaryPositionEmbeddingOptions;
using PreparedDynamicExpression = ThorImplementation::PreparedDynamicExpression;
using DynamicExpressionBuild = ThorImplementation::DynamicExpressionBuild;
using CudaKernelExpression = ThorImplementation::CudaKernelExpression;
using CudaKernelSourceInspection = ThorImplementation::CudaKernelSourceInspection;
using CudaKernelOutOfBandKeys = ThorImplementation::CudaKernelOutOfBandKeys;
using CudaKernelDimExpr = ThorImplementation::CudaKernelExpression::DimExpr;
using CudaKernelLaunchContext = ThorImplementation::CudaKernelExpression::LaunchContext;
using CudaKernelLaunchConfig = ThorImplementation::CudaKernelLaunchConfig;
using DynamicTensorMap = std::unordered_map<std::string, Tensor>;
using DynamicTensorScalarMap = std::unordered_map<std::string, TensorScalarBinding>;
using DynamicShapeMap = std::unordered_map<std::string, std::vector<uint64_t>>;

namespace {

class GilSafePythonObject {
   public:
    explicit GilSafePythonObject(nb::handle object) : object(object.ptr()) {
        nb::gil_scoped_acquire gil;
        Py_XINCREF(this->object);
    }

    GilSafePythonObject(const GilSafePythonObject&) = delete;
    GilSafePythonObject& operator=(const GilSafePythonObject&) = delete;

    ~GilSafePythonObject() {
        if (object == nullptr)
            return;

        nb::gil_scoped_acquire gil;
        Py_XDECREF(object);
    }

    nb::handle get() const { return nb::handle(object); }

   private:
    PyObject* object = nullptr;
};

dim3 dim3FromPython(nb::handle value, const char* name) {
    std::vector<uint32_t> dims = nb::cast<std::vector<uint32_t>>(value);
    if (dims.empty() || dims.size() > 3) {
        throw nb::value_error((std::string(name) + " must contain 1, 2, or 3 unsigned integer dimensions.").c_str());
    }

    while (dims.size() < 3) {
        dims.push_back(1);
    }
    return dim3(dims[0], dims[1], dims[2]);
}

CudaKernelLaunchConfig makeCudaKernelLaunchConfig(nb::handle grid, nb::handle block, uint32_t dynamic_shared_bytes) {
    CudaKernelLaunchConfig config;
    config.grid = dim3FromPython(grid, "grid");
    config.block = dim3FromPython(block, "block");
    config.dynamic_shared_bytes = dynamic_shared_bytes;
    return config;
}

CudaKernelDimExpr dimExprFromPython(nb::handle value) {
    if (nb::isinstance<CudaKernelDimExpr>(value)) {
        return nb::cast<CudaKernelDimExpr>(value);
    }
    if (nb::isinstance<nb::int_>(value)) {
        const int64_t dim = nb::cast<int64_t>(value);
        if (dim < 0) {
            throw nb::value_error("CudaKernelExpression dimensions must be non-negative.");
        }
        return CudaKernelDimExpr::constant(static_cast<uint64_t>(dim));
    }
    throw nb::type_error("CudaKernelExpression shape entries must be integers or CudaKernelDimExpr objects.");
}

std::vector<CudaKernelDimExpr> dimExprVectorFromPython(nb::handle shape) {
    std::vector<CudaKernelDimExpr> dims;
    for (nb::handle item : nb::cast<nb::sequence>(shape)) {
        dims.push_back(dimExprFromPython(item));
    }
    return dims;
}

std::variant<int32_t, uint32_t, int64_t, uint64_t, float, double, CudaKernelDimExpr> scalarValueFromPython(DataType type,
                                                                                                           nb::handle value) {
    if (nb::isinstance<CudaKernelDimExpr>(value)) {
        return nb::cast<CudaKernelDimExpr>(value);
    }

    switch (type) {
        case DataType::INT32:
            return static_cast<int32_t>(nb::cast<int64_t>(value));
        case DataType::UINT32:
            return static_cast<uint32_t>(nb::cast<uint64_t>(value));
        case DataType::INT64:
            return static_cast<int64_t>(nb::cast<int64_t>(value));
        case DataType::UINT64:
            return static_cast<uint64_t>(nb::cast<uint64_t>(value));
        case DataType::FP32:
            return static_cast<float>(nb::cast<double>(value));
        case DataType::FP64:
            return static_cast<double>(nb::cast<double>(value));
        default:
            throw nb::value_error("CudaKernelExpression scalar dtype is not supported for by-value kernel scalar arguments. Supported scalar dtypes are int32, uint32, int64, uint64, fp32, and fp64.");
    }
}

CudaKernelExpression::LaunchFn launchFnFromPython(nb::callable launch) {
    auto launch_ref = std::make_shared<GilSafePythonObject>(launch);
    return [launch_ref](const CudaKernelLaunchContext& ctx) -> CudaKernelLaunchConfig {
        nb::gil_scoped_acquire gil;
        nb::callable launch_callable = nb::borrow<nb::callable>(launch_ref->get());
        nb::object result = launch_callable(nb::cast(&ctx, nb::rv_policy::reference));
        return nb::cast<CudaKernelLaunchConfig>(result);
    };
}

nb::dict cudaKernelSourceInspectionToPython(const CudaKernelSourceInspection& info) {
    nb::dict entry;
    entry["name"] = info.name;
    entry["entrypoint"] = info.entrypoint;
    entry["source"] = info.source;
    entry["compiled_source"] = info.compiled_source;
    entry["compiled_source_hash"] = info.compiled_source_hash;
    entry["loaded_source_compilation_allowed"] = info.loaded_source_compilation_allowed;
    entry["source_encrypted"] = info.source_encrypted;
    if (!info.source_encryption_algorithm.empty()) {
        entry["source_encryption_algorithm"] = info.source_encryption_algorithm;
    }
    if (!info.source_decryption_key_fingerprint.empty()) {
        entry["source_decryption_key_fingerprint"] = info.source_decryption_key_fingerprint;
    }
    if (!info.signature_algorithm.empty()) {
        entry["signature_algorithm"] = info.signature_algorithm;
    }
    if (!info.signing_public_key_fingerprint.empty()) {
        entry["signing_public_key_fingerprint"] = info.signing_public_key_fingerprint;
    }
    if (!info.signature.empty()) {
        entry["signature"] = info.signature;
    }
    return entry;
}

nb::list cudaKernelSourceInspectionListToPython(const std::vector<CudaKernelSourceInspection>& infos) {
    nb::list result;
    for (const CudaKernelSourceInspection& info : infos) {
        result.append(cudaKernelSourceInspectionToPython(info));
    }
    return result;
}

nb::list cudaKernelOutOfBandKeysToPython(const std::vector<CudaKernelOutOfBandKeys>& key_sets) {
    nb::list result;
    for (const CudaKernelOutOfBandKeys& keys : key_sets) {
        nb::dict entry;
        entry["signing_public_key"] = keys.signing_public_key;
        entry["source_decryption_key"] = keys.source_decryption_key;
        result.append(std::move(entry));
    }
    return result;
}

}  // namespace

static nb::dict parameterFanOverridesToPython(const FusedEquation::ParameterFanOverrideMap& overrides) {
    nb::dict result;
    for (const auto& [name, hint] : overrides) {
        nb::dict entry;
        entry["fan_in"] = nb::int_(hint.fan_in);
        entry["fan_out"] = nb::int_(hint.fan_out);
        result[nb::str(name.c_str())] = std::move(entry);
    }
    return result;
}

void bind_physical_expression(nb::module_& physical) {
    physical.def(
        "cuda_kernel_signing_public_keys_from_json",
        [](const std::string& payload) {
            return ThorImplementation::collectCudaKernelSigningPublicKeys(nlohmann::json::parse(payload));
        },
        "payload"_a,
        R"nbdoc(
Return the out-of-band CudaKernelExpression public signing keys that were generated
by this process for a serialized expression JSON payload. The serialized payload
contains only public-key fingerprints, so this returns keys only while the
process-local ephemeral signing registry still has them.
)nbdoc");

    physical.def(
        "cuda_kernel_out_of_band_keys_from_json",
        [](const std::string& payload) {
            return cudaKernelOutOfBandKeysToPython(ThorImplementation::collectCudaKernelOutOfBandKeys(nlohmann::json::parse(payload)));
        },
        "payload"_a,
        R"nbdoc(
Return the out-of-band CudaKernelExpression key sets generated by this process
for a serialized expression JSON payload. Each entry contains the Ed25519 public
signing key and AES-256-GCM source decryption key. The serialized payload stores
only fingerprints, so this works only while the process-local ephemeral key
registries still have the keys.
)nbdoc");

    physical.def(
        "cuda_kernel_source_info_from_json",
        [](const std::string& payload) {
            return cudaKernelSourceInspectionListToPython(ThorImplementation::collectCudaKernelSourceInfo(nlohmann::json::parse(payload)));
        },
        "payload"_a,
        R"nbdoc(
Return CudaKernelExpression source entries from serialized expression JSON
without enabling CUDA source compilation. Encrypted payloads expose encryption
metadata and key fingerprints but not plaintext source.
)nbdoc");

    auto attention_layout = nb::enum_<AttentionTensorLayout>(physical, "AttentionTensorLayout")
                                .value("bhsd", AttentionTensorLayout::BHSD)
                                .value("bshd", AttentionTensorLayout::BSHD);
    attention_layout.attr("__module__") = "thor.physical";
    attention_layout.attr("__doc__") = R"nbdoc(
Attention tensor layout used by cuDNN SDPA expression stages.

The semantic tensor shape is always ``[B, H, S, D]``.  ``bhsd`` stores that order
directly.  ``bshd`` stores batch, sequence, heads, head dimension.  Ragged/packed
THD attention requires BSHD physical layouts for Q/K/V/O so ragged offsets index
packed token-contiguous storage.
)nbdoc";

    auto attention_mask_kind = nb::enum_<AttentionMaskKind>(physical, "AttentionMaskKind")
                                   .value("none", AttentionMaskKind::None)
                                   .value("causal_top_left", AttentionMaskKind::CausalTopLeft)
                                   .value("causal_bottom_right", AttentionMaskKind::CausalBottomRight)
                                   .value("sliding_window_top_left", AttentionMaskKind::SlidingWindowTopLeft)
                                   .value("sliding_window_bottom_right", AttentionMaskKind::SlidingWindowBottomRight);
    attention_mask_kind.attr("__module__") = "thor.physical";
    attention_mask_kind.attr("__doc__") = R"nbdoc(
Mask kinds supported by Thor's cuDNN SDPA path.

``causal_top_left`` and ``sliding_window_top_left`` use standard top-left diagonal
semantics.  ``causal_bottom_right`` and ``sliding_window_bottom_right`` support
decode-style alignment, but production cuDNN primary SDPA currently requires
additive bias, ALiBi, and dropout to be disabled for bottom-right/decode masks.
ALiBi requires a causal/sliding diagonal mask with ``diagonal_right_bound == 0``.
)nbdoc";

    auto rotary_scaling_kind = nb::enum_<RotaryScalingKind>(physical, "RotaryScalingKind")
                                   .value("none", RotaryScalingKind::None)
                                   .value("linear", RotaryScalingKind::Linear)
                                   .value("dynamic_ntk", RotaryScalingKind::DynamicNTK)
                                   .value("yarn", RotaryScalingKind::Yarn)
                                   .value("longrope", RotaryScalingKind::LongRope)
                                   .value("llama3", RotaryScalingKind::Llama3);
    rotary_scaling_kind.attr("__module__") = "thor.physical";
    rotary_scaling_kind.attr("__doc__") = R"nbdoc(
RoPE scaling parameterization for ``Expression.rotary_position_embedding`` and
``thor.layers.Attention``.  The high-level Attention layer supports ``none``,
``linear``, ``dynamic_ntk``, ``yarn``, ``longrope``, and ``llama3``.
)nbdoc";

    physical.def(
        "cudnn_frontend_attention_available",
        []() { return ThorImplementation::CudnnScaledDotProductAttention::frontendAvailable(); },
        R"nbdoc(
Return True when Thor was compiled with the cuDNN Frontend C++ headers needed by the cuDNN SDPA attention executor.
)nbdoc");

    auto cuda_kernel_dim_expr = nb::class_<CudaKernelDimExpr>(physical, "CudaKernelDimExpr");
    cuda_kernel_dim_expr.attr("__module__") = "thor.physical";
    cuda_kernel_dim_expr.def_static("constant", &CudaKernelDimExpr::constant, "value"_a)
        .def_static("dim", &CudaKernelDimExpr::dim, "tensor_name"_a, "axis"_a)
        .def_static("numel", &CudaKernelDimExpr::numel, "tensor_name"_a)
        .def("describe", &CudaKernelDimExpr::describe)
        .def("__repr__", [](const CudaKernelDimExpr& self) { return "CudaKernelDimExpr(" + self.describe() + ")"; });

    auto cuda_kernel_launch_config = nb::class_<CudaKernelLaunchConfig>(physical, "CudaKernelLaunchConfig");
    cuda_kernel_launch_config.attr("__module__") = "thor.physical";
    cuda_kernel_launch_config
        .def(
            "__init__",
            [](CudaKernelLaunchConfig* self, nb::object grid, nb::object block, uint32_t dynamic_shared_bytes) {
                new (self) CudaKernelLaunchConfig(makeCudaKernelLaunchConfig(grid, block, dynamic_shared_bytes));
            },
            "grid"_a,
            "block"_a,
            "dynamic_shared_bytes"_a = 0,
            R"nbdoc(
CUDA launch configuration for a CudaKernelExpression.

``grid`` and ``block`` may be 1-, 2-, or 3-element integer sequences. Missing
trailing dimensions default to 1. ``dynamic_shared_bytes`` is passed as the
kernel launch dynamic shared-memory byte count.
)nbdoc")
        .def_static(
            "grid_1d",
            [](uint64_t elements, uint32_t block_size, uint32_t dynamic_shared_bytes) {
                if (block_size == 0) {
                    throw nb::value_error("block_size must be nonzero.");
                }
                uint64_t grid_x = (elements + block_size - 1) / block_size;
                if (grid_x == 0) {
                    grid_x = 1;
                }
                if (grid_x > std::numeric_limits<uint32_t>::max()) {
                    throw nb::value_error("grid_x does not fit in uint32_t.");
                }
                return CudaKernelLaunchConfig{dim3(static_cast<uint32_t>(grid_x), 1, 1), dim3(block_size, 1, 1), dynamic_shared_bytes};
            },
            "elements"_a,
            "block_size"_a = 256,
            "dynamic_shared_bytes"_a = 0)
        .def_prop_rw(
            "dynamic_shared_bytes",
            [](const CudaKernelLaunchConfig& self) { return self.dynamic_shared_bytes; },
            [](CudaKernelLaunchConfig& self, uint32_t bytes) { self.dynamic_shared_bytes = bytes; })
        .def_prop_ro("grid",
                     [](const CudaKernelLaunchConfig& self) { return std::vector<uint32_t>{self.grid.x, self.grid.y, self.grid.z}; })
        .def_prop_ro("block",
                     [](const CudaKernelLaunchConfig& self) { return std::vector<uint32_t>{self.block.x, self.block.y, self.block.z}; });

    auto cuda_kernel_launch_context = nb::class_<CudaKernelLaunchContext>(physical, "CudaKernelLaunchContext");
    cuda_kernel_launch_context.attr("__module__") = "thor.physical";
    cuda_kernel_launch_context.def("dim", &CudaKernelLaunchContext::dim, "tensor_name"_a, "axis"_a)
        .def("numel", &CudaKernelLaunchContext::numel, "tensor_name"_a)
        .def("dtype", &CudaKernelLaunchContext::dtype, "tensor_name"_a)
        .def_prop_ro("device_num", [](const CudaKernelLaunchContext& self) { return self.device_num; });

    auto cuda_kernel_expression = nb::class_<CudaKernelExpression>(physical, "CudaKernelExpression");
    cuda_kernel_expression.attr("__module__") = "thor.physical";

    auto cuda_kernel_expression_builder = nb::class_<CudaKernelExpression::Builder>(physical, "CudaKernelExpressionBuilder");
    cuda_kernel_expression_builder.attr("__module__") = "thor.physical";
    cuda_kernel_expression_builder.def(nb::init<std::string>(), "name"_a)
        .def("source", &CudaKernelExpression::Builder::source, "cuda_source"_a, nb::rv_policy::reference_internal)
        .def("entry", &CudaKernelExpression::Builder::entry, "entrypoint"_a, nb::rv_policy::reference_internal)
        .def("input", &CudaKernelExpression::Builder::input, "name"_a, "dtype"_a, nb::rv_policy::reference_internal)
        .def("tensor_runtime_scalar_input",
             &CudaKernelExpression::Builder::tensorRuntimeScalarInput,
             "name"_a,
             "dtype"_a,
             nb::rv_policy::reference_internal)
        .def("host_runtime_scalar_input",
             &CudaKernelExpression::Builder::hostRuntimeScalarInput,
             "name"_a,
             "dtype"_a,
             nb::rv_policy::reference_internal)
        .def(
            "output",
            [](CudaKernelExpression::Builder& self, const std::string& name, DataType dtype, nb::sequence shape)
                -> CudaKernelExpression::Builder& { return self.output(name, dtype, dimExprVectorFromPython(shape)); },
            "name"_a,
            "dtype"_a,
            "shape"_a,
            nb::rv_policy::reference_internal)
        .def("output_like",
             &CudaKernelExpression::Builder::outputLike,
             "name"_a,
             "dtype"_a,
             "input_name"_a,
             nb::rv_policy::reference_internal)
        .def(
            "scalar",
            [](CudaKernelExpression::Builder& self, const std::string& name, DataType type, nb::object value)
                -> CudaKernelExpression::Builder& { return self.scalar(name, type, scalarValueFromPython(type, value)); },
            "name"_a,
            "type"_a,
            "value"_a,
            nb::rv_policy::reference_internal)
        .def(
            "launch",
            [](CudaKernelExpression::Builder& self, nb::callable launch) -> CudaKernelExpression::Builder& {
                return self.launch(launchFnFromPython(launch));
            },
            "launch"_a,
            nb::rv_policy::reference_internal)
        .def(
            "launch_grid_1d",
            [](CudaKernelExpression::Builder& self, nb::object elements, uint32_t block_size, uint32_t dynamic_shared_bytes)
                -> CudaKernelExpression::Builder& {
                return self.launchGrid1D(dimExprFromPython(elements), block_size, dynamic_shared_bytes);
            },
            "elements"_a,
            "block_size"_a = 256,
            "dynamic_shared_bytes"_a = 0,
            nb::rv_policy::reference_internal)
        .def("use_fast_math", &CudaKernelExpression::Builder::useFastMath, "enabled"_a = true, nb::rv_policy::reference_internal)
        .def("build", &CudaKernelExpression::Builder::build);

    cuda_kernel_expression.def_static("builder", &CudaKernelExpression::builder, "name"_a)
        .def_static("dim", &CudaKernelDimExpr::dim, "tensor_name"_a, "axis"_a)
        .def_static("numel", &CudaKernelDimExpr::numel, "tensor_name"_a)
        .def_static("constant_dim", &CudaKernelDimExpr::constant, "value"_a)
        .def("name", &CudaKernelExpression::name)
        .def_prop_ro("source", &CudaKernelExpression::source)
        .def_prop_ro("compiled_source", &CudaKernelExpression::compiledSource)
        .def_prop_ro("loaded_source_compilation_allowed", &CudaKernelExpression::loadedSourceCompilationAllowed)
        .def("source_info", [](const CudaKernelExpression& self) {
            const auto info = self.sourceInfo();
            CudaKernelSourceInspection inspection;
            inspection.name = info.name;
            inspection.entrypoint = info.entrypoint;
            inspection.source = info.source;
            inspection.compiled_source = info.compiled_source;
            inspection.compiled_source_hash = info.source_hash;
            inspection.loaded_source_compilation_allowed = info.loaded_source_compilation_allowed;
            return cudaKernelSourceInspectionToPython(inspection);
        })
        .def("source_info_json", [](const CudaKernelExpression& self) {
            const auto info = self.sourceInfo();
            CudaKernelSourceInspection inspection;
            inspection.name = info.name;
            inspection.entrypoint = info.entrypoint;
            inspection.source = info.source;
            inspection.compiled_source = info.compiled_source;
            inspection.compiled_source_hash = info.source_hash;
            inspection.loaded_source_compilation_allowed = info.loaded_source_compilation_allowed;
            return ThorImplementation::cudaKernelSourceInspectionToJson(inspection).dump(2);
        })
        .def("apply", &CudaKernelExpression::apply, "inputs"_a)
        .def("__call__", &CudaKernelExpression::apply, "inputs"_a)
        .def("as_dynamic_expression", &CudaKernelExpression::asDynamicExpression)
        .def(
            "stamp",
            [](const CudaKernelExpression& self,
               const std::unordered_map<std::string, Tensor>& inputs,
               const std::unordered_map<std::string, Tensor>& preallocated_outputs,
               Stream& stream,
               const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs) {
                return self.stamp(inputs, preallocated_outputs, stream, tensor_scalar_inputs);
            },
            "inputs"_a,
            "preallocated_outputs"_a,
            "stream"_a,
            "tensor_scalar_inputs"_a = std::unordered_map<std::string, TensorScalarBinding>{},
            R"nbdoc(
Compile, bind, and stamp this CUDA kernel expression directly.

Most users should use ``apply(...)`` to stitch the custom kernel into a normal
Thor expression graph. This direct path is useful for low-level tests and
standalone custom kernels.
)nbdoc");

    auto expr = nb::class_<Expression>(physical, "Expression");
    expr.attr("__module__") = "thor.physical";

    expr.def(nb::init_implicit<double>());
    // expr.def(nb::init_implicit<int64_t>());

    expr.def_static(
        "input",
        [](const std::string& name, nb::object output_dtype_obj, nb::object compute_dtype_obj) {
            std::optional<DataType> output_dtype = std::nullopt;
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            std::optional<DataType> compute_dtype = std::nullopt;
            if (!compute_dtype_obj.is_none()) {
                compute_dtype = nb::cast<DataType>(compute_dtype_obj);
            }
            return Expression::input(name, compute_dtype, output_dtype);
        },
        "name"_a,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
Create an input expression.

Parameters
----------
name : str
    Input name.
output_dtype : thor.DataType | None
    Optional dtype to cast the input value to when it enters the expression graph.
    The actual bound runtime tensor may have a different dtype.

Returns
-------
thor.physical.Expression
    An Expression representing that input.
)nbdoc");

    nb::class_<TensorScalarBinding>(physical, "TensorScalarBinding")
        .def(nb::init<>())
        .def(nb::init<const Tensor&, uint64_t, DataType>(), "buffer"_a, "source_dtype"_a, "byte_offset"_a = 0)
        .def_rw("buffer", &TensorScalarBinding::buffer)
        .def_rw("source_dtype", &TensorScalarBinding::sourceDType)
        .def_rw("byte_offset", &TensorScalarBinding::byteOffset);

    expr.def_static(
        "runtime_scalar",
        [](const std::string& name, nb::object output_dtype_obj, nb::object compute_dtype_obj) {
            std::optional<DataType> output_dtype = std::nullopt;
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            std::optional<DataType> compute_dtype = std::nullopt;
            if (!compute_dtype_obj.is_none()) {
                compute_dtype = nb::cast<DataType>(compute_dtype_obj);
            }
            return Expression::runtimeScalar(name, compute_dtype, output_dtype);
        },
        "name"_a,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
Create a runtime-bound scalar input expression.

Parameters
----------
name : str
    Runtime scalar input name.
output_dtype : thor.DataType | None
    Optional dtype cast applied to the runtime scalar as it enters the graph.
    Currently runtime scalar bindings are passed as fp32 values.

Returns
-------
thor.physical.Expression
    An Expression representing that runtime scalar input.
)nbdoc");

    expr.def_static(
        "tensor_runtime_scalar",
        [](const std::string& name, nb::object output_dtype_obj, nb::object compute_dtype_obj) {
            std::optional<DataType> output_dtype = std::nullopt;
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            std::optional<DataType> compute_dtype = std::nullopt;
            if (!compute_dtype_obj.is_none()) {
                compute_dtype = nb::cast<DataType>(compute_dtype_obj);
            }
            return Expression::tensorRuntimeScalar(name, compute_dtype, output_dtype);
        },
        "name"_a,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
Create a GPU tensor-backed runtime scalar input expression.

The scalar is loaded from a bound GPU buffer at stamp time using a
TensorScalarBinding (buffer, byte_offset, source_dtype).
)nbdoc");

    expr.def(
        "with_dtypes",
        [](const Expression& self, nb::object output_dtype_obj, nb::object compute_dtype_obj) {
            std::optional<DataType> output_dtype = std::nullopt;
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            std::optional<DataType> compute_dtype = std::nullopt;
            if (!compute_dtype_obj.is_none()) {
                compute_dtype = nb::cast<DataType>(compute_dtype_obj);
            }
            return self.withDTypes(compute_dtype, output_dtype);
        },
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
Return a new expression whose result node has local dtype overrides.

This only annotates the current expression result node. It does not recursively
rewrite the dtypes of ancestor nodes in the subexpression.

Parameters
----------
output_dtype : thor.DataType | None
    Optional output dtype override for this expression node.
compute_dtype : thor.DataType | None
    Optional compute dtype override for this expression node.

Returns
-------
thor.physical.Expression
    A new Expression with the requested local dtype overrides applied to its
    result node.
)nbdoc");

    expr.def(
        "with_output_dtype",
        [](const Expression& self, DataType output_dtype) { return self.withOutputDType(output_dtype); },
        "output_dtype"_a,
        R"nbdoc(
Return a new expression whose result node uses the requested output dtype.
)nbdoc");

    expr.def(
        "with_compute_dtype",
        [](const Expression& self, DataType compute_dtype) { return self.withComputeDType(compute_dtype); },
        "compute_dtype"_a,
        R"nbdoc(
Return a new expression whose result node uses the requested compute dtype.
)nbdoc");

    expr.def(
        "reshape",
        [](const Expression& self, const std::vector<uint64_t>& new_dims) { return self.reshape(new_dims); },
        "shape"_a,
        R"nbdoc(
Return a metadata-only reshape expression.

For contiguous tensors this is planned as a value alias rather than a materializing
fused kernel when no dtype conversion is requested.
)nbdoc");

    expr.def(
        "strided_view",
        [](const Expression& self, const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides, uint64_t element_offset) {
            return self.stridedView(dims, strides, element_offset);
        },
        "shape"_a,
        "strides"_a,
        "element_offset"_a = 0,
        R"nbdoc(
Return a zero-materialization storage alias with explicit element strides.

The alias shares the source allocation, starts at element_offset elements from the
source tensor's visible base pointer, and indexes the requested shape using the
provided element strides. This is intended for layout/descriptor adapters such as
packed-QKV attention views; generic fused kernels should materialize or lower a
layout-aware kernel before consuming non-dense views.
)nbdoc");

    expr.def_static(
        "constant_scalar",
        [](double value) { return Expression::constantScalar(value); },
        "value"_a,
        R"nbdoc(
Create a floating-point scalar constant expression.
)nbdoc");

    expr.def("__add__", [](const Expression& a, const Expression& b) { return a + b; }, "other"_a);
    expr.def("__sub__", [](const Expression& a, const Expression& b) { return a - b; }, "other"_a);
    expr.def("__mul__", [](const Expression& a, const Expression& b) { return a * b; }, "other"_a);
    expr.def("__truediv__", [](const Expression& a, const Expression& b) { return a / b; }, "other"_a);
    expr.def("__pow__", [](const Expression& a, const Expression& b) { return a.pow(b); });

    expr.def("__radd__", [](const Expression& a, const Expression& b) { return b + a; }, "other"_a);
    expr.def("__rsub__", [](const Expression& a, const Expression& b) { return b - a; }, "other"_a);
    expr.def("__rmul__", [](const Expression& a, const Expression& b) { return b * a; }, "other"_a);
    expr.def("__rtruediv__", [](const Expression& a, const Expression& b) { return b / a; }, "other"_a);
    expr.def("__rpow__", [](const Expression& a, const Expression& b) { return b.pow(a); }, "other"_a);
    expr.def("__matmul__", [](const Expression& a, const Expression& b) { return Expression::matmul(a, b); }, "other"_a);
    expr.def("__rmatmul__", [](const Expression& a, const Expression& b) { return Expression::matmul(b, a); }, "other"_a);
    expr.def("__imatmul__", [](const Expression& a, const Expression& b) { return Expression::matmul(a, b); }, "other"_a);

    expr.def("__neg__", [](const Expression& a) { return -a; });
    expr.def(
        "transpose",
        [](const Expression& a) { return a.transpose(); },
        R"nbdoc(
Return an expression with the last two dimensions swapped.

For rank-2 tensors this is a matrix transpose. For rank > 2, this is
a batched transpose over the trailing matrix dimensions. A final transpose
can be folded into a fused tiled materialization kernel.
)nbdoc");
    expr.def_prop_ro(
        "T",
        [](const Expression& a) { return a.transpose(); },
        R"nbdoc(
Shorthand for ``self.transpose()``.
)nbdoc");

    expr.def_static(
        "conv2d",
        [](const Expression& x,
           const Expression& w,
           int32_t stride_h,
           int32_t stride_w,
           int32_t pad_h,
           int32_t pad_w,
           nb::object output_dtype_obj,
           nb::object compute_dtype_obj) {
            std::optional<DataType> output_dtype = std::nullopt;
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            std::optional<DataType> compute_dtype = std::nullopt;
            if (!compute_dtype_obj.is_none()) {
                compute_dtype = nb::cast<DataType>(compute_dtype_obj);
            }
            return Expression::conv2d(x, w, stride_h, stride_w, pad_h, pad_w, compute_dtype, output_dtype);
        },
        "x"_a,
        "w"_a,
        "stride_h"_a = 1,
        "stride_w"_a = 1,
        "pad_h"_a = 0,
        "pad_w"_a = 0,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none());

    expr.def_static(
        "conv3d",
        [](const Expression& x,
           const Expression& w,
           int32_t stride_d,
           int32_t stride_h,
           int32_t stride_w,
           int32_t pad_d,
           int32_t pad_h,
           int32_t pad_w,
           nb::object output_dtype_obj,
           nb::object compute_dtype_obj) {
            std::optional<DataType> output_dtype = std::nullopt;
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            std::optional<DataType> compute_dtype = std::nullopt;
            if (!compute_dtype_obj.is_none()) {
                compute_dtype = nb::cast<DataType>(compute_dtype_obj);
            }
            return Expression::conv3d(x, w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, compute_dtype, output_dtype);
        },
        "x"_a,
        "w"_a,
        "stride_d"_a = 1,
        "stride_h"_a = 1,
        "stride_w"_a = 1,
        "pad_d"_a = 0,
        "pad_h"_a = 0,
        "pad_w"_a = 0,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none());

    expr.def_static(
        "matmul",
        [](const Expression& a,
           const Expression& b,
           bool transpose_a,
           bool transpose_b,
           nb::object output_dtype_obj,
           nb::object compute_dtype_obj) {
            std::optional<DataType> output_dtype = std::nullopt;
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            std::optional<DataType> compute_dtype = std::nullopt;
            if (!compute_dtype_obj.is_none()) {
                compute_dtype = nb::cast<DataType>(compute_dtype_obj);
            }
            return Expression::matmul(a, b, transpose_a, transpose_b, compute_dtype, output_dtype);
        },
        "a"_a,
        "b"_a,
        "transpose_a"_a = false,
        "transpose_b"_a = false,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none());

    expr.def_static(
        "gemm",
        [](const Expression& a,
           const Expression& b,
           const Expression& c,
           const Expression& alpha,
           const Expression& beta,
           bool transpose_a,
           bool transpose_b,
           bool transpose_c,
           nb::object output_dtype_obj,
           nb::object compute_dtype_obj) {
            std::optional<DataType> output_dtype = std::nullopt;
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            std::optional<DataType> compute_dtype = std::nullopt;
            if (!compute_dtype_obj.is_none()) {
                compute_dtype = nb::cast<DataType>(compute_dtype_obj);
            }
            return Expression::gemm(a, b, c, alpha, beta, transpose_a, transpose_b, transpose_c, compute_dtype, output_dtype);
        },
        "a"_a,
        "b"_a,
        "c"_a,
        "alpha"_a = 1.0,
        "beta"_a = 1.0,
        "transpose_a"_a = false,
        "transpose_b"_a = false,
        "transpose_c"_a = false,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none());

    auto parse_optional_dtype = [](const nb::object& dtype_obj) -> std::optional<DataType> {
        if (dtype_obj.is_none()) {
            return std::nullopt;
        }
        return nb::cast<DataType>(dtype_obj);
    };

    expr.def_static(
        "embedding_lookup",
        [parse_optional_dtype](const Expression& indices, const Expression& weights, nb::object padding_index_obj, nb::object output_dtype_obj) {
            std::optional<uint64_t> padding_index = std::nullopt;
            if (!padding_index_obj.is_none()) {
                padding_index = nb::cast<uint64_t>(padding_index_obj);
            }
            return Expression::embeddingLookup(indices, weights, padding_index, parse_optional_dtype(output_dtype_obj));
        },
        "indices"_a,
        "weights"_a,
        "padding_index"_a.none() = nb::none(),
        "output_dtype"_a.none() = nb::none(),
        R"nbdoc(
Embedding lookup expression. The indices tensor must be uint32 or uint64. The weights tensor must have shape
[vocabulary_size, embedding_dim], and the output shape is indices.shape + [embedding_dim]. When padding_index is set,
matching rows are written as zeros without reading the weight table.
)nbdoc");

    auto build_attention_from_python_args = [parse_optional_dtype](const Expression& q,
                                                                   const Expression& k,
                                                                   const Expression& v,
                                                                   AttentionTensorLayout q_layout,
                                                                   AttentionTensorLayout k_layout,
                                                                   AttentionTensorLayout v_layout,
                                                                   AttentionTensorLayout o_layout,
                                                                   AttentionMaskKind mask_kind,
                                                                   int64_t diagonal_left_bound,
                                                                   int64_t diagonal_right_bound,
                                                                   nb::object attention_scale_obj,
                                                                   bool use_alibi_mask,
                                                                   nb::object output_dtype_obj,
                                                                   nb::object compute_dtype_obj,
                                                                   nb::object bias_obj,
                                                                   nb::object q_seq_len_obj,
                                                                   nb::object kv_seq_len_obj,
                                                                   nb::object q_ragged_offsets_obj,
                                                                   nb::object kv_ragged_offsets_obj,
                                                                   nb::object page_table_k_obj,
                                                                   nb::object page_table_v_obj,
                                                                   int64_t paged_kv_max_sequence_length,
                                                                   float dropout_probability,
                                                                   nb::object dropout_seed_obj,
                                                                   nb::object dropout_offset_obj) -> Expression {
        AttentionOptions options;
        options.q_layout = q_layout;
        options.k_layout = k_layout;
        options.v_layout = v_layout;
        options.o_layout = o_layout;
        options.mask_kind = mask_kind;
        options.diagonal_left_bound = diagonal_left_bound;
        options.diagonal_right_bound = diagonal_right_bound;
        if (!attention_scale_obj.is_none()) {
            options.attention_scale = nb::cast<float>(attention_scale_obj);
        }
        options.use_alibi_mask = use_alibi_mask;
        options.output_dtype = parse_optional_dtype(output_dtype_obj);
        options.compute_dtype = parse_optional_dtype(compute_dtype_obj);
        options.dropout_probability = dropout_probability;
        options.paged_kv_max_sequence_length = paged_kv_max_sequence_length;

        const bool has_bias = !bias_obj.is_none();
        const bool has_q_seq_len = !q_seq_len_obj.is_none();
        const bool has_kv_seq_len = !kv_seq_len_obj.is_none();
        const bool has_q_ragged_offsets = !q_ragged_offsets_obj.is_none();
        const bool has_kv_ragged_offsets = !kv_ragged_offsets_obj.is_none();
        const bool has_page_table_k = !page_table_k_obj.is_none();
        const bool has_page_table_v = !page_table_v_obj.is_none();
        const bool has_dropout_seed = !dropout_seed_obj.is_none();
        const bool has_dropout_offset = !dropout_offset_obj.is_none();
        const bool uses_dropout = dropout_probability > 0.0f;

        if (has_q_seq_len != has_kv_seq_len) {
            throw std::runtime_error("q_seq_len and kv_seq_len must be provided together for padding-mask attention.");
        }
        if (has_q_ragged_offsets != has_kv_ragged_offsets) {
            throw std::runtime_error("q_ragged_offsets and kv_ragged_offsets must be provided together for ragged attention.");
        }
        if (has_page_table_k != has_page_table_v) {
            throw std::runtime_error("page_table_k and page_table_v must be provided together for paged KV attention.");
        }
        if (has_q_ragged_offsets && has_page_table_k) {
            throw std::runtime_error("ragged attention and paged KV cache cannot be combined.");
        }
        if (has_page_table_k && has_bias) {
            throw std::runtime_error("paged KV attention cannot currently be combined with additive bias.");
        }
        if (has_page_table_k && uses_dropout) {
            throw std::runtime_error("paged KV attention is inference-only and cannot currently be combined with dropout.");
        }
        if (has_page_table_k && paged_kv_max_sequence_length <= 0) {
            throw std::runtime_error("paged KV attention requires paged_kv_max_sequence_length > 0.");
        }
        if (has_page_table_k && !has_q_seq_len) {
            throw std::runtime_error("paged KV attention requires q_seq_len and kv_seq_len.");
        }
        if (has_q_ragged_offsets && !has_q_seq_len) {
            throw std::runtime_error(
                "ragged attention requires q_seq_len and kv_seq_len along with q_ragged_offsets and kv_ragged_offsets.");
        }
        if (has_dropout_seed != has_dropout_offset) {
            throw std::runtime_error("dropout_seed and dropout_offset must be provided together for attention dropout.");
        }
        if (uses_dropout && !has_dropout_seed) {
            throw std::runtime_error("attention dropout_probability > 0 requires dropout_seed and dropout_offset expressions.");
        }
        if (!uses_dropout && has_dropout_seed) {
            throw std::runtime_error("dropout_seed/dropout_offset were provided but attention dropout_probability is zero.");
        }

        if (has_q_seq_len) {
            options.use_padding_mask = true;
        }

        if (has_page_table_k) {
            const Expression& q_seq_len = nb::cast<Expression>(q_seq_len_obj);
            const Expression& kv_seq_len = nb::cast<Expression>(kv_seq_len_obj);
            const Expression& page_table_k = nb::cast<Expression>(page_table_k_obj);
            const Expression& page_table_v = nb::cast<Expression>(page_table_v_obj);
            return Expression::scaledDotProductAttentionPagedKv(
                q, k, v, q_seq_len, kv_seq_len, page_table_k, page_table_v, std::move(options));
        }

        if (has_q_ragged_offsets) {
            const Expression& q_seq_len = nb::cast<Expression>(q_seq_len_obj);
            const Expression& kv_seq_len = nb::cast<Expression>(kv_seq_len_obj);
            const Expression& q_offsets = nb::cast<Expression>(q_ragged_offsets_obj);
            const Expression& kv_offsets = nb::cast<Expression>(kv_ragged_offsets_obj);
            if (uses_dropout) {
                const Expression& dropout_seed = nb::cast<Expression>(dropout_seed_obj);
                const Expression& dropout_offset = nb::cast<Expression>(dropout_offset_obj);
                if (has_bias) {
                    return Expression::scaledDotProductAttentionRagged(q,
                                                                       k,
                                                                       v,
                                                                       nb::cast<Expression>(bias_obj),
                                                                       q_seq_len,
                                                                       kv_seq_len,
                                                                       q_offsets,
                                                                       kv_offsets,
                                                                       dropout_seed,
                                                                       dropout_offset,
                                                                       std::move(options));
                }
                return Expression::scaledDotProductAttentionRagged(
                    q, k, v, q_seq_len, kv_seq_len, q_offsets, kv_offsets, dropout_seed, dropout_offset, std::move(options));
            }
            if (has_bias) {
                return Expression::scaledDotProductAttentionRagged(
                    q, k, v, nb::cast<Expression>(bias_obj), q_seq_len, kv_seq_len, q_offsets, kv_offsets, std::move(options));
            }
            return Expression::scaledDotProductAttentionRagged(q, k, v, q_seq_len, kv_seq_len, q_offsets, kv_offsets, std::move(options));
        }

        if (uses_dropout) {
            const Expression& dropout_seed = nb::cast<Expression>(dropout_seed_obj);
            const Expression& dropout_offset = nb::cast<Expression>(dropout_offset_obj);
            if (has_q_seq_len) {
                const Expression& q_seq_len = nb::cast<Expression>(q_seq_len_obj);
                const Expression& kv_seq_len = nb::cast<Expression>(kv_seq_len_obj);
                if (has_bias) {
                    return Expression::scaledDotProductAttention(
                        q, k, v, nb::cast<Expression>(bias_obj), q_seq_len, kv_seq_len, dropout_seed, dropout_offset, std::move(options));
                }
                return Expression::scaledDotProductAttention(
                    q, k, v, q_seq_len, kv_seq_len, dropout_seed, dropout_offset, std::move(options));
            }
            if (has_bias) {
                return Expression::scaledDotProductAttentionWithDropout(
                    q, k, v, nb::cast<Expression>(bias_obj), dropout_seed, dropout_offset, std::move(options));
            }
            return Expression::scaledDotProductAttentionWithDropout(q, k, v, dropout_seed, dropout_offset, std::move(options));
        }

        if (has_q_seq_len) {
            const Expression& q_seq_len = nb::cast<Expression>(q_seq_len_obj);
            const Expression& kv_seq_len = nb::cast<Expression>(kv_seq_len_obj);
            if (has_bias) {
                return Expression::scaledDotProductAttention(
                    q, k, v, nb::cast<Expression>(bias_obj), q_seq_len, kv_seq_len, std::move(options));
            }
            return Expression::scaledDotProductAttention(q, k, v, q_seq_len, kv_seq_len, std::move(options));
        }
        if (has_bias) {
            return Expression::scaledDotProductAttention(q, k, v, nb::cast<Expression>(bias_obj), std::move(options));
        }
        return Expression::scaledDotProductAttention(q, k, v, std::move(options));
    };

    expr.def_static(
        "rotary_position_embedding",
        [parse_optional_dtype](const Expression& input,
                               uint32_t sequence_axis,
                               uint32_t head_dim_axis,
                               uint64_t rotary_dim,
                               double base,
                               int64_t position_offset,
                               bool interleaved,
                               bool inverse,
                               RotaryScalingKind scaling_kind,
                               double scaling_factor,
                               uint64_t original_max_position_embeddings,
                               std::optional<double> attention_factor,
                               double yarn_beta_fast,
                               double yarn_beta_slow,
                               double llama3_low_freq_factor,
                               double llama3_high_freq_factor,
                               std::vector<double> long_rope_short_factors,
                               std::vector<double> long_rope_long_factors,
                               nb::object output_dtype_obj,
                               nb::object compute_dtype_obj,
                               bool allow_in_place_materialization) {
            RotaryPositionEmbeddingOptions options;
            options.sequence_axis = sequence_axis;
            options.head_dim_axis = head_dim_axis;
            options.rotary_dim = rotary_dim;
            options.base = base;
            options.position_offset = position_offset;
            options.interleaved = interleaved;
            options.inverse = inverse;
            options.scaling_kind = scaling_kind;
            options.scaling_factor = scaling_factor;
            options.original_max_position_embeddings = original_max_position_embeddings;
            options.attention_factor = attention_factor;
            options.yarn_beta_fast = yarn_beta_fast;
            options.yarn_beta_slow = yarn_beta_slow;
            options.llama3_low_freq_factor = llama3_low_freq_factor;
            options.llama3_high_freq_factor = llama3_high_freq_factor;
            options.long_rope_short_factors = std::move(long_rope_short_factors);
            options.long_rope_long_factors = std::move(long_rope_long_factors);
            options.allow_in_place_materialization = allow_in_place_materialization;
            options.output_dtype = parse_optional_dtype(output_dtype_obj);
            options.compute_dtype = parse_optional_dtype(compute_dtype_obj);
            return Expression::rotaryPositionEmbedding(input, std::move(options));
        },
        "input"_a,
        "sequence_axis"_a = 2,
        "head_dim_axis"_a = 3,
        "rotary_dim"_a = 0,
        "base"_a = 10000.0,
        "position_offset"_a = 0,
        "interleaved"_a = false,
        "inverse"_a = false,
        "scaling_kind"_a = RotaryScalingKind::None,
        "scaling_factor"_a = 1.0,
        "original_max_position_embeddings"_a = 0,
        "attention_factor"_a.none() = nb::none(),
        "yarn_beta_fast"_a = 32.0,
        "yarn_beta_slow"_a = 1.0,
        "llama3_low_freq_factor"_a = 1.0,
        "llama3_high_freq_factor"_a = 4.0,
        "long_rope_short_factors"_a = std::vector<double>{},
        "long_rope_long_factors"_a = std::vector<double>{},
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        "allow_in_place_materialization"_a = false,
        R"nbdoc(
Apply rotary positional embedding as a fused expression primitive.

The tensor is interpreted as having a sequence axis and an innermost head-dim
axis.  rotary_dim=0 rotates the full head dimension; otherwise only the leading
rotary_dim channels are rotated.  The inverse flag applies the transpose rotation,
which is used by autodiff.
)nbdoc");

    expr.def_static(
        "rope",
        [parse_optional_dtype](const Expression& input,
                               uint32_t sequence_axis,
                               uint32_t head_dim_axis,
                               uint64_t rotary_dim,
                               double base,
                               int64_t position_offset,
                               bool interleaved,
                               bool inverse,
                               RotaryScalingKind scaling_kind,
                               double scaling_factor,
                               uint64_t original_max_position_embeddings,
                               std::optional<double> attention_factor,
                               double yarn_beta_fast,
                               double yarn_beta_slow,
                               double llama3_low_freq_factor,
                               double llama3_high_freq_factor,
                               std::vector<double> long_rope_short_factors,
                               std::vector<double> long_rope_long_factors,
                               nb::object output_dtype_obj,
                               nb::object compute_dtype_obj,
                               bool allow_in_place_materialization) {
            RotaryPositionEmbeddingOptions options;
            options.sequence_axis = sequence_axis;
            options.head_dim_axis = head_dim_axis;
            options.rotary_dim = rotary_dim;
            options.base = base;
            options.position_offset = position_offset;
            options.interleaved = interleaved;
            options.inverse = inverse;
            options.scaling_kind = scaling_kind;
            options.scaling_factor = scaling_factor;
            options.original_max_position_embeddings = original_max_position_embeddings;
            options.attention_factor = attention_factor;
            options.yarn_beta_fast = yarn_beta_fast;
            options.yarn_beta_slow = yarn_beta_slow;
            options.llama3_low_freq_factor = llama3_low_freq_factor;
            options.llama3_high_freq_factor = llama3_high_freq_factor;
            options.long_rope_short_factors = std::move(long_rope_short_factors);
            options.long_rope_long_factors = std::move(long_rope_long_factors);
            options.allow_in_place_materialization = allow_in_place_materialization;
            options.output_dtype = parse_optional_dtype(output_dtype_obj);
            options.compute_dtype = parse_optional_dtype(compute_dtype_obj);
            return Expression::rotaryPositionEmbedding(input, std::move(options));
        },
        "input"_a,
        "sequence_axis"_a = 2,
        "head_dim_axis"_a = 3,
        "rotary_dim"_a = 0,
        "base"_a = 10000.0,
        "position_offset"_a = 0,
        "interleaved"_a = false,
        "inverse"_a = false,
        "scaling_kind"_a = RotaryScalingKind::None,
        "scaling_factor"_a = 1.0,
        "original_max_position_embeddings"_a = 0,
        "attention_factor"_a.none() = nb::none(),
        "yarn_beta_fast"_a = 32.0,
        "yarn_beta_slow"_a = 1.0,
        "llama3_low_freq_factor"_a = 1.0,
        "llama3_high_freq_factor"_a = 4.0,
        "long_rope_short_factors"_a = std::vector<double>{},
        "long_rope_long_factors"_a = std::vector<double>{},
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        "allow_in_place_materialization"_a = false,
        R"nbdoc(Alias for rotary_position_embedding().)nbdoc");

    expr.def_static(
        "scaled_dot_product_attention",
        [build_attention_from_python_args](const Expression& q,
                                           const Expression& k,
                                           const Expression& v,
                                           AttentionTensorLayout q_layout,
                                           AttentionTensorLayout k_layout,
                                           AttentionTensorLayout v_layout,
                                           AttentionTensorLayout o_layout,
                                           AttentionMaskKind mask_kind,
                                           int64_t diagonal_left_bound,
                                           int64_t diagonal_right_bound,
                                           nb::object attention_scale_obj,
                                           bool use_alibi_mask,
                                           nb::object output_dtype_obj,
                                           nb::object compute_dtype_obj,
                                           nb::object bias_obj,
                                           nb::object q_seq_len_obj,
                                           nb::object kv_seq_len_obj,
                                           nb::object q_ragged_offsets_obj,
                                           nb::object kv_ragged_offsets_obj,
                                           nb::object page_table_k_obj,
                                           nb::object page_table_v_obj,
                                           int64_t paged_kv_max_sequence_length,
                                           float dropout_probability,
                                           nb::object dropout_seed_obj,
                                           nb::object dropout_offset_obj) {
            return build_attention_from_python_args(q,
                                                    k,
                                                    v,
                                                    q_layout,
                                                    k_layout,
                                                    v_layout,
                                                    o_layout,
                                                    mask_kind,
                                                    diagonal_left_bound,
                                                    diagonal_right_bound,
                                                    std::move(attention_scale_obj),
                                                    use_alibi_mask,
                                                    std::move(output_dtype_obj),
                                                    std::move(compute_dtype_obj),
                                                    std::move(bias_obj),
                                                    std::move(q_seq_len_obj),
                                                    std::move(kv_seq_len_obj),
                                                    std::move(q_ragged_offsets_obj),
                                                    std::move(kv_ragged_offsets_obj),
                                                    std::move(page_table_k_obj),
                                                    std::move(page_table_v_obj),
                                                    paged_kv_max_sequence_length,
                                                    dropout_probability,
                                                    std::move(dropout_seed_obj),
                                                    std::move(dropout_offset_obj));
        },
        "q"_a,
        "k"_a,
        "v"_a,
        "q_layout"_a = AttentionTensorLayout::BHSD,
        "k_layout"_a = AttentionTensorLayout::BHSD,
        "v_layout"_a = AttentionTensorLayout::BHSD,
        "o_layout"_a = AttentionTensorLayout::BHSD,
        "mask_kind"_a = AttentionMaskKind::None,
        "diagonal_left_bound"_a = 0,
        "diagonal_right_bound"_a = 0,
        "attention_scale"_a.none() = nb::none(),
        "use_alibi_mask"_a = false,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        "bias"_a.none() = nb::none(),
        "q_seq_len"_a.none() = nb::none(),
        "kv_seq_len"_a.none() = nb::none(),
        "q_ragged_offsets"_a.none() = nb::none(),
        "kv_ragged_offsets"_a.none() = nb::none(),
        "page_table_k"_a.none() = nb::none(),
        "page_table_v"_a.none() = nb::none(),
        "paged_kv_max_sequence_length"_a = 0,
        "dropout_probability"_a = 0.0f,
        "dropout_seed"_a.none() = nb::none(),
        "dropout_offset"_a.none() = nb::none(),
        R"nbdoc(
Create a cuDNN scaled-dot-product attention expression stage.

All tensors use semantic shape ``[B, H, S, D]``.  Layout arguments describe how
those tensors are handed to cuDNN.  The default layout is BHSD, matching Thor's
row-major physical tensor layout for rank-4 attention inputs.  ``output_dtype``
should normally match Q/K/V for the current cuDNN SDPA path; ``compute_dtype``
should normally be ``thor.DataType.fp32``.

FP16/BF16 production support:

* Q/K/V/O must all use the same FP16 or BF16 dtype.  Forward and backward are
  supported for self-attention, cross-attention, MHA, GQA, and MQA.
* Supported masks are ``none``, ``causal_top_left``, ``causal_bottom_right``,
  ``sliding_window_top_left``, and ``sliding_window_bottom_right``.
* ALiBi requires a causal/sliding diagonal mask and ``diagonal_right_bound == 0``.
* ``bias`` is additive score-space bias in ``[1|B, 1|Hq, 1|Sq, 1|Skv]`` semantic
  order and must use the compute dtype.  Forward supports sequence broadcast.
  Backward materializes sequence-broadcast bias to dense score space before
  cuDNN backward, then explicitly reduces dBias back to the requested bias shape.
* When ``q_ragged_offsets`` and ``kv_ragged_offsets`` are provided, they must be
  int32 GPU tensors with shape ``[B + 1]``.  They enable cuDNN packed/ragged
  variable-length attention and are passed through as Q/O and K/V ragged offsets.
  Ragged + additive-bias forward is supported, but ragged + additive-bias
  backward is rejected.
* When ``dropout_probability > 0``, ``dropout_seed`` and ``dropout_offset`` must
  be int64 GPU scalar expressions with shape ``[1, 1, 1, 1]``.  They are passed
  to cuDNN's Philox attention dropout path.

Paged KV cache:

* ``page_table_k`` and ``page_table_v`` must be int32 GPU tensors with shape
  ``[B, 1, ceil(Skv / block_size), 1]``.  Paged-KV attention requires
  ``q_seq_len`` and ``kv_seq_len`` and a positive ``paged_kv_max_sequence_length``.
* The production paged-KV path is FP16/BF16 forward-only/inference-only.  Bias,
  dropout, ragged offsets, and backward are rejected for paged KV.

FP8 support:

* FP8 is exposed by the lower-level FP8-specific expression path and by
  ``thor.layers.ScaledDotProductAttention`` with explicit scale/descale/amax
  tensors.  This generic expression wrapper documents the same validated surface:
  forward-only, same FP8 format for Q/K/V/O, head dimensions multiples of 16 and
  ``<= 128``, no additive bias, no dropout, no ALiBi, no ragged, no paged KV, no
  bottom-right/decode or sliding-window masks, and no decode-style ``Sq=1, Skv>1``.
* FP8 padding masks / sequence lengths are supported for forward.

Important combination rules:

* Bottom-right/decode masks currently require additive bias, ALiBi, and dropout
  to be disabled in the production cuDNN primary SDPA path.
* Experimental cuDNN support-surface probe environment variables can bypass some
  guards for measurement only; probe-only combinations are not support guarantees.
)nbdoc");

    expr.def_static(
        "attention",
        [build_attention_from_python_args](const Expression& q,
                                           const Expression& k,
                                           const Expression& v,
                                           AttentionTensorLayout q_layout,
                                           AttentionTensorLayout k_layout,
                                           AttentionTensorLayout v_layout,
                                           AttentionTensorLayout o_layout,
                                           AttentionMaskKind mask_kind,
                                           int64_t diagonal_left_bound,
                                           int64_t diagonal_right_bound,
                                           nb::object attention_scale_obj,
                                           bool use_alibi_mask,
                                           nb::object output_dtype_obj,
                                           nb::object compute_dtype_obj,
                                           nb::object bias_obj,
                                           nb::object q_seq_len_obj,
                                           nb::object kv_seq_len_obj,
                                           nb::object q_ragged_offsets_obj,
                                           nb::object kv_ragged_offsets_obj,
                                           nb::object page_table_k_obj,
                                           nb::object page_table_v_obj,
                                           int64_t paged_kv_max_sequence_length,
                                           float dropout_probability,
                                           nb::object dropout_seed_obj,
                                           nb::object dropout_offset_obj) {
            return build_attention_from_python_args(q,
                                                    k,
                                                    v,
                                                    q_layout,
                                                    k_layout,
                                                    v_layout,
                                                    o_layout,
                                                    mask_kind,
                                                    diagonal_left_bound,
                                                    diagonal_right_bound,
                                                    std::move(attention_scale_obj),
                                                    use_alibi_mask,
                                                    std::move(output_dtype_obj),
                                                    std::move(compute_dtype_obj),
                                                    std::move(bias_obj),
                                                    std::move(q_seq_len_obj),
                                                    std::move(kv_seq_len_obj),
                                                    std::move(q_ragged_offsets_obj),
                                                    std::move(kv_ragged_offsets_obj),
                                                    std::move(page_table_k_obj),
                                                    std::move(page_table_v_obj),
                                                    paged_kv_max_sequence_length,
                                                    dropout_probability,
                                                    std::move(dropout_seed_obj),
                                                    std::move(dropout_offset_obj));
        },
        "q"_a,
        "k"_a,
        "v"_a,
        "q_layout"_a = AttentionTensorLayout::BHSD,
        "k_layout"_a = AttentionTensorLayout::BHSD,
        "v_layout"_a = AttentionTensorLayout::BHSD,
        "o_layout"_a = AttentionTensorLayout::BHSD,
        "mask_kind"_a = AttentionMaskKind::None,
        "diagonal_left_bound"_a = 0,
        "diagonal_right_bound"_a = 0,
        "attention_scale"_a.none() = nb::none(),
        "use_alibi_mask"_a = false,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        "bias"_a.none() = nb::none(),
        "q_seq_len"_a.none() = nb::none(),
        "kv_seq_len"_a.none() = nb::none(),
        "q_ragged_offsets"_a.none() = nb::none(),
        "kv_ragged_offsets"_a.none() = nb::none(),
        "page_table_k"_a.none() = nb::none(),
        "page_table_v"_a.none() = nb::none(),
        "paged_kv_max_sequence_length"_a = 0,
        "dropout_probability"_a = 0.0f,
        "dropout_seed"_a.none() = nb::none(),
        "dropout_offset"_a.none() = nb::none(),
        R"nbdoc(Alias for scaled_dot_product_attention().)nbdoc");
    expr.def("__rpow__", [](const Expression& a, const Expression& b) { return b.pow(a); }, "other"_a);

    expr.def("__neg__", [](const Expression& a) { return -a; });

    expr.def_static("min", [](const Expression& a, const Expression& b) { return a.min(b); }, "a"_a, "b"_a);
    expr.def_static("max", [](const Expression& a, const Expression& b) { return a.max(b); }, "a"_a, "b"_a);

    expr.def_static(
        "abs",
        [](const Expression& x) { return x.abs(); },
        "x"_a,
        R"nbdoc(
Return the absolute value of the input expression x
)nbdoc");

    expr.def_static("exp", [](const Expression& x) { return x.exp(); }, "x"_a);
    expr.def_static("expm1", [](const Expression& x) { return x.expm1(); }, "x"_a);
    expr.def_static("exp2", [](const Expression& x) { return x.exp2(); }, "x"_a);
    expr.def_static("exp10", [](const Expression& x) { return x.exp10(); }, "x"_a);

    expr.def_static(
        "ln",
        [](const Expression& x) { return x.ln(); },
        "x"_a,
        R"nbdoc(
Return the elementwise natural logarithm of the input expression x
)nbdoc");
    expr.def_static("log", [](const Expression& x, double base) { return x.log(base); }, "x"_a, "base"_a = std::numbers::e);
    expr.def_static("log1p", [](const Expression& x) { return x.log1p(); }, "x"_a);
    expr.def_static("log2", [](const Expression& x) { return x.log2(); }, "x"_a);
    expr.def_static("log10", [](const Expression& x) { return x.log10(); }, "x"_a);

    expr.def_static(
        "sqrt",
        [](const Expression& x) { return x.sqrt(); },
        "x"_a,
        R"nbdoc(
Return the elementwise square root of the input expression x
)nbdoc");

    expr.def_static(
        "tanh",
        [](const Expression& x) { return x.tanh(); },
        "x"_a,
        R"nbdoc(
Return the elementwise hyperbolic tangent of the input expression x.

This lowers to Thor's TANH expression op, which is emitted with CUDA's built-in tanh implementation.
)nbdoc");

    expr.def_static(
        "normcdf",
        [](const Expression& x) { return x.normcdf(); },
        "x"_a,
        R"nbdoc(
Return the elementwise standard normal CDF of the input expression x.

This lowers to Thor's NORMCDF expression op, which is emitted with CUDA's built-in normcdf implementation.
)nbdoc");

    auto parse_softmax_mode = [](const std::string& mode) -> cudnnSoftmaxMode_t {
        if (mode == "channel") {
            return CUDNN_SOFTMAX_MODE_CHANNEL;
        }
        if (mode == "instance") {
            return CUDNN_SOFTMAX_MODE_INSTANCE;
        }
        throw std::runtime_error("softmax mode must be 'channel' or 'instance'.");
    };

    auto parse_softmax_algorithm = [](const std::string& algorithm) -> cudnnSoftmaxAlgorithm_t {
        if (algorithm == "accurate") {
            return CUDNN_SOFTMAX_ACCURATE;
        }
        if (algorithm == "fast") {
            return CUDNN_SOFTMAX_FAST;
        }
        throw std::runtime_error("softmax algorithm must be 'accurate' or 'fast'. Use log_softmax for CUDNN_SOFTMAX_LOG.");
    };

    expr.def_static(
        "softmax",
        [parse_softmax_mode, parse_softmax_algorithm](const Expression& x, const std::string& algorithm, const std::string& mode) {
            return x.softmax(parse_softmax_algorithm(algorithm), parse_softmax_mode(mode));
        },
        "x"_a,
        "algorithm"_a = "accurate",
        "mode"_a = "channel",
        R"nbdoc(
Return cuDNN softmax of the input expression x.

algorithm may be 'accurate' (default) or 'fast'. Log-softmax is a different operation; use log_softmax().
mode may be 'channel' (default) or 'instance'.
)nbdoc");

    expr.def_static(
        "log_softmax",
        [parse_softmax_mode](const Expression& x, const std::string& mode) { return x.logSoftmax(parse_softmax_mode(mode)); },
        "x"_a,
        "mode"_a = "channel",
        R"nbdoc(
Return cuDNN log-softmax of the input expression x.
)nbdoc");

    // Reductions
    auto parse_axes = [](const nb::object& axis) -> std::vector<uint64_t> {
        if (axis.is_none()) {
            return {};
        }
        if (nb::isinstance<nb::int_>(axis)) {
            return {nb::cast<uint64_t>(axis)};
        }
        return nb::cast<std::vector<uint64_t>>(axis);
    };

    expr.def_static(
        "unsqueeze",
        [parse_axes](const Expression& x, const nb::object& axis) { return x.unsqueeze(parse_axes(axis)); },
        "x"_a,
        "axis"_a,
        R"nbdoc(
Insert singleton dimensions at the specified output axes.

Parameters
----------
x : thor.physical.Expression
    Input expression.
axis : int | list[int]
    Output-axis positions at which singleton dimensions of size 1 are inserted.

Returns
-------
thor.physical.Expression
    An Expression that views the same logical values with the requested singleton axes inserted.
)nbdoc");

    expr.def_static(
        "squeeze",
        [parse_axes](const Expression& x, const nb::object& axis) {
            std::vector<uint64_t> all_singletons{UINT64_MAX};
            return x.squeeze(axis.is_none() ? all_singletons : parse_axes(axis));
        },
        "x"_a,
        "axis"_a.none() = nb::none(),
        R"nbdoc(
Remove singleton dimensions at the specified axes.

Parameters
----------
x : thor.physical.Expression
    Input expression.
axis : int | list[int]
    Axes that must be singleton dimensions of size 1.

Returns
-------
thor.physical.Expression
    An Expression that views the same logical values with the requested singleton axes removed.
)nbdoc");

    auto parse_squeeze_axes = [](const nb::object& squeeze) -> std::vector<uint64_t> {
        if (squeeze.is_none()) {
            return {};
        }

        if (nb::isinstance<nb::bool_>(squeeze)) {
            bool b = nb::cast<bool>(squeeze);
            if (!b) {
                return {};
            }
            return {UINT64_MAX};  // sentinel meaning "squeeze all singleton dims"
        }

        if (nb::isinstance<nb::int_>(squeeze)) {
            return {nb::cast<uint64_t>(squeeze)};
        }

        return nb::cast<std::vector<uint64_t>>(squeeze);
    };

    auto parse_reduction_compute_dtype = [](const std::string_view& op_name,
                                            const std::optional<DataType>& compute_dtype) -> std::optional<DataType> {
        // if (compute_dtype.has_value() && compute_dtype.value() != DataType::FP32) {
        //     throw std::runtime_error(std::string(op_name) + ": currently only supports compute_dtype=thor.DataType.fp32");
        // }
        return DataType::FP32;
    };

    auto parse_reduction_output_dtype = [](std::string_view op_name, std::optional<DataType> compute_dtype) -> std::optional<DataType> {
        // if (compute_dtype.has_value() && compute_dtype.value() != DataType::FP32) {
        //     throw std::runtime_error(std::string(op_name) + ": currently only supports output_dtype=thor.DataType.fp32");
        // }
        return DataType::FP32;
    };

    auto parse_arg_reduction_output_dtype = [](std::string_view op_name, std::optional<DataType> output_dtype) -> std::optional<DataType> {
        // if (output_dtype.has_value() && output_dtype.value() != DataType::UINT32) {
        //     throw std::runtime_error(std::string(op_name) + ": currently only supports output_dtype=thor.DataType.uint32");
        // }
        return DataType::UINT32;
    };

    static constexpr std::string_view kReductionDocTemplate = R"doc(
Reduce by {} across the specified axes.

Args:
    axis: int | list[int] | None
        Single axis or sequence of axes to reduce. If None, reduce across all axes.
    squeeze: bool | int | list[int]
        If False, keep reduced axes as singleton dimensions.
        If True, remove all singleton dimensions after reduction.
        If an int or sequence of ints, remove those specific singleton axes after reduction.
    compute_dtype: thor.DataType: default thor.DataType.fp32
        The data type used during compute. Currently only fp32 is supported for this operation.
    output_dtype: thor.DataType: default thor.DataType.fp32
        The data type that is written back to memory. Currently only fp32 is supported for this operation.
)doc";

    static constexpr std::string_view kArgReductionDocTemplate = R"doc(
Return the flattened index of the {} across the specified axes.

Args:
    axis: int | list[int] | None
        Single axis or sequence of axes to reduce. If None, reduce across all axes.
    squeeze: bool | int | list[int]
        If False, keep reduced axes as singleton dimensions.
        If True, remove all singleton dimensions after reduction.
        If an int or sequence of ints, remove those specific singleton axes after reduction.
    compute_dtype: thor.DataType: default thor.DataType.fp32
        The data type used during compute. Currently only fp32 is supported for this operation.
    output_dtype: thor.DataType: default thor.DataType.uint32
        The flattened reduced-space index dtype written back to memory. Currently only uint32 is supported.
)doc";

    std::string reduce_sum_doc = std::format(kReductionDocTemplate, "summation");
    expr.def_static(
        "reduce_sum",
        [parse_axes, parse_squeeze_axes, parse_reduction_compute_dtype, parse_reduction_output_dtype](
            const Expression& expr,
            nb::object axis,
            nb::object squeeze,
            std::optional<DataType> compute_dtype,
            std::optional<DataType> output_dtype) {
            parse_reduction_output_dtype("reduce_sum", output_dtype);
            return expr.reduce_sum(
                parse_axes(axis), parse_squeeze_axes(squeeze), parse_reduction_compute_dtype("reduce_sum", compute_dtype));
        },
        "expr"_a,
        "axis"_a = nb::none(),
        "squeeze"_a = false,
        "compute_dtype"_a.none() = DataType::FP32,
        "output_dtype"_a.none() = DataType::FP32,
        reduce_sum_doc.c_str());

    std::string reduce_prod_doc = std::format(kReductionDocTemplate, "product");
    expr.def_static(
        "reduce_prod",
        [parse_axes, parse_squeeze_axes, parse_reduction_compute_dtype, parse_reduction_output_dtype](
            const Expression& expr,
            nb::object axis,
            nb::object squeeze,
            std::optional<DataType> compute_dtype,
            std::optional<DataType> output_dtype) {
            parse_reduction_output_dtype("reduce_prod", output_dtype);
            return expr.reduce_prod(
                parse_axes(axis), parse_squeeze_axes(squeeze), parse_reduction_compute_dtype("reduce_prod", compute_dtype));
        },
        "expr"_a,
        "axis"_a = nb::none(),
        "squeeze"_a = false,
        "compute_dtype"_a.none() = DataType::FP32,
        "output_dtype"_a.none() = DataType::FP32,
        reduce_prod_doc.c_str());

    std::string reduce_min_doc = std::format(kReductionDocTemplate, "minimum");
    expr.def_static(
        "reduce_min",
        [parse_axes, parse_squeeze_axes, parse_reduction_compute_dtype, parse_reduction_output_dtype](
            const Expression& expr,
            nb::object axis,
            nb::object squeeze,
            std::optional<DataType> compute_dtype,
            std::optional<DataType> output_dtype) {
            parse_reduction_output_dtype("reduce_min", output_dtype);
            return expr.reduce_min(
                parse_axes(axis), parse_squeeze_axes(squeeze), parse_reduction_compute_dtype("reduce_min", compute_dtype));
        },
        "expr"_a,
        "axis"_a = nb::none(),
        "squeeze"_a = false,
        "compute_dtype"_a.none() = DataType::FP32,
        "output_dtype"_a.none() = DataType::FP32,
        reduce_min_doc.c_str());

    std::string reduce_max_doc = std::format(kReductionDocTemplate, "maximum");
    expr.def_static(
        "reduce_max",
        [parse_axes, parse_squeeze_axes, parse_reduction_compute_dtype, parse_reduction_output_dtype](
            const Expression& expr,
            nb::object axis,
            nb::object squeeze,
            std::optional<DataType> compute_dtype,
            std::optional<DataType> output_dtype) {
            parse_reduction_output_dtype("reduce_max", output_dtype);
            return expr.reduce_max(
                parse_axes(axis), parse_squeeze_axes(squeeze), parse_reduction_compute_dtype("reduce_max", compute_dtype));
        },
        "expr"_a,
        "axis"_a = nb::none(),
        "squeeze"_a = false,
        "compute_dtype"_a.none() = DataType::FP32,
        "output_dtype"_a.none() = DataType::FP32,
        reduce_max_doc.c_str());

    std::string argmin_doc = std::format(kArgReductionDocTemplate, "minimum");
    expr.def_static(
        "argmin",
        [parse_axes, parse_squeeze_axes, parse_reduction_compute_dtype, parse_arg_reduction_output_dtype](
            const Expression& expr,
            nb::object axis,
            nb::object squeeze,
            std::optional<DataType> compute_dtype,
            std::optional<DataType> output_dtype) {
            parse_arg_reduction_output_dtype("argmin", output_dtype);
            return expr.argmin(parse_axes(axis), parse_squeeze_axes(squeeze), parse_reduction_compute_dtype("argmin", compute_dtype));
        },
        "expr"_a,
        "axis"_a = nb::none(),
        "squeeze"_a = false,
        "compute_dtype"_a.none() = DataType::FP32,
        "output_dtype"_a.none() = DataType::UINT32,
        argmin_doc.c_str());

    std::string argmax_doc = std::format(kArgReductionDocTemplate, "maximum");
    expr.def_static(
        "argmax",
        [parse_axes, parse_squeeze_axes, parse_reduction_compute_dtype, parse_arg_reduction_output_dtype](
            const Expression& expr,
            nb::object axis,
            nb::object squeeze,
            std::optional<DataType> compute_dtype,
            std::optional<DataType> output_dtype) {
            parse_arg_reduction_output_dtype("argmax", output_dtype);
            return expr.argmax(parse_axes(axis), parse_squeeze_axes(squeeze), parse_reduction_compute_dtype("argmax", compute_dtype));
        },
        "expr"_a,
        "axis"_a = nb::none(),
        "squeeze"_a = false,
        "compute_dtype"_a.none() = DataType::FP32,
        "output_dtype"_a.none() = DataType::UINT32,
        argmax_doc.c_str());

    std::string reduce_mean_doc = std::format(kReductionDocTemplate, "arithmetic mean");
    expr.def_static(
        "reduce_mean",
        [parse_axes, parse_squeeze_axes, parse_reduction_compute_dtype, parse_reduction_output_dtype](
            const Expression& expr,
            nb::object axis,
            nb::object squeeze,
            std::optional<DataType> compute_dtype,
            std::optional<DataType> output_dtype) {
            parse_reduction_output_dtype("reduce_mean", output_dtype);
            return expr.reduce_mean(
                parse_axes(axis), parse_squeeze_axes(squeeze), parse_reduction_compute_dtype("reduce_mean", compute_dtype));
        },
        "expr"_a,
        "axis"_a = nb::none(),
        "squeeze"_a = false,
        "compute_dtype"_a.none() = DataType::FP32,
        "output_dtype"_a.none() = DataType::FP32,
        reduce_mean_doc.c_str());

    std::string reduce_norm1_doc = std::format(kReductionDocTemplate, "L1 norm");
    expr.def_static(
        "reduce_norm1",
        [parse_axes, parse_squeeze_axes, parse_reduction_compute_dtype, parse_reduction_output_dtype](
            const Expression& expr,
            nb::object axis,
            nb::object squeeze,
            std::optional<DataType> compute_dtype,
            std::optional<DataType> output_dtype) {
            parse_reduction_output_dtype("reduce_norm1", output_dtype);
            return expr.reduce_norm1(
                parse_axes(axis), parse_squeeze_axes(squeeze), parse_reduction_compute_dtype("reduce_norm1", compute_dtype));
        },
        "expr"_a,
        "axis"_a = nb::none(),
        "squeeze"_a = false,
        "compute_dtype"_a.none() = DataType::FP32,
        "output_dtype"_a.none() = DataType::FP32,
        reduce_norm1_doc.c_str());

    std::string reduce_norm2_doc = std::format(kReductionDocTemplate, "L2 norm");
    expr.def_static(
        "reduce_norm2",
        [parse_axes, parse_squeeze_axes, parse_reduction_compute_dtype, parse_reduction_output_dtype](
            const Expression& expr,
            nb::object axis,
            nb::object squeeze,
            std::optional<DataType> compute_dtype,
            std::optional<DataType> output_dtype) {
            parse_reduction_output_dtype("reduce_norm2", output_dtype);
            return expr.reduce_norm2(
                parse_axes(axis), parse_squeeze_axes(squeeze), parse_reduction_compute_dtype("reduce_norm", compute_dtype));
        },
        "expr"_a,
        "axis"_a = nb::none(),
        "squeeze"_a = false,
        "compute_dtype"_a.none() = DataType::FP32,
        "output_dtype"_a.none() = DataType::FP32,
        reduce_norm2_doc.c_str());

    auto outputs_type = nb::class_<Outputs>(physical, "Outputs");
    outputs_type.attr("__module__") = "thor.physical";
    outputs_type
        .def(
            "compile",
            [](const Outputs& self, int device_num, bool use_fast_math) {
                nb::gil_scoped_release release;
                return FusedEquation::compile(self.physicalOutputs(), device_num, use_fast_math);
            },
            "device_num"_a = 0,
            "use_fast_math"_a = false)
        .def("to_json", [](const Outputs& self) { return ExpressionDefinition::fromOutputs(self).architectureJsonWithCudaKernelManifestSignature().dump(); })
        .def_static(
            "from_json",
            [](const std::string& payload,
               bool allow_unsafe_loaded_cuda_kernel_source,
               const std::string& trusted_cuda_kernel_public_key,
               const std::string& trusted_cuda_kernel_source_decryption_key) {
                return Outputs::fromPhysicalOutputs(
                    ExpressionDefinition::deserialize(nlohmann::json::parse(payload),
                                                      allow_unsafe_loaded_cuda_kernel_source,
                                                      trusted_cuda_kernel_public_key,
                                                      trusted_cuda_kernel_source_decryption_key)
                        .outputs);
            },
            "payload"_a,
            "allow_unsafe_loaded_cuda_kernel_source"_a = false,
            "trusted_cuda_kernel_public_key"_a = "",
            "trusted_cuda_kernel_source_decryption_key"_a = "")
        .def("output_names", [](const Outputs& self) {
            std::vector<std::string> names;
            for (const NamedOutput& output : self.namedOutputs()) {
                names.push_back(output.name);
            }
            return names;
        });

    auto expression_definition_type = nb::class_<ExpressionDefinition>(physical, "ExpressionDefinition");
    expression_definition_type.attr("__module__") = "thor.physical";
    expression_definition_type.def_static("from_outputs", &ExpressionDefinition::fromOutputs, "outputs"_a)
        .def("to_json", [](const ExpressionDefinition& self) { return self.architectureJsonWithCudaKernelManifestSignature().dump(); })
        .def_static(
            "from_json",
            [](const std::string& payload,
               bool allow_unsafe_loaded_cuda_kernel_source,
               const std::string& trusted_cuda_kernel_public_key,
               const std::string& trusted_cuda_kernel_source_decryption_key) {
                return ExpressionDefinition::deserialize(nlohmann::json::parse(payload),
                                                         allow_unsafe_loaded_cuda_kernel_source,
                                                         trusted_cuda_kernel_public_key,
                                                         trusted_cuda_kernel_source_decryption_key);
            },
            "payload"_a,
            "allow_unsafe_loaded_cuda_kernel_source"_a = false,
            "trusted_cuda_kernel_public_key"_a = "",
            "trusted_cuda_kernel_source_decryption_key"_a = "")
        .def("cuda_kernel_source_info", [](const ExpressionDefinition& self) { return cudaKernelSourceInspectionListToPython(self.cudaKernelSourceInfo()); })
        .def("cuda_kernel_sources", &ExpressionDefinition::cudaKernelSources)
        .def("cuda_kernel_source_info_json", [](const ExpressionDefinition& self) { return self.cudaKernelSourceInfoJson().dump(2); })
        .def("cuda_kernel_signing_public_keys", &ExpressionDefinition::cudaKernelSigningPublicKeys)
        .def("cuda_kernel_out_of_band_keys", [](const ExpressionDefinition& self) { return cudaKernelOutOfBandKeysToPython(self.cudaKernelOutOfBandKeys()); })
        .def("allow_unsafe_loaded_cuda_kernel_source_compilation",
             [](ExpressionDefinition& self,
                const std::string& trusted_cuda_kernel_public_key,
                const std::string& trusted_cuda_kernel_source_decryption_key) {
                 self.allowUnsafeLoadedCudaKernelSourceCompilation(trusted_cuda_kernel_public_key, trusted_cuda_kernel_source_decryption_key);
             },
             "trusted_cuda_kernel_public_key"_a,
             "trusted_cuda_kernel_source_decryption_key"_a = "")
        .def_prop_ro("has_cuda_kernel_expressions", &ExpressionDefinition::hasCudaKernelExpressions)
        .def_prop_ro("expected_input_names", [](const ExpressionDefinition& self) { return self.expected_input_names; })
        .def_prop_ro("expected_output_names", [](const ExpressionDefinition& self) { return self.expected_output_names; })
        .def_prop_ro("canonical_hash", [](const ExpressionDefinition& self) { return self.canonical_hash; });

    expr.def_static(
        "outputs",
        [](nb::dict mapping) {
            std::vector<std::pair<std::string, Expression>> named_exprs;
            named_exprs.reserve(mapping.size());

            for (auto item : mapping) {
                nb::handle key = item.first;
                nb::handle value = item.second;

                if (!nb::isinstance<nb::str>(key)) {
                    throw std::runtime_error("Expression.outputs keys must be strings.");
                }
                if (!nb::isinstance<Expression>(value)) {
                    throw std::runtime_error("Expression.outputs values must be Expression objects.");
                }

                std::string name = nb::cast<std::string>(key);
                Expression out_expr = nb::cast<Expression>(value);

                named_exprs.emplace_back(std::move(name), std::move(out_expr));
            }

            return Expression::outputs(named_exprs);
        },
        "outputs"_a,
        R"nbdoc(
Create a terminal multi-output graph from a mapping of output names to expressions.

Args:
    outputs: dict[str, Expression]
        Mapping from output names to expressions. All expressions must belong to the same graph.

Returns:
    Outputs
        A terminal multi-output graph object that can be compiled together.
)nbdoc");

    expr.def_static(
        "compile",
        [](const Expression& expr, int device_num, bool use_fast_math) {
            nb::gil_scoped_release release;
            return FusedEquation::compile(expr.expression(), device_num, use_fast_math);
        },
        "expr"_a,
        "device_num"_a = 0,
        "use_fast_math"_a = false,
        R"nbdoc(
Compile an expression into a fused equation.

Parameters
----------
expr : thor.physical.Expression
    The expression to compile.
dtype : thor.DataType
    The tensor data type to target.
device_num : int, default 0
    The GPU device number.
use_fast_math : bool, default False
    Whether to enable fast-math optimizations during compilation.

Returns
-------
thor.physical.FusedEquation
    The compiled fused equation.
)nbdoc");

    expr.def_static(
        "compile",
        [](const Outputs& self, int device_num, bool use_fast_math) {
            nb::gil_scoped_release release;
            return FusedEquation::compile(self.physicalOutputs(), device_num, use_fast_math);
        },
        "expr"_a,
        "device_num"_a = 0,
        "use_fast_math"_a = false,
        R"nbdoc(
Compile an expression into a fused equation.

Parameters
----------
expr : thor.physical.Expression
    The expression to compile.
device_num : int, default 0
    The GPU device number.
use_fast_math : bool, default False
    Whether to enable fast-math optimizations during compilation.

Returns
-------
thor.physical.FusedEquation
    The compiled fused equation.
)nbdoc");
}

void bind_fused_equation(nb::module_& physical) {
    auto fused_equation = nb::class_<FusedEquation>(physical, "FusedEquation");
    fused_equation.attr("__module__") = "thor.physical";

    fused_equation.def(
        "stamp",
        [](const FusedEquation& self,
           const std::unordered_map<std::string, Tensor>& inputs,
           const Stream& stream,
           const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
           const std::optional<Tensor>& preallocated_output,
           const std::vector<uint64_t>& requestedOutputShape) {
            if (!preallocated_output.has_value() && requestedOutputShape.empty()) {
                // No single-output-only arguments were supplied.
                // Route through the general multi-output stamp path, which also works for single-output equations.
                const std::unordered_map<std::string, Tensor> preallocated_outputs{};
                const std::unordered_map<std::string, std::vector<uint64_t>> requestedOutputShapes{};
                return self.stamp(inputs, stream, tensor_scalar_inputs, preallocated_outputs, requestedOutputShapes);
            } else {
                return self.stampSingleOutput(inputs, stream, tensor_scalar_inputs, preallocated_output, requestedOutputShape);
            }
        },
        "inputs"_a,
        "stream"_a,
        nb::kw_only(),
        "tensor_scalar_inputs"_a = std::unordered_map<std::string, TensorScalarBinding>{},
        "preallocated_output"_a.none() = nb::none(),
        "requested_output_shape"_a = std::vector<uint64_t>{},
        R"nbdoc(
Create an executable instance of a fused equation.

Parameters
----------
inputs : dict[str, PhysicalTensor]
    Mapping from input names to tensors.
stream : thor.Stream
    Stream on which to stamp the equation.
tensor_scalar_inputs : dict[str, TensorScalarBinding], optional
    GPU-backed runtime scalar bindings.
preallocated_output : PhysicalTensor | None, optional
    Preallocated output tensor for the single output.
requested_output_shape : list[int], optional
    Requested output shape for the single output.

Returns
-------
thor.physical.Equation
    A stamped execution plan.
)nbdoc");

    fused_equation.def(
        "stamp",
        [](const FusedEquation& self,
           const std::unordered_map<std::string, Tensor>& inputs,
           const Stream& stream,
           const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
           const std::unordered_map<std::string, Tensor>& preallocated_outputs,
           const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) {
            return self.stamp(inputs, stream, tensor_scalar_inputs, preallocated_outputs, requestedOutputShapes);
        },
        "inputs"_a,
        "stream"_a,
        nb::kw_only(),
        "tensor_scalar_inputs"_a = std::unordered_map<std::string, TensorScalarBinding>{},
        "preallocated_outputs"_a = std::unordered_map<std::string, Tensor>{},
        "requested_output_shapes"_a = std::unordered_map<std::string, std::vector<uint64_t>>{},
        R"nbdoc(
Create an executable instance of a multi-output fused equation.

Parameters
----------
inputs : dict[str, PhysicalTensor]
    Mapping from input names to tensors.
stream : thor.Stream
    Stream on which to stamp the equation.
tensor_scalar_inputs : dict[str, TensorScalarBinding], optional
    GPU-backed runtime scalar bindings.
preallocated_outputs : dict[str, PhysicalTensor], optional
    Mapping from output names to preallocated output tensors.
requested_output_shapes : dict[str, list[int]], optional
    Mapping from output names to requested output shapes.

Returns
-------
thor.physical.Equation
    A stamped execution plan.
)nbdoc");

    fused_equation.def("run",
                       nb::overload_cast<const Tensor&, Tensor&, Stream&>(&FusedEquation::run, nb::const_),
                       "input"_a,
                       "output"_a,
                       "stream"_a,
                       R"nbdoc(
Run a fused equation with the thor.physical.PhysicalTensor's provided.

input: PhysicalTensor
output: PhysicalTensor
)nbdoc");

    fused_equation.def(
        "get_parameter_fan_overrides",
        [](const FusedEquation& self,
           const std::unordered_map<std::string, Tensor>& inputs,
           const std::vector<std::string>& parameter_names,
           const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
           const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes) {
            const std::unordered_set<std::string> parameter_name_set(parameter_names.begin(), parameter_names.end());
            return parameterFanOverridesToPython(
                self.getParameterFanOverrides(inputs, parameter_name_set, tensor_scalar_inputs, requested_output_shapes));
        },
        "inputs"_a,
        "parameter_names"_a,
        nb::kw_only(),
        "tensor_scalar_inputs"_a = std::unordered_map<std::string, TensorScalarBinding>{},
        "requested_output_shapes"_a = std::unordered_map<std::string, std::vector<uint64_t>>{},
        R"nbdoc(
Infer parameter initializer fan-in/fan-out overrides for the named parameter inputs.

Returns
-------
dict[str, dict[str, int]]
    Mapping from parameter name to {"fan_in": int, "fan_out": int}.
)nbdoc");

    fused_equation.def(
        "run",
        nb::overload_cast<const std::unordered_map<std::string, Tensor>&, const std::unordered_map<std::string, float>&, Tensor&, Stream&>(
            &FusedEquation::run, nb::const_),
        "inputs"_a,
        "scalar_inputs"_a,
        "output"_a,
        "stream"_a,
        R"nbdoc(
Run a fused equation with bound tensor and runtime scalar inputs.
)nbdoc");

    fused_equation.def("run",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&, Tensor&, Stream&>(&FusedEquation::run, nb::const_),
                       "inputs"_a,
                       "output"_a,
                       "stream"_a,
                       R"nbdoc(
Run a fused equation with the thor.physical.PhysicalTensor's provided.

inputs: dict[str, PhysicalTensor]
    A dict mapping input names to tensors
output: PhysicalTensor
)nbdoc");

    fused_equation.def("run",
                       nb::overload_cast<const Tensor&, std::unordered_map<std::string, Tensor>&, Stream&>(&FusedEquation::run, nb::const_),
                       "input"_a,
                       "outputs"_a,
                       "stream"_a,
                       R"nbdoc(
Run a fused equation with the thor.physical.PhysicalTensor's provided.

input: PhysicalTensor
outputs: dict[str, PhysicalTensor]
    A dict mapping output names to tensors
)nbdoc");

    fused_equation.def("run",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&,
                                         const std::unordered_map<std::string, float>&,
                                         std::unordered_map<std::string, Tensor>&,
                                         Stream&>(&FusedEquation::run, nb::const_),
                       "inputs"_a,
                       "scalar_inputs"_a,
                       "outputs"_a,
                       "stream"_a,
                       R"nbdoc(
Run a fused equation with bound tensor and runtime scalar inputs.
)nbdoc");

    fused_equation.def("run",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&, std::unordered_map<std::string, Tensor>&, Stream&>(
                           &FusedEquation::run, nb::const_),
                       "inputs"_a,
                       "outputs"_a,
                       "stream"_a,
                       R"nbdoc(
Run a fused equation with the thor.physical.PhysicalTensor's provided.

inputs: dict[str, PhysicalTensor]
    A dict mapping input names to tensors
outputs: dict[str, PhysicalTensor]
    A dict mapping output names to tensors
)nbdoc");

    fused_equation.def(
        "_debug_stage_kinds",
        [](const FusedEquation& self,
           const std::unordered_map<std::string, Tensor>& inputs,
           const std::unordered_map<std::string, float>& scalar_inputs,
           const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs) {
            std::shared_ptr<ThorImplementation::CompiledOutputs> compiled =
                self.compileForInputs(inputs, scalar_inputs, tensor_scalar_inputs);
            std::vector<std::string> result;
            result.reserve(compiled->stages.size());
            for (const auto& stage : compiled->stages) {
                std::string label = ThorImplementation::CompiledExecutionStage::kindToString(stage.kind);
                if (stage.kind == ThorImplementation::CompiledExecutionStage::Kind::Matmul && stage.matmul) {
                    auto epilogue_name = [](ThorImplementation::MatmulEpilogue epilogue) {
                        switch (epilogue) {
                            case ThorImplementation::MatmulEpilogue::Default:
                                return "default";
                            case ThorImplementation::MatmulEpilogue::Relu:
                                return "relu";
                            case ThorImplementation::MatmulEpilogue::Gelu:
                                return "gelu";
                        }
                        return "unknown";
                    };
                    auto backward_epilogue_name = [](ThorImplementation::MatmulBackwardEpilogue epilogue) {
                        switch (epilogue) {
                            case ThorImplementation::MatmulBackwardEpilogue::Default:
                                return "default";
                            case ThorImplementation::MatmulBackwardEpilogue::DRelu:
                                return "drelu";
                            case ThorImplementation::MatmulBackwardEpilogue::DGelu:
                                return "dgelu";
                        }
                        return "unknown";
                    };
                    std::ostringstream oss;
                    oss << label << "(op=" << ThorImplementation::opName(stage.matmul->op)
                        << ",lhsT=" << (stage.matmul->transpose_lhs ? 1 : 0) << ",rhsT=" << (stage.matmul->transpose_rhs ? 1 : 0)
                        << ",auxT=" << (stage.matmul->transpose_aux ? 1 : 0);
                    if (stage.matmul->epilogue != ThorImplementation::MatmulEpilogue::Default) {
                        oss << ",epilogue=" << epilogue_name(stage.matmul->epilogue);
                    }
                    if (stage.matmul->backward_epilogue != ThorImplementation::MatmulBackwardEpilogue::Default) {
                        oss << ",backward_epilogue=" << backward_epilogue_name(stage.matmul->backward_epilogue);
                    }
                    if (stage.matmul->bgrad_output_dtype.has_value()) {
                        oss << ",bgrad=1";
                    }
                    oss << ")";
                    label = oss.str();
                }
                result.push_back(std::move(label));
            }
            return result;
        },
        "inputs"_a,
        nb::kw_only(),
        "scalar_inputs"_a = std::unordered_map<std::string, float>{},
        "tensor_scalar_inputs"_a = std::unordered_map<std::string, TensorScalarBinding>{});

    fused_equation.def(
        "_debug_fused_kernel_launches",
        [](const FusedEquation& self,
           const std::unordered_map<std::string, Tensor>& inputs,
           const std::unordered_map<std::string, float>& scalar_inputs,
           const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs) {
            auto launch_kind_name = [](ThorImplementation::CompiledEquation::LaunchKind kind) {
                switch (kind) {
                    case ThorImplementation::CompiledEquation::LaunchKind::Flat:
                        return "Flat";
                    case ThorImplementation::CompiledEquation::LaunchKind::BroadcastSingle:
                        return "BroadcastSingle";
                    case ThorImplementation::CompiledEquation::LaunchKind::BroadcastGrouped:
                        return "BroadcastGrouped";
                    case ThorImplementation::CompiledEquation::LaunchKind::FusedTiledTranspose:
                        return "FusedTiledTranspose";
                }
                return "<unknown>";
            };

            std::shared_ptr<ThorImplementation::PreparedConvenienceRunPlan> prepared_plan =
                self.prepareConvenienceRunPlanForInputs(inputs, scalar_inputs, tensor_scalar_inputs);
            std::shared_ptr<ThorImplementation::CompiledOutputs> compiled = prepared_plan->compiled_outputs;
            std::vector<std::string> result;
            for (size_t stage_idx = 0; stage_idx < compiled->stages.size(); ++stage_idx) {
                const auto& stage = compiled->stages[stage_idx];
                if (stage.kind != ThorImplementation::CompiledExecutionStage::Kind::FusedKernel) {
                    continue;
                }
                if (stage_idx >= prepared_plan->stages.size()) {
                    throw std::runtime_error("Prepared debug launch plan is missing a stage.");
                }
                const std::shared_ptr<ThorImplementation::CompiledEquation>& launch_equation =
                    prepared_plan->stages[stage_idx].compiled_equation;
                if (!launch_equation) {
                    continue;
                }

                std::ostringstream oss;
                oss << "FusedKernel(stage=" << stage_idx << ",launch=" << launch_kind_name(launch_equation->launch_kind)
                    << ",outputs=" << stage.outputs.size() << ",elements_per_thread=" << launch_equation->elements_per_thread
                    << ",tiled_pack=" << launch_equation->tiled_transpose_pack_scalars
                    << ",logical_transpose=" << (launch_equation->uses_tiled_logical_transpose_consumer ? 1 : 0)
                    << ",logical_slot_bytes=" << launch_equation->tiled_logical_transpose_slot_bytes
                    << ",logical_dense_packed_loads=" << launch_equation->tiled_logical_transpose_dense_packed_input_load_count
                    << ",logical_vectorized_outputs=" << launch_equation->tiled_logical_transpose_vectorized_output_count << ")";
                result.push_back(oss.str());
            }
            return result;
        },
        "inputs"_a,
        nb::kw_only(),
        "scalar_inputs"_a = std::unordered_map<std::string, float>{},
        "tensor_scalar_inputs"_a = std::unordered_map<std::string, TensorScalarBinding>{},
        R"nbdoc(
Return compact debug descriptions for compiled fused-kernel launch metadata.

This is intended for tests that need to assert a specific optimized lowering,
for example tiled logical-transpose auto-swizzle pack width or vectorized lane math,
instead of only asserting that a stage compiled to a generic FusedKernel.
)nbdoc");

    fused_equation.def(
        "compile_backward",
        [](const FusedEquation& self,
           const std::vector<std::string>& wrt_names,
           std::optional<std::string> error_input_name,
           bool accumulate_grad_outputs) {
            nb::gil_scoped_release release;
            return self.compileBackward(wrt_names, error_input_name, accumulate_grad_outputs);
        },
        "wrt_names"_a = std::vector<std::string>{},
        "error_input_name"_a.none() = nb::none(),
        "accumulate_grad_outputs"_a = false,
        R"nbdoc(
Compile a backward equation for a single-output forward equation.

The compiled backward equation expects an additional input tensor named by
error_input_name, whose shape is compatible with the forward output.

Args:
    wrt_names: list[str]
        Input names to differentiate with respect to. If omitted, all forward
        root inputs are differentiated and need to be supplied to the backward
        expression.
    error_input_name: str | None
        Name for the upstream-gradient input tensor.
        I.e. the incoming error gradient (for the backward computation) from the
        layer downstream in the forward direction.
)nbdoc");

    fused_equation.def(
        "compile_backward",
        [](const FusedEquation& self,
           const std::vector<std::string>& wrt_names,
           const std::unordered_map<std::string, std::string>& feature_output_name_to_error_input_name,
           bool accumulate_grad_outputs) {
            if (feature_output_name_to_error_input_name.size() == 0)
                throw std::runtime_error("Cannot compute backward expression with no error inputs.");
            nb::gil_scoped_release release;
            return self.compileBackward(wrt_names, feature_output_name_to_error_input_name, accumulate_grad_outputs);
        },
        "wrt_names"_a,
        "feature_output_name_to_error_input_name"_a,
        "accumulate_grad_outputs"_a = false,
        R"nbdoc(
Compile a backward equation for a multi-output forward equation.

This overload makes the upstream gradient explicit for each named forward
output. The compiled backward equation will expect one additional input tensor
per entry in ``feature_output_name_to_error_input_name``.

Args:
    wrt_names: list[str]
        Input names to differentiate with respect to.
    feature_output_name_to_error_input_name: dict[str, str]
        Mapping from forward output name to the input name that should carry the
        corresponding upstream gradient tensor.

For example:

bwd = fwd.compile_backward(
    ["x", "w"],
    {
        "main": "__grad_main",
        "aux": "__grad_aux",
    },
)

Then the backward equation will supply x_grad and w_grad as outputs.

In the case that w is frozen and you don't want the gradient with respect to w,
you would instead do:

bwd = fwd.compile_backward(
    ["x"],
    {
        "main": "__grad_main",
        "aux": "__grad_aux",
    },
)

Then the backward equation will only supply x_grad as an ouput. When either
__grad_main or __grad_aux does not participate in the gradient computation
for x, the unused tensor will not be accessed - it will be ignored in that case.
)nbdoc");

    fused_equation.def("output_names",
                       &FusedEquation::getOutputNames,
                       R"nbdoc(
Returns
-------
list[int]
    A list of names of the outputs from this equation.
)nbdoc");

    fused_equation.def("output_shape",
                       nb::overload_cast<const Tensor&>(&FusedEquation::getOutputShape, nb::const_),
                       "input"_a,
                       R"nbdoc(
Get the shape of the output tensor for this equation, from the input tensors.

Parameters
----------
inputs: dict[str, PhysicalTensor]
    A dict mapping input names to tensors

Returns
-------
list[int]
    The output tensor dimensions.
)nbdoc");
    fused_equation.def("output_shape",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&>(&FusedEquation::getOutputShape, nb::const_),
                       "inputs"_a,
                       R"nbdoc(
Get the shape of the output tensor for this equation, from the input tensors.

Parameters
----------
inputs: dict[str, PhysicalTensor]
    A dict mapping input names to tensors

Returns
-------
list[int]
    The output tensor dimensions.
)nbdoc");

    fused_equation.def("output_shapes",
                       nb::overload_cast<const Tensor&>(&FusedEquation::getOutputShapes, nb::const_),
                       "input"_a,
                       R"nbdoc(
Get the shape of the output tensor for this equation, from the input tensors.

Parameters
----------
inputs: dict[str, PhysicalTensor]
    A dict mapping input names to tensors

Returns
-------
dict[str, list[int]]
    output name -> tensor dimensions.
)nbdoc");
    fused_equation.def("output_shapes",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&>(&FusedEquation::getOutputShapes, nb::const_),
                       "inputs"_a,
                       R"nbdoc(
Get the shape of the output tensor for this equation, from the input tensors.

Parameters
----------
inputs: dict[str, PhysicalTensor]
    A dict mapping input names to tensors

Returns
-------
dict[str, list[int]]
    output name -> tensor dimensions.
)nbdoc");
}

void bind_stamped_equation(nb::module_& physical) {
    auto stamped_equation = nb::class_<StampedExecutionPlan>(physical, "Equation");
    stamped_equation.attr("__module__") = "thor.physical";

    stamped_equation.def("run",
                         nb::overload_cast<>(&StampedExecutionPlan::run),
                         R"nbdoc(
Execute the stamped fused equation on the bound tensors.
        )nbdoc");
    stamped_equation.def("run",
                         nb::overload_cast<const std::unordered_map<std::string, float>&>(&StampedExecutionPlan::run),
                         "runtime_scalars"_a,
                         R"nbdoc(
Execute the stamped fused equation on the bound tensors, overriding any bound runtime scalar values for this run.
        )nbdoc");

    stamped_equation.def(
        "output",
        [](const StampedExecutionPlan& self) { return self.output(); },
        R"nbdoc(
Return the output tensor owned by this equation instance. Valid when the equation has a single output tensor.
)nbdoc");

    stamped_equation.def(
        "output",
        [](const StampedExecutionPlan& self, const std::string& name) { return self.output(name); },
        "name"_a,
        R"nbdoc(
Return a named output tensor from a stamped multi-output execution plan.
)nbdoc");

    stamped_equation.def(
        "outputs",
        [](const StampedExecutionPlan& self) { return self.getFinalOutputs(); },
        R"nbdoc(
Return a dict of named output tensor from a stamped multi-output execution plan.
)nbdoc");

    stamped_equation.def("output_names", [](const StampedExecutionPlan& self) { return self.outputNames(); });

    stamped_equation.def("flop_count",
                         &StampedExecutionPlan::flopCount,
                         R"nbdoc(
Return the semantic floating-point operation count represented by this stamped execution plan.

Conventions:
- elementwise arithmetic/transcendentals count as 1 op per output element
- GEMM / matmul / convolution use 2 FLOPs per multiply-accumulate
- shape-only ops and transpose count as 0
- this is a semantic model FLOP count, not a backend-instruction count
)nbdoc");

    stamped_equation.def("stage_flop_counts",
                         &StampedExecutionPlan::stageFlopCounts,
                         R"nbdoc(
Return the per-stage semantic FLOP counts for this stamped execution plan.
)nbdoc");

    stamped_equation.def("_debug_stage_kinds",
                         &StampedExecutionPlan::stageKindNames,
                         R"nbdoc(
Return the concrete runtime stage kinds in this stamped execution plan.

This reflects stages that will actually run, after runtime-stage elision such as
direct packed-QKV attention-backward outputs.
)nbdoc");
}

void bind_dynamic_expression(nb::module_& physical) {
    auto dynamic_expression_build = nb::class_<DynamicExpressionBuild>(physical, "DynamicExpressionBuild");
    dynamic_expression_build.attr("__module__") = "thor.physical";

    dynamic_expression_build.def(
        "__init__",
        [](DynamicExpressionBuild* self,
           const std::shared_ptr<FusedEquation>& equation,
           const DynamicTensorMap& stamp_inputs,
           const DynamicTensorScalarMap& tensor_scalar_inputs,
           const DynamicTensorMap& preallocated_outputs,
           const DynamicShapeMap& requested_output_shapes) {
            new (self) DynamicExpressionBuild{
                equation,
                stamp_inputs,
                tensor_scalar_inputs,
                preallocated_outputs,
                requested_output_shapes,
            };
        },
        "equation"_a,
        "stamp_inputs"_a,
        "tensor_scalar_inputs"_a = DynamicTensorScalarMap{},
        "preallocated_outputs"_a = DynamicTensorMap{},
        "requested_output_shapes"_a = DynamicShapeMap{},
        R"nbdoc(
Describe a prepared dynamic-expression build result.

Parameters
----------
equation : thor.physical.FusedEquation
    The compiled equation to stamp.
stamp_inputs : dict[str, PhysicalTensor]
    Input tensors bound into the prepared expression.
tensor_scalar_inputs : dict[str, thor.physical.TensorScalarBinding], optional
    Tensor-backed scalar bindings.
preallocated_outputs : dict[str, PhysicalTensor], optional
    Output tensors to bind when stamping.
requested_output_shapes : dict[str, list[int]], optional
    Per-output requested shapes used when stamping.
)nbdoc");

    dynamic_expression_build.def_rw("equation", &DynamicExpressionBuild::equation);
    dynamic_expression_build.def_rw("stamp_inputs", &DynamicExpressionBuild::stamp_inputs);
    dynamic_expression_build.def_rw("tensor_scalar_inputs", &DynamicExpressionBuild::tensor_scalar_inputs);
    dynamic_expression_build.def_rw("preallocated_outputs", &DynamicExpressionBuild::preallocated_outputs);
    dynamic_expression_build.def_rw("requested_output_shapes", &DynamicExpressionBuild::requested_output_shapes);

    auto prepared_dynamic_expression = nb::class_<PreparedDynamicExpression>(physical, "PreparedDynamicExpression");
    prepared_dynamic_expression.attr("__module__") = "thor.physical";

    prepared_dynamic_expression.def("stamp",
                                    nb::overload_cast<>(&PreparedDynamicExpression::stamp, nb::const_),
                                    R"nbdoc(
Stamp the prepared dynamic expression using its bound inputs and any default
preallocated outputs captured in the DynamicExpressionBuild.
)nbdoc");

    prepared_dynamic_expression.def(
        "stamp",
        nb::overload_cast<const DynamicTensorMap&, const DynamicShapeMap&>(&PreparedDynamicExpression::stamp, nb::const_),
        "preallocated_outputs_override"_a,
        "requested_output_shapes_override"_a = DynamicShapeMap{},
        R"nbdoc(
Stamp the prepared dynamic expression, overriding default preallocated outputs
and/or requested output shapes for this stamp.
)nbdoc");

    prepared_dynamic_expression.def(
        "compile_backward",
        [](const PreparedDynamicExpression& self,
           const std::vector<std::string>& wrt_names,
           std::optional<std::string> upstream_input_name,
           bool accumulate_grad_outputs,
           const DynamicTensorMap& additional_inputs,
           const DynamicTensorScalarMap& additional_tensor_scalar_inputs,
           const DynamicTensorMap& preallocated_grad_outputs,
           const DynamicShapeMap& requested_grad_output_shapes) {
            return self.compileBackward(wrt_names,
                                        upstream_input_name,
                                        accumulate_grad_outputs,
                                        additional_inputs,
                                        additional_tensor_scalar_inputs,
                                        preallocated_grad_outputs,
                                        requested_grad_output_shapes);
        },
        "wrt_names"_a = std::vector<std::string>{},
        "upstream_input_name"_a.none() = nb::none(),
        "accumulate_grad_outputs"_a = false,
        "additional_inputs"_a = DynamicTensorMap{},
        "additional_tensor_scalar_inputs"_a = DynamicTensorScalarMap{},
        "preallocated_grad_outputs"_a = DynamicTensorMap{},
        "requested_grad_output_shapes"_a = DynamicShapeMap{},
        R"nbdoc(
Prepare a backward dynamic expression for a single-output forward expression.
)nbdoc");

    prepared_dynamic_expression.def(
        "compile_backward",
        [](const PreparedDynamicExpression& self,
           const std::vector<std::string>& wrt_names,
           const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
           bool accumulate_grad_outputs,
           const DynamicTensorMap& additional_inputs,
           const DynamicTensorScalarMap& additional_tensor_scalar_inputs,
           const DynamicTensorMap& preallocated_grad_outputs,
           const DynamicShapeMap& requested_grad_output_shapes) {
            return self.compileBackward(wrt_names,
                                        upstream_input_names_by_output,
                                        accumulate_grad_outputs,
                                        additional_inputs,
                                        additional_tensor_scalar_inputs,
                                        preallocated_grad_outputs,
                                        requested_grad_output_shapes);
        },
        "wrt_names"_a,
        "upstream_input_names_by_output"_a,
        "accumulate_grad_outputs"_a = false,
        "additional_inputs"_a = DynamicTensorMap{},
        "additional_tensor_scalar_inputs"_a = DynamicTensorScalarMap{},
        "preallocated_grad_outputs"_a = DynamicTensorMap{},
        "requested_grad_output_shapes"_a = DynamicShapeMap{},
        R"nbdoc(
Prepare a backward dynamic expression for a multi-output forward expression.
)nbdoc");

    prepared_dynamic_expression.def(
        "get_parameter_fan_overrides",
        [](const PreparedDynamicExpression& self, const std::vector<std::string>& parameter_names) {
            const std::unordered_set<std::string> parameter_name_set(parameter_names.begin(), parameter_names.end());
            return parameterFanOverridesToPython(self.getParameterFanOverrides(parameter_name_set));
        },
        "parameter_names"_a,
        R"nbdoc(
Infer parameter initializer fan-in/fan-out overrides for the prepared dynamic expression.
)nbdoc");

    prepared_dynamic_expression.def(
        "stamp_backward",
        [](const PreparedDynamicExpression& self,
           const std::vector<std::string>& wrt_names,
           std::optional<std::string> upstream_input_name,
           bool accumulate_grad_outputs,
           const DynamicTensorMap& additional_inputs,
           const DynamicTensorScalarMap& additional_tensor_scalar_inputs,
           const DynamicTensorMap& preallocated_grad_outputs,
           const DynamicShapeMap& requested_grad_output_shapes) {
            return self.stampBackward(wrt_names,
                                      upstream_input_name,
                                      accumulate_grad_outputs,
                                      additional_inputs,
                                      additional_tensor_scalar_inputs,
                                      preallocated_grad_outputs,
                                      requested_grad_output_shapes);
        },
        "wrt_names"_a = std::vector<std::string>{},
        "upstream_input_name"_a.none() = nb::none(),
        "accumulate_grad_outputs"_a = false,
        "additional_inputs"_a = DynamicTensorMap{},
        "additional_tensor_scalar_inputs"_a = DynamicTensorScalarMap{},
        "preallocated_grad_outputs"_a = DynamicTensorMap{},
        "requested_grad_output_shapes"_a = DynamicShapeMap{},
        R"nbdoc(
Stamp a backward execution plan for a single-output forward expression.
)nbdoc");

    prepared_dynamic_expression.def(
        "stamp_backward",
        [](const PreparedDynamicExpression& self,
           const std::vector<std::string>& wrt_names,
           const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
           bool accumulate_grad_outputs,
           const DynamicTensorMap& additional_inputs,
           const DynamicTensorScalarMap& additional_tensor_scalar_inputs,
           const DynamicTensorMap& preallocated_grad_outputs,
           const DynamicShapeMap& requested_grad_output_shapes) {
            return self.stampBackward(wrt_names,
                                      upstream_input_names_by_output,
                                      accumulate_grad_outputs,
                                      additional_inputs,
                                      additional_tensor_scalar_inputs,
                                      preallocated_grad_outputs,
                                      requested_grad_output_shapes);
        },
        "wrt_names"_a,
        "upstream_input_names_by_output"_a,
        "accumulate_grad_outputs"_a = false,
        "additional_inputs"_a = DynamicTensorMap{},
        "additional_tensor_scalar_inputs"_a = DynamicTensorScalarMap{},
        "preallocated_grad_outputs"_a = DynamicTensorMap{},
        "requested_grad_output_shapes"_a = DynamicShapeMap{},
        R"nbdoc(
Stamp a backward execution plan for a multi-output forward expression.
)nbdoc");

    prepared_dynamic_expression.def_prop_ro(
        "equation",
        [](PreparedDynamicExpression& self) -> FusedEquation& { return const_cast<FusedEquation&>(self.equation()); },
        nb::rv_policy::reference_internal,
        R"nbdoc(
Return the compiled equation owned by this prepared dynamic expression.
)nbdoc");

    prepared_dynamic_expression.def_prop_ro("stamp_inputs", &PreparedDynamicExpression::stampInputs, nb::rv_policy::reference_internal);

    prepared_dynamic_expression.def_prop_ro(
        "tensor_scalar_inputs", &PreparedDynamicExpression::tensorScalarInputs, nb::rv_policy::reference_internal);

    prepared_dynamic_expression.def_prop_ro(
        "preallocated_outputs", &PreparedDynamicExpression::preallocatedOutputs, nb::rv_policy::reference_internal);

    prepared_dynamic_expression.def_prop_ro(
        "requested_output_shapes", &PreparedDynamicExpression::requestedOutputShapes, nb::rv_policy::reference_internal);

    auto dynamic_expression = nb::class_<DynamicExpression>(physical, "DynamicExpression");
    dynamic_expression.attr("__module__") = "thor.physical";

    dynamic_expression.def_static(
        "from_expression_definition",
        [](const ExpressionDefinition& definition, bool use_fast_math) {
            return DynamicExpression::fromExpressionDefinition(definition, use_fast_math);
        },
        "definition"_a,
        "use_fast_math"_a = false);

    dynamic_expression.def(
        "__init__",
        [](DynamicExpression* self, nb::callable builder) {
            auto builderRef = std::make_shared<GilSafePythonObject>(builder);

            new (self) DynamicExpression(
                [builderRef](const DynamicTensorMap& inputs, const DynamicTensorMap& outputs, Stream& stream) -> DynamicExpressionBuild {
                    nb::gil_scoped_acquire gil;

                    nb::callable builderCallable = nb::borrow<nb::callable>(builderRef->get());

                    nb::object result = builderCallable(inputs, outputs, stream);
                    return nb::cast<DynamicExpressionBuild>(result);
                });
        },
        "builder"_a,
        R"nbdoc(
Create a dynamic expression from a Python callable.

Parameters
----------
builder : Callable[
    [dict[str, PhysicalTensor], dict[str, PhysicalTensor], thor.Stream],
    thor.physical.DynamicExpressionBuild
]
    A callable that receives three arguments:

    - ``inputs``:
      A mapping from input name to the currently bound input tensor.

      These are the tensors that the caller supplied when preparing or
      stamping the dynamic expression. The builder may inspect their shape,
      dtype, placement, and other metadata in order to choose how to build
      the expression.

    - ``outputs``:
      A mapping from output name to caller-supplied output tensors.

      This mapping represents the final outputs that the caller would like
      the dynamic expression to write into.

      The builder may use ``outputs`` to:

      - validate that a provided output tensor has the expected shape, dtype,
        or placement
      - choose an output dtype or architecture that is compatible with the
        requested outputs
      - pass those tensors through as ``preallocated_outputs`` in the returned
        ``DynamicExpressionBuild`` so the compiled equation writes into them

      In other words, ``outputs`` is part of the builder's decision surface.
      It is not just informational metadata.

    - ``stream``:
      The stream that will be used for preparation / stamping / execution.

      The builder may inspect this if needed, but it should generally use the
      stream only as contextual information when constructing the returned
      ``DynamicExpressionBuild``.

Returns
-------
thor.physical.DynamicExpressionBuild
    An object describing the compiled equation and any default bindings to use
    when stamping. Typically this includes:

    - a compiled equation
    - the chosen input bindings
    - any caller-provided output tensors that should be reused as final outputs

Notes
-----
The callable is invoked synchronously from C++ when ``prepare(...)``,
``stamp(...)``, or ``stamp_backward(...)`` is called.

This means the builder runs on the preparation / stamping path, not on the
hot kernel execution path. It is therefore appropriate for the builder to make
shape-dependent, dtype-dependent, or architecture-dependent decisions.

The builder is expected to *describe* the computation by returning a
``DynamicExpressionBuild``. It should not directly enqueue the actual forward
or backward kernels itself.

Examples
--------
A simple example that chooses the expression based on the shape of the input:

.. code-block:: python

    import thor
    from thor.physical import Expression as ex
    from thor.physical import DynamicExpression, DynamicExpressionBuild, FusedEquation

    def make_dynamic_relu(device_num: int):
        def builder(inputs, outputs, stream):
            x_tensor = inputs["x"]

            if len(x_tensor.dims) != 2:
                raise ValueError("expected a rank-2 input")

            x = ex.input("x")
            y = ex.max(x, 0.0)

            named = ex.outputs({"y": y})
            equation = FusedEquation.compile(named.physical_outputs(), device_num)

            return DynamicExpressionBuild(
                equation=equation,
                stamp_inputs=inputs,
                preallocated_outputs=outputs,
            )

        return DynamicExpression(builder)

Using ``outputs`` to validate and reuse a caller-provided output tensor:

.. code-block:: python

    import thor
    from thor.physical import Expression as ex
    from thor.physical import DynamicExpression, DynamicExpressionBuild, FusedEquation

    def make_dynamic_identity(device_num: int):
        def builder(inputs, outputs, stream):
            x_tensor = inputs["x"]

            if "y" in outputs:
                y_tensor = outputs["y"]
                if y_tensor.dims != x_tensor.dims:
                    raise ValueError(
                        f"output y has dims {y_tensor.dims}, expected {x_tensor.dims}"
                    )
                if y_tensor.placement != x_tensor.placement:
                    raise ValueError("output y must be on the same device as x")

            x = ex.input("x")
            named = ex.outputs({"y": x})
            equation = FusedEquation.compile(named.physical_outputs(), device_num)

            return DynamicExpressionBuild(
                equation=equation,
                stamp_inputs=inputs,
                preallocated_outputs=outputs,
            )

        return DynamicExpression(builder)

A fully connected layer can be expressed by inspecting the bound parameter
tensors at build time:

.. code-block:: python

    import thor
    from thor.physical import Expression as ex
    from thor.physical import DynamicExpression, DynamicExpressionBuild, FusedEquation

    def fully_connected_dynamic_expression(device_num: int, has_bias: bool = True):
        def builder(inputs, outputs, stream):
            x_tensor = inputs["feature_input"]
            w_tensor = inputs["weights"]

            if len(x_tensor.dims) != 2:
                raise ValueError("feature_input must be rank 2")
            if len(w_tensor.dims) != 2:
                raise ValueError("weights must be rank 2")
            if x_tensor.dims[1] != w_tensor.dims[0]:
                raise ValueError("feature_input and weights have incompatible shapes")

            x = ex.input("feature_input")
            w = ex.input("weights", output_dtype=w_tensor.dtype, compute_dtype=w_tensor.dtype)
            y = x @ w

            if has_bias:
                b_tensor = inputs["biases"]
                if len(b_tensor.dims) != 1 or b_tensor.dims[0] != w_tensor.dims[1]:
                    raise ValueError("biases must have shape [out_features]")
                b = ex.input("biases", output_dtype=b_tensor.dtype, compute_dtype=b_tensor.dtype)
                y = y + b

            named = ex.outputs({"feature_output": y})
            equation = FusedEquation.compile(named.physical_outputs(), device_num)

            return DynamicExpressionBuild(
                equation=equation,
                stamp_inputs=inputs,
                preallocated_outputs=outputs,
            )

        return DynamicExpression(builder)

)nbdoc");

    dynamic_expression.def_prop_ro("serialized_definition", &DynamicExpression::getSerializedDefinition);

    dynamic_expression.def("prepare",
                           &DynamicExpression::prepare,
                           "inputs"_a,
                           "outputs"_a,
                           "stream"_a,
                           R"nbdoc(
Validate the provided tensors and stream, invoke the Python builder with
``(inputs, outputs, stream)``, and return a PreparedDynamicExpression.
)nbdoc");

    dynamic_expression.def(
        "stamp",
        nb::overload_cast<const DynamicTensorMap&, const DynamicTensorMap&, Stream&>(&DynamicExpression::stamp, nb::const_),
        "inputs"_a,
        "outputs"_a,
        "stream"_a,
        R"nbdoc(
Validate the provided tensors and stream, then stamp the dynamic expression.
)nbdoc");

    dynamic_expression.def(
        "stamp",
        nb::overload_cast<const DynamicTensorMap&, const DynamicTensorMap&, Stream&, const DynamicTensorMap&, const DynamicShapeMap&>(
            &DynamicExpression::stamp, nb::const_),
        "inputs"_a,
        "outputs"_a,
        "stream"_a,
        "preallocated_outputs"_a,
        "requested_output_shapes"_a = DynamicShapeMap{},
        R"nbdoc(
Validate the provided tensors and stream, then stamp the dynamic expression
with output overrides.
)nbdoc");

    dynamic_expression.def(
        "stamp_backward",
        [](const DynamicExpression& self,
           const DynamicTensorMap& inputs,
           const DynamicTensorMap& outputs,
           Stream& stream,
           const std::vector<std::string>& wrt_names,
           std::optional<std::string> upstream_input_name,
           bool accumulate_grad_outputs,
           const DynamicTensorMap& additional_inputs,
           const DynamicTensorScalarMap& additional_tensor_scalar_inputs,
           const DynamicTensorMap& preallocated_grad_outputs,
           const DynamicShapeMap& requested_grad_output_shapes) {
            return self.stampBackward(inputs,
                                      outputs,
                                      stream,
                                      wrt_names,
                                      upstream_input_name,
                                      accumulate_grad_outputs,
                                      additional_inputs,
                                      additional_tensor_scalar_inputs,
                                      preallocated_grad_outputs,
                                      requested_grad_output_shapes);
        },
        "inputs"_a,
        "outputs"_a,
        "stream"_a,
        "wrt_names"_a = std::vector<std::string>{},
        "upstream_input_name"_a.none() = nb::none(),
        "accumulate_grad_outputs"_a = false,
        "additional_inputs"_a = DynamicTensorMap{},
        "additional_tensor_scalar_inputs"_a = DynamicTensorScalarMap{},
        "preallocated_grad_outputs"_a = DynamicTensorMap{},
        "requested_grad_output_shapes"_a = DynamicShapeMap{},
        R"nbdoc(
Prepare and stamp a backward execution plan for a single-output forward expression.
)nbdoc");

    dynamic_expression.def(
        "stamp_backward",
        [](const DynamicExpression& self,
           const DynamicTensorMap& inputs,
           const DynamicTensorMap& outputs,
           Stream& stream,
           const std::vector<std::string>& wrt_names,
           const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
           bool accumulate_grad_outputs,
           const DynamicTensorMap& additional_inputs,
           const DynamicTensorScalarMap& additional_tensor_scalar_inputs,
           const DynamicTensorMap& preallocated_grad_outputs,
           const DynamicShapeMap& requested_grad_output_shapes) {
            return self.stampBackward(inputs,
                                      outputs,
                                      stream,
                                      wrt_names,
                                      upstream_input_names_by_output,
                                      accumulate_grad_outputs,
                                      additional_inputs,
                                      additional_tensor_scalar_inputs,
                                      preallocated_grad_outputs,
                                      requested_grad_output_shapes);
        },
        "inputs"_a,
        "outputs"_a,
        "stream"_a,
        "wrt_names"_a,
        "upstream_input_names_by_output"_a,
        "accumulate_grad_outputs"_a = false,
        "additional_inputs"_a = DynamicTensorMap{},
        "additional_tensor_scalar_inputs"_a = DynamicTensorScalarMap{},
        "preallocated_grad_outputs"_a = DynamicTensorMap{},
        "requested_grad_output_shapes"_a = DynamicShapeMap{},
        R"nbdoc(
Prepare and stamp a backward execution plan for a multi-output forward expression.
)nbdoc");
}
