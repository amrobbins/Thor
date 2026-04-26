#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

#include <sstream>

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
using NamedOutput = ThorImplementation::NamedOutput;
using DynamicExpression = ThorImplementation::DynamicExpression;
using TensorScalarBinding = ThorImplementation::TensorScalarBinding;
using PreparedDynamicExpression = ThorImplementation::PreparedDynamicExpression;
using DynamicExpressionBuild = ThorImplementation::DynamicExpressionBuild;
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
    auto expr = nb::class_<Expression>(physical, "Expression");
    expr.attr("__module__") = "thor.physical";

    expr.def(nb::init_implicit<double>());
    // expr.def(nb::init_implicit<int64_t>());

    expr.def_static(
        "input",
        [](const std::string& name, nb::object output_dtype_obj, nb::object compute_dtype_obj) {
            Optional<DataType> output_dtype = Optional<DataType>::empty();
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            Optional<DataType> compute_dtype = Optional<DataType>::empty();
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
            Optional<DataType> output_dtype = Optional<DataType>::empty();
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            Optional<DataType> compute_dtype = Optional<DataType>::empty();
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
            Optional<DataType> output_dtype = Optional<DataType>::empty();
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            Optional<DataType> compute_dtype = Optional<DataType>::empty();
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
            Optional<DataType> output_dtype = Optional<DataType>::empty();
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            Optional<DataType> compute_dtype = Optional<DataType>::empty();
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
Return the rank-2 transpose of this expression.

Transpose is lowered as its own staged operation.
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
            Optional<DataType> output_dtype = Optional<DataType>::empty();
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            Optional<DataType> compute_dtype = Optional<DataType>::empty();
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
        "matmul",
        [](const Expression& a,
           const Expression& b,
           bool transpose_a,
           bool transpose_b,
           nb::object output_dtype_obj,
           nb::object compute_dtype_obj) {
            Optional<DataType> output_dtype = Optional<DataType>::empty();
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            Optional<DataType> compute_dtype = Optional<DataType>::empty();
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
            Optional<DataType> output_dtype = Optional<DataType>::empty();
            if (!output_dtype_obj.is_none()) {
                output_dtype = nb::cast<DataType>(output_dtype_obj);
            }
            Optional<DataType> compute_dtype = Optional<DataType>::empty();
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
    expr.def_static("log2", [](const Expression& x) { return x.log2(); }, "x"_a);
    expr.def_static("log10", [](const Expression& x) { return x.log10(); }, "x"_a);

    expr.def_static(
        "sqrt",
        [](const Expression& x) { return x.sqrt(); },
        "x"_a,
        R"nbdoc(
Return the elementwise square root of the input expression x
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
                                            const std::optional<DataType>& compute_dtype) -> Optional<DataType> {
        // if (compute_dtype.has_value() && compute_dtype.value() != DataType::FP32) {
        //     throw std::runtime_error(std::string(op_name) + ": currently only supports compute_dtype=thor.DataType.fp32");
        // }
        return DataType::FP32;
    };

    auto parse_reduction_output_dtype = [](std::string_view op_name, std::optional<DataType> compute_dtype) -> Optional<DataType> {
        // if (compute_dtype.has_value() && compute_dtype.value() != DataType::FP32) {
        //     throw std::runtime_error(std::string(op_name) + ": currently only supports output_dtype=thor.DataType.fp32");
        // }
        return DataType::FP32;
    };

    auto parse_arg_reduction_output_dtype = [](std::string_view op_name, std::optional<DataType> output_dtype) -> Optional<DataType> {
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

    nb::class_<Outputs>(expr, "Outputs")
        .def(
            "compile",
            [](const Outputs& self, int device_num, bool use_fast_math) {
                nb::gil_scoped_release release;
                return FusedEquation::compile(self.physicalOutputs(), device_num, use_fast_math);
            },
            "device_num"_a = 0,
            "use_fast_math"_a = false)
        .def("output_names", [](const Outputs& self) {
            std::vector<std::string> names;
            for (const NamedOutput& output : self.namedOutputs()) {
                names.push_back(output.name);
            }
            return names;
        });

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
                    std::ostringstream oss;
                    oss << label << "(lhsT=" << (stage.matmul->transpose_lhs ? 1 : 0) << ",rhsT=" << (stage.matmul->transpose_rhs ? 1 : 0)
                        << ",auxT=" << (stage.matmul->transpose_aux ? 1 : 0) << ")";
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
