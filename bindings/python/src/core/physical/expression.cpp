#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "Utilities/TensorMathFusion/FusedEquation.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

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
        "constant_scalar",
        [](double value) { return Expression::constant_scalar(value); },
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

    fused_equation.def("stamp",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&, const Stream&, const std::vector<uint64_t>&>(
                           &FusedEquation::stamp, nb::const_),
                       "inputs"_a,
                       "stream"_a,
                       "requestedOutputShape"_a = std::vector<uint64_t>{},
                       R"nbdoc(
Create an executable instance of this fused equation with bound thor.physical.PhysicalTensor's.

inputs: dict[str, PhysicalTensor]
    A dict mapping input names to tensors
)nbdoc");

    fused_equation.def("stamp",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&,
                                         const std::unordered_map<std::string, float>&,
                                         const Stream&,
                                         const std::vector<uint64_t>&>(&FusedEquation::stamp, nb::const_),
                       "inputs"_a,
                       "scalar_inputs"_a,
                       "stream"_a,
                       "requestedOutputShape"_a = std::vector<uint64_t>{},
                       R"nbdoc(
Create an executable instance of this fused equation with bound tensor and runtime scalar inputs.
)nbdoc");

    fused_equation.def("stamp",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&,
                                         const std::unordered_map<std::string, float>&,
                                         const Stream&,
                                         const std::unordered_map<std::string, std::vector<uint64_t>>&>(&FusedEquation::stamp, nb::const_),
                       "inputs"_a,
                       "scalar_inputs"_a,
                       "stream"_a,
                       "requestedOutputShapes"_a = std::unordered_map<std::string, std::vector<uint64_t>>{},
                       R"nbdoc(
Create an executable instance of this fused equation with bound tensor and runtime scalar inputs.
)nbdoc");

    fused_equation.def("stamp",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&,
                                         const Stream&,
                                         const std::unordered_map<std::string, std::vector<uint64_t>>&>(&FusedEquation::stamp, nb::const_),
                       "inputs"_a,
                       "stream"_a,
                       "requestedOutputShapes"_a = std::unordered_map<std::string, std::vector<uint64_t>>{},
                       R"nbdoc(
Create an executable instance of this fused equation with bound thor.physical.PhysicalTensor's.

inputs: dict[str, PhysicalTensor]
    A dict mapping input names to tensors
)nbdoc");

    fused_equation.def("stamp",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&,
                                         const std::unordered_map<std::string, float>&,
                                         const std::unordered_map<std::string, Tensor>&,
                                         const Stream&,
                                         const std::vector<uint64_t>&>(&FusedEquation::stamp, nb::const_),
                       "inputs"_a,
                       "scalar_inputs"_a,
                       "preallocated_outputs"_a,
                       "stream"_a,
                       "requestedOutputShape"_a = std::vector<uint64_t>{},
                       R"nbdoc(
Create an executable instance of this fused equation with bound tensor and runtime scalar inputs.
)nbdoc");

    fused_equation.def("stamp",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&,
                                         const std::unordered_map<std::string, float>&,
                                         const std::unordered_map<std::string, Tensor>&,
                                         const Stream&,
                                         const std::unordered_map<std::string, std::vector<uint64_t>>&>(&FusedEquation::stamp, nb::const_),
                       "inputs"_a,
                       "scalar_inputs"_a,
                       "preallocated_outputs"_a,
                       "stream"_a,
                       "requestedOutputShapes"_a = std::unordered_map<std::string, std::vector<uint64_t>>{},
                       R"nbdoc(
Create an executable instance of this fused equation with bound tensor and runtime scalar inputs.
)nbdoc");

    fused_equation.def("stamp",
                       nb::overload_cast<const std::unordered_map<std::string, Tensor>&,
                                         const std::unordered_map<std::string, Tensor>&,
                                         const Stream&,
                                         const std::unordered_map<std::string, std::vector<uint64_t>>&>(&FusedEquation::stamp, nb::const_),
                       "inputs"_a,
                       "preallocated_outputs"_a,
                       "stream"_a,
                       "requestedOutputShapes"_a = std::unordered_map<std::string, std::vector<uint64_t>>{},
                       R"nbdoc(
Create an executable instance of this fused equation with bound thor.physical.PhysicalTensor's.

inputs: dict[str, PhysicalTensor]
    A dict mapping input names to tensors
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
                         &StampedExecutionPlan::run,
                         R"nbdoc(
Execute the stamped fused equation on the bound tensors.
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
}
