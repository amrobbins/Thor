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

    expr.def_static("input",
                    &Expression::input,
                    "input_name"_a,
                    R"nbdoc(
            Create an expression input node.

            Parameters
            ----------
            input_name : str
                The zero-based input number.

            Returns
            -------
            thor.physical.Expression
                Expression representing that input.
        )nbdoc");

    expr.def_static(
        "scalar",
        [](double value) { return Expression::scalar(value); },
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
        if (compute_dtype.has_value() && compute_dtype.value() != DataType::FP32) {
            throw std::runtime_error(std::string(op_name) + ": currently only supports compute_dtype=thor.DataType.fp32");
        }
        return DataType::FP32;
    };

    auto parse_reduction_output_dtype = [](std::string_view op_name, std::optional<DataType> compute_dtype) -> Optional<DataType> {
        if (compute_dtype.has_value() && compute_dtype.value() != DataType::FP32) {
            throw std::runtime_error(std::string(op_name) + ": currently only supports output_dtype=thor.DataType.fp32");
        }
        return DataType::FP32;
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
            [](const Outputs& self, DataType dtype, int device_num, bool use_fast_math) {
                return FusedEquation::compile(self.physicalOutputs(), dtype, device_num, use_fast_math);
            },
            "dtype"_a,
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
        [](const Expression& expr, DataType dtype, int device_num, bool use_fast_math) {
            nb::gil_scoped_release release;
            return FusedEquation::compile(expr.expression(), dtype, device_num, use_fast_math);
        },
        "expr"_a,
        "dtype"_a,
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

Example
-------
x = thor.physical.Expression.input(0)
y = thor.physical.Expression.input(1)
z = thor.physical.Expression.input(2)

expr = (x / y) * z + 1.5
eq = thor.physical.Expression.compile(
    expr,
    dtype=thor.DataType.fp32,
    device_num=0,
)

eq.run({'x': x_tensor, 'y': y_tensor, 'z': z_tensor}, out_tensor, stream)

# or

stamped = eq.stamp({'x': x_tensor, 'y': y_tensor, 'z': z_tensor}, stream)
stamped.run()
out = stamped.output_tensor
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
}

void bind_stamped_equation(nb::module_& physical) {
    auto stamped_equation = nb::class_<StampedExecutionPlan>(physical, "Equation");
    stamped_equation.attr("__module__") = "thor.physical";

    stamped_equation.def("run",
                         &StampedExecutionPlan::run,
                         R"nbdoc(
Execute the stamped fused equation on the bound tensors.
        )nbdoc");

    stamped_equation.def_prop_ro(
        "output_tensor",
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
