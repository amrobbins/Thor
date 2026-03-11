#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
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

void bind_physical_expression(nb::module_& physical) {
    auto expr = nb::class_<Expression>(physical, "Expression");
    expr.attr("__module__") = "thor.physical";

    expr.def(nb::init_implicit<double>());
    // expr.def(nb::init_implicit<int64_t>());

    expr.def_static("input",
                    &Expression::input,
                    "input_index"_a,
                    R"nbdoc(
            Create an expression input node.

            Parameters
            ----------
            input_index : int
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

    //     expr.def_static(
    //         "scalar",
    //         [](int64_t value) { return Expression::scalar(value); },
    //         "value"_a,
    //         R"nbdoc(
    // Create an integer scalar constant expression.
    // )nbdoc");

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

eq.run([x_tensor, y_tensor, z_tensor], out_tensor, stream)

# or

stamped = eq.stamp([x_tensor, y_tensor, z_tensor], stream)
stamped.run()
out = stamped.output_tensor
)nbdoc");
}

void bind_fused_equation(nb::module_& physical) {
    auto fused_equation = nb::class_<FusedEquation>(physical, "FusedEquation");
    fused_equation.attr("__module__") = "thor.physical";

    fused_equation.def("stamp",
                       &FusedEquation::stamp,
                       "inputs"_a,
                       "stream"_a,
                       "requestedOutputShape"_a = std::vector<uint64_t>{},
                       R"nbdoc(
Create an executable instance of this fused equation with bound thor.physical.PhysicalTensor's.
)nbdoc");

    fused_equation.def("run",
                       &FusedEquation::run,
                       "inputs"_a,
                       "output"_a,
                       "stream"_a,
                       R"nbdoc(
Run a fused equation with the thor.physical.PhysicalTensor's provided.
)nbdoc");
}

void bind_stamped_equation(nb::module_& physical) {
    auto stamped_equation = nb::class_<StampedEquation>(physical, "Equation");
    stamped_equation.attr("__module__") = "thor.physical";

    stamped_equation.def("run",
                         &StampedEquation::run,
                         R"nbdoc(
Execute the stamped fused equation on the bound tensors.
        )nbdoc");

    stamped_equation.def_prop_ro("output_tensor",
                                 &StampedEquation::getOutputTensor,
                                 R"nbdoc(
Return the output tensor owned by this equation instance.
        )nbdoc");
}
