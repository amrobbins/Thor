#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "Utilities/TensorMathFusion/Equation.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"

namespace nb = nanobind;
using namespace nb::literals;

using Expr = ThorImplementation::Expression;
using FusedEquation = ThorImplementation::FusedEquation;
using EquationInstance = ThorImplementation::Equation;

void bind_physical_expression(nb::module_& physical) {
    auto expr = nb::class_<Expr>(physical, "Expr");
    expr.attr("__module__") = "thor.physical";

    expr.def_static("input",
                    &Expr::input,
                    "input_index"_a,
                    R"nbdoc(
            Create an expression input node.

            Parameters
            ----------
            input_index : int
                The zero-based input number.

            Returns
            -------
            thor.physical.Expr
                Expression representing that input.
        )nbdoc");

    expr.def_static("scalar",
                    &Expr::scalar,
                    "value"_a,
                    R"nbdoc(
            Create a scalar constant expression.
        )nbdoc");

    expr.def("__add__", [](const Expr& a, const Expr& b) { return a + b; }, "other"_a);
    expr.def("__sub__", [](const Expr& a, const Expr& b) { return a - b; }, "other"_a);
    expr.def("__mul__", [](const Expr& a, const Expr& b) { return a * b; }, "other"_a);
    expr.def("__truediv__", [](const Expr& a, const Expr& b) { return a / b; }, "other"_a);

    expr.def("__add__", [](const Expr& a, float b) { return a + b; }, "other"_a);
    expr.def("__sub__", [](const Expr& a, float b) { return a - b; }, "other"_a);
    expr.def("__mul__", [](const Expr& a, float b) { return a * b; }, "other"_a);
    expr.def("__truediv__", [](const Expr& a, float b) { return a / b; }, "other"_a);

    expr.def("__radd__", [](const Expr& a, float b) { return Expr::scalar(b) + a; }, "other"_a);
    expr.def("__rsub__", [](const Expr& a, float b) { return Expr::scalar(b) - a; }, "other"_a);
    expr.def("__rmul__", [](const Expr& a, float b) { return Expr::scalar(b) * a; }, "other"_a);
    expr.def("__rtruediv__", [](const Expr& a, float b) { return Expr::scalar(b) / a; }, "other"_a);
}

void bind_fused_equation(nb::module_& physical) {
    auto fused_equation = nb::class_<FusedEquation>(physical, "FusedEquation");
    fused_equation.attr("__module__") = "thor.physical";

    fused_equation.def("instantiate",
                       &FusedEquation::instantiate,
                       "output"_a,
                       "stream"_a,
                       R"nbdoc(
Create an executable instance of this fused equation.
)nbdoc");
}

void bind_equation_instance(nb::module_& physical) {
    auto equation_instance = nb::class_<EquationInstance>(physical, "EquationInstance");
    equation_instance.attr("__module__") = "thor.physical";

    equation_instance.def("run",
                          &EquationInstance::run,
                          "inputs"_a,
                          R"nbdoc(
            Execute the fused equation with the given input tensors.
        )nbdoc");
}
