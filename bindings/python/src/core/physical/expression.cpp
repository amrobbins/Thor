#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "Utilities/TensorMathFusion/Equation.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"

namespace nb = nanobind;
using namespace nb::literals;

using Expression = ThorImplementation::Expression;
using FusedEquation = ThorImplementation::FusedEquation;
using Equation = ThorImplementation::Equation;
using DataType = ThorImplementation::TensorDescriptor::DataType;
using Tensor = ThorImplementation::Tensor;

void bind_physical_expression(nb::module_& physical) {
    auto expr = nb::class_<Expression>(physical, "Expression");
    expr.attr("__module__") = "thor.physical";

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
            thor.physical.Expr
                Expression representing that input.
        )nbdoc");

    expr.def_static("scalar",
                    &Expression::scalar,
                    "value"_a,
                    R"nbdoc(
            Create a scalar constant expression.
        )nbdoc");

    expr.def("__add__", [](const Expression& a, const Expression& b) { return a + b; }, "other"_a);
    expr.def("__sub__", [](const Expression& a, const Expression& b) { return a - b; }, "other"_a);
    expr.def("__mul__", [](const Expression& a, const Expression& b) { return a * b; }, "other"_a);
    expr.def("__truediv__", [](const Expression& a, const Expression& b) { return a / b; }, "other"_a);

    expr.def("__add__", [](const Expression& a, float b) { return a + b; }, "other"_a);
    expr.def("__sub__", [](const Expression& a, float b) { return a - b; }, "other"_a);
    expr.def("__mul__", [](const Expression& a, float b) { return a * b; }, "other"_a);
    expr.def("__truediv__", [](const Expression& a, float b) { return a / b; }, "other"_a);

    expr.def("__radd__", [](const Expression& a, float b) { return Expression::scalar(b) + a; }, "other"_a);
    expr.def("__rsub__", [](const Expression& a, float b) { return Expression::scalar(b) - a; }, "other"_a);
    expr.def("__rmul__", [](const Expression& a, float b) { return Expression::scalar(b) * a; }, "other"_a);
    expr.def("__rtruediv__", [](const Expression& a, float b) { return Expression::scalar(b) / a; }, "other"_a);
}

void bind_fused_equation(nb::module_& physical) {
    auto fused_equation = nb::class_<FusedEquation>(physical, "FusedEquation");
    fused_equation.attr("__module__") = "thor.physical";

    fused_equation.def("instantiate",
                       &FusedEquation::instantiate,
                       "stream"_a,
                       R"nbdoc(
Create an executable instance of this fused equation.
)nbdoc");
}

void bind_equation_instance(nb::module_& physical) {
    auto equation = nb::class_<Equation>(physical, "Equation");
    equation.attr("__module__") = "thor.physical";

    equation.def("run",
                 &Equation::run,
                 "inputs"_a,
                 R"nbdoc(
Execute the fused equation with the given input tensors.
        )nbdoc");

    equation.def("get_output_tensor",
                 &Equation::getOutputTensor,
                 R"nbdoc(
Return the output tensor owned by this equation instance.
        )nbdoc");
}

void bind_physical_compile(nb::module_& physical) {
    physical.def(
        "compile",
        [](const Expression& expr, const std::vector<Tensor>& inputs, DataType dtype, int device_num) {
            return FusedEquation::compile(expr.expression(), inputs, dtype, device_num);
        },
        "expr"_a,
        "inputs"_a,
        "dtype"_a,
        "device_num"_a = 0,
        R"nbdoc(
            Compile an expression into a fused equation.

            Parameters
            ----------
            expr : thor.physical.Expr
                The expression to compile.
            inputs :
                The list of input tensors
            dtype : thor.DataType
                The tensor data type to target.
            device_num : int, default 0
                The GPU device number.

            Returns
            -------
            thor.physical.FusedEquation
                The compiled fused equation.
        )nbdoc");
}
