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

    expr.def("__neg__", [](const Expression& a) { return -a; });

    expr.def("exp",
             nb::overload_cast<>(&Expression::exp, nb::const_),
             R"nbdoc(
Return the elementwise natural exponential of this expression: e^x_i for a tensor of x_i's.
)nbdoc");

    expr.def("exp2",
             nb::overload_cast<>(&Expression::exp2, nb::const_),
             R"nbdoc(
Return the elementwise base-2 exponential of this expression: 2^x_i for a tensor of x_i's.
)nbdoc");

    expr.def("exp10",
             nb::overload_cast<>(&Expression::exp10, nb::const_),
             R"nbdoc(
Return the elementwise base-10 exponential of this expression: 10^x_i for a tensor of x_i's.
)nbdoc");

    expr.def("log",
             nb::overload_cast<>(&Expression::log, nb::const_),
             R"nbdoc(
Return the elementwise natural logarithm of this expression: ln(x_i) for a tensor of x_i's.
)nbdoc");

    expr.def("log",
             nb::overload_cast<float>(&Expression::log, nb::const_),
             "base"_a,
             R"nbdoc(
Return the elementwise logarithm of this expression with the given base: log_base(x_i) for a tensor of x_i's.

Parameters
----------
base : float
    The logarithm base. Must be positive and not equal to 1.
)nbdoc");

    expr.def("log2",
             nb::overload_cast<>(&Expression::log2, nb::const_),
             R"nbdoc(
Return the elementwise base-2 logarithm of this expression: lg(x_i) for a tensor of x_i's.
)nbdoc");

    expr.def("log10",
             nb::overload_cast<>(&Expression::log10, nb::const_),
             R"nbdoc(
Return the elementwise base-10 logarithm of this expression: log_10(x_i) for a tensor of x_i's.
)nbdoc");

    expr.def("sqrt",
             &Expression::sqrt,
             R"nbdoc(
Return the elementwise square root of this expression: sqrt(x_i) for a tensor of x_i's.
)nbdoc");

    expr.def("pow",
             nb::overload_cast<const Expression&>(&Expression::pow, nb::const_),
             "exponent"_a,
             R"nbdoc(
from thor.physical.Expression.scalar import input, scalar

x = input(0)
y = input(1)

expr1 = x.pow(y)      # t1 ^ t2
expr2 = x.pow(2.0)    # t1 ^ s
expr3 = scalar(2.0).pow(x)   # s ^ t1
expr4 = scalar(2.0).pow(3.0)  # s1 ^ s2
)nbdoc");

    expr.def("pow", nb::overload_cast<float>(&Expression::pow, nb::const_), "exponent"_a);
}

void bind_fused_equation(nb::module_& physical) {
    auto fused_equation = nb::class_<FusedEquation>(physical, "FusedEquation");
    fused_equation.attr("__module__") = "thor.physical";

    fused_equation.def("stamp",
                       &FusedEquation::stamp,
                       "inputs"_a,
                       "stream"_a,
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

void bind_physical_compile(nb::module_& physical) {
    physical.def(
        "compile",
        [](const Expression& expr, DataType dtype, int device_num, bool use_fast_math) {
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

            Returns
            -------
            thor.physical.FusedEquation
                The compiled fused equation.

            Example
            -------
            x = thor.physical.Expression.input(0)
            y = thor.physical.Expression.input(1)
            z = thor.physical.Expression.input(1)
            expr = (x / y) * z + 1.5
            eq = thor.physical.compile(expr, dtype=thor.DataType.fp32, device_num=0)

            eq.run([x_tensor, y_tensor, z_tensor], out_tensor, stream)

            -- or --

            stamped = eq.stamp([x_tensor, y_tensor, z_tensor], stream)
            stamped.run()
            out = stamped.output_tensor

            Example Kernel:
                extern "C" __global__
                void fused_kernel(const float* in0, const float* in1, const float* in2, float* out, unsigned long long numel) {
                  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
                  if (idx >= numel) return;

                  float t0 = in0[idx];
                  float t1 = in1[idx];
                  float t2 = t0 / t1;
                  float t3 = in2[idx];
                  float t4 = t2 * t3;
                  float t5 = 1.5f;
                  float t6 = t4 + t5;

                  out[idx] = t6;
}
        )nbdoc");
}
