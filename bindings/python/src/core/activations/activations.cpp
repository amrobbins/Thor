#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "Utilities/Expression/Expression.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_glu(nb::module_ &m);
void bind_reglu(nb::module_ &m);
void bind_geglu(nb::module_ &m);
void bind_swiglu(nb::module_ &m);
void bind_bilinear_glu(nb::module_ &m);
void bind_elu(nb::module_ &m);
void bind_exponential(nb::module_ &m);
void bind_gelu(nb::module_ &m);
void bind_hard_sigmoid(nb::module_ &m);
void bind_hard_swish(nb::module_ &m);
void bind_hard_tanh(nb::module_ &m);
void bind_mish(nb::module_ &m);
void bind_relu(nb::module_ &m);
void bind_relu6(nb::module_ &m);
void bind_selu(nb::module_ &m);
void bind_sigmoid(nb::module_ &m);
void bind_soft_plus(nb::module_ &m);
void bind_soft_sign(nb::module_ &m);
void bind_softmax(nb::module_ &m);
void bind_swish(nb::module_ &m);
void bind_tanh(nb::module_ &m);
void bind_threshold(nb::module_ &m);

void bind_activations(nb::module_ &activations) {
    activations.doc() = "Thor activations";

    auto activation = nb::class_<Activation>(activations, "Activation");
    activation.attr("__module__") = "thor.activations";
    activation.def(
        "to_expression",
        [](const Activation& self, const ThorImplementation::Expression& input) { return self.toExpression(input); },
        "input"_a,
        R"nbdoc(Return an expression equivalent to applying this activation to the supplied expression.)nbdoc");

    bind_glu(activations);
    bind_reglu(activations);
    bind_geglu(activations);
    bind_swiglu(activations);
    bind_bilinear_glu(activations);
    bind_elu(activations);
    bind_exponential(activations);
    bind_gelu(activations);
    bind_hard_sigmoid(activations);
    bind_hard_swish(activations);
    bind_hard_tanh(activations);
    bind_mish(activations);
    bind_relu(activations);
    bind_relu6(activations);
    bind_selu(activations);
    bind_sigmoid(activations);
    bind_soft_plus(activations);
    bind_soft_sign(activations);
    bind_softmax(activations);
    bind_swish(activations);
    bind_tanh(activations);
    bind_threshold(activations);
}
