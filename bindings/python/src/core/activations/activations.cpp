#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_elu(nb::module_ &m);
void bind_exponential(nb::module_ &m);
void bind_gelu(nb::module_ &m);
void bind_hard_sigmoid(nb::module_ &m);
void bind_relu(nb::module_ &m);
void bind_selu(nb::module_ &m);
void bind_sigmoid(nb::module_ &m);
void bind_soft_plus(nb::module_ &m);
void bind_soft_sign(nb::module_ &m);
void bind_softmax(nb::module_ &m);
void bind_swish(nb::module_ &m);
void bind_tanh(nb::module_ &m);

void bind_activations(nb::module_ &activations) {
    activations.doc() = "Thor activations";

    nb::class_<Activation::Builder>(activations, "Activation");

    bind_elu(activations);
    bind_exponential(activations);
    bind_gelu(activations);
    bind_hard_sigmoid(activations);
    bind_relu(activations);
    bind_selu(activations);
    bind_sigmoid(activations);
    bind_soft_plus(activations);
    bind_soft_sign(activations);
    bind_softmax(activations);
    bind_swish(activations);
    bind_tanh(activations);
}
