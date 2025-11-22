#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_relu(nb::module_ &m);

void bind_activations(nb::module_ &activations) {
    activations.doc() = "Thor activations";

    nb::class_<Activation::Builder>(activations, "Activation");

    activations.def("Elu", []() { return "temp"; });
    activations.def("Exponential", []() { return "temp"; });
    activations.def("Gelu", []() { return "temp"; });
    activations.def("HardSigmoid", []() { return "temp"; });
    activations.def("Relu", []() { return "temp"; });
    bind_relu(activations);
    activations.def("Selu", []() { return "temp"; });
    activations.def("Sigmoid", []() { return "temp"; });
    activations.def("SoftMax", []() { return "temp"; });
    activations.def("SoftPlus", []() { return "temp"; });
    activations.def("SoftSign", []() { return "temp"; });
    activations.def("Swish", []() { return "temp"; });
    activations.def("Tanh", []() { return "temp"; });
}
