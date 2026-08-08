#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Layers/Utility/Einsum.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "bindings/python/src/core/cast.h"

#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;
namespace pybind = Thor::PythonBindings;

namespace {

std::vector<Tensor> featureInputsFromPython(const nb::object& featureInputsObj) {
    if (!nb::isinstance<nb::sequence>(featureInputsObj) || nb::isinstance<nb::str>(featureInputsObj)) {
        throw nb::type_error(("Einsum() argument 'feature_inputs': expected a non-empty sequence of thor.Tensor, got " +
                              pybind::pythonTypeName(featureInputsObj))
                                 .c_str());
    }

    nb::sequence featureInputs = pybind::castOrTypeError<nb::sequence>(
        featureInputsObj, "Einsum() argument 'feature_inputs'", "non-empty sequence of thor.Tensor", false);
    if (nb::len(featureInputs) == 0) {
        throw nb::value_error("Einsum() argument 'feature_inputs' must be non-empty.");
    }

    std::vector<Tensor> tensors;
    tensors.reserve(nb::len(featureInputs));
    size_t index = 0;
    for (nb::handle item : featureInputs) {
        const std::string context = "Einsum() argument 'feature_inputs'[" + std::to_string(index) + "]";
        tensors.push_back(pybind::castOrTypeError<Tensor>(item, context, "thor.Tensor", false));
        ++index;
    }
    return tensors;
}

}  // namespace

void bind_einsum(nb::module_& m) {
    auto einsum = nb::class_<Einsum, MultiConnectionLayer>(m, "Einsum");
    einsum.attr("__module__") = "thor.layers";

    einsum.def(
        "__init__",
        [](Einsum* self, Network& network, const std::string& equation, const nb::object& featureInputsObj) {
            std::vector<Tensor> featureInputs = featureInputsFromPython(featureInputsObj);

            Einsum::Builder builder;
            builder.network(network).equation(equation).featureInputs(std::move(featureInputs));
            Einsum built = builder.build();
            new (self) Einsum(std::move(built));
        },
        "network"_a,
        "equation"_a,
        "feature_inputs"_a,
        R"nbdoc(
Create and attach a symbolic Einsum layer to a Network.

The equation describes per-example feature dimensions. Thor's physical batch
axis is implicit and is always preserved. Einsum owns no trainable parameters;
all operands are ordinary graph tensors and gradients propagate to every live
operand occurrence.

Parameters
----------
network : thor.Network
    Network the layer should be added to.
equation : str
    Explicit einsum equation, for example ``"ik,kj->ij"``.
feature_inputs : sequence[thor.Tensor]
    Operand tensors in equation order. Repeating the same symbolic tensor in
    multiple operand positions is supported.
)nbdoc");

    einsum.def("get_equation", [](const Einsum& self) { return std::string(self.getEquation()); });
    einsum.def(
        "get_feature_output",
        [](Einsum& self) -> Tensor { return self.getFeatureOutput().value(); },
        R"nbdoc(Return the symbolic tensor produced by this einsum layer.)nbdoc");
}
