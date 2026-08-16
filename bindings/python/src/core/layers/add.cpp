#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/Add.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_add(nb::module_& m) {
    auto add = nb::class_<Add, MultiConnectionLayer>(m, "Add");
    add.attr("__module__") = "thor.layers";

    add.def(
        "__init__",
        [](Add* self, Network& network, const Tensor& left, const Tensor& right) {
            Add built = Add::Builder().network(network).left(left).right(right).build();
            new (self) Add(std::move(built));
        },
        "network"_a,
        "left"_a,
        "right"_a);

    add.def(
        "__init__",
        [](Add* self, Network& network, const RaggedTensor& left, const RaggedTensor& right) {
            Add built = Add::Builder().network(network).left(left).right(right).build();
            new (self) Add(std::move(built));
        },
        "network"_a,
        "left"_a,
        "right"_a);

    add.def("get_feature_output", [](const Add& self) {
        if (self.getUseRagged()) return nb::cast(self.getRaggedFeatureOutput().value());
        return nb::cast(self.getFeatureOutput().value());
    });
    add.def_prop_ro("use_ragged", &Add::getUseRagged);

    add.attr("__doc__") = R"nbdoc(
Elementwise addition for dense tensors or canonical rank-1 ragged tensors.

Ragged operands must share the exact same row-partition offsets tensor. The
result preserves that partition and executes only over the authoritative active
packed prefix.
)nbdoc";
}
