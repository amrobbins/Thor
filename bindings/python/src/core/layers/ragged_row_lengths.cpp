#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/RaggedRowLengths.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_ragged_row_lengths(nb::module_& m) {
    auto lengths = nb::class_<RaggedRowLengths, Layer>(m, "RaggedRowLengths");
    lengths.attr("__module__") = "thor.layers";

    lengths.def(
        "__init__",
        [](RaggedRowLengths* self, Network& network, const RaggedTensor& featureInput) {
            RaggedRowLengths built = RaggedRowLengths::Builder().network(network).featureInput(featureInput).build();
            new (self) RaggedRowLengths(std::move(built));
        },
        "network"_a,
        "feature_input"_a);
    lengths.def("get_feature_output", [](const RaggedRowLengths& self) { return self.getFeatureOutput().value(); });

    lengths.attr("__doc__") = R"nbdoc(
Materialize canonical ragged row lengths as dense INT32 logical ``[1]`` values.

For offsets ``[0, 371, 558, 612]`` the physical output is ``[[371], [187], [54]]``.
The layer depends only on row-partition offsets, not on packed values.
)nbdoc";
}
