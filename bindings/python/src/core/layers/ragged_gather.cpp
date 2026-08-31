#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/RaggedGather.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_ragged_gather(nb::module_& m) {
    auto layer = nb::class_<RaggedGather, MultiConnectionLayer>(m, "RaggedGather");
    layer.attr("__module__") = "thor.layers";
    layer.def(
        "__init__",
        [](RaggedGather* self, Network& network, RaggedTensor sourceInput, RaggedTensor indicesInput) {
            RaggedGather built = RaggedGather::Builder()
                                     .network(network)
                                     .sourceInput(sourceInput)
                                     .indicesInput(indicesInput)
                                     .build();
            new (self) RaggedGather(std::move(built));
        },
        "network"_a,
        "source_input"_a,
        "indices_input"_a,
        R"nbdoc(
Gather tokens independently within every row of a rank-1 RaggedTensor.

``indices_input`` must contain scalar UINT32 or UINT64 row-local indices. Its
row partition Q defines the output partition exactly, while ``source_input``
provides source partition P and the output value dtype/trailing shape. Thus
source and indices may have different row lengths. Duplicate indices are valid
and preserve their occurrence order; backward sums their gradient contributions
into the selected source token. Inactive packed source/index capacity is never
read.
)nbdoc");
    layer.def("get_feature_output", &RaggedGather::getRaggedFeatureOutput);
    layer.def("get_source_input", &RaggedGather::getRaggedSourceInput);
    layer.def("get_indices_input", &RaggedGather::getRaggedIndicesInput);
}
