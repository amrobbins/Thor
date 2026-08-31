#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/RaggedSequenceSlice.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <cstdint>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_ragged_sequence_slice(nb::module_& m) {
    auto layer = nb::class_<RaggedSequenceSlice, MultiConnectionLayer>(m, "RaggedSequenceSlice");
    layer.attr("__module__") = "thor.layers";
    layer.def(
        "__init__",
        [](RaggedSequenceSlice* self,
           Network& network,
           RaggedTensor featureInput,
           uint64_t start,
           uint64_t length) {
            RaggedSequenceSlice built = RaggedSequenceSlice::Builder()
                                               .network(network)
                                               .featureInput(featureInput)
                                               .start(start)
                                               .length(length)
                                               .build();
            new (self) RaggedSequenceSlice(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "start"_a,
        "length"_a,
        R"nbdoc(
Slice every logical row of a rank-1 RaggedTensor along the variable-length sequence axis.

``start`` is a non-negative row-local token offset and ``length`` must be
positive. Each row contributes at most ``length`` tokens beginning at ``start``;
short rows are clipped independently and rows no longer than ``start`` become
empty. Selected values are compacted and the layer explicitly produces a new
canonical offsets tensor rather than preserving the input partition. Inactive
packed capacity is never read, and backward writes exact zero to active input
positions outside the selected window while leaving inactive gradient capacity
undefined.
)nbdoc");
    layer.def("get_feature_output", &RaggedSequenceSlice::getRaggedFeatureOutput);
    layer.def_prop_ro("start", &RaggedSequenceSlice::getStart);
    layer.def_prop_ro("length", &RaggedSequenceSlice::getLength);
}
