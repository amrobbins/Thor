#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/SegmentedSoftmax.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_segmented_softmax(nb::module_& m) {
    auto layer = nb::class_<SegmentedSoftmax, MultiConnectionLayer>(m, "SegmentedSoftmax");
    layer.attr("__module__") = "thor.layers";
    layer.def(
        "__init__",
        [](SegmentedSoftmax* self, Network& network, const RaggedTensor& featureInput) {
            SegmentedSoftmax built = SegmentedSoftmax::Builder().network(network).featureInput(featureInput).build();
            new (self) SegmentedSoftmax(std::move(built));
        },
        "network"_a,
        "feature_input"_a);
    layer.def("get_feature_output", &SegmentedSoftmax::getRaggedFeatureOutput);
    layer.attr("__doc__") = R"nbdoc(
Softmax across the active tokens of each ragged row.

Each trailing component is normalized independently over its row's variable-length
token axis. The exact canonical offsets object is preserved and inactive packed
capacity is excluded. Values must be FP16, BF16, or FP32; FP64 is intentionally
unsupported. This is distinct from ordinary ``Softmax``, which is not a segmented
sequence-axis operation.
)nbdoc";
}
