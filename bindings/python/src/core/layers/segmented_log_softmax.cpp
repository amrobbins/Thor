#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/SegmentedLogSoftmax.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_segmented_log_softmax(nb::module_& m) {
    auto layer = nb::class_<SegmentedLogSoftmax, MultiConnectionLayer>(m, "SegmentedLogSoftmax");
    layer.attr("__module__") = "thor.layers";
    layer.def(
        "__init__",
        [](SegmentedLogSoftmax* self, Network& network, const RaggedTensor& featureInput) {
            SegmentedLogSoftmax built = SegmentedLogSoftmax::Builder().network(network).featureInput(featureInput).build();
            new (self) SegmentedLogSoftmax(std::move(built));
        },
        "network"_a,
        "feature_input"_a);
    layer.def("get_feature_output", &SegmentedLogSoftmax::getRaggedFeatureOutput);
    layer.attr("__doc__") = R"nbdoc(
Log-softmax across the active tokens of each ragged row.

Each trailing component is normalized independently over its row's variable-length
token axis. The exact canonical offsets object is preserved and inactive packed
capacity is excluded. Values must be FP16, BF16, or FP32; FP64 is intentionally
unsupported.
)nbdoc";
}
