#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/SegmentedBroadcast.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_segmented_broadcast(nb::module_& m) {
    auto layer = nb::class_<SegmentedBroadcast, MultiConnectionLayer>(m, "SegmentedBroadcast");
    layer.attr("__module__") = "thor.layers";
    layer.def(
        "__init__",
        [](SegmentedBroadcast* self, Network& network, const Tensor& featureInput, const RaggedTensor& partitionInput) {
            SegmentedBroadcast built = SegmentedBroadcast::Builder()
                                           .network(network)
                                           .featureInput(featureInput)
                                           .partitionInput(partitionInput)
                                           .build();
            new (self) SegmentedBroadcast(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "partition_input"_a);
    layer.def("get_feature_output", &SegmentedBroadcast::getRaggedFeatureOutput);
    layer.def("get_partition_input", &SegmentedBroadcast::getPartitionInput);
    layer.attr("__doc__") = R"nbdoc(
Broadcast one dense value per batch row to every active token in a ragged row.

``feature_input`` is a normal dense per-example tensor. ``partition_input``
provides only the canonical row offsets/capacity; its packed values are not read.
The output is a ``thor.RaggedTensor`` with the exact same offsets object and with
trailing value dimensions equal to ``feature_input``. Values must be FP16, BF16,
or FP32. FP64 is intentionally unsupported.
)nbdoc";
}
