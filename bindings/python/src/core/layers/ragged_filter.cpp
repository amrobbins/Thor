#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/RaggedFilter.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_ragged_filter(nb::module_& m) {
    auto layer = nb::class_<RaggedFilter, MultiConnectionLayer>(m, "RaggedFilter");
    layer.attr("__module__") = "thor.layers";
    layer.def(
        "__init__",
        [](RaggedFilter* self, Network& network, RaggedTensor featureInput, RaggedTensor maskInput) {
            RaggedFilter built = RaggedFilter::Builder()
                                     .network(network)
                                     .featureInput(featureInput)
                                     .maskInput(maskInput)
                                     .build();
            new (self) RaggedFilter(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "mask_input"_a,
        R"nbdoc(
Stable-filter every row of a rank-1 RaggedTensor with one BOOLEAN predicate per token.

``mask_input`` must be a scalar BOOLEAN RaggedTensor sharing the exact same
canonical offsets tensor and row-partition descriptor as ``feature_input``.
Selected active tokens preserve their row-local order and are compacted into a
new packed values tensor with a newly produced canonical offsets tensor. Neither
forward nor backward reads inactive packed capacity. Backward writes zero to
active filtered-out feature positions and scatters gradients only to retained
positions; the BOOLEAN mask is non-differentiable.
)nbdoc");
    layer.def("get_feature_output", &RaggedFilter::getRaggedFeatureOutput);
    layer.def("get_feature_input", &RaggedFilter::getRaggedFeatureInput);
    layer.def("get_mask_input", &RaggedFilter::getRaggedMaskInput);
}
