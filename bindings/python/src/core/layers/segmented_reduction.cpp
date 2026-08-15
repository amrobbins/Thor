#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/SegmentedReduction.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_segmented_reduction(nb::module_& m) {
    auto reduction = nb::class_<SegmentedReduction, MultiConnectionLayer>(m, "SegmentedReduction");
    reduction.attr("__module__") = "thor.layers";

    auto reductionType = nb::enum_<SegmentedReduction::Type>(reduction, "Type")
                             .value("sum", SegmentedReduction::Type::SUM)
                             .value("mean", SegmentedReduction::Type::MEAN)
                             .value("min", SegmentedReduction::Type::MIN)
                             .value("max", SegmentedReduction::Type::MAX);
    (void)reductionType;

    reduction.def(
        "__init__",
        [](SegmentedReduction* self,
           Network& network,
           const RaggedTensor& featureInput,
           SegmentedReduction::Type reductionType) {
            SegmentedReduction built = SegmentedReduction::Builder()
                                           .network(network)
                                           .featureInput(featureInput)
                                           .reductionType(reductionType)
                                           .build();
            new (self) SegmentedReduction(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "reduction_type"_a);

    reduction.def("get_feature_output", [](const SegmentedReduction& self) { return self.getFeatureOutput().value(); });
    reduction.def_prop_ro("reduction_type", &SegmentedReduction::getReductionType);

    reduction.attr("__doc__") = R"nbdoc(
Reduce each row of a packed ``thor.RaggedTensor`` independently.

The row partition supplies the reduction domains. ``sum``, ``mean``, ``min``,
and ``max`` preserve every fixed trailing value dimension and return a normal dense
``thor.Tensor`` feature shape. At execution time the physical
shape is ``[batch_size, *trailing_dimensions]`` (or ``[batch_size, 1]`` for
scalar ragged values). Empty rows follow the existing Thor segmented-reduction semantics.
)nbdoc";
}
