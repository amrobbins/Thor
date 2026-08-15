#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Slice.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <cstdint>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_slice(nb::module_& m) {
    auto slice = nb::class_<Slice, Layer>(m, "Slice");
    slice.attr("__module__") = "thor.layers";

    slice.def(
        "__init__",
        [](Slice* self, Network& network, nb::object featureInput, uint64_t axis, int64_t start, uint64_t length) {
            Slice::Builder builder;
            builder.network(network).axis(axis).start(start).length(length);
            if (nb::isinstance<RaggedTensor>(featureInput)) {
                builder.featureInput(nb::cast<RaggedTensor>(featureInput));
            } else if (nb::isinstance<Tensor>(featureInput)) {
                builder.featureInput(nb::cast<Tensor>(featureInput));
            } else {
                throw nb::type_error("Slice feature_input must be thor.Tensor or thor.RaggedTensor.");
            }
            Slice built = builder.build();
            new (self) Slice(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "axis"_a,
        "start"_a,
        "length"_a);

    slice.def("get_feature_output", [](const Slice& self) -> nb::object {
        if (std::optional<RaggedTensor> raggedOutput = self.getRaggedFeatureOutput(); raggedOutput.has_value()) {
            return nb::cast(raggedOutput.value());
        }
        return nb::cast(self.getFeatureOutput().value());
    });
    slice.def("get_use_ragged", &Slice::getUseRagged);
    slice.def_prop_ro("axis", &Slice::getAxis);
    slice.def_prop_ro("start", &Slice::getStart);
    slice.def_prop_ro("length", &Slice::getLength);

    slice.attr("__doc__") = R"nbdoc(
Slice a contiguous window from one logical tensor axis.

``feature_input`` may be a dense ``thor.Tensor`` or packed ``thor.RaggedTensor``. For ragged input, ``axis`` addresses only the fixed trailing value dimensions and the row partition is preserved.

The batch dimension is excluded from ``axis``. Negative ``start`` values are
resolved relative to the end of the selected logical axis. The operation is
serialized declaratively as ``axis``, ``start``, and ``length`` and remains
batch-polymorphic when the network is placed, cloned into training phases, or
saved and reloaded.
)nbdoc";
}
