#include <nanobind/nanobind.h>

#include <optional>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_drop_out(nb::module_ &m) {
    auto drop_out = nb::class_<DropOut, Layer>(m, "DropOut");
    drop_out.attr("__module__") = "thor.layers";

    drop_out
        .def(
            "__init__",
            [](DropOut *self, Network &network, nb::object feature_input, float drop_proportion) {
                if (drop_proportion < 0.0f || drop_proportion > 1.0f) {
                    string error_message =
                        "Drop Out instance: you must pass 0 <= drop_proportion <= 1. drop_proportion: " + to_string(drop_proportion);
                    throw nb::value_error(error_message.c_str());
                }

                DropOut::Builder builder;
                builder.network(network);
                if (nb::isinstance<RaggedTensor>(feature_input)) {
                    builder.featureInput(nb::cast<RaggedTensor>(feature_input));
                } else if (nb::isinstance<Tensor>(feature_input)) {
                    builder.featureInput(nb::cast<Tensor>(feature_input));
                } else {
                    throw nb::type_error("DropOut feature_input must be thor.Tensor or thor.RaggedTensor.");
                }
                DropOut built = builder.dropProportion(drop_proportion).build();

                // Move the dropout layer into the pre-allocated but uninitialized memory at self
                new (self) DropOut(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "drop_proportion"_a,

            R"nbdoc(
            Create and attach a DropOut layer to a Network.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor or thor.RaggedTensor
                Dense or packed-ragged input. Ragged input preserves its row partition and applies dropout only to active packed values.
            drop_proportion : float
                Fraction of units to drop (0.0 <= p <= 1.0).
            )nbdoc")
        .def(
            "get_feature_output",
            [](DropOut &self) -> nb::object {
                if (std::optional<RaggedTensor> raggedOutput = self.getRaggedFeatureOutput(); raggedOutput.has_value()) {
                    return nb::cast(raggedOutput.value());
                }
                return nb::cast(self.getFeatureOutput().value());
            },
            R"nbdoc(
            Return the logical output. Ragged inputs produce a RaggedTensor with the same row partition.
            )nbdoc")
        .def("get_use_ragged", &DropOut::getUseRagged)
        .def("get_drop_proportion", &DropOut::getDropProportion)
        .def("set_training_dropout_enabled",
             [](DropOut& layer, bool enabled) { layer.setTrainingDropoutEnabled(enabled); },
             "enabled"_a)
        .def("is_training_dropout_enabled",
             [](const DropOut& layer) { return layer.isTrainingDropoutEnabled(); });
}
