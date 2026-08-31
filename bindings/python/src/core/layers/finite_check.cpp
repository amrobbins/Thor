#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/FiniteCheck.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_finite_check(nb::module_ &m) {
    auto finite_check = nb::class_<FiniteCheck, Layer>(m, "FiniteCheck");
    finite_check.attr("__module__") = "thor.layers";

    finite_check.def(
        "__init__",
        [](FiniteCheck *self,
           Network &network,
           nb::object feature_input,
           string tensor_label,
           bool check_forward,
           bool check_backward,
           bool fail_on_non_finite,
           uint32_t max_reported_indices,
           bool enabled) {
            if (!check_forward && !check_backward)
                throw nb::value_error("FiniteCheck must check forward, backward, or both.");
            if (max_reported_indices > ThorImplementation::FINITE_CHECK_MAX_REPORTED_INDICES)
                throw nb::value_error("FiniteCheck max_reported_indices exceeds the supported maximum of 32.");

            FiniteCheck::Builder builder;
            builder.network(network);
            if (nb::isinstance<RaggedTensor>(feature_input)) {
                builder.featureInput(nb::cast<RaggedTensor>(feature_input));
            } else if (nb::isinstance<Tensor>(feature_input)) {
                builder.featureInput(nb::cast<Tensor>(feature_input));
            } else {
                throw nb::type_error("FiniteCheck feature_input must be thor.Tensor or thor.RaggedTensor.");
            }
            FiniteCheck built = builder.tensorLabel(std::move(tensor_label))
                                    .enabled(enabled)
                                    .checkForward(check_forward)
                                    .checkBackward(check_backward)
                                    .failOnNonFinite(fail_on_non_finite)
                                    .maxReportedIndices(max_reported_indices)
                                    .build();
            new (self) FiniteCheck(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "tensor_label"_a = "",
        "check_forward"_a = true,
        "check_backward"_a = true,
        "fail_on_non_finite"_a = true,
        "max_reported_indices"_a = 8,
        "enabled"_a = true,
        R"nbdoc(
Create and attach a zero-copy finite-value diagnostic layer.

Set ``enabled=False`` to leave the layer in the model as a zero-copy no-op. A
disabled FiniteCheck allocates no diagnostic workspace, launches no check, and
does not synchronize execution.

The forward activation and, when a backward path exists, the incoming gradient
are checked for NaN and infinity values. The layer aliases its input storage in
both directions and allocates no feature or gradient tensor of its own. For a
``RaggedTensor``, only the authoritative active packed prefix ending at
``offsets[B]`` is checked; undefined inactive capacity is deliberately ignored
and the exact row partition is preserved on the output.

On a failure, the report includes the user label, direction, tensor role, API
and physical tensor ids, dtype, shape, checked element count, counts of
NaN/+Inf/-Inf, and sample flat and multidimensional indices.
``fail_on_non_finite=True`` raises immediately; ``False`` writes the report to
stderr and continues.

FiniteCheck is intentionally a debugging barrier. GPU checks synchronize the
layer stream so that a host-visible report or exception is deterministic, and
therefore should not be left enabled in performance runs. Thor emits a warning
when an enabled FiniteCheck is first stamped.
)nbdoc");

    finite_check.def(
        "get_feature_output",
        [](FiniteCheck &self) -> nb::object {
            if (std::optional<RaggedTensor> raggedOutput = self.getRaggedFeatureOutput(); raggedOutput.has_value())
                return nb::cast(raggedOutput.value());
            return nb::cast(self.getFeatureOutput().value());
        },
        R"nbdoc(Return the logical output produced by this layer. Ragged inputs return a thor.RaggedTensor.)nbdoc");
    finite_check.def("get_use_ragged", &FiniteCheck::getUseRagged);
    finite_check.def("get_tensor_label", &FiniteCheck::getTensorLabel);
    finite_check.def("get_enabled", &FiniteCheck::getEnabled);
    finite_check.def("get_check_forward", &FiniteCheck::getCheckForward);
    finite_check.def("get_check_backward", &FiniteCheck::getCheckBackward);
    finite_check.def("get_fail_on_non_finite", &FiniteCheck::getFailOnNonFinite);
    finite_check.def("get_max_reported_indices", &FiniteCheck::getMaxReportedIndices);
}
