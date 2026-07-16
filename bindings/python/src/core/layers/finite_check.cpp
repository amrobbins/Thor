#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/FiniteCheck.h"
#include "DeepLearning/Api/Network/Network.h"
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
           const Tensor &feature_input,
           string tensor_label,
           bool check_forward,
           bool check_backward,
           bool fail_on_non_finite,
           uint32_t max_reported_indices) {
            if (!check_forward && !check_backward)
                throw nb::value_error("FiniteCheck must check forward, backward, or both.");
            if (max_reported_indices > ThorImplementation::FINITE_CHECK_MAX_REPORTED_INDICES)
                throw nb::value_error("FiniteCheck max_reported_indices exceeds the supported maximum of 32.");

            FiniteCheck::Builder builder;
            FiniteCheck built = builder.network(network)
                                    .featureInput(feature_input)
                                    .tensorLabel(std::move(tensor_label))
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
        R"nbdoc(
Create and attach a zero-copy finite-value diagnostic layer.

The forward activation and, when a backward path exists, the incoming gradient
are checked for NaN and infinity values. The layer aliases its input storage in
both directions and allocates no feature or gradient tensor of its own.

On a failure, the report includes the user label, direction, tensor role, API
and physical tensor ids, dtype, shape, counts of NaN/+Inf/-Inf, and sample flat
and multidimensional indices. ``fail_on_non_finite=True`` raises immediately;
``False`` writes the report to stderr and continues.

FiniteCheck is intentionally a debugging barrier. GPU checks synchronize the
layer stream so that a host-visible report or exception is deterministic, and
therefore should not be left in performance runs.
)nbdoc");

    finite_check.def(
        "get_feature_output",
        [](FiniteCheck &self) -> Tensor { return self.getFeatureOutput().value(); },
        R"nbdoc(Return the logical output tensor produced by this layer.)nbdoc");
    finite_check.def("get_tensor_label", &FiniteCheck::getTensorLabel);
    finite_check.def("get_check_forward", &FiniteCheck::getCheckForward);
    finite_check.def("get_check_backward", &FiniteCheck::getCheckBackward);
    finite_check.def("get_fail_on_non_finite", &FiniteCheck::getFailOnNonFinite);
    finite_check.def("get_max_reported_indices", &FiniteCheck::getMaxReportedIndices);
}
