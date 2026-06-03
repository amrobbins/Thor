#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "DeepLearning/Api/Layers/Metrics/LossMetric.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_loss_metric(nb::module_& metrics) {
    nb::enum_<LossMetric::Formula>(metrics, "LossFormula")
        .value("mean_squared_error", LossMetric::Formula::MEAN_SQUARED_ERROR)
        .value("mean_absolute_error", LossMetric::Formula::MEAN_ABSOLUTE_ERROR)
        .value("mean_absolute_percentage_error", LossMetric::Formula::MEAN_ABSOLUTE_PERCENTAGE_ERROR)
        .export_values();

    auto loss_metric = nb::class_<LossMetric, Metric>(metrics, "LossMetric");
    loss_metric.attr("__module__") = "thor.metrics";

    loss_metric.def(
        "__init__",
        [](LossMetric* self,
           Network& network,
           Tensor predictions,
           Tensor labels,
           LossMetric::Formula formula,
           std::optional<float> epsilon,
           std::optional<float> max_magnitude,
           std::optional<std::string> display_name) {
            LossMetric::Builder builder;
            builder.network(network).predictions(predictions).labels(labels);
            if (formula == LossMetric::Formula::MEAN_ABSOLUTE_PERCENTAGE_ERROR) {
                builder.meanAbsolutePercentageError(epsilon.value_or(0.0001f), max_magnitude.value_or(1000.0f));
            } else {
                builder.formula(formula);
            }
            if (display_name.has_value())
                builder.displayName(display_name.value());
            LossMetric built = builder.build();
            new (self) LossMetric(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "formula"_a = LossMetric::Formula::MEAN_SQUARED_ERROR,
        "epsilon"_a.none() = nb::none(),
        "max_magnitude"_a.none() = nb::none(),
        "display_name"_a.none() = nb::none(),
        R"nbdoc(Track a loss formula as a forward-only metric.)nbdoc");

    loss_metric.def_prop_ro("formula", &LossMetric::getFormula);
    loss_metric.def_prop_ro("predictions", &LossMetric::getPredictions);
    loss_metric.def_prop_ro("labels", &LossMetric::getLabels);
    loss_metric.def_prop_ro("display_name", &LossMetric::getDisplayName);
}
