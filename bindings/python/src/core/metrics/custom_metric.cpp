#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <utility>

#include "DeepLearning/Api/Layers/Metrics/CustomMetric.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/DynamicExpression.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;
using DynamicExpression = ThorImplementation::DynamicExpression;

void bind_custom_metric(nb::module_& metrics) {
    auto custom_metric = nb::class_<CustomMetric, Metric>(metrics, "CustomMetric");
    custom_metric.attr("__module__") = "thor.metrics";

    custom_metric.def(
        "__init__",
        [](CustomMetric* self,
           Network& network,
           DynamicExpression expression,
           Tensor predictions,
           Tensor labels,
           const std::string& predictionsName,
           const std::string& labelsName,
           const std::string& metricName,
           const std::string& displayName) {
            CustomMetric::Builder builder;
            builder.network(network)
                .expression(std::move(expression))
                .predictions(std::move(predictions))
                .labels(std::move(labels))
                .predictionsName(predictionsName)
                .labelsName(labelsName)
                .metricName(metricName)
                .displayName(displayName);

            CustomMetric built = builder.build();
            new (self) CustomMetric(std::move(built));
        },
        "network"_a,
        "expression"_a,
        "predictions"_a,
        "labels"_a,
        "predictions_name"_a = "predictions",
        "labels_name"_a = "labels",
        "metric_name"_a = "metric",
        "display_name"_a = "Metric",
        R"nbdoc(Construct an expression-backed CustomMetric.)nbdoc");

    custom_metric.def_prop_ro("predictions_name", &CustomMetric::getPredictionsName);
    custom_metric.def_prop_ro("labels_name", &CustomMetric::getLabelsName);
    custom_metric.def_prop_ro("metric_name", &CustomMetric::getMetricName);
    custom_metric.def_prop_ro("display_name", &CustomMetric::getDisplayName);

    custom_metric.attr("__doc__") = R"nbdoc(
Expression-backed custom metric.

Parameters
----------
network : thor.Network
expression : thor.physical.DynamicExpression
predictions : thor.Tensor
labels : thor.Tensor
predictions_name : str, default "predictions"
labels_name : str, default "labels"
metric_name : str, default "metric"
display_name : str, default "Metric"
)nbdoc";
}
