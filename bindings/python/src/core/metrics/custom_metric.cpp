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
           MetricAggregation aggregation,
           const std::string& predictionsName,
           const std::string& labelsName,
           const std::string& metricName,
           const std::string& displayName,
           bool usesBatchValidity) {
            CustomMetric::Builder builder;
            builder.network(network)
                .expression(std::move(expression))
                .predictions(std::move(predictions))
                .labels(std::move(labels))
                .aggregation(aggregation)
                .predictionsName(predictionsName)
                .labelsName(labelsName)
                .metricName(metricName)
                .displayName(displayName);
            if (usesBatchValidity)
                builder.usesBatchValidity();

            CustomMetric built = builder.build();
            new (self) CustomMetric(std::move(built));
        },
        "network"_a,
        "expression"_a,
        "predictions"_a,
        "labels"_a,
        "aggregation"_a,
        "predictions_name"_a = "predictions",
        "labels_name"_a = "labels",
        "metric_name"_a = "metric",
        "display_name"_a = "Metric",
        "uses_batch_validity"_a = false,
        R"nbdoc(Construct an expression-backed CustomMetric.)nbdoc");

    custom_metric.def_prop_ro("predictions_name", &CustomMetric::getPredictionsName);
    custom_metric.def_prop_ro("labels_name", &CustomMetric::getLabelsName);
    custom_metric.def_prop_ro("metric_name", &CustomMetric::getMetricName);
    custom_metric.def_prop_ro("display_name", &CustomMetric::getDisplayName);
    custom_metric.def_prop_ro("uses_batch_validity", &CustomMetric::usesBatchValidity);

    custom_metric.attr("__doc__") = R"nbdoc(
Expression-backed custom metric.

Parameters
----------
network : thor.Network
expression : thor.physical.DynamicExpression
predictions : thor.Tensor
labels : thor.Tensor
aggregation : thor.MetricAggregation
    Declares how this metric's scalar batch result combines across an epoch. A ``RATIO`` expression must also emit
    FP32 scalar outputs named ``thor.METRIC_AGGREGATION_NUMERATOR_NAME`` and
    ``thor.METRIC_AGGREGATION_DENOMINATOR_NAME``. These are internal sufficient statistics and do not become public
    network outputs.
predictions_name : str, default "predictions"
labels_name : str, default "labels"
metric_name : str, default "metric"
display_name : str, default "Metric"
uses_batch_validity : bool, default False
    Declares that the expression consumes runtime batch validity. Thor currently supplies it through the reserved
    ``__thor_batch_validity_mask`` FP32 prefix-mask input so invalid tail rows can be excluded from batch-coupled computation.
)nbdoc";
}
