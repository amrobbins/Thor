#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Metrics/Metric.h"

namespace nb = nanobind;

void bind_binary_accuracy(nb::module_ &metrics);
void bind_categorical_accuracy(nb::module_ &metrics);

void bind_metrics(nb::module_ &metrics) {
    metrics.doc() = "Thor metrics";

    auto metric = nb::class_<Thor::Metric>(metrics, "Metric");
    metric.attr("__module__") = "thor.metrics";

    bind_binary_accuracy(metrics);
    bind_categorical_accuracy(metrics);
}
