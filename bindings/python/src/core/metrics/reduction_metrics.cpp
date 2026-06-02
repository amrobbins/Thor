#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Metrics/ReductionMetrics.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

namespace {

template <typename MetricT>
void bind_unary_reduction_metric(nb::module_& metrics, const char* name, const char* doc) {
    auto metric = nb::class_<MetricT, Metric>(metrics, name);
    metric.attr("__module__") = "thor.metrics";

    metric.def(
        "__init__",
        [](MetricT* self, Network& network, Tensor values) {
            typename MetricT::Builder builder;
            builder.network(network).values(values);
            MetricT built = builder.build();
            new (self) MetricT(std::move(built));
        },
        "network"_a,
        "values"_a,
        doc);

    metric.def_prop_ro("values", &MetricT::getValues);
}

}  // namespace

void bind_reduction_metrics(nb::module_& metrics) {
    bind_unary_reduction_metric<Mean>(metrics, "Mean", R"nbdoc(Construct a Mean metric over a values tensor.)nbdoc");
    bind_unary_reduction_metric<Sum>(metrics, "Sum", R"nbdoc(Construct a Sum metric over a values tensor.)nbdoc");
    bind_unary_reduction_metric<Min>(metrics, "Min", R"nbdoc(Construct a Min metric over a values tensor.)nbdoc");
    bind_unary_reduction_metric<Max>(metrics, "Max", R"nbdoc(Construct a Max metric over a values tensor.)nbdoc");

    auto weighted_mean = nb::class_<WeightedMean, Metric>(metrics, "WeightedMean");
    weighted_mean.attr("__module__") = "thor.metrics";

    weighted_mean.def(
        "__init__",
        [](WeightedMean* self, Network& network, Tensor values, Tensor weights) {
            WeightedMean::Builder builder;
            builder.network(network).values(values).weights(weights);
            WeightedMean built = builder.build();
            new (self) WeightedMean(std::move(built));
        },
        "network"_a,
        "values"_a,
        "weights"_a,
        R"nbdoc(Construct a WeightedMean metric over values and weights tensors.)nbdoc");

    weighted_mean.def_prop_ro("values", &WeightedMean::getValues);
    weighted_mean.def_prop_ro("weights", &WeightedMean::getWeights);
}
