#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Metrics/BinaryAccuracy.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_binary_accuracy(nb::module_ &metrics) {
    auto binary_accuracy = nb::class_<BinaryAccuracy, Metric>(metrics, "BinaryAccuracy");
    binary_accuracy.attr("__module__") = "thor.metrics";

    binary_accuracy.def(
        "__init__",
        [](BinaryAccuracy *self, Network &network, Tensor predictions, Tensor labels) {
            BinaryAccuracy::Builder builder;
            builder.network(network).predictions(predictions).labels(labels);

            BinaryAccuracy built = builder.build();

            new (self) BinaryAccuracy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        R"nbdoc(Construct a Binary Accuracy metric.)nbdoc");

    binary_accuracy.def(
        "__init__",
        [](BinaryAccuracy *self, Network &network, RaggedTensor predictions, RaggedTensor labels) {
            BinaryAccuracy::Builder builder;
            builder.network(network).predictions(std::move(predictions)).labels(std::move(labels));
            BinaryAccuracy built = builder.build();
            new (self) BinaryAccuracy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        R"nbdoc(Construct a Binary Accuracy metric over same-partition ragged tokens.)nbdoc");

    binary_accuracy.def_prop_ro("ragged_predictions", &BinaryAccuracy::getRaggedPredictions);
    binary_accuracy.def_prop_ro("ragged_labels", &BinaryAccuracy::getRaggedLabels);

    binary_accuracy.attr("__doc__") = R"nbdoc(
Binary Accuracy metric.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
labels : thor.Tensor
)nbdoc";
}
