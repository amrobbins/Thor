#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Metrics/CategoricalAccuracy.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using LabelType = Loss::LabelType;

void bind_categorical_accuracy(nb::module_ &metrics) {
    auto categorical_accuracy = nb::class_<CategoricalAccuracy, Metric>(metrics, "CategoricalAccuracy");
    categorical_accuracy.attr("__module__") = "thor.metrics";

    categorical_accuracy.def(
        "__init__",
        [](CategoricalAccuracy *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           LabelType label_type,
           std::optional<int32_t> num_classes) {
            CategoricalAccuracy::Builder builder;
            builder.network(network).predictions(predictions).labels(labels);

            // Ensure everything matches up.
            if (label_type == LabelType::ONE_HOT) {
                if (predictions.getDimensions().size() != 1) {
                    string error_message = "CategoricalAccuracy instance: one_hot predictions must have 1 dimension but tensor format is " +
                                           predictions.getDescriptorString();
                    throw nb::value_error(error_message.c_str());
                }
                if (labels.getDimensions().size() != 1) {
                    string error_message = "CategoricalAccuracy instance: one_hot labels must have 1 dimension but tensor format is " +
                                           labels.getDescriptorString();
                    throw nb::value_error(error_message.c_str());
                }
                if (num_classes.has_value() &&
                    (uint64_t(num_classes.value()) != predictions.getDimensions()[0] || num_classes.value() <= 0)) {
                    string error_message = "CategoricalAccuracy instance: mismatch between num_classes " + to_string(num_classes.value()) +
                                           " and predictions tensor size " + to_string(predictions.getDimensions()[0]) +
                                           ". Either set num_classes to match, don't pass num_classes, or fix your predictions tensor.";
                    throw nb::value_error(error_message.c_str());
                }
                if (predictions.getDimensions()[0] != labels.getDimensions()[0]) {
                    string error_message = "CategoricalAccuracy instance: mismatch between predictions size " +
                                           to_string(predictions.getDimensions()[0]) + " and labels tensor size " +
                                           to_string(labels.getDimensions()[0]);
                    throw nb::value_error(error_message.c_str());
                }

                builder.receivesOneHotLabels();
            } else if (label_type == LabelType::INDEX) {
                if (!num_classes.has_value()) {
                    throw nb::value_error(
                        "CategoricalAccuracy instance: label_type set to LabelType.index but num_classes is None. You must pass "
                        "num_classes in this case.");
                } else if (num_classes.value() <= 0) {
                    string error_message =
                        "CategoricalAccuracy instance: num_classes must be a positive integer when using index labels. You passed "
                        "num_classes == " +
                        to_string(num_classes.value());
                    throw nb::value_error(error_message.c_str());
                }

                if (labels.getDimensions().size() != 1 || labels.getDimensions()[0] != 1) {
                    string error_message = "CategoricalAccuracy instance: labels tensor is not sized right. label tensor is " +
                                           labels.getDescriptorString() +
                                           ". labels must be a 1 dimensional tensor of size 1. since label_type == index";
                    throw nb::value_error(error_message.c_str());
                }

                builder.receivesClassIndexLabels(num_classes.value());
            } else {
                string error_message =
                    "Invalid value " + to_string((int)label_type) + " passed for enum LabelType to CategoricalAccuracy instance.";
                throw nb::value_error(error_message.c_str());
            }

            CategoricalAccuracy built = builder.build();

            new (self) CategoricalAccuracy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "label_type"_a,
        "num_classes"_a.none() = nb::none(),
        R"nbdoc(Construct a Categorical Accuracy metric.)nbdoc");

    categorical_accuracy.attr("__doc__") = R"nbdoc(
Categorical Accuracy metric.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
labels : thor.Tensor
)nbdoc";
}
