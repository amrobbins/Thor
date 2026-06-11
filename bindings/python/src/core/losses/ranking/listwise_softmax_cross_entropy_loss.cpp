#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/ListwiseSoftmaxCrossEntropyLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "bindings/python/src/core/losses/ranking/listwise_common.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

void bind_listwise_softmax_cross_entropy_loss(nb::module_ &ranking) {
    auto listwise_softmax_cross_entropy_loss =
        nb::class_<ListwiseSoftmaxCrossEntropyLoss, Loss>(ranking, "ListwiseSoftmaxCrossEntropyLoss");
    listwise_softmax_cross_entropy_loss.attr("__module__") = "thor.losses.ranking";

    listwise_softmax_cross_entropy_loss.def(
        "__init__",
        [](ListwiseSoftmaxCrossEntropyLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float temperature,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<Tensor> mask,
           std::optional<float> loss_weight) {
            const string loss_name = "ListwiseSoftmaxCrossEntropyLoss instance";
            ThorPython::Ranking::ListwiseCommon::validateFixedSizeListwiseTensorArguments(loss_name,
                                                                                         predictions,
                                                                                         labels,
                                                                                         loss_data_type,
                                                                                         reported_loss_shape,
                                                                                         mask);
            ThorPython::Ranking::ListwiseCommon::validatePositiveTemperature(loss_name, "temperature", temperature);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            ListwiseSoftmaxCrossEntropyLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).temperature(temperature).lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            if (mask.has_value())
                builder.mask(mask.value());
            ThorPython::Ranking::ListwiseCommon::setReportedLossShape(builder, reported_loss_shape);
            ListwiseSoftmaxCrossEntropyLoss built = builder.build();

            new (self) ListwiseSoftmaxCrossEntropyLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "temperature"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        "mask"_a.none() = nb::none(),
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a listwise softmax cross entropy loss over fixed-size query/document lists.)nbdoc");

    listwise_softmax_cross_entropy_loss.def_prop_ro("temperature", &ListwiseSoftmaxCrossEntropyLoss::getTemperature);

    listwise_softmax_cross_entropy_loss.attr("__doc__") = R"nbdoc(
Listwise softmax cross entropy over fixed-size query/document lists.

The predictions tensor contains unnormalized model scores for the documents in one fixed-size
list, and the labels tensor contains a target distribution or nonnegative target weights with
the same list dimension. An optional mask tensor may mark padded documents with values <= 0.5
and valid documents with values > 0.5. Masked documents contribute zero loss and zero prediction
gradient; fully masked rows produce zero raw loss and zero prediction gradient. The prediction
ranking distribution is computed from predictions / temperature across the valid documents,
and the raw loss is one scalar per list:

    -sum(labels * log_softmax(predictions / temperature))

When labels sum to one, this is standard listwise softmax cross entropy. When labels are
nonnegative weights, the gradient uses the per-list target mass.
)nbdoc";
}
