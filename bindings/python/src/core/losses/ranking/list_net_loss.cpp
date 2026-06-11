#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/ListNetLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "bindings/python/src/core/losses/ranking/listwise_common.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

void bind_list_net_loss(nb::module_ &ranking) {
    auto list_net_loss = nb::class_<ListNetLoss, Loss>(ranking, "ListNetLoss");
    list_net_loss.attr("__module__") = "thor.losses.ranking";

    list_net_loss.def(
        "__init__",
        [](ListNetLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float score_temperature,
           float label_temperature,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<Tensor> mask,
           std::optional<float> loss_weight) {
            const string loss_name = "ListNetLoss instance";
            ThorPython::Ranking::ListwiseCommon::validateFixedSizeListwiseTensorArguments(loss_name,
                                                                                         predictions,
                                                                                         labels,
                                                                                         loss_data_type,
                                                                                         reported_loss_shape,
                                                                                         mask);
            ThorPython::Ranking::ListwiseCommon::validatePositiveTemperature(loss_name, "score_temperature", score_temperature);
            ThorPython::Ranking::ListwiseCommon::validatePositiveTemperature(loss_name, "label_temperature", label_temperature);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            ListNetLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .scoreTemperature(score_temperature)
                .labelTemperature(label_temperature)
                .lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            if (mask.has_value())
                builder.mask(mask.value());
            ThorPython::Ranking::ListwiseCommon::setReportedLossShape(builder, reported_loss_shape);
            ListNetLoss built = builder.build();

            new (self) ListNetLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "score_temperature"_a = 1.0f,
        "label_temperature"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        "mask"_a.none() = nb::none(),
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a ListNet loss over fixed-size query/document lists.)nbdoc");

    list_net_loss.def_prop_ro("score_temperature", &ListNetLoss::getScoreTemperature);
    list_net_loss.def_prop_ro("label_temperature", &ListNetLoss::getLabelTemperature);

    list_net_loss.attr("__doc__") = R"nbdoc(
ListNet loss over fixed-size query/document lists.

The predictions tensor contains unnormalized model scores for the documents in one fixed-size
list, and the labels tensor contains relevance labels with the same list dimension. The target
ranking distribution is computed with softmax(labels / label_temperature). The prediction
ranking distribution is computed from predictions / score_temperature, and the raw loss is one
scalar per list:

    -sum(target * log_softmax(predictions / score_temperature))

The current implementation supports fixed-size lists. Use the optional mask tensor for padded
fixed-size lists. Mask values > 0.5 mark valid documents; values <= 0.5 mark padded documents.
Masked documents contribute zero loss and zero prediction gradient, and fully masked rows
produce zero raw loss and zero prediction gradient. True ragged query/document groups require
later segment/ragged scaffolding.
)nbdoc";
}
