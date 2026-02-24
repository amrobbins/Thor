#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Tensor::DataType;

void bind_binary_cross_entropy(nb::module_ &losses) {
    auto binary_cross_entropy = nb::class_<BinaryCrossEntropy, Loss>(losses, "BinaryCrossEntropy");
    binary_cross_entropy.attr("__module__") = "thor.losses";

    binary_cross_entropy.def(
        "__init__",
        [](BinaryCrossEntropy *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           std::optional<bool> reportsBatchLoss,
           std::optional<bool> reportsElementwiseLoss,
           DataType loss_data_type) {
            BinaryCrossEntropy::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).lossDataType(loss_data_type);

            if (!reportsBatchLoss.has_value() && !reportsElementwiseLoss.has_value()) {
                builder.reportsBatchLoss();
            } else {
                if (reportsBatchLoss.has_value() && reportsElementwiseLoss.has_value()) {
                    if (reportsBatchLoss.value() == reportsElementwiseLoss.value()) {
                        throw nb::value_error(
                            "reports_batch_loss and reports_elementwise_loss cannot be equal when both are provided. "
                            "Provide only one, or set them to opposite values.");
                    } else if (reportsBatchLoss.value() == true) {
                        builder.reportsBatchLoss();
                    } else {
                        builder.reportsElementwiseLoss();
                    }
                }
            }

            BinaryCrossEntropy built = builder.build();

            new (self) BinaryCrossEntropy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "reports_batch_loss"_a.none() = nb::none(),
        "reports_elementwise_loss"_a.none() = nb::none(),
        "loss_data_type"_a = DataType::FP32,
        // nb::sig("def __init__(self, "
        //         "network: thor.Network, "
        //         "predictions: thor.Tensor, "
        //         "labels: thor.Tensor, "
        //         "reports_batch_loss: bool | None = None, "
        //         "reports_elementwise_loss: bool | None = None, "
        //         "loss_data_type: thor.DataType = thor.DataType.fp32"
        //         ") -> None"),
        R"nbdoc(Construct a Binary Cross Entropy loss.)nbdoc");

    binary_cross_entropy.attr("__doc__") = R"nbdoc(
Binary cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
labels : thor.Tensor
reports_batch_loss : Optional[bool], default None
    If True, report a single batch-aggregated loss.
    When reports_batch_loss and reports_elementwise_loss are None, defaults to batch loss.
reports_elementwise_loss : Optional[bool], default None
    If True, report elementwise loss.
    When reports_batch_loss and reports_elementwise_loss are None, defaults to batch loss.
loss_data_type : thor.DataType, default thor.DataType.FP32
)nbdoc";
}
