#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

namespace {
void validateReportedLossShape(LossShape reported_loss_shape, const string &loss_name) {
    if (reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::CLASSWISE &&
        reported_loss_shape != LossShape::ELEMENTWISE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

template <typename BuilderT>
void setReportedLossShape(BuilderT &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::CLASSWISE) {
        builder.reportsClasswiseLoss();
    } else if (reported_loss_shape == LossShape::ELEMENTWISE) {
        builder.reportsElementwiseLoss();
    } else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

void validateCategoricalCommon(const string &loss_name, Tensor predictions, DataType loss_data_type, LossShape reported_loss_shape) {
    if (predictions.getDimensions().size() != 1 || predictions.getDimensions()[0] <= 1) {
        string error_message = loss_name + ": predictions must be a 1 dimensional tensor with more than one class but predictions is " +
                               predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (loss_data_type != DataType::FP16 && loss_data_type != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_categorical_cross_entropy(nb::module_ &losses) {
    auto categorical_cross_entropy = nb::class_<CategoricalCrossEntropy, Loss>(losses, "CategoricalCrossEntropy");
    categorical_cross_entropy.attr("__module__") = "thor.losses";

    categorical_cross_entropy.def(
        "__init__",
        [](CategoricalCrossEntropy *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           DataType loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "CategoricalCrossEntropy instance";
            validateCategoricalCommon(loss_name, predictions, loss_data_type, reported_loss_shape);
            if (labels.getDimensions().size() != 1 || labels.getDimensions()[0] <= 1) {
                string error_message = loss_name + ": labels must be a 1 dimensional dense class vector but labels is " +
                                       labels.getDescriptorString();
                throw nb::value_error(error_message.c_str());
            }
            if (predictions.getDimensions()[0] != labels.getDimensions()[0]) {
                string error_message = loss_name + ": mismatch between predictions size " + to_string(predictions.getDimensions()[0]) +
                                       " and labels tensor size " + to_string(labels.getDimensions()[0]);
                throw nb::value_error(error_message.c_str());
            }

            CategoricalCrossEntropy::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).lossDataType(loss_data_type)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            CategoricalCrossEntropy built = builder.build();

            new (self) CategoricalCrossEntropy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a = DataType::FP32,
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a dense/soft-label categorical cross-entropy loss.)nbdoc");

    categorical_cross_entropy.attr("__doc__") = R"nbdoc(
Dense categorical cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
    One-dimensional logits tensor with one element per class.
labels : thor.Tensor
    One-dimensional dense class target tensor. One-hot labels and soft labels are both supported.
loss_data_type : thor.DataType, default thor.DataType.FP32
reported_loss_shape : thor.losses.LossShape, default batch
    This setting does not affect training; it only controls the reported loss tensor shape.

Notes
-----
A softmax is applied internally to convert logits into probabilities:

    p_c = exp(z_c) / \sum_{j=1}^{C} exp(z_j)

The per-example dense categorical cross-entropy is then:

    L = -\sum_{c=1}^{C} y_c \log(p_c)

Use SparseCategoricalCrossEntropy when labels are integer class ids.
)nbdoc";

    auto sparse_categorical_cross_entropy =
        nb::class_<SparseCategoricalCrossEntropy, CategoricalCrossEntropy>(losses, "SparseCategoricalCrossEntropy");
    sparse_categorical_cross_entropy.attr("__module__") = "thor.losses";

    sparse_categorical_cross_entropy.def(
        "__init__",
        [](SparseCategoricalCrossEntropy *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           int32_t num_classes,
           DataType loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "SparseCategoricalCrossEntropy instance";
            validateCategoricalCommon(loss_name, predictions, loss_data_type, reported_loss_shape);
            if (num_classes <= 1) {
                string error_message = loss_name + ": num_classes must be greater than one. You passed num_classes == " +
                                       to_string(num_classes);
                throw nb::value_error(error_message.c_str());
            }
            if (predictions.getDimensions()[0] != uint64_t(num_classes)) {
                string error_message = loss_name + ": mismatch between num_classes " + to_string(num_classes) +
                                       " and predictions tensor size " + to_string(predictions.getDimensions()[0]) +
                                       ". Either set num_classes to match or fix your predictions tensor.";
                throw nb::value_error(error_message.c_str());
            }
            if (labels.getDimensions().size() != 1 || labels.getDimensions()[0] != 1) {
                string error_message = loss_name + ": labels must be a 1 dimensional tensor of size 1 but labels is " +
                                       labels.getDescriptorString();
                throw nb::value_error(error_message.c_str());
            }
            DataType labelsDataType = labels.getDataType();
            if (labelsDataType != DataType::UINT8 && labelsDataType != DataType::UINT16 && labelsDataType != DataType::UINT32) {
                string error_message = loss_name + ": labels must use uint8, uint16, or uint32 dtype for sparse class ids";
                throw nb::value_error(error_message.c_str());
            }

            SparseCategoricalCrossEntropy::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .numClasses(uint32_t(num_classes))
                .lossDataType(loss_data_type)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            SparseCategoricalCrossEntropy built = builder.build();

            new (self) SparseCategoricalCrossEntropy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "num_classes"_a,
        "loss_data_type"_a = DataType::FP32,
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a sparse categorical cross-entropy loss.)nbdoc");

    sparse_categorical_cross_entropy.attr("__doc__") = R"nbdoc(
Sparse categorical cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
    One-dimensional logits tensor with one element per class.
labels : thor.Tensor
    One-dimensional tensor of size 1 containing the true class id for each batch item.
num_classes : int
    Number of classes in predictions.
loss_data_type : thor.DataType, default thor.DataType.FP32
reported_loss_shape : thor.losses.LossShape, default batch
    This setting does not affect training; it only controls the reported loss tensor shape.

Notes
-----
Sparse categorical cross-entropy applies softmax internally and computes:

    L = -\log(p_true)

The logits gradient is dense and equivalent to p - one_hot(class_id).
)nbdoc";
}
