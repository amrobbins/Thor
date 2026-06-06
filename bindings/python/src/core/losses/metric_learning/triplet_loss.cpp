#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/TripletLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

namespace {
void validateReportedLossShape(LossShape reported_loss_shape, const string& loss_name) {
    if (reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::CLASSWISE &&
        reported_loss_shape != LossShape::ELEMENTWISE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(TripletLoss::Builder& builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::CLASSWISE) {
        builder.reportsPerOutputLoss();
    } else if (reported_loss_shape == LossShape::ELEMENTWISE) {
        builder.reportsElementwiseLoss();
    } else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

void validateTripletLossArguments(const string& loss_name,
                                  Tensor anchor,
                                  Tensor positive,
                                  Tensor negative,
                                  float margin,
                                  float eps,
                                  optional<DataType> loss_data_type,
                                  LossShape reported_loss_shape) {
    if (anchor.getDimensions().size() != 1 || anchor.getDimensions()[0] == 0) {
        string error_message = loss_name + ": anchor must be a 1 dimensional embedding tensor but anchor is " + anchor.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (positive.getDimensions() != anchor.getDimensions()) {
        string error_message = loss_name + ": positive dimensions " + positive.getDescriptorString() +
                               " must match anchor dimensions " + anchor.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (negative.getDimensions() != anchor.getDimensions()) {
        string error_message = loss_name + ": negative dimensions " + negative.getDescriptorString() +
                               " must match anchor dimensions " + anchor.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (anchor == positive || anchor == negative || positive == negative) {
        string error_message = loss_name + ": anchor, positive, and negative must be distinct tensors";
        throw nb::value_error(error_message.c_str());
    }
    if (anchor.getDataType() != DataType::FP16 && anchor.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": anchor must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (positive.getDataType() != anchor.getDataType() || negative.getDataType() != anchor.getDataType()) {
        string error_message = loss_name + ": anchor, positive, and negative must use the same fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (margin <= 0.0f) {
        string error_message = loss_name + ": margin must be greater than zero";
        throw nb::value_error(error_message.c_str());
    }
    if (eps <= 0.0f) {
        string error_message = loss_name + ": eps must be greater than zero";
        throw nb::value_error(error_message.c_str());
    }
    DataType effectiveLossDataType = loss_data_type.value_or(anchor.getDataType());
    if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_triplet_loss(nb::module_& losses) {
    auto triplet_loss = nb::class_<TripletLoss, Loss>(losses, "TripletLoss");
    triplet_loss.attr("__module__") = "thor.losses.metric_learning";

    triplet_loss.def(
        "__init__",
        [](TripletLoss* self,
           Network& network,
           Tensor anchor,
           Tensor positive,
           Tensor negative,
           float margin,
           float eps,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "TripletLoss instance";
            validateTripletLossArguments(loss_name, anchor, positive, negative, margin, eps, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(anchor.getDataType());
            TripletLoss::Builder builder;
            builder.network(network)
                .anchor(anchor)
                .positive(positive)
                .negative(negative)
                .margin(margin)
                .eps(eps)
                .lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            TripletLoss built = builder.build();

            new (self) TripletLoss(std::move(built));
        },
        "network"_a,
        "anchor"_a,
        "positive"_a,
        "negative"_a,
        "margin"_a = 1.0f,
        "eps"_a = 1.0e-6f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a Triplet margin loss over anchor, positive, and negative embeddings.)nbdoc");

    triplet_loss.def_prop_ro("margin", &TripletLoss::getMargin);
    triplet_loss.def_prop_ro("eps", &TripletLoss::getEps);
    triplet_loss.def("get_anchor", &TripletLoss::getAnchor);
    triplet_loss.def("get_positive", &TripletLoss::getPositive);
    triplet_loss.def("get_negative", &TripletLoss::getNegative);

    triplet_loss.attr("__doc__") = R"nbdoc(
Triplet margin loss over anchor, positive, and negative embedding tensors.

The raw loss per example is:

    max(||anchor - positive||_2 - ||anchor - negative||_2 + margin, 0)

All three inputs are differentiable. This loss expects already-formed triplets; triplet
mining is intentionally separate from the loss.
)nbdoc";
}
