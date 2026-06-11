#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <optional>
#include <string>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/BoxIouLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

namespace {

void validateReportedLossShape(LossShape reportedLossShape, const string& lossName) {
    if (reportedLossShape != LossShape::BATCH && reportedLossShape != LossShape::CLASSWISE &&
        reportedLossShape != LossShape::ELEMENTWISE && reportedLossShape != LossShape::RAW) {
        string errorMessage =
            "Invalid value " + to_string((int)reportedLossShape) + " passed for enum reported_loss_shape to " + lossName + ".";
        throw nb::value_error(errorMessage.c_str());
    }
}

void validateBoxTensor(const string& lossName, const string& tensorName, Tensor tensor) {
    const vector<uint64_t>& dims = tensor.getDimensions();
    if (!((dims.size() == 1 && dims[0] == 4) || (dims.size() == 2 && dims[1] == 4))) {
        string errorMessage = lossName + ": " + tensorName + " must have dimensions [4] or [boxes, 4] but got " +
                              tensor.getDescriptorString();
        throw nb::value_error(errorMessage.c_str());
    }
    if (tensor.getDataType() != DataType::FP16 && tensor.getDataType() != DataType::FP32) {
        string errorMessage = lossName + ": " + tensorName + " must use fp16 or fp32 dtype";
        throw nb::value_error(errorMessage.c_str());
    }
}

void validateBoxIouLossArguments(const string& lossName,
                                 Tensor predictions,
                                 Tensor labels,
                                 const string& boxFormat,
                                 float eps,
                                 optional<DataType> lossDataType,
                                 LossShape reportedLossShape) {
    validateBoxTensor(lossName, "predictions", predictions);
    validateBoxTensor(lossName, "labels", labels);
    if (predictions.getDimensions() != labels.getDimensions()) {
        string errorMessage = lossName + ": labels dimensions " + labels.getDescriptorString() +
                              " must match predictions dimensions " + predictions.getDescriptorString();
        throw nb::value_error(errorMessage.c_str());
    }
    if (predictions == labels) {
        string errorMessage = lossName + ": predictions and labels must be distinct tensors";
        throw nb::value_error(errorMessage.c_str());
    }
    if (boxFormat != "xyxy") {
        string errorMessage = lossName + ": box_format must be 'xyxy'";
        throw nb::value_error(errorMessage.c_str());
    }
    if (eps <= 0.0f) {
        string errorMessage = lossName + ": eps must be greater than zero";
        throw nb::value_error(errorMessage.c_str());
    }
    DataType effectiveLossDataType = lossDataType.value_or(predictions.getDataType());
    if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
        string errorMessage = lossName + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(errorMessage.c_str());
    }
    validateReportedLossShape(reportedLossShape, lossName);
}

template <typename BuilderT>
void setReportedLossShape(BuilderT& builder, LossShape reportedLossShape) {
    if (reportedLossShape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reportedLossShape == LossShape::CLASSWISE) {
        builder.reportsPerOutputLoss();
    } else if (reportedLossShape == LossShape::ELEMENTWISE) {
        builder.reportsElementwiseLoss();
    } else {
        THOR_THROW_IF_FALSE(reportedLossShape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

template <typename LossT>
void bindBoxIouLoss(nb::module_& losses, const char* className, const char* docLine) {
    auto cls = nb::class_<LossT, Loss>(losses, className);
    cls.attr("__module__") = "thor.losses.detection";

    cls.def(
        "__init__",
        [className](LossT* self,
                    Network& network,
                    Tensor predictions,
                    Tensor labels,
                    const string& boxFormat,
                    float eps,
                    optional<DataType> lossDataType,
                    LossShape reportedLossShape,
                    std::optional<float> loss_weight) {
            const string lossName = string(className) + " instance";
            validateBoxIouLossArguments(lossName, predictions, labels, boxFormat, eps, lossDataType, reportedLossShape);

            DataType effectiveLossDataType = lossDataType.value_or(predictions.getDataType());
            typename LossT::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).eps(eps).lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reportedLossShape);
            LossT built = builder.build();

            new (self) LossT(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "box_format"_a = "xyxy",
        "eps"_a = 1.0e-7f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        docLine);

    cls.def_prop_ro("eps", &LossT::getEps);
    cls.def_prop_ro("box_format", &LossT::getBoxFormat);
}

}  // namespace

void bind_box_iou_losses(nb::module_& losses) {
    bindBoxIouLoss<IoULoss>(losses, "IoULoss", "Construct an IoU box regression loss for xyxy boxes.");
    bindBoxIouLoss<GIoULoss>(losses, "GIoULoss", "Construct a Generalized IoU box regression loss for xyxy boxes.");
    bindBoxIouLoss<DIoULoss>(losses, "DIoULoss", "Construct a Distance IoU box regression loss for xyxy boxes.");
    bindBoxIouLoss<CIoULoss>(losses, "CIoULoss", "Construct a Complete IoU box regression loss for xyxy boxes.");
}
