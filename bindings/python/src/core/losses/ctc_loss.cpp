#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/CtcLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Loss/CtcLoss.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;
using CtcLossOobGradientMode = ThorImplementation::CtcLossOobGradientMode;

namespace {

string dimsToString(const vector<uint64_t>& dims) {
    string result = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i != 0)
            result += ", ";
        result += to_string(dims[i]);
    }
    result += "]";
    return result;
}

void validateReportedLossShape(LossShape reportedLossShape, const string& lossName) {
    if (reportedLossShape != LossShape::BATCH && reportedLossShape != LossShape::ELEMENTWISE && reportedLossShape != LossShape::RAW) {
        string errorMessage = lossName + ": reported_loss_shape must be batch, elementwise, or raw for CTCLoss";
        throw nb::value_error(errorMessage.c_str());
    }
}

void validateLengthTensor(const string& lossName, const string& tensorName, Tensor tensor) {
    const vector<uint64_t>& dims = tensor.getDimensions();
    if (dims != vector<uint64_t>{1}) {
        string errorMessage = lossName + ": " + tensorName + " must have dimensions [1] but got " + dimsToString(dims);
        throw nb::value_error(errorMessage.c_str());
    }
    if (tensor.getDataType() != DataType::INT32) {
        string errorMessage = lossName + ": " + tensorName + " must use int32 dtype";
        throw nb::value_error(errorMessage.c_str());
    }
}

void validateCtcLossArguments(const string& lossName,
                              Tensor logits,
                              Tensor labels,
                              Tensor labelLengths,
                              Tensor inputLengths,
                              optional<DataType> lossDataType,
                              LossShape reportedLossShape) {
    const vector<uint64_t>& logitsDims = logits.getDimensions();
    if (logitsDims.size() != 2 || logitsDims[0] == 0 || logitsDims[1] <= 1) {
        string errorMessage = lossName + ": logits must have dimensions [time, classes] with time > 0 and classes > 1 but got " +
                              dimsToString(logitsDims);
        throw nb::value_error(errorMessage.c_str());
    }
    if (logits.getDataType() != DataType::FP32) {
        string errorMessage = lossName + ": logits must use fp32 dtype";
        throw nb::value_error(errorMessage.c_str());
    }

    const vector<uint64_t>& labelDims = labels.getDimensions();
    if (labelDims.size() != 1 || labelDims[0] == 0 || labelDims[0] >= 256) {
        string errorMessage = lossName + ": labels must have dimensions [max_label_length] with 0 < max_label_length < 256 but got " +
                              dimsToString(labelDims);
        throw nb::value_error(errorMessage.c_str());
    }
    if (labels.getDataType() != DataType::INT32) {
        string errorMessage = lossName + ": labels must use int32 dtype";
        throw nb::value_error(errorMessage.c_str());
    }
    if (logitsDims[0] < labelDims[0]) {
        string errorMessage = lossName + ": max_label_length " + to_string(labelDims[0]) +
                              " must be less than or equal to logits time dimension " + to_string(logitsDims[0]);
        throw nb::value_error(errorMessage.c_str());
    }

    validateLengthTensor(lossName, "label_lengths", labelLengths);
    validateLengthTensor(lossName, "input_lengths", inputLengths);

    if (logits == labels || logits == labelLengths || logits == inputLengths || labels == labelLengths || labels == inputLengths ||
        labelLengths == inputLengths) {
        string errorMessage = lossName + ": logits, labels, label_lengths, and input_lengths must be distinct tensors";
        throw nb::value_error(errorMessage.c_str());
    }

    const DataType effectiveLossDataType = lossDataType.value_or(DataType::FP32);
    if (effectiveLossDataType != DataType::FP32) {
        string errorMessage = lossName + ": loss_data_type must be fp32";
        throw nb::value_error(errorMessage.c_str());
    }
    validateReportedLossShape(reportedLossShape, lossName);
}

void setReportedLossShape(CtcLoss::Builder& builder, LossShape reportedLossShape) {
    if (reportedLossShape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reportedLossShape == LossShape::ELEMENTWISE) {
        builder.reportsElementwiseLoss();
    } else {
        THOR_THROW_IF_FALSE(reportedLossShape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

}  // namespace

void bind_ctc_loss(nb::module_& losses) {
    auto oobMode = nb::enum_<CtcLossOobGradientMode>(losses, "CTCOobGradientMode")
                       .value("zero", CtcLossOobGradientMode::ZERO)
                       .value("skip", CtcLossOobGradientMode::SKIP);
    oobMode.attr("__module__") = "thor.losses";

    auto ctcLoss = nb::class_<CtcLoss, Loss>(losses, "CTCLoss");
    ctcLoss.attr("__module__") = "thor.losses";

    ctcLoss.def(
        "__init__",
        [](CtcLoss* self,
           Network& network,
           Tensor logits,
           Tensor labels,
           Tensor labelLengths,
           Tensor inputLengths,
           optional<DataType> lossDataType,
           LossShape reportedLossShape,
           CtcLossOobGradientMode oobGradientMode,
           optional<float> lossWeight) {
            const string lossName = "CTCLoss instance";
            validateCtcLossArguments(lossName, logits, labels, labelLengths, inputLengths, lossDataType, reportedLossShape);

            CtcLoss::Builder builder;
            builder.network(network)
                .logits(logits)
                .labels(labels)
                .labelLengths(labelLengths)
                .inputLengths(inputLengths)
                .lossDataType(lossDataType.value_or(DataType::FP32))
                .lossWeight(lossWeight.value_or(1.0f));
            setReportedLossShape(builder, reportedLossShape);
            if (oobGradientMode == CtcLossOobGradientMode::ZERO)
                builder.zeroOutOfBoundsGradients();
            else
                builder.skipOutOfBoundsGradients();

            CtcLoss built = builder.build();
            new (self) CtcLoss(std::move(built));
        },
        "network"_a,
        "logits"_a,
        "labels"_a,
        "label_lengths"_a,
        "input_lengths"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        "oob_gradient_mode"_a = CtcLossOobGradientMode::ZERO,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a cuDNN-backed CTC loss.)nbdoc");

    ctcLoss.def("get_label_lengths", &CtcLoss::getLabelLengths);
    ctcLoss.def("get_input_lengths", &CtcLoss::getInputLengths);
    ctcLoss.def_prop_ro("max_label_length", &CtcLoss::getMaxLabelLength);
    ctcLoss.def_prop_ro("oob_gradient_mode", &CtcLoss::getOobGradientMode);

    ctcLoss.attr("__doc__") = R"nbdoc(
cuDNN-backed CTC loss.

Parameters
----------
network : thor.Network
logits : thor.Tensor
    FP32 tensor with API dimensions ``[time, classes]``. The batch dimension is
    supplied by Thor placement, so the physical tensor is ``[batch, time, classes]``.
labels : thor.Tensor
    INT32 padded per-sample target tensor with API dimensions ``[max_label_length]``.
    At runtime, only the prefix ``labels[b, :label_lengths[b]]`` is read; padding
    values after the valid prefix are ignored.
label_lengths : thor.Tensor
    INT32 tensor with API dimensions ``[1]``. Physical shape is ``[batch, 1]``.
input_lengths : thor.Tensor
    INT32 tensor with API dimensions ``[1]``. Physical shape is ``[batch, 1]``.
loss_data_type : thor.DataType | None, default thor.DataType.fp32
    CTCLoss v1 supports fp32 only.
reported_loss_shape : thor.losses.LossShape, default batch
    CTCLoss supports batch, elementwise, and raw reporting. Classwise reporting is rejected.
oob_gradient_mode : thor.losses.CTCOobGradientMode, default zero
    Controls cuDNN out-of-bounds gradient handling for impossible samples.
loss_weight : float | None, keyword-only, default None

Notes
-----
This is intentionally a cuDNN-only CTC wrapper. There is no native or CPU fallback.
cuDNN applies softmax normalization internally, so pass unnormalized logits/activations.
The blank label follows cuDNN's fixed blank-index convention, class 0.
)nbdoc";
}
