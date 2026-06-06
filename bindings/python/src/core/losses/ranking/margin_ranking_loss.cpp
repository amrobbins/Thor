#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/MarginRankingLoss.h"
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

void setReportedLossShape(MarginRankingLoss::Builder& builder, LossShape reported_loss_shape) {
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

bool isTargetDType(DataType dtype) {
    return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 || dtype == DataType::INT64 ||
           dtype == DataType::FP16 || dtype == DataType::FP32;
}

void validateMarginRankingLossArguments(const string& loss_name,
                                        Tensor input1,
                                        Tensor input2,
                                        Tensor target,
                                        float margin,
                                        optional<DataType> loss_data_type,
                                        LossShape reported_loss_shape) {
    if (input1.getDimensions().empty()) {
        string error_message = loss_name + ": input1 must have at least one dimension but input1 is " + input1.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (input2.getDimensions() != input1.getDimensions()) {
        string error_message = loss_name + ": input2 dimensions " + input2.getDescriptorString() +
                               " must match input1 dimensions " + input1.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (target.getDimensions() != input1.getDimensions()) {
        string error_message = loss_name + ": target dimensions " + target.getDescriptorString() +
                               " must match input1 dimensions " + input1.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (input1 == input2 || input1 == target || input2 == target) {
        string error_message = loss_name + ": input1, input2, and target must be distinct tensors";
        throw nb::value_error(error_message.c_str());
    }
    if (input1.getDataType() != DataType::FP16 && input1.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": input1 must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (input2.getDataType() != input1.getDataType()) {
        string error_message = loss_name + ": input1 and input2 must use the same fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (!isTargetDType(target.getDataType())) {
        string error_message = loss_name + ": target must use int8, int16, int32, int64, fp16, or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (margin < 0.0f) {
        string error_message = loss_name + ": margin must be non-negative";
        throw nb::value_error(error_message.c_str());
    }
    DataType effectiveLossDataType = loss_data_type.value_or(input1.getDataType());
    if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_margin_ranking_loss(nb::module_& losses) {
    auto margin_ranking_loss = nb::class_<MarginRankingLoss, Loss>(losses, "MarginRankingLoss");
    margin_ranking_loss.attr("__module__") = "thor.losses.ranking";

    margin_ranking_loss.def(
        "__init__",
        [](MarginRankingLoss* self,
           Network& network,
           Tensor input1,
           Tensor input2,
           Tensor target,
           float margin,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "MarginRankingLoss instance";
            validateMarginRankingLossArguments(loss_name, input1, input2, target, margin, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(input1.getDataType());
            MarginRankingLoss::Builder builder;
            builder.network(network).input1(input1).input2(input2).target(target).margin(margin).lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            MarginRankingLoss built = builder.build();

            new (self) MarginRankingLoss(std::move(built));
        },
        "network"_a,
        "input1"_a,
        "input2"_a,
        "target"_a,
        "margin"_a = 0.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a MarginRankingLoss over two score tensors and a target tensor.)nbdoc");

    margin_ranking_loss.def_prop_ro("margin", &MarginRankingLoss::getMargin);
    margin_ranking_loss.def("get_input1", &MarginRankingLoss::getInput1);
    margin_ranking_loss.def("get_input2", &MarginRankingLoss::getInput2);
    margin_ranking_loss.def("get_target", &MarginRankingLoss::getTarget);

    margin_ranking_loss.attr("__doc__") = R"nbdoc(
Margin ranking loss over two score tensors and one target tensor.

The raw elementwise loss is:

    max(margin - target * (input1 - input2), 0)

For target = 1, this encourages input1 to rank above input2 by at least margin.
For target = -1, this encourages input2 to rank above input1 by at least margin.

Gradients are produced for input1 and input2. The target tensor is an auxiliary,
non-differentiable input.
)nbdoc";
}
