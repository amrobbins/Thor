#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/CosineEmbeddingLoss.h"
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

void setReportedLossShape(CosineEmbeddingLoss::Builder& builder, LossShape reported_loss_shape) {
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

void validateCosineEmbeddingLossArguments(const string& loss_name,
                                          Tensor input1,
                                          Tensor input2,
                                          Tensor target,
                                          float margin,
                                          float eps,
                                          optional<DataType> loss_data_type,
                                          LossShape reported_loss_shape) {
    if (input1.getDimensions().size() != 1 || input1.getDimensions()[0] == 0) {
        string error_message = loss_name + ": input1 must be a 1 dimensional embedding tensor but input1 is " + input1.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (input2.getDimensions() != input1.getDimensions()) {
        string error_message = loss_name + ": input2 dimensions " + input2.getDescriptorString() +
                               " must match input1 dimensions " + input1.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (target.getDimensions().size() != 1 || target.getDimensions()[0] != 1) {
        string error_message = loss_name + ": target must be a 1 dimensional tensor with exactly one label per example but target is " +
                               target.getDescriptorString();
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
    if (margin < -1.0f || margin > 1.0f) {
        string error_message = loss_name + ": margin must be between -1 and 1";
        throw nb::value_error(error_message.c_str());
    }
    if (eps <= 0.0f) {
        string error_message = loss_name + ": eps must be greater than zero";
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

void bind_cosine_embedding_loss(nb::module_& losses) {
    auto cosine_embedding_loss = nb::class_<CosineEmbeddingLoss, Loss>(losses, "CosineEmbeddingLoss");
    cosine_embedding_loss.attr("__module__") = "thor.losses.metric_learning";

    cosine_embedding_loss.def(
        "__init__",
        [](CosineEmbeddingLoss* self,
           Network& network,
           Tensor input1,
           Tensor input2,
           Tensor target,
           float margin,
           float eps,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "CosineEmbeddingLoss instance";
            validateCosineEmbeddingLossArguments(loss_name, input1, input2, target, margin, eps, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(input1.getDataType());
            CosineEmbeddingLoss::Builder builder;
            builder.network(network)
                .input1(input1)
                .input2(input2)
                .target(target)
                .margin(margin)
                .eps(eps)
                .lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            CosineEmbeddingLoss built = builder.build();

            new (self) CosineEmbeddingLoss(std::move(built));
        },
        "network"_a,
        "input1"_a,
        "input2"_a,
        "target"_a,
        "margin"_a = 0.0f,
        "eps"_a = 1.0e-8f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a CosineEmbeddingLoss over two embedding tensors and a target label tensor.)nbdoc");

    cosine_embedding_loss.def_prop_ro("margin", &CosineEmbeddingLoss::getMargin);
    cosine_embedding_loss.def_prop_ro("eps", &CosineEmbeddingLoss::getEps);
    cosine_embedding_loss.def("get_input1", &CosineEmbeddingLoss::getInput1);
    cosine_embedding_loss.def("get_input2", &CosineEmbeddingLoss::getInput2);
    cosine_embedding_loss.def("get_target", &CosineEmbeddingLoss::getTarget);

    cosine_embedding_loss.attr("__doc__") = R"nbdoc(
Cosine embedding loss over two embedding tensors and one target label per example.

For target > 0, the raw loss is:

    1 - cosine(input1, input2)

For target <= 0, the raw loss is:

    max(cosine(input1, input2) - margin, 0)

Gradients are produced for input1 and input2. The target tensor is an auxiliary,
non-differentiable input.
)nbdoc";
}
