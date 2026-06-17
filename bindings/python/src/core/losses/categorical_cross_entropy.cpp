#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <limits>

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


bool sparseLabelsMatchPredictionPrefix(Tensor predictions, Tensor labels) {
    const std::vector<uint64_t> predictionDims = predictions.getDimensions();
    const std::vector<uint64_t> labelDims = labels.getDimensions();
    if (predictionDims.empty())
        return false;
    const size_t prefixRank = predictionDims.size() - 1;
    if (prefixRank == 0) {
        return labelDims.size() == 1 && labelDims[0] == 1;
    }
    if (labelDims.size() == prefixRank) {
        for (size_t i = 0; i < prefixRank; ++i) {
            if (labelDims[i] != predictionDims[i])
                return false;
        }
        return true;
    }
    if (labelDims.size() == prefixRank + 1 && labelDims.back() == 1) {
        for (size_t i = 0; i < prefixRank; ++i) {
            if (labelDims[i] != predictionDims[i])
                return false;
        }
        return true;
    }
    return false;
}

std::string dimsToString(const std::vector<uint64_t>& dims) {
    std::string result = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i != 0)
            result += ", ";
        result += std::to_string(dims[i]);
    }
    result += "]";
    return result;
}

void validateCategoricalCommon(const string &loss_name, Tensor predictions, DataType loss_data_type, LossShape reported_loss_shape) {
    if (predictions.getDimensions().empty() || predictions.getDimensions().back() <= 1) {
        string error_message = loss_name + ": predictions must have at least one dimension and a final class dimension greater than one but predictions is " +
                               predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (loss_data_type != DataType::FP16 && loss_data_type != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
    validateReportedLossShape(reported_loss_shape, loss_name);
}

void validateSparseMask(const string &loss_name, Tensor predictions, Tensor mask) {
    if (!sparseLabelsMatchPredictionPrefix(predictions, mask)) {
        const std::vector<uint64_t> predictionDims = predictions.getDimensions();
        const std::vector<uint64_t> predictionPrefix(predictionDims.begin(), predictionDims.end() - 1);
        string error_message = loss_name + ": mask dimensions " + dimsToString(mask.getDimensions()) +
                               " must match predictions prefix dimensions " + dimsToString(predictionPrefix) +
                               " or that prefix with a trailing singleton";
        throw nb::value_error(error_message.c_str());
    }
    DataType maskDataType = mask.getDataType();
    if (maskDataType != DataType::BOOLEAN && maskDataType != DataType::UINT8 && maskDataType != DataType::FP16 &&
        maskDataType != DataType::FP32) {
        string error_message = loss_name + ": mask must use bool, uint8, fp16, or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
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
            if (predictions.getDimensions() != labels.getDimensions()) {
                string error_message = loss_name + ": dense labels dimensions " + dimsToString(labels.getDimensions()) +
                                       " must match predictions dimensions " + dimsToString(predictions.getDimensions());
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
    Logits tensor whose final dimension is the class dimension.
labels : thor.Tensor
    Dense class target tensor with the same dimensions as predictions. One-hot labels and soft labels are both supported.
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
           std::optional<float> loss_weight,
           std::optional<int64_t> ignore_index,
           std::optional<Tensor> mask) {
            const string loss_name = "SparseCategoricalCrossEntropy instance";
            validateCategoricalCommon(loss_name, predictions, loss_data_type, reported_loss_shape);
            if (num_classes <= 1) {
                string error_message = loss_name + ": num_classes must be greater than one. You passed num_classes == " +
                                       to_string(num_classes);
                throw nb::value_error(error_message.c_str());
            }
            if (predictions.getDimensions().back() != uint64_t(num_classes)) {
                string error_message = loss_name + ": mismatch between num_classes " + to_string(num_classes) +
                                       " and predictions final class dimension " + to_string(predictions.getDimensions().back()) +
                                       ". Either set num_classes to match or fix your predictions tensor.";
                throw nb::value_error(error_message.c_str());
            }
            if (!sparseLabelsMatchPredictionPrefix(predictions, labels)) {
                const std::vector<uint64_t> predictionDims = predictions.getDimensions();
                const std::vector<uint64_t> predictionPrefix(predictionDims.begin(), predictionDims.end() - 1);
                string error_message = loss_name + ": sparse labels dimensions " + dimsToString(labels.getDimensions()) +
                                       " must match predictions prefix dimensions " + dimsToString(predictionPrefix) +
                                       " or that prefix with a trailing singleton";
                throw nb::value_error(error_message.c_str());
            }
            DataType labelsDataType = labels.getDataType();
            if (labelsDataType != DataType::UINT8 && labelsDataType != DataType::UINT16 && labelsDataType != DataType::UINT32) {
                string error_message = loss_name + ": labels must use uint8, uint16, or uint32 dtype for sparse class ids";
                throw nb::value_error(error_message.c_str());
            }

            if (ignore_index.has_value() && (ignore_index.value() < 0 || ignore_index.value() > int64_t(std::numeric_limits<uint32_t>::max()))) {
                string error_message = loss_name + ": ignore_index must be between 0 and UINT32_MAX";
                throw nb::value_error(error_message.c_str());
            }
            if (mask.has_value())
                validateSparseMask(loss_name, predictions, mask.value());

            SparseCategoricalCrossEntropy::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .numClasses(uint32_t(num_classes))
                .lossDataType(loss_data_type);
            builder.lossWeight(loss_weight.value_or(1.0f));
            if (ignore_index.has_value())
                builder.ignoreIndex(uint32_t(ignore_index.value()));
            if (mask.has_value())
                builder.mask(mask.value());
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
        "ignore_index"_a.none() = nb::none(),
        "mask"_a.none() = nb::none(),
        R"nbdoc(Construct a sparse categorical cross-entropy loss.)nbdoc");

    sparse_categorical_cross_entropy.attr("__doc__") = R"nbdoc(
Sparse categorical cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
    Logits tensor whose final dimension is the class dimension.
labels : thor.Tensor
    Sparse integer class ids. Dimensions must match the prediction prefix dimensions, or that prefix with a trailing singleton.
num_classes : int
    Number of classes in predictions.
loss_data_type : thor.DataType, default thor.DataType.FP32
reported_loss_shape : thor.losses.LossShape, default batch
    This setting does not affect training; it only controls the reported loss tensor shape.
ignore_index : int, optional keyword-only
    Label id that contributes zero loss and zero logits gradient.
mask : thor.Tensor, optional keyword-only
    Prefix-shaped boolean/uint8/fp16/fp32 mask. Entries > 0.5 are valid; masked entries contribute zero loss and zero gradient.

Notes
-----
Sparse categorical cross-entropy is logits-native: it computes logsumexp(logits) - logits[class_id]
without materializing a separate softmax tensor or a per-class raw loss tensor. The raw loss shape is
the predictions prefix shape, e.g. predictions [B, S, V] produce raw loss [B, S].

The logits gradient is dense and equivalent to softmax(logits) - one_hot(class_id).
)nbdoc";
}
