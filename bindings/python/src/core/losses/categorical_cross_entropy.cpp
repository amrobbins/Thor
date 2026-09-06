#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <limits>

#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

namespace {
void validateReportedLossShape(LossShape reported_loss_shape, const string &loss_name) {
    if (reported_loss_shape != LossShape::NONE && reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::PER_OUTPUT &&
        reported_loss_shape != LossShape::PER_EXAMPLE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

template <typename BuilderT>
void setReportedLossShape(BuilderT &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::NONE) {
        builder.reportsNoLoss();
    } else if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::PER_OUTPUT) {
        builder.reportsPerOutputLoss();
    } else if (reported_loss_shape == LossShape::PER_EXAMPLE) {
        builder.reportsPerExampleLoss();
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

void validateRaggedCategorical(const string& loss_name,
                               const RaggedTensor& predictions,
                               const RaggedTensor& labels,
                               DataType loss_data_type,
                               LossShape reported_loss_shape) {
    if (predictions.getTrailingDimensions().size() != 1 || predictions.getTrailingDimensions().back() <= 1)
        throw nb::value_error("CategoricalCrossEntropy instance: ragged predictions must have exactly one trailing class dimension greater than one.");
    if (!predictions.sharesPartitionWith(labels))
        throw nb::value_error("CategoricalCrossEntropy instance: ragged predictions and labels must use the exact same row partition.");
    if (predictions.getBatchSize() != labels.getBatchSize() ||
        predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
        predictions.getTrailingDimensions() != labels.getTrailingDimensions())
        throw nb::value_error("CategoricalCrossEntropy instance: ragged predictions and labels must have identical value geometry.");
    if (predictions.getValuesDataType() != DataType::FP16 && predictions.getValuesDataType() != DataType::FP32)
        throw nb::value_error("CategoricalCrossEntropy instance: ragged predictions must use fp16 or fp32 dtype.");
    if (labels.getValuesDataType() != DataType::FP16 && labels.getValuesDataType() != DataType::FP32)
        throw nb::value_error("CategoricalCrossEntropy instance: ragged labels must use fp16 or fp32 dtype.");
    if (loss_data_type != DataType::FP16 && loss_data_type != DataType::FP32)
        throw nb::value_error("CategoricalCrossEntropy instance: loss_data_type must be fp16 or fp32.");
    if (reported_loss_shape == LossShape::PER_OUTPUT)
        throw nb::value_error("CategoricalCrossEntropy instance: per_output reporting is undefined for ragged predictions.");
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

bool raggedSparseValueIsScalar(const RaggedTensor& tensor) {
    const std::vector<uint64_t> trailing = tensor.getTrailingDimensions();
    return trailing.empty() || (trailing.size() == 1 && trailing.front() == 1);
}

void validateRaggedSparseCategorical(const string& loss_name,
                                     const RaggedTensor& predictions,
                                     const RaggedTensor& labels,
                                     int32_t num_classes,
                                     DataType loss_data_type,
                                     LossShape reported_loss_shape) {
    const std::vector<uint64_t> predictionTrailing = predictions.getTrailingDimensions();
    if (predictionTrailing.size() != 1 || predictionTrailing.back() <= 1)
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: ragged predictions must have exactly one trailing class dimension greater than one.");
    if (predictions.getValuesDataType() != DataType::FP16 && predictions.getValuesDataType() != DataType::FP32)
        throw nb::value_error("SparseCategoricalCrossEntropy instance: ragged predictions must use fp16 or fp32 dtype.");
    if (!raggedSparseValueIsScalar(labels))
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: ragged labels must be scalar per active token (trailing shape [] or [1]).");
    if (labels.getValuesDataType() != DataType::UINT8 && labels.getValuesDataType() != DataType::UINT16 &&
        labels.getValuesDataType() != DataType::UINT32)
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: labels must use uint8, uint16, or uint32 dtype for sparse class ids");
    if (!predictions.sharesPartitionWith(labels))
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: ragged predictions and labels must use the exact same row partition.");
    if (predictions.getBatchSize() != labels.getBatchSize() ||
        predictions.getMaxTotalValues() != labels.getMaxTotalValues())
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: ragged predictions and labels must have the same batch size and packed capacity.");
    if (num_classes <= 1) {
        string error_message = loss_name + ": num_classes must be greater than one. You passed num_classes == " +
                               to_string(num_classes);
        throw nb::value_error(error_message.c_str());
    }
    if (predictionTrailing.back() != uint64_t(num_classes)) {
        string error_message = loss_name + ": mismatch between num_classes " + to_string(num_classes) +
                               " and predictions trailing class dimension " + to_string(predictionTrailing.back()) +
                               ". Either set num_classes to match or fix your predictions tensor.";
        throw nb::value_error(error_message.c_str());
    }
    if (loss_data_type != DataType::FP16 && loss_data_type != DataType::FP32)
        throw nb::value_error("SparseCategoricalCrossEntropy instance: loss_data_type must be fp16 or fp32");
    if (reported_loss_shape == LossShape::PER_OUTPUT)
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: per_output reporting is undefined for ragged predictions.");
    validateReportedLossShape(reported_loss_shape, loss_name);
}

void validateRaggedSparseMask(const RaggedTensor& predictions, const RaggedTensor& mask) {
    if (!raggedSparseValueIsScalar(mask))
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: ragged mask must be scalar per active token (trailing shape [] or [1]).");
    if (!predictions.sharesPartitionWith(mask))
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: ragged predictions and mask must use the exact same row partition.");
    if (predictions.getBatchSize() != mask.getBatchSize() ||
        predictions.getMaxTotalValues() != mask.getMaxTotalValues())
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: ragged predictions and mask must have the same batch size and packed capacity.");
    const DataType maskDataType = mask.getValuesDataType();
    if (maskDataType != DataType::BOOLEAN && maskDataType != DataType::UINT8 && maskDataType != DataType::FP16 &&
        maskDataType != DataType::FP32)
        throw nb::value_error(
            "SparseCategoricalCrossEntropy instance: mask must use bool, uint8, fp16, or fp32 dtype");
}
}  // namespace

void bind_categorical_cross_entropy(nb::module_ &losses) {
    auto categorical_cross_entropy = nb::class_<CategoricalCrossEntropy, Loss>(losses, "CategoricalCrossEntropy");
    categorical_cross_entropy.attr("__module__") = "thor.losses";

    categorical_cross_entropy.def(
        "__init__",
        [](CategoricalCrossEntropy *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           DataType loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "CategoricalCrossEntropy instance";
            CategoricalCrossEntropy::Builder builder;
            builder.network(network).lossDataType(loss_data_type).lossWeight(loss_weight.value_or(1.0f));

            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                validateCategoricalCommon(loss_name, predictions, loss_data_type, reported_loss_shape);
                if (predictions.getDimensions() != labels.getDimensions()) {
                    string error_message = loss_name + ": dense labels dimensions " + dimsToString(labels.getDimensions()) +
                                           " must match predictions dimensions " + dimsToString(predictions.getDimensions());
                    throw nb::value_error(error_message.c_str());
                }
                builder.predictions(predictions).labels(labels);
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                validateRaggedCategorical(loss_name, predictions, labels, loss_data_type, reported_loss_shape);
                builder.predictions(predictions).labels(labels);
            } else {
                throw nb::type_error("CategoricalCrossEntropy predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }

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
        R"nbdoc(Construct a dense or rank-1 ragged dense-target categorical cross-entropy loss.)nbdoc");

    categorical_cross_entropy.def("get_predictions", [](const CategoricalCrossEntropy& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.getPredictions());
    });
    categorical_cross_entropy.def("get_labels", [](const CategoricalCrossEntropy& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    categorical_cross_entropy.def("get_raw_loss", [](const CategoricalCrossEntropy& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    categorical_cross_entropy.def("get_loss", [](const CategoricalCrossEntropy& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    categorical_cross_entropy.def_prop_ro("is_ragged", &CategoricalCrossEntropy::isRagged);

    categorical_cross_entropy.attr("__doc__") = R"nbdoc(
Dense-target categorical cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor or thor.RaggedTensor
    Logits whose final/trailing dimension is the class dimension. Ragged inputs must have exactly one trailing class dimension.
labels : thor.Tensor or thor.RaggedTensor
    Dense class targets matching predictions. Ragged labels must share the exact same row partition.
loss_data_type : thor.DataType, default thor.DataType.FP32
reported_loss_shape : thor.losses.LossShape, default batch
    This setting does not affect training; it only controls the reported loss tensor shape.

Notes
-----
A softmax is applied internally to convert logits into probabilities:

    p_c = exp(z_c) / \sum_{j=1}^{C} exp(z_j)

The per-example dense categorical cross-entropy is then:

    L = -\sum_{c=1}^{C} y_c \log(p_c)

For ragged inputs, ``raw`` preserves the partition, ``per_example`` sums over all active tokens/classes in each logical row, and ``batch`` averages those row sums over valid logical examples. ``per_output`` is undefined for ragged input.

Use SparseCategoricalCrossEntropy when labels are integer class ids; dense and rank-1 ragged sparse targets are supported.
)nbdoc";

    auto sparse_categorical_cross_entropy =
        nb::class_<SparseCategoricalCrossEntropy, CategoricalCrossEntropy>(losses, "SparseCategoricalCrossEntropy");
    sparse_categorical_cross_entropy.attr("__module__") = "thor.losses";

    sparse_categorical_cross_entropy.def(
        "__init__",
        [](SparseCategoricalCrossEntropy *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           int32_t num_classes,
           DataType loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight,
           std::optional<int64_t> ignore_index,
           nb::object maskObject) {
            const string loss_name = "SparseCategoricalCrossEntropy instance";
            if (ignore_index.has_value() && (ignore_index.value() < 0 || ignore_index.value() > int64_t(std::numeric_limits<uint32_t>::max()))) {
                string error_message = loss_name + ": ignore_index must be between 0 and UINT32_MAX";
                throw nb::value_error(error_message.c_str());
            }

            SparseCategoricalCrossEntropy::Builder builder;
            builder.network(network);

            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
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
                builder.predictions(predictions).labels(labels);
                if (!maskObject.is_none()) {
                    if (!nb::isinstance<Tensor>(maskObject))
                        throw nb::type_error("SparseCategoricalCrossEntropy dense predictions require mask to be thor.Tensor.");
                    Tensor mask = nb::cast<Tensor>(maskObject);
                    validateSparseMask(loss_name, predictions, mask);
                    builder.mask(mask);
                }
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                validateRaggedSparseCategorical(
                    loss_name, predictions, labels, num_classes, loss_data_type, reported_loss_shape);
                builder.predictions(predictions).labels(labels);
                if (!maskObject.is_none()) {
                    if (!nb::isinstance<RaggedTensor>(maskObject))
                        throw nb::type_error("SparseCategoricalCrossEntropy ragged predictions require mask to be thor.RaggedTensor.");
                    RaggedTensor mask = nb::cast<RaggedTensor>(maskObject);
                    validateRaggedSparseMask(predictions, mask);
                    builder.mask(mask);
                }
            } else {
                throw nb::type_error(
                    "SparseCategoricalCrossEntropy predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }

            builder.numClasses(uint32_t(num_classes)).lossDataType(loss_data_type).lossWeight(loss_weight.value_or(1.0f));
            if (ignore_index.has_value())
                builder.ignoreIndex(uint32_t(ignore_index.value()));
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
        "mask"_a = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged sparse categorical cross-entropy loss.)nbdoc");

    sparse_categorical_cross_entropy.attr("__doc__") = R"nbdoc(
Sparse categorical cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor or thor.RaggedTensor
    Logits tensor whose final/trailing dimension is the class dimension. Ragged predictions must have trailing shape ``[C]``.
labels : thor.Tensor or thor.RaggedTensor
    Sparse integer class ids. Dense dimensions must match the prediction prefix dimensions, or that prefix with a trailing singleton.
    Ragged labels must share the exact prediction row partition and have scalar trailing shape ``[]`` or ``[1]``.
num_classes : int
    Number of classes in predictions.
loss_data_type : thor.DataType, default thor.DataType.FP32
reported_loss_shape : thor.losses.LossShape, default batch
    This setting does not affect training; it only controls the reported loss tensor shape.
ignore_index : int, optional keyword-only
    Label id that contributes zero loss and zero logits gradient.
mask : thor.Tensor or thor.RaggedTensor, optional keyword-only
    Dense inputs use a prefix-shaped mask. Ragged inputs require a scalar-per-token ragged mask with the exact same row partition.
    Boolean/uint8/fp16/fp32 masks are supported. Entries > 0.5 are valid; masked entries contribute zero loss and zero gradient.

Notes
-----
Sparse categorical cross-entropy is logits-native: it computes logsumexp(logits) - logits[class_id]
without materializing a separate softmax tensor or a per-class raw loss tensor. The raw loss shape is
the predictions prefix shape, e.g. predictions [B, S, V] produce raw loss [B, S].

The logits gradient is dense and equivalent to softmax(logits) - one_hot(class_id).
For ragged inputs, ``raw`` is one scalar per active token and preserves the prediction row partition; ``per_example`` and
``batch`` use ragged row reductions. ``per_output`` is undefined.
)nbdoc";
}
