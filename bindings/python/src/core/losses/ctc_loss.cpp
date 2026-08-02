#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "DeepLearning/Api/Layers/Loss/CtcLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

namespace {

void setReportedLossShape(CtcLoss::Builder& builder, Loss::LossShape shape) {
    switch (shape) {
        case Loss::LossShape::NONE:
            builder.reportsNoLoss();
            return;
        case Loss::LossShape::BATCH:
            builder.reportsBatchLoss();
            return;
        case Loss::LossShape::PER_EXAMPLE:
            builder.reportsPerExampleLoss();
            return;
        case Loss::LossShape::RAW:
            builder.reportsRawLoss();
            return;
        case Loss::LossShape::PER_OUTPUT:
            throw nb::value_error("CtcLoss reported_loss_shape does not support per_output; use none, batch, per_example, or raw.");
    }
    throw nb::value_error("CtcLoss received an unknown reported_loss_shape value.");
}

std::string oobModeName(ThorImplementation::CtcLossOobGradientMode mode) {
    switch (mode) {
        case ThorImplementation::CtcLossOobGradientMode::ZERO:
            return "zero";
        case ThorImplementation::CtcLossOobGradientMode::SKIP:
            return "skip";
    }
    throw std::runtime_error("Unknown CTC out-of-bounds gradient mode.");
}

}  // namespace

void bind_ctc_loss(nb::module_& losses) {
    auto ctc = nb::class_<CtcLoss, Loss>(losses, "CtcLoss");
    ctc.attr("__module__") = "thor.losses";

    ctc.def(
        "__init__",
        [](CtcLoss* self,
           Network& network,
           Tensor logits,
           RaggedTensor labels,
           Tensor input_lengths,
           Loss::LossShape reported_loss_shape,
           std::optional<float> loss_weight,
           std::string out_of_bounds_gradients) {
            CtcLoss::Builder builder;
            builder.network(network).logits(logits).labels(labels).inputLengths(input_lengths).lossDataType(ThorImplementation::DataType::FP32);
            builder.lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);

            if (out_of_bounds_gradients == "zero") {
                builder.zeroOutOfBoundsGradients();
            } else if (out_of_bounds_gradients == "skip") {
                builder.skipOutOfBoundsGradients();
            } else {
                throw nb::value_error("CtcLoss out_of_bounds_gradients must be 'zero' or 'skip'.");
            }

            new (self) CtcLoss(std::move(builder.build()));
        },
        "network"_a,
        "logits"_a,
        "labels"_a,
        "input_lengths"_a,
        "reported_loss_shape"_a = Loss::LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        "out_of_bounds_gradients"_a = "zero",
        R"nbdoc(
Canonical cuDNN-backed CTC loss.

``labels`` must be a rank-1 ``thor.RaggedTensor`` with INT32 packed values and
canonical UINT32/UINT64 row-partition offsets. Label lengths are derived from
those offsets on device; there is no padded-label or separately supplied
label-length API.
        )nbdoc");

    ctc.def("get_labels", &CtcLoss::getRaggedLabels);
    ctc.def("get_ragged_labels", &CtcLoss::getRaggedLabels);
    ctc.def("get_input_lengths", &CtcLoss::getInputLengths);
    ctc.def("get_out_of_bounds_gradients", [](const CtcLoss& self) { return oobModeName(self.getOobGradientMode()); });
}
