#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Transpose.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/Expression.h"

#include <optional>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

using DataType = ThorImplementation::DataType;

namespace {

std::optional<DataType> optionalDataTypeFromPython(const nb::object &obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return nb::cast<DataType>(obj);
}

ThorImplementation::Expression makePythonEpilogueInput(const nb::object &outputDTypeObj, const nb::object &computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj);
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj);
    return Transpose::epilogueInput(computeDType, outputDType);
}

void applyPythonEpilogue(Transpose::Builder &builder, const nb::object &epilogue) {
    if (epilogue.is_none()) {
        return;
    }
    if (!nb::isinstance<ThorImplementation::Expression>(epilogue)) {
        throw nb::type_error("epilogue must be a thor.physical.Expression instance or None");
    }
    builder.epilogue(nb::cast<ThorImplementation::Expression>(epilogue));
}

}  // namespace

void bind_transpose(nb::module_ &m) {
    auto transpose = nb::class_<Transpose, Layer>(m, "Transpose");
    transpose.attr("__module__") = "thor.layers";

    transpose.def(
        "__init__",
        [](Transpose *self, Network &network, const Tensor &feature_input, nb::object output_dtype, nb::object epilogue) {
            const auto &dims = feature_input.getDimensions();
            if (dims.size() < 2) {
                throw nb::value_error("Transpose instance: feature_input must have rank >= 2.");
            }

            Transpose::Builder builder;
            builder.network(network).featureInput(feature_input);
            if (!output_dtype.is_none()) {
                builder.outputDataType(nb::cast<DataType>(output_dtype));
            }
            applyPythonEpilogue(builder, epilogue);

            Transpose built = builder.build();
            new (self) Transpose(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "output_dtype"_a.none() = nb::none(),
        "epilogue"_a.none() = nb::none());

    transpose.def_static(
        "epilogue_input",
        &makePythonEpilogueInput,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
            Return the single tensor input expression expected by a Transpose epilogue.
            )nbdoc");

    transpose.def(
        "get_feature_output",
        [](Transpose &self) -> Tensor {
            std::optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor
                The feature output tensor handle.
            )nbdoc");

    transpose.def("get_output_data_type", [](Transpose &self) { return self.getOutputDataType(); });

    transpose.attr("__doc__") = R"nbdoc(
            Create and attach a Transpose layer to a Network.

            The layer swaps the last two feature dimensions. The network batch
            dimension is preserved by the underlying physical expression, so a
            feature tensor shaped [X, Y] is materialized as [Y, X], while the
            stamped physical tensor behaves as [B, X, Y] -> [B, Y, X]. The
            optional output_dtype casts the transposed value before the optional
            epilogue expression is applied.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor
                Input feature tensor for this layer. Must have rank >= 2.
            output_dtype : thor.DataType | None, default None
                Optional dtype for the transposed layer output. Defaults to the
                input feature dtype.
            epilogue : thor.physical.Expression | None, default None
                Optional expression applied after the transpose/output dtype cast.
            )nbdoc";
}
