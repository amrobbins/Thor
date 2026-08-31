#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Transpose.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/Expression.h"
#include "bindings/python/src/core/cast.h"

#include <optional>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;
namespace pybind = Thor::PythonBindings;

using DataType = ThorImplementation::DataType;

namespace {

std::optional<DataType> optionalDataTypeFromPython(const nb::object &obj,
                                                   const char *functionName,
                                                   const char *argumentName) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return pybind::castArgument<DataType>(obj, functionName, argumentName, "thor.DataType or None", false);
}

ThorImplementation::Expression makePythonEpilogueInput(const nb::object &outputDTypeObj, const nb::object &computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj, "Transpose.epilogue_input", "output_dtype");
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj, "Transpose.epilogue_input", "compute_dtype");
    return Transpose::epilogueInput(computeDType, outputDType);
}

void applyPythonEpilogue(Transpose::Builder &builder, const nb::object &epilogue) {
    if (epilogue.is_none()) {
        return;
    }
    builder.epilogue(pybind::castArgument<ThorImplementation::Expression>(
        epilogue, "Transpose", "epilogue", "thor.physical.Expression or None", false));
}

}  // namespace

void bind_transpose(nb::module_ &m) {
    auto transpose = nb::class_<Transpose, Layer>(m, "Transpose");
    transpose.attr("__module__") = "thor.layers";

    transpose.def(
        "__init__",
        [](Transpose *self, Network &network, nb::object feature_input, nb::object output_dtype, nb::object epilogue) {
            Transpose::Builder builder;
            builder.network(network);
            if (nb::isinstance<RaggedTensor>(feature_input)) {
                RaggedTensor ragged = nb::cast<RaggedTensor>(feature_input);
                if (ragged.getTrailingDimensions().size() < 2) {
                    throw nb::value_error("Ragged Transpose instance: feature_input must have at least two trailing value dimensions.");
                }
                builder.featureInput(ragged);
            } else if (nb::isinstance<Tensor>(feature_input)) {
                Tensor tensor = nb::cast<Tensor>(feature_input);
                if (tensor.getDimensions().size() < 2) {
                    throw nb::value_error("Transpose instance: feature_input must have rank >= 2.");
                }
                builder.featureInput(tensor);
            } else {
                throw nb::type_error("Transpose feature_input must be thor.Tensor or thor.RaggedTensor.");
            }
            if (!output_dtype.is_none()) {
                builder.outputDataType(pybind::castArgument<DataType>(
                    output_dtype, "Transpose", "output_dtype", "thor.DataType or None", false));
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
        [](Transpose &self) -> nb::object {
            if (std::optional<RaggedTensor> raggedOutput = self.getRaggedFeatureOutput(); raggedOutput.has_value()) {
                return nb::cast(raggedOutput.value());
            }
            return nb::cast(self.getFeatureOutput().value());
        },
        R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor or thor.RaggedTensor
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
            feature_input : thor.Tensor or thor.RaggedTensor
                Dense inputs must have rank >= 2. Ragged inputs must have at least
                two trailing value dimensions; only those final two dimensions are
                transposed and the row partition is preserved.
            output_dtype : thor.DataType | None, default None
                Optional dtype for the transposed layer output. Defaults to the
                input feature dtype.
            epilogue : thor.physical.Expression | None, default None
                Optional expression applied after the transpose/output dtype cast.
            )nbdoc";
}
