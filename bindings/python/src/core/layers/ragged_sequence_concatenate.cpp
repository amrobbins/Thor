#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Layers/Utility/RaggedSequenceConcatenate.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "bindings/python/src/core/cast.h"

#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;
namespace pybind = Thor::PythonBindings;

void bind_ragged_sequence_concatenate(nb::module_& m) {
    auto layer = nb::class_<RaggedSequenceConcatenate, MultiConnectionLayer>(m, "RaggedSequenceConcatenate");
    layer.attr("__module__") = "thor.layers";
    layer.def(
        "__init__",
        [](RaggedSequenceConcatenate* self, Network& network, nb::object featureInputsObject) {
            nb::list featureInputs = pybind::castArgument<nb::list>(
                featureInputsObject,
                "RaggedSequenceConcatenate",
                "feature_inputs",
                "list[thor.RaggedTensor]",
                false);
            if (featureInputs.size() < 2) {
                throw nb::value_error("RaggedSequenceConcatenate feature_inputs must contain at least two RaggedTensor objects.");
            }
            RaggedSequenceConcatenate::Builder builder;
            builder.network(network);
            for (size_t i = 0; i < featureInputs.size(); ++i) {
                const std::string context =
                    "RaggedSequenceConcatenate() argument 'feature_inputs'[" + std::to_string(i) + "]";
                builder.featureInput(pybind::castOrTypeError<RaggedTensor>(
                    featureInputs[i], context, "thor.RaggedTensor", false));
            }
            new (self) RaggedSequenceConcatenate(std::move(builder.build()));
        },
        "network"_a,
        "feature_inputs"_a,
        R"nbdoc(
Concatenate rank-1 ragged inputs along their variable-length sequence axis.

Every input must have the same logical batch size, values dtype, offsets dtype,
and trailing value shape. Row partitions may differ. For each logical row, the
output contains row 0 from every input in argument order, then row 1 from every
input, and so on. The layer explicitly produces a new canonical offsets tensor;
it does not reuse any input partition. Inactive packed capacity is never read.
)nbdoc");
    layer.def("get_feature_output", &RaggedSequenceConcatenate::getRaggedFeatureOutput);
}
