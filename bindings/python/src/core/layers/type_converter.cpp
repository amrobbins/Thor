#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <optional>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;

void bind_type_converter(nb::module_ &m) {
    nb::class_<TypeConverter, Layer>(m, "TypeConverter")
        .def(
            "__init__",
            [](TypeConverter *self, Network &network, nb::object feature_input, const DataType &new_data_type) {
                TypeConverter::Builder builder;
                builder.network(network);
                if (nb::isinstance<RaggedTensor>(feature_input)) {
                    builder.featureInput(nb::cast<RaggedTensor>(feature_input));
                } else if (nb::isinstance<Tensor>(feature_input)) {
                    builder.featureInput(nb::cast<Tensor>(feature_input));
                } else {
                    throw nb::type_error("TypeConverter feature_input must be thor.Tensor or thor.RaggedTensor.");
                }
                TypeConverter built = builder.newDataType(new_data_type).build();

                // Move the TypeConverter layer into the pre-allocated but uninitialized memory at self.
                new (self) TypeConverter(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "new_data_type"_a,

            R"nbdoc(
            Create and attach an expression-backed TypeConverter to a Network.

            Parameters
            ----------
            network : thor.Network
                The network that the layer should be added to.
            feature_input : thor.Tensor or thor.RaggedTensor
                Dense or packed-ragged values to convert. Ragged inputs preserve their row partition.
            new_data_type : thor.DataType
                Data type of the output tensor (e.g. thor.DataType.fp16).
            )nbdoc")
        .def(
            "get_feature_output",
            [](TypeConverter &self) -> nb::object {
                if (std::optional<RaggedTensor> ragged_output = self.getRaggedFeatureOutput(); ragged_output.has_value()) {
                    return nb::cast(ragged_output.value());
                }
                return nb::cast(self.getFeatureOutput().value());
            },
            R"nbdoc(
Return the converted logical output. Ragged inputs produce a RaggedTensor with the same row partition.
)nbdoc")
        .def("get_use_ragged", &TypeConverter::getUseRagged);
}
