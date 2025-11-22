#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Thor::Tensor::DataType;

void bind_type_converter(nb::module_ &m) {
    nb::class_<TypeConverter, Layer>(m, "TypeConverter")
        .def(
            "__init__",
            [](TypeConverter *self, Network &network, const DataType &new_data_type) {
                TypeConverter::Builder builder;
                TypeConverter built = builder.network(network).newDataType(new_data_type).build();

                // Move the typeConverter layer into the pre-allocated but uninitialized memory at self
                new (self) TypeConverter(std::move(built));
            },
            "network"_a,
            "new_data_type"_a,

            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "new_data_type: thor.Tensor.DataType"
                    ") -> None"),

            R"nbdoc(
            Create and attach a TypeConverter to send data into a Network.

            Parameters
            ----------
            network : thor.Network
                The network that the layer should be added to.
            new_data_type : thor.DataType
                Data type of the output tensor (e.g. thor.Tensor.DataType.fp16).
            )nbdoc");
}
