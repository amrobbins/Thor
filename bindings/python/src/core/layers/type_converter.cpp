#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::TensorDescriptor::DataType;

void bind_type_converter(nb::module_ &m) {
    nb::class_<TypeConverter, Layer>(m, "TypeConverter")
        .def(
            "__init__",
            [](TypeConverter *self, Network &network, const Tensor &feature_input, const DataType &new_data_type) {
                TypeConverter::Builder builder;
                TypeConverter built = builder.network(network).featureInput(feature_input).newDataType(new_data_type).build();

                // Move the typeConverter layer into the pre-allocated but uninitialized memory at self
                new (self) TypeConverter(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "new_data_type"_a,

            R"nbdoc(
            Create and attach a TypeConverter to send data into a Network.

            Parameters
            ----------
            network : thor.Network
                The network that the layer should be added to.
            new_data_type : thor.DataType
                Data type of the output tensor (e.g. thor.DataType.fp16).
            )nbdoc");
}
