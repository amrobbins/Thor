#include <nanobind/nanobind.h>

#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;
using DataType = Thor::Tensor::DataType;

void bind_tensor(nb::module_ &m) {
    auto tensor_class = nb::class_<Tensor>(m, "Tensor")
                            .def(
                                "__init__",
                                [](Tensor *self, const vector<uint64_t> &dimensions, const DataType &data_type) {
                                    Tensor tensor(data_type, dimensions);

                                    // Move the tensor layer into the pre-allocated but uninitialized memory at self
                                    new (self) Tensor(std::move(tensor));
                                },
                                "dimensions"_a,
                                "data_type"_a,

                                nb::sig("def __init__(self, "
                                        "dimensions: list[int], "
                                        "data_type: thor.Tensor.DataType "
                                        ") -> None"),

                                R"nbdoc(
        A Tensor that is used to describe the shape of data and to record the
        connections between API elements.

        This tensor does not directly represent a single piece of allocated
        memory - it is possible that multiple instances of physical tensors will
        exist that represent this tensor.

        The actual allocated memory belongs to a physical tensor that is part
        of a stamped network; a corresponding physical tensor can be looked up in
        a stamped network using the ID of this API tensor.

        Parameters
        ----------
        dimensions : list[int]
            The dimensions of the tensor.
            The batch size dimension is **NOT** included here; the batch dimension
            will be created upon realization of a network via the stamping process.
        data_type : thor.Tensor.DataType
            Data type of all elements in the tensor.
        )nbdoc")
                            .def("get_id", &Tensor::getId)
                            .def("get_dimensions", &Tensor::getDimensions)
                            .def("get_data_type", &Tensor::getDataType)
                            .def("get_total_num_elements", &Tensor::getTotalNumElements)
                            .def_static("bytes_per_element", nb::overload_cast<DataType>(&Tensor::getBytesPerElement), "data_type"_a)
                            .def("get_bytes_per_element", nb::overload_cast<>(&Tensor::getBytesPerElement, nb::const_))
                            .def("get_total_size_in_bytes", &Tensor::getTotalSizeInBytes);

    nb::enum_<Tensor::DataType>(tensor_class, "DataType")
        .value("packed_bool", DataType::PACKED_BOOLEAN)
        .value("bool", DataType::BOOLEAN)
        .value("int8", DataType::INT8)
        .value("uint8", DataType::UINT8)
        .value("int16", DataType::INT16)
        .value("uint16", DataType::UINT16)
        .value("int32", DataType::INT32)
        .value("uint32", DataType::UINT32)
        .value("int64", DataType::INT64)
        .value("uint64", DataType::UINT64)
        .value("fp8_e4m3", DataType::FP8_E4M3)
        .value("fp8_e5m2", DataType::FP8_E5M2)
        .value("bf16", DataType::BF16)
        .value("fp16", DataType::FP16)
        .value("fp32", DataType::FP32)
        .value("fp64", DataType::FP64)
        .export_values();
}
