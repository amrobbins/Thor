#include <nanobind/nanobind.h>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using PhysicalTensor = ThorImplementation::Tensor;
using TensorPlacement = ThorImplementation::TensorPlacement;
using TensorDescriptor = ThorImplementation::TensorDescriptor;
using DataType = ThorImplementation::TensorDescriptor::DataType;

void bind_physical_tensor(nb::module_ &physical) {
    // Tensor Placement
    nb::object device_type = nb::enum_<TensorPlacement::MemDevices>(physical, "DeviceType")
                                 .value("invalid", TensorPlacement::MemDevices::INVALID)
                                 .value("cpu", TensorPlacement::MemDevices::CPU)
                                 .value("gpu", TensorPlacement::MemDevices::GPU);
    device_type.attr("__module__") = "thor.physical";

    nb::object placement = nb::class_<TensorPlacement>(physical, "Placement")
                               .def(nb::init<>())
                               .def(nb::init<TensorPlacement::MemDevices, int>(), "device_type"_a, "device_num"_a = 0)
                               .def("get_device_type", &TensorPlacement::getMemDevice)
                               .def("get_device_num", &TensorPlacement::getDeviceNum)
                               .def("__str__", &TensorPlacement::toString)
                               .def("__repr__", [](TensorPlacement &self) { return string("TensorPlacement(") + self.toString() + ")"; })
                               .def("__eq__", [](const TensorPlacement &a, const TensorPlacement &b) { return a == b; })
                               .def("__ne__", [](const TensorPlacement &a, const TensorPlacement &b) { return a != b; });
    placement.attr("__module__") = "thor.physical";

    // TensorDescriptor
    auto tensor_descriptor = nb::class_<TensorDescriptor>(physical, "Descriptor");
    tensor_descriptor.attr("__module__") = "thor.physical";

    tensor_descriptor.def(
        "__init__",
        [](TensorDescriptor *self, DataType data_type, const vector<uint64_t> &dims) {
            if (dims.empty())
                throw nb::value_error("TensorDescriptor: dimensions cannot be empty");
            new (self) TensorDescriptor(data_type, dims);
        },
        "data_type"_a,
        "dimensions"_a,
        R"nbdoc(
TensorDescriptor(data_type, dimensions)

Parameters
----------
data_type : thor.DataType
dimensions : list[int]
)nbdoc");

    tensor_descriptor.def("get_data_type", &TensorDescriptor::getDataType);
    tensor_descriptor.def("get_dimensions", [](const TensorDescriptor &self) { return self.getDimensions(); });
    tensor_descriptor.def("get_num_dimensions", &TensorDescriptor::getNumDimensions);
    tensor_descriptor.def("get_total_num_elements", &TensorDescriptor::getTotalNumElements);
    tensor_descriptor.def("__str__", &TensorDescriptor::toString);
    tensor_descriptor.def("__repr__", [](const TensorDescriptor &self) { return string("TensorDescriptor(") + self.toString() + ")"; });
    tensor_descriptor.def("__eq__", [](const TensorDescriptor &a, const TensorDescriptor &b) { return a == b; });
    tensor_descriptor.def("__ne__", [](const TensorDescriptor &a, const TensorDescriptor &b) { return a != b; });

    // Size helpers
    tensor_descriptor.def_static(
        "array_size_in_bytes",
        [](long num_elements, DataType dt) {
            if (num_elements < 0)
                throw nb::value_error("num_elements must be >= 0");
            return (uint64_t)TensorDescriptor::getArraySizeInBytes(num_elements, dt);
        },
        "num_elements"_a,
        "data_type"_a,
        R"nbdoc(
Return the number of bytes required to store num_elements of data_type.

Note: packed_boolean is handled correctly.
)nbdoc");

    tensor_descriptor.def("get_array_size_in_bytes", [](TensorDescriptor &self) { return (uint64_t)self.getArraySizeInBytes(); });

    tensor_descriptor.def("get_element_name", &TensorDescriptor::getElementName);

    tensor_descriptor.def_static("element_type_name", &TensorDescriptor::getElementTypeName, "data_type"_a);
    tensor_descriptor.def_static(
        "element_size_in_bytes", [](DataType dt) { return TensorDescriptor::getElementSizeInBytes(dt); }, "data_type"_a);

    tensor_descriptor.def("is_integral_type", [](const TensorDescriptor &self) { return self.isIntegralType(); });
    tensor_descriptor.def_static("is_integral_data_type", [](DataType dt) { return TensorDescriptor::isIntegralType(dt); }, "data_type"_a);

    tensor_descriptor.def("is_boolean_type", [](const TensorDescriptor &self) { return self.isBooleanType(); });
    tensor_descriptor.def_static("is_boolean_data_type", [](DataType dt) { return TensorDescriptor::isBooleanType(dt); }, "data_type"_a);

    tensor_descriptor.def("is_signed_type", [](const TensorDescriptor &self) { return self.isSignedType(); });
    tensor_descriptor.def_static("is_signed_data_type", [](DataType dt) { return TensorDescriptor::isSignedType(dt); }, "data_type"_a);

    // reshape / indexing helpers (safe)
    tensor_descriptor.def(
        "reshape",
        [](TensorDescriptor &self, const vector<uint64_t> &new_dims) {
            if (new_dims.empty())
                throw nb::value_error("new_dimensions cannot be empty");
            self.reshape(new_dims);
        },
        "new_dimensions"_a);

    tensor_descriptor.def(
        "flat_index",
        [](TensorDescriptor &self, const vector<uint64_t> &element) { return (uint64_t)self.getFlatIndex(element); },
        "element"_a);

    tensor_descriptor.def(
        "dimensional_index",
        [](TensorDescriptor &self, uint64_t flat_index) { return self.getDimensionalIndex((unsigned long)flat_index); },
        "flat_index"_a);

    tensor_descriptor.def(
        "dimension_stride", [](TensorDescriptor &self, uint32_t axis) { return (uint64_t)self.getDimensionStride(axis); }, "axis"_a);

    // Physical Tensor
    auto physical_tensor = nb::class_<PhysicalTensor>(physical, "PhysicalTensor");
    physical_tensor.attr("__module__") = "thor.physical";

    // placement + descriptor + optional alignment
    physical_tensor.def(
        "__init__",
        [](PhysicalTensor *self, TensorPlacement placement, TensorDescriptor descriptor, uint32_t alignment_bytes) {
            new (self) PhysicalTensor(placement, descriptor, alignment_bytes);
        },
        "placement"_a,
        "descriptor"_a,
        "alignment_bytes"_a = 256,
        R"nbdoc(
    Create a PhysicalTensor with owned storage.

    Parameters
    ----------
    placement : thor.physical.TensorPlacement
    descriptor : thor.physical.TensorDescriptor
    alignment_bytes : int, default 256
        Byte alignment for pinned memory allocated on the CPU. 256 byte alignment is supported by cuda natively.
    )nbdoc");

    // Copy support (nanobind can usually do this implicitly, but being explicit is fine)
    physical_tensor.def("__copy__", [](const PhysicalTensor &self) { return PhysicalTensor(self); });
    physical_tensor.def("__deepcopy__", [](const PhysicalTensor &self, nb::handle /*memo*/) { return PhysicalTensor(self); }, "memo"_a);

    // Optional: repr (minimal)
    physical_tensor.def("__repr__", [](const PhysicalTensor & /*self*/) { return string("<thor.physical.PhysicalTensor>"); });

    // TODO (once available): expose safe introspection
    // t.def("get_descriptor", &PhysicalTensor::getDescriptor);
    // t.def("get_placement", &PhysicalTensor::getPlacement);
    // t.def("get_bytes", &PhysicalTensor::getSizeInBytes);

    physical_tensor.attr("Placement") = placement;
    placement.attr("__qualname__") = "PhysicalTensor.Placement";

    physical_tensor.attr("DeviceType") = device_type;
    device_type.attr("__qualname__") = "PhysicalTensor.DeviceType";

    physical_tensor.attr("Descriptor") = tensor_descriptor;
    tensor_descriptor.attr("__qualname__") = "PhysicalTensor.Descriptor";
}
