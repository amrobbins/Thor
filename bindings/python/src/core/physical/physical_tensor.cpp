#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings/python/src/core/physical/NanobindDTypes.h"

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
            return TensorDescriptor::getArraySizeInBytes(num_elements, dt);
        },
        "num_elements"_a,
        "data_type"_a,
        R"nbdoc(
Return the number of bytes required to store num_elements of data_type.
)nbdoc");

    tensor_descriptor.def_static(
        "element_size_in_bytes", [](DataType dt) { return TensorDescriptor::getElementSizeInBytes(dt); }, "data_type"_a);
    tensor_descriptor.def("get_array_size_in_bytes", [](TensorDescriptor &self) { return self.getArraySizeInBytes(); });

    tensor_descriptor.def("get_element_type_name",
                          [](const TensorDescriptor &self) { return TensorDescriptor::getElementTypeName(self.getDataType()); });
    tensor_descriptor.def_static("element_type_name", [](DataType dt) { return TensorDescriptor::getElementTypeName(dt); }, "data_type"_a);

    tensor_descriptor.def("is_integral_type", [](const TensorDescriptor &self) { return self.isIntegralType(); });
    tensor_descriptor.def_static("is_integral_data_type", [](DataType dt) { return TensorDescriptor::isIntegralType(dt); }, "data_type"_a);

    tensor_descriptor.def("is_boolean_type", [](const TensorDescriptor &self) { return self.isBooleanType(); });
    tensor_descriptor.def_static("is_boolean_data_type", [](DataType dt) { return TensorDescriptor::isBooleanType(dt); }, "data_type"_a);

    tensor_descriptor.def("is_signed_type", [](const TensorDescriptor &self) { return self.isSignedType(); });
    tensor_descriptor.def_static("is_signed_data_type", [](DataType dt) { return TensorDescriptor::isSignedType(dt); }, "data_type"_a);

    // reshape / indexing helpers
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
        [](TensorDescriptor &self, const std::vector<uint64_t> &element) { return self.getFlatIndex(element); },
        "element"_a,
        R"nbdoc(
Return the flat index corresponding to a multidimensional element index.

Parameters
----------
element : Sequence[int]
    One index per tensor dimension. Its length must match the number
    of dimensions, and each index must be within bounds.

Returns
-------
int
    The flattened linear index of the element in row-major order.
    )nbdoc");

    tensor_descriptor.def(
        "dimensional_index",
        [](TensorDescriptor &self, uint64_t flat_index) { return self.getDimensionalIndex(flat_index); },
        "flat_index"_a,
        R"nbdoc(
Return the per dimension indexes of an element, given its flat index (element offset from the beginning of the tensor).

Parameters
----------
flat_index : int
    Offset of the element from the beginning of the tensor.

Returns
-------
Sequence[int]
    One index per tensor dimension, that addresses the element at offset flat_index.
    )nbdoc");

    tensor_descriptor.def(
        "dimension_stride",
        [](TensorDescriptor &self, uint32_t axis) { return self.getDimensionStride(axis); },
        "axis"_a,
        R"nbdoc(
Return the number of elements contained at the specified axis, before the next index in the axis.
For example:

    if tensor has shape [2][3][4]
    tensor.dimension_stride(axis=0) == 12
    tensor.dimension_stride(axis=1) == 4
    tensor.dimension_stride(axis=2) == 1

Parameters
----------
axis : int
    The dimension for which the stride is computed.

Returns
-------
int
    The number of elements between subsequent indexes in the specified dimension.
    )nbdoc");

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

    physical_tensor.def("__repr__", [](const PhysicalTensor & /*self*/) { return string("<thor.physical.PhysicalTensor>"); });

    physical_tensor.def("get_descriptor", &PhysicalTensor::getDescriptor);
    physical_tensor.def("get_placement", &PhysicalTensor::getPlacement);
    physical_tensor.def("get_size_in_bytes", &PhysicalTensor::getArraySizeInBytes);

    physical_tensor.def_prop_ro("dimensions", [](const PhysicalTensor &self) { return self.getDescriptor().getDimensions(); });

    physical_tensor.def("numpy", [](PhysicalTensor &self) -> nb::object {
        if (self.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU)
            throw nb::value_error("PhysicalTensor.numpy() requires CPU placement");

        TensorDescriptor desc = self.getDescriptor();

        std::vector<size_t> shape;
        for (uint64_t d : desc.getDimensions())
            shape.push_back(static_cast<size_t>(d));

        std::vector<int64_t> strides(shape.size());
        int64_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= static_cast<int64_t>(shape[i]);
        }

        nb::object owner = nb::find(&self);
        if (!owner.is_valid())
            throw std::runtime_error("PhysicalTensor.numpy(): could not find Python owner object");

        auto make_shape_tuple = [&]() -> nb::tuple {
            nb::list lst;
            for (size_t d : shape)
                lst.append(nb::int_(d));
            return nb::tuple(lst);
        };

        auto numel = [&]() -> size_t {
            size_t n = 1;
            for (size_t d : shape)
                n *= d;
            return n;
        };

        auto view_bytes_as_ml_dtype = [&](void *ptr, size_t itemsize, nb::object dtype_obj) -> nb::object {
            size_t nbytes = numel() * itemsize;

            // Zero-copy 1D byte array whose base object is the tensor owner.
            nb::object raw = nb::cast(nb::ndarray<uint8_t, nb::numpy>(static_cast<uint8_t *>(ptr), 1, &nbytes, owner, nullptr));

            // Reinterpret bytes as the requested ml_dtypes dtype, then reshape.
            return raw.attr("view")(dtype_obj).attr("reshape")(make_shape_tuple());
        };

        switch (desc.getDataType()) {
            case DataType::FP32:
                return nb::cast(nb::ndarray<float, nb::numpy>(self.getMemPtr<float>(), shape.size(), shape.data(), owner, strides.data()));

            case DataType::FP16:
                return nb::cast(nb::ndarray<half, nb::numpy>(self.getMemPtr<half>(), shape.size(), shape.data(), owner, strides.data()));

            case DataType::BF16: {
                nb::module_ mld = nb::module_::import_("ml_dtypes");
                return view_bytes_as_ml_dtype(self.getMemPtr<__nv_bfloat16>(), sizeof(__nv_bfloat16), mld.attr("bfloat16"));
            }

            case DataType::FP8_E4M3: {
                nb::module_ mld = nb::module_::import_("ml_dtypes");
                return view_bytes_as_ml_dtype(self.getMemPtr<__nv_fp8_e4m3>(), sizeof(__nv_fp8_e4m3), mld.attr("float8_e4m3fn"));
            }

            case DataType::FP8_E5M2: {
                nb::module_ mld = nb::module_::import_("ml_dtypes");
                return view_bytes_as_ml_dtype(self.getMemPtr<__nv_fp8_e5m2>(), sizeof(__nv_fp8_e5m2), mld.attr("float8_e5m2"));
            }

            default:
                throw nb::value_error("PhysicalTensor.numpy() does not support this dtype");
        }
    });

    physical_tensor.def(
        "copy_from_async",
        [](PhysicalTensor &self, const PhysicalTensor &source, Stream stream) { self.copyFromAsync(source, stream); },
        "source"_a,
        "stream"_a,
        R"nbdoc(
Asynchronously copy tensor contents from source into this tensor using the provided stream.

Parameters
----------
source : thor.physical.PhysicalTensor
    Source tensor to copy from.
stream : thor.physical.Stream
    Stream used for the copy.
)nbdoc");

    // physical_tensor.attr("Placement") = placement;
    // placement.attr("__qualname__") = "PhysicalTensor.Placement";
    //
    // physical_tensor.attr("DeviceType") = device_type;
    // device_type.attr("__qualname__") = "PhysicalTensor.DeviceType";

    physical_tensor.attr("Descriptor") = tensor_descriptor;
    tensor_descriptor.attr("__qualname__") = "PhysicalTensor.Descriptor";
    nb::delattr(physical, "Descriptor");

    physical_tensor.def("clone",
                        nb::overload_cast<>(&PhysicalTensor::clone, nb::const_),
                        R"nbdoc(
Create another tensor like this one with the same placement, data type, and dimensions.
)nbdoc");

    physical_tensor.def("clone",
                        nb::overload_cast<TensorPlacement>(&PhysicalTensor::clone, nb::const_),
                        "new_placement"_a,
                        R"nbdoc(
Create another tensor like this one but with a different placement.

Parameters
----------
new_placement : thor.physical.Placement
    Destination placement for the cloned tensor.
)nbdoc");

    physical_tensor.def("clone",
                        nb::overload_cast<TensorDescriptor::DataType>(&PhysicalTensor::clone, nb::const_),
                        "new_data_type"_a,
                        R"nbdoc(
Create another tensor like this one but with a different data type.

Parameters
----------
new_data_type : thor.DataType
    Destination data type for the cloned tensor.
)nbdoc");

    physical_tensor.def("clone",
                        nb::overload_cast<TensorPlacement, TensorDescriptor::DataType>(&PhysicalTensor::clone, nb::const_),
                        "new_placement"_a,
                        "new_data_type"_a,
                        R"nbdoc(
Create another tensor like this one but with a different placement and data type.

Parameters
----------
new_placement : thor.physical.Placement
    Destination placement for the cloned tensor.
new_data_type : thor.DataType
    Destination data type for the cloned tensor.
)nbdoc");

    physical_tensor.def("clone",
                        nb::overload_cast<std::vector<uint64_t>>(&PhysicalTensor::clone, nb::const_),
                        "new_dimensions"_a,
                        R"nbdoc(
Create another tensor like this one but with a different dimensions.

Parameters
----------
new_dimensions : list[int]
    New tensor dimensions.
)nbdoc");
}
