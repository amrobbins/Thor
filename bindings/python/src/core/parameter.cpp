#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <memory>
#include <optional>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;
using DataType = ThorImplementation::TensorDescriptor::DataType;
using PhysicalTensor = ThorImplementation::Tensor;

class PyParameter : public Parameter {
   public:
    NB_TRAMPOLINE(Parameter, 1);

    PhysicalTensor create_storage(const PhysicalTensor& inputTensor) const override { NB_OVERRIDE(create_storage, inputTensor); }
};

void bind_parameter(nb::module_& thor) {
    auto parameter = nb::class_<Parameter, PyParameter>(thor, "Parameter");
    parameter.attr("__module__") = "thor";

    parameter.def(
        "__init__",
        [](Parameter* self,
           const std::string& name,
           const std::vector<uint64_t>& shape,
           DataType dtype,
           std::shared_ptr<Initializer> initializer,
           bool trainable,
           std::shared_ptr<Optimizer> optimizer) {
            new (self) Parameter(name, shape, dtype, std::move(initializer), trainable, std::move(optimizer));
        },
        "name"_a,
        "shape"_a,
        "dtype"_a = DataType::FP32,
        "initializer"_a.none() = nb::none(),
        "trainable"_a = true,
        "optimizer"_a.none() = nb::none(),
        R"nbdoc(
Create a fixed-shape API parameter.

The parameter is logical at API construction time. During physical layer compilation,
it stamps to an implementation parameter and allocates storage on the same placement as
the layer input tensor passed to ``create_storage``.

Parameters
----------
name : str
    Parameter name. Names beginning with ``__`` are reserved.
shape : list[int]
    Physical parameter shape, including all non-batch dimensions. Parameters do not
    automatically receive a batch dimension.
dtype : thor.DataType, default thor.DataType.fp32
    Storage dtype.
initializer : thor.initializers.Initializer or None, default None
    Optional initializer used when the physical storage is initialized.
trainable : bool, default True
    Whether gradients and optimizer updates should be applied.
optimizer : thor.optimizers.Optimizer or None, default None
    Reserved for parameter-specific optimizer overrides. Layer-level optimizers are
    currently the primary supported path.
        )nbdoc");

    parameter.def_prop_ro("name", &Parameter::getName);
    parameter.def_prop_ro("shape", &Parameter::getShape);
    parameter.def_prop_ro("dtype", &Parameter::getDataType);
    parameter.def_prop_ro("trainable", &Parameter::isTrainable);
    parameter.def("is_trainable", &Parameter::isTrainable);
    parameter.def("is_training_enabled", &Parameter::isTrainingEnabled);
    parameter.def("set_training_enabled", &Parameter::setTrainingEnabled, "enabled"_a);
    parameter.def("has_optimizer", &Parameter::hasOptimizer);

    parameter.def(
        "createStorage",
        [](const Parameter& self,
           const PhysicalTensor& inputTensor,
           std::optional<std::vector<uint64_t>> shape,
           std::optional<DataType> dtype) {
            if (shape.has_value() || dtype.has_value()) {
                return self.createStorage(inputTensor, shape.value_or(self.getShape()), dtype.value_or(self.getDataType()));
            }
            return self.createStorage(inputTensor);
        },
        "input_tensor"_a,
        "shape"_a.none() = nb::none(),
        "dtype"_a.none() = nb::none(),
        R"nbdoc(
Default physical storage helper.

This intentionally bypasses the Python ``create_storage`` virtual override, so a
subclass can compute a dynamic shape or dtype and delegate allocation back to the
basic fixed-shape allocator::

    def create_storage(self, input_tensor):
        shape = [input_tensor.get_descriptor().get_dimensions()[-1]]
        dtype = input_tensor.get_descriptor().get_data_type()
        return self.createStorage(input_tensor, shape=shape, dtype=dtype)

When ``shape`` or ``dtype`` are omitted, this uses the parameter's stored fixed
shape and dtype.
        )nbdoc");

    parameter.def("create_storage",
                  &Parameter::create_storage,
                  "input_tensor"_a,
                  R"nbdoc(
Create physical storage for this parameter from a physical input tensor.

Subclasses may override this method and return a ``thor.physical.PhysicalTensor``.
The default implementation allocates this parameter's fixed shape and dtype on the
same placement as ``input_tensor``. Within an override, call ``createStorage`` to
delegate to the default allocator.
        )nbdoc");

    parameter.attr("__doc__") = R"nbdoc(
Logical API parameter used by custom trainable layers.

A ``Parameter`` describes storage that will be materialized later, when the API
network is stamped into physical layers. The base class is the common fixed-shape
case; subclass it and override ``create_storage(input_tensor)`` when storage shape,
dtype, or placement should be computed directly from the physical layer input.
    )nbdoc";
}
