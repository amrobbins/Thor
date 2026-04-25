#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
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
using StorageContext = ThorImplementation::Parameter::StorageContext;

class PyParameter : public Parameter {
   public:
    NB_TRAMPOLINE(Parameter, 1);

    PhysicalTensor create_storage(const StorageContext& context) const override {
        nb::gil_scoped_acquire gil;
        NB_OVERRIDE(create_storage, context);
    }
};

namespace {
PhysicalTensor createStorageHelper(const Parameter& self,
                                   const nb::object& input_or_context,
                                   std::optional<std::vector<uint64_t>> shape,
                                   std::optional<DataType> dtype) {
    auto create_from_input = [&](const PhysicalTensor& inputTensor) {
        if (shape.has_value() || dtype.has_value()) {
            return self.createStorage(inputTensor, shape.value_or(self.getShape()), dtype.value_or(self.getDataType()));
        }
        return self.createStorage(inputTensor);
    };

    if (nb::isinstance<StorageContext>(input_or_context)) {
        const StorageContext& context = nb::cast<const StorageContext&>(input_or_context);
        if (shape.has_value() || dtype.has_value()) {
            return self.createStorage(context, shape.value_or(self.getShape()), dtype.value_or(self.getDataType()));
        }
        return self.createStorage(context);
    }

    const PhysicalTensor& inputTensor = nb::cast<const PhysicalTensor&>(input_or_context);
    return create_from_input(inputTensor);
}
}  // namespace

void bind_parameter(nb::module_& thor) {
    auto parameter = nb::class_<Parameter, PyParameter>(thor, "Parameter");
    parameter.attr("__module__") = "thor";

    auto storage_context = nb::class_<StorageContext>(parameter, "StorageContext");
    storage_context.attr("__module__") = "thor";
    storage_context.attr("__qualname__") = "Parameter.StorageContext";
    storage_context.def(nb::init<>());
    storage_context.def(nb::init<const PhysicalTensor&>(), "feature_input"_a);
    storage_context.def(nb::init<std::unordered_map<std::string, PhysicalTensor>>(), "named_inputs"_a);
    storage_context.def_prop_ro("inputs", [](const StorageContext& self) { return self.namedInputs; });
    storage_context.def("has_input", &StorageContext::hasInput, "name"_a);
    storage_context.def("get_input", &StorageContext::getInput, "name"_a, nb::rv_policy::copy);
    storage_context.def("get_feature_input", &StorageContext::getFeatureInput);
    storage_context.def("input_names", &StorageContext::getInputNames);
    storage_context.def("input_names_string", &StorageContext::getInputNamesString);

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
        )nbdoc");

    parameter.def_prop_ro("name", &Parameter::getName);
    parameter.def_prop_ro("shape", &Parameter::getShape);
    parameter.def_prop_ro("dtype", &Parameter::getDataType);
    parameter.def_prop_ro("trainable", &Parameter::isTrainable);
    parameter.def("is_trainable", &Parameter::isTrainable);
    parameter.def("is_training_enabled", &Parameter::isTrainingEnabled);
    parameter.def("set_training_enabled", &Parameter::setTrainingEnabled, "enabled"_a);
    parameter.def("has_optimizer", &Parameter::hasOptimizer);

    parameter.def("createStorage",
                  &createStorageHelper,
                  "input_or_context"_a,
                  "shape"_a.none() = nb::none(),
                  "dtype"_a.none() = nb::none(),
                  R"nbdoc(
Default physical storage helper.

This intentionally bypasses the Python ``create_storage`` virtual override, so a
subclass can compute a dynamic shape or dtype and delegate allocation back to the
basic fixed-shape allocator::

    def create_storage(self, ctx):
        x = ctx.get_input("feature_input")
        shape = [x.get_descriptor().get_dimensions()[-1]]
        dtype = x.get_descriptor().get_data_type()
        return self.createStorage(x, shape=shape, dtype=dtype)

``input_or_context`` may be either a ``thor.physical.PhysicalTensor`` or a
``thor.Parameter.StorageContext``.
        )nbdoc");

    parameter.def(
        "create_storage",
        [](const Parameter& self, const nb::object& input_or_context) {
            if (nb::isinstance<StorageContext>(input_or_context)) {
                return self.create_storage(nb::cast<const StorageContext&>(input_or_context));
            }
            return self.create_storage(nb::cast<const PhysicalTensor&>(input_or_context));
        },
        "input_or_context"_a,
        R"nbdoc(
Create physical storage for this parameter from a physical input tensor or storage context.

Custom-layer parameters should generally override ``create_storage(ctx)`` and use
``ctx.get_input(name)`` so named feature inputs are available. The default
implementation allocates this parameter's fixed shape and dtype on the context's
single feature input placement.
        )nbdoc");

    parameter.attr("__doc__") = R"nbdoc(
Logical API parameter used by custom trainable layers.

A ``Parameter`` describes storage that will be materialized later, when the API
network is stamped into physical layers. The base class is the common fixed-shape
case; subclass it and override ``create_storage(ctx)`` when storage shape, dtype,
or placement should be computed directly from named physical layer inputs.
    )nbdoc";
}
