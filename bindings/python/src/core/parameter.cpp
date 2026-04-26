#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

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

namespace {

class GilSafePythonObject {
   public:
    explicit GilSafePythonObject(nb::handle object) : object(object.ptr()) {
        nb::gil_scoped_acquire gil;
        Py_XINCREF(this->object);
    }

    GilSafePythonObject(const GilSafePythonObject&) = delete;
    GilSafePythonObject& operator=(const GilSafePythonObject&) = delete;

    ~GilSafePythonObject() {
        if (object == nullptr)
            return;

        nb::gil_scoped_acquire gil;
        Py_XDECREF(object);
    }

    nb::handle get() const { return nb::handle(object); }

   private:
    PyObject* object = nullptr;
};

}  // namespace

void bind_parameter(nb::module_& thor) {
    auto parameter = nb::class_<Parameter>(thor, "Parameter");
    parameter.attr("__module__") = "thor";

    auto storage_context = nb::class_<StorageContext>(parameter, "StorageContext");
    storage_context.attr("__module__") = "thor";
    storage_context.attr("__qualname__") = "Parameter.StorageContext";
    storage_context.def(nb::init<>());
    storage_context.def(nb::init<std::unordered_map<std::string, PhysicalTensor>>(), "named_inputs"_a);
    storage_context.def(nb::init<PhysicalTensor>(), "feature_input"_a);
    storage_context.def_prop_ro("inputs", [](const StorageContext& self) { return self.namedInputs; });
    storage_context.def("has_input", &StorageContext::hasInput, "name"_a);
    storage_context.def("get_input", &StorageContext::getInput, "name"_a, nb::rv_policy::copy);
    storage_context.def("get_feature_input", &StorageContext::getFeatureInput, nb::rv_policy::copy);
    storage_context.def("input_names", &StorageContext::getInputNames);
    storage_context.def("input_names_string", &StorageContext::getInputNamesString);

    // Parameter definition time attribute resolution
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
Create an API parameter with storage attributes determined at parameter definition time.

Provide:
- ``shape``: the parameter shape
- ``dtype``: the parameter dtype, default ``fp32``

This form is for statically-shaped parameters. For compile-time-dynamic parameters, use
``create_storage_from_context=...`` instead.
    )nbdoc");

    // Layer compile time attribute resolution
    parameter.def(
        "__init__",
        [](Parameter* self,
           const std::string& name,
           nb::object create_storage_from_context,
           std::shared_ptr<Initializer> initializer,
           bool trainable,
           std::shared_ptr<Optimizer> optimizer) {
            if (create_storage_from_context.is_none()) {
                throw std::runtime_error("create_storage_from_context must be provided.");
            }

            if (!nb::isinstance<nb::callable>(create_storage_from_context)) {
                throw std::runtime_error("create_storage_from_context must be callable.");
            }

            auto createStorageRef = std::make_shared<GilSafePythonObject>(create_storage_from_context);

            auto createStorage = [createStorageRef](const StorageContext& context) -> PhysicalTensor {
                nb::gil_scoped_acquire gil;

                nb::callable createStorageCallable = nb::borrow<nb::callable>(createStorageRef->get());

                nb::object result = createStorageCallable(context);
                return nb::cast<PhysicalTensor>(result);
            };

            new (self) Parameter(name, std::move(createStorage), std::move(initializer), trainable, std::move(optimizer));
        },
        "name"_a,
        "create_storage_from_context"_a,
        "initializer"_a.none() = nb::none(),
        "trainable"_a = true,
        "optimizer"_a.none() = nb::none(),
        R"nbdoc(
Create an API parameter whose implementation storage is allocated at physical layer compile time.

Provide ``create_storage_from_context``. For single-input layers, the default feature input name is
``"feature_input"``, and ``Parameter.StorageContext.get_feature_input()`` returns that tensor when exactly one input is present.
    )nbdoc");

    parameter.def_static("allocate_storage",
                         &Parameter::allocateStorage,
                         "input_tensor"_a,
                         "shape"_a,
                         "dtype"_a,
                         R"nbdoc(
Allocate implementation storage on the same placement as ``input_tensor`` with the requested shape and dtype.
    )nbdoc");

    parameter.def_prop_ro("name", &Parameter::getName);
    parameter.def_prop_ro("trainable", &Parameter::isTrainable);
    parameter.def("is_trainable", &Parameter::isTrainable);
    parameter.def("is_training_enabled", &Parameter::isTrainingEnabled);
    parameter.def("set_training_enabled", &Parameter::setTrainingEnabled, "enabled"_a);
    parameter.def("has_optimizer", &Parameter::hasOptimizer);
}
