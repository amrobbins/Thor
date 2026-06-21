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

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/BoundParameter.h"
#include "DeepLearning/Api/Parameter/ParameterConstraint.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;
using DataType = ThorImplementation::DataType;
using PhysicalTensor = ThorImplementation::Tensor;
using StorageContext = ThorImplementation::PhysicalParameter::StorageContext;
using GlorotMode = ThorImplementation::Glorot::Mode;

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


std::vector<std::shared_ptr<ParameterConstraint>> constraintsFromPython(const nb::object& obj) {
    std::vector<std::shared_ptr<ParameterConstraint>> constraints;
    if (obj.is_none()) {
        return constraints;
    }

    auto appendConstraint = [&constraints](const nb::handle& handle) {
        std::shared_ptr<ParameterConstraint> constraint;
        try {
            constraint = nb::cast<std::shared_ptr<ParameterConstraint>>(handle);
        } catch (const std::exception&) {
            throw nb::type_error("parameter constraints must be thor.constraints.ParameterConstraint instances");
        }
        if (constraint == nullptr) {
            throw nb::value_error("parameter constraints may not contain None");
        }
        constraints.push_back(constraint->clone());
    };

    try {
        std::shared_ptr<ParameterConstraint> single = nb::cast<std::shared_ptr<ParameterConstraint>>(obj);
        if (single != nullptr) {
            constraints.push_back(single->clone());
            return constraints;
        }
    } catch (const std::exception&) {
    }

    if (!nb::isinstance<nb::sequence>(obj) || nb::isinstance<nb::str>(obj)) {
        throw nb::type_error("constraints must be a thor.constraints.ParameterConstraint, a sequence of constraints, or None");
    }

    nb::sequence seq = nb::cast<nb::sequence>(obj);
    constraints.reserve(nb::len(seq));
    for (nb::handle item : seq) {
        appendConstraint(item);
    }
    return constraints;
}
}  // namespace

void bind_parameter(nb::module_& thor) {
    auto parameter_reference = nb::class_<ParameterReference>(thor, "ParameterReference");
    parameter_reference.attr("__module__") = "thor";
    parameter_reference.def("__init__", [](ParameterReference* self, uint64_t parameterizable_id, const std::string& parameter_name) {
        new (self) ParameterReference(parameterizable_id, parameter_name);
    }, "parameterizable_id"_a, "parameter_name"_a);
    parameter_reference.def_prop_ro("parameterizable_id", &ParameterReference::getParameterizableId);
    parameter_reference.def_prop_ro("parameter_name", &ParameterReference::getParameterName);
    parameter_reference.def("is_initialized", &ParameterReference::isInitialized);
    parameter_reference.def("get_architecture_json", [](const ParameterReference& self) { return self.architectureJson().dump(); });
    parameter_reference.def("__eq__", &ParameterReference::operator==);

    auto bound_parameter = nb::class_<BoundParameter>(thor, "BoundParameter");
    bound_parameter.attr("__module__") = "thor";
    bound_parameter.def_prop_ro("name", &BoundParameter::getName);
    bound_parameter.def_prop_ro("trainable", &BoundParameter::isTrainable);
    bound_parameter.def("is_trainable", &BoundParameter::isTrainable);
    bound_parameter.def("is_training_enabled", &BoundParameter::isTrainingEnabled);
    bound_parameter.def("set_training_enabled", &BoundParameter::setTrainingEnabled, "enabled"_a);
    bound_parameter.def("has_optimizer", &BoundParameter::hasOptimizer);

    auto parameter_constraint = nb::class_<ParameterConstraint>(thor, "ParameterConstraint");
    parameter_constraint.attr("__module__") = "thor";
    parameter_constraint.def_prop_ro("constraint_type", &ParameterConstraint::getConstraintType);
    parameter_constraint.def("get_architecture_json", [](const ParameterConstraint& self) { return self.architectureJson().dump(); });

    auto non_negative_constraint = nb::class_<NonNegativeParameterConstraint, ParameterConstraint>(
        thor, "NonNegativeParameterConstraint");
    non_negative_constraint.attr("__module__") = "thor";
    non_negative_constraint.def(nb::init<>());
    non_negative_constraint.attr("__doc__") = R"nbdoc(
Post-update parameter constraint that clips parameter values to be non-negative.

This is a general Thor parameter constraint, not a layer-specific hack. It can be
attached to any trainable ParameterSpecification or to layer builders that expose
parameter-specific constraint arguments.
    )nbdoc";

    auto non_positive_constraint = nb::class_<NonPositiveParameterConstraint, ParameterConstraint>(
        thor, "NonPositiveParameterConstraint");
    non_positive_constraint.attr("__module__") = "thor";
    non_positive_constraint.def(nb::init<>());
    non_positive_constraint.attr("__doc__") = R"nbdoc(
Post-update parameter constraint that clips parameter values to be non-positive.
    )nbdoc";

    auto min_constraint = nb::class_<MinParameterConstraint, ParameterConstraint>(thor, "MinParameterConstraint");
    min_constraint.attr("__module__") = "thor";
    min_constraint.def(nb::init<double>(), "min_value"_a);
    min_constraint.def_prop_ro("min_value", &MinParameterConstraint::getMinValue);
    min_constraint.attr("__doc__") = R"nbdoc(
Post-update parameter constraint that clips parameter values to be at least min_value.
    )nbdoc";

    auto max_constraint = nb::class_<MaxParameterConstraint, ParameterConstraint>(thor, "MaxParameterConstraint");
    max_constraint.attr("__module__") = "thor";
    max_constraint.def(nb::init<double>(), "max_value"_a);
    max_constraint.def_prop_ro("max_value", &MaxParameterConstraint::getMaxValue);
    max_constraint.attr("__doc__") = R"nbdoc(
Post-update parameter constraint that clips parameter values to be at most max_value.
    )nbdoc";

    auto min_max_constraint = nb::class_<MinMaxParameterConstraint, ParameterConstraint>(thor, "MinMaxParameterConstraint");
    min_max_constraint.attr("__module__") = "thor";
    min_max_constraint.def(nb::init<double, double>(), "min_value"_a, "max_value"_a);
    min_max_constraint.def_prop_ro("min_value", &MinMaxParameterConstraint::getMinValue);
    min_max_constraint.def_prop_ro("max_value", &MinMaxParameterConstraint::getMaxValue);
    min_max_constraint.attr("__doc__") = R"nbdoc(
Post-update parameter constraint that clips parameter values into [min_value, max_value].
    )nbdoc";

    auto parameter = nb::class_<ParameterSpecification>(thor, "ParameterSpecification");
    parameter.attr("__module__") = "thor";

    auto storage_context = nb::class_<StorageContext>(parameter, "StorageContext");
    storage_context.attr("__module__") = "thor.parameters";
    storage_context.attr("__qualname__") = "ParameterSpecification.StorageContext";
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
        [](ParameterSpecification* self,
           const std::string& name,
           const std::vector<uint64_t>& shape,
           DataType dtype,
           std::shared_ptr<Initializer> initializer,
           bool trainable,
           std::shared_ptr<Optimizer> optimizer_override,
           std::optional<bool> training_initially_enabled,
           nb::object constraints) {
            if (initializer == nullptr)
                initializer = Glorot(GlorotMode::UNIFORM).clone();
            if (!training_initially_enabled.has_value())
                training_initially_enabled = trainable;

            ParameterSpecification::Builder builder;
            builder.name(name)
                .shape(shape)
                .dtype(dtype)
                .trainable(trainable)
                .trainingInitiallyEnabled(training_initially_enabled.value())
                .initializer(initializer);

            if (optimizer_override != nullptr) {
                builder.optimizer(optimizer_override);
            }
            builder.constraints(constraintsFromPython(constraints));

            ParameterSpecification built = builder.build();

            new (self) ParameterSpecification(built);
        },
        "name"_a,
        "shape"_a,
        "dtype"_a = DataType::FP32,
        "initializer"_a.none() = nb::none(),
        "trainable"_a = true,
        "optimizer"_a.none() = nb::none(),
        "training_initially_enabled"_a.none() = nb::none(),
        "constraints"_a.none() = nb::none(),
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
        [](ParameterSpecification* self,
           const std::string& name,
           nb::object create_storage_from_context,
           std::shared_ptr<Initializer> initializer,
           bool trainable,
           std::shared_ptr<Optimizer> optimizer_override,
           std::optional<bool> training_initially_enabled,
           nb::object constraints) {
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

            if (initializer == nullptr)
                initializer = Glorot(GlorotMode::UNIFORM).clone();
            if (!training_initially_enabled.has_value())
                training_initially_enabled = trainable;

            ParameterSpecification::Builder builder;
            builder.name(name)
                .createStorage(std::move(createStorage))
                .trainable(trainable)
                .trainingInitiallyEnabled(training_initially_enabled.value())
                .initializer(initializer);

            if (optimizer_override != nullptr) {
                builder.optimizer(optimizer_override);
            }
            builder.constraints(constraintsFromPython(constraints));

            ParameterSpecification built = builder.build();

            new (self) ParameterSpecification(built);
        },
        "name"_a,
        "create_storage_from_context"_a,
        "initializer"_a.none() = nb::none(),
        "trainable"_a = true,
        "optimizer_override"_a.none() = nb::none(),
        "training_initially_enabled"_a.none() = nb::none(),
        "constraints"_a.none() = nb::none(),
        R"nbdoc(
Create an API parameter whose implementation storage is allocated at physical layer compile time.

Provide ``create_storage_from_context``. For single-input layers, the default feature input name is
``"feature_input"``, and ``ParameterSpecification.StorageContext.get_feature_input()`` returns that tensor when
exactly one input is present.
    )nbdoc");

    parameter.def_static("allocate_storage",
                         &ParameterSpecification::allocateStorage,
                         "input_tensor"_a,
                         "shape"_a,
                         "dtype"_a,
                         R"nbdoc(
Allocate implementation storage on the same placement as ``input_tensor`` with the requested shape and dtype.
    )nbdoc");

    parameter.def_prop_ro("name", &ParameterSpecification::getName);
    parameter.def_prop_ro("trainable", &ParameterSpecification::isTrainable);
    parameter.def("is_trainable", &ParameterSpecification::isTrainable);
    parameter.def("is_training_initially_enabled", &ParameterSpecification::isTrainingInitiallyEnabled);
    parameter.def("has_optimizer", &ParameterSpecification::hasOptimizer);
    parameter.def("get_architecture_json", [](const ParameterSpecification& self) { return self.architectureJson().dump(); });
    parameter.def("has_constraints", &ParameterSpecification::hasConstraints);
    parameter.def("get_constraints", &ParameterSpecification::getConstraints);
}

