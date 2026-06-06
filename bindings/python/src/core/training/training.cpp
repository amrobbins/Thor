#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <utility>

#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Training/StepExecutable.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingStep.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_training(nb::module_& training) {
    training.doc() = "Thor training program scaffolding";

    auto gradient_clear_policy = nb::enum_<TrainingStep::GradientClearPolicy>(training, "GradientClearPolicy")
                                     .value("clear_before_step", TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP)
                                     .value("accumulate", TrainingStep::GradientClearPolicy::ACCUMULATE);
    gradient_clear_policy.attr("__module__") = "thor.training";

    auto training_input_binding = nb::class_<TrainingInputBinding>(training, "TrainingInputBinding");
    training_input_binding.attr("__module__") = "thor.training";
    training_input_binding.def(
        "__init__",
        [](TrainingInputBinding* self, const std::string& network_input_name, const std::string& batch_input_name) {
            new (self) TrainingInputBinding(network_input_name, batch_input_name);
        },
        "network_input_name"_a,
        "batch_input_name"_a);
    training_input_binding.def_prop_ro("network_input_name", &TrainingInputBinding::getNetworkInputName);
    training_input_binding.def_prop_ro("batch_input_name", &TrainingInputBinding::getBatchInputName);
    training_input_binding.def("is_initialized", &TrainingInputBinding::isInitialized);
    training_input_binding.def("get_architecture_json", &TrainingInputBinding::architectureJsonString);
    training_input_binding.def("__eq__", &TrainingInputBinding::operator==);

    auto training_step = nb::class_<TrainingStep>(training, "TrainingStep");
    training_step.attr("__module__") = "thor.training";
    training_step.def(
        "__init__",
        [](TrainingStep* self,
           const std::string& name,
           std::vector<Tensor> loss_roots,
           std::shared_ptr<Optimizer> optimizer,
           std::vector<ParameterReference> update_parameters,
           uint32_t repeat_count,
           TrainingStep::GradientClearPolicy gradient_clear_policy,
           std::vector<TrainingInputBinding> input_bindings) {
            new (self) TrainingStep(name,
                                    std::move(loss_roots),
                                    std::move(optimizer),
                                    std::move(update_parameters),
                                    repeat_count,
                                    gradient_clear_policy,
                                    std::move(input_bindings));
        },
        "name"_a,
        "loss_roots"_a,
        "optimizer"_a.none() = nb::none(),
        "update_parameters"_a = std::vector<ParameterReference>{},
        "repeat_count"_a = 1,
        "gradient_clear_policy"_a = TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
        "input_bindings"_a = std::vector<TrainingInputBinding>{});
    training_step.def_prop_ro("name", &TrainingStep::getName);
    training_step.def_prop_ro("repeat_count", &TrainingStep::getRepeatCount);
    training_step.def_prop_ro("gradient_clear_policy", &TrainingStep::getGradientClearPolicy);
    training_step.def("is_initialized", &TrainingStep::isInitialized);
    training_step.def("get_loss_roots", &TrainingStep::getLossRoots, nb::rv_policy::reference_internal);
    training_step.def("get_optimizer", &TrainingStep::getOptimizer);
    training_step.def("get_update_parameters", &TrainingStep::getUpdateParameters, nb::rv_policy::reference_internal);
    training_step.def("get_input_bindings", &TrainingStep::getInputBindings, nb::rv_policy::reference_internal);
    training_step.def("updates_parameter", &TrainingStep::updatesParameter, "parameter"_a);
    training_step.def("get_architecture_json", &TrainingStep::architectureJsonString);

    auto step_executable = nb::class_<StepExecutable>(training, "StepExecutable");
    step_executable.attr("__module__") = "thor.training";
    step_executable.def("is_initialized", &StepExecutable::isInitialized);
    step_executable.def_prop_ro("name", &StepExecutable::getName);
    step_executable.def_prop_ro("repeat_count", &StepExecutable::getRepeatCount);
    step_executable.def_prop_ro("gradient_clear_policy", &StepExecutable::getGradientClearPolicy);
    step_executable.def("get_loss_roots", &StepExecutable::getLossRoots, nb::rv_policy::reference_internal);
    step_executable.def("get_optimizer", &StepExecutable::getOptimizer);
    step_executable.def(
        "get_update_parameter_references", &StepExecutable::getUpdateParameterReferences, nb::rv_policy::reference_internal);
    step_executable.def("get_resolved_update_parameters", &StepExecutable::getResolvedUpdateParameters, nb::rv_policy::reference_internal);
    step_executable.def("get_input_bindings", &StepExecutable::getInputBindings, nb::rv_policy::reference_internal);
    step_executable.def("get_architecture_json", &StepExecutable::architectureJsonString);

    auto training_program = nb::class_<TrainingProgram>(training, "TrainingProgram");
    training_program.attr("__module__") = "thor.training";
    training_program.def(nb::init<>());
    training_program.def(nb::init<std::vector<TrainingStep>>(), "steps"_a);
    training_program.def("add_step", &TrainingProgram::addStep, "step"_a);
    training_program.def("get_num_steps", &TrainingProgram::getNumSteps);
    training_program.def("get_step", &TrainingProgram::getStep, "index"_a, nb::rv_policy::reference_internal);
    training_program.def("get_steps", &TrainingProgram::getSteps, nb::rv_policy::reference_internal);
    training_program.def("is_initialized", &TrainingProgram::isInitialized);
    training_program.def("get_architecture_json", &TrainingProgram::architectureJsonString);
    training_program.def("compile", &TrainingProgram::compile, "placed_network"_a);
}
