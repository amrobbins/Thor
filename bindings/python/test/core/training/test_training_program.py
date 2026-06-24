import json

import pytest
import thor


def _make_fc_network(name: str):
    n = thor.Network(name)
    x = thor.layers.NetworkInput(n, "input", [3], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(n, x.get_feature_output(), 2, True, activation=None)
    return n, fc


def test_parameter_references_are_logical_preplacement_handles():
    _, fc = _make_fc_network("test_training_program_parameter_refs")

    refs = fc.get_parameter_references()
    assert [ref.parameter_name for ref in refs] == ["weights", "biases"]
    assert all(ref.parameterizable_id == fc.get_id() for ref in refs)

    weights = fc.get_parameter_reference("weights")
    assert weights.parameter_name == "weights"
    assert weights.parameterizable_id == fc.get_id()


def test_network_trainable_parameter_references_follow_default_freeze_state():
    n, _ = _make_fc_network("test_training_program_network_refs")
    assert [ref.parameter_name for ref in n.get_trainable_parameter_references()] == ["weights", "biases"]

    n.freeze_training()
    assert n.get_trainable_parameter_references() == []
    assert [ref.parameter_name for ref in n.get_trainable_parameter_references(training_enabled_only=False)] == ["weights", "biases"]

    n.unfreeze_training()
    assert [ref.parameter_name for ref in n.get_trainable_parameter_references()] == ["weights", "biases"]


def test_training_step_and_program_are_ordered_logical_execution_specs():
    _, fc = _make_fc_network("test_training_program_ordered_steps")
    sgd = thor.optimizers.Sgd(initial_learning_rate=0.01)
    loss = thor.Tensor([1], thor.DataType.fp32)

    d_step = thor.training.TrainingStep(
        "discriminator",
        [loss],
        optimizer=sgd,
        update_parameters=fc.get_parameter_references(),
        repeat_count=2,
    )
    g_step = thor.training.TrainingStep(
        "generator",
        [loss],
        optimizer=sgd,
        update_parameters=[fc.get_parameter_reference("weights")],
    )

    program = thor.training.TrainingProgram([d_step, g_step])
    assert program.get_num_steps() == 2
    assert program.get_step(0).name == "discriminator"
    assert program.get_step(1).name == "generator"

    arch = json.loads(program.get_architecture_json())
    assert [step["name"] for step in arch["steps"]] == ["discriminator", "generator"]
    assert arch["steps"][0]["repeat_count"] == 2
    assert [param["parameter_name"] for param in arch["steps"][0]["update_parameters"]] == ["weights", "biases"]

    restored = thor.training.TrainingProgram.deserialize(program.get_architecture_json())
    assert restored.get_num_steps() == 2
    assert restored.get_step(0).name == "discriminator"
    assert restored.get_step(1).name == "generator"


def test_training_program_rejects_duplicate_step_names():
    sgd = thor.optimizers.Sgd(initial_learning_rate=0.01)
    loss = thor.Tensor([1], thor.DataType.fp32)
    step = thor.training.TrainingStep("same", [loss], optimizer=sgd, update_parameters=[thor.parameters.ParameterReference(1, "weights")])
    program = thor.training.TrainingProgram([step])

    with pytest.raises(RuntimeError, match="already contains a step named"):
        program.add_step(step)


def test_training_phase_enable_disable_and_serialization():
    loss = thor.Tensor([1], thor.DataType.fp32)
    forecast = thor.Tensor([100], thor.DataType.fp32)

    phase = thor.training.TrainingPhase(
        "daily_prediction",
        loss_roots=[loss],
        outputs={"forecast": forecast},
        depends_on=["feature_preprocessing"],
        enabled=False,
    )

    assert phase.name == "daily_prediction"
    assert not phase.enabled
    assert not phase.is_enabled()
    assert phase.get_loss_roots() == [loss]
    assert phase.get_outputs()["forecast"] == forecast
    assert phase.get_depends_on() == ["feature_preprocessing"]

    phase.enable()
    assert phase.enabled
    phase.enabled = False
    assert not phase.is_enabled()
    phase.set_enabled(True)
    assert phase.is_enabled()

    arch = json.loads(phase.get_architecture_json())
    assert arch["version"] == "1.0.0"
    assert arch["name"] == "daily_prediction"
    assert arch["enabled"] is True
    assert arch["depends_on"] == ["feature_preprocessing"]
    assert list(arch["outputs"].keys()) == ["forecast"]

    restored = thor.training.TrainingPhase.deserialize(phase.get_architecture_json())
    assert restored.name == "daily_prediction"
    assert restored.enabled
    assert restored.get_depends_on() == ["feature_preprocessing"]
    assert len(restored.get_loss_roots()) == 1
    assert set(restored.get_outputs().keys()) == {"forecast"}

def test_training_phase_can_own_regular_network_and_derive_outputs():
    phase_network = thor.Network("daily_phase_network")
    phase_input = thor.layers.NetworkInput(phase_network, "features", [8], thor.DataType.fp32)
    thor.layers.NetworkOutput(phase_network, "forecast", phase_input.get_feature_output(), thor.DataType.fp32)

    phase = thor.training.TrainingPhase("daily_prediction", network=phase_network, enabled=False)

    assert phase.name == "daily_prediction"
    assert phase.has_network()
    assert phase.get_network().get_network_name() == "daily_phase_network"
    assert not phase.enabled
    assert phase.get_loss_roots() == []
    assert set(phase.get_outputs().keys()) == {"forecast"}
    assert phase.get_depends_on() == []

    arch = json.loads(phase.get_architecture_json())
    assert arch["version"] == "1.1.0"
    assert arch["name"] == "daily_prediction"
    assert arch["enabled"] is False
    assert arch["network"]["name"] == "daily_phase_network"
    assert "depends_on" not in arch

    restored = thor.training.TrainingPhase.deserialize(phase.get_architecture_json())
    assert restored.name == "daily_prediction"
    assert restored.has_network()
    assert restored.get_network().get_network_name() == "daily_phase_network"
    assert not restored.enabled
    assert set(restored.get_outputs().keys()) == {"forecast"}


def test_training_phase_rejects_network_with_legacy_fields():
    phase_network = thor.Network("mixed_phase_network")
    loss = thor.Tensor([1], thor.DataType.fp32)

    with pytest.raises(ValueError, match="either network or legacy"):
        thor.training.TrainingPhase("mixed", network=phase_network, loss_roots=[loss])


def test_training_phase_validation_and_deserialization_errors_are_exposed():
    loss = thor.Tensor([1], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="requires a non-empty name"):
        thor.training.TrainingPhase("", loss_roots=[loss])

    with pytest.raises(RuntimeError, match="dependency names must be non-empty"):
        thor.training.TrainingPhase("daily_prediction", depends_on=[""])

    with pytest.raises(RuntimeError, match="cannot depend on itself"):
        thor.training.TrainingPhase("daily_prediction", depends_on=["daily_prediction"])

    with pytest.raises(RuntimeError, match="contains duplicate dependency 'daily_prediction'"):
        thor.training.TrainingPhase("aggregate_prediction", depends_on=["daily_prediction", "daily_prediction"])

    enabled_phase = thor.training.TrainingPhase("enabled_phase", loss_roots=[loss], enabled=True)
    disabled_phase = thor.training.TrainingPhase("disabled_phase", loss_roots=[loss], enabled=False)
    assert thor.training.TrainingPhase.deserialize(enabled_phase.get_architecture_json()).enabled
    assert not thor.training.TrainingPhase.deserialize(disabled_phase.get_architecture_json()).enabled

    bad_arch = json.loads(enabled_phase.get_architecture_json())
    bad_arch["version"] = "2.0.0"
    with pytest.raises(RuntimeError, match="Unsupported TrainingPhase version: 2.0.0"):
        thor.training.TrainingPhase.deserialize(json.dumps(bad_arch))


def test_training_step_phase_constructor_collects_enabled_phase_loss_roots():
    daily_loss = thor.Tensor([1], thor.DataType.fp32)
    aggregate_loss = thor.Tensor([1], thor.DataType.fp32)

    daily_phase = thor.training.TrainingPhase("daily_prediction", loss_roots=[daily_loss])
    aggregate_phase = thor.training.TrainingPhase(
        "aggregate_prediction",
        loss_roots=[aggregate_loss],
        depends_on=["daily_prediction"],
        enabled=False,
    )

    step = thor.training.TrainingStep("demand_forecast", phases=[daily_phase, aggregate_phase])
    assert step.enabled
    assert step.is_enabled()
    assert step.get_active_phase_names() == ["daily_prediction"]
    assert step.get_active_loss_roots() == [daily_loss]
    assert [phase.name for phase in step.get_phases()] == ["daily_prediction", "aggregate_prediction"]

    aggregate_phase.enable()
    assert step.get_active_phase_names() == ["daily_prediction", "aggregate_prediction"]
    assert step.get_active_loss_roots() == [daily_loss, aggregate_loss]
    step.validate_enabled_phase_dependencies()

    daily_phase.disable()
    with pytest.raises(RuntimeError, match="depends on disabled phase 'daily_prediction'"):
        step.validate_enabled_phase_dependencies()


def test_training_step_phase_dependency_validation_skips_disabled_phases_and_steps():
    daily_loss = thor.Tensor([1], thor.DataType.fp32)
    aggregate_loss = thor.Tensor([1], thor.DataType.fp32)

    daily_phase = thor.training.TrainingPhase("daily_prediction", loss_roots=[daily_loss])
    disabled_aggregate = thor.training.TrainingPhase(
        "aggregate_prediction",
        loss_roots=[aggregate_loss],
        depends_on=["missing_daily"],
        enabled=False,
    )
    step = thor.training.TrainingStep("demand_forecast", phases=[daily_phase, disabled_aggregate])
    step.validate_enabled_phase_dependencies()
    assert step.get_active_phase_names() == ["daily_prediction"]
    assert step.get_active_loss_roots() == [daily_loss]

    disabled_aggregate.enable()
    with pytest.raises(RuntimeError, match="enabled phase 'aggregate_prediction' depends on missing phase 'missing_daily'"):
        step.validate_enabled_phase_dependencies()

    disabled_daily = thor.training.TrainingPhase("daily_prediction", loss_roots=[daily_loss], enabled=False)
    enabled_aggregate = thor.training.TrainingPhase(
        "aggregate_prediction",
        loss_roots=[aggregate_loss],
        depends_on=["daily_prediction"],
        enabled=True,
    )
    disabled_step = thor.training.TrainingStep(
        "demand_forecast",
        phases=[disabled_daily, enabled_aggregate],
        enabled=False,
    )
    disabled_step.validate_enabled_phase_dependencies()
    assert disabled_step.get_active_phase_names() == []
    assert disabled_step.get_active_loss_roots() == []


def test_training_step_dependency_errors_are_specific_and_searchable():
    loss = thor.Tensor([1], thor.DataType.fp32)

    disabled_daily = thor.training.TrainingPhase("daily_prediction", loss_roots=[loss], enabled=False)
    aggregate = thor.training.TrainingPhase("aggregate_prediction", loss_roots=[loss], depends_on=["daily_prediction"])
    step = thor.training.TrainingStep("demand_forecast", phases=[disabled_daily, aggregate])
    with pytest.raises(
        RuntimeError,
        match=r"TrainingStep 'demand_forecast' enabled phase 'aggregate_prediction' depends on disabled phase 'daily_prediction'\.",
    ):
        step.validate_enabled_phase_dependencies()

    missing = thor.training.TrainingPhase("aggregate_prediction", loss_roots=[loss], depends_on=["daily_prediction"])
    missing_step = thor.training.TrainingStep("demand_forecast", phases=[missing])
    with pytest.raises(RuntimeError, match="enabled phase 'aggregate_prediction' depends on missing phase 'daily_prediction'"):
        missing_step.validate_enabled_phase_dependencies()

    aggregate_first = thor.training.TrainingPhase("aggregate_prediction", loss_roots=[loss], depends_on=["daily_prediction"])
    daily_second = thor.training.TrainingPhase("daily_prediction", loss_roots=[loss])
    forward_step = thor.training.TrainingStep("demand_forecast", phases=[aggregate_first, daily_second])
    with pytest.raises(RuntimeError, match="but that dependency does not appear earlier in the step"):
        forward_step.validate_enabled_phase_dependencies()


def test_training_step_forward_only_phase_can_be_active_without_contributing_loss_roots():
    features = thor.Tensor([16], thor.DataType.fp32)
    daily_loss = thor.Tensor([1], thor.DataType.fp32)
    aggregate_loss = thor.Tensor([1], thor.DataType.fp32)

    preprocessing = thor.training.TrainingPhase("feature_preprocessing", outputs={"features": features})
    daily = thor.training.TrainingPhase("daily_prediction", loss_roots=[daily_loss], depends_on=["feature_preprocessing"])
    aggregate = thor.training.TrainingPhase(
        "aggregate_prediction",
        loss_roots=[aggregate_loss],
        depends_on=["daily_prediction"],
        enabled=False,
    )
    step = thor.training.TrainingStep("demand_forecast", phases=[preprocessing, daily, aggregate])

    assert step.get_active_phase_names() == ["feature_preprocessing", "daily_prediction"]
    assert step.get_active_loss_roots() == [daily_loss]

    aggregate.enable()
    assert step.get_active_phase_names() == ["feature_preprocessing", "daily_prediction", "aggregate_prediction"]
    assert step.get_active_loss_roots() == [daily_loss, aggregate_loss]


def test_training_step_phase_serialization_and_legacy_json_without_phases():
    daily_loss = thor.Tensor([1], thor.DataType.fp32)
    aggregate_loss = thor.Tensor([1], thor.DataType.fp32)
    daily = thor.training.TrainingPhase("daily_prediction", loss_roots=[daily_loss])
    aggregate = thor.training.TrainingPhase(
        "aggregate_prediction",
        loss_roots=[aggregate_loss],
        depends_on=["daily_prediction"],
        enabled=False,
    )

    step = thor.training.TrainingStep(
        "demand_forecast",
        phases=[daily, aggregate],
        repeat_count=3,
        gradient_clear_policy=thor.training.GradientClearPolicy.accumulate,
    )
    restored = thor.training.TrainingStep.deserialize(step.get_architecture_json())
    assert restored.repeat_count == 3
    assert restored.gradient_clear_policy == thor.training.GradientClearPolicy.accumulate
    assert [phase.name for phase in restored.get_phases()] == ["daily_prediction", "aggregate_prediction"]
    assert restored.get_phases()[0].enabled
    assert not restored.get_phases()[1].enabled
    assert restored.get_phases()[1].get_depends_on() == ["daily_prediction"]
    assert restored.get_active_phase_names() == ["daily_prediction"]

    legacy_arch = json.loads(step.get_architecture_json())
    legacy_arch["version"] = "1.0.0"
    del legacy_arch["phases"]
    legacy_step = thor.training.TrainingStep.deserialize(json.dumps(legacy_arch))
    assert [phase.name for phase in legacy_step.get_phases()] == ["demand_forecast_phase"]
    assert len(legacy_step.get_active_loss_roots()) == 2

    bad_arch = json.loads(step.get_architecture_json())
    bad_arch["version"] = "9.9.9"
    with pytest.raises(RuntimeError, match="Unsupported TrainingStep version: 9.9.9"):
        thor.training.TrainingStep.deserialize(json.dumps(bad_arch))


def test_training_step_constructor_requires_loss_roots_or_phases_but_not_both():
    loss = thor.Tensor([1], thor.DataType.fp32)
    phase = thor.training.TrainingPhase("daily_prediction", loss_roots=[loss])

    with pytest.raises(ValueError, match="exactly one of loss_roots or phases"):
        thor.training.TrainingStep("bad")

    with pytest.raises(ValueError, match="exactly one of loss_roots or phases"):
        thor.training.TrainingStep("bad", [loss], phases=[phase])


def test_training_program_holds_training_steps_by_reference_in_python():
    loss = thor.Tensor([1], thor.DataType.fp32)
    daily_phase = thor.training.TrainingPhase("daily_prediction", loss_roots=[loss])
    aggregate_phase = thor.training.TrainingPhase(
        "aggregate_prediction",
        loss_roots=[loss],
        depends_on=["daily_prediction"],
        enabled=False,
    )
    step = thor.training.TrainingStep("demand_forecast", phases=[daily_phase, aggregate_phase])
    program = thor.training.TrainingProgram([step])

    assert program.get_step(0).is_enabled()
    step.disable()
    assert not program.get_step(0).is_enabled()

    program.get_step(0).enable()
    assert step.is_enabled()

    assert program.get_step(0).get_active_phase_names() == ["daily_prediction"]
    assert not program.get_step(0).get_phases()[1].enabled
    aggregate_phase.enable()
    assert step.get_phases()[1].enabled
    assert program.get_step(0).get_phases()[1].enabled
    assert program.get_step(0).get_active_phase_names() == ["daily_prediction", "aggregate_prediction"]

    program.get_step(0).get_phases()[1].disable()
    assert not aggregate_phase.enabled
    assert program.get_step(0).get_active_phase_names() == ["daily_prediction"]


def test_training_step_allows_updates_without_step_optimizer_for_parameter_overrides():
    loss = thor.Tensor([1], thor.DataType.fp32)
    weights = thor.parameters.ParameterReference(1, "weights")

    step = thor.training.TrainingStep("per_parameter", [loss], update_parameters=[weights])

    assert step.get_optimizer() is None
    assert step.get_update_parameters() == [weights]
    arch = json.loads(step.get_architecture_json())
    assert "optimizer" not in arch

    restored = thor.training.TrainingStep.deserialize(step.get_architecture_json())
    assert restored.get_optimizer() is None
    assert restored.get_update_parameters() == [weights]


def test_training_program_rejects_empty_programs():
    with pytest.raises(RuntimeError, match="at least one TrainingStep"):
        thor.training.TrainingProgram([])

    program = thor.training.TrainingProgram()
    assert not program.is_initialized()
    with pytest.raises(RuntimeError, match="at least one TrainingStep"):
        program.get_architecture_json()

    with pytest.raises(RuntimeError, match="at least one TrainingStep"):
        thor.training.TrainingProgram.deserialize(json.dumps({"version": "1.0.0", "steps": []}))


def test_training_program_rejects_out_of_range_access_and_unsupported_version():
    loss = thor.Tensor([1], thor.DataType.fp32)
    step = thor.training.TrainingStep("daily", [loss])
    program = thor.training.TrainingProgram([step])

    with pytest.raises(RuntimeError, match="step index out of range"):
        program.get_step(1)

    arch = json.loads(program.get_architecture_json())
    arch["version"] = "2.0.0"
    with pytest.raises(RuntimeError, match="Unsupported TrainingProgram version: 2.0.0"):
        thor.training.TrainingProgram.deserialize(json.dumps(arch))


def test_training_input_bindings_are_serialized_and_validated():
    loss = thor.Tensor([1], thor.DataType.fp32)
    z_binding = thor.training.TrainingInputBinding("z", "z_discriminator")

    assert z_binding.network_input_name == "z"
    assert z_binding.batch_input_name == "z_discriminator"

    binding_arch = json.loads(z_binding.get_architecture_json())
    assert binding_arch["network_input_name"] == "z"
    assert binding_arch["batch_input_name"] == "z_discriminator"
    assert thor.training.TrainingInputBinding.deserialize(z_binding.get_architecture_json()) == z_binding

    step = thor.training.TrainingStep(
        "discriminator",
        [loss],
        input_bindings=[thor.training.TrainingInputBinding("real_images", "real_images"), z_binding],
    )
    assert [binding.network_input_name for binding in step.get_input_bindings()] == ["real_images", "z"]

    arch = json.loads(step.get_architecture_json())
    assert [binding["batch_input_name"] for binding in arch["input_bindings"]] == ["real_images", "z_discriminator"]

    restored_step = thor.training.TrainingStep.deserialize(step.get_architecture_json())
    assert restored_step.name == "discriminator"
    assert [binding.batch_input_name for binding in restored_step.get_input_bindings()] == ["real_images", "z_discriminator"]

    with pytest.raises(RuntimeError, match="duplicate binding"):
        thor.training.TrainingStep(
            "bad",
            [loss],
            input_bindings=[
                thor.training.TrainingInputBinding("input", "a"),
                thor.training.TrainingInputBinding("input", "b"),
            ],
        )


@pytest.mark.cuda
def test_training_program_compile_plans_step_executables_and_resolves_update_parameters():
    n = thor.Network("test_training_program_step_compile")
    x = thor.layers.NetworkInput(n, "input", [3], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(n, "labels", [1], thor.DataType.fp32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01)
    fc = thor.layers.FullyConnected(
        n,
        x.get_feature_output(),
        1,
        True,
        activation=None,
        weights_optimizer=optimizer,
        biases_optimizer=optimizer,
    )
    loss_root = fc.get_feature_output()
    thor.layers.NetworkOutput(n, "scores", loss_root, thor.DataType.fp32)
    thor.layers.NetworkOutput(n, "labels_out", labels.get_feature_output(), thor.DataType.fp32)

    placed = n.place(2, inference_only=False)
    assert placed.has_network_input("input")
    assert not placed.has_network_input("missing_input")
    assert set(placed.get_network_input_names()) == {"input", "labels"}
    assert placed.has_api_tensor(loss_root)

    weights = fc.get_parameter_reference("weights")
    resolved = placed.resolve_parameter_reference(weights)
    assert resolved.name == "weights"
    assert resolved.is_trainable()
    assert resolved.is_training_enabled()

    step = thor.training.TrainingStep(
        "generator",
        [loss_root],
        optimizer=optimizer,
        update_parameters=[weights],
        repeat_count=2,
        gradient_clear_policy=thor.training.GradientClearPolicy.accumulate,
        input_bindings=[thor.training.TrainingInputBinding("input", "z_generator")],
    )
    program = thor.training.TrainingProgram([step])
    executables = program.compile(placed)

    assert len(executables) == 1
    executable = executables[0]
    assert executable.name == "generator"
    assert executable.repeat_count == 2
    assert len(executable.get_resolved_loss_roots()) == 1
    assert [param.name for param in executable.get_resolved_update_parameters()] == ["weights"]
    assert [binding.batch_input_name for binding in executable.get_input_bindings()] == ["z_generator"]
    resolved_input_bindings = {
        binding.network_input_name: binding.batch_input_name for binding in executable.get_resolved_input_bindings()
    }
    assert resolved_input_bindings == {"input": "z_generator", "labels": "labels"}
    assert executable.get_required_batch_input_names() == ["labels", "z_generator"]

    arch = json.loads(executable.get_architecture_json())
    assert arch["planned"] is True
    assert arch["resolved_loss_root_count"] == 1
    assert arch["resolved_update_parameter_count"] == 1
    assert arch["input_bindings"][0]["network_input_name"] == "input"
    assert len(arch["resolved_input_bindings"]) == 2
    assert arch["required_batch_input_names"] == ["labels", "z_generator"]

    bad_input_step = thor.training.TrainingStep(
        "bad_input",
        [loss_root],
        optimizer=optimizer,
        update_parameters=[weights],
        input_bindings=[thor.training.TrainingInputBinding("missing_input", "batch")],
    )
    with pytest.raises(RuntimeError, match="unknown NetworkInput"):
        thor.training.TrainingProgram([bad_input_step]).compile(placed)

    bad_loss_step = thor.training.TrainingStep(
        "bad_loss",
        [thor.Tensor([1], thor.DataType.fp32)],
        optimizer=optimizer,
        update_parameters=[weights],
    )
    with pytest.raises(RuntimeError, match="does not belong to network"):
        thor.training.TrainingProgram([bad_loss_step]).compile(placed)

@pytest.mark.cuda
def test_training_program_compile_reflects_python_phase_enable_mutation():
    n = thor.Network("test_training_program_python_phase_compile")
    x = thor.layers.NetworkInput(n, "input", [3], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(n, "labels", [1], thor.DataType.fp32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01)
    fc = thor.layers.FullyConnected(
        n,
        x.get_feature_output(),
        1,
        True,
        activation=None,
        weights_optimizer=optimizer,
        biases_optimizer=optimizer,
    )
    daily_loss = fc.get_feature_output()
    aggregate_loss = labels.get_feature_output()
    thor.layers.NetworkOutput(n, "scores", daily_loss, thor.DataType.fp32)
    thor.layers.NetworkOutput(n, "labels_out", aggregate_loss, thor.DataType.fp32)

    placed = n.place(2, inference_only=False)

    daily_phase = thor.training.TrainingPhase("daily_prediction", loss_roots=[daily_loss])
    aggregate_phase = thor.training.TrainingPhase(
        "aggregate_prediction",
        loss_roots=[aggregate_loss],
        depends_on=["daily_prediction"],
        enabled=False,
    )
    step = thor.training.TrainingStep(
        "demand_forecast",
        phases=[daily_phase, aggregate_phase],
        optimizer=optimizer,
        update_parameters=[fc.get_parameter_reference("weights")],
    )
    program = thor.training.TrainingProgram([step])

    daily_only = program.compile(placed)
    assert len(daily_only) == 1
    assert daily_only[0].get_active_phase_names() == ["daily_prediction"]
    assert len(daily_only[0].get_resolved_loss_roots()) == 1

    aggregate_phase.enable()
    joint = program.compile(placed)
    assert len(joint) == 1
    assert joint[0].get_active_phase_names() == ["daily_prediction", "aggregate_prediction"]
    assert len(joint[0].get_resolved_loss_roots()) == 2

    daily_phase.disable()
    with pytest.raises(RuntimeError, match="depends on disabled phase 'daily_prediction'"):
        program.compile(placed)

@pytest.mark.cuda
def test_training_program_compile_skips_disabled_step_even_if_it_has_invalid_phase_dependencies():
    n = thor.Network("test_training_program_compile_skips_disabled_bad_dependency")
    x = thor.layers.NetworkInput(n, "input", [3], thor.DataType.fp32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01)
    fc = thor.layers.FullyConnected(
        n,
        x.get_feature_output(),
        1,
        True,
        activation=None,
        weights_optimizer=optimizer,
        biases_optimizer=optimizer,
    )
    loss = fc.get_feature_output()
    thor.layers.NetworkOutput(n, "scores", loss, thor.DataType.fp32)
    placed = n.place(2, inference_only=False)

    bad_phase = thor.training.TrainingPhase("aggregate_prediction", loss_roots=[loss], depends_on=["missing_daily"])
    skipped_step = thor.training.TrainingStep(
        "skipped_bad_dependency",
        phases=[bad_phase],
        optimizer=optimizer,
        update_parameters=[fc.get_parameter_reference("weights")],
        enabled=False,
    )
    valid_step = thor.training.TrainingStep(
        "valid_reference",
        [loss],
        optimizer=optimizer,
        update_parameters=[fc.get_parameter_reference("weights")],
    )
    program = thor.training.TrainingProgram([skipped_step, valid_step])

    executables = program.compile(placed)
    assert len(executables) == 1
    assert executables[0].name == "valid_reference"

    skipped_step.enable()
    with pytest.raises(RuntimeError, match="enabled phase 'aggregate_prediction' depends on missing phase 'missing_daily'"):
        program.compile(placed)

