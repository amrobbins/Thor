import json

import pytest
import thor


def _make_fc_network(name: str):
    n = thor.Network(name)
    x = thor.layers.NetworkInput(n, "input", [3], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(n, x.get_feature_output(), 2, True, activation=None)
    return n, fc


def _make_phase_network(
    name: str,
    *,
    input_name: str = "examples",
    output_name: str = "prediction",
    input_external: bool = True,
    output_external: bool = True,
    with_loss: bool = True,
):
    network = thor.Network(name)
    x = thor.layers.NetworkInput(
        network,
        input_name,
        [1],
        thor.DataType.fp32,
        external=input_external,
    )
    prediction = x.get_feature_output()
    loss_root = None
    if with_loss:
        labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32)
        loss = thor.losses.MSE(
            network,
            prediction,
            labels.get_feature_output(),
            thor.DataType.fp32,
        )
        loss_root = loss.get_loss()
        thor.layers.NetworkOutput(network, f"{output_name}_loss", loss_root, thor.DataType.fp32)
    thor.layers.NetworkOutput(
        network,
        output_name,
        prediction,
        thor.DataType.fp32,
        external=output_external,
    )
    return network, loss_root, prediction


def _make_phase(
    name: str,
    *,
    input_name: str = "examples",
    output_name: str | None = None,
    input_external: bool = True,
    output_external: bool = True,
    with_loss: bool = True,
    enabled: bool = True,
):
    phase_output_name = output_name or name
    network, _loss_root, prediction = _make_phase_network(
        f"{name}_network",
        input_name=input_name,
        output_name=phase_output_name,
        input_external=input_external,
        output_external=output_external,
        with_loss=with_loss,
    )
    phase = thor.training.TrainingPhase(name, network=network, enabled=enabled)
    phase_loss_roots = phase.get_loss_roots()
    canonical_loss_root = phase_loss_roots[0] if phase_loss_roots else None
    return phase, canonical_loss_root, prediction

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
    phase_network, _loss_root, forecast = _make_phase_network(
        "daily_phase_network",
        output_name="forecast",
    )

    phase = thor.training.TrainingPhase(
        "daily_prediction",
        network=phase_network,
        enabled=False,
    )

    assert phase.name == "daily_prediction"
    assert phase.get_network() is not None
    assert phase.get_network().get_network_name() == "daily_phase_network"
    assert not phase.enabled
    assert not phase.is_enabled()
    assert len(phase.get_loss_roots()) == 1
    assert set(phase.get_outputs().keys()) == {"forecast", "forecast_loss"}
    assert phase.get_outputs()["forecast"] == forecast

    phase.enable()
    assert phase.enabled
    phase.enabled = False
    assert not phase.is_enabled()
    phase.set_enabled(True)
    assert phase.is_enabled()

    arch = json.loads(phase.get_architecture_json())
    assert arch["version"] == "1.1.0"
    assert arch["name"] == "daily_prediction"
    assert arch["enabled"] is True
    assert arch["network"]["name"] == "daily_phase_network"

    restored = thor.training.TrainingPhase.deserialize(phase.get_architecture_json())
    assert restored.name == "daily_prediction"
    assert restored.enabled
    assert restored.get_network() is not None
    assert restored.get_network().get_network_name() == "daily_phase_network"
    assert len(restored.get_loss_roots()) == 1
    assert set(restored.get_outputs().keys()) == {"forecast", "forecast_loss"}


def test_training_phase_external_flags_derive_local_and_external_outputs():
    phase_network, _, hidden = _make_phase_network(
        "feature_preprocessing_network",
        output_name="hidden",
        output_external=False,
        with_loss=False,
    )

    phase = thor.training.TrainingPhase("feature_preprocessing", network=phase_network, enabled=True)

    assert phase.name == "feature_preprocessing"
    assert phase.get_network() is not None
    assert phase.get_loss_roots() == []
    assert set(phase.get_outputs().keys()) == {"hidden"}
    assert phase.get_outputs()["hidden"] == hidden

    arch = json.loads(phase.get_architecture_json())
    assert arch["version"] == "1.1.0"
    assert arch["network"]["name"] == "feature_preprocessing_network"

    restored = thor.training.TrainingPhase.deserialize(phase.get_architecture_json())
    assert restored.get_network() is not None
    assert restored.get_network().get_network_name() == "feature_preprocessing_network"
    assert set(restored.get_outputs().keys()) == {"hidden"}


def test_training_phase_construction_and_deserialization_errors_are_exposed():
    phase_network, _, _ = _make_phase_network("enabled_phase_network", output_name="prediction")

    with pytest.raises(RuntimeError, match="requires a non-empty name"):
        thor.training.TrainingPhase("", network=phase_network)

    enabled_phase = thor.training.TrainingPhase("enabled_phase", network=phase_network, enabled=True)
    disabled_network, _, _ = _make_phase_network("disabled_phase_network", output_name="prediction")
    disabled_phase = thor.training.TrainingPhase("disabled_phase", network=disabled_network, enabled=False)
    assert thor.training.TrainingPhase.deserialize(enabled_phase.get_architecture_json()).enabled
    assert not thor.training.TrainingPhase.deserialize(disabled_phase.get_architecture_json()).enabled

    bad_arch = json.loads(enabled_phase.get_architecture_json())
    bad_arch["version"] = "2.0.0"
    with pytest.raises(RuntimeError, match="Unsupported TrainingPhase version: 2.0.0"):
        thor.training.TrainingPhase.deserialize(json.dumps(bad_arch))

def test_training_step_phase_constructor_collects_enabled_phase_loss_roots():
    daily_phase, daily_loss, _ = _make_phase("daily_prediction")
    aggregate_phase, aggregate_loss, _ = _make_phase(
        "aggregate_prediction",
        input_name="daily_prediction",
        input_external=False,
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
    daily_phase.disable()
    assert step.get_active_phase_names() == ["aggregate_prediction"]
    assert step.get_active_loss_roots() == [aggregate_loss]


def test_training_step_active_phase_view_skips_disabled_phases_and_steps():
    daily_phase, daily_loss, _ = _make_phase("daily_prediction")
    disabled_aggregate, aggregate_loss, _ = _make_phase(
        "aggregate_prediction",
        input_name="missing_daily",
        input_external=False,
        enabled=False,
    )
    step = thor.training.TrainingStep("demand_forecast", phases=[daily_phase, disabled_aggregate])
    assert step.get_active_phase_names() == ["daily_prediction"]
    assert step.get_active_loss_roots() == [daily_loss]

    disabled_aggregate.enable()
    assert step.get_active_phase_names() == ["daily_prediction", "aggregate_prediction"]
    assert step.get_active_loss_roots() == [daily_loss, aggregate_loss]

    disabled_daily, _, _ = _make_phase("daily_prediction", enabled=False)
    enabled_aggregate, _, _ = _make_phase(
        "aggregate_prediction",
        input_name="daily_prediction",
        input_external=False,
        enabled=True,
    )
    disabled_step = thor.training.TrainingStep(
        "demand_forecast",
        phases=[disabled_daily, enabled_aggregate],
        enabled=False,
    )
    assert disabled_step.get_active_phase_names() == []
    assert disabled_step.get_active_loss_roots() == []


def test_training_step_phase_names_are_independent_of_inferred_wiring_order():
    aggregate_phase, aggregate_loss, _ = _make_phase(
        "aggregate_prediction",
        input_name="daily_prediction",
        input_external=False,
    )
    daily_phase, daily_loss, _ = _make_phase("daily_prediction")

    step = thor.training.TrainingStep("demand_forecast", phases=[aggregate_phase, daily_phase])

    # Explicit phase dependency lists are gone. TrainingStep preserves the user's phase
    # list for the logical API view; phase-graph composition topologically orders
    # execution later from NetworkInput/NetworkOutput names.
    assert step.get_active_phase_names() == ["aggregate_prediction", "daily_prediction"]
    assert step.get_active_loss_roots() == [aggregate_loss, daily_loss]

def test_training_step_forward_only_phase_can_be_active_without_contributing_loss_roots():
    preprocessing, _, _ = _make_phase(
        "feature_preprocessing",
        output_name="features",
        output_external=False,
        with_loss=False,
    )
    daily, daily_loss, _ = _make_phase(
        "daily_prediction",
        input_name="features",
        input_external=False,
    )
    aggregate, aggregate_loss, _ = _make_phase(
        "aggregate_prediction",
        input_name="daily_prediction",
        input_external=False,
        enabled=False,
    )
    step = thor.training.TrainingStep("demand_forecast", phases=[preprocessing, daily, aggregate])

    assert step.get_active_phase_names() == ["feature_preprocessing", "daily_prediction"]
    assert step.get_active_loss_roots() == [daily_loss]

    aggregate.enable()
    assert step.get_active_phase_names() == ["feature_preprocessing", "daily_prediction", "aggregate_prediction"]
    assert step.get_active_loss_roots() == [daily_loss, aggregate_loss]


def test_training_step_phase_serialization_preserves_network_backed_phases():
    daily, _, _ = _make_phase("daily_prediction")
    aggregate, _, _ = _make_phase(
        "aggregate_prediction",
        input_name="daily_prediction",
        input_external=False,
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
    assert all(phase.get_network() is not None for phase in restored.get_phases())
    assert restored.get_active_phase_names() == ["daily_prediction"]

    bad_arch = json.loads(step.get_architecture_json())
    bad_arch["version"] = "9.9.9"
    with pytest.raises(RuntimeError, match="Unsupported TrainingStep version: 9.9.9"):
        thor.training.TrainingStep.deserialize(json.dumps(bad_arch))


def test_training_step_constructor_requires_loss_roots_or_phases_but_not_both():
    loss = thor.Tensor([1], thor.DataType.fp32)
    phase, _, _ = _make_phase("daily_prediction")

    with pytest.raises(ValueError, match="exactly one of loss_roots or phases"):
        thor.training.TrainingStep("bad")

    with pytest.raises(ValueError, match="exactly one of loss_roots or phases"):
        thor.training.TrainingStep("bad", [loss], phases=[phase])


def test_training_program_holds_training_steps_by_reference_in_python():
    daily_phase, _, _ = _make_phase("daily_prediction")
    aggregate_phase, _, _ = _make_phase(
        "aggregate_prediction",
        input_name="daily_prediction",
        input_external=False,
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

def test_training_program_reflects_python_phase_enable_mutation():
    daily_phase, daily_loss, _ = _make_phase("daily_prediction")
    aggregate_phase, aggregate_loss, _ = _make_phase(
        "aggregate_prediction",
        input_name="daily_prediction",
        input_external=False,
        enabled=False,
    )
    step = thor.training.TrainingStep(
        "demand_forecast",
        phases=[daily_phase, aggregate_phase],
    )
    program = thor.training.TrainingProgram([step])

    assert program.get_step(0).get_active_phase_names() == ["daily_prediction"]
    assert program.get_step(0).get_active_loss_roots() == [daily_loss]

    aggregate_phase.enable()
    assert program.get_step(0).get_active_phase_names() == ["daily_prediction", "aggregate_prediction"]
    assert program.get_step(0).get_active_loss_roots() == [daily_loss, aggregate_loss]

    daily_phase.disable()
    assert program.get_step(0).get_active_phase_names() == ["aggregate_prediction"]
    assert program.get_step(0).get_active_loss_roots() == [aggregate_loss]

@pytest.mark.cuda
def test_training_program_compile_skips_disabled_network_backed_step_even_if_not_placeable():
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

    bad_phase_network = thor.Network("aggregate_prediction_missing_compile_phase_network")
    missing_input = thor.layers.NetworkInput(bad_phase_network, "missing_daily", [1], thor.DataType.fp32, external=False)
    labels = thor.layers.NetworkInput(bad_phase_network, "labels", [1], thor.DataType.fp32)
    bad_loss = thor.losses.MSE(bad_phase_network, missing_input.get_feature_output(), labels.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(bad_phase_network, "aggregate_prediction", bad_loss.get_loss(), thor.DataType.fp32)
    bad_phase = thor.training.TrainingPhase("aggregate_prediction", network=bad_phase_network)
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
    with pytest.raises(RuntimeError):
        program.compile(placed)

