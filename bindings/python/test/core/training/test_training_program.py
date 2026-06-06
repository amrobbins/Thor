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
    step = thor.training.TrainingStep("same", [loss], optimizer=sgd, update_parameters=[thor.ParameterReference(1, "weights")])
    program = thor.training.TrainingProgram([step])

    with pytest.raises(RuntimeError, match="already contains a step named"):
        program.add_step(step)


def test_training_step_rejects_updates_without_optimizer():
    loss = thor.Tensor([1], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="requires an optimizer"):
        thor.training.TrainingStep("bad", [loss], update_parameters=[thor.ParameterReference(1, "weights")])


def test_training_program_rejects_empty_programs():
    with pytest.raises(RuntimeError, match="at least one TrainingStep"):
        thor.training.TrainingProgram([])

    program = thor.training.TrainingProgram()
    assert not program.is_initialized()
    with pytest.raises(RuntimeError, match="at least one TrainingStep"):
        program.get_architecture_json()

    with pytest.raises(RuntimeError, match="at least one TrainingStep"):
        thor.training.TrainingProgram.deserialize(json.dumps({"version": "1.0.0", "steps": []}))


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
