import ctypes
import json
import os
import re
import sys

import numpy as np
import pytest
import thor
from integration_flags import integration_flag_enabled, integration_skip_reason

RUN_TRAINING_INTEGRATION = integration_flag_enabled("THOR_RUN_TRAINING_INTEGRATION")
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RUN_STATUS_RE = re.compile(r"INFO runs\[(?P<run>[^\]|]+)(?:\|[^\]]+)?\]:.*\bstatus=(?P<status>completed|failed|cancelled|interrupted|oom|running|starting|not_started)\b")


def _flush_native_stdio_for_capture():
    ctypes.CDLL(None).fflush(None)


def _captured_terminal_statuses(captured_text: str):
    plain_text = _ANSI_RE.sub("", captured_text)
    seen = {}
    for match in _RUN_STATUS_RE.finditer(plain_text):
        seen[match.group("run")] = match.group("status")
    return seen


def _fit_runs_and_capture_text(runs, capfd, *, epochs: int, test_loader=None):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    results = runs.fit(epochs=epochs, test_loader=test_loader)
    _flush_native_stdio_for_capture()
    captured = capfd.readouterr()
    captured_text = captured.out + captured.err

    # Mirror the trainer integration tests: native fprintf output is captured for
    # assertions, then replayed outside capture so failures still show useful logs.
    with capfd.disabled():
        sys.stdout.write(captured_text)
        sys.stdout.flush()

    return results, captured_text


def _regression_arrays(*, dtype=np.float32):
    x = np.array(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([[-1.0], [1.0], [1.0], [-1.0]], dtype=np.float32)
    return np.ascontiguousarray(x, dtype=dtype), np.ascontiguousarray(y, dtype=dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _regression_one_batch_loader(*, dtype=np.float32):
    x, y = _regression_arrays(dtype=dtype)
    loader_cls = thor.training.NumpyFloat16BatchLoader if dtype == np.float16 else thor.training.NumpyFloat32BatchLoader
    return loader_cls(
        x,
        y,
        x,
        y,
        batch_size=4,
        example_input_name="examples",
        label_input_name="labels",
        dataset_name="training_runs_regression_one_batch",
    )


def _categorical_arrays(*, dtype=np.float32):
    x = np.array(
        [
            [2.0, -1.0],
            [-1.0, 2.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=np.float32,
    )
    y = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(x, dtype=dtype), np.ascontiguousarray(y, dtype=dtype)


def _categorical_one_batch_loader(*, dtype=np.float32):
    x, y = _categorical_arrays(dtype=dtype)
    loader_cls = thor.training.NumpyFloat16BatchLoader if dtype == np.float16 else thor.training.NumpyFloat32BatchLoader
    return loader_cls(
        x,
        y,
        x,
        y,
        batch_size=4,
        example_input_name="examples",
        label_input_name="class_targets",
        dataset_name="training_runs_categorical_one_batch",
    )


def _categorical_mixed_accuracy_arrays(*, dtype=np.float32):
    x, _ = _categorical_arrays(dtype=np.float32)
    y = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(x, dtype=dtype), np.ascontiguousarray(y, dtype=dtype)


def _categorical_mixed_accuracy_one_batch_loader(*, dtype=np.float32):
    x, y = _categorical_mixed_accuracy_arrays(dtype=dtype)
    loader_cls = thor.training.NumpyFloat16BatchLoader if dtype == np.float16 else thor.training.NumpyFloat32BatchLoader
    return loader_cls(
        x,
        y,
        x,
        y,
        batch_size=4,
        example_input_name="examples",
        label_input_name="class_targets",
        dataset_name="training_runs_categorical_mixed_accuracy_one_batch",
    )


def _softmax_cross_entropy_and_accuracy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    losses = -np.sum(labels * np.log(np.maximum(probabilities, 1.0e-12)), axis=-1)
    accuracy = np.mean(np.argmax(scores, axis=-1) == np.argmax(labels, axis=-1))
    return float(np.mean(losses)), float(accuracy)


def _build_tiny_regressor(name: str):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32)

    predictions = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        1,
        True,
        activation=None,
        weights_initializer=thor.initializers.UniformRandom(0.0, 0.0),
        biases_initializer=thor.initializers.UniformRandom(0.0, 0.0),
    )
    loss = thor.losses.MSE(
        network,
        predictions.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
        loss_weight=2.0,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction", predictions.get_feature_output(), thor.DataType.fp32)
    return network


def _build_tiny_classifier(name: str):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(network, "class_targets", [2], thor.DataType.fp32)

    scores = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        2,
        True,
        activation=None,
        weights_initializer=thor.initializers.UniformRandom(0.0, 0.0),
        biases_initializer=thor.initializers.UniformRandom(0.0, 0.0),
    )
    loss = thor.losses.CategoricalCrossEntropy(
        network,
        scores.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "scores", scores.get_feature_output(), thor.DataType.fp32)
    return network


def _make_tiny_categorical_trainer(
    name: str,
    *,
    save_model_dir=None,
    save_model_overwrite=False,
):
    return thor.training.Trainer(
        _build_tiny_classifier(name),
        _categorical_one_batch_loader(),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
        save_model_dir=save_model_dir,
        save_model_overwrite=save_model_overwrite,
    )


def _build_signature_only_network(name: str, *, input_dtype=thor.DataType.fp32, output_dtype=thor.DataType.fp32):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], input_dtype)
    thor.layers.NetworkOutput(network, "prediction", examples.get_feature_output(), output_dtype)
    return network


def _make_signature_only_trainer(name: str, *, input_dtype=thor.DataType.fp32, output_dtype=thor.DataType.fp32):
    return thor.training.Trainer(
        _build_signature_only_network(name, input_dtype=input_dtype, output_dtype=output_dtype),
        _regression_one_batch_loader(),
    )


def _make_tiny_regression_trainer(
    name: str,
    *,
    optimizer=True,
    optimizer_obj=None,
    save_model_dir=None,
    save_model_overwrite=False,
    check_best_model_every_epochs=1,
    min_early_completion_epochs=0,
    model_selection_score=None,
    restart_conditions=None,
    early_completion_policies=None,
):
    if optimizer_obj is None:
        optimizer_obj = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0) if optimizer else None
    return thor.training.Trainer(
        _build_tiny_regressor(name),
        _regression_one_batch_loader(),
        optimizer=optimizer_obj,
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
        save_model_dir=save_model_dir,
        save_model_overwrite=save_model_overwrite,
        check_best_model_every_epochs=check_best_model_every_epochs,
        min_early_completion_epochs=min_early_completion_epochs,
        model_selection_score=model_selection_score,
        restart_conditions=restart_conditions,
        early_completion_policies=early_completion_policies,
    )


def test_training_runs_binding_rejects_empty_runs():
    with pytest.raises(RuntimeError, match="at least one run"):
        thor.training.TrainingRuns([])


def test_training_runs_binding_rejects_invalid_failure_policy():
    with pytest.raises(ValueError, match="failure_policy"):
        thor.training.TrainingRuns([], failure_policy="fail_fast")


def test_training_runs_binding_rejects_invalid_max_parallel_runs():
    with pytest.raises(RuntimeError, match="maxParallelRuns"):
        thor.training.TrainingRuns([], max_parallel_runs=0)


def test_training_runs_binding_accepts_reported_losses():
    trainer = _make_tiny_regression_trainer("training_runs_binding_reported_losses")

    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "tiny_ensemble")],
        reported_losses={"tiny_ensemble": ["loss"]},
    )

    assert runs.reported_losses["tiny_ensemble"] == ["loss"]
    assert not hasattr(runs, "ensemble_metrics")
    assert not hasattr(thor.training, "MetricSpec")


def test_training_runs_binding_rejects_invalid_reported_losses():
    trainer = _make_tiny_regression_trainer("training_runs_binding_invalid_reported_losses")

    with pytest.raises(RuntimeError, match="requested reported loss .*missing"):
        thor.training.TrainingRuns(
            [("fold_0", trainer, "tiny_ensemble")],
            reported_losses={"tiny_ensemble": ["missing"]},
        )

    with pytest.raises(TypeError, match="reported_losses"):
        thor.training.TrainingRuns(
            [("fold_0", trainer, "tiny_ensemble")],
            reported_losses={"tiny_ensemble": [{"name": "loss"}]},
        )


def test_trainer_binding_accepts_pathlike_save_model_dir_for_training_runs_artifact(tmp_path):
    trainer = _make_tiny_regression_trainer(
        "training_runs_pathlike_save_model_dir",
        save_model_dir=tmp_path / "model_artifact",
    )

    runs = thor.training.TrainingRuns([("fold_0", trainer, "tiny_ensemble")])

    assert runs is not None


def test_trainer_binding_rejects_empty_save_model_dir():
    with pytest.raises((ValueError, RuntimeError), match="save_model_dir must not be empty"):
        _make_tiny_regression_trainer("training_runs_empty_save_model_dir", save_model_dir="")


def test_trainer_fit_rejects_existing_save_model_dir_before_training(tmp_path):
    save_dir = tmp_path / "existing_model_artifact"
    save_dir.mkdir()
    trainer = _make_tiny_regression_trainer(
        "training_runs_existing_save_model_dir_single_trainer",
        save_model_dir=save_dir,
    )

    with pytest.raises(RuntimeError, match="save_model_dir.*already exists"):
        trainer.fit(epochs=1)


def test_training_runs_fit_rejects_existing_save_model_dir_before_training(tmp_path):
    existing_save_dir = tmp_path / "existing_fold_0"
    existing_save_dir.mkdir()
    trainer0 = _make_tiny_regression_trainer(
        "training_runs_existing_save_model_dir_fold_0",
        save_model_dir=existing_save_dir,
    )
    trainer1 = _make_tiny_regression_trainer(
        "training_runs_existing_save_model_dir_fold_1",
        save_model_dir=tmp_path / "fresh_fold_1",
    )
    runs = thor.training.TrainingRuns([("fold_0", trainer0), ("fold_1", trainer1)])

    with pytest.raises(RuntimeError, match="save_model_dir.*already exists"):
        runs.fit(epochs=1)


def test_trainer_binding_accepts_best_model_candidate_cadence_and_min_early_completion_epochs():
    trainer = _make_tiny_regression_trainer(
        "training_runs_best_candidate_cadence",
        check_best_model_every_epochs=3,
        min_early_completion_epochs=7,
    )

    assert trainer is not None
    assert trainer.min_early_completion_epochs == 7
    assert trainer.completed_training_epochs == 0


def test_trainer_binding_rejects_zero_best_model_candidate_cadence():
    with pytest.raises(RuntimeError, match="check_best_model_every_epochs"):
        _make_tiny_regression_trainer(
            "training_runs_best_candidate_invalid_cadence",
            check_best_model_every_epochs=0,
        )


def test_trainer_binding_accepts_custom_model_selection_score():
    trainer = _make_tiny_regression_trainer(
        "training_runs_custom_model_selection_score",
        model_selection_score=lambda validation_loss, training_loss, epoch: validation_loss if validation_loss is not None else training_loss,
    )

    assert trainer is not None


def test_trainer_binding_rejects_non_callable_model_selection_score():
    with pytest.raises(TypeError, match="model_selection_score"):
        _make_tiny_regression_trainer(
            "training_runs_custom_model_selection_score_invalid",
            model_selection_score=3.0,
        )


def test_trainer_restart_condition_binding_defaults():
    condition = thor.training.RestartCondition()

    assert isinstance(condition, thor.training.TrainingRestartCondition)
    assert condition.progress_check_epochs == 3
    assert condition.progress_improvement_min_percentage == 5.0
    assert condition.max_restarts == 5

    trainer = _make_tiny_regression_trainer(
        "trainer_restart_condition_binding",
        restart_conditions=[thor.training.RestartCondition(progress_check_epochs=2, progress_improvement_min_percentage=10.0)],
    )

    assert trainer is not None


def test_training_runs_restart_condition_binding_defaults_and_validation():
    condition = thor.training.RestartPolicy(run_name="fold_0")

    assert isinstance(condition, thor.training.TrainingRunsRestartPolicy)
    assert condition.run_name == "fold_0"
    assert condition.ensemble_group is None
    assert condition.progress_check_epochs == 3
    assert condition.progress_improvement_min_percentage == 5.0
    assert condition.max_restarts == 5

    trainer = _make_tiny_regression_trainer("training_runs_restart_condition_unknown")
    with pytest.raises(RuntimeError, match="unknown run_name"):
        thor.training.TrainingRuns(
            [("fold_0", trainer)],
            failure_policy="continue",
            restart_conditions=[thor.training.RestartPolicy(run_name="missing")],
        )


def test_training_runs_restart_condition_accepts_ensemble_group_target():
    trainer = _make_tiny_regression_trainer("training_runs_restart_condition_group")
    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "tiny_ensemble")],
        failure_policy="continue",
        restart_conditions=[
            thor.training.RestartPolicy(
                ensemble_group="tiny_ensemble",
                progress_check_epochs=2,
                progress_improvement_min_percentage=10.0,
                max_restarts=1,
            )
        ],
    )

    assert runs is not None

def test_training_runs_restart_condition_accepts_multiple_conditions_for_same_group():
    trainer = _make_tiny_regression_trainer("training_runs_restart_condition_group_multiple")
    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "tiny_ensemble")],
        failure_policy="continue",
        restart_conditions=[
            thor.training.RestartPolicy(
                ensemble_group="tiny_ensemble",
                progress_check_epochs=3,
                progress_improvement_min_percentage=5.0,
                max_restarts=5,
            ),
            thor.training.RestartPolicy(
                ensemble_group="tiny_ensemble",
                progress_check_epochs=15,
                progress_improvement_min_percentage=20.0,
                max_restarts=5,
            ),
        ],
    )

    assert runs is not None


def test_training_runs_restart_condition_accepts_multiple_conditions_for_same_run():
    trainer = _make_tiny_regression_trainer("training_runs_restart_condition_run_multiple")
    runs = thor.training.TrainingRuns(
        [("fold_0", trainer)],
        failure_policy="continue",
        restart_conditions=[
            thor.training.RestartPolicy(run_name="fold_0", progress_check_epochs=3),
            thor.training.RestartPolicy(run_name="fold_0", progress_check_epochs=15, progress_improvement_min_percentage=20.0),
        ],
    )

    assert runs is not None


def test_trainer_early_completion_policy_binding_accepts_callable():
    policy = thor.training.EarlyCompletionPolicy(
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch - best_epoch >= 1
    )
    assert isinstance(policy, thor.training.TrainingEarlyCompletionPolicy)

    trainer = _make_tiny_regression_trainer(
        "trainer_early_completion_policy_binding",
        early_completion_policies=[policy],
    )

    assert trainer is not None


def test_training_runs_early_completion_rule_binding_defaults_and_validation():
    rule = thor.training.EarlyCompletionRule(
        lambda current_score, best_score, current_epoch, best_epoch: False,
        run_name="fold_0",
    )

    assert isinstance(rule, thor.training.TrainingRunsEarlyCompletionRule)
    assert rule.run_name == "fold_0"
    assert rule.ensemble_group is None

    trainer = _make_tiny_regression_trainer("training_runs_early_completion_rule_unknown")
    with pytest.raises(RuntimeError, match="unknown run_name"):
        thor.training.TrainingRuns(
            [("fold_0", trainer)],
            failure_policy="continue",
            early_completion_rules=[
                thor.training.EarlyCompletionRule(
                    lambda current_score, best_score, current_epoch, best_epoch: False,
                    run_name="missing",
                )
            ],
        )


def test_training_runs_early_completion_rule_accepts_ensemble_group_target():
    trainer = _make_tiny_regression_trainer("training_runs_early_completion_rule_group")
    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "tiny_ensemble")],
        failure_policy="continue",
        early_completion_rules=[
            thor.training.EarlyCompletionRule(
                lambda current_score, best_score, current_epoch, best_epoch: False,
                ensemble_group="tiny_ensemble",
            )
        ],
    )

    assert runs is not None


def test_training_runs_result_status_names_are_exposed():
    assert thor.training.TrainingRunStatus.completed.name == "completed"
    assert thor.training.TrainingRunCompletionReason.early_completed.name == "early_completed"
    assert thor.training.TrainingRunsFailurePolicy.cancel_siblings.name == "cancel_siblings"


def test_training_runs_binding_rejects_duplicate_save_model_dirs(tmp_path):
    shared_dir = tmp_path / "shared_checkpoint"
    equivalent_shared_dir = tmp_path / "nested" / ".." / "shared_checkpoint"
    trainer0 = _make_tiny_regression_trainer(
        "training_runs_duplicate_save_dir_0",
        save_model_dir=str(shared_dir),
    )
    trainer1 = _make_tiny_regression_trainer(
        "training_runs_duplicate_save_dir_1",
        save_model_dir=str(equivalent_shared_dir),
    )

    with pytest.raises(RuntimeError, match="save_model_dir collision"):
        thor.training.TrainingRuns([("fold_0", trainer0), ("fold_1", trainer1)])


def test_training_runs_fit_rejects_ensemble_without_trainer_save_model_dir():
    trainer = _make_tiny_regression_trainer("training_runs_missing_ensemble_save_dir")
    runs = thor.training.TrainingRuns([("fold_0", trainer, "tiny_ensemble")])

    with pytest.raises(RuntimeError, match="save_model_dir"):
        runs.fit(epochs=1)


def test_training_runs_binding_rejects_invalid_ensemble_weight():
    trainer = _make_tiny_regression_trainer("training_runs_invalid_ensemble_weight")

    with pytest.raises(RuntimeError, match="ensemble_weight"):
        thor.training.TrainingRuns([("fold_0", trainer, "tiny_ensemble", 0.0)])


def test_training_runs_rejects_ensemble_input_dtype_mismatch_before_fit():
    trainer0 = _make_signature_only_trainer("training_runs_input_dtype_0", input_dtype=thor.DataType.fp32)
    trainer1 = _make_signature_only_trainer("training_runs_input_dtype_1", input_dtype=thor.DataType.fp16)

    with pytest.raises(RuntimeError, match="incompatible input signatures"):
        thor.training.TrainingRuns([("fold_0", trainer0, "tiny_ensemble"), ("fold_1", trainer1, "tiny_ensemble")])


def test_training_runs_rejects_ensemble_output_dtype_mismatch_before_fit():
    trainer0 = _make_signature_only_trainer("training_runs_output_dtype_0", output_dtype=thor.DataType.fp32)
    trainer1 = _make_signature_only_trainer("training_runs_output_dtype_1", output_dtype=thor.DataType.fp16)

    with pytest.raises(RuntimeError, match="incompatible output signatures"):
        thor.training.TrainingRuns([("fold_0", trainer0, "tiny_ensemble"), ("fold_1", trainer1, "tiny_ensemble")])


def test_training_runs_accepts_min_successful_models_for_known_ensemble_group():
    trainer0 = _make_signature_only_trainer("training_runs_min_success_group_0")
    trainer1 = _make_signature_only_trainer("training_runs_min_success_group_1")

    runs = thor.training.TrainingRuns(
        [("fold_0", trainer0, "tiny_ensemble"), ("fold_1", trainer1, "tiny_ensemble")],
        min_successful_models={"tiny_ensemble": 1},
    )

    assert runs is not None


def test_training_runs_rejects_min_successful_models_unknown_group():
    trainer0 = _make_signature_only_trainer("training_runs_min_success_unknown_0")
    trainer1 = _make_signature_only_trainer("training_runs_min_success_unknown_1")

    with pytest.raises(RuntimeError, match="unknown ensemble_group"):
        thor.training.TrainingRuns(
            [("fold_0", trainer0, "tiny_ensemble"), ("fold_1", trainer1, "tiny_ensemble")],
            min_successful_models={"tinny_ensemble": 1},
        )


def test_training_runs_rejects_min_successful_models_larger_than_group():
    trainer0 = _make_signature_only_trainer("training_runs_min_success_too_large_0")
    trainer1 = _make_signature_only_trainer("training_runs_min_success_too_large_1")

    with pytest.raises(RuntimeError, match="only has 2 member"):
        thor.training.TrainingRuns(
            [("fold_0", trainer0, "tiny_ensemble"), ("fold_1", trainer1, "tiny_ensemble")],
            min_successful_models={"tiny_ensemble": 3},
        )


def test_training_runs_rejects_min_successful_models_non_dict():
    trainer0 = _make_signature_only_trainer("training_runs_min_success_non_dict_0")
    trainer1 = _make_signature_only_trainer("training_runs_min_success_non_dict_1")

    with pytest.raises(TypeError, match="min_successful_models"):
        thor.training.TrainingRuns(
            [("fold_0", trainer0, "tiny_ensemble"), ("fold_1", trainer1, "tiny_ensemble")],
            min_successful_models=1,
        )


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_evaluates_saved_adam_model_on_test_loader(capfd, tmp_path):
    trainer = _make_tiny_regression_trainer(
        "training_runs_saved_adam_test_eval",
        optimizer_obj=thor.optimizers.Adam(),
        save_model_dir=tmp_path / "adam_model",
        save_model_overwrite=True,
    )
    runs = thor.training.TrainingRuns([("fold_0", trainer, "tiny_ensemble")])

    results, captured_text = _fit_runs_and_capture_text(
        runs,
        capfd,
        epochs=1,
        test_loader=_regression_one_batch_loader(),
    )

    assert results.all_completed()
    assert results["fold_0"].final_test_loss is not None
    plain_text = _ANSI_RE.sub("", captured_text)
    assert re.search(r"INFO runs\[fold_0\|tiny_ensemble\]:.*test_loss=", plain_text)




def _training_selection_metadata(save_dir):
    with open(save_dir / "training_selection_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)

def _prediction_from_saved_tiny_regressor(save_dir, network_name: str):
    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    x, _ = _regression_arrays(dtype=np.float32)
    outputs = placed.infer({"examples": _cpu_tensor(x, thor.DataType.fp32)})
    return np.array(outputs["prediction"].numpy(), copy=True)


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_saved_trained_model_load_place_inference_only_infer_sequence(tmp_path):
    save_dir = tmp_path / "saved_trained_inference_sequence"
    trainer = _make_tiny_regression_trainer(
        "saved_trained_inference_sequence",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
    )

    result = trainer.fit(1)

    assert result.status == "completed"
    assert result.best_epoch == 1
    assert save_dir.exists()

    loaded = thor.Network("saved_trained_inference_sequence")
    loaded.load(str(save_dir))
    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    assert isinstance(placed, thor.runtime.PlacedNetwork)
    assert set(placed.get_network_input_names()) == {"examples"}

    x, _ = _regression_arrays(dtype=np.float32)
    outputs = placed.infer({"examples": _cpu_tensor(x, thor.DataType.fp32)})

    assert set(outputs) == {"prediction"}
    prediction = np.array(outputs["prediction"].numpy(), copy=True)
    assert prediction.shape == (4, 1)
    assert np.all(np.isfinite(prediction))
    assert not np.allclose(prediction, 0.0, atol=1e-7)


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_saved_training_graph_inference_prunes_loss_and_label_only_inputs(tmp_path):
    save_dir = tmp_path / "saved_training_graph_prunes_loss_labels"
    trainer = _make_tiny_regression_trainer(
        "saved_training_graph_prunes_loss_labels",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
    )
    trainer.fit(1)

    loaded = thor.Network("saved_training_graph_prunes_loss_labels")
    loaded.load(str(save_dir))
    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    assert set(placed.get_network_input_names()) == {"examples"}

    x, _ = _regression_arrays(dtype=np.float32)
    outputs = placed.infer({"examples": _cpu_tensor(x, thor.DataType.fp32)})

    assert set(outputs) == {"prediction"}
    prediction = np.array(outputs["prediction"].numpy(), copy=True)
    assert prediction.shape == (4, 1)
    assert np.all(np.isfinite(prediction))


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_trainer_best_candidate_snapshot_contains_trained_weights(tmp_path):
    save_dir = tmp_path / "best_candidate_model"
    trainer = _make_tiny_regression_trainer(
        "trainer_best_candidate_snapshot_trained_weights",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
    )

    trainer.fit(1)

    prediction = _prediction_from_saved_tiny_regressor(save_dir, "trainer_best_candidate_snapshot_trained_weights")

    assert save_dir.exists()
    assert not np.allclose(prediction, 0.0, atol=1e-7)


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_trainer_custom_model_selection_score_controls_saved_candidate(tmp_path):
    first_epoch_dir = tmp_path / "custom_score_first_epoch"
    one_epoch_reference_dir = tmp_path / "one_epoch_reference"
    two_epoch_reference_dir = tmp_path / "two_epoch_reference"

    _make_tiny_regression_trainer(
        "trainer_custom_model_selection_first_epoch",
        save_model_dir=first_epoch_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        model_selection_score=lambda validation_loss, training_loss, epoch: float(epoch),
    ).fit(2)

    _make_tiny_regression_trainer(
        "trainer_custom_model_selection_one_epoch_reference",
        save_model_dir=one_epoch_reference_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
    ).fit(1)

    _make_tiny_regression_trainer(
        "trainer_custom_model_selection_two_epoch_reference",
        save_model_dir=two_epoch_reference_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        model_selection_score=lambda validation_loss, training_loss, epoch: -float(epoch),
    ).fit(2)

    selected_prediction = _prediction_from_saved_tiny_regressor(first_epoch_dir, "trainer_custom_model_selection_first_epoch")
    one_epoch_prediction = _prediction_from_saved_tiny_regressor(one_epoch_reference_dir, "trainer_custom_model_selection_one_epoch_reference")
    two_epoch_prediction = _prediction_from_saved_tiny_regressor(two_epoch_reference_dir, "trainer_custom_model_selection_two_epoch_reference")

    assert np.allclose(selected_prediction, one_epoch_prediction, atol=1e-6)
    assert not np.allclose(selected_prediction, two_epoch_prediction, atol=1e-7)






@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_trainer_fit_returns_result_and_persists_selection_metadata(tmp_path):
    save_dir = tmp_path / "trainer_fit_result_metadata"
    early_policy = thor.training.EarlyCompletionPolicy(
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch >= 2
    )
    trainer = _make_tiny_regression_trainer(
        "trainer_fit_result_metadata",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        model_selection_score=lambda validation_loss, training_loss, epoch: float(epoch),
        early_completion_policies=[early_policy],
    )

    result = trainer.fit(50)

    assert isinstance(result, thor.training.TrainingRunResult)
    assert result.status == "completed"
    assert result.result == "early_completed"
    assert result.early_completed is True
    assert result.completed_epoch == 2
    assert result.best_epoch == 1
    assert result.best_score == pytest.approx(1.0)
    assert save_dir.exists()

    metadata = _training_selection_metadata(save_dir)
    assert metadata["schema_version"] == 1
    assert metadata["best_epoch"] == 1
    assert metadata["best_score"] == pytest.approx(1.0)
    assert metadata["completed_epoch"] == 2
    assert metadata["completion_reason"] == "early_completed"
    assert metadata["check_best_model_every_epochs"] == 1
    assert metadata["min_early_completion_epochs"] == 0


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_trainer_model_selection_score_none_skips_candidate_for_epoch(tmp_path):
    save_dir = tmp_path / "custom_score_none_skips_epoch"
    one_epoch_reference_dir = tmp_path / "custom_score_none_one_epoch_reference"
    two_epoch_reference_dir = tmp_path / "custom_score_none_two_epoch_reference"

    result = _make_tiny_regression_trainer(
        "trainer_custom_model_selection_none_skips_epoch",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        model_selection_score=lambda validation_loss, training_loss, epoch: None if epoch == 1 else float(epoch),
    ).fit(2)

    _make_tiny_regression_trainer(
        "trainer_custom_model_selection_none_one_epoch_reference",
        save_model_dir=one_epoch_reference_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
    ).fit(1)

    _make_tiny_regression_trainer(
        "trainer_custom_model_selection_none_two_epoch_reference",
        save_model_dir=two_epoch_reference_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
    ).fit(2)

    assert result.status == "completed"
    assert result.result == "completed"
    assert result.completed_epoch == 2
    assert result.best_epoch == 2
    assert result.best_score == pytest.approx(2.0)

    metadata = _training_selection_metadata(save_dir)
    assert metadata["best_epoch"] == 2
    assert metadata["best_score"] == pytest.approx(2.0)
    assert metadata["completed_epoch"] == 2
    assert metadata["completion_reason"] == "completed"

    selected_prediction = _prediction_from_saved_tiny_regressor(save_dir, "trainer_custom_model_selection_none_skips_epoch")
    one_epoch_prediction = _prediction_from_saved_tiny_regressor(
        one_epoch_reference_dir, "trainer_custom_model_selection_none_one_epoch_reference"
    )
    two_epoch_prediction = _prediction_from_saved_tiny_regressor(
        two_epoch_reference_dir, "trainer_custom_model_selection_none_two_epoch_reference"
    )

    assert np.allclose(selected_prediction, two_epoch_prediction, atol=1e-6)
    assert not np.allclose(selected_prediction, one_epoch_prediction, atol=1e-7)


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_trainer_min_early_completion_epochs_can_complete_without_candidate_or_saved_model(tmp_path):
    save_dir = tmp_path / "min_early_completion_no_candidate"
    trainer = _make_tiny_regression_trainer(
        "trainer_min_early_completion_no_candidate",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        min_early_completion_epochs=5,
        model_selection_score=lambda validation_loss, training_loss, epoch: float(epoch),
    )

    result = trainer.fit(2)

    assert result.status == "completed"
    assert result.result == "completed"
    assert result.completed_epoch == 2
    assert result.best_epoch is None
    assert result.best_score is None
    assert trainer.completed_training_epochs == 2
    assert not save_dir.exists()

    final_dir = tmp_path / "manual_final_model_after_no_candidate"
    trainer.save_model(final_dir)
    assert final_dir.exists()
    prediction = _prediction_from_saved_tiny_regressor(final_dir, "trainer_min_early_completion_no_candidate")
    assert not np.allclose(prediction, 0.0, atol=1e-7)


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_trainer_min_early_completion_epochs_uses_cumulative_epoch_across_fit_calls(tmp_path):
    save_dir = tmp_path / "min_early_completion_cumulative"
    early_policy = thor.training.EarlyCompletionPolicy(
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch >= 3
    )
    trainer = _make_tiny_regression_trainer(
        "trainer_min_early_completion_cumulative",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        min_early_completion_epochs=3,
        model_selection_score=lambda validation_loss, training_loss, epoch: float(epoch),
        early_completion_policies=[early_policy],
    )

    first_result = trainer.fit(2)
    assert first_result.completed_epoch == 2
    assert first_result.best_epoch is None
    assert trainer.completed_training_epochs == 2
    assert not save_dir.exists()

    second_result = trainer.fit(10)

    assert second_result.status == "completed"
    assert second_result.result == "early_completed"
    assert second_result.early_completed is True
    assert second_result.completed_epoch == 3
    assert second_result.best_epoch == 3
    assert second_result.best_score == pytest.approx(3.0)
    assert trainer.completed_training_epochs == 3
    assert save_dir.exists()

    metadata = _training_selection_metadata(save_dir)
    assert metadata["best_epoch"] == 3
    assert metadata["best_score"] == pytest.approx(3.0)
    assert metadata["completed_epoch"] == 3
    assert metadata["completion_reason"] == "early_completed"
    assert metadata["check_best_model_every_epochs"] == 1
    assert metadata["min_early_completion_epochs"] == 3


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_min_early_completion_epochs_uses_cumulative_epoch_across_fit_calls(tmp_path):
    save_dir = tmp_path / "training_runs_min_early_completion_cumulative"
    early_policy = thor.training.EarlyCompletionPolicy(
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch >= 3
    )
    trainer = _make_tiny_regression_trainer(
        "training_runs_min_early_completion_cumulative",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        min_early_completion_epochs=3,
        model_selection_score=lambda validation_loss, training_loss, epoch: float(epoch),
        early_completion_policies=[early_policy],
    )
    runs = thor.training.TrainingRuns([("fold_0", trainer)], failure_policy="continue")

    first_results = runs.fit(epochs=2)
    first_result = first_results["fold_0"]
    assert first_results.all_completed()
    assert first_result.status == "completed"
    assert first_result.result == "completed"
    assert first_result.completed_epoch == 2
    assert first_result.best_epoch is None
    assert first_result.best_score is None
    assert trainer.completed_training_epochs == 2
    assert not save_dir.exists()

    second_results = runs.fit(epochs=10)
    second_result = second_results["fold_0"]
    assert second_results.all_completed()
    assert second_result.status == "completed"
    assert second_result.result == "early_completed"
    assert second_result.early_completed is True
    assert second_result.completed_epoch == 3
    assert second_result.best_epoch == 3
    assert second_result.best_score == pytest.approx(3.0)
    assert trainer.completed_training_epochs == 3
    assert save_dir.exists()

    metadata = _training_selection_metadata(save_dir)
    assert metadata["best_epoch"] == 3
    assert metadata["best_score"] == pytest.approx(3.0)
    assert metadata["completed_epoch"] == 3
    assert metadata["completion_reason"] == "early_completed"
    assert metadata["check_best_model_every_epochs"] == 1
    assert metadata["min_early_completion_epochs"] == 3



@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_restart_policy_uses_cumulative_epoch_across_fit_calls():
    trainer = _make_tiny_regression_trainer(
        "training_runs_restart_policy_cumulative_epoch",
        optimizer_obj=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
    )
    runs = thor.training.TrainingRuns(
        [("fold_0", trainer)],
        failure_policy="continue",
        restart_conditions=[
            thor.training.RestartPolicy(
                run_name="fold_0",
                progress_check_epochs=3,
                progress_improvement_min_percentage=100.0,
                max_restarts=0,
            )
        ],
    )

    first_results = runs.fit(epochs=2)
    assert first_results.all_completed()
    assert first_results["fold_0"].status == "completed"
    assert trainer.completed_training_epochs == 2

    second_results = runs.fit(epochs=1)
    second_result = second_results["fold_0"]
    assert second_results.any_failed()
    assert second_result.status == "failed"
    assert second_result.exception_type == "TrainingRestartConditionExceeded"
    assert "progress_check_epochs=3" in second_result.exception_message
    assert trainer.completed_training_epochs == 2



@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_early_completion_stops_early_and_saves_best_candidate(capfd, tmp_path):
    early_dir = tmp_path / "early_completed_best"
    one_epoch_reference_dir = tmp_path / "one_epoch_reference"

    early_policy = thor.training.EarlyCompletionPolicy(
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch >= 2
    )
    trainer = _make_tiny_regression_trainer(
        "training_runs_early_completion_best_candidate",
        save_model_dir=early_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        model_selection_score=lambda validation_loss, training_loss, epoch: float(epoch),
        early_completion_policies=[early_policy],
    )
    runs = thor.training.TrainingRuns([("fold_0", trainer)], failure_policy="continue")

    results, captured_text = _fit_runs_and_capture_text(runs, capfd, epochs=50)

    result = results["fold_0"]
    assert results.all_completed()
    assert result.status == "completed"
    assert result.result == "early_completed"
    assert result.early_completed is True
    assert result.completed_epoch == 2
    assert result.best_epoch == 1
    assert result.best_score == pytest.approx(1.0)
    assert result.final_training_stats.epoch == 2
    assert result.final_training_stats.epochs == 50
    assert early_dir.exists()

    plain_text = _ANSI_RE.sub("", captured_text)
    assert re.search(
        r"INFO runs\[fold_0\]:.*status=completed.*result=early_completed.*completed_epoch=2.*best_epoch=1.*best_score=1\.000000",
        plain_text,
    )

    _make_tiny_regression_trainer(
        "training_runs_early_completion_one_epoch_reference",
        save_model_dir=one_epoch_reference_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
    ).fit(1)

    selected_prediction = _prediction_from_saved_tiny_regressor(early_dir, "training_runs_early_completion_best_candidate")
    one_epoch_prediction = _prediction_from_saved_tiny_regressor(
        one_epoch_reference_dir, "training_runs_early_completion_one_epoch_reference"
    )

    assert np.allclose(selected_prediction, one_epoch_prediction, atol=1e-6)

@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_fits_two_tiny_trainers_on_one_gpu_and_prefixes_stats(capfd, tmp_path):
    runs = thor.training.TrainingRuns(
        [
            (
                "fold_0",
                _make_tiny_regression_trainer(
                    "training_runs_cuda_two_0",
                    save_model_dir=tmp_path / "fold_0_model",
                    save_model_overwrite=True,
                ),
                "tiny_ensemble",
                1.0,
            ),
            (
                "longer_fold_1",
                _make_tiny_regression_trainer(
                    "training_runs_cuda_two_1",
                    save_model_dir=tmp_path / "longer_fold_1_model",
                    save_model_overwrite=True,
                ),
                "tiny_ensemble",
                2.0,
            ),
        ],
    )

    test_loader = _regression_one_batch_loader()
    results, captured_text = _fit_runs_and_capture_text(runs, capfd, epochs=1, test_loader=test_loader)

    assert len(results) == 2
    assert results.all_completed()
    for run_name in ("fold_0", "longer_fold_1"):
        result = results[run_name]
        assert result.status == "completed"
        assert result.ensemble_group == "tiny_ensemble"
        assert result.final_training_loss is not None
        assert result.final_validation_loss is not None
        assert result.final_test_loss is not None
        assert result.final_loss("train") == result.final_training_loss
        assert result.final_loss("validate") == result.final_validation_loss
        assert result.final_loss("test") == result.final_test_loss
        assert result.final_training_step is not None
        assert result.final_validation_step is not None
        assert result.final_test_step is not None

    plain_text = _ANSI_RE.sub("", captured_text)
    statuses = _captured_terminal_statuses(captured_text)
    assert statuses["fold_0"] == "completed"
    assert statuses["longer_fold_1"] == "completed"
    assert "INFO runs summary:" in plain_text
    assert "\nINFO runs final: ==================== final results" in plain_text
    assert "INFO runs final:" in plain_text
    assert "INFO runs final: =====================================================" in plain_text
    assert "completed=2" in plain_text
    assert results.status_counts["completed"] == 2
    assert results.has_ensembles
    assert len(results.ensembles) == 1
    ensemble = results.ensemble("tiny_ensemble")
    assert ensemble.ensemble_group == "tiny_ensemble"
    assert ensemble.all_completed()
    assert ensemble.total_weight == pytest.approx(3.0)
    assert ensemble.ensemble_train_loss is not None
    assert (tmp_path / "fold_0_model").exists()
    assert (tmp_path / "longer_fold_1_model").exists()
    assert results["fold_0"].saved_model_dir == str(tmp_path / "fold_0_model")
    assert results["longer_fold_1"].saved_model_dir == str(tmp_path / "longer_fold_1_model")

    ensemble_artifact_dir = tmp_path / "tiny_ensemble_artifact"
    manifest_path = results.save_ensemble("tiny_ensemble", ensemble_artifact_dir)
    assert manifest_path == str(ensemble_artifact_dir / "ensemble_manifest.json")
    loaded_ensemble = thor.EnsembleModel.load(ensemble_artifact_dir)
    assert loaded_ensemble.get_num_members() == 2
    assert loaded_ensemble.get_aggregation() == "weighted_mean"
    assert loaded_ensemble.get_member_names() == ("fold_0", "longer_fold_1")
    assert loaded_ensemble.get_member_weights() == (1.0, 2.0)
    assert loaded_ensemble.get_input_names() == ("examples",)
    assert loaded_ensemble.get_output_names() == ("prediction",)

    test_examples, test_labels = _regression_arrays(dtype=np.float32)
    loaded_outputs = loaded_ensemble.infer({"examples": _cpu_tensor(test_examples, thor.DataType.fp32)})
    assert set(loaded_outputs) == {"prediction"}
    loaded_predictions = loaded_outputs["prediction"].numpy()
    loaded_ensemble_mae = float(np.mean(np.abs(loaded_predictions - test_labels)))
    assert ensemble.ensemble_test_loss == pytest.approx(loaded_ensemble_mae, rel=1e-5, abs=1e-6)

    assert (ensemble_artifact_dir / loaded_ensemble.members[0].path / "training_selection_metadata.json").exists()
    assert (ensemble_artifact_dir / loaded_ensemble.members[1].path / "training_selection_metadata.json").exists()
    manifest = json.loads((ensemble_artifact_dir / "ensemble_manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["ensemble_group"] == "tiny_ensemble"
    assert manifest["execution"] == "parallel_single_gpu"
    assert manifest["members"][0]["selection"]["status"] == "completed"

    with pytest.raises(RuntimeError, match="already exists"):
        results.save_ensemble("tiny_ensemble", ensemble_artifact_dir)
    assert results.save_ensemble("tiny_ensemble", ensemble_artifact_dir, overwrite=True) == str(
        ensemble_artifact_dir / "ensemble_manifest.json"
    )

    assert ensemble.ensemble_test_loss is not None
    assert len(ensemble.members) == 2
    assert [member.run_name for member in ensemble.members] == ["fold_0", "longer_fold_1"]
    assert [member.weight for member in ensemble.members] == [1.0, 2.0]
    assert len(ensemble.output_signature) >= 1
    assert "INFO runs ensemble:" in plain_text
    assert "INFO runs ensemble[tiny_ensemble]:" in plain_text
    assert "INFO runs[fold_0|tiny_ensemble]:" in plain_text
    assert "INFO runs[longer_fold_1|tiny_ensemble]:" in plain_text
    assert "ensemble_group=tiny_ensemble" not in plain_text
    assert "aggregation=ensemble_eval" in plain_text
    assert "ensemble_train_loss=" in plain_text
    assert "ensemble_test_loss=" in plain_text
    assert "weighted_train_loss=" not in plain_text
    assert "weighted_validate_loss=" not in plain_text
    assert re.search(r"INFO runs\[fold_0\|tiny_ensemble\]:\s+status=completed", plain_text)
    for run_name in ("fold_0", "longer_fold_1"):
        assert re.search(
            rf"INFO runs\[{re.escape(run_name)}\|tiny_ensemble\]:.*train_loss=.*validate_loss=.*test_loss=",
            plain_text,
        ), f"final report did not include per-run test_loss for {run_name}:\n{plain_text}"
    assert "train_loss=" in plain_text
    assert "validate_loss=" in plain_text
    assert "test_loss=" in plain_text
    assert "final_train_loss=" not in plain_text
    assert "final_validate_loss=" not in plain_text
    assert "phase=unknown" not in plain_text
    assert "INFO trainer:" not in plain_text


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_categorical_report_matches_loaded_ensemble_predictions(capfd, tmp_path):
    runs = thor.training.TrainingRuns(
        [
            (
                "fold_0",
                _make_tiny_categorical_trainer(
                    "training_runs_categorical_fold_0",
                    save_model_dir=tmp_path / "categorical_fold_0_model",
                    save_model_overwrite=True,
                ),
                "tiny_categorical_ensemble",
                1.0,
            ),
            (
                "fold_1",
                _make_tiny_categorical_trainer(
                    "training_runs_categorical_fold_1",
                    save_model_dir=tmp_path / "categorical_fold_1_model",
                    save_model_overwrite=True,
                ),
                "tiny_categorical_ensemble",
                2.0,
            ),
        ],
    )

    results, _ = _fit_runs_and_capture_text(
        runs,
        capfd,
        epochs=1,
        test_loader=_categorical_mixed_accuracy_one_batch_loader(),
    )

    assert results.all_completed()
    ensemble = results.ensemble("tiny_categorical_ensemble")
    assert ensemble.ensemble_test_loss is not None
    assert ensemble.ensemble_test_accuracy is not None

    ensemble_artifact_dir = tmp_path / "tiny_categorical_ensemble_artifact"
    results.save_ensemble("tiny_categorical_ensemble", ensemble_artifact_dir)
    loaded_ensemble = thor.EnsembleModel.load(ensemble_artifact_dir)
    assert loaded_ensemble.get_aggregation() == "weighted_mean"
    assert loaded_ensemble.get_member_weights() == (1.0, 2.0)
    assert loaded_ensemble.get_input_names() == ("examples",)
    assert loaded_ensemble.get_output_names() == ("scores",)

    test_examples, test_labels = _categorical_mixed_accuracy_arrays(dtype=np.float32)
    loaded_outputs = loaded_ensemble.infer({"examples": _cpu_tensor(test_examples, thor.DataType.fp32)})
    assert set(loaded_outputs) == {"scores"}
    loaded_scores = np.array(loaded_outputs["scores"].numpy(), copy=True)
    expected_loss, expected_accuracy = _softmax_cross_entropy_and_accuracy(loaded_scores, test_labels)
    assert expected_accuracy == pytest.approx(0.75, abs=0.0)
    assert 0.0 < expected_accuracy < 1.0

    assert ensemble.ensemble_test_loss == pytest.approx(expected_loss, rel=1e-5, abs=1e-6)
    assert ensemble.ensemble_test_accuracy == pytest.approx(expected_accuracy, rel=1e-5, abs=1e-6)


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_fits_five_tiny_trainers_on_one_gpu(capfd):
    run_specs = [
        (f"fold_{index}", _make_tiny_regression_trainer(f"training_runs_cuda_five_{index}")) for index in range(5)
    ]
    runs = thor.training.TrainingRuns(run_specs, max_parallel_runs=2)

    results, captured_text = _fit_runs_and_capture_text(runs, capfd, epochs=1)

    assert len(results) == 5
    assert results.all_completed()
    plain_text = _ANSI_RE.sub("", captured_text)
    statuses = _captured_terminal_statuses(captured_text)
    assert "INFO runs summary:" in plain_text
    assert "INFO runs final:" in plain_text
    assert "completed=5" in plain_text
    assert results.status_counts["completed"] == 5
    assert "phase=unknown" not in plain_text
    for index in range(5):
        run_name = f"fold_{index}"
        assert results[run_name].status == "completed"
        assert results[run_name].final_training_loss is not None
        assert results[run_name].final_validation_loss is not None
        assert results[run_name].final_test_loss is None
        assert statuses[run_name] == "completed"


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_continue_policy_allows_real_cuda_sibling_to_finish_after_failure(capfd):
    bad_trainer = _make_tiny_regression_trainer("training_runs_cuda_continue_bad", optimizer=False)
    good_trainer = _make_tiny_regression_trainer("training_runs_cuda_continue_good", optimizer=True)
    runs = thor.training.TrainingRuns(
        [("bad_fold", bad_trainer), ("good_fold", good_trainer)],
        failure_policy="continue",
    )

    results, captured_text = _fit_runs_and_capture_text(runs, capfd, epochs=1)
    plain_text = _ANSI_RE.sub("", captured_text)

    assert len(results) == 2
    assert results.any_failed()
    assert not results.any_cancelled()
    assert results.status_counts["failed"] == 1
    assert results.status_counts["completed"] == 1
    assert results["bad_fold"].status == "failed"
    assert results["bad_fold"].exception_message
    assert results["good_fold"].status == "completed"
    assert results["good_fold"].final_training_loss is not None
    assert "INFO runs[bad_fold]:" in plain_text
    assert "INFO runs[good_fold]:" in plain_text
    assert "INFO runs final: total=2" in plain_text
    assert "INFO runs summary: total=1" not in plain_text


def _demand_end_to_end_arrays():
    product_ids = [f"product_{index}" for index in range(6)]
    row_groups = []
    trend_rows = []
    seasonality_rows = []
    monotone_rows = []
    for product_index, product_id in enumerate(product_ids):
        for date_index in range(2):
            row_groups.append(product_id)
            trend = np.array([float(product_index + 1)], dtype=np.float32)
            seasonality = np.array([float(date_index), float(product_index - date_index)], dtype=np.float32)
            monotone = np.array([float(product_index + date_index + 1)], dtype=np.float32)
            trend_rows.append(trend)
            seasonality_rows.append(seasonality)
            monotone_rows.append(monotone)

    trend_inputs = np.ascontiguousarray(np.stack(trend_rows, axis=0), dtype=np.float32)
    seasonality_inputs = np.ascontiguousarray(np.stack(seasonality_rows, axis=0), dtype=np.float32)
    monotone_inputs = np.ascontiguousarray(np.stack(monotone_rows, axis=0), dtype=np.float32)
    observed_future_demand = np.ascontiguousarray(
        np.concatenate([trend_inputs, seasonality_inputs, monotone_inputs], axis=1),
        dtype=np.float32,
    )
    return product_ids, tuple(row_groups), {
        "trend_inputs": trend_inputs,
        "seasonality_inputs": seasonality_inputs,
        "monotone_increasing_inputs": monotone_inputs,
        "observed_future_demand": observed_future_demand,
    }


def _fold_split_with_holdout(fold, *, test_keys, test_groups):
    return thor.data.StratifiedTrainValidationTestSplit(
        train_keys=fold.train_keys,
        validate_keys=fold.validate_keys,
        test_keys=tuple(test_keys),
        train_groups=fold.train_groups,
        validate_groups=fold.validate_groups,
        test_groups=tuple(test_groups),
    )


def _build_demand_end_to_end_network(name: str) -> thor.Network:
    network = thor.Network(name)
    trend = thor.layers.NetworkInput(network, "trend_inputs", [1], thor.DataType.fp32)
    seasonality = thor.layers.NetworkInput(network, "seasonality_inputs", [2], thor.DataType.fp32)
    monotone = thor.layers.NetworkInput(network, "monotone_increasing_inputs", [1], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(network, "observed_future_demand", [4], thor.DataType.fp32)
    features = thor.layers.Concatenate(
        network,
        [
            trend.get_feature_output(),
            seasonality.get_feature_output(),
            monotone.get_feature_output(),
        ],
        0,
    )
    forecast = thor.layers.FullyConnected(
        network,
        features.get_feature_output(),
        4,
        True,
        activation=None,
        weights_initializer=thor.initializers.UniformRandom(0.0, 0.0),
        biases_initializer=thor.initializers.UniformRandom(0.0, 0.0),
    )
    quantile = thor.layers.CustomLayer(
        network=network,
        inputs={"forecast": forecast.get_feature_output()},
        output_names=["forecast_quantile_high"],
        build=lambda context: {"forecast_quantile_high": context.input("forecast") + 1.0},
    )
    loss = thor.losses.MSE(network, forecast.get_feature_output(), labels.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "forecast", forecast.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "forecast_quantile_high", quantile["forecast_quantile_high"], thor.DataType.fp32)
    return network


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns demand-style k-fold ensemble smoke test",
    ),
)
def test_training_runs_demand_style_kfold_full_path_saves_loadable_ensemble(capfd, tmp_path):
    product_ids, row_groups, tensors = _demand_end_to_end_arrays()
    split = thor.data.StratifiedSplitter(
        product_ids,
        [float(index) for index in range(len(product_ids))],
        mode="quantile",
        num_bins=3,
        seed=23,
    ).holdout_plus_k_fold(test_size=2, k=2)

    def make_trainer(*, fold, run_name, test_keys, test_groups):
        fold_split = _fold_split_with_holdout(fold, test_keys=test_keys, test_groups=test_groups)
        numpy_splits = thor.data.make_numpy_dict_splits(tensors, split=fold_split, groups=row_groups)
        loader = thor.training.NumpyFloat32DictBatchLoader(
            train=numpy_splits.train,
            validate=numpy_splits.validate,
            test=numpy_splits.test,
            batch_size=2,
            randomize_train=False,
            dataset_name=f"demand_kfold_{run_name}",
        )
        return thor.training.Trainer(
            _build_demand_end_to_end_network(f"demand_kfold_network_{run_name}"),
            loader,
            optimizer=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
            stats_interval_s=0.0,
            max_in_flight_batches=2,
            scalar_tensors_to_report=["loss"],
            stats_color="never",
            save_model_dir=tmp_path / f"{run_name}_model",
            save_model_overwrite=True,
        )

    runs = thor.training.training_runs_from_k_fold_split(
        split,
        make_trainer=make_trainer,
        ensemble_group="brand_demand_cv2",
        run_name_template="brand_fold_{fold_index}",
        max_parallel_runs=2,
        min_successful_models={"brand_demand_cv2": 2},
    )
    holdout_split = thor.data.StratifiedTrainValidationTestSplit(
        train_keys=split.test_keys,
        validate_keys=split.test_keys,
        test_keys=split.test_keys,
        train_groups=split.test_groups,
        validate_groups=split.test_groups,
        test_groups=split.test_groups,
    )
    holdout_numpy = thor.data.make_numpy_dict_splits(tensors, split=holdout_split, groups=row_groups)
    test_loader = thor.training.NumpyFloat32DictBatchLoader(
        train=holdout_numpy.train,
        validate=holdout_numpy.validate,
        test=holdout_numpy.test,
        batch_size=2,
        randomize_train=False,
        dataset_name="demand_kfold_holdout",
    )

    results, captured_text = _fit_runs_and_capture_text(runs, capfd, epochs=1, test_loader=test_loader)

    assert results.all_completed()
    assert results.has_ensembles
    ensemble_result = results.ensemble("brand_demand_cv2")
    assert ensemble_result.all_completed()
    assert ensemble_result.ensemble_test_loss is not None
    assert "INFO runs ensemble[brand_demand_cv2]:" in _ANSI_RE.sub("", captured_text)

    ensemble_artifact_dir = tmp_path / "brand_demand_cv2_ensemble"
    results.save_ensemble("brand_demand_cv2", ensemble_artifact_dir)
    ensemble = thor.EnsembleModel.load(ensemble_artifact_dir)
    assert ensemble.get_num_members() == 2
    assert ensemble.get_input_names() == ("trend_inputs", "seasonality_inputs", "monotone_increasing_inputs")
    assert ensemble.get_output_names() == ("forecast", "forecast_quantile_high")

    deploy_inputs = {
        "trend_inputs": _cpu_tensor(holdout_numpy.test["trend_inputs"], thor.DataType.fp32),
        "seasonality_inputs": _cpu_tensor(holdout_numpy.test["seasonality_inputs"], thor.DataType.fp32),
        "monotone_increasing_inputs": _cpu_tensor(holdout_numpy.test["monotone_increasing_inputs"], thor.DataType.fp32),
    }
    outputs = ensemble.infer(deploy_inputs)

    assert set(outputs) == {"forecast", "forecast_quantile_high"}
    assert outputs["forecast"].get_placement().get_device_type() == thor.physical.DeviceType.cpu
    assert outputs["forecast_quantile_high"].get_placement().get_device_type() == thor.physical.DeviceType.cpu
    assert outputs["forecast"].numpy().shape == holdout_numpy.test["observed_future_demand"].shape
    assert outputs["forecast_quantile_high"].numpy().shape == holdout_numpy.test["observed_future_demand"].shape
    assert np.all(np.isfinite(outputs["forecast"].numpy()))
    assert np.all(np.isfinite(outputs["forecast_quantile_high"].numpy()))
