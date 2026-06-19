import ctypes
import os
import re
import sys

import numpy as np
import pytest
import thor

RUN_TRAINING_INTEGRATION = os.environ.get("THOR_RUN_TRAINING_INTEGRATION") == "1"
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


def _fit_runs_and_capture_text(runs, capfd, *, epochs: int):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    results = runs.fit(epochs=epochs)
    _flush_native_stdio_for_capture()
    captured = capfd.readouterr()
    captured_text = captured.out + captured.err

    # Mirror the trainer integration tests: native fprintf output is captured for
    # assertions, then replayed outside capture so failures still show useful logs.
    with capfd.disabled():
        sys.stdout.write(captured_text)
        sys.stdout.flush()

    return results, captured_text


def _regression_one_batch_loader(*, dtype=np.float32):
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
    x = np.ascontiguousarray(x, dtype=dtype)
    y = np.ascontiguousarray(y, dtype=dtype)
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
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction", predictions.get_feature_output(), thor.DataType.fp32)
    return network


def _build_signature_only_network(name: str, *, input_dtype=thor.DataType.fp32, output_dtype=thor.DataType.fp32):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], input_dtype)
    thor.layers.NetworkOutput(network, "prediction", examples.get_feature_output(), output_dtype)
    return network


def _make_signature_only_trainer(name: str, *, input_dtype=thor.DataType.fp32, output_dtype=thor.DataType.fp32):
    return thor.training.Trainer(
        _build_signature_only_network(name, input_dtype=input_dtype, output_dtype=output_dtype),
        _regression_one_batch_loader(),
        stats=False,
    )


def _make_tiny_regression_trainer(name: str, *, optimizer=True, stats=True, save_model_dir=None, save_model_overwrite=False):
    optimizer_obj = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0) if optimizer else None
    return thor.training.Trainer(
        _build_tiny_regressor(name),
        _regression_one_batch_loader(),
        optimizer=optimizer_obj,
        stats=stats,
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
        save_model_dir=save_model_dir,
        save_model_overwrite=save_model_overwrite,
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


def test_trainer_binding_accepts_pathlike_save_model_dir_for_training_runs_artifact(tmp_path):
    trainer = _make_tiny_regression_trainer(
        "training_runs_pathlike_save_model_dir",
        stats=False,
        save_model_dir=tmp_path / "model_artifact",
    )

    runs = thor.training.TrainingRuns([("fold_0", trainer, "tiny_ensemble")])

    assert runs is not None


def test_trainer_binding_rejects_empty_save_model_dir():
    with pytest.raises((ValueError, RuntimeError), match="save_model_dir must not be empty"):
        _make_tiny_regression_trainer("training_runs_empty_save_model_dir", stats=False, save_model_dir="")


def test_training_runs_result_status_names_are_exposed():
    assert thor.training.TrainingRunStatus.completed.name == "completed"
    assert thor.training.TrainingRunsFailurePolicy.cancel_siblings.name == "cancel_siblings"


def test_training_runs_binding_rejects_duplicate_save_model_dirs(tmp_path):
    shared_dir = tmp_path / "shared_checkpoint"
    equivalent_shared_dir = tmp_path / "nested" / ".." / "shared_checkpoint"
    trainer0 = _make_tiny_regression_trainer(
        "training_runs_duplicate_save_dir_0",
        stats=False,
        save_model_dir=str(shared_dir),
    )
    trainer1 = _make_tiny_regression_trainer(
        "training_runs_duplicate_save_dir_1",
        stats=False,
        save_model_dir=str(equivalent_shared_dir),
    )

    with pytest.raises(RuntimeError, match="save_model_dir collision"):
        thor.training.TrainingRuns([("fold_0", trainer0), ("fold_1", trainer1)])


def test_training_runs_fit_rejects_ensemble_without_trainer_save_model_dir():
    trainer = _make_tiny_regression_trainer("training_runs_missing_ensemble_save_dir", stats=False)
    runs = thor.training.TrainingRuns([("fold_0", trainer, "tiny_ensemble")])

    with pytest.raises(RuntimeError, match="save_model_dir"):
        runs.fit(epochs=1)


def test_training_runs_binding_rejects_invalid_ensemble_weight():
    trainer = _make_tiny_regression_trainer("training_runs_invalid_ensemble_weight", stats=False)

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


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason="set THOR_RUN_TRAINING_INTEGRATION=1 to run opt-in TrainingRuns CUDA integration tests",
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

    results, captured_text = _fit_runs_and_capture_text(runs, capfd, epochs=1)

    assert len(results) == 2
    assert results.all_completed()
    for run_name in ("fold_0", "longer_fold_1"):
        result = results[run_name]
        assert result.status == "completed"
        assert result.ensemble_group == "tiny_ensemble"
        assert result.final_training_loss is not None
        assert result.final_validation_loss is not None
        assert result.final_test_loss is None
        assert result.final_loss("train") == result.final_training_loss
        assert result.final_loss("validate") == result.final_validation_loss
        assert result.final_loss("test") is None
        assert result.final_training_step is not None
        assert result.final_validation_step is not None
        assert result.final_test_step is None

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
    assert ensemble.ensemble_test_loss is None
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
    assert "ensemble_test_loss=" not in plain_text
    assert "weighted_train_loss=" not in plain_text
    assert "weighted_validate_loss=" not in plain_text
    assert re.search(r"INFO runs\[fold_0\|tiny_ensemble\]:\s+status=completed", plain_text)
    assert "train_loss=" in plain_text
    assert "validate_loss=" in plain_text
    assert "final_train_loss=" not in plain_text
    assert "final_validate_loss=" not in plain_text
    assert "phase=unknown" not in plain_text
    assert "INFO trainer:" not in plain_text


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason="set THOR_RUN_TRAINING_INTEGRATION=1 to run opt-in TrainingRuns CUDA integration tests",
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
    reason="set THOR_RUN_TRAINING_INTEGRATION=1 to run opt-in TrainingRuns CUDA integration tests",
)
def test_training_runs_continue_policy_allows_real_cuda_sibling_to_finish_after_failure(capfd):
    bad_trainer = _make_tiny_regression_trainer("training_runs_cuda_continue_bad", optimizer=False, stats=False)
    good_trainer = _make_tiny_regression_trainer("training_runs_cuda_continue_good", optimizer=True, stats=True)
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
