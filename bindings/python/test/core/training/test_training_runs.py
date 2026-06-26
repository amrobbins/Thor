import ctypes
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import weakref
from pathlib import Path

import numpy as np
import pytest
import thor
from integration_flags import integration_flag_enabled, integration_skip_reason

RUN_TRAINING_INTEGRATION = integration_flag_enabled("THOR_RUN_TRAINING_INTEGRATION")
RUN_TRAINING_INTEGRATION_LARGE = integration_flag_enabled("THOR_RUN_TRAINING_INTEGRATION_LARGE")
AIRFOIL_SELF_NOISE_URL = os.environ.get(
    "THOR_AIRFOIL_SELF_NOISE_URL",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
)
AIRFOIL_QUANTILE_CACHE_DIR = Path(
    os.environ.get("THOR_AIRFOIL_QUANTILE_CACHE_DIR", "/tmp/thor_airfoil_quantile_training"))
AIRFOIL_QUANTILE_STATS_COLOR = os.environ.get("THOR_AIRFOIL_QUANTILE_STATS_COLOR", "auto")
AIRFOIL_QUANTILE_STATS_INTERVAL_S = float(os.environ.get("THOR_AIRFOIL_QUANTILE_STATS_INTERVAL_S", "0.0"))
AIRFOIL_QUANTILE_SUMMARY_LOGS_PER_SECOND = float(os.environ.get("THOR_AIRFOIL_QUANTILE_SUMMARY_LOGS_PER_SECOND", "2.0"))
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RUN_STATUS_RE = re.compile(
    r"INFO runs\[(?P<run>[^\]|]+)(?:\|[^\]]+)?\]:.*\bstatus=(?P<status>completed|failed|cancelled|interrupted|oom|running|starting|not_started)\b"
)


def _flush_native_stdio_for_capture():
    ctypes.CDLL(None).fflush(None)


def _captured_terminal_statuses(captured_text: str):
    plain_text = _ANSI_RE.sub("", captured_text)
    seen = {}
    for match in _RUN_STATUS_RE.finditer(plain_text):
        seen[match.group("run")] = match.group("status")
    return seen


class _NativeOutputTee:
    """Mirror native stdout/stderr immediately while keeping text for assertions."""

    def __init__(self):
        self._saved_fds = {}
        self._tee_processes = []
        self._capture_paths = []
        self._saved_force_color = None
        self._had_force_color = False
        self._set_force_color_for_tty_tee = False

    def __enter__(self):
        _flush_native_stdio_for_capture()
        self._had_force_color = "FORCE_COLOR" in os.environ
        self._saved_force_color = os.environ.get("FORCE_COLOR")
        self._set_force_color_for_tty_tee = False
        if os.isatty(1) and not os.environ.get("NO_COLOR"):
            # Native stdout/stderr are about to be redirected to pipes so the
            # helper tee process can mirror output and capture it for assertions.
            # Preserve color=auto terminal behavior by forcing color only when
            # the original stdout was a TTY. Shell redirection still leaves this
            # unset, so redirected files stay plain.
            os.environ["FORCE_COLOR"] = "1"
            self._set_force_color_for_tty_tee = True
        tee_exe = shutil.which("tee")
        assert tee_exe is not None, "the temporary native-output tee requires /usr/bin/tee on PATH"

        for fd in (1, 2):
            saved_fd = os.dup(fd)
            read_fd, write_fd = os.pipe()
            capture_file = tempfile.NamedTemporaryFile(
                prefix=f"thor_training_runs_fit_fd{fd}_", suffix=".log", delete=False)
            capture_path = capture_file.name
            capture_file.close()

            process = subprocess.Popen(
                [tee_exe, capture_path],
                stdin=read_fd,
                stdout=saved_fd,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            os.close(read_fd)
            os.dup2(write_fd, fd)
            os.close(write_fd)

            self._saved_fds[fd] = saved_fd
            self._tee_processes.append(process)
            self._capture_paths.append(capture_path)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            _flush_native_stdio_for_capture()
        finally:
            for fd, saved_fd in self._saved_fds.items():
                os.dup2(saved_fd, fd)

            for process in self._tee_processes:
                process.wait()

            for saved_fd in self._saved_fds.values():
                os.close(saved_fd)

            if self._set_force_color_for_tty_tee:
                if self._had_force_color:
                    os.environ["FORCE_COLOR"] = self._saved_force_color
                else:
                    os.environ.pop("FORCE_COLOR", None)

        return False

    def text(self) -> str:
        parts = []
        for capture_path in self._capture_paths:
            path = Path(capture_path)
            try:
                parts.append(path.read_text(errors="replace"))
            finally:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        return "".join(parts)


def _expects_color_for_stats_color_mode(mode: str) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if mode == "always":
        return True
    if mode == "never":
        return False
    if os.environ.get("CLICOLOR_FORCE") not in {None, "", "0"}:
        return True
    if os.environ.get("FORCE_COLOR") not in {None, "", "0"}:
        return True
    return os.isatty(1)


def _fit_runs_and_capture_text(runs, capfd, *, epochs: int, test_loader=None):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    with capfd.disabled():
        with _NativeOutputTee() as tee:
            results = runs.fit(epochs=epochs, test_loader=test_loader)
    return results, tee.text()


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


def _non_finite_regression_one_batch_loader(non_finite_phase: str, *, dtype=np.float32):
    x, y = _regression_arrays(dtype=dtype)
    train_y = y.copy()
    validate_y = y.copy()
    if non_finite_phase == "train":
        train_y[0, 0] = np.inf
    elif non_finite_phase == "validate":
        validate_y[0, 0] = np.inf
    else:
        raise ValueError(f"unsupported non_finite_phase: {non_finite_phase}")

    loader_cls = thor.training.NumpyFloat16BatchLoader if dtype == np.float16 else thor.training.NumpyFloat32BatchLoader
    return loader_cls(
        x,
        np.ascontiguousarray(train_y, dtype=dtype),
        x,
        np.ascontiguousarray(validate_y, dtype=dtype),
        batch_size=4,
        example_input_name="examples",
        label_input_name="labels",
        dataset_name=f"training_runs_non_finite_{non_finite_phase}_loss",
    )


def _weighted_regression_arrays(*, dtype=np.float32):
    x, y = _regression_arrays(dtype=dtype)
    example_weights = np.array([[1.0], [0.5], [2.0], [3.0]], dtype=np.float32)
    return x, y, np.ascontiguousarray(example_weights, dtype=dtype)


def _weighted_regression_one_batch_loader(*, dtype=np.float32):
    x, y, example_weights = _weighted_regression_arrays(dtype=dtype)
    tensors = {
        "examples": x,
        "labels": y,
        "example_weights": example_weights,
    }
    return thor.training.NumpyFloat32DictBatchLoader(
        train={
            name: value.copy() for name, value in tensors.items()
        },
        validate={
            name: value.copy() for name, value in tensors.items()
        },
        test={
            name: value.copy() for name, value in tensors.items()
        },
        batch_size=4,
        dataset_name="training_runs_weighted_regression_one_batch",
        randomize_train=False,
    )


def _mae_quantile_regression_arrays(*, dtype=np.float32):
    # Positive, continuous targets make this a better regression/forecast smoke
    # dataset than the symmetric +/-1 toy data used by the generic tiny regressor.
    # With zero-initialized prediction heads and effectively-frozen training, the
    # deterministic mean target value is 3.5.  That gives MAE=3.5, p10 pinball
    # loss=0.35, and p90 pinball loss=3.15.
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([[1.0], [2.0], [4.0], [7.0]], dtype=np.float32)
    return np.ascontiguousarray(x, dtype=dtype), np.ascontiguousarray(y, dtype=dtype)


def _mae_quantile_regression_one_batch_loader(*, dtype=np.float32):
    x, y = _mae_quantile_regression_arrays(dtype=dtype)
    loader_cls = thor.training.NumpyFloat16BatchLoader if dtype == np.float16 else thor.training.NumpyFloat32BatchLoader
    return loader_cls(
        x,
        y,
        x,
        y,
        batch_size=4,
        example_input_name="examples",
        label_input_name="demand",
        dataset_name="training_runs_mae_quantile_regression_one_batch",
    )


def _download_file_if_missing(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _ensure_airfoil_self_noise_arrays(*, cache_root: Path = AIRFOIL_QUANTILE_CACHE_DIR):
    # UCI Airfoil Self-Noise is a compact real regression dataset: 1503 rows,
    # five numeric features, and one continuous sound-pressure target.
    data_path = cache_root / "downloads" / "airfoil_self_noise.dat"
    _download_file_if_missing(AIRFOIL_SELF_NOISE_URL, data_path)
    data = np.loadtxt(data_path, dtype=np.float32)
    assert data.ndim == 2
    assert data.shape[0] >= 1000
    assert data.shape[1] == 6

    features = np.ascontiguousarray(data[:, :5], dtype=np.float32)
    target = np.ascontiguousarray(data[:, 5:6], dtype=np.float32)

    feature_mean = features.mean(axis=0, keepdims=True)
    feature_std = features.std(axis=0, keepdims=True)
    feature_std[feature_std == 0.0] = 1.0
    features = np.ascontiguousarray((features - feature_mean) / feature_std, dtype=np.float32)

    target_mean = target.mean(axis=0, keepdims=True)
    target_std = target.std(axis=0, keepdims=True)
    target_std[target_std == 0.0] = 1.0
    target = np.ascontiguousarray((target - target_mean) / target_std, dtype=np.float32)
    return features, target


def _airfoil_cv3_indices(num_examples: int, *, seed: int = 746):
    assert num_examples >= 1000
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_examples)
    test_count = int(round(num_examples * 0.10))
    test_indices = np.ascontiguousarray(indices[:test_count], dtype=np.int64)
    cv_indices = indices[test_count:]
    validate_folds = [np.ascontiguousarray(fold, dtype=np.int64) for fold in np.array_split(cv_indices, 3)]
    folds = []
    for fold_index, validate_indices in enumerate(validate_folds):
        train_indices = np.concatenate([fold for index, fold in enumerate(validate_folds) if index != fold_index])
        folds.append(
            {
                "fold_index": fold_index,
                "train_indices": np.ascontiguousarray(train_indices, dtype=np.int64),
                "validate_indices": validate_indices,
                "test_indices": test_indices,
            })
    return folds, test_indices


def _airfoil_loader_from_indices(
    features: np.ndarray,
    target: np.ndarray,
    *,
    train_indices: np.ndarray,
    validate_indices: np.ndarray,
    batch_size: int,
    dataset_name: str,
):
    return thor.training.NumpyFloat32BatchLoader(
        np.ascontiguousarray(features[train_indices], dtype=np.float32),
        np.ascontiguousarray(target[train_indices], dtype=np.float32),
        np.ascontiguousarray(features[validate_indices], dtype=np.float32),
        np.ascontiguousarray(target[validate_indices], dtype=np.float32),
        batch_size=batch_size,
        example_input_name="examples",
        label_input_name="demand",
        dataset_name=dataset_name,
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


def _categorical_mixed_labels_arrays(*, dtype=np.float32):
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


def _categorical_mixed_labels_one_batch_loader(*, dtype=np.float32):
    x, y = _categorical_mixed_labels_arrays(dtype=dtype)
    loader_cls = thor.training.NumpyFloat16BatchLoader if dtype == np.float16 else thor.training.NumpyFloat32BatchLoader
    return loader_cls(
        x,
        y,
        x,
        y,
        batch_size=4,
        example_input_name="examples",
        label_input_name="class_targets",
        dataset_name="training_runs_categorical_mixed_labels_one_batch",
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
        loss_weight=2.0,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction", predictions.get_feature_output(), thor.DataType.fp32)
    return network


def _build_tiny_regressor_with_label_mean_report(name: str):
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
    true_mean = thor.metrics.Mean(network, labels.get_feature_output())
    prediction_mean = thor.metrics.Mean(network, predictions.get_feature_output())
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction", predictions.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "true_mean", true_mean.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction_mean", prediction_mean.get_metric(), thor.DataType.fp32)
    return network


def _build_tiny_regressor_with_hidden_metric_report(name: str):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], thor.DataType.fp32)
    context = thor.layers.NetworkInput(network, "context", [2], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32)

    hidden = thor.layers.CustomLayer(
        network=network,
        inputs={
            "examples": examples.get_feature_output(),
            "context": context.get_feature_output(),
        },
        output_names=["hidden"],
        build=lambda layer_context: {
            "hidden": layer_context.input("examples") + layer_context.input("context"),
        },
    )
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
    hidden_mean = thor.metrics.Mean(network, hidden["hidden"])
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction", predictions.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "hidden_mean", hidden_mean.get_metric(), thor.DataType.fp32)
    return network


def _build_tiny_regressor_with_hidden_loss_report(name: str):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32)

    hidden_prediction = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        1,
        True,
        activation=None,
        weights_initializer=thor.initializers.UniformRandom(0.0, 0.0),
        biases_initializer=thor.initializers.UniformRandom(0.0, 0.0),
    )
    deployable_prediction = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        1,
        True,
        activation=None,
        weights_initializer=thor.initializers.UniformRandom(0.0, 0.0),
        biases_initializer=thor.initializers.UniformRandom(0.0, 0.0),
    )
    hidden_loss = thor.losses.MSE(
        network,
        hidden_prediction.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "hidden_loss", hidden_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction", deployable_prediction.get_feature_output(), thor.DataType.fp32)
    return network


def _build_weighted_tiny_regressor(name: str):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], thor.DataType.fp32)
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32)
    example_weights = thor.layers.NetworkInput(network, "example_weights", [1], thor.DataType.fp32)

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
        example_weights=example_weights.get_feature_output(),
    )
    thor.layers.NetworkOutput(network, "weighted_mse_loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction", predictions.get_feature_output(), thor.DataType.fp32)
    return network


def _build_named_graph_loss_regressor(name: str):
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
    graph_loss = thor.losses.MSE(
        network,
        predictions.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
        loss_weight=2.0,
    )
    # Name the graph loss through the graph itself.  This test is about rejecting
    # synthetic prediction-output metrics, not about the later generated-name
    # cleanup for losses that do not yet have a graph-owned display name.
    thor.layers.NetworkOutput(network, "graph_loss", graph_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction", predictions.get_feature_output(), thor.DataType.fp32)
    return network


def _build_two_loss_regressor(name: str):
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
    mse = thor.losses.MSE(
        network,
        predictions.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
        loss_weight=2.0,
    )
    mae = thor.losses.MAE(
        network,
        predictions.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
        loss_weight=3.0,
    )
    thor.layers.NetworkOutput(network, "mae_loss", mae.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "mse_loss", mse.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "prediction", predictions.get_feature_output(), thor.DataType.fp32)
    return network


def _zero_initialized_regression_head(network: thor.Network, features: thor.Tensor):
    return thor.layers.FullyConnected(
        network,
        features,
        1,
        True,
        activation=None,
        weights_initializer=thor.initializers.UniformRandom(0.0, 0.0),
        biases_initializer=thor.initializers.UniformRandom(0.0, 0.0),
    )


def _build_mae_low_high_quantile_regressor(name: str):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], thor.DataType.fp32)
    demand = thor.layers.NetworkInput(network, "demand", [1], thor.DataType.fp32)

    point_forecast = _zero_initialized_regression_head(network, examples.get_feature_output())
    low_quantile_forecast = _zero_initialized_regression_head(network, examples.get_feature_output())
    high_quantile_forecast = _zero_initialized_regression_head(network, examples.get_feature_output())

    mae = thor.losses.MAE(
        network,
        point_forecast.get_feature_output(),
        demand.get_feature_output(),
        thor.DataType.fp32,
    )
    low_quantile = thor.losses.QuantileLoss(
        network,
        low_quantile_forecast.get_feature_output(),
        demand.get_feature_output(),
        0.1,
        thor.DataType.fp32,
    )
    high_quantile = thor.losses.QuantileLoss(
        network,
        high_quantile_forecast.get_feature_output(),
        demand.get_feature_output(),
        0.9,
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "mae_loss", mae.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "quantile_low_loss", low_quantile.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "quantile_high_loss", high_quantile.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "forecast", point_forecast.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "forecast_p10", low_quantile_forecast.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "forecast_p90", high_quantile_forecast.get_feature_output(), thor.DataType.fp32)
    return network


def _build_airfoil_mae_low_high_quantile_regressor(name: str, *, width: int = 16):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [5], thor.DataType.fp32)
    demand = thor.layers.NetworkInput(network, "demand", [1], thor.DataType.fp32)

    hidden = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        width,
        True,
        activation=thor.activations.Relu(),
    )
    point_forecast = thor.layers.FullyConnected(
        network,
        hidden.get_feature_output(),
        1,
        True,
        activation=None,
    )
    low_quantile_forecast = thor.layers.FullyConnected(
        network,
        hidden.get_feature_output(),
        1,
        True,
        activation=None,
    )
    high_quantile_forecast = thor.layers.FullyConnected(
        network,
        hidden.get_feature_output(),
        1,
        True,
        activation=None,
    )

    mae = thor.losses.MAE(
        network,
        point_forecast.get_feature_output(),
        demand.get_feature_output(),
        thor.DataType.fp32,
    )
    low_quantile = thor.losses.QuantileLoss(
        network,
        low_quantile_forecast.get_feature_output(),
        demand.get_feature_output(),
        0.1,
        thor.DataType.fp32,
    )
    high_quantile = thor.losses.QuantileLoss(
        network,
        high_quantile_forecast.get_feature_output(),
        demand.get_feature_output(),
        0.9,
        thor.DataType.fp32,
    )
    mae_accuracy = thor.metrics.LossMetric(
        network,
        point_forecast.get_feature_output(),
        demand.get_feature_output(),
        formula=thor.metrics.LossFormula.mean_absolute_error,
        display_name="MAE",
    )
    thor.layers.NetworkOutput(network, "mae_loss", mae.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "quantile_low_loss", low_quantile.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "quantile_high_loss", high_quantile.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "mae_accuracy", mae_accuracy.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "forecast", point_forecast.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "forecast_p10", low_quantile_forecast.get_feature_output(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "forecast_p90", high_quantile_forecast.get_feature_output(), thor.DataType.fp32)
    return network


def _select_airfoil_repeated_two_batch_indices(
    target: np.ndarray,
    source_indices: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return two identical train batches plus the one-batch probe baseline.

    The two-phase gradient checks use zeroed features so the forecast heads are
    bias-only and deterministic.  Pick a deterministic high-mean slice from the
    fold so the zero-initialized point head receives a clear non-zero gradient.
    """
    source_indices = np.asarray(source_indices, dtype=np.int64)
    assert source_indices.shape[0] >= batch_size
    target_values = np.asarray(target[source_indices, 0], dtype=np.float32)
    order = np.argsort(target_values)
    probe_indices = np.ascontiguousarray(source_indices[order[-batch_size:]], dtype=np.int64)
    train_indices = np.ascontiguousarray(np.concatenate([probe_indices, probe_indices]), dtype=np.int64)
    baseline_mse = float(np.mean(np.square(target[probe_indices], dtype=np.float32)))
    assert baseline_mse > 0.0
    return train_indices, probe_indices, baseline_mse


def _build_airfoil_two_phase_joint_quantile_regressor(
    name: str,
) -> tuple[thor.Network, thor.training.TrainingProgram, thor.training.TrainingPhase]:
    network = thor.Network(name)

    point_network = thor.Network(f"{name}_point_forecast_phase")
    point_examples = thor.layers.NetworkInput(point_network, "examples", [5], thor.DataType.fp32)
    point_demand = thor.layers.NetworkInput(point_network, "demand", [1], thor.DataType.fp32)

    zero = thor.initializers.UniformRandom(0.0, 0.0)
    point_forecast = thor.layers.FullyConnected(
        point_network,
        point_examples.get_feature_output(),
        1,
        True,
        activation=None,
        weights_initializer=zero,
        biases_initializer=zero,
    )
    point_mse = thor.losses.MSE(
        point_network,
        point_forecast.get_feature_output(),
        point_demand.get_feature_output(),
        thor.DataType.fp32,
    )
    point_mae = thor.metrics.LossMetric(
        point_network,
        point_forecast.get_feature_output(),
        point_demand.get_feature_output(),
        formula=thor.metrics.LossFormula.mean_absolute_error,
        display_name="Point MAE",
    )

    thor.layers.NetworkOutput(point_network, "point_mse_loss", point_mse.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(point_network, "point_mae_accuracy", point_mae.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(point_network, "forecast", point_forecast.get_feature_output(), thor.DataType.fp32)

    joint_network = thor.Network(f"{name}_joint_and_quantile_phase")
    joint_examples = thor.layers.NetworkInput(joint_network, "examples", [5], thor.DataType.fp32)
    joint_demand = thor.layers.NetworkInput(joint_network, "demand", [1], thor.DataType.fp32)
    point_forecast_input = thor.layers.NetworkInput(
        joint_network,
        "forecast",
        [1],
        thor.DataType.fp32,
        external=False,
    )

    high_quantile_forecast = thor.layers.FullyConnected(
        joint_network,
        joint_examples.get_feature_output(),
        1,
        True,
        activation=None,
        weights_initializer=zero,
        biases_initializer=zero,
    )

    # Phase 2 intentionally depends on the phase-1 forecast through a distinct
    # CustomLayer and uses the same optimum, but it must not be numerically
    # identical to point_mse_loss.  Scaling both prediction and target by 2 keeps
    # the objective compatible with point_mse_loss while making the reported
    # value exactly 4x the point loss and the MAE metric exactly 2x the point
    # MAE.  Keep the prediction and target scaling in separate CustomLayers: the
    # graph-loss discovery code must still be able to identify ``demand`` as a
    # training-only label input, which would fail if a single multi-input
    # CustomLayer mixed the prediction path and label path.
    joint_forecast_projection = thor.layers.CustomLayer(
        network=joint_network,
        inputs={
            "point_forecast": point_forecast_input.get_feature_output()
        },
        output_names=["joint_forecast"],
        build=lambda context: {
            "joint_forecast": context.input("point_forecast") * 2.0
        },
    )
    joint_demand_projection = thor.layers.CustomLayer(
        network=joint_network,
        inputs={
            "demand": joint_demand.get_feature_output()
        },
        output_names=["joint_demand"],
        build=lambda context: {
            "joint_demand": context.input("demand") * 2.0
        },
    )
    joint_mse = thor.losses.MSE(
        joint_network,
        joint_forecast_projection["joint_forecast"],
        joint_demand_projection["joint_demand"],
        thor.DataType.fp32,
    )
    high_quantile = thor.losses.QuantileLoss(
        joint_network,
        high_quantile_forecast.get_feature_output(),
        joint_demand.get_feature_output(),
        0.9,
        thor.DataType.fp32,
    )
    joint_mae = thor.metrics.LossMetric(
        joint_network,
        joint_forecast_projection["joint_forecast"],
        joint_demand_projection["joint_demand"],
        formula=thor.metrics.LossFormula.mean_absolute_error,
        display_name="Joint MAE",
    )

    thor.layers.NetworkOutput(joint_network, "joint_mse_loss", joint_mse.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(joint_network, "quantile_high_loss", high_quantile.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(joint_network, "joint_mae_accuracy", joint_mae.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(
        joint_network, "forecast_p90", high_quantile_forecast.get_feature_output(), thor.DataType.fp32)

    point_phase = thor.training.TrainingPhase(
        "point_forecast",
        network=point_network,
        enabled=True,
    )
    second_phase = thor.training.TrainingPhase(
        "joint_and_quantile",
        network=joint_network,
        enabled=False,
    )
    step = thor.training.TrainingStep(
        "airfoil_two_phase_step",
        phases=[point_phase, second_phase],
        optimizer=thor.optimizers.Sgd(initial_learning_rate=0.02, momentum=0.0),
    )
    program = thor.training.TrainingProgram([step])
    return network, program, second_phase


def _build_airfoil_two_phase_mae_then_mse_regressor(
    name: str,
    *,
    width: int = 32,
) -> tuple[thor.Network, thor.training.TrainingProgram, thor.training.TrainingPhase]:
    """Real two-phase Airfoil model: MAE pretrain, then MSE head using MAE forecast.

    Phase 1 is an ordinary network that exports hidden state and the MAE forecast.
    Phase 2 is another ordinary network whose non-external inputs are satisfied by
    those phase-1 NetworkOutputs when both phases are enabled.
    """
    network = thor.Network(name)

    mae_network = thor.Network(f"{name}_mae_pretrain_phase")
    mae_examples = thor.layers.NetworkInput(mae_network, "examples", [5], thor.DataType.fp32)
    mae_demand = thor.layers.NetworkInput(mae_network, "demand", [1], thor.DataType.fp32)

    hidden = thor.layers.FullyConnected(
        mae_network,
        mae_examples.get_feature_output(),
        width,
        True,
        activation=thor.activations.Relu(),
    )
    hidden = thor.layers.FullyConnected(
        mae_network,
        hidden.get_feature_output(),
        width,
        True,
        activation=thor.activations.Relu(),
    )
    mae_forecast = thor.layers.FullyConnected(
        mae_network,
        hidden.get_feature_output(),
        1,
        True,
        activation=None,
    )
    mae_loss = thor.losses.MAE(
        mae_network,
        mae_forecast.get_feature_output(),
        mae_demand.get_feature_output(),
        thor.DataType.fp32,
    )
    mae_head_mae = thor.metrics.LossMetric(
        mae_network,
        mae_forecast.get_feature_output(),
        mae_demand.get_feature_output(),
        formula=thor.metrics.LossFormula.mean_absolute_error,
        display_name="MAE Head MAE",
    )

    thor.layers.NetworkOutput(mae_network, "mae_loss", mae_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(mae_network, "mae_head_mae_accuracy", mae_head_mae.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(
        mae_network,
        "hidden",
        hidden.get_feature_output(),
        thor.DataType.fp32,
        external=False,
    )
    # ``mae_forecast`` is both a phase-local dependency consumed by the MSE
    # phase and the prediction tensor for ``mae_loss``/``mae_head_mae_accuracy``
    # ensemble evaluation, so keep it external/materialized.  Pure internal
    # exports, such as ``hidden``, stay ``external=False``.
    thor.layers.NetworkOutput(
        mae_network,
        "mae_forecast",
        mae_forecast.get_feature_output(),
        thor.DataType.fp32,
    )

    mse_network = thor.Network(f"{name}_mse_finetune_phase")
    mse_demand = thor.layers.NetworkInput(mse_network, "demand", [1], thor.DataType.fp32)
    mse_hidden_input = thor.layers.NetworkInput(
        mse_network,
        "hidden",
        [width],
        thor.DataType.fp32,
        external=False,
    )
    mse_mae_forecast_input = thor.layers.NetworkInput(
        mse_network,
        "mae_forecast",
        [1],
        thor.DataType.fp32,
        external=False,
    )
    mse_features = thor.layers.Concatenate(
        mse_network,
        [mse_hidden_input.get_feature_output(),
         mse_mae_forecast_input.get_feature_output()],
        0,
    )
    mse_hidden = thor.layers.FullyConnected(
        mse_network,
        mse_features.get_feature_output(),
        width,
        True,
        activation=thor.activations.Relu(),
    )
    mse_forecast = thor.layers.FullyConnected(
        mse_network,
        mse_hidden.get_feature_output(),
        1,
        True,
        activation=None,
    )
    mse_loss = thor.losses.MSE(
        mse_network,
        mse_forecast.get_feature_output(),
        mse_demand.get_feature_output(),
        thor.DataType.fp32,
    )
    mse_head_mae = thor.metrics.LossMetric(
        mse_network,
        mse_forecast.get_feature_output(),
        mse_demand.get_feature_output(),
        formula=thor.metrics.LossFormula.mean_absolute_error,
        display_name="MSE Head MAE",
    )

    thor.layers.NetworkOutput(mse_network, "mse_loss", mse_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(mse_network, "mse_head_mae_accuracy", mse_head_mae.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(mse_network, "mse_forecast", mse_forecast.get_feature_output(), thor.DataType.fp32)

    mae_phase = thor.training.TrainingPhase(
        "mae_pretrain",
        network=mae_network,
        enabled=True,
    )
    mse_phase = thor.training.TrainingPhase(
        "mse_finetune",
        network=mse_network,
        enabled=False,
    )
    step = thor.training.TrainingStep(
        "airfoil_mae_then_mse_step",
        phases=[mae_phase, mse_phase],
        optimizer=thor.optimizers.Adam(),
    )
    return network, thor.training.TrainingProgram([step]), mse_phase


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


def _make_signature_only_trainer(
    name: str,
    *,
    input_dtype=thor.DataType.fp32,
    output_dtype=thor.DataType.fp32,
    save_model_dir=None,
    save_model_overwrite=False,
):
    return thor.training.Trainer(
        _build_signature_only_network(name, input_dtype=input_dtype, output_dtype=output_dtype),
        _regression_one_batch_loader(),
        save_model_dir=save_model_dir,
        save_model_overwrite=save_model_overwrite,
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


def _make_non_finite_tiny_regression_trainer(
    name: str,
    non_finite_phase: str,
    *,
    restart_conditions=None,
):
    return thor.training.Trainer(
        _build_tiny_regressor(name),
        _non_finite_regression_one_batch_loader(non_finite_phase),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=1.0e-12, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
        restart_conditions=restart_conditions,
    )


def _make_tiny_regression_with_label_mean_report_trainer(
    name: str,
    *,
    save_model_dir=None,
    save_model_overwrite=False,
):
    return thor.training.Trainer(
        _build_tiny_regressor_with_label_mean_report(name),
        _regression_one_batch_loader(),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["true_mean", "prediction_mean"],
        stats_color="never",
        save_model_dir=save_model_dir,
        save_model_overwrite=save_model_overwrite,
    )


def _make_tiny_regression_with_hidden_metric_report_trainer(
    name: str,
):
    return thor.training.Trainer(
        _build_tiny_regressor_with_hidden_metric_report(name),
        _regression_one_batch_loader(),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["hidden_mean"],
        stats_color="never",
    )


def _make_tiny_regression_with_hidden_loss_report_trainer(
    name: str,
):
    return thor.training.Trainer(
        _build_tiny_regressor_with_hidden_loss_report(name),
        _regression_one_batch_loader(),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["hidden_loss"],
        stats_color="never",
    )


def _make_weighted_tiny_regression_trainer(
    name: str,
    *,
    save_model_dir=None,
    save_model_overwrite=False,
):
    return thor.training.Trainer(
        _build_weighted_tiny_regressor(name),
        _weighted_regression_one_batch_loader(),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=1.0e-12, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["weighted_mse_loss"],
        stats_color="never",
        save_model_dir=save_model_dir,
        save_model_overwrite=save_model_overwrite,
    )


def _make_named_graph_loss_regression_trainer(
    name: str,
    *,
    save_model_dir=None,
    save_model_overwrite=False,
):
    return thor.training.Trainer(
        _build_named_graph_loss_regressor(name),
        _regression_one_batch_loader(),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=1.0e-12, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=[],
        stats_color="never",
        save_model_dir=save_model_dir,
        save_model_overwrite=save_model_overwrite,
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: 0.0,
    )


def _make_two_loss_regression_trainer(
    name: str,
    *,
    save_model_dir=None,
    save_model_overwrite=False,
):
    return thor.training.Trainer(
        _build_two_loss_regressor(name),
        _regression_one_batch_loader(),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=1.0e-12, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["mse_loss", "mae_loss"],
        stats_color="never",
        save_model_dir=save_model_dir,
        save_model_overwrite=save_model_overwrite,
    )


def _make_mae_low_high_quantile_regression_trainer(
    name: str,
    *,
    save_model_dir=None,
    save_model_overwrite=False,
):
    return thor.training.Trainer(
        _build_mae_low_high_quantile_regressor(name),
        _mae_quantile_regression_one_batch_loader(),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=1.0e-12, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["mae_loss", "quantile_low_loss", "quantile_high_loss"],
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


def test_training_runs_binding_accepts_reported_losses():
    trainer = _make_tiny_regression_trainer("training_runs_binding_reported_losses")

    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "tiny_ensemble")],
        reported_losses={
            "tiny_ensemble": ["loss"]
        },
    )

    assert runs.reported_losses["tiny_ensemble"] == ["loss"]
    assert not hasattr(runs, "ensemble_metrics")
    assert not hasattr(thor.training, "MetricSpec")


def test_training_runs_reported_losses_survive_label_mean_report_metric():
    trainer = _make_tiny_regression_with_label_mean_report_trainer(
        "training_runs_reported_losses_with_label_mean_report"
    )

    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "tiny_ensemble")],
        reported_losses={
            "tiny_ensemble": ["loss"]
        },
    )

    assert runs.reported_losses["tiny_ensemble"] == ["loss"]


def test_training_runs_reported_metrics_discovers_metric_outputs_without_prediction_role():
    trainer = _make_tiny_regression_with_label_mean_report_trainer(
        "training_runs_reported_metrics_with_label_mean_report"
    )

    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "tiny_ensemble")],
        reported_metrics={
            "tiny_ensemble": ["true_mean", "prediction_mean"]
        },
    )

    assert runs.reported_metrics["tiny_ensemble"] == ["true_mean", "prediction_mean"]


def test_training_runs_reported_metrics_keeps_explicit_hidden_metric_output():
    trainer = _make_tiny_regression_with_hidden_metric_report_trainer(
        "training_runs_reported_metrics_with_hidden_metric_report"
    )

    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "tiny_ensemble")],
        reported_metrics={
            "tiny_ensemble": ["hidden_mean"]
        },
    )

    assert runs.reported_metrics["tiny_ensemble"] == ["hidden_mean"]


def test_training_runs_reported_losses_keeps_explicit_hidden_loss_output():
    trainer = _make_tiny_regression_with_hidden_loss_report_trainer(
        "training_runs_reported_losses_with_hidden_loss_report"
    )

    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "tiny_ensemble")],
        reported_losses={
            "tiny_ensemble": ["hidden_loss"]
        },
    )

    assert runs.reported_losses["tiny_ensemble"] == ["hidden_loss"]


def _make_two_phase_trainer_with_inactive_future_reports(name: str):
    base_network = thor.Network(name)

    first_network = thor.Network(f"{name}_first_phase")
    first_examples = thor.layers.NetworkInput(first_network, "examples", [2], thor.DataType.fp32)
    first_labels = thor.layers.NetworkInput(first_network, "labels", [1], thor.DataType.fp32)
    first_prediction = thor.layers.FullyConnected(
        first_network,
        first_examples.get_feature_output(),
        1,
        True,
        activation=None,
    )
    first_loss = thor.losses.MAE(
        first_network,
        first_prediction.get_feature_output(),
        first_labels.get_feature_output(),
        thor.DataType.fp32,
    )
    first_metric = thor.metrics.LossMetric(
        first_network,
        first_prediction.get_feature_output(),
        first_labels.get_feature_output(),
        formula=thor.metrics.LossFormula.mean_absolute_error,
        display_name="first MAE",
    )
    thor.layers.NetworkOutput(first_network, "first_loss", first_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(first_network, "first_metric", first_metric.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(
        first_network, "first_prediction", first_prediction.get_feature_output(), thor.DataType.fp32)

    second_network = thor.Network(f"{name}_second_phase")
    second_labels = thor.layers.NetworkInput(second_network, "labels", [1], thor.DataType.fp32)
    first_prediction_input = thor.layers.NetworkInput(
        second_network,
        "first_prediction",
        [1],
        thor.DataType.fp32,
        external=False,
    )
    second_prediction = thor.layers.FullyConnected(
        second_network,
        first_prediction_input.get_feature_output(),
        1,
        True,
        activation=None,
    )
    second_loss = thor.losses.MSE(
        second_network,
        second_prediction.get_feature_output(),
        second_labels.get_feature_output(),
        thor.DataType.fp32,
    )
    second_metric = thor.metrics.LossMetric(
        second_network,
        second_prediction.get_feature_output(),
        second_labels.get_feature_output(),
        formula=thor.metrics.LossFormula.mean_absolute_error,
        display_name="second MAE",
    )
    thor.layers.NetworkOutput(second_network, "second_loss", second_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(second_network, "second_metric", second_metric.get_metric(), thor.DataType.fp32)
    thor.layers.NetworkOutput(
        second_network, "second_prediction", second_prediction.get_feature_output(), thor.DataType.fp32)

    first_phase = thor.training.TrainingPhase("first", network=first_network)
    second_phase = thor.training.TrainingPhase("second", network=second_network, enabled=False)
    program = thor.training.TrainingProgram(
        [
            thor.training.TrainingStep(
                "two_phase_step",
                phases=[first_phase, second_phase],
                optimizer=thor.optimizers.Sgd(initial_learning_rate=1.0e-12, momentum=0.0),
            )
        ])

    return thor.training.Trainer(
        base_network,
        _regression_one_batch_loader(),
        training_program=program,
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        scalar_tensors_to_report=["first_loss", "second_loss"],
        stats_color="never",
    )


def test_training_runs_accepts_reported_names_from_inactive_future_phase():
    trainer = _make_two_phase_trainer_with_inactive_future_reports("training_runs_future_phase_reported_names")

    runs = thor.training.TrainingRuns(
        [("fold_0", trainer, "two_phase_ensemble")],
        reported_losses={
            "two_phase_ensemble": ["first_loss", "second_loss"],
        },
        reported_metrics={
            "two_phase_ensemble": ["first_metric", "second_metric"],
        },
    )

    assert runs.reported_losses["two_phase_ensemble"] == ["first_loss", "second_loss"]
    assert runs.reported_metrics["two_phase_ensemble"] == ["first_metric", "second_metric"]


def test_training_runs_binding_rejects_invalid_reported_losses():
    trainer = _make_tiny_regression_trainer("training_runs_binding_invalid_reported_losses")

    with pytest.raises(RuntimeError, match="requested reported loss .*missing"):
        thor.training.TrainingRuns(
            [("fold_0", trainer, "tiny_ensemble")],
            reported_losses={
                "tiny_ensemble": ["missing"]
            },
        )

    with pytest.raises(TypeError, match="reported_losses"):
        thor.training.TrainingRuns(
            [("fold_0", trainer, "tiny_ensemble")],
            reported_losses={
                "tiny_ensemble": [{
                    "name": "loss"
                }]
            },
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
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: validation_loss if validation_loss is not None else training_loss,
    )

    assert trainer is not None


def test_trainer_binding_rejects_non_callable_model_selection_score():
    with pytest.raises(TypeError, match="model_selection_score"):
        _make_tiny_regression_trainer(
            "training_runs_custom_model_selection_score_invalid",
            model_selection_score=3.0,
        )


def test_restart_policy_binding_defaults_and_aliases():
    condition = thor.training.RestartPolicy()

    assert isinstance(condition, thor.training.TrainingRestartPolicy)
    assert condition.run_name is None
    assert condition.ensemble_group is None
    assert condition.progress_check_epochs == 3
    assert condition.progress_improvement_min_percentage == 5.0
    assert condition.max_restarts == 5
    assert thor.training.RestartCondition is thor.training.RestartPolicy
    assert thor.training.TrainingRestartCondition is thor.training.RestartPolicy
    assert thor.training.TrainingRunsRestartPolicy is thor.training.RestartPolicy

    trainer = _make_tiny_regression_trainer(
        "trainer_restart_policy_binding",
        restart_conditions=[
            thor.training.RestartPolicy(progress_check_epochs=2, progress_improvement_min_percentage=10.0)
        ],
    )

    assert trainer is not None


def test_trainer_restart_policy_warns_and_ignores_training_runs_targets():
    with pytest.warns(RuntimeWarning, match="ignore RestartPolicy.*ensemble_group"):
        trainer = _make_tiny_regression_trainer(
            "trainer_restart_policy_ignores_group_target",
            restart_conditions=[
                thor.training.RestartPolicy(
                    ensemble_group="ignored_group",
                    progress_check_epochs=2,
                    progress_improvement_min_percentage=10.0,
                )
            ],
        )

    assert trainer is not None


def test_training_runs_restart_policy_binding_defaults_and_validation():
    condition = thor.training.RestartPolicy(run_name="fold_0")

    assert isinstance(condition, thor.training.TrainingRestartPolicy)
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


def test_training_runs_restart_policy_accepts_global_untargeted_policy():
    trainer0 = _make_tiny_regression_trainer("training_runs_restart_policy_global_0")
    trainer1 = _make_tiny_regression_trainer("training_runs_restart_policy_global_1")
    runs = thor.training.TrainingRuns(
        [("fold_0", trainer0), ("fold_1", trainer1)],
        failure_policy="continue",
        restart_conditions=[
            thor.training.RestartPolicy(
                progress_check_epochs=2,
                progress_improvement_min_percentage=10.0,
                max_restarts=1,
            )
        ],
    )

    assert runs is not None


def test_training_runs_restart_policy_accepts_ensemble_group_target():
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


def test_training_runs_restart_policy_accepts_multiple_conditions_for_same_group():
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


def test_training_runs_restart_policy_accepts_multiple_conditions_for_same_run():
    trainer = _make_tiny_regression_trainer("training_runs_restart_condition_run_multiple")
    runs = thor.training.TrainingRuns(
        [("fold_0", trainer)],
        failure_policy="continue",
        restart_conditions=[
            thor.training.RestartPolicy(run_name="fold_0", progress_check_epochs=3),
            thor.training.RestartPolicy(
                run_name="fold_0", progress_check_epochs=15, progress_improvement_min_percentage=20.0),
        ],
    )

    assert runs is not None


def test_trainer_early_completion_policy_binding_accepts_callable():
    policy = thor.training.EarlyCompletionPolicy(
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch - best_epoch >= 1)
    assert isinstance(policy, thor.training.TrainingEarlyCompletionPolicy)

    trainer = _make_tiny_regression_trainer(
        "trainer_early_completion_policy_binding",
        early_completion_policies=[policy],
    )

    assert trainer is not None


def test_training_runs_early_completion_rule_binding_defaults_and_validation():
    rule = thor.training.EarlyCompletionRule(
        lambda current_score,
        best_score,
        current_epoch,
        best_epoch: False,
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
                    lambda current_score,
                    best_score,
                    current_epoch,
                    best_epoch: False,
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
                lambda current_score,
                best_score,
                current_epoch,
                best_epoch: False,
                ensemble_group="tiny_ensemble",
            )
        ],
    )

    assert runs is not None




def test_training_runs_early_completion_callback_cycle_is_collectable():
    class Owner:
        def __init__(self):
            self.stop_after_epochs = 1
            trainer = _make_tiny_regression_trainer("training_runs_callback_cycle_collectable")

            def stop_when_stale(current_score, best_score, current_epoch, best_epoch):
                del current_score, best_score
                return current_epoch >= best_epoch + self.stop_after_epochs

            self.runs = thor.training.TrainingRuns(
                [("fold_0", trainer, "tiny_ensemble")],
                failure_policy="continue",
                early_completion_rules=[
                    thor.training.EarlyCompletionRule(
                        stop_when_stale,
                        ensemble_group="tiny_ensemble",
                    )
                ],
            )

    owner = Owner()
    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()

    assert owner_ref() is None


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
        min_successful_models={
            "tiny_ensemble": 1
        },
    )

    assert runs is not None


def test_training_runs_rejects_min_successful_models_unknown_group():
    trainer0 = _make_signature_only_trainer("training_runs_min_success_unknown_0")
    trainer1 = _make_signature_only_trainer("training_runs_min_success_unknown_1")

    with pytest.raises(RuntimeError, match="unknown ensemble_group"):
        thor.training.TrainingRuns(
            [("fold_0", trainer0, "tiny_ensemble"), ("fold_1", trainer1, "tiny_ensemble")],
            min_successful_models={
                "tinny_ensemble": 1
            },
        )


def test_training_runs_rejects_min_successful_models_larger_than_group():
    trainer0 = _make_signature_only_trainer("training_runs_min_success_too_large_0")
    trainer1 = _make_signature_only_trainer("training_runs_min_success_too_large_1")

    with pytest.raises(RuntimeError, match="only has 2 member"):
        thor.training.TrainingRuns(
            [("fold_0", trainer0, "tiny_ensemble"), ("fold_1", trainer1, "tiny_ensemble")],
            min_successful_models={
                "tiny_ensemble": 3
            },
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


@pytest.mark.cuda
@pytest.mark.training_integration
def test_training_runs_weighted_mse_example_weights_drives_training_loss(capfd, tmp_path):
    # This intentionally covers the end-to-end TrainingRuns path, not just loss
    # construction or forward inference.  The regression bug was that a weighted
    # loss could produce an active raw loss tensor that the physical training
    # loss-root logic did not recognize as being driven by a physical loss layer.
    _, _, example_weights = _weighted_regression_arrays(dtype=np.float32)
    expected_weighted_mse = float(np.mean(example_weights))

    runs = thor.training.TrainingRuns(
        [
            (
                "fold_0",
                _make_weighted_tiny_regression_trainer(
                    "training_runs_weighted_mse_example_weights",
                    save_model_dir=tmp_path / "weighted_mse_model",
                    save_model_overwrite=True,
                ),
                "weighted_mse_ensemble",
                1.0,
            )
        ],
        reported_losses={
            "weighted_mse_ensemble": ["weighted_mse_loss"]
        },
    )

    results, captured_text = _fit_runs_and_capture_text(
        runs,
        capfd,
        epochs=1,
        test_loader=_weighted_regression_one_batch_loader(),
    )

    assert results.all_completed(), [
        (result.run_name, result.status, result.result, result.exception_type, result.exception_message)
        for result in results
    ]
    result = results["fold_0"]
    assert result.status == "completed"
    assert result.final_training_loss == pytest.approx(expected_weighted_mse, rel=1e-5, abs=1e-6)
    assert result.final_validation_loss == pytest.approx(expected_weighted_mse, rel=1e-5, abs=1e-6)
    assert result.final_test_loss == pytest.approx(expected_weighted_mse, rel=1e-5, abs=1e-6)
    assert result.final_training_stats.metrics["weighted_mse_loss"] == pytest.approx(
        expected_weighted_mse, rel=1e-5, abs=1e-6)
    assert result.final_validation_stats.metrics["weighted_mse_loss"] == pytest.approx(
        expected_weighted_mse, rel=1e-5, abs=1e-6)
    assert result.final_test_stats.metrics["weighted_mse_loss"] == pytest.approx(
        expected_weighted_mse, rel=1e-5, abs=1e-6)

    ensemble = results.ensemble("weighted_mse_ensemble")
    assert [metric.name for metric in ensemble.named_metrics] == ["weighted_mse_loss"]
    assert ensemble.ensemble_test_loss == pytest.approx(expected_weighted_mse, rel=1e-5, abs=1e-6)

    plain_text = _ANSI_RE.sub("", captured_text)
    assert "train_weighted_mse_loss=" in plain_text
    assert "validate_weighted_mse_loss=" in plain_text
    assert "test_weighted_mse_loss=" in plain_text
    assert "ensemble_test_weighted_mse_loss=" in plain_text


@pytest.mark.cuda
@pytest.mark.training_integration
def test_training_runs_weighted_cross_phase_backprop_after_phase_enable_does_not_crash():
    # Regression reproducer for the SkuForecaster stage-1 -> stage-2 crash.  The
    # important shape is:
    #   phase 1 exports a normal prediction output,
    #   phase 2 later consumes it through an external=False NetworkInput,
    #   phase 2 uses weighted losses, and
    #   there is no StopGradient between the phase-1 output and phase-2 losses.
    #
    # Today this can poison CUDA state with cudaErrorIllegalAddress.  Run the
    # scenario in a child process so the pytest worker survives the abort and can
    # report the failing command output.  Once the Thor-side bug is fixed, the
    # child process should complete successfully.
    reproducer = r'''
import tempfile
from pathlib import Path

import numpy as np
import thor


def make_loader(seed):
    rng = np.random.default_rng(seed)
    n = 24
    trend = np.ascontiguousarray(rng.normal(size=(n, 2)).astype(np.float32))
    seasonality = np.ascontiguousarray(rng.normal(size=(n, 3)).astype(np.float32))
    daily_labels = np.ascontiguousarray((3.0 + rng.random(size=(n, 1)) * 5.0).astype(np.float32))
    aggregate_labels = np.ascontiguousarray((daily_labels * 2.0 + rng.random(size=(n, 1))).astype(np.float32))
    example_weights = np.ascontiguousarray((0.5 + rng.random(size=(n, 1)) * 2.0).astype(np.float32))
    tensors = {
        "trend_inputs": trend,
        "seasonality_inputs": seasonality,
        "forecast_daily_labels": daily_labels,
        "forecast_aggregate_labels": aggregate_labels,
        "example_weights": example_weights,
    }
    return thor.training.NumpyFloat32DictBatchLoader(
        train={name: value.copy() for name, value in tensors.items()},
        validate={name: value.copy() for name, value in tensors.items()},
        batch_size=8,
        dataset_name=f"weighted_cross_phase_backprop_{seed}",
        randomize_train=False,
        batch_queue_depth=4,
    )


def make_network(name):
    root = thor.Network(name)

    daily = thor.Network(f"{name}_daily_phase")
    daily_trend = thor.layers.NetworkInput(daily, "trend_inputs", [2], thor.DataType.fp32)
    daily_seasonality = thor.layers.NetworkInput(daily, "seasonality_inputs", [3], thor.DataType.fp32)
    daily_labels = thor.layers.NetworkInput(daily, "forecast_daily_labels", [1], thor.DataType.fp32)
    daily_weights = thor.layers.NetworkInput(daily, "example_weights", [1], thor.DataType.fp32)
    daily_all = thor.layers.Concatenate(
        daily,
        [daily_trend.get_feature_output(), daily_seasonality.get_feature_output()],
        0,
    ).get_feature_output()
    daily_hidden = thor.layers.FullyConnected(
        daily,
        daily_all,
        16,
        True,
        activation=thor.activations.SoftPlus(),
        weights_initializer=thor.initializers.Glorot(),
        biases_initializer=thor.initializers.Glorot(),
    ).get_feature_output()
    daily_forecast = thor.layers.FullyConnected(
        daily,
        daily_hidden,
        1,
        True,
        activation=thor.activations.SoftPlus(),
        weights_initializer=thor.initializers.Glorot(),
        biases_initializer=thor.initializers.Glorot(),
    ).get_feature_output()
    daily_quantile = thor.layers.FullyConnected(
        daily,
        daily_forecast,
        1,
        False,
        activation=thor.activations.SoftPlus(),
        weights_initializer=thor.initializers.UniformRandom(0.95, 1.05),
    ).get_feature_output()
    daily_loss = thor.losses.MAE(
        daily,
        daily_forecast,
        daily_labels.get_feature_output(),
        thor.DataType.fp32,
        False,
        loss_weight=10.0,
        example_weights=daily_weights.get_feature_output(),
    )
    daily_quantile_loss = thor.losses.QuantileLoss(
        daily,
        daily_quantile,
        daily_labels.get_feature_output(),
        0.85,
        thor.DataType.fp32,
        loss_weight=26.667,
        example_weights=daily_weights.get_feature_output(),
    )
    thor.layers.NetworkOutput(daily, "forecast_daily_loss", daily_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(
        daily,
        "forecast_daily_quantile_high_loss",
        daily_quantile_loss.get_loss(),
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(daily, "forecast_daily", daily_forecast, thor.DataType.fp32)
    thor.layers.NetworkOutput(daily, "forecast_daily_quantile_high", daily_quantile, thor.DataType.fp32)

    aggregate = thor.Network(f"{name}_aggregate_phase")
    aggregate_trend = thor.layers.NetworkInput(aggregate, "trend_inputs", [2], thor.DataType.fp32)
    aggregate_seasonality = thor.layers.NetworkInput(aggregate, "seasonality_inputs", [3], thor.DataType.fp32)
    aggregate_labels = thor.layers.NetworkInput(aggregate, "forecast_aggregate_labels", [1], thor.DataType.fp32)
    aggregate_weights = thor.layers.NetworkInput(aggregate, "example_weights", [1], thor.DataType.fp32)
    daily_forecast_input = thor.layers.NetworkInput(
        aggregate,
        "forecast_daily",
        [1],
        thor.DataType.fp32,
        external=False,
    )
    daily_quantile_input = thor.layers.NetworkInput(
        aggregate,
        "forecast_daily_quantile_high",
        [1],
        thor.DataType.fp32,
        external=False,
    )
    aggregate_features = thor.layers.Concatenate(
        aggregate,
        [
            daily_forecast_input.get_feature_output(),
            aggregate_trend.get_feature_output(),
            aggregate_seasonality.get_feature_output(),
        ],
        0,
    ).get_feature_output()
    aggregate_forecast = thor.layers.FullyConnected(
        aggregate,
        aggregate_features,
        1,
        False,
        activation=thor.activations.SoftPlus(),
        weights_initializer=thor.initializers.UniformRandom(0.95, 1.05),
    ).get_feature_output()
    aggregate_quantile_features = thor.layers.Concatenate(
        aggregate,
        [
            daily_quantile_input.get_feature_output(),
            aggregate_trend.get_feature_output(),
            aggregate_seasonality.get_feature_output(),
        ],
        0,
    ).get_feature_output()
    aggregate_quantile = thor.layers.FullyConnected(
        aggregate,
        aggregate_quantile_features,
        1,
        False,
        activation=thor.activations.SoftPlus(),
        weights_initializer=thor.initializers.UniformRandom(0.95, 1.05),
    ).get_feature_output()
    aggregate_loss = thor.losses.MAE(
        aggregate,
        aggregate_forecast,
        aggregate_labels.get_feature_output(),
        thor.DataType.fp32,
        False,
        loss_weight=1.0,
        example_weights=aggregate_weights.get_feature_output(),
    )
    aggregate_quantile_loss = thor.losses.QuantileLoss(
        aggregate,
        aggregate_quantile,
        aggregate_labels.get_feature_output(),
        0.85,
        thor.DataType.fp32,
        loss_weight=2.6667,
        example_weights=aggregate_weights.get_feature_output(),
    )
    thor.layers.NetworkOutput(aggregate, "forecast_aggregate_loss", aggregate_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(
        aggregate,
        "forecast_aggregate_quantile_high_loss",
        aggregate_quantile_loss.get_loss(),
        thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(aggregate, "forecast_aggregate", aggregate_forecast, thor.DataType.fp32)
    thor.layers.NetworkOutput(aggregate, "forecast_aggregate_quantile_high", aggregate_quantile, thor.DataType.fp32)

    daily_phase = thor.training.TrainingPhase("daily", network=daily, enabled=True)
    aggregate_phase = thor.training.TrainingPhase("aggregate", network=aggregate, enabled=False)
    optimizer = thor.optimizers.Adam(
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        amsgrad=True,
    )
    step = thor.training.TrainingStep(
        "daily_then_aggregate",
        phases=[daily_phase, aggregate_phase],
        optimizer=optimizer,
    )
    return root, thor.training.TrainingProgram([step]), aggregate_phase


def make_trainer(name, seed, model_root):
    network, program, aggregate_phase = make_network(name)
    trainer = thor.training.Trainer(
        network,
        make_loader(seed),
        training_program=program,
        stats_interval_s=0.0,
        max_in_flight_batches=4,
        scalar_tensors_to_report=[
            "forecast_daily_loss",
            "forecast_daily_quantile_high_loss",
            "forecast_aggregate_loss",
            "forecast_aggregate_quantile_high_loss",
        ],
        stats_color="never",
        save_model_dir=model_root / name,
        save_model_overwrite=True,
        save_optimizer_state=True,
        check_best_model_every_epochs=1,
    )
    return trainer, aggregate_phase


loss_names = [
    "forecast_daily_loss",
    "forecast_daily_quantile_high_loss",
    "forecast_aggregate_loss",
    "forecast_aggregate_quantile_high_loss",
]
with tempfile.TemporaryDirectory(prefix="thor_weighted_cross_phase_") as tmp:
    model_root = Path(tmp) / "members"
    run_specs = []
    aggregate_phases = []
    for fold in range(3):
        run_name = f"fold_{fold}"
        trainer, aggregate_phase = make_trainer(f"weighted_cross_phase_{fold}", 1000 + fold, model_root)
        run_specs.append((run_name, trainer, "weighted_cross_phase_cv3", 1.0))
        aggregate_phases.append(aggregate_phase)

    first_runs = thor.training.TrainingRuns(
        run_specs,
        reported_losses={"weighted_cross_phase_cv3": loss_names[:2]},
        max_parallel_runs=3,
    )
    first_result = first_runs.fit(epochs=1)
    if not first_result.all_completed():
        raise RuntimeError("pretrain failed")

    for phase in aggregate_phases:
        phase.enable()

    second_runs = thor.training.TrainingRuns(
        run_specs,
        reported_losses={"weighted_cross_phase_cv3": loss_names},
        max_parallel_runs=3,
    )
    second_result = second_runs.fit(epochs=1)
    if not second_result.all_completed():
        raise RuntimeError("joint fit failed")
'''
    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-c", reproducer],
            text=True,
            capture_output=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "weighted cross-phase backprop reproducer timed out\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}")

    assert completed.returncode == 0, (
        "weighted cross-phase backprop reproducer failed; this is expected "
        "before the Thor-side fix for phase-2 weighted losses backpropagating "
        "through phase-1 non-external outputs.\n"
        f"returncode={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}")


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.parametrize(
    ("non_finite_phase", "message_fragment"),
    [
        ("train", "non-finite training loss"),
        ("validate", "non-finite validation loss"),
    ],
)
def test_training_runs_non_finite_train_or_validation_loss_fails_run(non_finite_phase, message_fragment):
    runs = thor.training.TrainingRuns(
        [
            (
                "fold_0",
                _make_non_finite_tiny_regression_trainer(
                    f"training_runs_non_finite_{non_finite_phase}_loss_fails",
                    non_finite_phase,
                ),
            )
        ],
        failure_policy="continue",
    )

    results = runs.fit(epochs=1)
    result = results["fold_0"]

    assert results.any_failed()
    assert not results.all_completed()
    assert result.status == "failed"
    assert result.result == "failed"
    assert result.exception_type == "TrainingNonFiniteLossDetected"
    assert message_fragment in result.exception_message


@pytest.mark.cuda
@pytest.mark.training_integration
def test_training_runs_non_finite_loss_uses_restart_policy_before_failure():
    runs = thor.training.TrainingRuns(
        [
            (
                "fold_0",
                _make_non_finite_tiny_regression_trainer(
                    "training_runs_non_finite_loss_restart_exhausted",
                    "train",
                ),
            )
        ],
        failure_policy="continue",
        restart_conditions=[
            thor.training.RestartPolicy(
                run_name="fold_0",
                progress_check_epochs=3,
                progress_improvement_min_percentage=5.0,
                max_restarts=1,
            )
        ],
    )

    results = runs.fit(epochs=1)
    result = results["fold_0"]

    assert results.any_failed()
    assert result.status == "failed"
    assert result.exception_type == "TrainingNonFiniteLossDetected"
    assert "non-finite training loss" in result.exception_message
    assert "Restart policy exhausted after 2 failed attempts" in result.exception_message
    assert "max_restarts=1" in result.exception_message


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_default_reported_losses_reports_all_graph_losses(capfd, tmp_path):
    runs = thor.training.TrainingRuns(
        [
            (
                "fold_0",
                _make_two_loss_regression_trainer(
                    "training_runs_all_graph_losses_fold_0",
                    save_model_dir=tmp_path / "fold_0_model",
                    save_model_overwrite=True,
                ),
                "two_loss_ensemble",
                1.0,
            ),
            (
                "fold_1",
                _make_two_loss_regression_trainer(
                    "training_runs_all_graph_losses_fold_1",
                    save_model_dir=tmp_path / "fold_1_model",
                    save_model_overwrite=True,
                ),
                "two_loss_ensemble",
                2.0,
            ),
        ],
    )

    results, captured_text = _fit_runs_and_capture_text(
        runs,
        capfd,
        epochs=1,
        test_loader=_regression_one_batch_loader(),
    )

    assert results.all_completed()
    ensemble = results.ensemble("two_loss_ensemble")
    assert [metric.name for metric in ensemble.named_metrics] == ["mae_loss", "mse_loss"]
    assert all(metric.test_value is not None for metric in ensemble.named_metrics)
    named_test_sum = sum(metric.test_value for metric in ensemble.named_metrics)
    assert ensemble.ensemble_test_loss == pytest.approx(named_test_sum, rel=1e-6, abs=1e-6)

    # The near-zero-learning-rate trainer leaves predictions effectively at zero, so
    # the graph-owned weighted losses are deterministic: MAE=1 * 3.0 and MSE=1 * 2.0.
    by_name = {
        metric.name: metric for metric in ensemble.named_metrics
    }
    assert by_name["mae_loss"].test_value == pytest.approx(3.0, rel=1e-5, abs=1e-6)
    assert by_name["mse_loss"].test_value == pytest.approx(2.0, rel=1e-5, abs=1e-6)
    assert ensemble.ensemble_test_loss == pytest.approx(5.0, rel=1e-5, abs=1e-6)

    plain_text = _ANSI_RE.sub("", captured_text)
    ensemble_line = next(line for line in plain_text.splitlines() if "INFO runs ensemble[two_loss_ensemble]:" in line)
    assert "ensemble_test_mae_loss=" in ensemble_line
    assert "ensemble_test_mse_loss=" in ensemble_line


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_reported_losses_filter_selects_graph_loss_subset(capfd, tmp_path):
    runs = thor.training.TrainingRuns(
        [
            (
                "fold_0",
                _make_two_loss_regression_trainer(
                    "training_runs_filtered_graph_losses_fold_0",
                    save_model_dir=tmp_path / "fold_0_model",
                    save_model_overwrite=True,
                ),
                "two_loss_ensemble",
                1.0,
            )
        ],
        reported_losses={
            "two_loss_ensemble": ["mae_loss"]
        },
    )

    results, captured_text = _fit_runs_and_capture_text(
        runs,
        capfd,
        epochs=1,
        test_loader=_regression_one_batch_loader(),
    )

    assert results.all_completed()
    ensemble = results.ensemble("two_loss_ensemble")
    assert [metric.name for metric in ensemble.named_metrics] == ["mae_loss"]
    assert ensemble.ensemble_test_loss == pytest.approx(ensemble.named_metrics[0].test_value, rel=1e-6, abs=1e-6)
    assert ensemble.ensemble_test_loss == pytest.approx(3.0, rel=1e-5, abs=1e-6)

    ensemble_line = next(
        line for line in _ANSI_RE.sub("", captured_text).splitlines()
        if "INFO runs ensemble[two_loss_ensemble]:" in line)
    assert "ensemble_test_mae_loss=" in ensemble_line
    assert "ensemble_test_mse_loss=" not in ensemble_line


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_reports_mae_plus_low_high_quantile_losses(capfd, tmp_path):
    reported_losses = ["mae_loss", "quantile_low_loss", "quantile_high_loss"]
    runs = thor.training.TrainingRuns(
        [
            (
                "fold_0",
                _make_mae_low_high_quantile_regression_trainer(
                    "training_runs_mae_quantile_losses_fold_0",
                    save_model_dir=tmp_path / "fold_0_model",
                    save_model_overwrite=True,
                ),
                "demand_quantile_ensemble",
                1.0,
            ),
            (
                "fold_1",
                _make_mae_low_high_quantile_regression_trainer(
                    "training_runs_mae_quantile_losses_fold_1",
                    save_model_dir=tmp_path / "fold_1_model",
                    save_model_overwrite=True,
                ),
                "demand_quantile_ensemble",
                2.0,
            ),
        ],
        reported_losses={
            "demand_quantile_ensemble": reported_losses
        },
    )

    results, captured_text = _fit_runs_and_capture_text(
        runs,
        capfd,
        epochs=1,
        test_loader=_mae_quantile_regression_one_batch_loader(),
    )

    assert results.all_completed()
    assert results.has_ensembles
    ensemble = results.ensemble("demand_quantile_ensemble")
    assert ensemble.all_completed()
    assert [metric.name for metric in ensemble.named_metrics] == reported_losses
    assert all(metric.test_value is not None for metric in ensemble.named_metrics)

    by_name = {
        metric.name: metric for metric in ensemble.named_metrics
    }
    assert by_name["mae_loss"].test_value == pytest.approx(3.5, rel=1e-5, abs=1e-6)
    assert by_name["quantile_low_loss"].test_value == pytest.approx(0.35, rel=1e-5, abs=1e-6)
    assert by_name["quantile_high_loss"].test_value == pytest.approx(3.15, rel=1e-5, abs=1e-6)
    assert ensemble.ensemble_test_loss == pytest.approx(7.0, rel=1e-5, abs=1e-6)
    assert ensemble.ensemble_test_loss == pytest.approx(
        sum(metric.test_value for metric in ensemble.named_metrics),
        rel=1e-6,
        abs=1e-6,
    )

    plain_text = _ANSI_RE.sub("", captured_text)
    ensemble_line = next(
        line for line in plain_text.splitlines() if "INFO runs ensemble[demand_quantile_ensemble]:" in line)
    assert "ensemble_test_mae_loss=" in ensemble_line
    assert "ensemble_test_quantile_low_loss=" in ensemble_line
    assert "ensemble_test_quantile_high_loss=" in ensemble_line

    ensemble_artifact_dir = tmp_path / "demand_quantile_ensemble_artifact"
    artifact_path = results.save_ensemble("demand_quantile_ensemble", ensemble_artifact_dir)
    assert artifact_path == str(ensemble_artifact_dir)
    assert not (ensemble_artifact_dir / "ensemble_manifest.json").exists()
    assert not (ensemble_artifact_dir / "members").exists()
    loaded_network = thor.Network.load(str(ensemble_artifact_dir), network_name="ensemble_demand_quantile_ensemble")
    placed_ensemble = loaded_network.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert set(placed_ensemble.get_network_input_names()) == {"examples"}
    x, _ = _mae_quantile_regression_arrays(dtype=np.float32)
    ensemble_outputs = placed_ensemble.infer({
        "examples": _cpu_tensor(x, thor.DataType.fp32)
    })
    assert set(ensemble_outputs) == {"forecast", "forecast_p10", "forecast_p90"}


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests with downloaded Airfoil Self-Noise data",
    ),
)
def test_training_runs_airfoil_cv3_reports_mae_plus_low_high_quantile_losses(capfd, tmp_path):
    features, target = _ensure_airfoil_self_noise_arrays()
    assert features.shape[0] == target.shape[0]
    assert features.shape[1] == 5
    assert target.shape[1] == 1

    folds, holdout_indices = _airfoil_cv3_indices(features.shape[0])
    assert len(folds) == 3
    assert holdout_indices.shape[0] == int(round(features.shape[0] * 0.10))

    reported_losses = ["mae_loss", "quantile_low_loss", "quantile_high_loss"]
    reported_metrics = ["mae_accuracy"]
    batch_size = 128

    def make_fold_trainer(*, fold: dict, run_name: str):
        loader = _airfoil_loader_from_indices(
            features,
            target,
            train_indices=fold["train_indices"],
            validate_indices=fold["validate_indices"],
            batch_size=batch_size,
            dataset_name=f"airfoil_self_noise_cv3_{run_name}",
        )
        assert loader.get_num_train_examples() == int(fold["train_indices"].shape[0])
        assert loader.get_num_validate_examples() == int(fold["validate_indices"].shape[0])
        assert loader.get_num_train_examples() > loader.get_num_validate_examples()
        assert loader.get_num_train_batches() > 1
        assert loader.get_num_validate_batches() > 1
        return thor.training.Trainer(
            _build_airfoil_mae_low_high_quantile_regressor(
                f"airfoil_self_noise_quantile_{run_name}",
                width=16,
            ),
            loader,
            optimizer=thor.optimizers.Adam(),
            stats_interval_s=AIRFOIL_QUANTILE_STATS_INTERVAL_S,
            max_in_flight_batches=2,
            scalar_tensors_to_report=reported_losses,
            stats_color=AIRFOIL_QUANTILE_STATS_COLOR,
            save_model_dir=tmp_path / f"{run_name}_model",
            save_model_overwrite=True,
        )

    run_specs = []
    for fold in folds:
        fold_index = int(fold["fold_index"])
        run_specs.append(
            (
                f"fold_{fold_index}",
                make_fold_trainer(fold=fold, run_name=f"fold_{fold_index}"),
                "airfoil_noise_cv3",
            ))

    runs = thor.training.TrainingRuns(
        run_specs,
        reported_losses={
            "airfoil_noise_cv3": reported_losses,
        },
        reported_metrics={
            "airfoil_noise_cv3": reported_metrics,
        },
        max_parallel_runs=3,
        max_summary_logs_per_second=AIRFOIL_QUANTILE_SUMMARY_LOGS_PER_SECOND,
    )

    test_loader = _airfoil_loader_from_indices(
        features,
        target,
        train_indices=holdout_indices,
        validate_indices=holdout_indices,
        batch_size=batch_size,
        dataset_name="airfoil_self_noise_cv3_holdout_test",
    )
    assert test_loader.get_num_validate_examples() == int(holdout_indices.shape[0])
    assert test_loader.get_num_validate_batches() > 1

    results, captured_text = _fit_runs_and_capture_text(
        runs,
        capfd,
        epochs=5,
        test_loader=test_loader,
    )

    plain_text = _ANSI_RE.sub("", captured_text)
    if _expects_color_for_stats_color_mode(AIRFOIL_QUANTILE_STATS_COLOR):
        assert "\x1b[" in captured_text
    assert len(results) == 3
    assert results.all_completed(), [
        (result.run_name, result.status, result.result, result.exception_type, result.exception_message)
        for result in results
    ]
    assert results.has_ensembles
    assert "INFO runs ensemble[airfoil_noise_cv3]:" in plain_text
    assert "ensemble_train_mae_accuracy=" in plain_text
    assert "ensemble_test_mae_accuracy=" in plain_text

    ensemble = results.ensemble("airfoil_noise_cv3")
    assert ensemble.all_completed()
    assert ensemble.total_weight == pytest.approx(3.0)
    assert len(ensemble.members) == 3
    assert [metric.name for metric in ensemble.named_metrics] == reported_losses
    assert [metric.name for metric in ensemble.reported_metrics] == reported_metrics
    assert ensemble.ensemble_train_loss is not None
    assert ensemble.ensemble_test_loss is not None
    assert math.isfinite(ensemble.ensemble_train_loss)
    assert math.isfinite(ensemble.ensemble_test_loss)
    assert ensemble.ensemble_train_loss > 0.0
    assert ensemble.ensemble_test_loss > 0.0

    graph_metric_by_name = {
        metric.name: metric for metric in ensemble.reported_metrics
    }
    mae_accuracy = graph_metric_by_name["mae_accuracy"]
    assert mae_accuracy.train_value is not None
    assert mae_accuracy.test_value is not None
    assert math.isfinite(mae_accuracy.train_value)
    assert math.isfinite(mae_accuracy.test_value)
    assert mae_accuracy.train_value > 0.0
    assert mae_accuracy.test_value > 0.0

    train_metric_sum = 0.0
    test_metric_sum = 0.0
    for metric in ensemble.named_metrics:
        assert metric.train_value is not None
        assert metric.test_value is not None
        assert math.isfinite(metric.train_value)
        assert math.isfinite(metric.test_value)
        assert metric.train_value > 0.0
        assert metric.test_value > 0.0
        train_metric_sum += metric.train_value
        test_metric_sum += metric.test_value
        assert f"ensemble_train_{metric.name}=" in plain_text
        assert f"ensemble_test_{metric.name}=" in plain_text

    assert ensemble.ensemble_train_loss == pytest.approx(train_metric_sum, rel=1e-5, abs=1e-6)
    assert ensemble.ensemble_test_loss == pytest.approx(test_metric_sum, rel=1e-5, abs=1e-6)

    for fold_index in range(3):
        run_name = f"fold_{fold_index}"
        result = results[run_name]
        assert result.status == "completed"
        assert result.ensemble_group == "airfoil_noise_cv3"
        assert result.ensemble_weight == pytest.approx(1.0)
        assert result.final_training_loss is not None
        assert result.final_validation_loss is not None
        assert result.final_test_loss is not None
        assert result.final_training_stats.metrics["mae_accuracy"] > 0.0
        assert result.final_validation_stats.metrics["mae_accuracy"] > 0.0
        assert result.final_test_stats.metrics["mae_accuracy"] > 0.0
        assert math.isfinite(result.final_training_loss)
        assert math.isfinite(result.final_validation_loss)
        assert math.isfinite(result.final_test_loss)
        assert result.final_training_loss > 0.0
        assert result.final_validation_loss > 0.0
        assert result.final_test_loss > 0.0
        assert re.search(
            rf"INFO runs\[{re.escape(run_name)}\|airfoil_noise_cv3\]:.*train_loss=.*validate_loss=.*test_loss=.*train_mae_accuracy=.*validate_mae_accuracy=.*test_mae_accuracy=",
            plain_text,
        )

    ensemble_artifact_dir = tmp_path / "airfoil_noise_cv3_ensemble"
    artifact_path = results.save_ensemble("airfoil_noise_cv3", ensemble_artifact_dir)
    assert artifact_path == str(ensemble_artifact_dir)
    assert not (ensemble_artifact_dir / "ensemble_manifest.json").exists()
    assert not (ensemble_artifact_dir / "members").exists()
    loaded_network = thor.Network.load(str(ensemble_artifact_dir), network_name="ensemble_airfoil_noise_cv3")
    placed_ensemble = loaded_network.place(
        batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert set(placed_ensemble.get_network_input_names()) == {"examples"}


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION_LARGE,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests with downloaded Airfoil Self-Noise data",
    ),
)
def test_training_runs_airfoil_cv3_two_phase_pretrain_then_joint_multi_loss_metric_ensemble(capfd, tmp_path):
    features, target = _ensure_airfoil_self_noise_arrays()
    features = np.zeros_like(features, dtype=np.float32)
    folds, holdout_indices = _airfoil_cv3_indices(features.shape[0])
    assert len(folds) == 3

    ensemble_group = "airfoil_two_phase_cv3"
    reported_losses = ["point_mse_loss", "joint_mse_loss", "quantile_high_loss"]
    reported_metrics = ["point_mae_accuracy", "joint_mae_accuracy"]
    batch_size = 64
    pretrain_epochs = 20
    joint_epochs = 20
    baselines_by_run = {}
    probe_indices_by_run = {}
    programs_by_run = {}
    second_phases_by_run = {}

    def make_fold_trainer(
        *,
        fold: dict,
        run_name: str,
        dataset_prefix: str,
        model_prefix: str,
        enable_second_phase: bool,
        store_main_handles: bool = False,
    ):
        train_indices, probe_indices, baseline_mse = _select_airfoil_repeated_two_batch_indices(
            target,
            fold["train_indices"],
            batch_size=batch_size,
        )
        if store_main_handles:
            baselines_by_run[run_name] = baseline_mse
            probe_indices_by_run[run_name] = probe_indices
        network, program, second_phase = _build_airfoil_two_phase_joint_quantile_regressor(
            f"{model_prefix}_{run_name}",
        )
        if enable_second_phase:
            second_phase.enable()
        if store_main_handles:
            programs_by_run[run_name] = program
            second_phases_by_run[run_name] = second_phase
        loader = _airfoil_loader_from_indices(
            features,
            target,
            train_indices=train_indices,
            validate_indices=probe_indices,
            batch_size=batch_size,
            dataset_name=f"{dataset_prefix}_{run_name}",
        )
        assert loader.get_num_train_batches() == 2
        assert loader.get_num_validate_batches() == 1
        expected_phases = ["point_forecast", "joint_and_quantile"] if enable_second_phase else ["point_forecast"]
        assert program.get_step(0).get_active_phase_names() == expected_phases
        return thor.training.Trainer(
            network,
            loader,
            training_program=program,
            stats_interval_s=AIRFOIL_QUANTILE_STATS_INTERVAL_S,
            max_in_flight_batches=1,
            scalar_tensors_to_report=reported_losses,
            stats_color=AIRFOIL_QUANTILE_STATS_COLOR,
            save_model_dir=tmp_path / f"{model_prefix}_{run_name}_model",
            save_model_overwrite=True,
        )

    def make_run_specs(
        *,
        dataset_prefix: str,
        model_prefix: str,
        enable_second_phase: bool,
        store_main_handles: bool = False,
        group_name: str = ensemble_group,
        run_name_prefix: str = "fold",
    ):
        specs = []
        for fold in folds:
            fold_index = int(fold["fold_index"])
            fold_name = f"fold_{fold_index}"
            run_name = fold_name if run_name_prefix == "fold" else f"{run_name_prefix}_{fold_name}"
            specs.append(
                (
                    run_name,
                    make_fold_trainer(
                        fold=fold,
                        run_name=run_name,
                        dataset_prefix=dataset_prefix,
                        model_prefix=model_prefix,
                        enable_second_phase=enable_second_phase,
                        store_main_handles=store_main_handles,
                    ),
                    group_name,
                ))
        return specs

    def run_spec_by_name(run_specs):
        return {
            run_name: trainer for run_name, trainer, _ in run_specs
        }

    def control_run_name(prefix: str, run_name: str):
        return run_name if prefix == "fold" else f"{prefix}_{run_name}"

    def make_runs(
        run_specs,
        *,
        group_name: str = ensemble_group,
        loss_names: list[str] | None = None,
        metric_names: list[str] | None = None,
    ):
        return thor.training.TrainingRuns(
            run_specs,
            reported_losses={
                group_name: reported_losses if loss_names is None else loss_names
            },
            reported_metrics={
                group_name: reported_metrics if metric_names is None else metric_names
            },
            max_parallel_runs=3,
            max_summary_logs_per_second=AIRFOIL_QUANTILE_SUMMARY_LOGS_PER_SECOND,
        )

    def completed_or_status_payload(results):
        return [
            (result.run_name, result.status, result.result, result.exception_type, result.exception_message)
            for result in results
        ]

    # Control: phase 2 is enabled from the first update.  Because joint_mse_loss
    # consumes a scaled projection of the phase-1 forecast with the same optimum,
    # this should add same-direction gradient through the point forecast path and
    # learn faster than the disabled pretrain.  The scaling also makes the joint
    # loss/metric distinct from the point loss/metric in reports.
    enabled_start_group = "airfoil_two_phase_enabled_start_cv3"
    enabled_start_prefix = "enabled_start"
    enabled_start_specs = make_run_specs(
        dataset_prefix="airfoil_two_phase_enabled_start_cv3",
        model_prefix="airfoil_two_phase_enabled_start",
        enable_second_phase=True,
        group_name=enabled_start_group,
        run_name_prefix=enabled_start_prefix,
    )
    enabled_start_trainers = run_spec_by_name(enabled_start_specs)
    enabled_start_results, enabled_start_text = _fit_runs_and_capture_text(
        make_runs(enabled_start_specs, group_name=enabled_start_group),
        capfd,
        epochs=pretrain_epochs,
    )
    if _expects_color_for_stats_color_mode(AIRFOIL_QUANTILE_STATS_COLOR):
        assert "\x1b[" in enabled_start_text
    assert enabled_start_results.all_completed(), completed_or_status_payload(enabled_start_results)
    for trainer in enabled_start_trainers.values():
        assert trainer.completed_training_epochs == pretrain_epochs

    # Main run: pre-train with phase 2 disabled.  This is the model instance that
    # is reused below after enabling phase 2, so its epochs must advance
    # cumulatively across fits.  The enabled-start and disabled-throughout runs are
    # separate control models with their own epoch accounting.
    run_specs = make_run_specs(
        dataset_prefix="airfoil_two_phase_cv3",
        model_prefix="airfoil_two_phase",
        enable_second_phase=False,
        store_main_handles=True,
    )
    active_pretrain_losses = ["point_mse_loss"]
    active_pretrain_metrics = ["point_mae_accuracy"]
    main_trainers = run_spec_by_name(run_specs)
    first_runs = make_runs(run_specs, loss_names=active_pretrain_losses, metric_names=active_pretrain_metrics)
    first_results, first_text = _fit_runs_and_capture_text(first_runs, capfd, epochs=pretrain_epochs)
    first_plain_text = _ANSI_RE.sub("", first_text)
    if _expects_color_for_stats_color_mode(AIRFOIL_QUANTILE_STATS_COLOR):
        assert "\x1b[" in first_text

    assert first_results.all_completed(), completed_or_status_payload(first_results)
    for trainer in main_trainers.values():
        assert trainer.completed_training_epochs == pretrain_epochs
    assert "INFO runs ensemble[airfoil_two_phase_cv3]:" in first_plain_text
    assert "ensemble_train_point_mae_accuracy=" in first_plain_text
    assert "ensemble_train_joint_mae_accuracy=" not in first_plain_text
    assert "ensemble_train_joint_mse_loss=" not in first_plain_text
    assert "ensemble_train_quantile_high_loss=" not in first_plain_text
    first_ensemble = first_results.ensemble(ensemble_group)
    assert [metric.name for metric in first_ensemble.named_metrics] == active_pretrain_losses
    assert [metric.name for metric in first_ensemble.reported_metrics] == active_pretrain_metrics

    pretrain_point_loss_by_run = {}
    pretrain_point_mae_by_run = {}

    for run_name, _, _ in run_specs:
        result = first_results[run_name]
        enabled_result = enabled_start_results[control_run_name(enabled_start_prefix, run_name)]
        assert result.status == "completed"
        baseline_mse = baselines_by_run[run_name]
        point_after_disabled = result.final_training_stats.metrics["point_mse_loss"]
        point_mae_after_disabled = result.final_training_stats.metrics["point_mae_accuracy"]
        point_enabled_start = enabled_result.final_training_stats.metrics["point_mse_loss"]
        joint_enabled_start = enabled_result.final_training_stats.metrics["joint_mse_loss"]

        pretrain_point_loss_by_run[run_name] = point_after_disabled
        pretrain_point_mae_by_run[run_name] = point_mae_after_disabled

        assert point_after_disabled < baseline_mse * 0.90, (
            run_name,
            baseline_mse,
            point_after_disabled,
        )
        # Final per-run losses and reported named loss metrics are both final-epoch
        # means, so the aggregate loss should match the active named loss exactly
        # in this pretrain-only phase.
        assert result.final_training_loss is not None
        assert result.final_training_loss > 0.0
        assert result.final_training_loss == pytest.approx(point_after_disabled, rel=1e-5, abs=1e-6)
        assert result.final_validation_loss == pytest.approx(
            result.final_validation_stats.metrics["point_mse_loss"], rel=1e-5, abs=1e-6)
        assert "joint_mse_loss" not in result.final_training_stats.metrics
        assert "quantile_high_loss" not in result.final_training_stats.metrics
        assert "joint_mae_accuracy" not in result.final_training_stats.metrics
        assert "joint_mse_loss" not in result.final_validation_stats.metrics
        assert "quantile_high_loss" not in result.final_validation_stats.metrics
        assert "joint_mae_accuracy" not in result.final_validation_stats.metrics

        # If the disabled second phase leaked gradients, the disabled pretrain
        # would behave like this enabled-from-start control because joint_mse_loss
        # has the same optimum and seeds same-direction gradients through a scaled
        # projection of the same forecast tensor.  The disabled run must remain
        # measurably worse, while still learning from phase 1 alone.
        assert point_enabled_start < point_after_disabled * 0.95, (
            run_name,
            point_after_disabled,
            point_enabled_start,
        )
        assert joint_enabled_start == pytest.approx(point_enabled_start * 4.0, rel=1e-5, abs=1e-6)
        assert joint_enabled_start != pytest.approx(point_enabled_start, rel=1e-5, abs=1e-6)
        assert result.final_training_stats.metrics["point_mae_accuracy"] > 0.0
        assert result.final_validation_stats.metrics["point_mae_accuracy"] > 0.0

    # Control: phase 2 stays disabled for the same total epoch count as the main
    # pretrain+joint path.  The main run should beat this once phase 2 is enabled,
    # proving the enabled second phase does backpropagate through the shared path.
    disabled_throughout_group = "airfoil_two_phase_disabled_throughout_cv3"
    disabled_throughout_prefix = "disabled_throughout"
    disabled_throughout_specs = make_run_specs(
        dataset_prefix="airfoil_two_phase_disabled_throughout_cv3",
        model_prefix="airfoil_two_phase_disabled_throughout",
        enable_second_phase=False,
        group_name=disabled_throughout_group,
        run_name_prefix=disabled_throughout_prefix,
    )
    disabled_throughout_trainers = run_spec_by_name(disabled_throughout_specs)
    disabled_throughout_results, disabled_throughout_text = _fit_runs_and_capture_text(
        make_runs(
            disabled_throughout_specs,
            group_name=disabled_throughout_group,
            loss_names=active_pretrain_losses,
            metric_names=active_pretrain_metrics,
        ),
        capfd,
        epochs=pretrain_epochs + joint_epochs,
    )
    if _expects_color_for_stats_color_mode(AIRFOIL_QUANTILE_STATS_COLOR):
        assert "\x1b[" in disabled_throughout_text
    assert disabled_throughout_results.all_completed(), completed_or_status_payload(disabled_throughout_results)
    for trainer in disabled_throughout_trainers.values():
        assert trainer.completed_training_epochs == pretrain_epochs + joint_epochs

    for second_phase in second_phases_by_run.values():
        second_phase.enable()
    for program in programs_by_run.values():
        assert program.get_step(0).get_active_phase_names() == ["point_forecast", "joint_and_quantile"]

    # This test is about end-to-end phase mechanics and composed ensemble
    # evaluation, not distribution-shift generalization.  The model sees zeroed
    # features, so it is intentionally bias-only; evaluating on the global
    # holdout can make the test loss look catastrophically bad just because the
    # selected training/probe target slice has a different mean.  Use the same
    # deterministic fold probe slices as an in-distribution test loader so the
    # test path exercises real ensemble evaluation while still behaving like the
    # fitted phase-mechanics problem.
    ensemble_test_indices = np.ascontiguousarray(
        np.concatenate([probe_indices_by_run[run_name] for run_name, _, _ in run_specs]),
        dtype=np.int64,
    )
    test_loader = _airfoil_loader_from_indices(
        features,
        target,
        train_indices=ensemble_test_indices,
        validate_indices=ensemble_test_indices,
        batch_size=batch_size,
        dataset_name="airfoil_two_phase_cv3_in_distribution_test",
    )
    assert test_loader.get_num_validate_examples() == int(ensemble_test_indices.shape[0])
    assert test_loader.get_num_validate_batches() == len(run_specs)
    second_runs = make_runs(run_specs)
    second_results, second_text = _fit_runs_and_capture_text(
        second_runs,
        capfd,
        epochs=joint_epochs,
        test_loader=test_loader,
    )
    second_plain_text = _ANSI_RE.sub("", second_text)
    if _expects_color_for_stats_color_mode(AIRFOIL_QUANTILE_STATS_COLOR):
        assert "\x1b[" in second_text

    assert second_results.all_completed(), completed_or_status_payload(second_results)
    for trainer in main_trainers.values():
        assert trainer.completed_training_epochs == pretrain_epochs + joint_epochs
    assert "INFO runs ensemble[airfoil_two_phase_cv3]:" in second_plain_text
    for loss_name in reported_losses:
        assert f"ensemble_train_{loss_name}=" in second_plain_text
        assert f"ensemble_test_{loss_name}=" in second_plain_text
    for metric_name in reported_metrics:
        assert f"ensemble_train_{metric_name}=" in second_plain_text
        assert f"ensemble_test_{metric_name}=" in second_plain_text

    second_ensemble = second_results.ensemble(ensemble_group)
    assert second_ensemble.all_completed()
    assert len(second_ensemble.members) == 3
    assert [metric.name for metric in second_ensemble.named_metrics] == reported_losses
    assert [metric.name for metric in second_ensemble.reported_metrics] == reported_metrics
    assert second_ensemble.ensemble_train_loss is not None
    assert second_ensemble.ensemble_test_loss is not None
    assert math.isfinite(second_ensemble.ensemble_train_loss)
    assert math.isfinite(second_ensemble.ensemble_test_loss)
    assert second_ensemble.ensemble_train_loss > 0.0
    assert second_ensemble.ensemble_test_loss > 0.0

    ensemble_loss_by_name = {
        metric.name: metric for metric in second_ensemble.named_metrics
    }
    ensemble_metric_by_name = {
        metric.name: metric for metric in second_ensemble.reported_metrics
    }
    assert ensemble_loss_by_name["joint_mse_loss"].train_value == pytest.approx(
        ensemble_loss_by_name["point_mse_loss"].train_value * 4.0, rel=1e-5, abs=1e-6)
    assert ensemble_loss_by_name["joint_mse_loss"].test_value == pytest.approx(
        ensemble_loss_by_name["point_mse_loss"].test_value * 4.0, rel=1e-5, abs=1e-6)
    assert ensemble_metric_by_name["joint_mae_accuracy"].train_value == pytest.approx(
        ensemble_metric_by_name["point_mae_accuracy"].train_value * 2.0, rel=1e-4, abs=1e-6)
    assert ensemble_metric_by_name["joint_mae_accuracy"].test_value == pytest.approx(
        ensemble_metric_by_name["point_mae_accuracy"].test_value * 2.0, rel=1e-4, abs=1e-6)
    assert ensemble_loss_by_name["point_mse_loss"].test_value < 0.25
    assert ensemble_metric_by_name["point_mae_accuracy"].test_value < 0.75
    assert second_ensemble.ensemble_test_loss < second_ensemble.ensemble_train_loss * 2.0

    for run_name, _, _ in run_specs:
        result = second_results[run_name]
        disabled_result = disabled_throughout_results[control_run_name(disabled_throughout_prefix, run_name)]
        assert result.status == "completed"
        assert result.final_test_loss is not None

        point_after_joint = result.final_training_stats.metrics["point_mse_loss"]
        joint_after_joint = result.final_training_stats.metrics["joint_mse_loss"]
        quantile_after_joint = result.final_training_stats.metrics["quantile_high_loss"]
        point_mae_after_joint = result.final_training_stats.metrics["point_mae_accuracy"]
        joint_mae_after_joint = result.final_training_stats.metrics["joint_mae_accuracy"]
        point_disabled_throughout = disabled_result.final_training_stats.metrics["point_mse_loss"]
        assert "joint_mse_loss" not in disabled_result.final_training_stats.metrics
        assert "quantile_high_loss" not in disabled_result.final_training_stats.metrics
        assert "joint_mae_accuracy" not in disabled_result.final_training_stats.metrics

        assert point_after_joint < pretrain_point_loss_by_run[run_name] * 0.95, (
            run_name,
            pretrain_point_loss_by_run[run_name],
            point_after_joint,
        )
        assert joint_after_joint < pretrain_point_loss_by_run[run_name] * 4.0 * 0.95, (
            run_name,
            pretrain_point_loss_by_run[run_name],
            joint_after_joint,
        )
        assert quantile_after_joint > 0.0
        assert point_mae_after_joint < pretrain_point_mae_by_run[run_name] * 0.95, (
            run_name,
            pretrain_point_mae_by_run[run_name],
            point_mae_after_joint,
        )
        assert joint_mae_after_joint < pretrain_point_mae_by_run[run_name] * 2.0 * 0.95, (
            run_name,
            pretrain_point_mae_by_run[run_name],
            joint_mae_after_joint,
        )
        assert joint_after_joint == pytest.approx(point_after_joint * 4.0, rel=1e-5, abs=1e-6)
        assert joint_after_joint != pytest.approx(point_after_joint, rel=1e-5, abs=1e-6)
        assert joint_mae_after_joint == pytest.approx(point_mae_after_joint * 2.0, rel=1e-5, abs=1e-6)
        assert joint_mae_after_joint != pytest.approx(point_mae_after_joint, rel=1e-5, abs=1e-6)
        assert point_after_joint < point_disabled_throughout, (
            run_name,
            point_disabled_throughout,
            point_after_joint,
        )
        assert joint_after_joint * 0.25 < point_disabled_throughout, (
            run_name,
            point_disabled_throughout,
            joint_after_joint,
        )

        for loss_name in reported_losses:
            assert result.final_training_stats.metrics[loss_name] > 0.0
            assert result.final_validation_stats.metrics[loss_name] > 0.0
            assert result.final_test_stats.metrics[loss_name] > 0.0
        for metric_name in reported_metrics:
            assert result.final_training_stats.metrics[metric_name] > 0.0
            assert result.final_validation_stats.metrics[metric_name] > 0.0
            assert result.final_test_stats.metrics[metric_name] > 0.0
        final_line_match = re.search(
            rf"^INFO runs\[{re.escape(run_name)}\|airfoil_two_phase_cv3\]:.*status=completed.*train_loss=.*validate_loss=.*test_loss=.*$",
            second_plain_text,
            flags=re.MULTILINE,
        )
        assert final_line_match is not None
        final_line = final_line_match.group(0)
        for column_name in ("train_loss", "validate_loss", "test_loss"):
            assert f"{column_name}=" in final_line
        for prefix in ("train", "validate", "test"):
            for loss_name in reported_losses:
                assert f"{prefix}_{loss_name}=" in final_line
            for metric_name in reported_metrics:
                assert f"{prefix}_{metric_name}=" in final_line


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION_LARGE,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns Airfoil two-phase MAE/MSE holdout integration test",
    ),
)
def test_training_runs_airfoil_cv3_two_phase_mae_pretrain_then_mse_head_holdout(capfd, tmp_path):
    features, target = _ensure_airfoil_self_noise_arrays()
    assert features.shape[0] == target.shape[0]
    assert features.shape[1] == 5
    assert target.shape[1] == 1

    folds, holdout_indices = _airfoil_cv3_indices(features.shape[0])
    assert len(folds) == 3
    assert holdout_indices.shape[0] == int(round(features.shape[0] * 0.10))

    ensemble_group = "airfoil_mae_then_mse_cv3"
    reported_losses = ["mae_loss", "mse_loss"]
    reported_metrics = ["mae_head_mae_accuracy", "mse_head_mae_accuracy"]
    batch_size = 128
    pretrain_epochs = 30
    mse_epochs = 40
    programs_by_run = {}
    mse_phases_by_run = {}
    validation_zero_mae_by_run = {}
    validation_zero_mse_by_run = {}

    holdout_target = target[holdout_indices]
    holdout_zero_mae = float(np.mean(np.abs(holdout_target), dtype=np.float32))
    holdout_zero_mse = float(np.mean(np.square(holdout_target, dtype=np.float32), dtype=np.float32))
    assert holdout_zero_mae > 0.0
    assert holdout_zero_mse > 0.0

    def make_fold_trainer(*, fold: dict, run_name: str):
        loader = _airfoil_loader_from_indices(
            features,
            target,
            train_indices=fold["train_indices"],
            validate_indices=fold["validate_indices"],
            batch_size=batch_size,
            dataset_name=f"airfoil_mae_then_mse_cv3_{run_name}",
        )
        assert loader.get_num_train_batches() > 1
        assert loader.get_num_validate_batches() > 1
        validation_target = target[fold["validate_indices"]]
        validation_zero_mae_by_run[run_name] = float(np.mean(np.abs(validation_target), dtype=np.float32))
        validation_zero_mse_by_run[run_name] = float(
            np.mean(np.square(validation_target, dtype=np.float32), dtype=np.float32))
        network, program, mse_phase = _build_airfoil_two_phase_mae_then_mse_regressor(
            f"airfoil_mae_then_mse_{run_name}",
            width=32,
        )
        programs_by_run[run_name] = program
        mse_phases_by_run[run_name] = mse_phase
        assert program.get_step(0).get_active_phase_names() == ["mae_pretrain"]
        return thor.training.Trainer(
            network,
            loader,
            training_program=program,
            stats_interval_s=AIRFOIL_QUANTILE_STATS_INTERVAL_S,
            max_in_flight_batches=2,
            scalar_tensors_to_report=reported_losses,
            stats_color=AIRFOIL_QUANTILE_STATS_COLOR,
            save_model_dir=tmp_path / f"{run_name}_model",
            save_model_overwrite=True,
        )

    run_specs = []
    for fold in folds:
        fold_index = int(fold["fold_index"])
        run_name = f"fold_{fold_index}"
        run_specs.append((run_name, make_fold_trainer(fold=fold, run_name=run_name), ensemble_group))

    main_trainers = {
        run_name: trainer for run_name, trainer, _ in run_specs
    }

    def make_runs():
        return thor.training.TrainingRuns(
            run_specs,
            reported_losses={
                ensemble_group: reported_losses
            },
            reported_metrics={
                ensemble_group: reported_metrics
            },
            max_parallel_runs=3,
            max_summary_logs_per_second=AIRFOIL_QUANTILE_SUMMARY_LOGS_PER_SECOND,
        )

    def completed_or_status_payload(results):
        return [
            (result.run_name, result.status, result.result, result.exception_type, result.exception_message)
            for result in results
        ]

    pretrain_results, pretrain_text = _fit_runs_and_capture_text(
        make_runs(),
        capfd,
        epochs=pretrain_epochs,
    )
    pretrain_plain_text = _ANSI_RE.sub("", pretrain_text)
    if _expects_color_for_stats_color_mode(AIRFOIL_QUANTILE_STATS_COLOR):
        assert "\x1b[" in pretrain_text
    assert pretrain_results.all_completed(), completed_or_status_payload(pretrain_results)
    assert "mse_loss=" not in pretrain_plain_text
    assert "mse_head_mae_accuracy=" not in pretrain_plain_text
    for run_name, trainer in main_trainers.items():
        assert trainer.completed_training_epochs == pretrain_epochs
        result = pretrain_results[run_name]
        assert result.final_validation_stats.metrics["mae_loss"] < validation_zero_mae_by_run[run_name] * 0.98, (
            run_name,
            validation_zero_mae_by_run[run_name],
            result.final_validation_stats.metrics["mae_loss"],
        )
        assert "mse_loss" not in result.final_training_stats.metrics
        assert "mse_loss" not in result.final_validation_stats.metrics
        assert "mse_head_mae_accuracy" not in result.final_training_stats.metrics
        assert "mse_head_mae_accuracy" not in result.final_validation_stats.metrics

    for mse_phase in mse_phases_by_run.values():
        mse_phase.enable()
    for program in programs_by_run.values():
        assert program.get_step(0).get_active_phase_names() == ["mae_pretrain", "mse_finetune"]

    test_loader = _airfoil_loader_from_indices(
        features,
        target,
        train_indices=holdout_indices,
        validate_indices=holdout_indices,
        batch_size=batch_size,
        dataset_name="airfoil_mae_then_mse_cv3_holdout_test",
    )
    assert test_loader.get_num_validate_examples() == int(holdout_indices.shape[0])
    assert test_loader.get_num_validate_batches() > 1

    final_results, final_text = _fit_runs_and_capture_text(
        make_runs(),
        capfd,
        epochs=mse_epochs,
        test_loader=test_loader,
    )
    final_plain_text = _ANSI_RE.sub("", final_text)
    if _expects_color_for_stats_color_mode(AIRFOIL_QUANTILE_STATS_COLOR):
        assert "\x1b[" in final_text
    assert final_results.all_completed(), completed_or_status_payload(final_results)
    assert "INFO runs ensemble[airfoil_mae_then_mse_cv3]:" in final_plain_text
    for loss_name in reported_losses:
        assert f"ensemble_train_{loss_name}=" in final_plain_text
        assert f"ensemble_test_{loss_name}=" in final_plain_text
    for metric_name in reported_metrics:
        assert f"ensemble_train_{metric_name}=" in final_plain_text
        assert f"ensemble_test_{metric_name}=" in final_plain_text

    for trainer in main_trainers.values():
        assert trainer.completed_training_epochs == pretrain_epochs + mse_epochs

    ensemble = final_results.ensemble(ensemble_group)
    assert ensemble.all_completed()
    assert len(ensemble.members) == 3
    assert [metric.name for metric in ensemble.named_metrics] == reported_losses
    assert [metric.name for metric in ensemble.reported_metrics] == reported_metrics
    assert ensemble.ensemble_train_loss is not None
    assert ensemble.ensemble_test_loss is not None
    assert math.isfinite(ensemble.ensemble_train_loss)
    assert math.isfinite(ensemble.ensemble_test_loss)
    assert ensemble.ensemble_train_loss > 0.0
    assert ensemble.ensemble_test_loss > 0.0

    ensemble_loss_by_name = {
        metric.name: metric for metric in ensemble.named_metrics
    }
    ensemble_metric_by_name = {
        metric.name: metric for metric in ensemble.reported_metrics
    }
    assert ensemble_loss_by_name["mae_loss"].test_value < holdout_zero_mae * 0.98, (
        holdout_zero_mae,
        ensemble_loss_by_name["mae_loss"].test_value,
    )
    assert ensemble_loss_by_name["mse_loss"].test_value < holdout_zero_mse * 0.98, (
        holdout_zero_mse,
        ensemble_loss_by_name["mse_loss"].test_value,
    )
    assert ensemble_metric_by_name["mae_head_mae_accuracy"].test_value == pytest.approx(
        ensemble_loss_by_name["mae_loss"].test_value, rel=1e-4, abs=1e-6)
    assert ensemble_metric_by_name["mse_head_mae_accuracy"].test_value < holdout_zero_mae * 1.05, (
        holdout_zero_mae,
        ensemble_metric_by_name["mse_head_mae_accuracy"].test_value,
    )

    for run_name, _, _ in run_specs:
        result = final_results[run_name]
        assert result.status == "completed"
        assert result.final_test_loss is not None
        for loss_name in reported_losses:
            assert result.final_training_stats.metrics[loss_name] > 0.0
            assert result.final_validation_stats.metrics[loss_name] > 0.0
            assert result.final_test_stats.metrics[loss_name] > 0.0
        for metric_name in reported_metrics:
            assert result.final_training_stats.metrics[metric_name] > 0.0
            assert result.final_validation_stats.metrics[metric_name] > 0.0
            assert result.final_test_stats.metrics[metric_name] > 0.0
        assert result.final_test_stats.metrics["mae_loss"] < holdout_zero_mae * 1.10, (
            run_name,
            holdout_zero_mae,
            result.final_test_stats.metrics["mae_loss"],
        )
        assert result.final_test_stats.metrics["mse_loss"] < holdout_zero_mse * 1.10, (
            run_name,
            holdout_zero_mse,
            result.final_test_stats.metrics["mse_loss"],
        )
        assert result.final_test_stats.metrics["mae_head_mae_accuracy"] == pytest.approx(
            result.final_test_stats.metrics["mae_loss"], rel=1e-4, abs=1e-6)
        assert result.final_test_stats.metrics["mse_head_mae_accuracy"] < holdout_zero_mae * 1.15, (
            run_name,
            holdout_zero_mae,
            result.final_test_stats.metrics["mse_head_mae_accuracy"],
        )
        assert result.final_training_loss == pytest.approx(
            result.final_training_stats.metrics["mae_loss"] + result.final_training_stats.metrics["mse_loss"],
            rel=1e-5,
            abs=1e-6,
        )
        assert result.final_validation_loss == pytest.approx(
            result.final_validation_stats.metrics["mae_loss"] + result.final_validation_stats.metrics["mse_loss"],
            rel=1e-5,
            abs=1e-6,
        )
        assert result.final_test_loss == pytest.approx(
            result.final_test_stats.metrics["mae_loss"] + result.final_test_stats.metrics["mse_loss"],
            rel=1e-5,
            abs=1e-6,
        )
        final_line_match = re.search(
            rf"^INFO runs\[{re.escape(run_name)}\|airfoil_mae_then_mse_cv3\]:.*status=completed.*train_loss=.*validate_loss=.*test_loss=.*$",
            final_plain_text,
            flags=re.MULTILINE,
        )
        assert final_line_match is not None
        final_line = final_line_match.group(0)
        for column_name in ("train_loss", "validate_loss", "test_loss"):
            assert f"{column_name}=" in final_line
        for prefix in ("train", "validate", "test"):
            for loss_name in reported_losses:
                assert f"{prefix}_{loss_name}=" in final_line
            for metric_name in reported_metrics:
                assert f"{prefix}_{metric_name}=" in final_line


@pytest.mark.cuda
@pytest.mark.training_integration
@pytest.mark.skipif(
    not RUN_TRAINING_INTEGRATION,
    reason=integration_skip_reason(
        "THOR_RUN_TRAINING_INTEGRATION",
        description="opt-in TrainingRuns CUDA integration tests",
    ),
)
def test_training_runs_graph_loss_does_not_invent_prediction_loss(capfd, tmp_path):
    trainer = _make_named_graph_loss_regression_trainer(
        "training_runs_graph_loss_no_synthetic_prediction_loss",
        save_model_dir=tmp_path / "model",
        save_model_overwrite=True,
    )
    runs = thor.training.TrainingRuns([("fold_0", trainer, "graph_loss_ensemble")])

    results, captured_text = _fit_runs_and_capture_text(
        runs,
        capfd,
        epochs=1,
        test_loader=_regression_one_batch_loader(),
    )

    assert results.all_completed()
    ensemble = results.ensemble("graph_loss_ensemble")
    assert len(ensemble.named_metrics) == 1
    assert ensemble.named_metrics[0].name == "graph_loss"
    assert ensemble.named_metrics[0].test_value == pytest.approx(2.0, rel=1e-5, abs=1e-6)
    assert ensemble.ensemble_test_loss == pytest.approx(2.0, rel=1e-5, abs=1e-6)

    # This must be the graph-owned loss value, not a synthetic MAE/CE value derived
    # by looking at a prediction output and guessing a label tensor on the CPU.
    ensemble_line = next(
        line for line in _ANSI_RE.sub("", captured_text).splitlines()
        if "INFO runs ensemble[graph_loss_ensemble]:" in line)
    assert "ensemble_test_graph_loss=" in ensemble_line


def _training_selection_metadata(save_dir):
    with open(save_dir / "training_selection_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _training_artifact_latest_dir(save_dir):
    return save_dir / "latest"


def _training_artifact_best_dir(save_dir):
    return save_dir / "best"


def _training_artifact_selected_dir(save_dir):
    best_dir = _training_artifact_best_dir(save_dir)
    if best_dir.exists():
        return best_dir
    latest_dir = _training_artifact_latest_dir(save_dir)
    if latest_dir.exists():
        return latest_dir
    return save_dir


def _prediction_from_saved_tiny_regressor(save_dir, network_name: str, *, artifact="selected"):
    if artifact == "selected":
        model_dir = _training_artifact_selected_dir(save_dir)
    elif artifact == "latest":
        model_dir = _training_artifact_latest_dir(save_dir)
    elif artifact == "best":
        model_dir = _training_artifact_best_dir(save_dir)
    elif artifact == "direct":
        model_dir = save_dir
    else:
        raise ValueError(f"unsupported artifact selector: {artifact}")
    loaded = thor.Network(network_name)
    loaded.load(str(model_dir))
    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    x, _ = _regression_arrays(dtype=np.float32)
    outputs = placed.infer({
        "examples": _cpu_tensor(x, thor.DataType.fp32)
    })
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
    assert _training_artifact_latest_dir(save_dir).exists()
    assert _training_artifact_best_dir(save_dir).exists()

    loaded = thor.Network("saved_trained_inference_sequence")
    loaded.load(str(_training_artifact_latest_dir(save_dir)))
    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    assert isinstance(placed, thor.runtime.PlacedNetwork)
    assert set(placed.get_network_input_names()) == {"examples"}

    x, _ = _regression_arrays(dtype=np.float32)
    outputs = placed.infer({
        "examples": _cpu_tensor(x, thor.DataType.fp32)
    })

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
    loaded.load(str(_training_artifact_latest_dir(save_dir)))
    placed = loaded.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    assert set(placed.get_network_input_names()) == {"examples"}

    x, _ = _regression_arrays(dtype=np.float32)
    outputs = placed.infer({
        "examples": _cpu_tensor(x, thor.DataType.fp32)
    })

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

    assert save_dir.exists()
    assert _training_artifact_latest_dir(save_dir).exists()
    assert _training_artifact_best_dir(save_dir).exists()
    prediction = _prediction_from_saved_tiny_regressor(
        save_dir, "trainer_best_candidate_snapshot_trained_weights", artifact="best")
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
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: float(epoch),
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
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: -float(epoch),
    ).fit(2)

    selected_prediction = _prediction_from_saved_tiny_regressor(
        first_epoch_dir, "trainer_custom_model_selection_first_epoch")
    one_epoch_prediction = _prediction_from_saved_tiny_regressor(
        one_epoch_reference_dir, "trainer_custom_model_selection_one_epoch_reference")
    two_epoch_prediction = _prediction_from_saved_tiny_regressor(
        two_epoch_reference_dir, "trainer_custom_model_selection_two_epoch_reference")

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
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch >= 2)
    trainer = _make_tiny_regression_trainer(
        "trainer_fit_result_metadata",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: float(epoch),
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
    assert _training_artifact_latest_dir(save_dir).exists()
    assert _training_artifact_best_dir(save_dir).exists()

    metadata = _training_selection_metadata(save_dir)
    assert metadata["schema_version"] == 2
    assert metadata["latest_epoch"] == 2
    assert metadata["latest_score"] == pytest.approx(2.0)
    assert metadata["latest_training_loss"] is not None
    assert metadata["latest_validation_loss"] is not None
    assert metadata["has_best_candidate"] is True
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
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: None if epoch == 1 else float(epoch),
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
    assert metadata["schema_version"] == 2
    assert metadata["latest_epoch"] == 2
    assert metadata["latest_score"] == pytest.approx(2.0)
    assert metadata["has_best_candidate"] is True
    assert metadata["best_epoch"] == 2
    assert metadata["best_score"] == pytest.approx(2.0)
    assert metadata["completed_epoch"] == 2
    assert metadata["completion_reason"] == "completed"

    selected_prediction = _prediction_from_saved_tiny_regressor(
        save_dir, "trainer_custom_model_selection_none_skips_epoch")
    one_epoch_prediction = _prediction_from_saved_tiny_regressor(
        one_epoch_reference_dir, "trainer_custom_model_selection_none_one_epoch_reference")
    two_epoch_prediction = _prediction_from_saved_tiny_regressor(
        two_epoch_reference_dir, "trainer_custom_model_selection_none_two_epoch_reference")

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
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: float(epoch),
    )

    result = trainer.fit(2)

    assert result.status == "completed"
    assert result.result == "completed"
    assert result.completed_epoch == 2
    assert result.best_epoch is None
    assert result.best_score is None
    assert trainer.completed_training_epochs == 2
    assert save_dir.exists()
    assert _training_artifact_latest_dir(save_dir).exists()
    assert not _training_artifact_best_dir(save_dir).exists()

    metadata = _training_selection_metadata(save_dir)
    assert metadata["schema_version"] == 2
    assert metadata["latest_epoch"] == 2
    assert metadata["latest_score"] is None
    assert metadata["latest_training_loss"] is not None
    assert metadata["latest_validation_loss"] is not None
    assert metadata["has_best_candidate"] is False
    assert metadata["best_epoch"] is None
    assert metadata["best_score"] is None
    assert metadata["completed_epoch"] == 2
    assert metadata["completion_reason"] == "completed"

    final_dir = tmp_path / "manual_final_model_after_no_candidate"
    trainer.save_model(final_dir)
    assert final_dir.exists()
    prediction = _prediction_from_saved_tiny_regressor(
        final_dir, "trainer_min_early_completion_no_candidate", artifact="direct")
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
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch >= 3)
    trainer = _make_tiny_regression_trainer(
        "trainer_min_early_completion_cumulative",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        min_early_completion_epochs=3,
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: float(epoch),
        early_completion_policies=[early_policy],
    )

    first_result = trainer.fit(2)
    assert first_result.completed_epoch == 2
    assert first_result.best_epoch is None
    assert trainer.completed_training_epochs == 2
    assert save_dir.exists()
    assert _training_artifact_latest_dir(save_dir).exists()
    assert not _training_artifact_best_dir(save_dir).exists()

    first_metadata = _training_selection_metadata(save_dir)
    assert first_metadata["latest_epoch"] == 2
    assert first_metadata["has_best_candidate"] is False
    assert first_metadata["best_epoch"] is None

    second_result = trainer.fit(10)

    assert second_result.status == "completed"
    assert second_result.result == "early_completed"
    assert second_result.early_completed is True
    assert second_result.completed_epoch == 3
    assert second_result.best_epoch == 3
    assert second_result.best_score == pytest.approx(3.0)
    assert trainer.completed_training_epochs == 3
    assert save_dir.exists()
    assert _training_artifact_latest_dir(save_dir).exists()
    assert _training_artifact_best_dir(save_dir).exists()

    metadata = _training_selection_metadata(save_dir)
    assert metadata["schema_version"] == 2
    assert metadata["latest_epoch"] == 3
    assert metadata["latest_score"] == pytest.approx(3.0)
    assert metadata["has_best_candidate"] is True
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
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch >= 3)
    trainer = _make_tiny_regression_trainer(
        "training_runs_min_early_completion_cumulative",
        save_model_dir=save_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        min_early_completion_epochs=3,
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: float(epoch),
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
    assert save_dir.exists()
    assert _training_artifact_latest_dir(save_dir).exists()
    assert not _training_artifact_best_dir(save_dir).exists()

    first_metadata = _training_selection_metadata(save_dir)
    assert first_metadata["latest_epoch"] == 2
    assert first_metadata["has_best_candidate"] is False
    assert first_metadata["best_epoch"] is None

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
    assert _training_artifact_latest_dir(save_dir).exists()
    assert _training_artifact_best_dir(save_dir).exists()

    metadata = _training_selection_metadata(save_dir)
    assert metadata["schema_version"] == 2
    assert metadata["latest_epoch"] == 3
    assert metadata["latest_score"] == pytest.approx(3.0)
    assert metadata["has_best_candidate"] is True
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
def test_training_runs_early_completion_stops_early_and_saves_best_candidate(capfd, tmp_path):
    early_dir = tmp_path / "early_completed_best"
    one_epoch_reference_dir = tmp_path / "one_epoch_reference"
    two_epoch_reference_dir = tmp_path / "two_epoch_reference"

    early_policy = thor.training.EarlyCompletionPolicy(
        lambda current_score, best_score, current_epoch, best_epoch: current_epoch >= 2)
    trainer = _make_tiny_regression_trainer(
        "training_runs_early_completion_best_candidate",
        save_model_dir=early_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
        model_selection_score=lambda validation_loss,
        training_loss,
        epoch: float(epoch),
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
    assert _training_artifact_latest_dir(early_dir).exists()
    assert _training_artifact_best_dir(early_dir).exists()

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

    _make_tiny_regression_trainer(
        "training_runs_early_completion_two_epoch_reference",
        save_model_dir=two_epoch_reference_dir,
        save_model_overwrite=True,
        check_best_model_every_epochs=1,
    ).fit(2)

    selected_prediction = _prediction_from_saved_tiny_regressor(
        early_dir, "training_runs_early_completion_best_candidate", artifact="best")
    latest_prediction = _prediction_from_saved_tiny_regressor(
        early_dir, "training_runs_early_completion_best_candidate", artifact="latest")
    one_epoch_prediction = _prediction_from_saved_tiny_regressor(
        one_epoch_reference_dir, "training_runs_early_completion_one_epoch_reference")
    two_epoch_prediction = _prediction_from_saved_tiny_regressor(
        two_epoch_reference_dir, "training_runs_early_completion_two_epoch_reference")

    assert np.allclose(selected_prediction, one_epoch_prediction, atol=1e-6)
    assert np.allclose(latest_prediction, two_epoch_prediction, atol=1e-6)
    assert not np.allclose(selected_prediction, latest_prediction, atol=1e-7)


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
    artifact_path = results.save_ensemble("tiny_ensemble", ensemble_artifact_dir)
    assert artifact_path == str(ensemble_artifact_dir)
    assert not (ensemble_artifact_dir / "ensemble_manifest.json").exists()
    assert not (ensemble_artifact_dir / "members").exists()

    loaded_network = thor.Network.load(str(ensemble_artifact_dir), network_name="ensemble_tiny_ensemble")
    placed_ensemble = loaded_network.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert set(placed_ensemble.get_network_input_names()) == {"examples"}
    x, _ = _regression_arrays(dtype=np.float32)
    ensemble_outputs = placed_ensemble.infer({
        "examples": _cpu_tensor(x, thor.DataType.fp32)
    })
    assert set(ensemble_outputs) == {"prediction"}
    ensemble_prediction = np.array(ensemble_outputs["prediction"].numpy(), copy=True)

    fold_0_prediction = _prediction_from_saved_tiny_regressor(
        tmp_path / "fold_0_model", results["fold_0"].saved_model_network_name)
    fold_1_prediction = _prediction_from_saved_tiny_regressor(
        tmp_path / "longer_fold_1_model", results["longer_fold_1"].saved_model_network_name)
    expected_prediction = (fold_0_prediction + 2.0 * fold_1_prediction) / 3.0
    assert ensemble_prediction.shape == expected_prediction.shape
    np.testing.assert_allclose(ensemble_prediction, expected_prediction, rtol=1e-5, atol=1e-5)

    with pytest.raises(RuntimeError, match="already exists"):
        results.save_ensemble("tiny_ensemble", ensemble_artifact_dir)
    assert results.save_ensemble("tiny_ensemble", ensemble_artifact_dir, overwrite=True) == str(ensemble_artifact_dir)

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
def test_training_runs_save_ensemble_excludes_label_only_report_inputs(capfd, tmp_path):
    runs = thor.training.TrainingRuns(
        [
            (
                "fold_0",
                _make_tiny_regression_with_label_mean_report_trainer(
                    "training_runs_label_report_save_ensemble_fold_0",
                    save_model_dir=tmp_path / "fold_0_model",
                    save_model_overwrite=True,
                ),
                "tiny_ensemble",
            ),
            (
                "fold_1",
                _make_tiny_regression_with_label_mean_report_trainer(
                    "training_runs_label_report_save_ensemble_fold_1",
                    save_model_dir=tmp_path / "fold_1_model",
                    save_model_overwrite=True,
                ),
                "tiny_ensemble",
            ),
        ],
        reported_metrics={
            "tiny_ensemble": ["true_mean", "prediction_mean"],
        },
    )

    results, _ = _fit_runs_and_capture_text(runs, capfd, epochs=1)
    assert results.all_completed()
    graph_metric_by_name = {
        metric.name: metric for metric in results.ensemble("tiny_ensemble").reported_metrics
    }
    assert graph_metric_by_name["true_mean"].train_value is not None
    assert graph_metric_by_name["prediction_mean"].train_value is not None

    ensemble_artifact_dir = tmp_path / "tiny_ensemble_label_report_artifact"
    assert results.save_ensemble("tiny_ensemble", ensemble_artifact_dir) == str(ensemble_artifact_dir)
    loaded_network = thor.Network.load(str(ensemble_artifact_dir), network_name="ensemble_tiny_ensemble")
    placed_ensemble = loaded_network.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert set(placed_ensemble.get_network_input_names()) == {"examples"}
    x, _ = _regression_arrays(dtype=np.float32)
    ensemble_outputs = placed_ensemble.infer({
        "examples": _cpu_tensor(x, thor.DataType.fp32)
    })
    assert set(ensemble_outputs) == {"prediction"}


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
        test_loader=_categorical_mixed_labels_one_batch_loader(),
    )

    assert results.all_completed()
    ensemble = results.ensemble("tiny_categorical_ensemble")
    assert ensemble.ensemble_test_loss is not None

    ensemble_artifact_dir = tmp_path / "tiny_categorical_ensemble_artifact"
    assert results.save_ensemble("tiny_categorical_ensemble", ensemble_artifact_dir) == str(ensemble_artifact_dir)
    loaded_network = thor.Network.load(str(ensemble_artifact_dir), network_name="ensemble_tiny_categorical_ensemble")
    placed_ensemble = loaded_network.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert set(placed_ensemble.get_network_input_names()) == {"examples"}


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
        inputs={
            "forecast": forecast.get_feature_output()
        },
        output_names=["forecast_quantile_high"],
        build=lambda context: {
            "forecast_quantile_high": context.input("forecast") + 1.0
        },
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
    ).holdout_plus_k_fold(
        test_size=2, k=2)

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
        min_successful_models={
            "brand_demand_cv2": 2
        },
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
    assert results.save_ensemble("brand_demand_cv2", ensemble_artifact_dir) == str(ensemble_artifact_dir)
    loaded_network = thor.Network.load(str(ensemble_artifact_dir), network_name="ensemble_brand_demand_cv2")
    placed_ensemble = loaded_network.place(4, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert set(placed_ensemble.get_network_input_names()) == {
        "trend_inputs", "seasonality_inputs", "monotone_increasing_inputs"
    }
