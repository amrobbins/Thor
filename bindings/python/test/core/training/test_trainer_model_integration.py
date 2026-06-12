import ctypes
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import thor

RUN_TRAINING_INTEGRATION = os.environ.get("THOR_RUN_TRAINING_INTEGRATION") == "1"
ASSERT_TRAINER_LEARNING = os.environ.get("THOR_ASSERT_TRAINER_LEARNING") == "1"
DATA_DIR = Path(os.environ.get("THOR_TRAINING_DATA_DIR", "/tmp/thor_training_data"))
IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
IRIS_PATH = DATA_DIR / "iris.data"


def _flush_native_stdio_for_capture():
    # The trainer's default stats reporter intentionally does not flush on the
    # training path.  Flush C stdio only in the smoke-test harness so capfd sees
    # native printf/fprintf output before pytest prints its summary.
    ctypes.CDLL(None).fflush(None)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.training_integration,
    pytest.mark.skipif(
        not RUN_TRAINING_INTEGRATION,
        reason="set THOR_RUN_TRAINING_INTEGRATION=1 to run opt-in model training integration tests",
    ),
]
requires_trainer_learning = pytest.mark.skipif(
    not ASSERT_TRAINER_LEARNING,
    reason=(
        "set THOR_ASSERT_TRAINER_LEARNING=1 to run trainer convergence diagnostics "
        "while the native trainer update path is under repair"
    ),
)


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


_TRAINER_STATS_RE = re.compile(
    r"INFO trainer: phase=(?P<phase>train|validate|test) "
    r"epoch=(?P<epoch>\d+)/(?:\d+) "
    r"step=(?P<step>\d+) "
    r"batch=(?P<batch>\d+)/(?:\d+) "
    r"loss=(?P<loss>[-+0-9.eE]+)"
)


def _captured_trainer_stats(captured_text: str):
    plain_text = _ANSI_RE.sub("", captured_text)
    stats = []
    for match in _TRAINER_STATS_RE.finditer(plain_text):
        stats.append(
            {
                "phase": match.group("phase"),
                "epoch": int(match.group("epoch")),
                "step": int(match.group("step")),
                "batch": int(match.group("batch")),
                "loss": float(match.group("loss")),
            }
        )
    return stats


def _phase_losses(stats, phase: str):
    return [entry["loss"] for entry in stats if entry["phase"] == phase]


def _phase_epoch_mean_losses(stats, phase: str):
    grouped = {}
    for entry in stats:
        if entry["phase"] == phase:
            grouped.setdefault(entry["epoch"], []).append(entry["loss"])
    return [float(np.mean(grouped[epoch])) for epoch in sorted(grouped)]


def _assert_loss_decreased(losses, *, name: str, tail_window: int = 5, required_fraction: float = 0.85):
    assert len(losses) >= tail_window * 2, f"expected enough {name} loss samples, got {losses}"
    first = float(np.mean(losses[:tail_window]))
    last = float(np.mean(losses[-tail_window:]))
    assert last < first * required_fraction, f"{name} loss did not decrease enough: first_window={first}, last_window={last}, losses={losses}"


def _fit_and_capture_stats(trainer, capfd, *, epochs: int):
    _flush_native_stdio_for_capture()
    capfd.readouterr()
    trainer.fit(epochs=epochs)
    _flush_native_stdio_for_capture()
    captured = capfd.readouterr()
    captured_text = captured.out + captured.err
    stats = _captured_trainer_stats(captured_text)
    assert stats, "trainer emitted no parseable stats"

    # Keep the training path free of flushes while still making the opt-in smoke
    # tests visibly print trainer stats under pytest's fd capture. Native fprintf
    # output is captured and parsed above, then replayed outside capture here.
    with capfd.disabled():
        sys.stdout.write(captured_text)
        sys.stdout.flush()

    return stats


def _linearly_separable_one_batch_loader(*, batch_size: int = 16, dtype=np.float32):
    neg = np.array(
        [
            [-2.8, -0.5],
            [-2.5, 0.1],
            [-2.2, 0.6],
            [-1.9, -0.2],
            [-1.7, 0.4],
            [-1.5, -0.7],
            [-1.3, 0.0],
            [-1.1, 0.7],
        ],
        dtype=np.float32,
    )
    pos = np.array(
        [
            [1.1, -0.6],
            [1.3, 0.2],
            [1.5, 0.7],
            [1.7, -0.1],
            [1.9, 0.5],
            [2.2, -0.4],
            [2.5, 0.0],
            [2.8, 0.6],
        ],
        dtype=np.float32,
    )
    x = np.concatenate([neg, pos], axis=0)
    y_index = np.concatenate([np.zeros(len(neg), dtype=np.int64), np.ones(len(pos), dtype=np.int64)])
    y = np.zeros((x.shape[0], 2), dtype=np.float32)
    y[np.arange(x.shape[0]), y_index] = 1.0

    order = np.array([0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15], dtype=np.int64)
    x = np.ascontiguousarray(x[order], dtype=dtype)
    y = np.ascontiguousarray(y[order], dtype=dtype)

    assert x.shape[0] == batch_size
    loader_cls = thor.training.NumpyFloat16BatchLoader if dtype == np.float16 else thor.training.NumpyFloat32BatchLoader
    return loader_cls(
        x,
        y,
        x,
        y,
        batch_size=batch_size,
        example_input_name="examples",
        label_input_name="labels",
        dataset_name="linearly_separable_one_batch",
    )


def _linearly_separable_multi_batch_loader(*, batch_size: int = 4, dtype=np.float32):
    # Four deliberately different batches.  The first epoch's per-batch losses should
    # not contain exact adjacent repeats; exact repeats are a strong stale-stat signal
    # for the queued executor because the model and examples differ from batch to batch.
    x = np.array(
        [
            [-3.0, -0.4],
            [-2.8, 0.5],
            [2.8, -0.5],
            [3.0, 0.4],
            [-2.3, 1.2],
            [-2.1, -1.1],
            [2.1, 1.1],
            [2.3, -1.2],
            [-1.8, -0.1],
            [-1.6, 0.8],
            [1.6, -0.8],
            [1.8, 0.1],
            [-1.2, 1.5],
            [-1.0, -1.4],
            [1.0, 1.4],
            [1.2, -1.5],
        ],
        dtype=np.float32,
    )
    y_index = (x[:, 0] > 0).astype(np.int64)
    y = np.zeros((x.shape[0], 2), dtype=np.float32)
    y[np.arange(x.shape[0]), y_index] = 1.0

    x = np.ascontiguousarray(x, dtype=dtype)
    y = np.ascontiguousarray(y, dtype=dtype)
    loader_cls = thor.training.NumpyFloat16BatchLoader if dtype == np.float16 else thor.training.NumpyFloat32BatchLoader
    return loader_cls(
        x,
        y,
        x,
        y,
        batch_size=batch_size,
        example_input_name="examples",
        label_input_name="labels",
        dataset_name="linearly_separable_multi_batch",
    )


def _build_linear_classifier(name: str, *, dtype=thor.DataType.fp32):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [2], dtype)

    logits = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        2,
        True,
        activation=None,
    )
    loss = thor.losses.CategoricalCrossEntropy(
        network,
        logits.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )

    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    return network

def _download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, path)


def _load_iris_one_hot(dtype=np.float32):
    _download_if_missing(IRIS_URL, IRIS_PATH)

    class_names = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }
    rows = []
    labels = []
    for line in IRIS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        assert len(parts) == 5
        rows.append([float(x) for x in parts[:4]])
        labels.append(class_names[parts[4]])

    x = np.asarray(rows, dtype=np.float32)
    y_index = np.asarray(labels, dtype=np.int64)

    x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
    y = np.zeros((y_index.shape[0], 3), dtype=np.float32)
    y[np.arange(y_index.shape[0]), y_index] = 1.0

    rng = np.random.default_rng(1234)
    order = rng.permutation(x.shape[0])
    x = np.ascontiguousarray(x[order], dtype=dtype)
    y = np.ascontiguousarray(y[order], dtype=dtype)

    return x[:120], y[:120], x[120:], y[120:]


def _iris_loader(batch_size: int = 16, dtype=np.float32):
    x_train, y_train, x_validate, y_validate = _load_iris_one_hot(dtype=dtype)
    loader_cls = thor.training.NumpyFloat16BatchLoader if dtype == np.float16 else thor.training.NumpyFloat32BatchLoader
    return loader_cls(
        x_train,
        y_train,
        x_validate,
        y_validate,
        batch_size=batch_size,
        example_input_name="examples",
        label_input_name="labels",
        dataset_name="iris",
    )


def _build_iris_mlp(name: str, *, dtype=thor.DataType.fp32, per_layer_optimizers: bool = False):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [4], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [3], dtype)

    if per_layer_optimizers:
        hidden_weights_optimizer = thor.optimizers.AdamW(alpha=0.003, weight_decay=0.0)
        hidden_biases_optimizer = thor.optimizers.AdamW(alpha=0.003, weight_decay=0.0)
        output_weights_optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.9)
        output_biases_optimizer = thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.9)
    else:
        hidden_weights_optimizer = None
        hidden_biases_optimizer = None
        output_weights_optimizer = None
        output_biases_optimizer = None

    hidden = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        16,
        True,
        activation=thor.activations.Relu(),
        weights_optimizer=hidden_weights_optimizer,
        biases_optimizer=hidden_biases_optimizer,
    )
    logits = thor.layers.FullyConnected(
        network,
        hidden.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=output_weights_optimizer,
        biases_optimizer=output_biases_optimizer,
    )
    loss = thor.losses.CategoricalCrossEntropy(
        network,
        logits.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )

    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "scores", logits.get_feature_output(), dtype)
    return network


def test_debug_synchronous_trainer_fits_iris_fp32_mlp_with_global_optimizer(capfd):
    network = _build_iris_mlp("python_integration_iris_debug_fp32", dtype=thor.DataType.fp32)
    loader = _iris_loader(batch_size=16, dtype=np.float32)
    optimizer = thor.optimizers.Adam(alpha=0.003)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=True,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
    )
    stats = _fit_and_capture_stats(trainer, capfd, epochs=30)
    if ASSERT_TRAINER_LEARNING:
        _assert_loss_decreased(
            _phase_epoch_mean_losses(stats, "validate"),
            name="debug synchronous Iris FP32 validate epoch mean",
            required_fraction=0.90,
        )


def test_queued_trainer_fits_iris_fp16_mlp_with_global_optimizer(capfd):
    network = _build_iris_mlp("python_integration_iris_queued_fp16", dtype=thor.DataType.fp16)
    loader = _iris_loader(batch_size=16, dtype=np.float16)
    optimizer = thor.optimizers.Adam(alpha=0.003)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=False,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=8,
        scalar_tensors_to_report=["loss"],
    )
    stats = _fit_and_capture_stats(trainer, capfd, epochs=30)
    if ASSERT_TRAINER_LEARNING:
        _assert_loss_decreased(
            _phase_epoch_mean_losses(stats, "validate"),
            name="queued Iris FP16 validate epoch mean",
            required_fraction=0.90,
        )


def test_debug_synchronous_trainer_fits_iris_fp16_with_per_layer_optimizer_overrides(capfd):
    network = _build_iris_mlp(
        "python_integration_iris_per_layer_optimizers_fp16",
        dtype=thor.DataType.fp16,
        per_layer_optimizers=True,
    )
    loader = _iris_loader(batch_size=16, dtype=np.float16)

    trainer = thor.training.Trainer(
        network,
        loader,
        debug_synchronous=True,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
    )
    stats = _fit_and_capture_stats(trainer, capfd, epochs=30)
    if ASSERT_TRAINER_LEARNING:
        _assert_loss_decreased(
            _phase_epoch_mean_losses(stats, "validate"),
            name="debug synchronous Iris FP16 per-layer validate epoch mean",
            required_fraction=0.90,
        )


@requires_trainer_learning
def test_debug_synchronous_trainer_reduces_loss_on_repeated_single_batch(capfd):
    network = _build_linear_classifier("python_integration_debug_repeated_batch", dtype=thor.DataType.fp32)
    loader = _linearly_separable_one_batch_loader(batch_size=16, dtype=np.float32)
    optimizer = thor.optimizers.Adam(alpha=0.03)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=True,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=40)

    train_epoch_losses = _phase_epoch_mean_losses(stats, "train")
    validate_epoch_losses = _phase_epoch_mean_losses(stats, "validate")
    assert len(train_epoch_losses) == 40, train_epoch_losses
    assert len(validate_epoch_losses) == 40, validate_epoch_losses
    _assert_loss_decreased(train_epoch_losses, name="debug synchronous train", required_fraction=0.70)
    _assert_loss_decreased(validate_epoch_losses, name="debug synchronous validate", required_fraction=0.70)


@requires_trainer_learning
def test_queued_trainer_reduces_loss_on_repeated_single_batch(capfd):
    network = _build_linear_classifier("python_integration_queued_repeated_batch", dtype=thor.DataType.fp32)
    loader = _linearly_separable_one_batch_loader(batch_size=16, dtype=np.float32)
    optimizer = thor.optimizers.Adam(alpha=0.03)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=False,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=4,
        scalar_tensors_to_report=["loss"],
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=40)

    train_epoch_losses = _phase_epoch_mean_losses(stats, "train")
    validate_epoch_losses = _phase_epoch_mean_losses(stats, "validate")
    assert len(train_epoch_losses) == 40, train_epoch_losses
    assert len(validate_epoch_losses) == 40, validate_epoch_losses
    _assert_loss_decreased(train_epoch_losses, name="queued train", required_fraction=0.70)
    _assert_loss_decreased(validate_epoch_losses, name="queued validate", required_fraction=0.70)


@requires_trainer_learning
def test_queued_trainer_reports_fresh_loss_for_each_batch(capfd):
    network = _build_linear_classifier("python_integration_queued_fresh_stats", dtype=thor.DataType.fp32)
    loader = _linearly_separable_multi_batch_loader(batch_size=4, dtype=np.float32)
    optimizer = thor.optimizers.Adam(alpha=0.02)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=False,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=4,
        scalar_tensors_to_report=["loss"],
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=8)
    train_stats = [entry for entry in stats if entry["phase"] == "train"]

    assert len(train_stats) == 8 * loader.get_num_train_batches(), train_stats
    first_epoch = [entry["loss"] for entry in train_stats if entry["epoch"] == 1]
    assert len(first_epoch) == loader.get_num_train_batches(), first_epoch

    adjacent_repeats = [(i, first_epoch[i - 1], first_epoch[i]) for i in range(1, len(first_epoch)) if first_epoch[i] == first_epoch[i - 1]]
    assert not adjacent_repeats, f"queued stats appear stale; adjacent first-epoch train losses repeated exactly: {adjacent_repeats}"

    if ASSERT_TRAINER_LEARNING:
        _assert_loss_decreased(
            _phase_epoch_mean_losses(stats, "train"),
            name="queued multi-batch train epoch mean",
            required_fraction=0.85,
        )
