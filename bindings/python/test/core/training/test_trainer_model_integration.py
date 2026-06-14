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
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

_TRAINER_STATS_RE = re.compile(
    r"INFO trainer:\s+phase=\s*(?P<phase>train|validate|test)\s+"
    r"epoch=\s*(?P<epoch>\d+)/(?:\d+)\s+"
    r"step=\s*(?P<step>\d+)\s+"
    r"batch=\s*(?P<batch>\d+)/(?:\d+)\s+"
    r"loss=\s*(?P<loss>[-+0-9.eE]+)")


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
            })
    return stats


def _phase_losses(stats, phase: str):
    return [entry["loss"] for entry in stats if entry["phase"] == phase]


def _phase_epoch_mean_losses(stats, phase: str):
    grouped = {}
    for entry in stats:
        if entry["phase"] == phase:
            grouped.setdefault(entry["epoch"], []).append(entry["loss"])
    return [float(np.mean(grouped[epoch])) for epoch in sorted(grouped)]


def _assert_zero_initialized_two_class_batch_loss(loss: float, *, batch_size: int, name: str):
    del batch_size
    expected = float(np.log(2.0))
    message = f"{name}: expected zero-logit two-class CE mean loss {expected}, got {loss}"
    assert loss == pytest.approx(expected, rel=2e-3, abs=2e-3), message


def _assert_zero_initialized_regression_mean_loss(loss: float, *, expected: float, name: str):
    assert loss == pytest.approx(
        expected, rel=2e-3, abs=2e-3), f"{name}: expected zero-init mean MSE loss {expected}, got {loss}"


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


def _axis_separable_one_batch_loader(*, batch_size: int = 4, dtype=np.float32):
    # Small enough to hand-check and intentionally symmetric so zero-initialized
    # logits produce exactly batch_size * log(2) loss before the first update.
    x = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_index = (x[:, 0] > 0.0).astype(np.int64)
    y = np.zeros((x.shape[0], 2), dtype=np.float32)
    y[np.arange(x.shape[0]), y_index] = 1.0

    assert x.shape[0] == batch_size
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
        dataset_name="axis_separable_one_batch",
    )


def _regression_one_batch_loader(*, batch_size: int = 4, dtype=np.float32):
    # Smallest useful training target: a single affine output trained with MSE.
    # With zero weights and zero bias, predictions are all 0 and Thor reports
    # mean(y ** 2) == 1.0, which makes the first loss hand-checkable.
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

    assert x.shape[0] == batch_size
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
        dataset_name="regression_one_batch",
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


def _build_linear_classifier(
    name: str,
    *,
    dtype=thor.DataType.fp32,
    weights_initializer=None,
    biases_initializer=None,
):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [2], dtype)

    logits = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        2,
        True,
        activation=None,
        weights_initializer=weights_initializer,
        biases_initializer=biases_initializer,
    )
    loss = thor.losses.CategoricalCrossEntropy(
        network,
        logits.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )

    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    return network


def _build_zero_initialized_linear_classifier(name: str, *, dtype=thor.DataType.fp32):
    zero = thor.initializers.UniformRandom(0.0, 0.0)
    return _build_linear_classifier(
        name,
        dtype=dtype,
        weights_initializer=zero,
        biases_initializer=zero,
    )


def _build_linear_regressor(
    name: str,
    *,
    dtype=thor.DataType.fp32,
    weights_initializer=None,
    biases_initializer=None,
):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [2], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [1], dtype)

    predictions = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        1,
        True,
        activation=None,
        weights_initializer=weights_initializer,
        biases_initializer=biases_initializer,
    )
    loss = thor.losses.MSE(
        network,
        predictions.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )

    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "predictions", predictions.get_feature_output(), dtype)
    return network


def _build_zero_initialized_linear_regressor(name: str, *, dtype=thor.DataType.fp32):
    zero = thor.initializers.UniformRandom(0.0, 0.0)
    return _build_linear_regressor(
        name,
        dtype=dtype,
        weights_initializer=zero,
        biases_initializer=zero,
    )


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
    return _build_iris_two_layer_classifier(
        name,
        dtype=dtype,
        hidden_activation=thor.activations.Relu(),
        per_layer_optimizers=per_layer_optimizers,
    )


def _build_iris_two_layer_classifier(
    name: str,
    *,
    dtype=thor.DataType.fp32,
    hidden_activation=None,
    per_layer_optimizers: bool = False,
):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [4], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [3], dtype)

    hidden_weights_initializer = None
    hidden_biases_initializer = None
    output_weights_initializer = None
    output_biases_initializer = None

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
        activation=hidden_activation,
        weights_initializer=hidden_weights_initializer,
        biases_initializer=hidden_biases_initializer,
        weights_optimizer=hidden_weights_optimizer,
        biases_optimizer=hidden_biases_optimizer,
    )
    logits = thor.layers.FullyConnected(
        network,
        hidden.get_feature_output(),
        3,
        True,
        activation=None,
        weights_initializer=output_weights_initializer,
        biases_initializer=output_biases_initializer,
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


def _make_deep_layer_optimizer_override(layer_index: int):
    # Fresh optimizer objects per parameter keep optimizer-owned state distinct
    # while still covering mixed per-layer/per-parameter override plumbing.
    if layer_index % 2 == 0:
        return thor.optimizers.AdamW(alpha=0.002, weight_decay=0.0)
    return thor.optimizers.Sgd(initial_learning_rate=0.015, momentum=0.9)


def _build_iris_deep_classifier(
    name: str,
    *,
    dtype=thor.DataType.fp32,
    hidden_layers,
    use_layer_norm: bool = False,
    per_layer_optimizer_overrides: bool = False,
    emit_extra_outputs: bool = False,
):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [4], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [3], dtype)

    features = examples.get_feature_output()
    for layer_index, (num_output_features, activation) in enumerate(hidden_layers):
        if per_layer_optimizer_overrides:
            weights_optimizer = _make_deep_layer_optimizer_override(layer_index)
            biases_optimizer = _make_deep_layer_optimizer_override(layer_index)
        else:
            weights_optimizer = None
            biases_optimizer = None

        dense = thor.layers.FullyConnected(
            network,
            features,
            num_output_features,
            True,
            activation=activation,
            weights_optimizer=weights_optimizer,
            biases_optimizer=biases_optimizer,
        )
        features = dense.get_feature_output()

        # Add normalization between hidden layers, not after the final hidden
        # projection, so the next trainable layer consumes normalized activations.
        if use_layer_norm and layer_index + 1 < len(hidden_layers):
            norm = thor.layers.LayerNorm(network, features)
            features = norm.get_feature_output()

    output_layer_index = len(hidden_layers)
    if per_layer_optimizer_overrides:
        output_weights_optimizer = _make_deep_layer_optimizer_override(output_layer_index)
        output_biases_optimizer = _make_deep_layer_optimizer_override(output_layer_index)
    else:
        output_weights_optimizer = None
        output_biases_optimizer = None

    logits = thor.layers.FullyConnected(
        network,
        features,
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
    if emit_extra_outputs:
        thor.layers.NetworkOutput(network, "features", features, dtype)
    return network


def _build_iris_two_input_multi_output_classifier(name: str, *, dtype=thor.DataType.fp32):
    network = thor.Network(name)
    examples_left = thor.layers.NetworkInput(network, "examples_left", [4], dtype)
    examples_right = thor.layers.NetworkInput(network, "examples_right", [4], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [3], dtype)

    # Keep this queued-trainer coverage fully serializable.  The native queued
    # runner serializes its network/program setup, so using a Python CustomLayer
    # here would correctly fail unless it was built from a serializable
    # ExpressionDefinition.  Two ordinary branches still exercise explicit
    # input binding, multiple NetworkInputs, multiple NetworkOutputs, and
    # multiple loss roots in one TrainingStep.
    left_hidden = thor.layers.FullyConnected(
        network,
        examples_left.get_feature_output(),
        16,
        True,
        activation=thor.activations.Relu(),
    )
    left_logits = thor.layers.FullyConnected(
        network,
        left_hidden.get_feature_output(),
        3,
        True,
        activation=None,
    )
    left_loss = thor.losses.CategoricalCrossEntropy(
        network,
        left_logits.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )

    right_hidden = thor.layers.FullyConnected(
        network,
        examples_right.get_feature_output(),
        12,
        True,
        activation=thor.activations.Tanh(),
    )
    right_logits = thor.layers.FullyConnected(
        network,
        right_hidden.get_feature_output(),
        3,
        True,
        activation=None,
    )
    right_loss = thor.losses.CategoricalCrossEntropy(
        network,
        right_logits.get_feature_output(),
        labels.get_feature_output(),
        thor.DataType.fp32,
    )

    thor.layers.NetworkOutput(network, "loss", left_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "aux_loss", right_loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "scores_left", left_logits.get_feature_output(), dtype)
    thor.layers.NetworkOutput(network, "scores_right", right_logits.get_feature_output(), dtype)
    return network, [left_loss.get_loss(), right_loss.get_loss()]


def _build_iris_frozen_hidden_classifier(name: str, *, dtype=thor.DataType.fp32):
    network = thor.Network(name)
    examples = thor.layers.NetworkInput(network, "examples", [4], dtype)
    labels = thor.layers.NetworkInput(network, "labels", [3], dtype)

    hidden = thor.layers.FullyConnected(
        network,
        examples.get_feature_output(),
        32,
        True,
        activation=thor.activations.Relu(),
    )
    frozen_hidden = thor.layers.FullyConnected(
        network,
        hidden.get_feature_output(),
        24,
        True,
        activation=thor.activations.Relu(),
    )
    frozen_hidden.freeze_training()
    logits = thor.layers.FullyConnected(
        network,
        frozen_hidden.get_feature_output(),
        3,
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
    thor.layers.NetworkOutput(network, "scores", logits.get_feature_output(), dtype)
    thor.layers.NetworkOutput(network, "frozen_features", frozen_hidden.get_feature_output(), dtype)
    return network


def _assert_phase_batch_count(stats, phase: str, *, epochs: int, batches_per_epoch: int):
    losses = _phase_losses(stats, phase)
    expected = epochs * batches_per_epoch
    assert len(losses) == expected, f"expected {expected} {phase} losses, got {len(losses)}: {losses}"


# Diagnostic ladder, from deterministic single-batch checks to real Iris training.


def test_debug_synchronous_trainer_reports_deterministic_zero_init_mse_one_batch_loss(capfd):
    batch_size = 4
    network = _build_zero_initialized_linear_regressor(
        "python_integration_debug_zero_init_mse_one_batch_loss",
        dtype=thor.DataType.fp32,
    )
    loader = _regression_one_batch_loader(batch_size=batch_size, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=1.0e-12, momentum=0.0)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=True,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=1)
    train_losses = _phase_losses(stats, "train")
    validate_losses = _phase_losses(stats, "validate")

    assert len(train_losses) == 1, train_losses
    assert len(validate_losses) == 1, validate_losses
    _assert_zero_initialized_regression_mean_loss(
        train_losses[0],
        expected=1.0,
        name="debug synchronous zero-init MSE first train batch",
    )
    _assert_zero_initialized_regression_mean_loss(
        validate_losses[0],
        expected=1.0,
        name="debug synchronous near-zero-update MSE validate batch",
    )


def test_debug_synchronous_trainer_applies_deterministic_one_batch_mse_sgd_update(capfd):
    batch_size = 4
    network = _build_zero_initialized_linear_regressor(
        "python_integration_debug_one_batch_mse_sgd_update",
        dtype=thor.DataType.fp32,
    )
    loader = _regression_one_batch_loader(batch_size=batch_size, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.1, momentum=0.0)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=True,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=2)
    train_losses = _phase_losses(stats, "train")
    validate_losses = _phase_losses(stats, "validate")

    assert len(train_losses) == 2, train_losses
    assert len(validate_losses) == 2, validate_losses
    _assert_zero_initialized_regression_mean_loss(
        train_losses[0],
        expected=1.0,
        name="debug synchronous MSE pre-update train batch",
    )

    # Same deterministic batch is used for validation.  If the first train batch
    # actually updated parameters, either the immediately following validation
    # loss or the next epoch's train loss must move below the zero-init baseline.
    post_first_update_loss = min(validate_losses[0], train_losses[1])
    assert post_first_update_loss < train_losses[0] * 0.90, (
        "one-batch MSE SGD update did not materially change the next loss: "
        f"train_losses={train_losses}, validate_losses={validate_losses}")


def test_debug_synchronous_trainer_reduces_deterministic_repeated_batch_mse_loss_with_sgd(capfd):
    batch_size = 4
    network = _build_zero_initialized_linear_regressor(
        "python_integration_debug_repeated_batch_mse_sgd",
        dtype=thor.DataType.fp32,
    )
    loader = _regression_one_batch_loader(batch_size=batch_size, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.05, momentum=0.0)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=True,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=12)
    train_epoch_losses = _phase_epoch_mean_losses(stats, "train")
    validate_epoch_losses = _phase_epoch_mean_losses(stats, "validate")

    assert len(train_epoch_losses) == 12, train_epoch_losses
    assert len(validate_epoch_losses) == 12, validate_epoch_losses
    _assert_zero_initialized_regression_mean_loss(
        train_epoch_losses[0],
        expected=1.0,
        name="debug synchronous repeated-batch MSE initial train epoch",
    )
    _assert_loss_decreased(
        train_epoch_losses,
        name="debug synchronous deterministic repeated-batch MSE train",
        tail_window=3,
        required_fraction=0.50,
    )
    _assert_loss_decreased(
        validate_epoch_losses,
        name="debug synchronous deterministic repeated-batch MSE validate",
        tail_window=3,
        required_fraction=0.50,
    )


def test_queued_trainer_reduces_deterministic_repeated_batch_mse_loss_with_sgd(capfd):
    batch_size = 4
    network = _build_zero_initialized_linear_regressor(
        "python_integration_queued_repeated_batch_mse_sgd",
        dtype=thor.DataType.fp32,
    )
    loader = _regression_one_batch_loader(batch_size=batch_size, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.05, momentum=0.0)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=False,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=4,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=12)
    train_epoch_losses = _phase_epoch_mean_losses(stats, "train")
    validate_epoch_losses = _phase_epoch_mean_losses(stats, "validate")

    assert len(train_epoch_losses) == 12, train_epoch_losses
    assert len(validate_epoch_losses) == 12, validate_epoch_losses
    _assert_zero_initialized_regression_mean_loss(
        train_epoch_losses[0],
        expected=1.0,
        name="queued repeated-batch MSE initial train epoch",
    )
    _assert_loss_decreased(
        train_epoch_losses,
        name="queued deterministic repeated-batch MSE train",
        tail_window=3,
        required_fraction=0.50,
    )
    _assert_loss_decreased(
        validate_epoch_losses,
        name="queued deterministic repeated-batch MSE validate",
        tail_window=3,
        required_fraction=0.50,
    )


def test_debug_synchronous_trainer_reports_deterministic_zero_init_categorical_ce_one_batch_loss(capfd):
    batch_size = 4
    network = _build_zero_initialized_linear_classifier(
        "python_integration_debug_zero_init_categorical_ce_one_batch_loss",
        dtype=thor.DataType.fp32,
    )
    loader = _axis_separable_one_batch_loader(batch_size=batch_size, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=1.0e-12, momentum=0.0)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=True,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=1)
    train_losses = _phase_losses(stats, "train")
    validate_losses = _phase_losses(stats, "validate")

    assert len(train_losses) == 1, train_losses
    assert len(validate_losses) == 1, validate_losses
    _assert_zero_initialized_two_class_batch_loss(
        train_losses[0],
        batch_size=batch_size,
        name="debug synchronous zero-init first train batch",
    )
    _assert_zero_initialized_two_class_batch_loss(
        validate_losses[0],
        batch_size=batch_size,
        name="debug synchronous near-zero-update validate batch",
    )


def test_debug_synchronous_trainer_applies_deterministic_one_batch_sgd_update(capfd):
    batch_size = 4
    network = _build_zero_initialized_linear_classifier(
        "python_integration_debug_one_batch_sgd_update",
        dtype=thor.DataType.fp32,
    )
    loader = _axis_separable_one_batch_loader(batch_size=batch_size, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.1, momentum=0.0)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=True,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=2)
    train_losses = _phase_losses(stats, "train")
    validate_losses = _phase_losses(stats, "validate")

    assert len(train_losses) == 2, train_losses
    assert len(validate_losses) == 2, validate_losses
    _assert_zero_initialized_two_class_batch_loss(
        train_losses[0],
        batch_size=batch_size,
        name="debug synchronous pre-update train batch",
    )

    # Same deterministic batch is used for validation.  If the first train batch
    # actually updated parameters, either the immediately following validation
    # loss or the next epoch's train loss must move below the zero-init baseline.
    post_first_update_loss = min(validate_losses[0], train_losses[1])
    assert post_first_update_loss < train_losses[0] * 0.90, (
        "one-batch SGD update did not materially change the next loss: "
        f"train_losses={train_losses}, validate_losses={validate_losses}")


def test_debug_synchronous_trainer_reduces_deterministic_repeated_batch_loss_with_sgd(capfd):
    batch_size = 4
    network = _build_zero_initialized_linear_classifier(
        "python_integration_debug_repeated_batch_sgd",
        dtype=thor.DataType.fp32,
    )
    loader = _axis_separable_one_batch_loader(batch_size=batch_size, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.05, momentum=0.0)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=True,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=12)
    train_epoch_losses = _phase_epoch_mean_losses(stats, "train")
    validate_epoch_losses = _phase_epoch_mean_losses(stats, "validate")

    assert len(train_epoch_losses) == 12, train_epoch_losses
    assert len(validate_epoch_losses) == 12, validate_epoch_losses
    _assert_zero_initialized_two_class_batch_loss(
        train_epoch_losses[0],
        batch_size=batch_size,
        name="debug synchronous repeated-batch initial train epoch",
    )
    _assert_loss_decreased(
        train_epoch_losses,
        name="debug synchronous deterministic repeated-batch train",
        tail_window=3,
        required_fraction=0.60,
    )
    _assert_loss_decreased(
        validate_epoch_losses,
        name="debug synchronous deterministic repeated-batch validate",
        tail_window=3,
        required_fraction=0.60,
    )


def test_queued_trainer_reduces_deterministic_repeated_batch_loss_with_sgd(capfd):
    batch_size = 4
    network = _build_zero_initialized_linear_classifier(
        "python_integration_queued_repeated_batch_sgd",
        dtype=thor.DataType.fp32,
    )
    loader = _axis_separable_one_batch_loader(batch_size=batch_size, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.05, momentum=0.0)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=False,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=4,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=12)
    train_epoch_losses = _phase_epoch_mean_losses(stats, "train")
    validate_epoch_losses = _phase_epoch_mean_losses(stats, "validate")

    assert len(train_epoch_losses) == 12, train_epoch_losses
    assert len(validate_epoch_losses) == 12, validate_epoch_losses
    _assert_zero_initialized_two_class_batch_loss(
        train_epoch_losses[0],
        batch_size=batch_size,
        name="queued repeated-batch initial train epoch",
    )
    _assert_loss_decreased(
        train_epoch_losses,
        name="queued deterministic repeated-batch train",
        tail_window=3,
        required_fraction=0.60,
    )
    _assert_loss_decreased(
        validate_epoch_losses,
        name="queued deterministic repeated-batch validate",
        tail_window=3,
        required_fraction=0.60,
    )


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
    _assert_loss_decreased(
        _phase_epoch_mean_losses(stats, "validate"),
        name="debug synchronous Iris FP16 per-layer validate epoch mean",
        required_fraction=0.90,
    )


def test_debug_synchronous_trainer_fits_iris_fp32_deep_mixed_activation_mlp_with_sgd_momentum(capfd):
    network = _build_iris_deep_classifier(
        "python_integration_iris_debug_fp32_deep_mixed_activation_sgd",
        dtype=thor.DataType.fp32,
        hidden_layers=[
            (32, thor.activations.Tanh()),
            (24, thor.activations.Relu()),
            (16, thor.activations.Gelu()),
            (12, thor.activations.Swish()),
        ],
    )
    loader = _iris_loader(batch_size=16, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.02, momentum=0.9)

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
    stats = _fit_and_capture_stats(trainer, capfd, epochs=35)
    _assert_loss_decreased(
        _phase_epoch_mean_losses(stats, "validate"),
        name="debug synchronous Iris FP32 deep mixed-activation SGD validate epoch mean",
        required_fraction=0.95,
    )


def test_queued_trainer_fits_iris_fp32_layer_norm_mlp_with_adamw(capfd):
    network = _build_iris_deep_classifier(
        "python_integration_iris_queued_fp32_layer_norm_adamw",
        dtype=thor.DataType.fp32,
        hidden_layers=[
            (32, thor.activations.Gelu()),
            (32, thor.activations.Swish()),
            (16, thor.activations.Tanh()),
        ],
        use_layer_norm=True,
    )
    loader = _iris_loader(batch_size=16, dtype=np.float32)
    optimizer = thor.optimizers.AdamW(alpha=0.002, weight_decay=0.0)

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
    stats = _fit_and_capture_stats(trainer, capfd, epochs=35)
    _assert_loss_decreased(
        _phase_epoch_mean_losses(stats, "validate"),
        name="queued Iris FP32 layer-norm AdamW validate epoch mean",
        required_fraction=0.95,
    )


def test_queued_trainer_fits_iris_fp16_deep_mlp_with_nadam(capfd):
    network = _build_iris_deep_classifier(
        "python_integration_iris_queued_fp16_deep_nadam",
        dtype=thor.DataType.fp16,
        hidden_layers=[
            (32, thor.activations.Relu()),
            (24, thor.activations.Gelu()),
            (16, thor.activations.Swish()),
        ],
    )
    loader = _iris_loader(batch_size=16, dtype=np.float16)
    optimizer = thor.optimizers.NAdam(alpha=0.002)

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
    stats = _fit_and_capture_stats(trainer, capfd, epochs=35)
    _assert_loss_decreased(
        _phase_epoch_mean_losses(stats, "validate"),
        name="queued Iris FP16 deep NAdam validate epoch mean",
        required_fraction=0.95,
    )


def test_queued_trainer_fits_iris_fp32_two_input_multi_output_network_with_explicit_bindings(capfd):
    network, loss_roots = _build_iris_two_input_multi_output_classifier(
        "python_integration_iris_queued_fp32_two_input_multi_output",
        dtype=thor.DataType.fp32,
    )
    loader = _iris_loader(batch_size=16, dtype=np.float32)
    optimizer = thor.optimizers.Adam(alpha=0.003)
    training_program = thor.training.TrainingProgram([
        thor.training.TrainingStep(
            "two_input_iris",
            loss_roots,
            optimizer=optimizer,
            update_parameters=network.get_trainable_parameter_references(),
            input_bindings=[
                thor.training.TrainingInputBinding("examples_left", "examples"),
                thor.training.TrainingInputBinding("examples_right", "examples"),
                thor.training.TrainingInputBinding("labels", "labels"),
            ],
        )
    ])

    trainer = thor.training.Trainer(
        network,
        loader,
        training_program=training_program,
        debug_synchronous=False,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=8,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )
    stats = _fit_and_capture_stats(trainer, capfd, epochs=30)
    _assert_phase_batch_count(stats, "train", epochs=30, batches_per_epoch=loader.get_num_train_batches())
    _assert_phase_batch_count(stats, "validate", epochs=30, batches_per_epoch=loader.get_num_validate_batches())
    _assert_loss_decreased(
        _phase_epoch_mean_losses(stats, "validate"),
        name="queued Iris FP32 two-input multi-output validate epoch mean",
        required_fraction=0.95,
    )


def test_queued_trainer_fits_iris_fp32_deep_mlp_with_per_layer_optimizer_overrides(capfd):
    network = _build_iris_deep_classifier(
        "python_integration_iris_queued_fp32_deep_per_layer_optimizers",
        dtype=thor.DataType.fp32,
        hidden_layers=[
            (32, thor.activations.Relu()),
            (24, thor.activations.Gelu()),
            (16, thor.activations.Tanh()),
        ],
        per_layer_optimizer_overrides=True,
    )
    loader = _iris_loader(batch_size=16, dtype=np.float32)

    trainer = thor.training.Trainer(
        network,
        loader,
        debug_synchronous=False,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=8,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )
    stats = _fit_and_capture_stats(trainer, capfd, epochs=30)
    _assert_loss_decreased(
        _phase_epoch_mean_losses(stats, "validate"),
        name="queued Iris FP32 deep per-layer optimizer validate epoch mean",
        required_fraction=0.95,
    )


def test_debug_synchronous_trainer_fits_iris_fp32_with_frozen_hidden_layer(capfd):
    network = _build_iris_frozen_hidden_classifier(
        "python_integration_iris_debug_fp32_frozen_hidden",
        dtype=thor.DataType.fp32,
    )
    all_trainable = network.get_trainable_parameter_references(training_enabled_only=False)
    enabled_trainable = network.get_trainable_parameter_references(training_enabled_only=True)
    assert len(enabled_trainable) < len(all_trainable)

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
        stats_color="never",
    )
    stats = _fit_and_capture_stats(trainer, capfd, epochs=35)
    _assert_loss_decreased(
        _phase_epoch_mean_losses(stats, "validate"),
        name="debug synchronous Iris FP32 frozen-hidden validate epoch mean",
        required_fraction=0.98,
    )


def test_queued_trainer_reports_iris_validation_with_extra_non_scalar_outputs(capfd):
    epochs = 24
    network = _build_iris_deep_classifier(
        "python_integration_iris_queued_validation_extra_outputs",
        dtype=thor.DataType.fp32,
        hidden_layers=[
            (24, thor.activations.Relu()),
            (12, thor.activations.Tanh()),
        ],
        emit_extra_outputs=True,
    )
    loader = _iris_loader(batch_size=16, dtype=np.float32)
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
        stats_color="never",
    )
    stats = _fit_and_capture_stats(trainer, capfd, epochs=epochs)
    _assert_phase_batch_count(stats, "train", epochs=epochs, batches_per_epoch=loader.get_num_train_batches())
    _assert_phase_batch_count(stats, "validate", epochs=epochs, batches_per_epoch=loader.get_num_validate_batches())
    _assert_loss_decreased(
        _phase_epoch_mean_losses(stats, "validate"),
        name="queued Iris FP32 validation with extra non-scalar outputs epoch mean",
        required_fraction=0.95,
    )


def test_queued_trainer_updates_with_materialized_loss_and_prediction_output(capfd):
    batch_size = 4
    network = _build_zero_initialized_linear_regressor(
        "python_integration_queued_materialized_mse_loss_and_prediction_output",
        dtype=thor.DataType.fp32,
    )
    loader = _regression_one_batch_loader(batch_size=batch_size, dtype=np.float32)
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.05, momentum=0.0)

    trainer = thor.training.Trainer(
        network,
        loader,
        optimizer=optimizer,
        debug_synchronous=False,
        stats=True,
        stats_interval_s=0.0,
        max_in_flight_batches=4,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )

    stats = _fit_and_capture_stats(trainer, capfd, epochs=8)
    train_losses = _phase_epoch_mean_losses(stats, "train")
    validate_losses = _phase_epoch_mean_losses(stats, "validate")
    assert len(train_losses) == 8, train_losses
    assert len(validate_losses) == 8, validate_losses
    _assert_zero_initialized_regression_mean_loss(
        train_losses[0],
        expected=1.0,
        name="queued materialized MSE loss initial train epoch",
    )
    _assert_loss_decreased(
        validate_losses,
        name="queued materialized MSE loss with prediction output validate",
        tail_window=2,
        required_fraction=0.70,
    )


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

    adjacent_repeats = [
        (i, first_epoch[i - 1], first_epoch[i])
        for i in range(1, len(first_epoch))
        if first_epoch[i] == first_epoch[i - 1]
    ]
    assert not adjacent_repeats, f"queued stats appear stale; adjacent first-epoch train losses repeated exactly: {adjacent_repeats}"

    _assert_loss_decreased(
        _phase_epoch_mean_losses(stats, "train"),
        name="queued multi-batch train epoch mean",
        tail_window=3,
        required_fraction=0.85,
    )
