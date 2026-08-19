import math

import numpy as np
import pytest
import thor
from thor.physical import Expression as ex


R = math.log(16.0)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _bounded_explicit(network: thor.Network, raw):
    def normalize(context):
        return {"normalized": context.input("raw") * (1.0 / R)}

    normalized = thor.layers.CustomLayer(
        network=network,
        inputs={"raw": raw},
        output_names=["normalized"],
        build=normalize,
        parameters=[],
    )["normalized"]
    unit = thor.activations.Tanh().add_to_network(network, normalized)

    def rescale(context):
        return {"bounded": context.input("unit") * R}

    return thor.layers.CustomLayer(
        network=network,
        inputs={"unit": unit},
        output_names=["bounded"],
        build=rescale,
        parameters=[],
    )["bounded"]


def _bounded_epilogue():
    z = thor.layers.FullyConnected.epilogue_input(
        output_dtype=thor.DataType.fp32,
        compute_dtype=thor.DataType.fp32,
    )
    return ex.tanh(z * (1.0 / R)) * R


def _add_base_log_rate(network: thor.Network, base, residual):
    def build(context):
        return {"log_rate": context.input("base") + context.input("residual")}

    return thor.layers.CustomLayer(
        network=network,
        inputs={"base": base, "residual": residual},
        output_names=["log_rate"],
        build=build,
        parameters=[],
    )["log_rate"]


def _build_poisson_head(name: str, *, fused: bool, initial_bias: float = 0.0):
    network = thor.Network(name)
    features = thor.layers.NetworkInput(network, "features", [1], thor.DataType.fp32).get_feature_output()
    base = thor.layers.NetworkInput(network, "base_log_rate", [1], thor.DataType.fp32).get_feature_output()
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32).get_feature_output()

    zero = thor.initializers.UniformRandom(0.0, 0.0)
    bias_init = thor.initializers.UniformRandom(initial_bias, initial_bias)
    raw_layer = thor.layers.FullyConnected(
        network,
        features,
        1,
        True,
        activation=None,
        weights_initializer=zero,
        biases_initializer=bias_init,
        weights_data_type=thor.DataType.fp32,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.fp32,
        epilogue=_bounded_epilogue() if fused else None,
    )
    residual = raw_layer.get_feature_output()
    if not fused:
        residual = _bounded_explicit(network, residual)

    log_rate = _add_base_log_rate(network, base, residual)
    loss = thor.losses.distribution.PoissonNLLLoss(
        network,
        log_rate,
        labels,
        log_input=True,
        full=False,
        eps=1.0e-8,
        loss_data_type=thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "residual", residual, thor.DataType.fp32)
    return network


def _training_data(*, target: float, rows: int = 4):
    tensors = {
        "features": np.zeros((rows, 1), dtype=np.float32),
        "base_log_rate": np.full((rows, 1), math.log(10.0), dtype=np.float32),
        "labels": np.full((rows, 1), target, dtype=np.float32),
    }
    dataset = thor.data.NumpyDataset(tensors)
    indices = np.arange(rows, dtype=np.int64)
    splits = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=indices,
        validate_indices=indices,
    )
    return thor.data.TrainingData(
        dataset=dataset,
        splits=splits,
        batching=thor.data.BatchPolicy(batch_size=rows, randomize_train=False),
        dataset_name=f"fc_epilogue_poisson_target_{target:g}",
        device_storage="off",
    )


def _make_trainer(
    name: str,
    *,
    fused: bool,
    target: float,
    initial_bias: float = 0.0,
    lr: float = 0.02,
    debug_synchronous: bool = False,
):
    return thor.training.Trainer(
        _build_poisson_head(name, fused=fused, initial_bias=initial_bias),
        data=_training_data(target=target),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=lr, momentum=0.0),
        debug_synchronous=debug_synchronous,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )


def _fit(
    name: str,
    *,
    fused: bool,
    target: float,
    epochs: int,
    initial_bias: float = 0.0,
    lr: float = 0.02,
    debug_synchronous: bool = False,
):
    trainer = _make_trainer(
        name,
        fused=fused,
        target=target,
        initial_bias=initial_bias,
        lr=lr,
        debug_synchronous=debug_synchronous,
    )
    result = trainer.fit(epochs=epochs, check_best_model_every_epochs=0)
    assert result.status == "completed"
    assert np.isfinite(result.final_training_loss)
    return float(result.final_training_loss)


def _expected_after_one_sgd_update(*, target: float, initial_bias: float, lr: float):
    unit = math.tanh(initial_bias / R)
    residual = R * unit
    log_rate = math.log(10.0) + residual
    rate = math.exp(log_rate)
    bias_gradient = (rate - target) * (1.0 - unit * unit)
    updated_bias = initial_bias - lr * bias_gradient
    updated_residual = R * math.tanh(updated_bias / R)
    updated_log_rate = math.log(10.0) + updated_residual
    updated_loss = math.exp(updated_log_rate) - target * updated_log_rate
    return updated_loss, updated_residual


def _saved_residual_after_fit(trainer, *, network_name: str, target: float, save_dir):
    trainer.save_model(str(save_dir), overwrite=True, save_optimizer_state=False)
    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    rows = 4
    placed = loaded.place(rows, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    available_inputs = {
        "features": _cpu_tensor(np.zeros((rows, 1), dtype=np.float32), thor.DataType.fp32),
        "base_log_rate": _cpu_tensor(
            np.full((rows, 1), math.log(10.0), dtype=np.float32), thor.DataType.fp32
        ),
        "labels": _cpu_tensor(np.full((rows, 1), target, dtype=np.float32), thor.DataType.fp32),
    }
    required_inputs = set(placed.get_network_input_names())
    assert required_inputs <= set(available_inputs)
    outputs = placed.infer({name: available_inputs[name] for name in required_inputs})
    residual = np.array(outputs["residual"].numpy(), copy=True)
    assert residual.shape == (rows, 1)
    assert np.all(np.isfinite(residual))
    return float(np.mean(residual))


@pytest.mark.cuda
def test_fc_rtanh_epilogue_training_matches_explicit_rtanh_gradient_path():
    """A fused R*tanh(z/R) FC epilogue must train like the explicit expression graph.

    Starting away from zero exercises the tanh derivative rather than only the
    identity slope at initialization. Two epochs make the reported final loss
    depend on the first optimizer update, so a wrong epilogue backward rule is
    observable even though Thor does not expose parameter gradients directly.
    """
    explicit_loss = _fit(
        "fc_rtanh_explicit_gradient_equivalence",
        fused=False,
        target=20.0,
        epochs=2,
        initial_bias=-0.5,
        lr=0.01,
    )
    fused_loss = _fit(
        "fc_rtanh_fused_gradient_equivalence",
        fused=True,
        target=20.0,
        epochs=2,
        initial_bias=-0.5,
        lr=0.01,
    )
    assert fused_loss == pytest.approx(explicit_loss, rel=2.0e-5, abs=2.0e-5)


@pytest.mark.cuda
def test_fc_explicit_rtanh_async_first_update_is_stable_under_repetition():
    """The explicit FC -> R*tanh(z/R) path establishes the Trainer/control baseline."""
    target = 20.0
    initial_bias = -0.5
    lr = 0.01
    expected_loss, _ = _expected_after_one_sgd_update(
        target=target, initial_bias=initial_bias, lr=lr
    )

    for iteration in range(12):
        trainer = _make_trainer(
            f"fc_rtanh_explicit_async_stress_{iteration}",
            fused=False,
            target=target,
            initial_bias=initial_bias,
            lr=lr,
        )
        result = trainer.fit(epochs=1, check_best_model_every_epochs=0)
        assert result.status == "completed"
        assert float(result.final_validation_loss) == pytest.approx(expected_loss, rel=2.0e-5, abs=2.0e-5)


@pytest.mark.cuda
def test_fc_fused_rtanh_async_first_update_is_stable_and_reports_saved_state_on_failure(tmp_path):
    """Stress the intermittent first-update lag and identify whether weights were actually updated.

    If validation observes stale parameters, save the fully synchronized model immediately.
    The failure message then distinguishes a validation/update ordering race from an update
    that was actually skipped: an updated saved residual means validation ran too early; an
    unchanged saved residual means the optimizer update itself did not commit correctly.
    """
    target = 20.0
    initial_bias = -0.5
    lr = 0.01
    expected_loss, expected_residual = _expected_after_one_sgd_update(
        target=target, initial_bias=initial_bias, lr=lr
    )
    initial_residual = R * math.tanh(initial_bias / R)

    for iteration in range(32):
        network_name = f"fc_rtanh_fused_async_stress_{iteration}"
        trainer = _make_trainer(
            network_name,
            fused=True,
            target=target,
            initial_bias=initial_bias,
            lr=lr,
        )
        result = trainer.fit(epochs=1, check_best_model_every_epochs=0)
        assert result.status == "completed"
        observed_loss = float(result.final_validation_loss)
        if observed_loss != pytest.approx(expected_loss, rel=2.0e-5, abs=2.0e-5):
            saved_residual = _saved_residual_after_fit(
                trainer,
                network_name=network_name,
                target=target,
                save_dir=tmp_path / f"failed_iteration_{iteration}",
            )
            saved_state = (
                "updated"
                if saved_residual == pytest.approx(expected_residual, rel=2.0e-5, abs=2.0e-5)
                else "still_initial"
                if saved_residual == pytest.approx(initial_residual, rel=2.0e-5, abs=2.0e-5)
                else "unexpected"
            )
            pytest.fail(
                "fused FC epilogue first update was not visible to validation: "
                f"iteration={iteration} validation_loss={observed_loss:.9f} "
                f"expected_validation_loss={expected_loss:.9f} "
                f"saved_residual={saved_residual:.9f} "
                f"expected_updated_residual={expected_residual:.9f} "
                f"initial_residual={initial_residual:.9f} saved_state={saved_state}"
            )


@pytest.mark.cuda
def test_fc_fused_rtanh_synchronous_first_update_is_stable_under_repetition():
    """If only the async stress fails, the defect is an execution-ordering problem, not VJP math."""
    target = 20.0
    initial_bias = -0.5
    lr = 0.01
    expected_loss, _ = _expected_after_one_sgd_update(
        target=target, initial_bias=initial_bias, lr=lr
    )

    for iteration in range(12):
        trainer = _make_trainer(
            f"fc_rtanh_fused_sync_stress_{iteration}",
            fused=True,
            target=target,
            initial_bias=initial_bias,
            lr=lr,
            debug_synchronous=True,
        )
        result = trainer.fit(epochs=1, check_best_model_every_epochs=0)
        assert result.status == "completed"
        assert float(result.final_validation_loss) == pytest.approx(expected_loss, rel=2.0e-5, abs=2.0e-5)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("target", "name"),
    [(20.0, "requires_positive_residual"), (5.0, "requires_negative_residual")],
)
def test_fc_rtanh_epilogue_poisson_gradient_moves_loss_in_correct_direction(target, name):
    """From residual zero, Poisson training must move toward the target rate.

    Base rate is 10. For target 20 the correct residual gradient is positive;
    for target 5 it is negative. In either case one SGD update must lower the
    next epoch's loss. A sign error in the fused epilogue backward path fails
    one or both cases immediately.
    """
    initial_loss = 10.0 - target * math.log(10.0)
    trained_loss = _fit(
        f"fc_rtanh_poisson_direction_{name}",
        fused=True,
        target=target,
        epochs=2,
        initial_bias=0.0,
        lr=0.01,
    )
    assert trained_loss < initial_loss - 1.0e-3


@pytest.mark.cuda
def test_fc_rtanh_epilogue_recovers_from_negative_residual_when_target_requires_increase():
    """The fused head must be able to climb back from a substantial negative correction."""
    target = 20.0
    initial_bias = -2.0
    initial_residual = R * math.tanh(initial_bias / R)
    initial_log_rate = math.log(10.0) + initial_residual
    initial_loss = math.exp(initial_log_rate) - target * initial_log_rate

    trained_loss = _fit(
        "fc_rtanh_poisson_negative_residual_recovery",
        fused=True,
        target=target,
        epochs=12,
        initial_bias=initial_bias,
        lr=0.02,
    )
    assert trained_loss < initial_loss - 0.25


def _build_scalar_bias_poisson_head(
    name: str,
    *,
    initial_bias: float,
    force_materialized_optimizer: bool,
):
    """Minimal trainable CustomLayer used to isolate dense optimizer-update fusion.

    A one-input CustomLayer is eligible for Thor's dense optimizer-update fusion.
    Adding a second equation input deliberately makes canFuseOptimizerUpdatesForApplication()
    return false, forcing the same parameter through the materialized-gradient update path.
    """
    network = thor.Network(name)
    base = thor.layers.NetworkInput(network, "base_log_rate", [1], thor.DataType.fp32).get_feature_output()
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32).get_feature_output()

    inputs = {"base": base}
    if force_materialized_optimizer:
        dummy = thor.layers.NetworkInput(network, "dummy", [1], thor.DataType.fp32).get_feature_output()
        inputs["dummy"] = dummy

    bias = thor.parameters.ParameterSpecification(
        name="bias",
        shape=[1],
        dtype=thor.DataType.fp32,
        initializer=thor.initializers.UniformRandom(initial_bias, initial_bias),
        trainable=True,
    )

    def build(context):
        log_rate = context.input("base") + context.param("bias")
        if force_materialized_optimizer:
            # Keep a second declared mathematical input on the layer. Even if the
            # expression compiler folds this zero contribution away, optimizer
            # fusion eligibility remains conservative because not every declared
            # equation input is represented in the prepared stamp inputs.
            log_rate = log_rate + context.input("dummy") * 0.0
        return {"log_rate": log_rate}

    log_rate = thor.layers.CustomLayer(
        network=network,
        inputs=inputs,
        output_names=["log_rate"],
        parameters=[bias],
        build=build,
    )["log_rate"]

    loss = thor.losses.distribution.PoissonNLLLoss(
        network,
        log_rate,
        labels,
        log_input=True,
        full=False,
        eps=1.0e-8,
        loss_data_type=thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    return network


def _scalar_bias_training_data(*, target: float, force_materialized_optimizer: bool, rows: int = 4):
    tensors = {
        "base_log_rate": np.full((rows, 1), math.log(10.0), dtype=np.float32),
        "labels": np.full((rows, 1), target, dtype=np.float32),
    }
    if force_materialized_optimizer:
        tensors["dummy"] = np.zeros((rows, 1), dtype=np.float32)
    dataset = thor.data.NumpyDataset(tensors)
    indices = np.arange(rows, dtype=np.int64)
    splits = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=indices,
        validate_indices=indices,
    )
    return thor.data.TrainingData(
        dataset=dataset,
        splits=splits,
        batching=thor.data.BatchPolicy(batch_size=rows, randomize_train=False),
        dataset_name=(
            "scalar_bias_poisson_materialized"
            if force_materialized_optimizer
            else "scalar_bias_poisson_fused_optimizer"
        ),
        device_storage="off",
    )


def _expected_direct_bias_after_one_sgd_update(*, target: float, initial_bias: float, lr: float):
    initial_log_rate = math.log(10.0) + initial_bias
    rate = math.exp(initial_log_rate)
    updated_bias = initial_bias - lr * (rate - target)
    updated_log_rate = math.log(10.0) + updated_bias
    return math.exp(updated_log_rate) - target * updated_log_rate


def _fit_scalar_bias_once(
    name: str,
    *,
    target: float,
    initial_bias: float,
    lr: float,
    force_materialized_optimizer: bool,
):
    trainer = thor.training.Trainer(
        _build_scalar_bias_poisson_head(
            name,
            initial_bias=initial_bias,
            force_materialized_optimizer=force_materialized_optimizer,
        ),
        data=_scalar_bias_training_data(
            target=target,
            force_materialized_optimizer=force_materialized_optimizer,
        ),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=lr, momentum=0.0),
        debug_synchronous=False,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )
    result = trainer.fit(epochs=1, check_best_model_every_epochs=0)
    assert result.status == "completed"
    return float(result.final_validation_loss)


def _build_scalar_residual_through_multi_input_add_head(name: str, *, initial_bias: float):
    """Pointwise trainable residual whose downstream add runs on a different input stream.

    This is intentionally free of matmul.  The residual branch is produced on the
    `features` stream, while `_add_base_log_rate()` chooses its first (`base`) input
    as the CustomLayer compute stream.  Its backward therefore has to publish the
    residual gradient from the base stream back to the residual/features stream.
    """
    network = thor.Network(name)
    features = thor.layers.NetworkInput(network, "features", [1], thor.DataType.fp32).get_feature_output()
    base = thor.layers.NetworkInput(network, "base_log_rate", [1], thor.DataType.fp32).get_feature_output()
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32).get_feature_output()

    bias = thor.parameters.ParameterSpecification(
        name="bias",
        shape=[1],
        dtype=thor.DataType.fp32,
        initializer=thor.initializers.UniformRandom(initial_bias, initial_bias),
        trainable=True,
    )

    def build_residual(context):
        # Keep the feature input live so this trainable layer is unambiguously owned
        # by the features connection stream while retaining scalar/pointwise math.
        return {"residual": context.input("features") + context.param("bias")}

    residual = thor.layers.CustomLayer(
        network=network,
        inputs={"features": features},
        output_names=["residual"],
        parameters=[bias],
        build=build_residual,
    )["residual"]
    log_rate = _add_base_log_rate(network, base, residual)
    loss = thor.losses.distribution.PoissonNLLLoss(
        network,
        log_rate,
        labels,
        log_input=True,
        full=False,
        eps=1.0e-8,
        loss_data_type=thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    return network


def _fit_scalar_residual_through_multi_input_add_once(
    name: str, *, target: float, initial_bias: float, lr: float
):
    rows = 4
    dataset = thor.data.NumpyDataset(
        {
            "features": np.zeros((rows, 1), dtype=np.float32),
            "base_log_rate": np.full((rows, 1), math.log(10.0), dtype=np.float32),
            "labels": np.full((rows, 1), target, dtype=np.float32),
        }
    )
    indices = np.arange(rows, dtype=np.int64)
    trainer = thor.training.Trainer(
        _build_scalar_residual_through_multi_input_add_head(name, initial_bias=initial_bias),
        data=thor.data.TrainingData(
            dataset=dataset,
            splits=thor.data.DatasetSplitManifest(
                dataset=dataset,
                train_indices=indices,
                validate_indices=indices,
            ),
            batching=thor.data.BatchPolicy(batch_size=rows, randomize_train=False),
            dataset_name="scalar_residual_multi_input_backward_stream_handoff",
            device_storage="off",
        ),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=lr, momentum=0.0),
        debug_synchronous=False,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )
    result = trainer.fit(epochs=1, check_best_model_every_epochs=0)
    assert result.status == "completed"
    return float(result.final_validation_loss)


@pytest.mark.cuda
def test_multi_input_custom_layer_backward_publishes_gradient_to_upstream_stream_under_repetition():
    """A downstream multi-input CustomLayer must publish each dx to its upstream stream."""
    target = 20.0
    initial_bias = -0.5
    lr = 0.01
    expected_loss = _expected_direct_bias_after_one_sgd_update(
        target=target,
        initial_bias=initial_bias,
        lr=lr,
    )

    for iteration in range(64):
        observed_loss = _fit_scalar_residual_through_multi_input_add_once(
            f"scalar_residual_multi_input_backward_stream_stress_{iteration}",
            target=target,
            initial_bias=initial_bias,
            lr=lr,
        )
        assert observed_loss == pytest.approx(expected_loss, rel=2.0e-5, abs=2.0e-5), (
            "multi-input CustomLayer did not publish its residual gradient to the upstream stream: "
            f"iteration={iteration} observed={observed_loss:.9f} expected={expected_loss:.9f}"
        )


@pytest.mark.cuda
def test_single_parameter_dense_fused_optimizer_async_first_update_is_stable_under_repetition():
    """Isolate the queued dense optimizer-fusion path without FC/matmul/epilogue complexity."""
    target = 20.0
    initial_bias = -0.5
    lr = 0.01
    expected_loss = _expected_direct_bias_after_one_sgd_update(
        target=target,
        initial_bias=initial_bias,
        lr=lr,
    )

    for iteration in range(32):
        observed_loss = _fit_scalar_bias_once(
            f"scalar_bias_fused_optimizer_async_stress_{iteration}",
            target=target,
            initial_bias=initial_bias,
            lr=lr,
            force_materialized_optimizer=False,
        )
        assert observed_loss == pytest.approx(expected_loss, rel=2.0e-5, abs=2.0e-5), (
            f"dense fused optimizer update was not visible on iteration {iteration}: "
            f"observed={observed_loss:.9f} expected={expected_loss:.9f}"
        )


@pytest.mark.cuda
def test_single_parameter_materialized_optimizer_async_first_update_is_stable_under_repetition():
    """Control: force materialized gradients while keeping the same queued one-batch training semantics."""
    target = 20.0
    initial_bias = -0.5
    lr = 0.01
    expected_loss = _expected_direct_bias_after_one_sgd_update(
        target=target,
        initial_bias=initial_bias,
        lr=lr,
    )

    for iteration in range(32):
        observed_loss = _fit_scalar_bias_once(
            f"scalar_bias_materialized_optimizer_async_stress_{iteration}",
            target=target,
            initial_bias=initial_bias,
            lr=lr,
            force_materialized_optimizer=True,
        )
        assert observed_loss == pytest.approx(expected_loss, rel=2.0e-5, abs=2.0e-5), (
            f"materialized optimizer update was not visible on iteration {iteration}: "
            f"observed={observed_loss:.9f} expected={expected_loss:.9f}"
        )


def _build_fc_no_bias_explicit_rtanh_head(name: str, *, initial_weight: float):
    """One-parameter real FullyConnected control: matmul/VJP without the bias parameter branch."""
    network = thor.Network(name)
    features = thor.layers.NetworkInput(network, "features", [1], thor.DataType.fp32).get_feature_output()
    base = thor.layers.NetworkInput(network, "base_log_rate", [1], thor.DataType.fp32).get_feature_output()
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32).get_feature_output()

    weight_init = thor.initializers.UniformRandom(initial_weight, initial_weight)
    raw = thor.layers.FullyConnected(
        network,
        features,
        1,
        False,
        activation=None,
        weights_initializer=weight_init,
        weights_data_type=thor.DataType.fp32,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.fp32,
    ).get_feature_output()
    residual = _bounded_explicit(network, raw)
    log_rate = _add_base_log_rate(network, base, residual)
    loss = thor.losses.distribution.PoissonNLLLoss(
        network,
        log_rate,
        labels,
        log_input=True,
        full=False,
        eps=1.0e-8,
        loss_data_type=thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    return network


def _build_two_parameter_affine_explicit_rtanh_head(
    name: str,
    *,
    initial_weight: float,
    initial_bias: float,
    force_materialized_optimizer: bool,
):
    """Generic x@W+b equivalent used to isolate multi-parameter optimizer fusion from FullyConnected."""
    network = thor.Network(name)
    features = thor.layers.NetworkInput(network, "features", [1], thor.DataType.fp32).get_feature_output()
    base = thor.layers.NetworkInput(network, "base_log_rate", [1], thor.DataType.fp32).get_feature_output()
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32).get_feature_output()

    inputs = {"features": features}
    if force_materialized_optimizer:
        dummy = thor.layers.NetworkInput(network, "dummy", [1], thor.DataType.fp32).get_feature_output()
        inputs["dummy"] = dummy

    weights = thor.parameters.ParameterSpecification(
        name="weights",
        shape=[1, 1],
        dtype=thor.DataType.fp32,
        initializer=thor.initializers.UniformRandom(initial_weight, initial_weight),
        trainable=True,
    )
    biases = thor.parameters.ParameterSpecification(
        name="biases",
        shape=[1],
        dtype=thor.DataType.fp32,
        initializer=thor.initializers.UniformRandom(initial_bias, initial_bias),
        trainable=True,
    )

    def build(context):
        raw = context.input("features") @ context.param("weights") + context.param("biases")
        if force_materialized_optimizer:
            # A second equation-bound feature input disables dense optimizer fusion while
            # leaving the affine math and both trainable parameters unchanged.
            raw = raw + context.input("dummy") * 0.0
        return {"raw": raw}

    raw = thor.layers.CustomLayer(
        network=network,
        inputs=inputs,
        output_names=["raw"],
        parameters=[weights, biases],
        build=build,
    )["raw"]
    residual = _bounded_explicit(network, raw)
    log_rate = _add_base_log_rate(network, base, residual)
    loss = thor.losses.distribution.PoissonNLLLoss(
        network,
        log_rate,
        labels,
        log_input=True,
        full=False,
        eps=1.0e-8,
        loss_data_type=thor.DataType.fp32,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    return network


def _fc_structure_training_data(*, target: float, feature_value: float, include_dummy: bool, rows: int = 4):
    tensors = {
        "features": np.full((rows, 1), feature_value, dtype=np.float32),
        "base_log_rate": np.full((rows, 1), math.log(10.0), dtype=np.float32),
        "labels": np.full((rows, 1), target, dtype=np.float32),
    }
    if include_dummy:
        tensors["dummy"] = np.zeros((rows, 1), dtype=np.float32)
    dataset = thor.data.NumpyDataset(tensors)
    indices = np.arange(rows, dtype=np.int64)
    return thor.data.TrainingData(
        dataset=dataset,
        splits=thor.data.DatasetSplitManifest(
            dataset=dataset,
            train_indices=indices,
            validate_indices=indices,
        ),
        batching=thor.data.BatchPolicy(batch_size=rows, randomize_train=False),
        dataset_name="fc_structure_first_update_diagnostic",
        device_storage="off",
    )


def _fit_fc_structure_once(network, *, target: float, feature_value: float, lr: float, include_dummy: bool):
    trainer = thor.training.Trainer(
        network,
        data=_fc_structure_training_data(
            target=target,
            feature_value=feature_value,
            include_dummy=include_dummy,
        ),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=lr, momentum=0.0),
        debug_synchronous=False,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=["loss"],
        stats_color="never",
    )
    result = trainer.fit(epochs=1, check_best_model_every_epochs=0)
    assert result.status == "completed"
    return float(result.final_validation_loss)


def _expected_explicit_rtanh_affine_after_one_sgd_update(
    *,
    target: float,
    initial_weight: float,
    initial_bias: float,
    feature_value: float,
    lr: float,
    train_weight: bool,
    train_bias: bool,
):
    raw = feature_value * initial_weight + initial_bias
    unit = math.tanh(raw / R)
    residual = R * unit
    log_rate = math.log(10.0) + residual
    rate = math.exp(log_rate)
    raw_gradient = (rate - target) * (1.0 - unit * unit)
    updated_weight = initial_weight - (lr * raw_gradient * feature_value if train_weight else 0.0)
    updated_bias = initial_bias - (lr * raw_gradient if train_bias else 0.0)
    updated_raw = feature_value * updated_weight + updated_bias
    updated_residual = R * math.tanh(updated_raw / R)
    updated_log_rate = math.log(10.0) + updated_residual
    return math.exp(updated_log_rate) - target * updated_log_rate


@pytest.mark.cuda
def test_real_fc_single_weight_async_first_update_is_stable_under_repetition():
    """A real FC matmul with only one trainable parameter should not need a multi-branch optimizer plan."""
    target = 20.0
    initial_weight = -0.5
    feature_value = 1.0
    lr = 0.01
    expected_loss = _expected_explicit_rtanh_affine_after_one_sgd_update(
        target=target,
        initial_weight=initial_weight,
        initial_bias=0.0,
        feature_value=feature_value,
        lr=lr,
        train_weight=True,
        train_bias=False,
    )

    for iteration in range(32):
        observed_loss = _fit_fc_structure_once(
            _build_fc_no_bias_explicit_rtanh_head(
                f"fc_single_weight_async_stress_{iteration}",
                initial_weight=initial_weight,
            ),
            target=target,
            feature_value=feature_value,
            lr=lr,
            include_dummy=False,
        )
        assert observed_loss == pytest.approx(expected_loss, rel=2.0e-5, abs=2.0e-5), (
            f"single-parameter real FC update was not visible on iteration {iteration}: "
            f"observed={observed_loss:.9f} expected={expected_loss:.9f}"
        )


@pytest.mark.cuda
def test_two_parameter_affine_fused_optimizer_async_first_update_is_stable_under_repetition():
    """Two-parameter matmul+bias CustomLayer exercises the branched dense fused optimizer plan."""
    target = 20.0
    initial_weight = 0.0
    initial_bias = -0.5
    feature_value = 0.0
    lr = 0.01
    expected_loss = _expected_explicit_rtanh_affine_after_one_sgd_update(
        target=target,
        initial_weight=initial_weight,
        initial_bias=initial_bias,
        feature_value=feature_value,
        lr=lr,
        train_weight=True,
        train_bias=True,
    )

    for iteration in range(64):
        observed_loss = _fit_fc_structure_once(
            _build_two_parameter_affine_explicit_rtanh_head(
                f"two_parameter_affine_fused_async_stress_{iteration}",
                initial_weight=initial_weight,
                initial_bias=initial_bias,
                force_materialized_optimizer=False,
            ),
            target=target,
            feature_value=feature_value,
            lr=lr,
            include_dummy=False,
        )
        assert observed_loss == pytest.approx(expected_loss, rel=2.0e-5, abs=2.0e-5), (
            f"two-parameter fused optimizer update was not visible on iteration {iteration}: "
            f"observed={observed_loss:.9f} expected={expected_loss:.9f}"
        )


@pytest.mark.cuda
def test_two_parameter_affine_materialized_optimizer_async_first_update_is_stable_under_repetition():
    """Same two-parameter affine graph, but force materialized gradients as the optimizer-fusion control."""
    target = 20.0
    initial_weight = 0.0
    initial_bias = -0.5
    feature_value = 0.0
    lr = 0.01
    expected_loss = _expected_explicit_rtanh_affine_after_one_sgd_update(
        target=target,
        initial_weight=initial_weight,
        initial_bias=initial_bias,
        feature_value=feature_value,
        lr=lr,
        train_weight=True,
        train_bias=True,
    )

    for iteration in range(64):
        observed_loss = _fit_fc_structure_once(
            _build_two_parameter_affine_explicit_rtanh_head(
                f"two_parameter_affine_materialized_async_stress_{iteration}",
                initial_weight=initial_weight,
                initial_bias=initial_bias,
                force_materialized_optimizer=True,
            ),
            target=target,
            feature_value=feature_value,
            lr=lr,
            include_dummy=True,
        )
        assert observed_loss == pytest.approx(expected_loss, rel=2.0e-5, abs=2.0e-5), (
            f"two-parameter materialized optimizer update was not visible on iteration {iteration}: "
            f"observed={observed_loss:.9f} expected={expected_loss:.9f}"
        )
