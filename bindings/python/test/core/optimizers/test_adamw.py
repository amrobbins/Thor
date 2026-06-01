import json
import math

import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream


LOSS_SCALING_FACTOR = 4.0


def _net():
    return thor.Network("test_net_adamw")


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    return PhysicalTensor(
        Placement(DeviceType.cpu, 0),
        PhysicalTensor.Descriptor(dtype, shape),
    )


def _host_to_gpu(values: np.ndarray, stream: Stream) -> PhysicalTensor:
    host = _cpu_tensor(list(values.shape), thor.DataType.fp32)
    host.numpy()[...] = values.astype(np.float32)
    device = PhysicalTensor(
        Placement(DeviceType.gpu, 0),
        PhysicalTensor.Descriptor(thor.DataType.fp32, list(values.shape)),
    )
    device.copy_from_async(host, stream)
    return device


def _copy_to_numpy(tensor: PhysicalTensor, stream: Stream) -> np.ndarray:
    host = _cpu_tensor(list(tensor.dimensions), thor.DataType.fp32)
    host.copy_from_async(tensor, stream)
    stream.synchronize()
    return host.numpy().copy()


def _adamw_runtime_scalars(
    *,
    step: int,
    batch_size: int,
    alpha: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
) -> dict[str, float]:
    alpha_t = alpha * math.sqrt(1.0 - math.pow(beta2, float(step))) / (1.0 - math.pow(beta1, float(step)))
    return {
        "alphaT": alpha_t,
        "alphaWeightDecay": alpha * weight_decay,
        "invBatchLossScale": 1.0 / (float(batch_size) * LOSS_SCALING_FACTOR),
    }


def _adamw_reference_step(
    *,
    weights: np.ndarray,
    gradient: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    step: int,
    batch_size: int,
    alpha: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scalars = _adamw_runtime_scalars(
        step=step,
        batch_size=batch_size,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    )
    g = gradient.astype(np.float32) * scalars["invBatchLossScale"]
    m_next = beta1 * m.astype(np.float32) + (1.0 - beta1) * g
    v_next = beta2 * v.astype(np.float32) + (1.0 - beta2) * g * g
    weights_next = (
        weights.astype(np.float32)
        - scalars["alphaWeightDecay"] * weights.astype(np.float32)
        - scalars["alphaT"] * m_next / (np.sqrt(v_next) + epsilon)
    )
    return weights_next.astype(np.float32), m_next.astype(np.float32), v_next.astype(np.float32)


def _compile_adamw_step_equation(*, beta1: float, beta2: float, epsilon: float):
    weights = ex.input("weights_in", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32)
    gradient = ex.input("gradient", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32)
    m = ex.input("m_in", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32)
    v = ex.input("v_in", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32)

    alpha_t = ex.runtime_scalar("alphaT", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32)
    alpha_weight_decay = ex.runtime_scalar(
        "alphaWeightDecay",
        output_dtype=thor.DataType.fp32,
        compute_dtype=thor.DataType.fp32,
    )
    inv_batch_loss_scale = ex.runtime_scalar(
        "invBatchLossScale",
        output_dtype=thor.DataType.fp32,
        compute_dtype=thor.DataType.fp32,
    )

    g = gradient * inv_batch_loss_scale
    m_next = beta1 * m + (1.0 - beta1) * g
    v_next = beta2 * v + (1.0 - beta2) * g * g
    weights_next = weights - alpha_weight_decay * weights - alpha_t * m_next / (ex.sqrt(v_next) + epsilon)

    return ex.outputs({
        "weights": weights_next,
        "m": m_next,
        "v": v_next,
    }).compile(device_num=0, use_fast_math=False)


def _run_adamw_expression_step(
    equation,
    *,
    weights: PhysicalTensor,
    gradient: PhysicalTensor,
    m: PhysicalTensor,
    v: PhysicalTensor,
    stream: Stream,
    step: int,
    batch_size: int,
    alpha: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
):
    stamped = equation.stamp({
        "weights_in": weights,
        "gradient": gradient,
        "m_in": m,
        "v_in": v,
    }, stream)
    stamped.run(_adamw_runtime_scalars(
        step=step,
        batch_size=batch_size,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    ))
    return stamped.outputs()


def _only_fully_connected_layer(network: thor.Network):
    layers = [layer for layer in json.loads(network.get_architecture_json())["layers"] if layer["layer_type"] == "fully_connected"]
    assert len(layers) == 1
    return layers[0]


def test_adamw_constructs_defaults():
    n = _net()
    opt = thor.optimizers.AdamW(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.AdamW)


def test_adamw_constructs_defaults_without_network():
    opt = thor.optimizers.AdamW()
    assert opt is not None
    assert isinstance(opt, thor.optimizers.AdamW)


def test_adamw_constructs_custom_params():
    n = _net()
    opt = thor.optimizers.AdamW(
        network=n,
        alpha=1e-3,
        beta1=0.85,
        beta2=0.99,
        epsilon=1e-7,
        weight_decay=0.03,
    )
    assert isinstance(opt, thor.optimizers.AdamW)


def test_adamw_allows_zero_weight_decay():
    assert isinstance(thor.optimizers.AdamW(weight_decay=0.0), thor.optimizers.AdamW)


def test_adamw_rejects_non_positive_alpha():
    n = _net()
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.AdamW(network=n, alpha=0.0)
    with pytest.raises(ValueError, match=r"alpha must be > 0"):
        thor.optimizers.AdamW(network=n, alpha=-1.0)


def test_adamw_rejects_beta1_out_of_range():
    n = _net()
    with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
        thor.optimizers.AdamW(network=n, beta1=-0.01)
    with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
        thor.optimizers.AdamW(beta1=1.0)
    with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
        thor.optimizers.AdamW(network=n, beta1=1.01)


def test_adamw_rejects_beta2_out_of_range():
    n = _net()
    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.AdamW(beta2=-0.01)
    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.AdamW(network=n, beta2=1.0)
    with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
        thor.optimizers.AdamW(network=n, beta2=1.01)


def test_adamw_rejects_non_positive_epsilon():
    n = _net()
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.AdamW(network=n, epsilon=0.0)
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.optimizers.AdamW(network=n, epsilon=-1e-7)


def test_adamw_rejects_negative_weight_decay():
    n = _net()
    with pytest.raises(ValueError, match=r"weight_decay must be >= 0"):
        thor.optimizers.AdamW(network=n, weight_decay=-0.01)


def test_adamw_rejects_wrong_types():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.AdamW("not a network")

    with pytest.raises(TypeError):
        thor.optimizers.AdamW(network=n, alpha="1e-3")

    with pytest.raises(TypeError):
        thor.optimizers.AdamW(network=n, beta1="0.9")

    with pytest.raises(TypeError):
        thor.optimizers.AdamW(network=n, beta2="0.999")

    with pytest.raises(TypeError):
        thor.optimizers.AdamW(network=n, epsilon="1e-7")

    with pytest.raises(TypeError):
        thor.optimizers.AdamW(network=n, weight_decay="0.01")


def test_adamw_rejects_wrong_arity_and_kwargs():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.AdamW(1e-3, 0.9, 0.999, 1e-7, 0.01, 123, network=n)

    with pytest.raises(TypeError):
        thor.optimizers.AdamW(bogus=123, network=n)

    with pytest.raises(TypeError):
        thor.optimizers.AdamW(alpha=1e-3, extra=123, network=n)


def test_adamw_is_optimizer_subclass_if_exposed():
    Optimizer = getattr(thor.optimizers, "Optimizer", None)
    if Optimizer is None:
        pytest.skip("thor.optimizers.Optimizer not exposed in Python")
    n = _net()
    assert isinstance(thor.optimizers.AdamW(network=n), Optimizer)


def test_adamw_multiple_optimizers_on_same_network_throws():
    n = _net()
    opt = thor.optimizers.AdamW(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.AdamW)

    with pytest.raises(RuntimeError, match=r".*Multiple default optimizers.*"):
        thor.optimizers.AdamW(network=n)


def test_adamw_parameter_override_serializes_in_layer_architecture():
    n = thor.Network("test_net_adamw_parameter_override")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    weights_optimizer = thor.optimizers.AdamW(
        alpha=0.002,
        beta1=0.75,
        beta2=0.93,
        epsilon=1e-5,
        weight_decay=0.04,
    )
    biases_optimizer = thor.optimizers.AdamW(
        alpha=0.003,
        beta1=0.7,
        beta2=0.91,
        epsilon=1e-6,
        weight_decay=0.0,
    )

    thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=weights_optimizer,
        biases_optimizer=biases_optimizer,
    )

    fc_layer = _only_fully_connected_layer(n)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    biases_json = fc_layer["parameters"]["biases"]["optimizer_override"]

    assert weights_json["optimizer_type"] == "adamw"
    assert weights_json["alpha"] == pytest.approx(0.002)
    assert weights_json["beta1"] == pytest.approx(0.75)
    assert weights_json["beta2"] == pytest.approx(0.93)
    assert weights_json["epsilon"] == pytest.approx(1e-5)
    assert weights_json["weight_decay"] == pytest.approx(0.04)

    assert biases_json["optimizer_type"] == "adamw"
    assert biases_json["alpha"] == pytest.approx(0.003)
    assert biases_json["beta1"] == pytest.approx(0.7)
    assert biases_json["beta2"] == pytest.approx(0.91)
    assert biases_json["epsilon"] == pytest.approx(1e-6)
    assert biases_json["weight_decay"] == pytest.approx(0.0)


def test_adamw_network_save_load_preserves_parameter_override(tmp_path):
    n = thor.Network("test_net_adamw_save_load")
    input_layer = thor.layers.NetworkInput(n, "input", [4], thor.DataType.fp32)
    fc = thor.layers.FullyConnected(
        n,
        input_layer.get_feature_output(),
        3,
        True,
        activation=None,
        weights_optimizer=thor.optimizers.AdamW(
            alpha=0.002,
            beta1=0.75,
            beta2=0.93,
            epsilon=1e-5,
            weight_decay=0.04,
        ),
    )
    thor.layers.NetworkOutput(n, "output", fc.get_feature_output(), thor.DataType.fp32)

    save_dir = tmp_path / "adamw_network"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network("test_net_adamw_save_load")
    loaded.load(str(save_dir))

    fc_layer = _only_fully_connected_layer(loaded)
    weights_json = fc_layer["parameters"]["weights"]["optimizer_override"]
    assert weights_json["optimizer_type"] == "adamw"
    assert weights_json["alpha"] == pytest.approx(0.002)
    assert weights_json["beta1"] == pytest.approx(0.75)
    assert weights_json["beta2"] == pytest.approx(0.93)
    assert weights_json["epsilon"] == pytest.approx(1e-5)
    assert weights_json["weight_decay"] == pytest.approx(0.04)


@pytest.mark.cuda
def test_adamw_dense_step_formula_matches_numpy_reference_for_two_steps():
    alpha = 0.01
    beta1 = 0.8
    beta2 = 0.95
    epsilon = 1e-4
    weight_decay = 0.03

    weights_np = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    m_np = np.zeros_like(weights_np)
    v_np = np.zeros_like(weights_np)
    gradient1_np = np.array([[0.5, -1.25, 2.0], [1.5, -0.75, 0.25]], dtype=np.float32)
    gradient2_np = np.array([[-1.0, 0.5, -0.25], [2.0, -1.5, 0.75]], dtype=np.float32)

    expected_weights, expected_m, expected_v = _adamw_reference_step(
        weights=weights_np,
        gradient=gradient1_np,
        m=m_np,
        v=v_np,
        step=1,
        batch_size=1,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
    )
    expected_weights, expected_m, expected_v = _adamw_reference_step(
        weights=expected_weights,
        gradient=gradient2_np,
        m=expected_m,
        v=expected_v,
        step=2,
        batch_size=2,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
    )

    stream = Stream(gpu_num=0)
    equation = _compile_adamw_step_equation(beta1=beta1, beta2=beta2, epsilon=epsilon)

    weights_gpu = _host_to_gpu(weights_np, stream)
    m_gpu = _host_to_gpu(m_np, stream)
    v_gpu = _host_to_gpu(v_np, stream)
    gradient1_gpu = _host_to_gpu(gradient1_np, stream)
    gradient2_gpu = _host_to_gpu(gradient2_np, stream)

    step1_outputs = _run_adamw_expression_step(
        equation,
        weights=weights_gpu,
        gradient=gradient1_gpu,
        m=m_gpu,
        v=v_gpu,
        stream=stream,
        step=1,
        batch_size=1,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    )
    step2_outputs = _run_adamw_expression_step(
        equation,
        weights=step1_outputs["weights"],
        gradient=gradient2_gpu,
        m=step1_outputs["m"],
        v=step1_outputs["v"],
        stream=stream,
        step=2,
        batch_size=2,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    )

    got_weights = _copy_to_numpy(step2_outputs["weights"], stream)
    got_m = _copy_to_numpy(step2_outputs["m"], stream)
    got_v = _copy_to_numpy(step2_outputs["v"], stream)

    np.testing.assert_allclose(got_weights, expected_weights, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_m, expected_m, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_v, expected_v, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_adamw_weight_decay_is_decoupled_from_adaptive_gradient_term_numerical():
    alpha = 0.02
    beta1 = 0.7
    beta2 = 0.9
    epsilon = 1e-5
    weight_decay = 0.2

    weights_np = np.array([2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    m_np = np.zeros_like(weights_np)
    v_np = np.zeros_like(weights_np)
    zero_gradient_np = np.zeros_like(weights_np)

    expected_weights, expected_m, expected_v = _adamw_reference_step(
        weights=weights_np,
        gradient=zero_gradient_np,
        m=m_np,
        v=v_np,
        step=1,
        batch_size=4,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
    )

    stream = Stream(gpu_num=0)
    equation = _compile_adamw_step_equation(beta1=beta1, beta2=beta2, epsilon=epsilon)

    outputs = _run_adamw_expression_step(
        equation,
        weights=_host_to_gpu(weights_np, stream),
        gradient=_host_to_gpu(zero_gradient_np, stream),
        m=_host_to_gpu(m_np, stream),
        v=_host_to_gpu(v_np, stream),
        stream=stream,
        step=1,
        batch_size=4,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    )

    got_weights = _copy_to_numpy(outputs["weights"], stream)
    got_m = _copy_to_numpy(outputs["m"], stream)
    got_v = _copy_to_numpy(outputs["v"], stream)

    np.testing.assert_allclose(got_weights, expected_weights, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_weights, weights_np * (1.0 - alpha * weight_decay), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_m, expected_m, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_v, expected_v, rtol=1e-6, atol=1e-6)
