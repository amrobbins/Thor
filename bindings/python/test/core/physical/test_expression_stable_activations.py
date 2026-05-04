import numpy as np
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream, Placement, DeviceType, numpy_dtypes


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _copy_numpy_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu_placement = Placement(DeviceType.cpu, 0)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)

    host_desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
    host_tensor = PhysicalTensor(cpu_placement, host_desc)
    host_tensor.numpy()[...] = arr.astype(_numpy_storage_dtype(dtype), copy=False)

    gpu_tensor = PhysicalTensor(gpu_placement, host_desc)
    gpu_tensor.copy_from_async(host_tensor, stream)
    return gpu_tensor


def _copy_gpu_to_numpy(tensor: PhysicalTensor, stream: Stream) -> np.ndarray:
    cpu_placement = Placement(DeviceType.cpu, 0)
    host_tensor = PhysicalTensor(cpu_placement, tensor.get_descriptor())
    host_tensor.copy_from_async(tensor, stream)
    stream.synchronize()
    return host_tensor.numpy().copy()


def _run_expr(expr, inputs: dict[str, tuple[np.ndarray, thor.DataType]], gpu_num: int = 0) -> np.ndarray:
    stream = Stream(gpu_num=gpu_num)

    gpu_inputs = {}
    for name, (arr, dtype) in inputs.items():
        gpu_inputs[name] = _copy_numpy_to_gpu(arr, dtype, stream, gpu_num=gpu_num)

    eq = ex.compile(expr, device_num=gpu_num)
    stamped = eq.stamp(gpu_inputs, stream)
    stamped.run()

    return _copy_gpu_to_numpy(stamped.output(), stream)


def _stable_sigmoid_expr(x):
    # Equivalent to:
    #   x >= 0: 1 / (1 + exp(-x))
    #   x <  0: exp(x) / (1 + exp(x))
    # but expressed without a branch and without exp of a large positive value.
    return ex.exp(-ex.max(-x, 0.0)) / (1.0 + ex.exp(-ex.abs(x)))


def _stable_tanh_expr(x):
    return 2.0 * _stable_sigmoid_expr(2.0 * x) - 1.0


def _stable_softplus_expr(x):
    return ex.max(x, 0.0) + ex.ln(1.0 + ex.exp(-ex.abs(x)))


def _stable_elu_expr(x, alpha: float = 1.0):
    return ex.max(x, 0.0) + alpha * (ex.exp(ex.min(x, 0.0)) - 1.0)


def _stable_selu_expr(x):
    return 1.05070098 * ex.max(x, 0.0) + 1.758099326 * (ex.exp(ex.min(x, 0.0)) - 1.0)


def _stable_gelu_expr(x):
    inner = 0.797884561 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + _stable_tanh_expr(inner))


def _stable_swish_expr(x):
    return x * _stable_sigmoid_expr(x)


def _stable_sigmoid_np(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.maximum(-x, 0.0)) / (1.0 + np.exp(-np.abs(x)))


def _stable_tanh_np(x: np.ndarray) -> np.ndarray:
    return 2.0 * _stable_sigmoid_np(2.0 * x) - 1.0


def _stable_softplus_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _stable_elu_np(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.maximum(x, 0.0) + alpha * np.expm1(np.minimum(x, 0.0))


def _stable_selu_np(x: np.ndarray) -> np.ndarray:
    return 1.05070098 * np.maximum(x, 0.0) + 1.758099326 * np.expm1(np.minimum(x, 0.0))


def _stable_gelu_np(x: np.ndarray) -> np.ndarray:
    inner = 0.797884561 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + _stable_tanh_np(inner))


def _stable_swish_np(x: np.ndarray) -> np.ndarray:
    return x * _stable_sigmoid_np(x)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("name", "expr_builder", "np_builder"),
    [
        ("sigmoid", _stable_sigmoid_expr, _stable_sigmoid_np),
        ("tanh", _stable_tanh_expr, _stable_tanh_np),
        ("softplus", _stable_softplus_expr, _stable_softplus_np),
        ("elu", _stable_elu_expr, _stable_elu_np),
        ("selu", _stable_selu_expr, _stable_selu_np),
        ("gelu", _stable_gelu_expr, _stable_gelu_np),
        ("swish", _stable_swish_expr, _stable_swish_np),
    ],
)
def test_stable_activation_expressions_handle_extreme_fp32_inputs(name, expr_builder, np_builder):
    del name

    dtype = thor.DataType.fp32
    x_np = np.array(
        [-1000.0, -200.0, -100.0, -80.0, -20.0, -3.0, -1.0, 0.0, 1.0, 3.0, 20.0, 80.0, 100.0, 200.0, 1000.0],
        dtype=np.float32,
    )

    x = ex.input("x")
    got = _run_expr(expr_builder(x), {"x": (x_np, dtype)})
    expected = np_builder(x_np).astype(np.float32)

    assert np.isfinite(got).all()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_stable_softplus_does_not_overflow_where_naive_expansion_would():
    dtype = thor.DataType.fp32
    x_np = np.array([80.0, 100.0, 200.0, 1000.0], dtype=np.float32)

    x = ex.input("x")
    got = _run_expr(_stable_softplus_expr(x), {"x": (x_np, dtype)})
    expected = _stable_softplus_np(x_np).astype(np.float32)

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_stable_tanh_saturates_instead_of_overflowing_exp2x():
    dtype = thor.DataType.fp32
    x_np = np.array([-1000.0, -200.0, -100.0, 100.0, 200.0, 1000.0], dtype=np.float32)

    x = ex.input("x")
    got = _run_expr(_stable_tanh_expr(x), {"x": (x_np, dtype)})

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32), rtol=0.0, atol=0.0)


@pytest.mark.cuda
def test_stable_elu_and_selu_do_not_evaluate_positive_exp_branch():
    dtype = thor.DataType.fp32
    x_np = np.array([80.0, 100.0, 200.0, 1000.0], dtype=np.float32)

    x = ex.input("x")
    got_elu = _run_expr(_stable_elu_expr(x, alpha=1.25), {"x": (x_np, dtype)})
    got_selu = _run_expr(_stable_selu_expr(x), {"x": (x_np, dtype)})

    assert np.isfinite(got_elu).all()
    assert np.isfinite(got_selu).all()
    np.testing.assert_allclose(got_elu, x_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_selu, (1.05070098 * x_np).astype(np.float32), rtol=1e-5, atol=1e-6)
