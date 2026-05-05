import math

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
    return ex.tanh(x)


def _stable_softplus_expr(x):
    return ex.max(x, 0.0) + ex.log1p(ex.exp(-ex.abs(x)))


def _stable_elu_expr(x, alpha: float = 1.0):
    return ex.max(x, 0.0) + alpha * ex.expm1(ex.min(x, 0.0))


def _stable_selu_expr(x):
    return 1.05070098 * ex.max(x, 0.0) + 1.758099326 * ex.expm1(ex.min(x, 0.0))


def _stable_gelu_expr(x):
    return x * ex.normcdf(x)


def _stable_swish_expr(x):
    return x * _stable_sigmoid_expr(x)


def _stable_sigmoid_np(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.maximum(-x, 0.0)) / (1.0 + np.exp(-np.abs(x)))


def _stable_tanh_np(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _stable_softplus_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _stable_elu_np(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.maximum(x, 0.0) + alpha * np.expm1(np.minimum(x, 0.0))


def _stable_selu_np(x: np.ndarray) -> np.ndarray:
    return 1.05070098 * np.maximum(x, 0.0) + 1.758099326 * np.expm1(np.minimum(x, 0.0))


def _normcdf_np(x: np.ndarray) -> np.ndarray:
    # Avoid scipy dependency; math.erfc is scalar, so vectorize it for the small numerical test arrays.
    erfc = np.vectorize(math.erfc, otypes=[np.float64])
    return (0.5 * erfc((-x.astype(np.float64)) / math.sqrt(2.0))).astype(np.float32)


def _stable_gelu_np(x: np.ndarray) -> np.ndarray:
    return x * _normcdf_np(x)


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
    got = _run_expr(expr_builder(x), {
        "x": (x_np, dtype)
    })
    expected = np_builder(x_np).astype(np.float32)

    assert np.isfinite(got).all()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_stable_softplus_does_not_overflow_where_naive_expansion_would():
    dtype = thor.DataType.fp32
    x_np = np.array([80.0, 100.0, 200.0, 1000.0], dtype=np.float32)

    x = ex.input("x")
    got = _run_expr(_stable_softplus_expr(x), {
        "x": (x_np, dtype)
    })
    expected = _stable_softplus_np(x_np).astype(np.float32)

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_tanh_expression_op_saturates_instead_of_overflowing_exp2x():
    dtype = thor.DataType.fp32
    x_np = np.array([-1000.0, -200.0, -100.0, 100.0, 200.0, 1000.0], dtype=np.float32)

    x = ex.input("x")
    got = _run_expr(ex.tanh(x), {
        "x": (x_np, dtype)
    })

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32), rtol=0.0, atol=0.0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (thor.DataType.fp32, 1e-5, 1e-6),
        (thor.DataType.fp16, 3e-3, 3e-3),
        (thor.DataType.bf16, 8e-3, 8e-3),
        (thor.DataType.fp8_e4m3, 2.5e-1, 2.5e-1),
        (thor.DataType.fp8_e5m2, 3.5e-1, 3.5e-1),
    ],
)
def test_tanh_expression_op_matches_numpy_for_supported_storage_dtypes(dtype, rtol, atol):
    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.linspace(-4.0, 4.0, num=17, dtype=np.float32).astype(storage_dtype)

    x = ex.input("x")
    got = _run_expr(ex.tanh(x), {
        "x": (x_np, dtype)
    })
    expected = np.tanh(x_np.astype(np.float32)).astype(storage_dtype)

    assert got.dtype == storage_dtype
    assert got.shape == expected.shape
    assert np.isfinite(got.astype(np.float32)).all()
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


@pytest.mark.cuda
def test_log1p_expression_op_preserves_small_exp_softplus_term():
    dtype = thor.DataType.fp32
    x_np = np.array([-1000.0, -200.0, -100.0, -20.0, -1.0, 0.0], dtype=np.float32)

    x = ex.input("x")
    got = _run_expr(ex.log1p(ex.exp(x)), {
        "x": (x_np, dtype)
    })
    expected = np.log1p(np.exp(x_np)).astype(np.float32)

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.cuda
def test_expm1_expression_op_preserves_values_near_zero():
    dtype = thor.DataType.fp32
    x_np = np.array([-1.0e-4, -1.0e-6, 0.0, 1.0e-6, 1.0e-4], dtype=np.float32)

    x = ex.input("x")
    got = _run_expr(ex.expm1(x), {
        "x": (x_np, dtype)
    })
    expected = np.expm1(x_np).astype(np.float32)

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-9)


@pytest.mark.cuda
def test_normcdf_expression_op_matches_reference_and_saturates():
    dtype = thor.DataType.fp32
    x_np = np.array([-1000.0, -20.0, -8.0, -1.0, 0.0, 1.0, 8.0, 20.0, 1000.0], dtype=np.float32)

    x = ex.input("x")
    got = _run_expr(ex.normcdf(x), {
        "x": (x_np, dtype)
    })
    expected = _normcdf_np(x_np).astype(np.float32)

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.cuda
def test_exact_gelu_uses_normcdf_expression_op():
    dtype = thor.DataType.fp32
    x_np = np.array([-20.0, -8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0, 20.0], dtype=np.float32)

    x = ex.input("x")
    got = _run_expr(_stable_gelu_expr(x), {
        "x": (x_np, dtype)
    })
    expected = _stable_gelu_np(x_np).astype(np.float32)

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_stable_elu_and_selu_do_not_evaluate_positive_exp_branch():
    dtype = thor.DataType.fp32
    x_np = np.array([80.0, 100.0, 200.0, 1000.0], dtype=np.float32)

    x = ex.input("x")
    got_elu = _run_expr(_stable_elu_expr(x, alpha=1.25), {
        "x": (x_np, dtype)
    })
    got_selu = _run_expr(_stable_selu_expr(x), {
        "x": (x_np, dtype)
    })

    assert np.isfinite(got_elu).all()
    assert np.isfinite(got_selu).all()
    np.testing.assert_allclose(got_elu, x_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_selu, (1.05070098 * x_np).astype(np.float32), rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (thor.DataType.fp16, 4e-3, 4e-3),
        (thor.DataType.bf16, 1e-2, 1e-2),
        (thor.DataType.fp8_e4m3, 2.5e-1, 2.5e-1),
        (thor.DataType.fp8_e5m2, 3.5e-1, 3.5e-1),
    ],
)
@pytest.mark.parametrize(
    ("name", "expr_builder", "np_builder", "values"),
    [
        ("expm1", lambda x: ex.expm1(x), np.expm1, [-5.0, -2.0, -1.0, -0.25, -0.03125, 0.0, 0.03125, 0.25, 1.0, 2.0]),
        ("log1p", lambda x: ex.log1p(x), np.log1p, [-0.75, -0.5, -0.25, -0.03125, 0.0, 0.03125, 0.25, 1.0, 4.0, 8.0]),
        ("normcdf", lambda x: ex.normcdf(x), _normcdf_np, [-6.0, -3.0, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 6.0]),
        ("tanh", lambda x: ex.tanh(x), np.tanh, [-6.0, -3.0, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 6.0]),
    ],
)
def test_special_function_expression_ops_match_numpy_for_low_precision_storage_dtypes(
        dtype, rtol, atol, name, expr_builder, np_builder, values):
    del name

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(values, dtype=np.float32).astype(storage_dtype)

    x = ex.input("x")
    got = _run_expr(expr_builder(x), {
        "x": (x_np, dtype)
    })
    expected = np_builder(x_np.astype(np.float32)).astype(storage_dtype)

    assert got.dtype == storage_dtype
    assert got.shape == expected.shape
    assert np.isfinite(got.astype(np.float32)).all()
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


def _softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def _log_softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


@pytest.mark.cuda
def test_cudnn_softmax_expression_matches_stable_numpy_for_extreme_fp32_inputs():
    dtype = thor.DataType.fp32
    x_np = np.array(
        [
            [1000.0, 999.0, 998.0, -1000.0],
            [-1000.0, -999.0, -998.0, 1000.0],
            [0.0, 1.0, -1.0, 2.0],
        ],
        dtype=np.float32,
    )

    x = ex.input("x")
    got = _run_expr(ex.softmax(x), {
        "x": (x_np, dtype)
    })
    expected = _softmax_np(x_np).astype(np.float32)

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_cudnn_log_softmax_expression_matches_stable_numpy_for_extreme_fp32_inputs():
    dtype = thor.DataType.fp32
    x_np = np.array(
        [
            [1000.0, 999.0, 998.0, -1000.0],
            [-1000.0, -999.0, -998.0, 1000.0],
            [0.0, 1.0, -1.0, 2.0],
        ],
        dtype=np.float32,
    )

    x = ex.input("x")
    got = _run_expr(ex.log_softmax(x), {
        "x": (x_np, dtype)
    })
    expected = _log_softmax_np(x_np).astype(np.float32)

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
