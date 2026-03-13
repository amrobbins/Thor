import math
import numpy as np
import ml_dtypes
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream, Placement, DeviceType

FLOAT_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
    thor.DataType.fp32,
]


def _numpy_storage_dtype(dtype: thor.DataType):
    if dtype == thor.DataType.fp32:
        return np.float32
    if dtype == thor.DataType.fp16:
        return np.float16
    if dtype == thor.DataType.bf16:
        return ml_dtypes.bfloat16
    if dtype == thor.DataType.fp8_e4m3:
        return ml_dtypes.float8_e4m3fn
    if dtype == thor.DataType.fp8_e5m2:
        return ml_dtypes.float8_e5m2
    raise AssertionError(f"Unhandled dtype: {dtype}")


def _rtol_atol(dtype: thor.DataType) -> tuple[float, float]:
    if dtype == thor.DataType.fp32:
        return 3e-5, 3e-6
    if dtype == thor.DataType.fp16:
        return 5e-3, 5e-3
    if dtype == thor.DataType.bf16:
        return 2e-2, 2e-2
    if dtype == thor.DataType.fp8_e4m3:
        return 1.5e-1, 1.5e-1
    if dtype == thor.DataType.fp8_e5m2:
        return 2.5e-1, 2.5e-1
    raise AssertionError(f"Unhandled dtype: {dtype}")


def _compute_dtype_for_reference(dtype: thor.DataType):
    if dtype == thor.DataType.fp8_e4m3:
        return ml_dtypes.bfloat16
    if dtype == thor.DataType.fp8_e5m2:
        return ml_dtypes.bfloat16
    return _numpy_storage_dtype(dtype)


def _cpu_tensor(shape: list[int]) -> thor.physical.PhysicalTensor:
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(thor.DataType.fp32, shape)
    return thor.physical.PhysicalTensor(placement, descriptor)


def _gpu_tensor(shape: list[int], gpu_num: int = 0) -> thor.physical.PhysicalTensor:
    placement = thor.physical.Placement(thor.physical.DeviceType.gpu, gpu_num)
    descriptor = thor.physical.PhysicalTensor.Descriptor(thor.DataType.fp32, shape)
    return thor.physical.PhysicalTensor(placement, descriptor)


def _cpu_numpy_view(t: thor.physical.PhysicalTensor) -> np.ndarray:
    arr = t.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    return arr


def _copy_numpy_to_gpu(values: np.ndarray, gpu_num: int = 0) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=np.float32, order="C")
    cpu = _cpu_tensor(list(values.shape))
    gpu = _gpu_tensor(list(values.shape), gpu_num=gpu_num)
    stream = thor.physical.Stream(gpu_num)

    cpu_view = _cpu_numpy_view(cpu)
    cpu_view[...] = values
    gpu.copy_from_async(cpu, stream)
    return gpu


def _copy_gpu_to_numpy(t: thor.physical.PhysicalTensor, stream) -> np.ndarray:
    cpu = _cpu_tensor(list(t.get_descriptor().get_dimensions()))
    cpu.copy_from_async(t, stream)
    stream.synchronize()
    return np.array(_cpu_numpy_view(cpu), copy=True)


def _run_expr(expr, *inputs: np.ndarray, gpu_num: int = 0, use_fast_math: bool = False) -> np.ndarray:
    assert len(inputs) >= 1
    first_shape = tuple(inputs[0].shape)
    for arr in inputs:
        assert arr.dtype == np.float32
        assert tuple(arr.shape) == first_shape

    eq = ex.compile(
        expr,
        dtype=thor.DataType.fp32,
        device_num=gpu_num,
        use_fast_math=use_fast_math,
    )

    gpu_inputs = [_copy_numpy_to_gpu(arr, gpu_num=gpu_num) for arr in inputs]
    gpu_output = _gpu_tensor(list(first_shape), gpu_num=gpu_num)
    stream = thor.physical.Stream(gpu_num)

    eq.run(gpu_inputs, gpu_output, stream)
    return _copy_gpu_to_numpy(gpu_output, stream)


def _assert_close(got: np.ndarray, expected: np.ndarray, *, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(got, expected.astype(np.float32), rtol=rtol, atol=atol)


@pytest.mark.cuda
def test_add_sub_mul_div_fp32_numerical():
    x = ex.input(0)
    y = ex.input(1)
    expr = ((x + y) - 2.0) * (x / (y + 1.0))

    x_np = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    y_np = np.array([2, 3, 4, 5, 6, 7], dtype=np.float32)

    expected = ((x_np + y_np) - 2.0) * (x_np / (y_np + 1.0))
    got = _run_expr(expr, x_np, y_np)

    _assert_close(got, expected)


@pytest.mark.cuda
def test_pow_fp32_numerical():
    x = ex.input(0)
    y = ex.input(1)
    expr = x**y

    x_np = np.array([1.5, 2.0, 3.0, 4.0, 5.5], dtype=np.float32)
    y_np = np.array([2.0, 3.0, 1.5, 0.5, 2.0], dtype=np.float32)

    expected = np.power(x_np, y_np)
    got = _run_expr(expr, x_np, y_np)

    _assert_close(got, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
def test_rpow_fp32_numerical():
    x = ex.input(0)
    expr = 2.0**x

    x_np = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    expected = np.power(2.0, x_np)
    got = _run_expr(expr, x_np)

    _assert_close(got, expected)


@pytest.mark.cuda
def test_negation_fp32_numerical():
    x = ex.input(0)
    y = ex.input(1)
    expr = -(x**y) + 3.0

    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_np = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)

    expected = -(np.power(x_np, y_np)) + 3.0
    got = _run_expr(expr, x_np, y_np)

    _assert_close(got, expected)


@pytest.mark.cuda
def test_min_max_fp32_numerical():
    x = ex.input(0)
    y = ex.input(1)
    expr = ex.max(ex.min(x, y), 3.0)

    x_np = np.array([1, 5, 2, 8, 4, 6], dtype=np.float32)
    y_np = np.array([2, 4, 7, 1, 3, 9], dtype=np.float32)

    expected = np.maximum(np.minimum(x_np, y_np), 3.0)
    got = _run_expr(expr, x_np, y_np)

    _assert_close(got, expected)


@pytest.mark.cuda
def test_exp_family_fp32_numerical():
    x = ex.input(0)

    x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    got_exp = _run_expr(ex.exp(x), x_np)
    got_exp2 = _run_expr(ex.exp2(x), x_np)
    got_exp10 = _run_expr(ex.exp10(x), x_np)

    _assert_close(got_exp, np.exp(x_np), rtol=2e-5, atol=2e-6)
    _assert_close(got_exp2, np.exp2(x_np), rtol=2e-5, atol=2e-6)
    _assert_close(got_exp10, np.power(10.0, x_np), rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
def test_log_family_fp32_numerical():
    x = ex.input(0)

    x_np = np.array([0.25, 0.5, 1.0, 2.0, 8.0, 10.0], dtype=np.float32)

    got_ln = _run_expr(ex.ln(x), x_np)
    got_log_default = _run_expr(ex.log(x), x_np)
    got_log2 = _run_expr(ex.log2(x), x_np)
    got_log10 = _run_expr(ex.log10(x), x_np)
    got_log_base7 = _run_expr(ex.log(x, 7.0), x_np)

    _assert_close(got_ln, np.log(x_np), rtol=2e-5, atol=2e-6)
    _assert_close(got_log_default, np.log(x_np), rtol=2e-5, atol=2e-6)
    _assert_close(got_log2, np.log2(x_np), rtol=2e-5, atol=2e-6)
    _assert_close(got_log10, np.log10(x_np), rtol=2e-5, atol=2e-6)
    _assert_close(got_log_base7, np.log(x_np) / math.log(7.0), rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
def test_sqrt_fp32_numerical():
    x = ex.input(0)
    expr = ex.sqrt(x)

    x_np = np.array([0.0, 0.25, 1.0, 4.0, 9.0, 16.0], dtype=np.float32)

    expected = np.sqrt(x_np)
    got = _run_expr(expr, x_np)

    _assert_close(got, expected)


@pytest.mark.cuda
def test_scalar_only_expression_fp32_numerical():
    expr = ex.exp(ex.scalar(2.0)) + ex.log2(ex.scalar(8.0)) - ex.sqrt(ex.scalar(9.0))

    # Need a tensor shape for execution. Use a dummy input expression plus a zero multiplier
    # so the result stays scalar-valued elementwise.
    x = ex.input(0)
    lifted_expr = (x * 0.0) + expr

    x_np = np.zeros((6,), dtype=np.float32)
    expected_value = np.exp(np.float32(2.0)) + np.log2(np.float32(8.0)) - np.sqrt(np.float32(9.0))
    expected = np.full_like(x_np, expected_value, dtype=np.float32)

    got = _run_expr(lifted_expr, x_np)
    _assert_close(got, expected)


@pytest.mark.cuda
def test_nested_expression_fp32_numerical():
    x = ex.input(0)
    y = ex.input(1)
    z = ex.input(2)

    expr = ex.max(
        ex.sqrt(ex.exp2((x + 3.0) * (y - 1.0))),
        ex.min((z / 2.0)**2.0, ex.log2(y + 8.0)),
    )

    x_np = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    y_np = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    z_np = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

    expected = np.maximum(
        np.sqrt(np.exp2((x_np + 3.0) * (y_np - 1.0))),
        np.minimum(np.power(z_np / 2.0, 2.0), np.log2(y_np + 8.0)),
    )
    got = _run_expr(expr, x_np, y_np, z_np)

    _assert_close(got, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
def test_nested_expression_fp32_numerical_stamped():
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)
    cpu_placement = Placement(DeviceType.cpu)
    descriptor = PhysicalTensor.Descriptor(thor.DataType.fp32, dimensions=[2, 4])

    # Allocate Tensors
    x_host = PhysicalTensor(cpu_placement, descriptor)
    y_host = PhysicalTensor(cpu_placement, descriptor)
    z_host = PhysicalTensor(cpu_placement, descriptor)
    x_gpu = PhysicalTensor(gpu_placement, descriptor)
    y_gpu = PhysicalTensor(gpu_placement, descriptor)
    z_gpu = PhysicalTensor(gpu_placement, descriptor)

    # Initialize cpu tensors, then copy to gpu tensors
    x_np_view = x_host.numpy()
    y_np_view = y_host.numpy()
    z_np_view = z_host.numpy()
    x_np_view[:] = [[0.5, 1.0, 1.5, 2.0], [0.8, 0.7, 2.5, 4.1]]
    y_np_view[:] = [[2.0, 3.0, 4.0, 5.0], [8.0, 2.1, 0.6, 2.0]]
    z_np_view[:] = [[2.0, 4.0, 6.0, 8.0], [1.0, 2.5, 3.2, 3.3]]
    x_gpu.copy_from_async(x_host, stream)
    y_gpu.copy_from_async(y_host, stream)
    z_gpu.copy_from_async(z_host, stream)

    # thor expression
    x = ex.input(0)
    y = ex.input(1)
    z = ex.input(2)
    expr = ex.max(
        ex.sqrt(ex.exp2((x + 3.0) * (y - 1.0))),
        ex.min((z / 2.0)**2.0, ex.log2(y + 8.0)),
    )

    # Corresponding numpy expression
    expected = np.maximum(
        np.sqrt(np.exp2((x_np_view + 3.0) * (y_np_view - 1.0))),
        np.minimum(np.power(z_np_view / 2.0, 2.0), np.log2(y_np_view + 8.0)),
    )

    fused_equation = ex.compile(
        expr,
        dtype=thor.DataType.fp32,
        device_num=gpu_num,
        use_fast_math=False,
    )

    stamped_equation = fused_equation.stamp([x_gpu, y_gpu, z_gpu], stream)

    stamped_equation.run()

    out_gpu = stamped_equation.output_tensor
    out_cpu = out_gpu.clone(cpu_placement)  # allocate matching CPU tensor for copy-back
    out_cpu.copy_from_async(out_gpu, stream)
    stream.synchronize()

    np.testing.assert_allclose(out_cpu.numpy(), expected.astype(np.float32), rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@pytest.mark.parametrize("use_fast_math", [False, True])
def test_fast_math_toggle_fp32_numerical(use_fast_math: bool):
    x = ex.input(0)
    y = ex.input(1)
    expr = ex.exp(ex.log2(x + 8.0) + (y**2.0) / 3.0)

    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_np = np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float32)

    got = _run_expr(expr, x_np, y_np, use_fast_math=use_fast_math)
    expected = np.exp(np.log2(x_np + 8.0) + np.power(y_np, 2.0) / 3.0)

    _assert_close(
        got,
        expected,
        rtol=8e-4 if use_fast_math else 3e-5,
        atol=5e-5 if use_fast_math else 3e-6,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nested_expression_numerical_stamped(dtype: thor.DataType):
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)
    cpu_placement = Placement(DeviceType.cpu)
    descriptor = PhysicalTensor.Descriptor(dtype, dimensions=[2, 4])

    x_host = PhysicalTensor(cpu_placement, descriptor)
    y_host = PhysicalTensor(cpu_placement, descriptor)
    z_host = PhysicalTensor(cpu_placement, descriptor)
    x_gpu = PhysicalTensor(gpu_placement, descriptor)
    y_gpu = PhysicalTensor(gpu_placement, descriptor)
    z_gpu = PhysicalTensor(gpu_placement, descriptor)

    x_np_view = x_host.numpy()
    y_np_view = y_host.numpy()
    z_np_view = z_host.numpy()

    x_init = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
    y_init = np.array([[1.1, 1.2, 1.3, 1.4], [1.5, 1.2, 1.1, 1.3]], dtype=np.float32)
    z_init = np.array([[1.0, 1.5, 2.0, 2.5], [0.8, 1.2, 1.6, 2.0]], dtype=np.float32)

    host_np_dtype = _numpy_storage_dtype(dtype)
    x_np_view[:] = x_init.astype(host_np_dtype)
    y_np_view[:] = y_init.astype(host_np_dtype)
    z_np_view[:] = z_init.astype(host_np_dtype)

    x_gpu.copy_from_async(x_host, stream)
    y_gpu.copy_from_async(y_host, stream)
    z_gpu.copy_from_async(z_host, stream)

    x = ex.input(0)
    y = ex.input(1)
    z = ex.input(2)
    expr = ex.max(
        ex.sqrt(ex.exp2((x + 3.0) * (y - 1.0))),
        ex.min((z / 2.0)**2.0, ex.log2(y + 8.0)),
    )

    compute_np_dtype = _compute_dtype_for_reference(dtype)
    x_ref = x_np_view.astype(compute_np_dtype)
    y_ref = y_np_view.astype(compute_np_dtype)
    z_ref = z_np_view.astype(compute_np_dtype)

    expected = np.maximum(
        np.sqrt(np.exp2((x_ref + 3.0) * (y_ref - 1.0))),
        np.minimum((z_ref / 2.0)**2.0, np.log2(y_ref + 8.0)),
    )

    # Mirror the kernel's final store dtype.
    expected = expected.astype(_numpy_storage_dtype(dtype))

    fused_equation = ex.compile(
        expr,
        dtype=dtype,
        device_num=gpu_num,
        use_fast_math=False,
    )

    stamped_equation = fused_equation.stamp([x_gpu, y_gpu, z_gpu], stream)
    stamped_equation.run()

    out_gpu = stamped_equation.output_tensor
    out_cpu = out_gpu.clone(cpu_placement)
    out_cpu.copy_from_async(out_gpu, stream)
    stream.synchronize()

    rtol, atol = _rtol_atol(dtype)
    np.testing.assert_allclose(out_cpu.numpy(), expected, rtol=rtol, atol=atol)
