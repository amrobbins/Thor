import numpy as np
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream, Placement, DeviceType

FLOAT_DTYPES = [
    thor.DataType.fp16,
    # thor.DataType.bf16,
    # thor.DataType.fp8_e4m3,
    # thor.DataType.fp8_e5m2,
    # thor.DataType.fp32,
]


def _numpy_storage_dtype(dtype: thor.DataType):
    return thor.physical.numpy_dtypes.from_thor(dtype)


def _numpy_compute_dtype(dtype: thor.DataType):
    if dtype == thor.DataType.fp8_e4m3 or dtype == thor.DataType.fp8_e5m2:
        return thor.physical.numpy_dtypes.bf16
    return _numpy_storage_dtype(dtype)


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


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, shape)
    return thor.physical.PhysicalTensor(placement, descriptor)


def _gpu_tensor(shape: list[int], dtype: thor.DataType, gpu_num: int = 0) -> thor.physical.PhysicalTensor:
    placement = thor.physical.Placement(thor.physical.DeviceType.gpu, gpu_num)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, shape)
    return thor.physical.PhysicalTensor(placement, descriptor)


def _cpu_numpy_view(t: thor.physical.PhysicalTensor, dtype: thor.DataType) -> np.ndarray:
    arr = t.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == _numpy_storage_dtype(dtype)
    return arr


def _copy_numpy_to_gpu(values: np.ndarray, stream, dtype: thor.DataType, gpu_num=0) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=_numpy_storage_dtype(dtype), order="C")
    cpu = _cpu_tensor(list(values.shape), dtype)
    gpu = _gpu_tensor(list(values.shape), dtype, gpu_num=gpu_num)

    cpu_view = _cpu_numpy_view(cpu, dtype)
    cpu_view[...] = values
    gpu.copy_from_async(cpu, stream)
    return gpu


def _copy_gpu_to_numpy(t: thor.physical.PhysicalTensor, stream, dtype: thor.DataType) -> np.ndarray:
    cpu = _cpu_tensor(list(t.get_descriptor().get_dimensions()), dtype)
    cpu.copy_from_async(t, stream)
    stream.synchronize()
    return np.array(_cpu_numpy_view(cpu, dtype), copy=True)


def _run_expr_broadcast(
    expr,
    input_names: list[str],
    *inputs: np.ndarray,
    dtype: thor.DataType,
    gpu_num: int = 0,
    use_fast_math: bool = False,
) -> np.ndarray:
    assert len(inputs) >= 1
    inputs = [np.asarray(arr, dtype=_numpy_storage_dtype(dtype), order="C") for arr in inputs]

    expected_shape = np.broadcast_shapes(*[arr.shape for arr in inputs])

    eq = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=use_fast_math,
    )

    stream = thor.physical.Stream(gpu_num)
    gpu_inputs = {
        input_names[i]: _copy_numpy_to_gpu(inputs[i], stream, dtype, gpu_num=gpu_num) for i in range(len(inputs))
    }
    gpu_output = _gpu_tensor(list(expected_shape), dtype, gpu_num=gpu_num)

    eq.run(gpu_inputs, gpu_output, stream)
    return _copy_gpu_to_numpy(gpu_output, stream, dtype)


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType, *, rtol=None, atol=None):
    default_rtol, default_atol = _rtol_atol(dtype)
    if rtol is None:
        rtol = default_rtol
    if atol is None:
        atol = default_atol

    expected = expected.astype(_numpy_storage_dtype(dtype))
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_same_rank_singleton_axis_add(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = x + y

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([[1.0], [2.0], [3.0]], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    expected = x_np.astype(compute_dtype) + y_np.astype(compute_dtype)
    got = _run_expr_broadcast(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_left_pad_add(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = x + y

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([10.0, 20.0, 30.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=thor.physical.numpy_dtypes.fp32,
    ).astype(storage_dtype)

    expected = x_np.astype(compute_dtype) + y_np.astype(compute_dtype)
    got = _run_expr_broadcast(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_multi_axis_nested_expression(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    expr = ex.max((x + y) * (z + 1.0), 2.0)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype=thor.physical.numpy_dtypes.fp32,
    ).astype(storage_dtype)

    y_np = np.array(
        [[[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0], [4.5, 5.0, 5.5, 6.0]]],
        dtype=thor.physical.numpy_dtypes.fp32,
    ).astype(storage_dtype)

    z_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)
    z_ref = z_np.astype(compute_dtype)

    expected = np.maximum((x_ref + y_ref) * (z_ref + 1.0), 2.0)
    got = _run_expr_broadcast(expr, ['x', 'y', 'z'], x_np, y_np, z_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_div_pow_expression(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = (x / (y + 1.0))**2.0

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [2.0, 4.0, 6.0],
            [8.0, 10.0, 12.0],
        ],
        dtype=thor.physical.numpy_dtypes.fp32,
    ).astype(storage_dtype)

    y_np = np.array([[1.0], [3.0]], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected = np.power(x_ref / (y_ref + 1.0), 2.0)
    got = _run_expr_broadcast(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_scalar_lift_like_expression(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = (x * 0.5) + y - 3.0

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([2.0, 4.0, 6.0, 8.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([[10.0]], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    expected = (x_np.astype(compute_dtype) * 0.5) + y_np.astype(compute_dtype) - 3.0
    got = _run_expr_broadcast(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_three_inputs_mixed_ranks(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")
    expr = x + y * z

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ],
        dtype=thor.physical.numpy_dtypes.fp32,
    ).astype(storage_dtype)

    y_np = np.array(
        [
            [[10.0]],
            [[20.0]],
        ],
        dtype=thor.physical.numpy_dtypes.fp32,
    ).astype(storage_dtype)

    z_np = np.array([0.1, 0.2, 0.3], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    expected = x_np.astype(compute_dtype) + y_np.astype(compute_dtype) * z_np.astype(compute_dtype)
    got = _run_expr_broadcast(expr, ['x', 'y', 'z'], x_np, y_np, z_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_requested_output_shape_add_singletons(dtype: thor.DataType):
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)

    x = ex.input("x")
    y = ex.input("y")
    expr = x + y

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1.0, 2.0, 3.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([[10.0, 20.0, 30.0]], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    expected = x_np.astype(compute_dtype) + y_np.astype(compute_dtype)

    x_gpu = _copy_numpy_to_gpu(x_np, stream, dtype, gpu_num=gpu_num)
    y_gpu = _copy_numpy_to_gpu(y_np, stream, dtype, gpu_num=gpu_num)

    fused_equation = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=False,
    )

    stamped_equation = fused_equation.stamp({
        'x': x_gpu,
        'y': y_gpu
    }, stream, requestedOutputShape=[1, 1, 3])
    stamped_equation.run()

    out_gpu = stamped_equation.output()
    out_cpu = out_gpu.clone(Placement(DeviceType.cpu))
    out_cpu.copy_from_async(out_gpu, stream)
    stream.synchronize()

    got = out_cpu.numpy()
    expected_reshaped = expected.reshape((1, 1, 3))

    assert got.shape == expected_reshaped.shape
    _assert_close(got, expected_reshaped, dtype)


@pytest.mark.cuda
def test_broadcast_nested_expression_fp32_numerical_stamped():
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)
    cpu_placement = Placement(DeviceType.cpu)

    x_desc = PhysicalTensor.Descriptor(thor.DataType.fp32, dimensions=[2, 1, 4])
    y_desc = PhysicalTensor.Descriptor(thor.DataType.fp32, dimensions=[1, 3, 4])
    z_desc = PhysicalTensor.Descriptor(thor.DataType.fp32, dimensions=[4])

    x_host = PhysicalTensor(cpu_placement, x_desc)
    y_host = PhysicalTensor(cpu_placement, y_desc)
    z_host = PhysicalTensor(cpu_placement, z_desc)

    x_gpu = PhysicalTensor(gpu_placement, x_desc)
    y_gpu = PhysicalTensor(gpu_placement, y_desc)
    z_gpu = PhysicalTensor(gpu_placement, z_desc)

    x_np_view = x_host.numpy()
    y_np_view = y_host.numpy()
    z_np_view = z_host.numpy()

    x_np_view[:] = [
        [[0.5, 1.0, 1.5, 2.0]],
        [[0.8, 0.7, 2.5, 4.1]],
    ]
    y_np_view[:] = [[
        [2.0, 3.0, 4.0, 5.0],
        [8.0, 2.1, 0.6, 2.0],
        [1.2, 3.4, 5.6, 7.8],
    ]]
    z_np_view[:] = [2.0, 4.0, 6.0, 8.0]

    x_gpu.copy_from_async(x_host, stream)
    y_gpu.copy_from_async(y_host, stream)
    z_gpu.copy_from_async(z_host, stream)

    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    expr = ex.max(
        ex.sqrt(ex.exp2((x + 3.0) * (y - 1.0))),
        ex.min((z / 2.0)**2.0, ex.log2(y + 8.0)),
    )

    expected = np.maximum(
        np.sqrt(np.exp2((x_np_view + 3.0) * (y_np_view - 1.0))),
        np.minimum(np.power(z_np_view / 2.0, 2.0), np.log2(y_np_view + 8.0)),
    )

    fused_equation = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=False,
    )

    stamped_equation = fused_equation.stamp({
        'x': x_gpu,
        'y': y_gpu,
        'z': z_gpu
    }, stream)
    stamped_equation.run()

    out_gpu = stamped_equation.output()
    out_cpu = out_gpu.clone(cpu_placement)
    out_cpu.copy_from_async(out_gpu, stream)
    stream.synchronize()

    got = out_cpu.numpy()
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected.astype(np.float32), rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_incompatible_shapes_raises(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = x + y

    x_np = np.zeros((2, 3), dtype=thor.physical.numpy_dtypes.fp32).astype(_numpy_storage_dtype(dtype))
    y_np = np.zeros((2, 4), dtype=thor.physical.numpy_dtypes.fp32).astype(_numpy_storage_dtype(dtype))

    eq = ex.compile(
        expr,
        device_num=0,
        use_fast_math=False,
    )

    stream = thor.physical.Stream(0)
    x_gpu = _copy_numpy_to_gpu(x_np, stream, dtype, gpu_num=0)
    y_gpu = _copy_numpy_to_gpu(y_np, stream, dtype, gpu_num=0)
    out_gpu = _gpu_tensor([2, 4], dtype, gpu_num=0)

    with pytest.raises(RuntimeError, match="broadcast|compatible|axis|dimension"):
        eq.run({
            'x': x_gpu,
            'y': y_gpu
        }, out_gpu, stream)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_four_dimensional_multi_axis(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = (x + y) * 0.5

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = (np.arange(2 * 1 * 4 * 1, dtype=thor.physical.numpy_dtypes.fp32).reshape(2, 1, 4, 1) +
            1.0).astype(storage_dtype)
    y_np = ((np.arange(1 * 3 * 1 * 5, dtype=thor.physical.numpy_dtypes.fp32).reshape(1, 3, 1, 5) + 10.0) /
            10.0).astype(storage_dtype)

    expected = (x_np.astype(compute_dtype) + y_np.astype(compute_dtype)) * 0.5
    got = _run_expr_broadcast(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_all_singleton_axes_rhs(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = x + (y * 2.0) - 1.0

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = (np.arange(2 * 3 * 4, dtype=thor.physical.numpy_dtypes.fp32).reshape(2, 3, 4) + 1.0).astype(storage_dtype)
    y_np = np.array([[[3.5]]], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    expected = x_np.astype(compute_dtype) + (y_np.astype(compute_dtype) * 2.0) - 1.0
    got = _run_expr_broadcast(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_rightmost_vector(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = (x * 2.0) + y

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = (np.arange(2 * 3 * 4, dtype=thor.physical.numpy_dtypes.fp32).reshape(2, 3, 4) + 1.0).astype(storage_dtype)
    y_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    expected = (x_np.astype(compute_dtype) * 2.0) + y_np.astype(compute_dtype)
    got = _run_expr_broadcast(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_repeated_same_input(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = x + x * y - ex.min(x, y + 100.0)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype=thor.physical.numpy_dtypes.fp32,
    ).astype(storage_dtype)

    y_np = np.array(
        [[[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0], [4.5, 5.0, 5.5, 6.0]]],
        dtype=thor.physical.numpy_dtypes.fp32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected = x_ref + x_ref * y_ref - np.minimum(x_ref, y_ref + 100.0)
    got = _run_expr_broadcast(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_direct_run_requested_output_shape_add_singletons(dtype: thor.DataType):
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)

    x = ex.input("x")
    y = ex.input("y")
    expr = x + y

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1.0, 2.0, 3.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([[10.0, 20.0, 30.0]], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    expected = (x_np.astype(compute_dtype) + y_np.astype(compute_dtype)).reshape(1, 1, 3)

    x_gpu = _copy_numpy_to_gpu(x_np, stream, dtype, gpu_num=gpu_num)
    y_gpu = _copy_numpy_to_gpu(y_np, stream, dtype, gpu_num=gpu_num)
    out_gpu = _gpu_tensor([1, 1, 3], dtype, gpu_num=gpu_num)

    fused_equation = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=False,
    )

    fused_equation.run({
        'x': x_gpu,
        'y': y_gpu
    }, out_gpu, stream)

    got = _copy_gpu_to_numpy(out_gpu, stream, dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_broadcast_stamped_reused_twice(dtype: thor.DataType):
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)
    cpu_placement = Placement(DeviceType.cpu)

    x_desc = PhysicalTensor.Descriptor(dtype, dimensions=[2, 1, 4])
    y_desc = PhysicalTensor.Descriptor(dtype, dimensions=[1, 3, 4])

    x_host = PhysicalTensor(cpu_placement, x_desc)
    y_host = PhysicalTensor(cpu_placement, y_desc)
    x_gpu = PhysicalTensor(gpu_placement, x_desc)
    y_gpu = PhysicalTensor(gpu_placement, y_desc)

    x_np_view = x_host.numpy()
    y_np_view = y_host.numpy()

    host_np_dtype = _numpy_storage_dtype(dtype)

    x_np_view[:] = np.array([
        [[1.0, 2.0, 3.0, 4.0]],
        [[5.0, 6.0, 7.0, 8.0]],
    ], dtype=thor.physical.numpy_dtypes.fp32).astype(host_np_dtype)

    y_np_view[:] = np.array(
        [[
            [0.5, 1.0, 1.5, 2.0],
            [2.5, 3.0, 3.5, 4.0],
            [4.5, 5.0, 5.5, 6.0],
        ]], dtype=thor.physical.numpy_dtypes.fp32).astype(host_np_dtype)

    x_gpu.copy_from_async(x_host, stream)
    y_gpu.copy_from_async(y_host, stream)

    x = ex.input("x")
    y = ex.input("y")
    expr = (x + y) * (x - y + 2.0)

    fused_equation = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=False,
    )

    stamped_equation = fused_equation.stamp({
        'x': x_gpu,
        'y': y_gpu
    }, stream)

    compute_np_dtype = _numpy_compute_dtype(dtype)
    expected_first = (x_np_view.astype(compute_np_dtype) + y_np_view.astype(compute_np_dtype)) * (
        x_np_view.astype(compute_np_dtype) - y_np_view.astype(compute_np_dtype) + 2.0)

    stamped_equation.run()

    out_gpu_1 = stamped_equation.output()
    out_cpu_1 = out_gpu_1.clone(cpu_placement)
    out_cpu_1.copy_from_async(out_gpu_1, stream)
    stream.synchronize()

    _assert_close(out_cpu_1.numpy(), expected_first, dtype)

    x_np_view[:] = np.array([
        [[2.0, 4.0, 6.0, 8.0]],
        [[1.5, 2.5, 3.5, 4.5]],
    ], dtype=thor.physical.numpy_dtypes.fp32).astype(host_np_dtype)

    y_np_view[:] = np.array(
        [[
            [1.0, 0.5, 2.0, 1.5],
            [3.0, 2.5, 4.0, 3.5],
            [5.0, 4.5, 6.0, 5.5],
        ]], dtype=thor.physical.numpy_dtypes.fp32).astype(host_np_dtype)

    x_gpu.copy_from_async(x_host, stream)
    y_gpu.copy_from_async(y_host, stream)

    expected_second = (x_np_view.astype(compute_np_dtype) + y_np_view.astype(compute_np_dtype)) * (
        x_np_view.astype(compute_np_dtype) - y_np_view.astype(compute_np_dtype) + 2.0)

    stamped_equation.run()

    out_gpu_2 = stamped_equation.output()
    out_cpu_2 = out_gpu_2.clone(cpu_placement)
    out_cpu_2.copy_from_async(out_gpu_2, stream)
    stream.synchronize()

    _assert_close(out_cpu_2.numpy(), expected_second, dtype)
