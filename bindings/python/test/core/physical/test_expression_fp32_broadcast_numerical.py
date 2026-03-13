import numpy as np
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream, Placement, DeviceType


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


def _copy_numpy_to_gpu(values: np.ndarray, stream, gpu_num=0) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=np.float32, order="C")
    cpu = _cpu_tensor(list(values.shape))
    gpu = _gpu_tensor(list(values.shape), gpu_num=gpu_num)

    cpu_view = _cpu_numpy_view(cpu)
    cpu_view[...] = values
    gpu.copy_from_async(cpu, stream)
    return gpu


def _copy_gpu_to_numpy(t: thor.physical.PhysicalTensor, stream) -> np.ndarray:
    cpu = _cpu_tensor(list(t.get_descriptor().get_dimensions()))
    cpu.copy_from_async(t, stream)
    stream.synchronize()
    return np.array(_cpu_numpy_view(cpu), copy=True)


def _run_expr_broadcast(expr, *inputs: np.ndarray, gpu_num: int = 0, use_fast_math: bool = False) -> np.ndarray:
    assert len(inputs) >= 1
    inputs = [np.asarray(arr, dtype=np.float32, order="C") for arr in inputs]

    expected_shape = np.broadcast_shapes(*[arr.shape for arr in inputs])

    eq = ex.compile(
        expr,
        dtype=thor.DataType.fp32,
        device_num=gpu_num,
        use_fast_math=use_fast_math,
    )

    stream = thor.physical.Stream(gpu_num)
    gpu_inputs = [_copy_numpy_to_gpu(arr, stream) for arr in inputs]
    gpu_output = _gpu_tensor(list(expected_shape), gpu_num=gpu_num)

    eq.run(gpu_inputs, gpu_output, stream)
    return _copy_gpu_to_numpy(gpu_output, stream)


def _assert_close(got: np.ndarray, expected: np.ndarray, *, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(got, expected.astype(np.float32), rtol=rtol, atol=atol)


@pytest.mark.cuda
def test_broadcast_same_rank_singleton_axis_add():
    x = ex.input(0)
    y = ex.input(1)
    expr = x + y

    x_np = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)  # (3, 1)
    y_np = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)  # (1, 4)

    expected = x_np + y_np
    got = _run_expr_broadcast(expr, x_np, y_np)

    assert got.shape == expected.shape
    _assert_close(got, expected)


@pytest.mark.cuda
def test_broadcast_left_pad_add():
    x = ex.input(0)
    y = ex.input(1)
    expr = x + y

    x_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)  # (3,)
    y_np = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )  # (2, 3)

    expected = x_np + y_np
    got = _run_expr_broadcast(expr, x_np, y_np)

    assert got.shape == expected.shape
    _assert_close(got, expected)


@pytest.mark.cuda
def test_broadcast_multi_axis_nested_expression():
    x = ex.input(0)
    y = ex.input(1)
    z = ex.input(2)

    expr = ex.max((x + y) * (z + 1.0), 2.0)

    x_np = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype=np.float32,
    )  # (2, 1, 4)

    y_np = np.array(
        [[[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0], [4.5, 5.0, 5.5, 6.0]]],
        dtype=np.float32,
    )  # (1, 3, 4)

    z_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # (4,)

    expected = np.maximum((x_np + y_np) * (z_np + 1.0), 2.0)
    got = _run_expr_broadcast(expr, x_np, y_np, z_np)

    assert got.shape == expected.shape
    _assert_close(got, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
def test_broadcast_div_pow_expression():
    x = ex.input(0)
    y = ex.input(1)
    expr = (x / (y + 1.0))**2.0

    x_np = np.array(
        [
            [2.0, 4.0, 6.0],
            [8.0, 10.0, 12.0],
        ],
        dtype=np.float32,
    )  # (2, 3)

    y_np = np.array([[1.0], [3.0]], dtype=np.float32)  # (2, 1)

    expected = np.power(x_np / (y_np + 1.0), 2.0)
    got = _run_expr_broadcast(expr, x_np, y_np)

    assert got.shape == expected.shape
    _assert_close(got, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
def test_broadcast_scalar_lift_like_expression():
    x = ex.input(0)
    y = ex.input(1)
    expr = (x * 0.5) + y - 3.0

    x_np = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)  # (4,)
    y_np = np.array([[10.0]], dtype=np.float32)  # (1, 1)

    expected = (x_np * 0.5) + y_np - 3.0
    got = _run_expr_broadcast(expr, x_np, y_np)

    assert got.shape == expected.shape
    _assert_close(got, expected)


@pytest.mark.cuda
def test_broadcast_three_inputs_mixed_ranks():
    x = ex.input(0)
    y = ex.input(1)
    z = ex.input(2)
    expr = x + y * z

    x_np = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ],
        dtype=np.float32,
    )  # (2, 1, 3)

    y_np = np.array(
        [
            [[10.0]],
            [[20.0]],
        ],
        dtype=np.float32,
    )  # (2, 1, 1)

    z_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # (3,)

    expected = x_np + y_np * z_np
    got = _run_expr_broadcast(expr, x_np, y_np, z_np)

    assert got.shape == expected.shape
    _assert_close(got, expected)


@pytest.mark.cuda
def test_broadcast_requested_output_shape_add_singletons():
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)

    x = ex.input(0)
    y = ex.input(1)
    expr = x + y

    x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # (3,)
    y_np = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)  # (1, 3)
    expected = x_np + y_np  # (1, 3)

    x_gpu = _copy_numpy_to_gpu(x_np, stream, gpu_num=gpu_num)
    y_gpu = _copy_numpy_to_gpu(y_np, stream, gpu_num=gpu_num)

    fused_equation = ex.compile(
        expr,
        dtype=thor.DataType.fp32,
        device_num=gpu_num,
        use_fast_math=False,
    )

    stamped_equation = fused_equation.stamp([x_gpu, y_gpu], stream, requestedOutputShape=[1, 1, 3])
    stamped_equation.run()

    out_gpu = stamped_equation.output_tensor
    out_cpu = out_gpu.clone(Placement(DeviceType.cpu))
    out_cpu.copy_from_async(out_gpu, stream)
    stream.synchronize()

    got = out_cpu.numpy()
    expected_reshaped = expected.reshape((1, 1, 3)).astype(np.float32)

    assert got.shape == expected_reshaped.shape
    _assert_close(got, expected_reshaped)


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

    x = ex.input(0)
    y = ex.input(1)
    z = ex.input(2)

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
        dtype=thor.DataType.fp32,
        device_num=gpu_num,
        use_fast_math=False,
    )

    stamped_equation = fused_equation.stamp([x_gpu, y_gpu, z_gpu], stream)
    stamped_equation.run()

    out_gpu = stamped_equation.output_tensor
    out_cpu = out_gpu.clone(cpu_placement)
    out_cpu.copy_from_async(out_gpu, stream)
    stream.synchronize()

    got = out_cpu.numpy()
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected.astype(np.float32), rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
def test_broadcast_incompatible_shapes_raises():
    x = ex.input(0)
    y = ex.input(1)
    expr = x + y

    x_np = np.zeros((2, 3), dtype=np.float32)
    y_np = np.zeros((2, 4), dtype=np.float32)

    eq = ex.compile(
        expr,
        dtype=thor.DataType.fp32,
        device_num=0,
        use_fast_math=False,
    )

    stream = thor.physical.Stream(0)
    x_gpu = _copy_numpy_to_gpu(x_np, stream, gpu_num=0)
    y_gpu = _copy_numpy_to_gpu(y_np, stream, gpu_num=0)
    out_gpu = _gpu_tensor([2, 4], gpu_num=0)

    with pytest.raises(RuntimeError, match="broadcast|compatible|axis|dimension"):
        eq.run([x_gpu, y_gpu], out_gpu, stream)


@pytest.mark.cuda
def test_broadcast_four_dimensional_multi_axis():
    x = ex.input(0)
    y = ex.input(1)
    expr = (x + y) * 0.5

    x_np = np.arange(2 * 1 * 4 * 1, dtype=np.float32).reshape(2, 1, 4, 1) + 1.0
    y_np = (np.arange(1 * 3 * 1 * 5, dtype=np.float32).reshape(1, 3, 1, 5) + 10.0) / 10.0

    expected = (x_np + y_np) * 0.5
    got = _run_expr_broadcast(expr, x_np, y_np)

    assert got.shape == expected.shape
    _assert_close(got, expected)


@pytest.mark.cuda
def test_broadcast_all_singleton_axes_rhs():
    x = ex.input(0)
    y = ex.input(1)
    expr = x + (y * 2.0) - 1.0

    x_np = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 1.0
    y_np = np.array([[[3.5]]], dtype=np.float32)  # (1, 1, 1)

    expected = x_np + (y_np * 2.0) - 1.0
    got = _run_expr_broadcast(expr, x_np, y_np)

    assert got.shape == expected.shape
    _assert_close(got, expected)


@pytest.mark.cuda
def test_broadcast_rightmost_vector():
    x = ex.input(0)
    y = ex.input(1)
    expr = (x * 2.0) + y

    x_np = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 1.0
    y_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)  # (4,)

    expected = (x_np * 2.0) + y_np
    got = _run_expr_broadcast(expr, x_np, y_np)

    assert got.shape == expected.shape
    _assert_close(got, expected)


@pytest.mark.cuda
def test_broadcast_repeated_same_input():
    x = ex.input(0)
    y = ex.input(1)
    expr = x + x * y - ex.min(x, y + 100.0)

    x_np = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype=np.float32,
    )  # (2, 1, 4)

    y_np = np.array(
        [[[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0], [4.5, 5.0, 5.5, 6.0]]],
        dtype=np.float32,
    )  # (1, 3, 4)

    expected = x_np + x_np * y_np - np.minimum(x_np, y_np + 100.0)
    got = _run_expr_broadcast(expr, x_np, y_np)

    assert got.shape == expected.shape
    _assert_close(got, expected)


@pytest.mark.cuda
def test_broadcast_direct_run_requested_output_shape_add_singletons():
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)

    x = ex.input(0)
    y = ex.input(1)
    expr = x + y

    x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # (3,)
    y_np = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)  # (1, 3)
    expected = (x_np + y_np).reshape(1, 1, 3).astype(np.float32)  # (1, 1, 3)

    x_gpu = _copy_numpy_to_gpu(x_np, stream, gpu_num=gpu_num)
    y_gpu = _copy_numpy_to_gpu(y_np, stream, gpu_num=gpu_num)
    out_gpu = _gpu_tensor([1, 1, 3], gpu_num=gpu_num)

    fused_equation = ex.compile(
        expr,
        dtype=thor.DataType.fp32,
        device_num=gpu_num,
        use_fast_math=False,
    )

    fused_equation.run([x_gpu, y_gpu], out_gpu, stream)

    got = _copy_gpu_to_numpy(out_gpu, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected)


@pytest.mark.cuda
def test_broadcast_stamped_reused_twice():
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)
    cpu_placement = Placement(DeviceType.cpu)

    x_desc = PhysicalTensor.Descriptor(thor.DataType.fp32, dimensions=[2, 1, 4])
    y_desc = PhysicalTensor.Descriptor(thor.DataType.fp32, dimensions=[1, 3, 4])

    x_host = PhysicalTensor(cpu_placement, x_desc)
    y_host = PhysicalTensor(cpu_placement, y_desc)
    x_gpu = PhysicalTensor(gpu_placement, x_desc)
    y_gpu = PhysicalTensor(gpu_placement, y_desc)

    x_np_view = x_host.numpy()
    y_np_view = y_host.numpy()

    x_np_view[:] = [
        [[1.0, 2.0, 3.0, 4.0]],
        [[5.0, 6.0, 7.0, 8.0]],
    ]
    y_np_view[:] = [[
        [0.5, 1.0, 1.5, 2.0],
        [2.5, 3.0, 3.5, 4.0],
        [4.5, 5.0, 5.5, 6.0],
    ]]

    x_gpu.copy_from_async(x_host, stream)
    y_gpu.copy_from_async(y_host, stream)

    x = ex.input(0)
    y = ex.input(1)
    expr = (x + y) * (x - y + 2.0)

    fused_equation = ex.compile(
        expr,
        dtype=thor.DataType.fp32,
        device_num=gpu_num,
        use_fast_math=False,
    )

    stamped_equation = fused_equation.stamp([x_gpu, y_gpu], stream)

    expected_first = (x_np_view + y_np_view) * (x_np_view - y_np_view + 2.0)

    stamped_equation.run()

    out_gpu_1 = stamped_equation.output_tensor
    out_cpu_1 = out_gpu_1.clone(cpu_placement)
    out_cpu_1.copy_from_async(out_gpu_1, stream)
    stream.synchronize()

    np.testing.assert_allclose(
        out_cpu_1.numpy(),
        expected_first.astype(np.float32),
        rtol=2e-5,
        atol=2e-6,
    )

    # Mutate inputs, reuse the same stamped equation, and check again.
    x_np_view[:] = [
        [[2.0, 4.0, 6.0, 8.0]],
        [[1.5, 2.5, 3.5, 4.5]],
    ]
    y_np_view[:] = [[
        [1.0, 0.5, 2.0, 1.5],
        [3.0, 2.5, 4.0, 3.5],
        [5.0, 4.5, 6.0, 5.5],
    ]]

    x_gpu.copy_from_async(x_host, stream)
    y_gpu.copy_from_async(y_host, stream)

    expected_second = (x_np_view + y_np_view) * (x_np_view - y_np_view + 2.0)

    stamped_equation.run()

    out_gpu_2 = stamped_equation.output_tensor
    out_cpu_2 = out_gpu_2.clone(cpu_placement)
    out_cpu_2.copy_from_async(out_gpu_2, stream)
    stream.synchronize()

    np.testing.assert_allclose(
        out_cpu_2.numpy(),
        expected_second.astype(np.float32),
        rtol=2e-5,
        atol=2e-6,
    )
