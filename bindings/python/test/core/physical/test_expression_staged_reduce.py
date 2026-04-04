import numpy as np
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream, Placement, DeviceType, numpy_dtypes


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _gpu_tensor(shape: list[int], dtype: thor.DataType, gpu_num: int = 0) -> thor.physical.PhysicalTensor:
    placement = Placement(DeviceType.gpu, gpu_num)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _numpy_compute_dtype(dtype: thor.DataType) -> np.dtype:
    if dtype == thor.DataType.fp8_e4m3:
        return numpy_dtypes.fp16
    if dtype == thor.DataType.fp8_e5m2:
        return numpy_dtypes.fp16
    return numpy_dtypes.from_thor(dtype)


FLOAT_DTYPES = [
    thor.DataType.fp32,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
]


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    got32 = got.astype(np.float32)
    expected32 = expected.astype(np.float32)

    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got32, expected32, rtol=1e-5, atol=1e-6)
    elif dtype in (thor.DataType.fp16, thor.DataType.bf16):
        np.testing.assert_allclose(got32, expected32, rtol=5e-2, atol=5e-2)
    elif dtype == thor.DataType.fp8_e4m3:
        np.testing.assert_allclose(got32, expected32, rtol=1e-1, atol=1e-1)
    elif dtype == thor.DataType.fp8_e5m2:
        np.testing.assert_allclose(got32, expected32, rtol=2e-1, atol=0.5)
    else:
        raise AssertionError(f"Unhandled dtype: {dtype}")


def _apply_numpy_squeeze(arr: np.ndarray, squeeze) -> np.ndarray:
    """
    Match the thor API semantics:

    squeeze=False / None -> no squeeze
    squeeze=True         -> squeeze all singleton dims
    squeeze=int          -> squeeze that singleton axis
    squeeze=list[int]    -> squeeze those singleton axes

    Important: for list[int], squeeze in descending axis order so axis positions
    remain stable as dimensions are removed.
    """
    if squeeze is False or squeeze is None:
        return arr

    if squeeze is True:
        return np.squeeze(arr)

    if isinstance(squeeze, int):
        return np.squeeze(arr, axis=squeeze)

    axes = list(squeeze)
    out = arr
    for axis in sorted(axes, reverse=True):
        out = np.squeeze(out, axis=axis)
    return out


def _run_staged_expr(
    expr,
    input_names: list[str],
    *inputs: np.ndarray,
    dtype: thor.DataType,
    gpu_num: int = 0,
    use_fast_math: bool = False,
) -> np.ndarray:
    assert len(inputs) >= 1
    assert len(inputs) == len(input_names)

    stream = Stream(gpu_num=gpu_num)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)
    cpu_placement = Placement(DeviceType.cpu, 0)

    input_tensors_gpu: dict[str, PhysicalTensor] = {}

    for name, arr in zip(input_names, inputs):
        host_desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
        host_tensor = PhysicalTensor(cpu_placement, host_desc)
        host_np = host_tensor.numpy()
        host_np[...] = arr

        gpu_tensor = PhysicalTensor(gpu_placement, host_desc)
        gpu_tensor.copy_from_async(host_tensor, stream)
        input_tensors_gpu[name] = gpu_tensor

    eq = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=use_fast_math,
    )

    stamped = eq.stamp(input_tensors_gpu, stream)
    stamped.run()

    out_gpu = stamped.output()
    out_host = PhysicalTensor(cpu_placement, out_gpu.get_descriptor())
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()

    return out_host.numpy().copy()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_sum_staged_with_prologue_and_epilogue(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    expr = ex.sqrt(ex.reduce_sum(((x + 3.0) * (y - 1.0)), axis=1, squeeze=False))

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    y_np = np.array(
        [[1.5, 2.0, 2.5, 3.0], [1.25, 1.5, 1.75, 2.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected = np.sqrt(np.sum((x_ref + 3.0) * (y_ref - 1.0), axis=1, keepdims=True))
    expected = expected.astype(storage_dtype)

    got = _run_staged_expr(expr, ["x", "y"], x_np, y_np, dtype=dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_sum_squeeze_false_keeps_reduced_axes(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_sum((x * 2.0 + 1.0), axis=1, squeeze=False)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    expected = np.sum(x_ref * 2.0 + 1.0, axis=1, keepdims=True).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_sum_squeeze_true_removes_all_singletons(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_sum((x + 1.0), axis=[1, 2], squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    reduced = np.sum(x_ref + 1.0, axis=(1, 2), keepdims=True)
    expected = _apply_numpy_squeeze(reduced, True).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_mean_multi_axis_with_epilogue_and_squeeze_int(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_mean(((x + 2.0) * 0.5), axis=[1, 2], squeeze=2) + 4.0

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    reduced = np.mean((x_ref + 2.0) * 0.5, axis=(1, 2), keepdims=True)
    expected = (_apply_numpy_squeeze(reduced, 2) + 4.0).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_prod_staged(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_prod((x + 1.0), axis=1, squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    reduced = np.prod(x_ref + 1.0, axis=1, keepdims=True)
    expected = _apply_numpy_squeeze(reduced, True).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_min_staged(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = ex.reduce_min(((x * 2.0) - y), axis=1, squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    y_np = np.array(
        [[1.0, 6.0, 2.0], [0.0, 1.0, 5.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)
    reduced = np.min((x_ref * 2.0) - y_ref, axis=1, keepdims=True)
    expected = _apply_numpy_squeeze(reduced, True).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x", "y"], x_np, y_np, dtype=dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_max_staged_with_epilogue(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_max(ex.ln((x / 2.0 + 1.0)), axis=1, squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    reduced = np.max(x_ref / 2.0 + 1.0, axis=1, keepdims=True)
    expected = np.log(_apply_numpy_squeeze(reduced, True)).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_norm1_staged(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_norm1((x - 2.0), axis=1, squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[1.0, 2.5, 4.0], [0.0, 3.0, 5.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    reduced = np.sum(np.abs(x_ref - 2.0), axis=1, keepdims=True)
    expected = _apply_numpy_squeeze(reduced, True).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_norm2_staged(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_norm2((x - 1.5), axis=1, squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    reduced = np.sqrt(np.sum(np.square(x_ref - 1.5), axis=1, keepdims=True))
    expected = _apply_numpy_squeeze(reduced, True).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_two_reduction_boundaries_staged(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    expr = ex.reduce_sum(((x + 1.0) * y), axis=1, squeeze=False)
    expr = ex.reduce_mean((expr + 2.0), axis=0, squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    y_np = np.array(
        [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    stage1 = np.sum((x_ref + 1.0) * y_ref, axis=1, keepdims=True)
    stage2 = np.mean(stage1 + 2.0, axis=0, keepdims=True)
    expected = _apply_numpy_squeeze(stage2, True).astype(storage_dtype)
    if len(expected.shape) == 0:
        expected.shape = (1,)

    got = _run_staged_expr(expr, ["x", "y"], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_multiple_reduce_stages_with_prologue_and_epilogue(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    expr = ex.reduce_sum(((x + 1.0) * (y - 0.5)), axis=2, squeeze=False)
    expr = ex.reduce_max((expr * 0.25 + 2.0), axis=1, squeeze=False)
    expr = ex.sqrt(expr + 1.0)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    y_np = np.array(
        [
            [[1.5, 2.0], [2.5, 3.0], [3.5, 4.0]],
            [[1.25, 1.75], [2.25, 2.75], [3.25, 3.75]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    stage1 = np.sum((x_ref + 1.0) * (y_ref - 0.5), axis=2, keepdims=True)
    stage2 = np.max(stage1 * 0.25 + 2.0, axis=1, keepdims=True)
    expected = np.sqrt(stage2 + 1.0).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x", "y"], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_sum_squeeze_specific_axes_list(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_sum((x + 1.0), axis=[1, 2], squeeze=[1, 2])

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    reduced = np.sum(x_ref + 1.0, axis=(1, 2), keepdims=True)
    expected = _apply_numpy_squeeze(reduced, [1, 2]).astype(storage_dtype)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_run_convenience_rejects_multi_stage_reduction_expression():
    gpu_num = 0
    stream = Stream(gpu_num=gpu_num)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)
    cpu_placement = Placement(DeviceType.cpu, 0)

    dtype = thor.DataType.fp32
    descriptor = PhysicalTensor.Descriptor(dtype, [2, 3])

    x_host = PhysicalTensor(cpu_placement, descriptor)
    x_host.numpy()[:] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    x_gpu = PhysicalTensor(gpu_placement, descriptor)
    x_gpu.copy_from_async(x_host, stream)

    out_desc = PhysicalTensor.Descriptor(dtype, [2])
    out_gpu = PhysicalTensor(gpu_placement, out_desc)

    x = ex.input("x")
    expr = ex.reduce_sum((x + 1.0), axis=1, squeeze=True)

    eq = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=False,
    )

    with pytest.raises(RuntimeError, match="single-stage fused expressions|staged expressions"):
        eq.run({
            "x": x_gpu
        }, out_gpu, stream)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_stamp_output_tensor_matches_expected_shape_squeeze_false(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_sum((x + 1.0), axis=1, squeeze=False)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    gpu_placement = Placement(DeviceType.gpu, 0)
    cpu_placement = Placement(DeviceType.cpu, 0)

    host_desc = PhysicalTensor.Descriptor(dtype, list(x_np.shape))
    x_host = PhysicalTensor(cpu_placement, host_desc)
    x_host.numpy()[:] = x_np

    x_gpu = PhysicalTensor(gpu_placement, host_desc)
    x_gpu.copy_from_async(x_host, stream)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu
    }, stream)

    assert list(stamped.output().dimensions) == [2, 1]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_stamp_output_tensor_matches_expected_shape_squeeze_true(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_sum((x + 1.0), axis=1, squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    gpu_placement = Placement(DeviceType.gpu, 0)
    cpu_placement = Placement(DeviceType.cpu, 0)

    host_desc = PhysicalTensor.Descriptor(dtype, list(x_np.shape))
    x_host = PhysicalTensor(cpu_placement, host_desc)
    x_host.numpy()[:] = x_np

    x_gpu = PhysicalTensor(gpu_placement, host_desc)
    x_gpu.copy_from_async(x_host, stream)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu
    }, stream)

    assert list(stamped.output().get_descriptor().get_dimensions()) == [2]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_stamp_output_tensor_matches_expected_shape_squeeze_specific_axis(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_sum((x + 1.0), axis=[1, 2], squeeze=2)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    gpu_placement = Placement(DeviceType.gpu, 0)
    cpu_placement = Placement(DeviceType.cpu, 0)

    host_desc = PhysicalTensor.Descriptor(dtype, list(x_np.shape))
    x_host = PhysicalTensor(cpu_placement, host_desc)
    x_host.numpy()[:] = x_np

    x_gpu = PhysicalTensor(gpu_placement, host_desc)
    x_gpu.copy_from_async(x_host, stream)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu
    }, stream)

    assert list(stamped.output().get_descriptor().get_dimensions()) == [2, 1]


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("input_shape", "reduce_axis", "squeeze", "expected_shape"),
    [
        # No squeeze: reduced dims stay as 1s
        ((2, 3, 4), 1, False, (2, 1, 4)),
        ((2, 3, 4), [1, 2], False, (2, 1, 1)),
        ((2, 3, 4), None, False, (1, 1, 1)),

        # Squeeze all singleton dims after reduction
        ((2, 3, 4), 1, True, (2, 4)),
        ((2, 3, 4), [1, 2], True, (2,)),
        ((2, 3, 4), None, True, (1,)),  # keep at least one dim

        # Squeeze one specific reduced axis
        ((2, 3, 4), 1, 1, (2, 4)),
        ((2, 3, 4), [1, 2], 1, (2, 1)),
        ((2, 3, 4), [1, 2], 2, (2, 1)),

        # Squeeze explicit list of reduced axes
        ((2, 3, 4), [1, 2], [1, 2], (2,)),
        ((2, 3, 4, 5), [1, 3], [1, 3], (2, 4)),

        # Existing singleton dims mixed with reduced singleton dims
        ((2, 1, 3, 1, 4), 2, False, (2, 1, 1, 1, 4)),
        ((2, 1, 3, 1, 4), 2, True, (2, 4)),
        ((2, 1, 3, 1, 4), 2, 2, (2, 1, 1, 4)),
        ((2, 1, 3, 1, 4), 2, [1, 2, 3], (2, 4)),

        # Full reduction from a tensor already containing singleton dims
        ((2, 1, 3, 1), None, False, (1, 1, 1, 1)),
        ((2, 1, 3, 1), None, True, (1,)),
        ((2, 1, 3, 1), None, [0, 1, 2, 3], (1,)),
    ],
)
def test_reduce_sum_squeeze_matrix(
    input_shape: tuple[int, ...],
    reduce_axis,
    squeeze,
    expected_shape: tuple[int, ...],
):
    dtype = thor.DataType.fp32
    x = ex.input("x")
    expr = ex.reduce_sum((x + 1.0), axis=reduce_axis, squeeze=squeeze)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    numel = int(np.prod(input_shape))
    x_np = np.arange(1, numel + 1, dtype=np.float32).reshape(input_shape).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)

    if reduce_axis is None:
        axes_tuple = tuple(range(x_ref.ndim))
    elif isinstance(reduce_axis, int):
        axes_tuple = (reduce_axis,)
    else:
        axes_tuple = tuple(reduce_axis)

    reduced = np.sum(x_ref + 1.0, axis=axes_tuple, keepdims=True)
    expected = _apply_numpy_squeeze(reduced, squeeze).astype(storage_dtype)
    if (len(expected.shape) == 0):
        expected.shape = (1,)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)

    assert got.shape == expected_shape
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_parallel_fanout_reductions_join_before_final_epilogue(dtype: thor.DataType):
    """
    Exercises:

      shared fused trunk
            |
        +---+---+
        |       |
      reduce   reduce
        |       |
        +---+---+
            |
      final broadcast epilogue

    If dependency waits are missing, the final epilogue can read unfinished
    reduction outputs. If the final helper-stream join back to the caller stream
    is missing, the host copy in _run_staged_expr can race the GPU work.
    """
    x = ex.input("x")

    trunk = ex.exp(x * 0.03125 + 1.0)
    left = ex.reduce_sum(trunk, axis=2, squeeze=False)  # [B, M, 1]
    right = ex.reduce_mean(trunk, axis=1, squeeze=False)  # [B, 1, N]
    expr = ex.sqrt(left + right + 1.0)  # [B, M, N]

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    shape = (4, 48, 64)
    x_np = np.linspace(0.1, 3.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    trunk_ref = np.exp(x_ref * 0.03125 + 1.0)
    expected = np.sqrt(np.sum(trunk_ref, axis=2, keepdims=True) + np.mean(trunk_ref, axis=1, keepdims=True) +
                       1.0).astype(storage_dtype)

    # Run multiple times to better exercise helper-stream ordering.
    for _ in range(4):
        got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)
        assert got.shape == expected.shape
        _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_parallel_branch_and_chain_dependencies_join_correctly(dtype: thor.DataType):
    """
    Exercises:

      left prologue -> left reduction -> left epilogue --+
                                                        +-> final join epilogue
                      right prologue -> right reduction -+

    This checks that a stage depending on a dependent chain and an independent
    branch waits on both producer paths correctly.
    """
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    left = ex.reduce_sum((x + 1.0) * (y + 0.5), axis=2, squeeze=False)  # [B, M, 1]
    left = ex.sqrt(left + 2.0)  # [B, M, 1]

    right = ex.reduce_max(z * 0.25 + 3.0, axis=1, squeeze=False)  # [B, 1, N]

    expr = ex.ln(left + right + 1.0)  # [B, M, N]

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    shape = (5, 32, 40)

    x_np = np.linspace(0.25, 2.25, num=int(np.prod(shape)), dtype=np.float32).reshape(shape).astype(storage_dtype)
    y_np = np.linspace(0.5, 1.5, num=int(np.prod(shape)), dtype=np.float32).reshape(shape).astype(storage_dtype)
    z_np = np.linspace(1.0, 5.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)
    z_ref = z_np.astype(compute_dtype)

    left_ref = np.sum((x_ref + 1.0) * (y_ref + 0.5), axis=2, keepdims=True)
    left_ref = np.sqrt(left_ref + 2.0)

    right_ref = np.max(z_ref * 0.25 + 3.0, axis=1, keepdims=True)

    expected = np.log(left_ref + right_ref + 1.0).astype(storage_dtype)

    for _ in range(4):
        got = _run_staged_expr(expr, ["x", "y", "z"], x_np, y_np, z_np, dtype=dtype)
        assert got.shape == expected.shape
        _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_multi_level_diamond_then_downstream_reduction_synchronizes(dtype: thor.DataType):
    """
    Exercises a deeper DAG:

          shared trunk
           /      \
       reduce    reduce
           \      /
            join fuse
                |
            downstream reduction
                |
            final epilogue

    This verifies that event waits are correct not only for the immediate join,
    but also for later stages that depend on that joined result.
    """
    x = ex.input("x")

    trunk = (x + 1.0) * 0.5
    left = ex.reduce_sum(trunk, axis=2, squeeze=False)  # [B, M, 1]
    right = ex.reduce_max(trunk, axis=1, squeeze=False)  # [B, 1, N]
    joined = left + right + 2.0  # [B, M, N]
    reduced = ex.reduce_mean(joined, axis=0, squeeze=False)  # [1, M, N]
    expr = ex.sqrt(reduced + 1.0)  # [1, M, N]

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    shape = (4, 24, 32)
    x_np = np.linspace(1.0, 7.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    trunk_ref = (x_ref + 1.0) * 0.5
    left_ref = np.sum(trunk_ref, axis=2, keepdims=True)
    right_ref = np.max(trunk_ref, axis=1, keepdims=True)
    joined_ref = left_ref + right_ref + 2.0
    reduced_ref = np.mean(joined_ref, axis=0, keepdims=True)
    expected = np.sqrt(reduced_ref + 1.0).astype(storage_dtype)

    for _ in range(4):
        got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)
        assert got.shape == expected.shape
        _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmax_staged_single_axis_squeeze_false(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = ex.argmax(((x * 2.0) - y), axis=1, squeeze=False)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]],
            [[2.0, 1.0], [0.0, 6.0], [4.0, 3.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    y_np = np.array(
        [
            [[0.0, 1.0], [4.0, 0.0], [1.0, 2.0]],
            [[1.0, 0.0], [0.0, 5.0], [2.0, 1.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    ref = (x_np.astype(compute_dtype) * 2.0) - y_np.astype(compute_dtype)
    expected = np.argmax(ref, axis=1).astype(np.uint32)
    expected = np.expand_dims(expected, axis=1)  # keep reduced axis as singleton

    got = _run_staged_expr(expr, ["x", "y"], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmin_staged_multi_axis_flattened_indices_squeeze_false(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = ex.argmin(((x + 1.0) * 0.5) - y, axis=[1, 2], squeeze=False)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[4.0, 8.0], [2.0, 7.0], [5.0, 9.0]],
            [[6.0, 3.0], [1.0, 4.0], [8.0, 2.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    y_np = np.array(
        [
            [[0.0, 1.0], [3.0, 0.0], [1.0, 2.0]],
            [[2.0, 0.0], [0.0, 1.0], [3.0, 0.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    ref = ((x_np.astype(compute_dtype) + 1.0) * 0.5) - y_np.astype(compute_dtype)
    expected = np.argmin(ref.reshape(ref.shape[0], -1), axis=1).astype(np.uint32)
    expected = expected.reshape(ref.shape[0], 1, 1)

    got = _run_staged_expr(expr, ["x", "y"], x_np, y_np, dtype=dtype)

    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmax_staged_all_axes_scalar_like_output(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.argmax((x * 1.5) - 2.0, axis=None, squeeze=False)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]],  # Don't change these numbers, they catch an fp8 kernel bug as set
        dtype=np.float32,
    ).astype(storage_dtype)

    ref = (x_np.astype(compute_dtype) * 1.5) - 2.0
    expected = np.array([[np.argmax(ref.reshape(-1))]], dtype=np.uint32)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)

    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmin_staged_single_axis_squeeze_true(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.argmin((x - 3.0) * (x + 1.0), axis=1, squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[4.0, 8.0], [2.0, 7.0], [5.0, 9.0]],
            [[6.0, 3.0], [1.0, 4.0], [8.0, 2.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    ref = (x_np.astype(compute_dtype) - 3.0) * (x_np.astype(compute_dtype) + 1.0)
    reduced = np.argmin(ref, axis=1)
    expected = reduced.astype(np.uint32)

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)

    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmax_staged_multi_axis_squeeze_specific_axes_list(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.argmax((x * 0.25) + 7.0, axis=[1, 2], squeeze=[1, 2])

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 9.0], [3.0, 4.0], [5.0, 2.0]],
            [[8.0, 0.0], [6.0, 7.0], [1.0, 3.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    ref = (x_np.astype(compute_dtype) * 0.25) + 7.0
    reduced = np.argmax(ref.reshape(ref.shape[0], -1), axis=1).astype(np.uint32)
    expected = reduced

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)

    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def _host_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu = Placement(DeviceType.cpu, 0)
    gpu = Placement(DeviceType.gpu, gpu_num)
    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
    host = PhysicalTensor(cpu, desc)
    host.numpy()[...] = arr
    device = PhysicalTensor(gpu, desc)
    device.copy_from_async(host, stream)
    return device


def _run_staged_expr_with_preallocated(
    expr,
    input_names: list[str],
    *inputs: np.ndarray,
    dtype: thor.DataType,
    output_dtype: thor.DataType | None = None,
    output_shape: list[int] | None = None,
    gpu_num: int = 0,
    use_fast_math: bool = False,
):
    assert len(inputs) >= 1
    assert len(inputs) == len(input_names)

    stream = Stream(gpu_num=gpu_num)
    gpu_placement = Placement(DeviceType.gpu, gpu_num)
    cpu_placement = Placement(DeviceType.cpu, 0)

    input_tensors_gpu: dict[str, PhysicalTensor] = {}

    for name, arr in zip(input_names, inputs):
        host_desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
        host_tensor = PhysicalTensor(cpu_placement, host_desc)
        host_tensor.numpy()[...] = arr

        gpu_tensor = PhysicalTensor(gpu_placement, host_desc)
        gpu_tensor.copy_from_async(host_tensor, stream)
        input_tensors_gpu[name] = gpu_tensor

    eq = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=use_fast_math,
    )

    output_name = eq.output_names()[0]

    if output_shape is None:
        output_shape = list(eq.output_shape(input_tensors_gpu))
    if output_dtype is None:
        # For reductions this is the storage dtype; argmin/argmax callers should pass uint32 explicitly.
        output_dtype = dtype

    out_desc = PhysicalTensor.Descriptor(output_dtype, list(output_shape))
    out_gpu = PhysicalTensor(gpu_placement, out_desc)

    stamped = eq.stamp(
        input_tensors_gpu,
        stream,
        preallocated_outputs={
            output_name: out_gpu
        },
    )
    stamped.run()

    out_host = PhysicalTensor(cpu_placement, out_desc)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()

    return out_host.numpy().copy(), out_gpu, stamped, eq, input_tensors_gpu


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_sum_full_reduction_preallocated_output(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_sum((x + 1.0), axis=None, squeeze=False)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).astype(storage_dtype)

    expected = np.sum(x_np.astype(compute_dtype) + 1.0, axis=(0, 1, 2), keepdims=True).astype(storage_dtype)

    got, out_gpu, stamped, _, _ = _run_staged_expr_with_preallocated(
        expr,
        ["x"],
        x_np,
        dtype=dtype,
        output_dtype=thor.DataType.fp32,
        output_shape=[1, 1, 1],
    )

    assert got.shape == (1, 1, 1)
    assert list(out_gpu.dimensions) == [1, 1, 1]
    assert list(stamped.output().dimensions) == [1, 1, 1]
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_sum_preallocated_output_wrong_shape_raises(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.reduce_sum((x + 1.0), axis=None, squeeze=False)

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    gpu_placement = Placement(DeviceType.gpu, 0)
    cpu_placement = Placement(DeviceType.cpu, 0)

    host_desc = PhysicalTensor.Descriptor(dtype, list(x_np.shape))
    x_host = PhysicalTensor(cpu_placement, host_desc)
    x_host.numpy()[...] = x_np

    x_gpu = PhysicalTensor(gpu_placement, host_desc)
    x_gpu.copy_from_async(x_host, stream)

    eq = ex.compile(expr, device_num=0, use_fast_math=False)
    output_name = eq.output_names()[0]

    wrong_out = PhysicalTensor(gpu_placement, PhysicalTensor.Descriptor(dtype, [2, 1, 1]))

    with pytest.raises(RuntimeError, match="incompatible|requested output|Output tensor dimensions"):
        eq.stamp(
            {
                "x": x_gpu
            },
            stream,
            preallocated_outputs={
                output_name: wrong_out
            },
        )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_min_staged_preallocated_output(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = ex.reduce_min(((x * 2.0) - y), axis=1, squeeze=True)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    y_np = np.array(
        [[1.0, 6.0, 2.0], [0.0, 1.0, 5.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    ref = (x_np.astype(compute_dtype) * 2.0) - y_np.astype(compute_dtype)
    expected = np.min(ref, axis=1, keepdims=True).squeeze(1).astype(storage_dtype)

    got, _, _, _, _ = _run_staged_expr_with_preallocated(
        expr,
        ["x", "y"],
        x_np,
        y_np,
        dtype=dtype,
        output_dtype=thor.DataType.fp32,
        output_shape=[2],
    )

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmax_all_axes_preallocated_output(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.argmax((x * 1.5) - 2.0, axis=None, squeeze=False)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    ref = (x_np.astype(compute_dtype) * 1.5) - 2.0
    expected = np.array([[np.argmax(ref.reshape(-1))]], dtype=np.uint32)

    got, out_gpu, stamped, _, _ = _run_staged_expr_with_preallocated(
        expr,
        ["x"],
        x_np,
        dtype=dtype,
        output_dtype=thor.DataType.uint32,
        output_shape=[1, 1],
    )

    assert got.shape == expected.shape
    assert list(out_gpu.dimensions) == [1, 1]
    assert list(stamped.output().dimensions) == [1, 1]
    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmin_multi_axis_preallocated_output(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = ex.argmin(((x + 1.0) * 0.5) - y, axis=[1, 2], squeeze=False)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array(
        [
            [[4.0, 8.0], [2.0, 7.0], [5.0, 9.0]],
            [[6.0, 3.0], [1.0, 4.0], [8.0, 2.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    y_np = np.array(
        [
            [[0.0, 1.0], [3.0, 0.0], [1.0, 2.0]],
            [[2.0, 0.0], [0.0, 1.0], [3.0, 0.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    ref = ((x_np.astype(compute_dtype) + 1.0) * 0.5) - y_np.astype(compute_dtype)
    expected = np.argmin(ref.reshape(ref.shape[0], -1), axis=1).astype(np.uint32).reshape(ref.shape[0], 1, 1)

    got, _, _, _, _ = _run_staged_expr_with_preallocated(
        expr,
        ["x", "y"],
        x_np,
        y_np,
        dtype=dtype,
        output_dtype=thor.DataType.uint32,
        output_shape=[2, 1, 1],
    )

    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# Add to: bindings/python/test/core/physical/test_expression_backward_phase1.py
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_min_preallocated_output_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_min(x, axis=1, squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[3.0, 1.0, 4.0], [2.0, -5.0, 0.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.array(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    out_name = bwd_eq.output_names()[0]
    out_gpu = _gpu_tensor(list(x_np.shape), dtype, gpu_num=0)

    stamped = bwd_eq.stamp(
        inputs_gpu,
        stream,
        preallocated_outputs={
            out_name: out_gpu
        },
    )
    stamped.run()

    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_max_preallocated_output_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_max(x, axis=1, squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[3.0, 1.0, 4.0], [2.0, -5.0, 0.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    out_name = bwd_eq.output_names()[0]
    out_gpu = _gpu_tensor(list(x_np.shape), dtype, gpu_num=0)

    stamped = bwd_eq.stamp(
        inputs_gpu,
        stream,
        preallocated_outputs={
            out_name: out_gpu
        },
    )
    stamped.run()

    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_min_preallocated_output_wrong_shape_raises(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_min(x, axis=1, squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[3.0, 1.0, 4.0], [2.0, -5.0, 0.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    out_name = bwd_eq.output_names()[0]
    wrong_out = _gpu_tensor([2, 1], dtype, gpu_num=0)

    with pytest.raises(RuntimeError, match="dimensions|shape|incompatible"):
        bwd_eq.stamp(
            inputs_gpu,
            stream,
            preallocated_outputs={
                out_name: wrong_out
            },
        )
