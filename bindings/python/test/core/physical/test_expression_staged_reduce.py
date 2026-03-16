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
    # thor.DataType.fp16,
    # thor.DataType.bf16,
]


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    elif dtype in (thor.DataType.fp16, thor.DataType.bf16):
        np.testing.assert_allclose(got, expected, rtol=3e-2, atol=3e-2)
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
        dtype=dtype,
        device_num=gpu_num,
        use_fast_math=use_fast_math,
    )

    stamped = eq.stamp(input_tensors_gpu, stream)
    stamped.run()

    out_gpu = stamped.output_tensor
    out_host = PhysicalTensor(cpu_placement, out_gpu.get_descriptor())
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()

    return out_host.numpy().copy()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_reduce_sum_staged_with_prologue_and_epilogue(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    expr = ex.sqrt(((x + 3.0) * (y - 1.0)).reduce_sum(axis=1, squeeze=False))

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
    expr = (x * 2.0 + 1.0).reduce_sum(axis=1, squeeze=False)

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
    expr = (x + 1.0).reduce_sum(axis=[1, 2], squeeze=True)

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
    expr = ((x + 2.0) * 0.5).reduce_mean(axis=[1, 2], squeeze=2) + 4.0

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
    expr = (x + 1.0).reduce_prod(axis=1, squeeze=True)

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
    expr = ((x * 2.0) - y).reduce_min(axis=1, squeeze=True)

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
    expr = ex.ln((x / 2.0 + 1.0).reduce_max(axis=1, squeeze=True))

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
    expr = (x - 2.0).reduce_norm1(axis=1, squeeze=True)

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
    expr = (x - 1.5).reduce_norm2(axis=1, squeeze=True)

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

    expr = ((x + 1.0) * y).reduce_sum(axis=1, squeeze=False)
    expr = (expr + 2.0).reduce_mean(axis=0, squeeze=True)

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

    expr = ((x + 1.0) * (y - 0.5)).reduce_sum(axis=2, squeeze=False)
    expr = (expr * 0.25 + 2.0).reduce_max(axis=1, squeeze=False)
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
    expr = (x + 1.0).reduce_sum(axis=[1, 2], squeeze=[1, 2])

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
    expr = (x + 1.0).reduce_sum(axis=1, squeeze=True)

    eq = ex.compile(
        expr,
        dtype=dtype,
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
    expr = (x + 1.0).reduce_sum(axis=1, squeeze=False)

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

    eq = ex.compile(expr, dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu
    }, stream)

    assert list(stamped.output_tensor.dimensions) == [2, 1]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_stamp_output_tensor_matches_expected_shape_squeeze_true(dtype: thor.DataType):
    x = ex.input("x")
    expr = (x + 1.0).reduce_sum(axis=1, squeeze=True)

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

    eq = ex.compile(expr, dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu
    }, stream)

    assert list(stamped.output_tensor.get_descriptor().get_dimensions()) == [2]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_stamp_output_tensor_matches_expected_shape_squeeze_specific_axis(dtype: thor.DataType):
    x = ex.input("x")
    expr = (x + 1.0).reduce_sum(axis=[1, 2], squeeze=2)

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

    eq = ex.compile(expr, dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu
    }, stream)

    assert list(stamped.output_tensor.get_descriptor().get_dimensions()) == [2, 1]


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("input_shape", "reduce_axis", "squeeze", "expected_shape"),
    [
        # No squeeze: reduced dims stay as 1s
        ((2, 3, 4), 1, False, (2, 1, 4)),
        ((2, 3, 4), [1, 2], False, (2, 1, 1)),
        # ((2, 3, 4), None, False, (1, 1, 1)),

        # Squeeze all singleton dims after reduction
        ((2, 3, 4), 1, True, (2, 4)),
        ((2, 3, 4), [1, 2], True, (2,)),
        # ((2, 3, 4), None, True, (1,)),  # keep at least one dim

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

        # Reduce over singleton axes and selectively squeeze them
        # ((2, 1, 3, 1), [1, 3], False, (2, 1, 3, 1)),
        # ((2, 1, 3, 1), [1, 3], True, (2, 3)),
        # ((2, 1, 3, 1), [1, 3], [1], (2, 3, 1)),
        # ((2, 1, 3, 1), [1, 3], [3], (2, 1, 3)),
        # ((2, 1, 3, 1), [1, 3], [1, 3], (2, 3)),

        # Full reduction from a tensor already containing singleton dims
        # ((2, 1, 3, 1), None, False, (1, 1, 1, 1)),
        # ((2, 1, 3, 1), None, True, (1,)),
        # ((2, 1, 3, 1), None, [0, 1, 2, 3], (1,)),
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
    expr = (x + 1.0).reduce_sum(axis=reduce_axis, squeeze=squeeze)

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

    got = _run_staged_expr(expr, ["x"], x_np, dtype=dtype)

    assert got.shape == expected_shape
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)
