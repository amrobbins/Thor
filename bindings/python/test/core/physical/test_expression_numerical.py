import math
import numpy as np
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


def _copy_numpy_to_gpu(
        values: np.ndarray,
        stream: thor.physical.Stream,
        dtype: thor.DataType,
        gpu_num: int = 0) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=_numpy_storage_dtype(dtype), order="C")
    cpu = _cpu_tensor(list(values.shape), dtype)
    gpu = _gpu_tensor(list(values.shape), dtype, gpu_num=gpu_num)

    cpu_view = _cpu_numpy_view(cpu, dtype)
    cpu_view[...] = values
    gpu.copy_from_async(cpu, stream)
    return gpu


def _copy_gpu_to_numpy(t: thor.physical.PhysicalTensor, dtype: thor.DataType, stream) -> np.ndarray:
    cpu = _cpu_tensor(list(t.get_descriptor().get_dimensions()), dtype)
    cpu.copy_from_async(t, stream)
    stream.synchronize()
    return np.array(_cpu_numpy_view(cpu, dtype), copy=True)


def _run_expr(
    expr,
    input_names: list[str],
    *inputs: np.ndarray,
    dtype: thor.DataType,
    gpu_num: int = 0,
    use_fast_math: bool = False,
) -> np.ndarray:
    assert len(inputs) >= 1
    first_shape = tuple(inputs[0].shape)
    for arr in inputs:
        assert tuple(arr.shape) == first_shape
        assert arr.dtype == _numpy_storage_dtype(dtype)

    eq = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=use_fast_math,
    )

    stream = thor.physical.Stream(gpu_num)
    gpu_inputs = {
        input_names[i]: _copy_numpy_to_gpu(inputs[i], stream, dtype, gpu_num=gpu_num) for i in range(len(inputs))
    }
    gpu_output = _gpu_tensor(list(first_shape), dtype, gpu_num=gpu_num)

    eq.run(gpu_inputs, gpu_output, stream)
    return _copy_gpu_to_numpy(gpu_output, dtype, stream)


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
def test_add_sub_mul_div_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = ((x + y) - 2.0) * (x / (y + 1.0))

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1, 2, 3, 4, 5, 6], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([2, 3, 4, 5, 6, 7], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    def to_storage(arr: np.ndarray) -> np.ndarray:
        return arr.astype(storage_dtype)

    # Model Thor's semantics:
    # each node computes in compute_dtype, then materializes to output_dtype (= storage dtype here).
    sum_ref = to_storage(x_np.astype(compute_dtype) + y_np.astype(compute_dtype))
    minus_ref = to_storage(sum_ref.astype(compute_dtype) - 2.0)
    denom_ref = to_storage(y_np.astype(compute_dtype) + 1.0)
    div_ref = to_storage(x_np.astype(compute_dtype) / denom_ref.astype(compute_dtype))
    expected = to_storage(minus_ref.astype(compute_dtype) * div_ref.astype(compute_dtype))

    got = _run_expr(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pow_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = x**y

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1.5, 2.0, 3.0, 4.0, 5.5], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([2.0, 3.0, 1.5, 0.5, 2.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected = np.power(x_ref, y_ref)
    got = _run_expr(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rpow_numerical(dtype: thor.DataType):
    x = ex.input("x")
    expr = 2.0**x

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    x_ref = x_np.astype(compute_dtype)

    expected = np.power(2.0, x_ref)
    got = _run_expr(expr, ['x'], x_np, dtype=dtype)

    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_negation_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = -(x**y) + 3.0

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([2.0, 2.0, 2.0, 2.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected = -(np.power(x_ref, y_ref)) + 3.0
    got = _run_expr(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_max_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = ex.max(ex.min(x, y), 3.0)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1, 5, 2, 8, 4, 6], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([2, 4, 7, 1, 3, 9], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected = np.maximum(np.minimum(x_ref, y_ref), 3.0)
    got = _run_expr(expr, ['x', 'y'], x_np, y_np, dtype=dtype)

    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_exp_family_numerical(dtype: thor.DataType):
    x = ex.input("x")

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    x_ref = x_np.astype(compute_dtype)

    got_exp = _run_expr(ex.exp(x), ['x'], x_np, dtype=dtype)
    got_exp2 = _run_expr(ex.exp2(x), ['x'], x_np, dtype=dtype)
    got_exp10 = _run_expr(ex.exp10(x), ['x'], x_np, dtype=dtype)

    _assert_close(got_exp, np.exp(x_ref), dtype)
    _assert_close(got_exp2, np.exp2(x_ref), dtype)
    _assert_close(got_exp10, np.power(10.0, x_ref), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_log_family_numerical(dtype: thor.DataType):
    x = ex.input("x")

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([0.25, 0.5, 1.0, 2.0, 8.0, 10.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    x_ref = x_np.astype(compute_dtype)

    got_ln = _run_expr(ex.ln(x), ['x'], x_np, dtype=dtype)
    got_log_default = _run_expr(ex.log(x), ['x'], x_np, dtype=dtype)
    got_log2 = _run_expr(ex.log2(x), ['x'], x_np, dtype=dtype)
    got_log10 = _run_expr(ex.log10(x), ['x'], x_np, dtype=dtype)
    got_log_base7 = _run_expr(ex.log(x, 7.0), ['x'], x_np, dtype=dtype)

    _assert_close(got_ln, np.log(x_ref), dtype)
    _assert_close(got_log_default, np.log(x_ref), dtype)
    _assert_close(got_log2, np.log2(x_ref), dtype)
    _assert_close(got_log10, np.log10(x_ref), dtype)
    _assert_close(got_log_base7, np.log(x_ref) / math.log(7.0), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sqrt_numerical(dtype: thor.DataType):
    x = ex.input("x")
    expr = ex.sqrt(x)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([0.0, 0.25, 1.0, 4.0, 9.0, 16.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    x_ref = x_np.astype(compute_dtype)

    expected = np.sqrt(x_ref)
    got = _run_expr(expr, ['x'], x_np, dtype=dtype)

    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scalar_only_expression_numerical(dtype: thor.DataType):
    expr = ex.exp(ex.constant_scalar(2.0)) + ex.log2(ex.constant_scalar(8.0)) - ex.sqrt(ex.constant_scalar(9.0))

    # Need a tensor shape for execution. Use a dummy input expression plus a zero multiplier
    # so the result stays scalar-valued elementwise.
    x = ex.input("x")
    lifted_expr = (x * 0.0) + expr

    compute_dtype = _numpy_compute_dtype(thor.DataType.fp32)

    x_np = np.zeros((6,), dtype=compute_dtype)
    expected_value = np.exp(compute_dtype(2.0)) + np.log2(compute_dtype(8.0)) - np.sqrt(compute_dtype(9.0))
    expected = np.full(x_np.shape, expected_value, dtype=compute_dtype)

    # There are no input tensors, just scalars, so they default to fp32
    got = _run_expr(lifted_expr, ['x'], x_np, dtype=thor.DataType.fp32)
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nested_expression_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    expr = ex.max(
        ex.sqrt(ex.exp2((x + 3.0) * (y - 1.0))),
        ex.min((z / 2.0)**2.0, ex.log2(y + 8.0)),
    )

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([0.5, 1.0, 1.5, 2.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([2.0, 3.0, 4.0, 1.3], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    z_np = np.array([2.0, 4.0, -0.9, 0.5], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)
    z_ref = z_np.astype(compute_dtype)

    expected = np.maximum(
        np.sqrt(np.exp2((x_ref + 3.0) * (y_ref - 1.0))),
        np.minimum(np.power(z_ref / 2.0, 2.0), np.log2(y_ref + 8.0)),
    )
    got = _run_expr(expr, ['x', 'y', 'z'], x_np, y_np, z_np, dtype=dtype)

    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("use_fast_math", [False, True])
def test_fast_math_toggle_numerical(dtype: thor.DataType, use_fast_math: bool):
    x = ex.input("x")
    y = ex.input("y")
    expr = ex.exp(ex.log2(x + 8.0) + (y**2.0) / 3.0)

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)

    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)
    y_np = np.array([1.0, 1.25, 1.5, 1.75], dtype=thor.physical.numpy_dtypes.fp32).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    got = _run_expr(expr, ['x', 'y'], x_np, y_np, use_fast_math=use_fast_math, dtype=dtype)
    expected = np.exp(np.log2(x_ref + 8.0) + np.power(y_ref, 2.0) / 3.0)

    rtol, atol = _rtol_atol(dtype)
    if dtype == thor.DataType.bf16:
        rtol = max(rtol, 3e-2)
        atol = max(atol, 3e-2)

    if use_fast_math and dtype != thor.DataType.fp32:
        rtol = max(rtol, 5e-2)
        atol = max(atol, 3e-2)

    _assert_close(got, expected, dtype, rtol=rtol, atol=atol)


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

    x_init = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=thor.physical.numpy_dtypes.fp32)
    y_init = np.array([[1.1, 1.2, 1.3, 1.4], [1.5, 1.2, 1.1, 1.3]], dtype=thor.physical.numpy_dtypes.fp32)
    z_init = np.array([[1.0, 1.5, 2.0, 2.5], [0.8, 1.2, 1.6, 2.0]], dtype=thor.physical.numpy_dtypes.fp32)

    host_np_dtype = _numpy_storage_dtype(dtype)
    x_np_view[:] = x_init.astype(host_np_dtype)
    y_np_view[:] = y_init.astype(host_np_dtype)
    z_np_view[:] = z_init.astype(host_np_dtype)

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

    compute_np_dtype = _numpy_compute_dtype(dtype)
    x_ref = x_np_view.astype(compute_np_dtype)
    y_ref = y_np_view.astype(compute_np_dtype)
    z_ref = z_np_view.astype(compute_np_dtype)

    expected = np.maximum(
        np.sqrt(np.exp2((x_ref + 3.0) * (y_ref - 1.0))),
        np.minimum((z_ref / 2.0)**2.0, np.log2(y_ref + 8.0)),
    )

    expected = expected.astype(_numpy_storage_dtype(dtype))

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

    rtol, atol = _rtol_atol(dtype)
    np.testing.assert_allclose(out_cpu.numpy(), expected, rtol=rtol, atol=atol)


@pytest.mark.cuda
def test_fused_equation_stamp_as_type_single_output_requested_shape_accepts_squeezed_layout():
    x = ex.input("x", output_dtype=thor.DataType.fp16)
    expr = ex.reduce_sum(x, axis=2, squeeze=False)  # actual logical output [2, 3, 1]

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    stream = thor.physical.Stream(gpu_num=0)

    # Requested layout [2, 3] should be accepted because singleton dims are ignored.
    stamped = eq.stamp(
        {
            "x": x_gpu,
        },
        stream,
        requested_output_shape=[2, 3],
    )

    assert stamped is not None


@pytest.mark.cuda
def test_fused_equation_stamp_as_type_single_output_requested_shape_rejects_incompatible_layout():
    x = ex.input("x", output_dtype=thor.DataType.fp16)
    expr = ex.reduce_sum(x, axis=2, squeeze=False)  # actual logical output [2, 3, 1]

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    stream = thor.physical.Stream(gpu_num=0)

    with pytest.raises(RuntimeError, match="Output tensor dimensions are incompatible"):
        eq.stamp(
            {
                "x": x_gpu,
            },
            stream,
            requested_output_shape=[2, 4],
        )


@pytest.mark.cuda
def test_fused_equation_stamp_as_type_multi_output_requested_shapes_with_broadcast_accepts_compatible_layouts():
    x = ex.input("x", output_dtype=thor.DataType.fp16)
    y = ex.input("y")

    xy = x + y
    y_shift = y - 0.5

    outs = ex.outputs(
        {
            "wide_sum": xy,  # [17, 3]
            "wide_mix": xy * y_shift,  # [17, 3]
            "narrow_shift": x + 1.0,  # [17, 1]
        })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([17, 1], thor.DataType.fp32)
    y_gpu = _gpu_tensor([1, 3], thor.DataType.fp32)
    stream = thor.physical.Stream(gpu_num=0)

    stamped = eq.stamp(
        {
            "x": x_gpu,
            "y": y_gpu,
        },
        stream,
        requested_output_shapes={
            "wide_sum": [17, 3],
            "wide_mix": [17, 3],
            "narrow_shift": [17],  # compatible with logical [17, 1]
        },
    )

    assert stamped is not None


@pytest.mark.cuda
def test_fused_equation_stamp_as_type_single_output_requested_shape_accepts_compatible_layout():
    x = ex.input("x", output_dtype=thor.DataType.fp16)
    expr = ex.reduce_sum(x, axis=2, squeeze=False)  # logical output [2, 3, 1]

    eq = ex.compile(expr, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    stream = thor.physical.Stream(gpu_num=0)

    stamped = eq.stamp(
        {
            "x": x_gpu,
        },
        stream,
        requested_output_shape=[2, 3, 1],
    )

    assert list(stamped.output().dimensions) == [2, 3, 1]


@pytest.mark.cuda
def test_fused_equation_stamp_as_type_multi_output_requested_shapes_with_broadcast_rejects_incompatible_layout():
    x = ex.input("x", output_dtype=thor.DataType.fp16)
    y = ex.input("y")

    xy = x + y

    outs = ex.outputs({
        "wide_sum": xy,  # [17, 3]
        "narrow_shift": x + 1.0,  # [17, 1]
    })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([17, 1], thor.DataType.fp32)
    y_gpu = _gpu_tensor([1, 3], thor.DataType.fp32)
    stream = thor.physical.Stream(gpu_num=0)

    with pytest.raises(RuntimeError, match="Output tensor dimensions are incompatible"):
        eq.stamp(
            {
                "x": x_gpu,
                "y": y_gpu,
            },
            stream,
            requested_output_shapes={
                "wide_sum": [17, 4],  # incompatible
                "narrow_shift": [17],
            },
        )


@pytest.mark.cuda
def test_fused_equation_stamp_as_type_multi_output_requested_shapes_with_reduction_and_epilogue_accepts_compatible_layouts(
):
    x = ex.input("x", output_dtype=thor.DataType.fp16)
    y = ex.input("y")

    trunk = (x + 1.0) * (y - 0.5)

    outs = ex.outputs(
        {
            "reduced": ex.reduce_sum(trunk, axis=2, squeeze=False),  # [2, 3, 1]
            "final": ex.sqrt(ex.reduce_sum(trunk, axis=2, squeeze=False) + 1.0),  # [2, 3, 1]
            "pointwise": x + 2.0,  # [2, 3, 4]
        })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    y_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    stream = thor.physical.Stream(gpu_num=0)

    stamped = eq.stamp(
        {
            "x": x_gpu,
            "y": y_gpu,
        },
        stream,
        requested_output_shapes={
            "reduced": [2, 3],
            "final": [2, 3],
            "pointwise": [2, 3, 4],
        },
    )

    assert stamped is not None


@pytest.mark.cuda
def test_fused_equation_stamp_as_type_multi_output_requested_shapes_with_reduction_and_epilogue_rejects_incompatible_layout(
):
    x = ex.input("x", output_dtype=thor.DataType.fp16)
    y = ex.input("y")

    trunk = (x + 1.0) * (y - 0.5)

    outs = ex.outputs(
        {
            "reduced": ex.reduce_sum(trunk, axis=2, squeeze=False),  # [2, 3, 1]
            "final": ex.sqrt(ex.reduce_sum(trunk, axis=2, squeeze=False) + 1.0),  # [2, 3, 1]
            "pointwise": x + 2.0,  # [2, 3, 4]
        })

    eq = ex.compile(outs, device_num=0, use_fast_math=False)

    x_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    y_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)
    stream = thor.physical.Stream(gpu_num=0)

    with pytest.raises(RuntimeError, match="Output tensor dimensions are incompatible"):
        eq.stamp(
            {
                "x": x_gpu,
                "y": y_gpu,
            },
            stream,
            requested_output_shapes={
                "reduced": [2, 4],  # incompatible with logical [2, 3, 1]
                "final": [2, 3],
                "pointwise": [2, 3, 4],
            },
        )


def _host_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu = Placement(DeviceType.cpu, 0)
    gpu = Placement(DeviceType.gpu, gpu_num)
    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
    host = PhysicalTensor(cpu, desc)
    host.numpy()[...] = arr
    device = PhysicalTensor(gpu, desc)
    device.copy_from_async(host, stream)
    return device


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_abs_forward_numerical(dtype: thor.DataType):
    x = ex.input("x")
    out = ex.abs(x)

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[-2.0, 0.0, 3.5], [4.25, -5.75, 0.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    expected = np.abs(x_np.astype(np.float32)).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output()
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)
