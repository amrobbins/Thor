import numpy as np
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream, Placement, DeviceType, numpy_dtypes

FLOAT_DTYPES = [
    thor.DataType.fp32,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
]

REDUCTION_DTYPES = [
    thor.DataType.fp32,
    thor.DataType.fp16,
]


def _numpy_compute_dtype(dtype: thor.DataType):
    if dtype == thor.DataType.fp32:
        return numpy_dtypes.fp32
    if dtype == thor.DataType.bf16:
        return numpy_dtypes.bf16
    return numpy_dtypes.fp16


def rtol_atol(dtype: thor.DataType) -> tuple[float, float]:
    if dtype == thor.DataType.fp32:
        return 1e-6, 1e-6
    return 5e-2, 5e-2


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    return PhysicalTensor(
        Placement(DeviceType.cpu, 0),
        PhysicalTensor.Descriptor(dtype, shape),
    )


def _fill_cpu_tensor(t: PhysicalTensor, values, dtype: thor.DataType) -> np.ndarray:
    storage_dtype = thor.physical.numpy_dtypes.from_thor(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)
    view = t.numpy()
    view[:] = np.array(values, dtype=storage_dtype).astype(compute_dtype)
    return view


def _clone_to_gpu(t: PhysicalTensor, stream: Stream) -> PhysicalTensor:
    return t.clone_copy_async(Placement(DeviceType.gpu, 0), stream)


def _clone_to_cpu(t: PhysicalTensor, stream: Stream) -> PhysicalTensor:
    return t.clone_copy_async(Placement(DeviceType.cpu, 0), stream)


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    rtol, atol = rtol_atol(dtype)
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_compile_smoke(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    compiled = outs.compile(
        dtype=dtype,
        device_num=0,
        use_fast_math=False,
    )

    assert compiled is not None


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_two_pointwise_results_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([4], dtype)
    y_cpu = _cpu_tensor([4], dtype)

    x_np_view = _fill_cpu_tensor(x_cpu, [1, 2, 3, 4], dtype)
    y_np_view = _fill_cpu_tensor(y_cpu, [5, 6, 7, 8], dtype)

    expected_sum = x_np_view + y_np_view
    expected_prod = x_np_view * y_np_view

    x_gpu = _clone_to_gpu(x_cpu, stream)
    y_gpu = _clone_to_gpu(y_cpu, stream)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    sum_cpu = _clone_to_cpu(stamped.output("sum"), stream)
    prod_cpu = _clone_to_cpu(stamped.output("prod"), stream)
    stream.synchronize()

    rtol, atol = rtol_atol(dtype)
    np.testing.assert_allclose(sum_cpu.numpy(), expected_sum, rtol=rtol, atol=atol)
    np.testing.assert_allclose(prod_cpu.numpy(), expected_prod, rtol=rtol, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_two_pointwise_results_broadcast_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 1, 4], dtype)
    y_cpu = _cpu_tensor([1, 3, 4], dtype)

    x_np_view = _fill_cpu_tensor(
        x_cpu,
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype,
    )
    y_np_view = _fill_cpu_tensor(
        y_cpu,
        [[
            [0.5, 1.0, 1.5, 2.0],
            [2.5, 3.0, 3.5, 4.0],
            [4.5, 5.0, 5.5, 6.0],
        ]],
        dtype,
    )

    expected_sum = x_np_view + y_np_view
    expected_prod = x_np_view * y_np_view

    x_gpu = _clone_to_gpu(x_cpu, stream)
    y_gpu = _clone_to_gpu(y_cpu, stream)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    sum_cpu = _clone_to_cpu(stamped.output("sum"), stream)
    prod_cpu = _clone_to_cpu(stamped.output("prod"), stream)
    stream.synchronize()

    rtol, atol = rtol_atol(dtype)
    np.testing.assert_allclose(sum_cpu.numpy(), expected_sum, rtol=rtol, atol=atol)
    np.testing.assert_allclose(prod_cpu.numpy(), expected_prod, rtol=rtol, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_shared_trunk_three_results_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    trunk = (x + y) * 2.0

    outs = ex.outputs({
        "plus_one": trunk + 1.0,
        "minus_three": trunk - 3.0,
        "square": trunk * trunk,
    })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([4], dtype)
    y_cpu = _cpu_tensor([4], dtype)

    x_np_view = _fill_cpu_tensor(x_cpu, [1.25, 1.50, 1.25, 0.75], dtype)
    y_np_view = _fill_cpu_tensor(y_cpu, [0.75, 1.00, 1.25, 1.75], dtype)

    trunk_ref = (x_np_view + y_np_view) * 2.0
    expected_plus_one = trunk_ref + 1.0
    expected_minus_three = trunk_ref - 3.0
    expected_square = (trunk_ref * trunk_ref)

    x_gpu = _clone_to_gpu(x_cpu, stream)
    y_gpu = _clone_to_gpu(y_cpu, stream)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    plus_one_cpu = _clone_to_cpu(stamped.output("plus_one"), stream)
    minus_three_cpu = _clone_to_cpu(stamped.output("minus_three"), stream)
    square_cpu = _clone_to_cpu(stamped.output("square"), stream)
    stream.synchronize()

    rtol, atol = rtol_atol(dtype)
    np.testing.assert_allclose(plus_one_cpu.numpy(), expected_plus_one, rtol=rtol, atol=atol)
    np.testing.assert_allclose(minus_three_cpu.numpy(), expected_minus_three, rtol=rtol, atol=atol)
    np.testing.assert_allclose(square_cpu.numpy(), expected_square, rtol=rtol, atol=atol)

    with pytest.raises(RuntimeError):
        stamped.output()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_disjoint_input_groups_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "left": a + b,
        "right": x * y,
    })

    stream = Stream(gpu_num=0)

    a_cpu = _cpu_tensor([4], dtype)
    b_cpu = _cpu_tensor([4], dtype)
    x_cpu = _cpu_tensor([4], dtype)
    y_cpu = _cpu_tensor([4], dtype)

    a_np = _fill_cpu_tensor(a_cpu, [1, 2, 3, 4], dtype)
    b_np = _fill_cpu_tensor(b_cpu, [10, 20, 30, 40], dtype)
    x_np = _fill_cpu_tensor(x_cpu, [2, 3, 4, 5], dtype)
    y_np = _fill_cpu_tensor(y_cpu, [6, 7, 8, 9], dtype)

    expected_left = a_np + b_np
    expected_right = x_np * y_np

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp(
        {
            "a": _clone_to_gpu(a_cpu, stream),
            "b": _clone_to_gpu(b_cpu, stream),
            "x": _clone_to_gpu(x_cpu, stream),
            "y": _clone_to_gpu(y_cpu, stream),
        },
        stream,
    )
    stamped.run()

    left_cpu = _clone_to_cpu(stamped.output("left"), stream)
    right_cpu = _clone_to_cpu(stamped.output("right"), stream)
    stream.synchronize()

    _assert_close(left_cpu.numpy(), expected_left, dtype)
    _assert_close(right_cpu.numpy(), expected_right, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_single_output_supports_output_tensor(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
    })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([3], dtype)
    y_cpu = _cpu_tensor([3], dtype)

    x_np = _fill_cpu_tensor(x_cpu, [1, 2, 3], dtype)
    y_np = _fill_cpu_tensor(y_cpu, [4, 5, 6], dtype)
    expected = x_np + y_np

    x_gpu = _clone_to_gpu(x_cpu, stream)
    y_gpu = _clone_to_gpu(y_cpu, stream)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    output_tensor_cpu = _clone_to_cpu(stamped.output(), stream)
    named_output_cpu = _clone_to_cpu(stamped.output("sum"), stream)
    stream.synchronize()

    assert isinstance(stamped.output(), PhysicalTensor)
    _assert_close(output_tensor_cpu.numpy(), expected, dtype)
    _assert_close(named_output_cpu.numpy(), expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_unknown_name_rejected_after_run(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([3], dtype)
    y_cpu = _cpu_tensor([3], dtype)
    _fill_cpu_tensor(x_cpu, [1, 2, 3], dtype)
    _fill_cpu_tensor(y_cpu, [4, 5, 6], dtype)

    x_gpu = _clone_to_gpu(x_cpu, stream)
    y_gpu = _clone_to_gpu(y_cpu, stream)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()
    stream.synchronize()

    with pytest.raises(RuntimeError):
        stamped.output("does_not_exist")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_output_stamp_missing_input_raises(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([4], dtype)
    _fill_cpu_tensor(x_cpu, [1, 2, 3, 4], dtype)
    x_gpu = _clone_to_gpu(x_cpu, stream)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)

    with pytest.raises(RuntimeError):
        eq.stamp({
            "x": x_gpu
        }, stream)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_output_stamp_unexpected_input_raises(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([4], dtype)
    y_cpu = _cpu_tensor([4], dtype)
    z_cpu = _cpu_tensor([4], dtype)

    _fill_cpu_tensor(x_cpu, [1, 2, 3, 4], dtype)
    _fill_cpu_tensor(y_cpu, [5, 6, 7, 8], dtype)
    _fill_cpu_tensor(z_cpu, [9, 10, 11, 12], dtype)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)

    with pytest.raises(RuntimeError):
        eq.stamp(
            {
                "x": _clone_to_gpu(x_cpu, stream),
                "y": _clone_to_gpu(y_cpu, stream),
                "z": _clone_to_gpu(z_cpu, stream),
            },
            stream,
        )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_stamp_wrong_input_name_raises(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([4], dtype)
    y_cpu = _cpu_tensor([4], dtype)
    _fill_cpu_tensor(x_cpu, [1, 2, 3, 4], dtype)
    _fill_cpu_tensor(y_cpu, [5, 6, 7, 8], dtype)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)

    with pytest.raises(RuntimeError):
        eq.stamp(
            {
                "x": _clone_to_gpu(x_cpu, stream),
                "b": _clone_to_gpu(y_cpu, stream),
            },
            stream,
        )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", REDUCTION_DTYPES)
def test_outputs_multiple_reductions_from_shared_trunk(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    trunk = x + y

    outs = ex.outputs(
        {
            "sum0": ex.reduce_sum(trunk, axis=0, squeeze=False),
            "max1": ex.reduce_max(trunk, axis=1, squeeze=False),
        })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 2], dtype)
    y_cpu = _cpu_tensor([2, 2], dtype)

    x_np = _fill_cpu_tensor(x_cpu, [[1, 2], [3, 4]], dtype)
    y_np = _fill_cpu_tensor(y_cpu, [[5, 6], [7, 8]], dtype)
    trunk_ref = x_np + y_np

    expected_sum0 = trunk_ref.sum(axis=0, keepdims=True)
    expected_max1 = trunk_ref.max(axis=1, keepdims=True)

    x_gpu = _clone_to_gpu(x_cpu, stream)
    y_gpu = _clone_to_gpu(y_cpu, stream)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    sum0_cpu = _clone_to_cpu(stamped.output("sum0"), stream)
    max1_cpu = _clone_to_cpu(stamped.output("max1"), stream)
    stream.synchronize()

    _assert_close(sum0_cpu.numpy(), expected_sum0, dtype)
    _assert_close(max1_cpu.numpy(), expected_max1, dtype)

    with pytest.raises(RuntimeError):
        stamped.output()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", REDUCTION_DTYPES)
def test_outputs_pointwise_and_reductions_mixed_plan(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    trunk = x + y

    outs = ex.outputs(
        {
            "trunk": trunk,
            "sum0": ex.reduce_sum(trunk, axis=0, squeeze=False),
            "max1": ex.reduce_max(trunk, axis=1, squeeze=False),
        })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 3], dtype)
    y_cpu = _cpu_tensor([2, 3], dtype)

    x_np = _fill_cpu_tensor(x_cpu, [[1, 2, 3], [4, 5, 6]], dtype)
    y_np = _fill_cpu_tensor(y_cpu, [[10, 20, 30], [40, 50, 60]], dtype)
    trunk_ref = x_np + y_np

    expected_trunk = trunk_ref
    expected_sum0 = trunk_ref.sum(axis=0, keepdims=True)
    expected_max1 = trunk_ref.max(axis=1, keepdims=True)

    x_gpu = _clone_to_gpu(x_cpu, stream)
    y_gpu = _clone_to_gpu(y_cpu, stream)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    trunk_cpu = _clone_to_cpu(stamped.output("trunk"), stream)
    sum0_cpu = _clone_to_cpu(stamped.output("sum0"), stream)
    max1_cpu = _clone_to_cpu(stamped.output("max1"), stream)
    stream.synchronize()

    _assert_close(trunk_cpu.numpy(), expected_trunk, dtype)
    _assert_close(sum0_cpu.numpy(), expected_sum0, dtype)
    _assert_close(max1_cpu.numpy(), expected_max1, dtype)

    with pytest.raises(RuntimeError):
        stamped.output()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", REDUCTION_DTYPES)
def test_outputs_reduction_squeeze_true_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    trunk = x + y

    outs = ex.outputs(
        {
            "sum_all": ex.reduce_sum(trunk, axis=1, squeeze=True),
            "max_all": ex.reduce_max(trunk, axis=0, squeeze=True),
        })

    stream = Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 3], dtype)
    y_cpu = _cpu_tensor([2, 3], dtype)

    x_np = _fill_cpu_tensor(x_cpu, [[1, 2, 3], [4, 5, 6]], dtype)
    y_np = _fill_cpu_tensor(y_cpu, [[6, 5, 4], [3, 2, 1]], dtype)
    trunk_ref = x_np + y_np

    expected_sum_all = trunk_ref.sum(axis=1)
    expected_max_all = trunk_ref.max(axis=0)

    x_gpu = _clone_to_gpu(x_cpu, stream)
    y_gpu = _clone_to_gpu(y_cpu, stream)

    eq = outs.compile(dtype=dtype, device_num=0, use_fast_math=False)
    stamped = eq.stamp({
        "x": x_gpu,
        "y": y_gpu
    }, stream)
    stamped.run()

    sum_all_cpu = _clone_to_cpu(stamped.output("sum_all"), stream)
    max_all_cpu = _clone_to_cpu(stamped.output("max_all"), stream)
    stream.synchronize()

    _assert_close(sum_all_cpu.numpy(), expected_sum_all, dtype)
    _assert_close(max_all_cpu.numpy(), expected_max_all, dtype)
