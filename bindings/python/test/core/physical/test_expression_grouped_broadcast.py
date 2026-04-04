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
    # thor.DataType.fp16,
]


def _clone_to_gpu(t: PhysicalTensor, stream: Stream) -> PhysicalTensor:
    return t.clone_copy_async(Placement(DeviceType.gpu, 0), stream)


def _clone_to_cpu(t: PhysicalTensor, stream: Stream) -> PhysicalTensor:
    return t.clone_copy_async(Placement(DeviceType.cpu, 0), stream)


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    return PhysicalTensor(
        Placement(DeviceType.cpu, 0),
        PhysicalTensor.Descriptor(dtype, shape),
    )


def _numpy_compute_dtype(dtype: thor.DataType):
    if dtype == thor.DataType.fp32:
        return numpy_dtypes.fp32
    if dtype == thor.DataType.bf16:
        return numpy_dtypes.bf16
    return numpy_dtypes.fp16


def _fill_cpu_tensor(t: PhysicalTensor, values, dtype: thor.DataType) -> np.ndarray:
    storage_dtype = thor.physical.numpy_dtypes.from_thor(dtype)
    compute_dtype = _numpy_compute_dtype(dtype)
    view = t.numpy()
    view[:] = np.array(values, dtype=storage_dtype).astype(compute_dtype)
    return view


def rtol_atol(dtype: thor.DataType) -> tuple[float, float]:
    if dtype == thor.DataType.fp32:
        return 1e-6, 1e-6
    elif dtype == thor.DataType.fp8_e4m3 or dtype == thor.DataType.fp8_e5m2:
        return 0.25, 0.25
    return 5e-2, 5e-2


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    rtol, atol = rtol_atol(dtype)
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_grouped_broadcast_mixed_domains_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    # Three outputs that remain connected by shared inputs, but resolve to different shapes:
    # xy_sum   -> [2, 3, 4]
    # xz_prod  -> [2, 1, 4]
    # y_shift  -> [1, 3, 4]
    outs = ex.outputs({
        "xy_sum": x + y,
        "xz_prod": x * z,
        "y_shift": y - 1.0,
    })

    stream = thor.physical.Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 1, 4], dtype)
    y_cpu = _cpu_tensor([1, 3, 4], dtype)
    z_cpu = _cpu_tensor([2, 1, 4], dtype)

    x_np = _fill_cpu_tensor(
        x_cpu,
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype,
    )
    y_np = _fill_cpu_tensor(
        y_cpu,
        [[
            [0.5, 1.0, 1.5, 2.0],
            [2.5, 3.0, 3.5, 4.0],
            [4.5, 5.0, 5.5, 6.0],
        ]],
        dtype,
    )
    z_np = _fill_cpu_tensor(
        z_cpu,
        [
            [[2.0, 3.0, 4.0, 5.0]],
            [[1.5, 2.5, 3.5, 4.5]],
        ],
        dtype,
    )

    expected_xy_sum = x_np + y_np
    expected_xz_prod = x_np * z_np
    expected_y_shift = y_np - 1.0

    eq = outs.compile(device_num=0, use_fast_math=False)
    stamped = eq.stamp(
        {
            "x": _clone_to_gpu(x_cpu, stream),
            "y": _clone_to_gpu(y_cpu, stream),
            "z": _clone_to_gpu(z_cpu, stream),
        },
        stream,
    )
    stamped.run()

    xy_sum_cpu = _clone_to_cpu(stamped.output("xy_sum"), stream)
    xz_prod_cpu = _clone_to_cpu(stamped.output("xz_prod"), stream)
    y_shift_cpu = _clone_to_cpu(stamped.output("y_shift"), stream)
    stream.synchronize()

    _assert_close(xy_sum_cpu.numpy(), expected_xy_sum, dtype)
    _assert_close(xz_prod_cpu.numpy(), expected_xz_prod, dtype)
    _assert_close(y_shift_cpu.numpy(), expected_y_shift, dtype)

    assert list(xy_sum_cpu.numpy().shape) == [2, 3, 4]
    assert list(xz_prod_cpu.numpy().shape) == [2, 1, 4]
    assert list(y_shift_cpu.numpy().shape) == [1, 3, 4]

    with pytest.raises(RuntimeError):
        stamped.output()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_mixed_broadcast_and_flat_domains_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    flat_trunk = x * z  # [2, 1, 4], no broadcast within this output group

    outs = ex.outputs(
        {
            "xy_sum": x + y,  # broadcasted -> [2, 3, 4]
            "flat_plus_one": flat_trunk + 1.0,  # flat domain -> [2, 1, 4]
            "flat_square": flat_trunk * flat_trunk,  # flat domain -> [2, 1, 4]
        })

    stream = thor.physical.Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 1, 4], dtype)
    y_cpu = _cpu_tensor([1, 3, 4], dtype)
    z_cpu = _cpu_tensor([2, 1, 4], dtype)

    x_np = _fill_cpu_tensor(
        x_cpu,
        [
            [[1.0, 2.0, 3.0, 2.25]],
            [[1.8, -0.2, 1.5, 2.5]],
        ],
        dtype,
    )
    y_np = _fill_cpu_tensor(
        y_cpu,
        [[
            [0.5, 1.0, 1.5, 2.0],
            [2.5, 1.75, 1.25, 0.5],
            [1.75, 2.25, 2.5, 1.8],
        ]],
        dtype,
    )
    z_np = _fill_cpu_tensor(
        z_cpu,
        [
            [[2.0, -0.05, 0.5, 0.5]],
            [[0.5, 0.5, 0.25, -0.25]],
        ],
        dtype,
    )

    flat_trunk_ref = x_np * z_np
    expected_xy_sum = x_np + y_np
    expected_flat_plus_one = flat_trunk_ref + 1.0
    expected_flat_square = flat_trunk_ref * flat_trunk_ref

    eq = outs.compile(device_num=0, use_fast_math=False)
    stamped = eq.stamp(
        {
            "x": _clone_to_gpu(x_cpu, stream),
            "y": _clone_to_gpu(y_cpu, stream),
            "z": _clone_to_gpu(z_cpu, stream),
        },
        stream,
    )
    stamped.run()

    xy_sum_cpu = _clone_to_cpu(stamped.output("xy_sum"), stream)
    flat_plus_one_cpu = _clone_to_cpu(stamped.output("flat_plus_one"), stream)
    flat_square_cpu = _clone_to_cpu(stamped.output("flat_square"), stream)
    stream.synchronize()

    _assert_close(xy_sum_cpu.numpy(), expected_xy_sum, dtype)
    _assert_close(flat_plus_one_cpu.numpy(), expected_flat_plus_one, dtype)
    _assert_close(flat_square_cpu.numpy(), expected_flat_square, dtype)

    assert list(xy_sum_cpu.numpy().shape) == [2, 3, 4]
    assert list(flat_plus_one_cpu.numpy().shape) == [2, 1, 4]
    assert list(flat_square_cpu.numpy().shape) == [2, 1, 4]

    with pytest.raises(RuntimeError):
        stamped.output()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_grouped_broadcast_same_shape_group_and_smaller_group(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    trunk = x + y

    # Two outputs share the same resolved shape [2, 3, 4], one is smaller [2, 1, 4].
    outs = ex.outputs({
        "trunk_plus_one": trunk + 1.0,
        "trunk_square": trunk * trunk,
        "xz_prod": x * z,
    })

    stream = thor.physical.Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 1, 4], dtype)
    y_cpu = _cpu_tensor([1, 3, 4], dtype)
    z_cpu = _cpu_tensor([2, 1, 4], dtype)

    x_np = _fill_cpu_tensor(
        x_cpu,
        [
            [[1.0, 0.0, 1.5, 1.5]],
            [[1.25, -0.5, 0.5, -0.75]],
        ],
        dtype,
    )
    y_np = _fill_cpu_tensor(
        y_cpu,
        [[
            [0.25, 0.50, 0.75, 1.00],
            [1.25, 1.50, 1.75, -0.50],
            [0.5, 1.25, 0.75, 1.00],
        ]],
        dtype,
    )
    z_np = _fill_cpu_tensor(
        z_cpu,
        [
            [[0.5, 0.75, 0.0, 0.25]],
            [[1.5, 1.25, 1.0, 0.75]],
        ],
        dtype,
    )

    trunk_ref = x_np + y_np
    expected_trunk_plus_one = trunk_ref + 1.0
    expected_trunk_square = trunk_ref * trunk_ref
    expected_xz_prod = x_np * z_np

    eq = outs.compile(device_num=0, use_fast_math=False)
    stamped = eq.stamp(
        {
            "x": _clone_to_gpu(x_cpu, stream),
            "y": _clone_to_gpu(y_cpu, stream),
            "z": _clone_to_gpu(z_cpu, stream),
        },
        stream,
    )
    stamped.run()

    trunk_plus_one_cpu = _clone_to_cpu(stamped.output("trunk_plus_one"), stream)
    trunk_square_cpu = _clone_to_cpu(stamped.output("trunk_square"), stream)
    xz_prod_cpu = _clone_to_cpu(stamped.output("xz_prod"), stream)
    stream.synchronize()

    _assert_close(trunk_plus_one_cpu.numpy(), expected_trunk_plus_one, dtype)
    _assert_close(trunk_square_cpu.numpy(), expected_trunk_square, dtype)
    _assert_close(xz_prod_cpu.numpy(), expected_xz_prod, dtype)

    assert list(trunk_plus_one_cpu.numpy().shape) == [2, 3, 4]
    assert list(trunk_square_cpu.numpy().shape) == [2, 3, 4]
    assert list(xz_prod_cpu.numpy().shape) == [2, 1, 4]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_grouped_broadcast_requested_shapes_match(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    outs = ex.outputs({
        "xy_sum": x + y,  # [2, 3, 4]
        "xz_prod": x * z,  # [2, 1, 4]
    })

    stream = thor.physical.Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 1, 4], dtype)
    y_cpu = _cpu_tensor([1, 3, 4], dtype)
    z_cpu = _cpu_tensor([2, 1, 4], dtype)

    x_np = _fill_cpu_tensor(
        x_cpu,
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype,
    )
    y_np = _fill_cpu_tensor(
        y_cpu,
        [[
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]],
        dtype,
    )
    z_np = _fill_cpu_tensor(
        z_cpu,
        [
            [[2.0, 2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0, 3.0]],
        ],
        dtype,
    )

    expected_xy_sum = x_np + y_np
    expected_xz_prod = x_np * z_np

    eq = outs.compile(device_num=0, use_fast_math=False)
    stamped = eq.stamp(
        {
            "x": _clone_to_gpu(x_cpu, stream),
            "y": _clone_to_gpu(y_cpu, stream),
            "z": _clone_to_gpu(z_cpu, stream),
        },
        stream,
        requested_output_shapes={
            "xy_sum": [2, 3, 4],
            "xz_prod": [2, 1, 4],
        },
    )
    stamped.run()

    xy_sum_cpu = _clone_to_cpu(stamped.output("xy_sum"), stream)
    xz_prod_cpu = _clone_to_cpu(stamped.output("xz_prod"), stream)
    stream.synchronize()

    _assert_close(xy_sum_cpu.numpy(), expected_xy_sum, dtype)
    _assert_close(xz_prod_cpu.numpy(), expected_xz_prod, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_grouped_broadcast_requested_shape_mismatch_raises(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    outs = ex.outputs({
        "xy_sum": x + y,  # [2, 3, 4]
        "xz_prod": x * z,  # [2, 1, 4]
    })

    stream = thor.physical.Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 1, 4], dtype)
    y_cpu = _cpu_tensor([1, 3, 4], dtype)
    z_cpu = _cpu_tensor([2, 1, 4], dtype)

    _fill_cpu_tensor(
        x_cpu,
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype,
    )
    _fill_cpu_tensor(
        y_cpu,
        [[
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]],
        dtype,
    )
    _fill_cpu_tensor(
        z_cpu,
        [
            [[2.0, 2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0, 3.0]],
        ],
        dtype,
    )

    eq = outs.compile(device_num=0, use_fast_math=False)

    with pytest.raises(RuntimeError):
        eq.stamp(
            {
                "x": _clone_to_gpu(x_cpu, stream),
                "y": _clone_to_gpu(y_cpu, stream),
                "z": _clone_to_gpu(z_cpu, stream),
            },
            stream,
            requested_output_shapes={
                "xy_sum": [2, 3, 4],
                "xz_prod": [2, 3, 4],  # intentionally wrong
            },
        )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", REDUCTION_DTYPES)
def test_outputs_grouped_broadcast_with_reduction_mixed_plan(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    trunk = x + y

    # trunk      -> [2, 3, 4]
    # xz_prod    -> [2, 1, 4]
    # sum_last   -> reduction of trunk over axis 2 => [2, 3, 1]
    outs = ex.outputs({
        "trunk": trunk,
        "xz_prod": x * z,
        "sum_last": ex.reduce_sum(trunk, axis=2, squeeze=False),
    })

    stream = thor.physical.Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 1, 4], dtype)
    y_cpu = _cpu_tensor([1, 3, 4], dtype)
    z_cpu = _cpu_tensor([2, 1, 4], dtype)

    x_np = _fill_cpu_tensor(
        x_cpu,
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[2.0, 3.0, 4.0, 5.0]],
        ],
        dtype,
    )
    y_np = _fill_cpu_tensor(
        y_cpu,
        [[
            [10.0, 20.0, 30.0, 40.0],
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
        ]],
        dtype,
    )
    z_np = _fill_cpu_tensor(
        z_cpu,
        [
            [[2.0, 2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0, 3.0]],
        ],
        dtype,
    )

    trunk_ref = x_np + y_np
    expected_trunk = trunk_ref
    expected_xz_prod = x_np * z_np
    expected_sum_last = trunk_ref.sum(axis=2, keepdims=True)

    eq = outs.compile(device_num=0, use_fast_math=False)
    stamped = eq.stamp(
        {
            "x": _clone_to_gpu(x_cpu, stream),
            "y": _clone_to_gpu(y_cpu, stream),
            "z": _clone_to_gpu(z_cpu, stream),
        },
        stream,
    )
    stamped.run()

    trunk_cpu = _clone_to_cpu(stamped.output("trunk"), stream)
    xz_prod_cpu = _clone_to_cpu(stamped.output("xz_prod"), stream)
    sum_last_cpu = _clone_to_cpu(stamped.output("sum_last"), stream)
    stream.synchronize()

    _assert_close(trunk_cpu.numpy(), expected_trunk, dtype)
    _assert_close(xz_prod_cpu.numpy(), expected_xz_prod, dtype)
    _assert_close(sum_last_cpu.numpy(), expected_sum_last, dtype)

    assert list(trunk_cpu.numpy().shape) == [2, 3, 4]
    assert list(xz_prod_cpu.numpy().shape) == [2, 1, 4]
    assert list(sum_last_cpu.numpy().shape) == [2, 3, 1]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", REDUCTION_DTYPES)
def test_outputs_grouped_broadcast_two_reductions_from_broadcasted_trunk(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    trunk = x + y  # [2, 3, 4]

    outs = ex.outputs(
        {
            "sum_axis1": ex.reduce_sum(trunk, axis=1, squeeze=False),  # [2, 1, 4]
            "max_axis2": ex.reduce_max(trunk, axis=2, squeeze=False),  # [2, 3, 1]
        })

    stream = thor.physical.Stream(gpu_num=0)

    x_cpu = _cpu_tensor([2, 1, 4], dtype)
    y_cpu = _cpu_tensor([1, 3, 4], dtype)

    x_np = _fill_cpu_tensor(
        x_cpu,
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype,
    )
    y_np = _fill_cpu_tensor(
        y_cpu,
        [[
            [0.5, 1.0, 1.5, 2.0],
            [2.5, 3.0, 3.5, 4.0],
            [4.5, 5.0, 5.5, 6.0],
        ]],
        dtype,
    )

    trunk_ref = x_np + y_np
    expected_sum_axis1 = trunk_ref.sum(axis=1, keepdims=True)
    expected_max_axis2 = trunk_ref.max(axis=2, keepdims=True)

    eq = outs.compile(device_num=0, use_fast_math=False)
    stamped = eq.stamp(
        {
            "x": _clone_to_gpu(x_cpu, stream),
            "y": _clone_to_gpu(y_cpu, stream),
        },
        stream,
    )
    stamped.run()

    sum_axis1_cpu = _clone_to_cpu(stamped.output("sum_axis1"), stream)
    max_axis2_cpu = _clone_to_cpu(stamped.output("max_axis2"), stream)
    stream.synchronize()

    _assert_close(sum_axis1_cpu.numpy(), expected_sum_axis1, dtype)
    _assert_close(max_axis2_cpu.numpy(), expected_max_axis2, dtype)

    assert list(sum_axis1_cpu.numpy().shape) == [2, 1, 4]
    assert list(max_axis2_cpu.numpy().shape) == [2, 3, 1]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_outputs_grouped_broadcast_different_input_shapes(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    outs = ex.outputs(
        {
            "broadcast1": x * z,  # [1,3] * [3, 3] = [3, 3]
            "broadcast2": y * z,  # [3,1] * [3, 3] = [3, 3]
        })

    stream = thor.physical.Stream(gpu_num=0)

    x_cpu = _cpu_tensor([1, 3], dtype)
    y_cpu = _cpu_tensor([3, 1], dtype)
    z_cpu = _cpu_tensor([3, 3], dtype)

    x_np = _fill_cpu_tensor(
        x_cpu,
        [
            [1.0, 1.25, 1.75],
        ],
        dtype,
    )
    y_np = _fill_cpu_tensor(
        y_cpu,
        [
            [2.0],
            [2.4],
            [3.1],
        ],
        dtype,
    )
    z_np = _fill_cpu_tensor(
        z_cpu,
        [
            [0.25, 1.6, 0.77],
            [0.82, 0.1, 2.4],
            [1.3, 2.1, 0.45],
        ],
        dtype,
    )

    expected_broadcast1 = x_np * z_np
    expected_broadcast2 = y_np * z_np

    eq = outs.compile(device_num=0, use_fast_math=False)
    stamped = eq.stamp(
        {
            "x": _clone_to_gpu(x_cpu, stream),
            "y": _clone_to_gpu(y_cpu, stream),
            "z": _clone_to_gpu(z_cpu, stream),
        },
        stream,
    )
    stamped.run()

    broadcast1_cpu = _clone_to_cpu(stamped.output("broadcast1"), stream)
    broadcast2_cpu = _clone_to_cpu(stamped.output("broadcast2"), stream)
    stream.synchronize()

    _assert_close(broadcast1_cpu.numpy(), expected_broadcast1, dtype)
    _assert_close(broadcast2_cpu.numpy(), expected_broadcast2, dtype)

    assert list(broadcast1_cpu.numpy().shape) == [3, 3]
    assert list(broadcast2_cpu.numpy().shape) == [3, 3]
