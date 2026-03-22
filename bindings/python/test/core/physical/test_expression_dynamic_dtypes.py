import numpy as np
import pytest
import thor
from thor.physical import Expression as ex
from thor.physical import PhysicalTensor, Stream, Placement, DeviceType, numpy_dtypes

ALL_FLOAT_DTYPES = [
    thor.DataType.fp32,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
]

VECTORIZABLE_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
]


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


def _default_compute_dtype(dtype: thor.DataType) -> thor.DataType:
    if dtype in (thor.DataType.fp8_e4m3, thor.DataType.fp8_e5m2):
        return thor.DataType.fp16
    return dtype


def _promote_tensor_value_dtype(a: thor.DataType, b: thor.DataType) -> thor.DataType:
    if a == b:
        return a

    if {a, b} == {thor.DataType.fp16, thor.DataType.bf16}:
        return thor.DataType.fp32

    if thor.DataType.fp32 in (a, b):
        return thor.DataType.fp32

    if thor.DataType.bf16 in (a, b):
        return thor.DataType.bf16

    if thor.DataType.fp16 in (a, b):
        return thor.DataType.fp16

    # Mixed fp8 families promote to fp16.
    return thor.DataType.fp16


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    elif dtype == thor.DataType.fp16:
        np.testing.assert_allclose(got, expected, rtol=3e-2, atol=3e-2)
    elif dtype == thor.DataType.bf16:
        np.testing.assert_allclose(got, expected, rtol=4e-2, atol=4e-2)
    elif dtype == thor.DataType.fp8_e4m3:
        np.testing.assert_allclose(got, expected, rtol=2.5e-1, atol=2.5e-1)
    elif dtype == thor.DataType.fp8_e5m2:
        np.testing.assert_allclose(got, expected, rtol=3.5e-1, atol=3.5e-1)
    else:
        raise AssertionError(f"Unhandled dtype: {dtype}")


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


def _run_expr(expr, inputs: dict[str, tuple[np.ndarray, thor.DataType]], gpu_num: int = 0, use_fast_math: bool = False):
    stream = Stream(gpu_num=gpu_num)

    gpu_inputs = {}
    for name, (arr, dtype) in inputs.items():
        gpu_inputs[name] = _copy_numpy_to_gpu(arr, dtype, stream, gpu_num=gpu_num)

    eq = ex.compile(
        expr,
        device_num=gpu_num,
        use_fast_math=use_fast_math,
    )

    stamped = eq.stamp(gpu_inputs, stream)
    stamped.run()

    out_gpu = stamped.output()
    out_np = _copy_gpu_to_numpy(out_gpu, stream)
    return out_np


def _run_outputs(
        outs, inputs: dict[str, tuple[np.ndarray, thor.DataType]], gpu_num: int = 0, use_fast_math: bool = False):
    stream = Stream(gpu_num=gpu_num)

    gpu_inputs = {}
    for name, (arr, dtype) in inputs.items():
        gpu_inputs[name] = _copy_numpy_to_gpu(arr, dtype, stream, gpu_num=gpu_num)

    eq = ex.compile(
        outs,
        device_num=gpu_num,
        use_fast_math=use_fast_math,
    )

    stamped = eq.stamp(gpu_inputs, stream)
    stamped.run()

    result = {}
    names = eq.output_names()
    if len(names) == 1:
        result[names[0]] = _copy_gpu_to_numpy(stamped.output(), stream)
    else:
        for name in names:
            result[name] = _copy_gpu_to_numpy(stamped.output(name), stream)

    return result


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES)
def test_dynamic_dtype_single_input_defaults_follow_tensor_storage_policy(dtype: thor.DataType):
    x = ex.input("x")
    expr = ((x + 1.0) * (x - 0.5)) + 2.0

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(_default_compute_dtype(dtype))

    x_np = np.linspace(-2.0, 2.0, num=17, dtype=np.float32).reshape(17).astype(storage_dtype)
    x_ref = x_np.astype(compute_dtype)

    expected = (((x_ref + 1.0) * (x_ref - 0.5)) + 2.0).astype(storage_dtype)

    got = _run_expr(expr, {
        "x": (x_np, dtype)
    })

    assert got.dtype == storage_dtype
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("lhs_dtype", "rhs_dtype", "expected_output_dtype"),
    [
        (thor.DataType.fp8_e4m3, thor.DataType.fp8_e4m3, thor.DataType.fp8_e4m3),
        (thor.DataType.fp8_e4m3, thor.DataType.fp8_e5m2, thor.DataType.fp16),
        (thor.DataType.fp8_e4m3, thor.DataType.fp16, thor.DataType.fp16),
        (thor.DataType.fp8_e5m2, thor.DataType.bf16, thor.DataType.bf16),
        (thor.DataType.fp16, thor.DataType.bf16, thor.DataType.fp32),
        (thor.DataType.fp16, thor.DataType.fp32, thor.DataType.fp32),
        (thor.DataType.bf16, thor.DataType.fp32, thor.DataType.fp32),
    ],
)
def test_dynamic_dtype_pairwise_tensor_promotion(
        lhs_dtype: thor.DataType, rhs_dtype: thor.DataType, expected_output_dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = x + y

    lhs_storage = _numpy_storage_dtype(lhs_dtype)
    rhs_storage = _numpy_storage_dtype(rhs_dtype)
    out_storage = _numpy_storage_dtype(expected_output_dtype)
    out_compute = _numpy_compute_dtype(_default_compute_dtype(expected_output_dtype))

    x_np = np.linspace(-1.0, 1.0, num=15, dtype=np.float32).reshape(3, 5).astype(lhs_storage)
    y_np = np.linspace(0.25, 2.25, num=15, dtype=np.float32).reshape(3, 5).astype(rhs_storage)

    expected = (x_np.astype(out_compute) + y_np.astype(out_compute)).astype(out_storage)

    got = _run_expr(expr, {
        "x": (x_np, lhs_dtype),
        "y": (y_np, rhs_dtype)
    })

    assert got.dtype == out_storage
    assert got.shape == expected.shape
    _assert_close(got, expected, expected_output_dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES)
def test_constants_do_not_affect_tensor_promotion(dtype: thor.DataType):
    x = ex.input("x")
    expr = (x + 1.0) * 2.0 - 0.25

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(_default_compute_dtype(dtype))

    x_np = np.linspace(0.0, 3.0, num=11, dtype=np.float32).reshape(11).astype(storage_dtype)
    expected = (((x_np.astype(compute_dtype) + 1.0) * 2.0) - 0.25).astype(storage_dtype)

    got = _run_expr(expr, {
        "x": (x_np, dtype)
    })

    assert got.dtype == storage_dtype
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_mixed_dtype_multi_output_fused_stage_produces_distinct_output_dtypes_and_values():
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")

    outs = ex.outputs(
        {
            "sum16": x + y,  # fp16
            "mix32": x + z,  # fp32
            "shift16": x + 1.0,  # fp16; constant must not promote
        })

    x_dtype = thor.DataType.fp16
    y_dtype = thor.DataType.fp16
    z_dtype = thor.DataType.fp32

    x_np = np.linspace(-1.0, 1.0, num=17, dtype=np.float32).reshape(17).astype(_numpy_storage_dtype(x_dtype))
    y_np = np.linspace(2.0, 4.0, num=17, dtype=np.float32).reshape(17).astype(_numpy_storage_dtype(y_dtype))
    z_np = np.linspace(-0.5, 0.5, num=17, dtype=np.float32).reshape(17).astype(_numpy_storage_dtype(z_dtype))

    expected_sum16 = (
        x_np.astype(_numpy_compute_dtype(thor.DataType.fp16)) +
        y_np.astype(_numpy_compute_dtype(thor.DataType.fp16))).astype(_numpy_storage_dtype(thor.DataType.fp16))

    expected_mix32 = (
        x_np.astype(_numpy_compute_dtype(thor.DataType.fp32)) +
        z_np.astype(_numpy_compute_dtype(thor.DataType.fp32))).astype(_numpy_storage_dtype(thor.DataType.fp32))

    expected_shift16 = (x_np.astype(_numpy_compute_dtype(thor.DataType.fp16)) + 1.0).astype(
        _numpy_storage_dtype(thor.DataType.fp16))

    got = _run_outputs(
        outs,
        {
            "x": (x_np, x_dtype),
            "y": (y_np, y_dtype),
            "z": (z_np, z_dtype),
        },
    )

    assert got["sum16"].dtype == _numpy_storage_dtype(thor.DataType.fp16)
    assert got["mix32"].dtype == _numpy_storage_dtype(thor.DataType.fp32)
    assert got["shift16"].dtype == _numpy_storage_dtype(thor.DataType.fp16)

    _assert_close(got["sum16"], expected_sum16, thor.DataType.fp16)
    _assert_close(got["mix32"], expected_mix32, thor.DataType.fp32)
    _assert_close(got["shift16"], expected_shift16, thor.DataType.fp16)


@pytest.mark.cuda
def test_mixed_dtype_staged_reduction_and_epilogue_outputs_preserve_expected_dtypes_and_values():
    x = ex.input("x")
    y = ex.input("y")

    trunk = (x + 1.0) * (y - 0.5)  # x fp16 + y fp32 -> trunk fp32

    outs = ex.outputs(
        {
            "reduced": ex.reduce_sum(trunk, axis=2, squeeze=False),  # fp32
            "final": ex.sqrt(ex.reduce_sum(trunk, axis=2, squeeze=False) + 1.0),  # fp32
            "pointwise": x + 2.0,  # fp16
        })

    x_dtype = thor.DataType.fp16
    y_dtype = thor.DataType.fp32

    x_np = np.linspace(0.25, 2.25, num=24, dtype=np.float32).reshape(2, 3, 4).astype(_numpy_storage_dtype(x_dtype))
    y_np = np.linspace(1.0, 3.0, num=24, dtype=np.float32).reshape(2, 3, 4).astype(_numpy_storage_dtype(y_dtype))

    x_plus_one_ref = (x_np.astype(_numpy_compute_dtype(thor.DataType.fp16)) + 1.0).astype(
        _numpy_storage_dtype(thor.DataType.fp16))
    y_minus_half_ref = (y_np.astype(_numpy_compute_dtype(thor.DataType.fp32)) - 0.5).astype(
        _numpy_storage_dtype(thor.DataType.fp32))

    trunk_ref = (
        x_plus_one_ref.astype(_numpy_compute_dtype(thor.DataType.fp32)) *
        y_minus_half_ref.astype(_numpy_compute_dtype(thor.DataType.fp32)))

    expected_reduced = np.sum(trunk_ref, axis=2, keepdims=True).astype(_numpy_storage_dtype(thor.DataType.fp32))
    expected_final = np.sqrt(np.sum(trunk_ref, axis=2, keepdims=True) + 1.0).astype(
        _numpy_storage_dtype(thor.DataType.fp32))
    expected_pointwise = (x_np.astype(_numpy_compute_dtype(thor.DataType.fp16)) + 2.0).astype(
        _numpy_storage_dtype(thor.DataType.fp16))

    got = _run_outputs(
        outs,
        {
            "x": (x_np, x_dtype),
            "y": (y_np, y_dtype),
        },
    )

    assert got["reduced"].dtype == _numpy_storage_dtype(thor.DataType.fp32)
    assert got["final"].dtype == _numpy_storage_dtype(thor.DataType.fp32)
    assert got["pointwise"].dtype == _numpy_storage_dtype(thor.DataType.fp16)

    assert got["reduced"].shape == (2, 3, 1)
    assert got["final"].shape == (2, 3, 1)
    assert got["pointwise"].shape == (2, 3, 4)

    _assert_close(got["reduced"], expected_reduced, thor.DataType.fp32)
    _assert_close(got["final"], expected_final, thor.DataType.fp32)
    _assert_close(got["pointwise"], expected_pointwise, thor.DataType.fp16)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", VECTORIZABLE_DTYPES)
def test_vectorized_flat_homogeneous_stage_handles_odd_numel_tail(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    expr = ((x + y) * (x - 0.5)) + 1.0

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(_default_compute_dtype(dtype))

    # 17 elements -> odd packed tail
    x_np = np.linspace(-1.0, 1.0, num=17, dtype=np.float32).reshape(17).astype(storage_dtype)
    y_np = np.linspace(0.25, 1.25, num=17, dtype=np.float32).reshape(17).astype(storage_dtype)

    expected = (((x_np.astype(compute_dtype) + y_np.astype(compute_dtype)) * (x_np.astype(compute_dtype) - 0.5)) +
                1.0).astype(storage_dtype)

    got = _run_expr(expr, {
        "x": (x_np, dtype),
        "y": (y_np, dtype)
    })

    assert got.dtype == storage_dtype
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", VECTORIZABLE_DTYPES)
def test_vectorized_flat_multi_output_homogeneous_stage_handles_odd_numel_tail(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
        "shifted": x + 1.0,
    })

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(_default_compute_dtype(dtype))

    x_np = np.linspace(0.1, 1.7, num=17, dtype=np.float32).reshape(17).astype(storage_dtype)
    y_np = np.linspace(1.0, 2.6, num=17, dtype=np.float32).reshape(17).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected = {
        "sum": (x_ref + y_ref).astype(storage_dtype),
        "prod": (x_ref * y_ref).astype(storage_dtype),
        "shifted": (x_ref + 1.0).astype(storage_dtype),
    }

    got = _run_outputs(outs, {
        "x": (x_np, dtype),
        "y": (y_np, dtype)
    })

    for name in expected:
        assert got[name].dtype == storage_dtype
        assert got[name].shape == expected[name].shape
        _assert_close(got[name], expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", VECTORIZABLE_DTYPES)
def test_vectorized_specialized_broadcast_same_domain_multi_output_handles_odd_numel_tail(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "sum": x + y,
        "prod": x * y,
    })

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(_default_compute_dtype(dtype))

    # Broadcast result shape [17, 3] => 51 elements, odd tail
    x_np = np.linspace(0.25, 2.25, num=17, dtype=np.float32).reshape(17, 1).astype(storage_dtype)
    y_np = np.linspace(1.0, 1.5, num=3, dtype=np.float32).reshape(1, 3).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected = {
        "sum": (x_ref + y_ref).astype(storage_dtype),
        "prod": (x_ref * y_ref).astype(storage_dtype),
    }

    got = _run_outputs(outs, {
        "x": (x_np, dtype),
        "y": (y_np, dtype)
    })

    for name in expected:
        assert got[name].dtype == storage_dtype
        assert got[name].shape == expected[name].shape
        _assert_close(got[name], expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", VECTORIZABLE_DTYPES)
def test_vectorized_specialized_broadcast_grouped_domains_handles_odd_numel_tails(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs(
        {
            "xy_sum": x + y,  # [17, 3] => 51 elements
            "x_shift": x + 1.0,  # [17, 1] => 17 elements
            "y_shift": y - 2.0,  # [1, 3] => 3 elements
        })

    storage_dtype = _numpy_storage_dtype(dtype)
    compute_dtype = _numpy_compute_dtype(_default_compute_dtype(dtype))

    x_np = np.linspace(0.25, 2.25, num=17, dtype=np.float32).reshape(17, 1).astype(storage_dtype)
    y_np = np.linspace(1.0, 1.5, num=3, dtype=np.float32).reshape(1, 3).astype(storage_dtype)

    x_ref = x_np.astype(compute_dtype)
    y_ref = y_np.astype(compute_dtype)

    expected = {
        "xy_sum": (x_ref + y_ref).astype(storage_dtype),
        "x_shift": (x_ref + 1.0).astype(storage_dtype),
        "y_shift": (y_ref - 2.0).astype(storage_dtype),
    }

    got = _run_outputs(outs, {
        "x": (x_np, dtype),
        "y": (y_np, dtype)
    })

    for name in expected:
        assert got[name].dtype == storage_dtype
        assert got[name].shape == expected[name].shape
        _assert_close(got[name], expected[name], dtype)


@pytest.mark.cuda
def test_mixed_dtype_stage_with_odd_numel_runs_correctly_even_when_not_vectorizable():
    x = ex.input("x")
    y = ex.input("y")
    expr = ((x + y) * 1.25) - 0.75  # fp16 + fp32 => fp32 scalar fallback

    x_dtype = thor.DataType.fp16
    y_dtype = thor.DataType.fp32
    out_dtype = thor.DataType.fp32

    x_np = np.linspace(-1.0, 1.0, num=17, dtype=np.float32).reshape(17).astype(_numpy_storage_dtype(x_dtype))
    y_np = np.linspace(2.0, 3.0, num=17, dtype=np.float32).reshape(17).astype(_numpy_storage_dtype(y_dtype))

    expected = (
        ((x_np.astype(_numpy_compute_dtype(out_dtype)) + y_np.astype(_numpy_compute_dtype(out_dtype))) * 1.25) -
        0.75).astype(_numpy_storage_dtype(out_dtype))

    got = _run_expr(expr, {
        "x": (x_np, x_dtype),
        "y": (y_np, y_dtype)
    })

    assert got.dtype == _numpy_storage_dtype(out_dtype)
    assert got.shape == expected.shape
    _assert_close(got, expected, out_dtype)
