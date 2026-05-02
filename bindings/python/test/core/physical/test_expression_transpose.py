import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

FLOAT_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
    thor.DataType.fp32,
]

BACKWARD_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp32,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
]


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _cast_reference_to_storage_dtype_with_saturation(values: np.ndarray, dtype: thor.DataType) -> np.ndarray:
    values32 = values.astype(np.float32)
    storage_dtype = _numpy_storage_dtype(dtype)
    if dtype == thor.DataType.fp8_e4m3:
        values32 = np.clip(values32, -448.0, 448.0)
    return values32.astype(storage_dtype)


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
        np.testing.assert_allclose(got32, expected32, rtol=2e-1, atol=5e-1)
    else:
        raise AssertionError(f"Unhandled dtype: {dtype}")


def _host_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu = Placement(DeviceType.cpu, 0)
    gpu = Placement(DeviceType.gpu, gpu_num)
    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
    host = PhysicalTensor(cpu, desc)
    host.numpy()[...] = arr
    device = PhysicalTensor(gpu, desc)
    device.copy_from_async(host, stream)
    return device


def _copy_to_host(tensor: PhysicalTensor, dtype: thor.DataType, stream: Stream) -> np.ndarray:
    _host_same_dtype = _cpu_tensor(list(tensor.dimensions), tensor.dtype)
    host = _cpu_tensor(list(tensor.dimensions), dtype)
    _host_same_dtype.copy_from_async(tensor, stream)
    host.copy_from_async(_host_same_dtype, stream)
    stream.synchronize()
    return host.numpy().copy()


def _make_matrix(shape: tuple[int, int], dtype: thor.DataType, scale: float = 0.125, bias: float = -1.5) -> np.ndarray:
    values = np.arange(1, 1 + shape[0] * shape[1], dtype=np.float32).reshape(shape) * scale + bias
    return _cast_reference_to_storage_dtype_with_saturation(values, dtype)


@pytest.mark.cuda
def test_transpose_binding_method_and_property_exist():
    x = ex.input("x")

    method_expr = x.transpose()
    property_expr = x.T

    assert isinstance(method_expr, thor.physical.Expression)
    assert isinstance(property_expr, thor.physical.Expression)

    eq_method = ex.compile(method_expr, device_num=0)
    eq_property = ex.compile(property_expr, device_num=0)

    dtype = thor.DataType.fp32
    x_np = _make_matrix((7, 5), dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    assert eq_method.output_shape(inputs_gpu) == eq_property.output_shape(inputs_gpu) == [5, 7]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(64, 64), (65, 97), (97, 65), (33, 31)])
def test_transpose_forward_numerical(dtype: thor.DataType, shape: tuple[int, int]):
    x = ex.input("x")
    eq = ex.compile(x.transpose(), device_num=0)

    x_np = _make_matrix(shape, dtype)
    expected = _cast_reference_to_storage_dtype_with_saturation(x_np.astype(np.float32).T, dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (thor.DataType.fp32, thor.DataType.fp32),
        (thor.DataType.fp32, thor.DataType.fp16),
        (thor.DataType.fp16, thor.DataType.fp32),
        (thor.DataType.bf16, thor.DataType.fp16),
        (thor.DataType.fp8_e4m3, thor.DataType.fp32),
        (thor.DataType.fp32, thor.DataType.fp8_e5m2),
    ],
)
def test_transpose_forward_output_dtype_override_numerical(input_dtype: thor.DataType, output_dtype: thor.DataType):
    x = ex.input("x")
    eq = ex.compile(x.transpose().with_output_dtype(output_dtype), device_num=0)

    x_np = _make_matrix((65, 97), input_dtype, scale=0.0625, bias=-0.75)
    expected = _cast_reference_to_storage_dtype_with_saturation(x_np.astype(np.float32).T, output_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, input_dtype, stream)
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), output_dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, output_dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_transpose_backward_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    fwd_eq = ex.compile(x.transpose(), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    x_np = _make_matrix((33, 65), dtype, scale=0.1, bias=-2.0)
    grad_np = _make_matrix((65, 33), dtype, scale=0.2, bias=0.25)
    expected = _cast_reference_to_storage_dtype_with_saturation(grad_np.astype(np.float32).T, dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output("x_grad"), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [thor.DataType.fp16, thor.DataType.fp32])
def test_transpose_pointwise_after_stage_numerical(dtype: thor.DataType):
    x = ex.input("x")
    out = ex.exp(x.transpose() * 0.25 - 0.5)
    eq = ex.compile(out, device_num=0)

    # Keep exp() inputs within fp16 range to avoid NumPy reference-cast overflow warnings.
    x_np = _make_matrix((65, 97), dtype, scale=0.005, bias=-1.0)
    expected = _cast_reference_to_storage_dtype_with_saturation(np.exp(x_np.astype(np.float32).T * 0.25 - 0.5), dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_transpose_rejects_non_rank2_tensor_at_stamp_time():
    x = ex.input("x")
    eq = ex.compile(x.transpose(), device_num=0)

    dtype = thor.DataType.fp32
    x_np = np.arange(1, 1 + 2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    with pytest.raises(RuntimeError, match="rank-2"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [thor.DataType.fp16, thor.DataType.fp32])
def test_explicit_transpose_is_absorbed_into_matmul_stage(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    eq = ex.compile(a.transpose() @ b, device_num=0)

    a_np = _make_matrix((3, 2), dtype, scale=0.125, bias=-0.5)
    b_np = _make_matrix((3, 4), dtype, scale=0.1, bias=0.25)
    expected = _cast_reference_to_storage_dtype_with_saturation(
        a_np.astype(np.float32).T @ b_np.astype(np.float32), dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    stage_kinds = eq._debug_stage_kinds(inputs_gpu)
    assert stage_kinds == ["Matmul(lhsT=1,rhsT=0,auxT=0)"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [thor.DataType.fp16, thor.DataType.fp32])
def test_explicit_transpose_is_absorbed_into_operator_lowered_gemm_stage(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    eq = ex.compile(a @ b.transpose() + c, device_num=0)

    a_np = _make_matrix((2, 3), dtype, scale=0.125, bias=-0.5)
    b_np = _make_matrix((4, 3), dtype, scale=0.1, bias=0.25)
    c_np = _make_matrix((2, 4), dtype, scale=0.05, bias=-0.125)
    expected = _cast_reference_to_storage_dtype_with_saturation(
        a_np.astype(np.float32) @ b_np.astype(np.float32).T + c_np.astype(np.float32),
        dtype,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    stage_kinds = eq._debug_stage_kinds(inputs_gpu)
    assert stage_kinds == ["Matmul(lhsT=0,rhsT=1,auxT=0)"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [thor.DataType.fp16, thor.DataType.fp32])
def test_explicit_transpose_is_absorbed_into_explicit_gemm_stage(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    eq = ex.compile(ex.gemm(a.transpose(), b, c, alpha=1.25, beta=-0.5), device_num=0)

    a_np = _make_matrix((3, 2), dtype, scale=0.125, bias=-0.5)
    b_np = _make_matrix((3, 4), dtype, scale=0.1, bias=0.25)
    c_np = _make_matrix((2, 4), dtype, scale=0.05, bias=-0.125)
    expected = _cast_reference_to_storage_dtype_with_saturation(
        1.25 * (a_np.astype(np.float32).T @ b_np.astype(np.float32)) - 0.5 * c_np.astype(np.float32),
        dtype,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    stage_kinds = eq._debug_stage_kinds(inputs_gpu)
    assert stage_kinds == ["Matmul(lhsT=1,rhsT=0,auxT=0)"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)
