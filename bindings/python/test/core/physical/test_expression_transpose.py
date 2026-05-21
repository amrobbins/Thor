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

    assert eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_transpose_binary_consumer_fuses_and_is_numerical():
    x = ex.input("x")
    y = ex.input("y")
    out = (x.transpose() * 1.5 + y * 0.25 - 0.125).with_output_dtype(thor.DataType.fp32)
    eq = ex.compile(out, device_num=0)

    dtype = thor.DataType.fp32
    x_np = _make_matrix((65, 97), dtype, scale=0.01, bias=-0.75)
    y_np = _make_matrix((97, 65), dtype, scale=-0.0075, bias=0.5)
    expected = x_np.astype(np.float32).T * 1.5 + y_np.astype(np.float32) * 0.25 - 0.125

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == list(expected.shape)
    assert eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_transpose_consumer_fuses_with_pre_transpose_broadcast_numerical():
    x = ex.input("x")
    bias = ex.input("bias")
    out = ex.exp((x + bias).transpose() * 0.125 - 0.25).with_output_dtype(thor.DataType.fp32)
    eq = ex.compile(out, device_num=0)

    dtype = thor.DataType.fp32
    x_np = _make_matrix((7, 11), dtype, scale=0.01, bias=-0.5)
    bias_np = _make_matrix((1, 11), dtype, scale=0.005, bias=0.125)
    expected = np.exp((x_np.astype(np.float32) + bias_np.astype(np.float32)).T * 0.125 - 0.25)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "bias": _host_to_gpu(bias_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == list(expected.shape)
    assert eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_batched_transpose_consumer_fuses_and_is_numerical():
    x = ex.input("x")
    scale = ex.input("scale")
    out = ex.ln(x.transpose() + scale).with_output_dtype(thor.DataType.fp32)
    eq = ex.compile(out, device_num=0)

    dtype = thor.DataType.fp32
    shape = (3, 5, 7)
    x_np = np.arange(1, 1 + int(np.prod(shape)), dtype=np.float32).reshape(shape) * 0.01 + 1.0
    scale_np = np.full((1, 7, 1), 0.5, dtype=np.float32)
    expected = np.log(np.swapaxes(x_np, -1, -2) + scale_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "scale": _host_to_gpu(scale_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == list(expected.shape)
    assert eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 4, 5)], ids=["rank3", "rank4"])
def test_transpose_forward_batched_trailing_dims_numerical(shape: tuple[int, ...]):
    x = ex.input("x")
    eq = ex.compile(x.transpose(), device_num=0)

    dtype = thor.DataType.fp32
    x_np = (np.arange(1, 1 + int(np.prod(shape)), dtype=np.float32).reshape(shape) * 0.125) - 1.0
    expected = np.swapaxes(x_np, -1, -2)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    assert eq.output_shape(inputs_gpu) == list(expected.shape)

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_tiled_logical_transpose_consumer_handles_pre_and_post_broadcast_numerical():
    x = ex.input("x")
    pre_bias = ex.input("pre_bias")
    post_bias = ex.input("post_bias")

    out = ex.exp(((x + pre_bias) * 0.25).transpose() + post_bias - 0.125).with_output_dtype(thor.DataType.fp32)
    eq = ex.compile(out, device_num=0)

    dtype = thor.DataType.fp32
    x_shape = (2, 5, 7)
    x_np = np.arange(1, 1 + int(np.prod(x_shape)), dtype=np.float32).reshape(x_shape) * 0.01 - 0.5
    pre_bias_np = np.linspace(-0.125, 0.125, num=7, dtype=np.float32).reshape(1, 1, 7)
    post_bias_np = np.linspace(0.05, 0.15, num=7, dtype=np.float32).reshape(1, 7, 1)
    expected = np.exp(np.swapaxes((x_np + pre_bias_np) * 0.25, -1, -2) + post_bias_np - 0.125)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "pre_bias": _host_to_gpu(pre_bias_np, dtype, stream),
        "post_bias": _host_to_gpu(post_bias_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == list(expected.shape)
    assert eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]

    stamped = eq.stamp(inputs_gpu, stream)
    assert stamped._debug_stage_kinds() == ["FusedKernel"]
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_tiled_logical_transpose_consumer_rank4_fuses_and_is_numerical():
    x = ex.input("x")
    y = ex.input("y")

    out = ex.sqrt((x.transpose() * 0.125) + y + 1.25).with_output_dtype(thor.DataType.fp32)
    eq = ex.compile(out, device_num=0)

    dtype = thor.DataType.fp32
    x_shape = (2, 3, 5, 7)
    x_np = np.arange(1, 1 + int(np.prod(x_shape)), dtype=np.float32).reshape(x_shape) * 0.002 + 0.5
    y_np = np.linspace(0.1, 0.4, num=2 * 3 * 7 * 5, dtype=np.float32).reshape(2, 3, 7, 5)
    expected = np.sqrt(np.swapaxes(x_np, -1, -2) * 0.125 + y_np + 1.25)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == list(expected.shape)
    assert eq._debug_stage_kinds(inputs_gpu) == ["FusedKernel"]

    stamped = eq.stamp(inputs_gpu, stream)
    assert stamped._debug_stage_kinds() == ["FusedKernel"]
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_transpose_rejects_rank1_tensor_at_stamp_time():
    x = ex.input("x")
    eq = ex.compile(x.transpose(), device_num=0)

    dtype = thor.DataType.fp32
    x_np = np.arange(1, 6, dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    with pytest.raises(RuntimeError, match="rank >= 2"):
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
    assert stage_kinds == ["Matmul(op=MATMUL,lhsT=1,rhsT=0,auxT=0)"]

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
    assert stage_kinds == ["Matmul(op=GEMM,lhsT=0,rhsT=1,auxT=0)"]

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
    assert stage_kinds == ["Matmul(op=GEMM,lhsT=1,rhsT=0,auxT=0)"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)
