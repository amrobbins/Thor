import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes


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


FLOAT_DTYPES = [thor.DataType.fp32]


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_supported_subset_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    z = ex.input("z")
    w = ex.input("w")
    v = ex.input("v")
    u = ex.input("u")

    loss = ex.reduce_sum((x * y) + (x / z) - ex.sqrt(w) + ex.exp(v) + ex.ln(u), axis=[0, 1], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x", "y", "z", "w", "v", "u"])
    assert bwd_eq.output_names() == ["x_grad", "y_grad", "z_grad", "w_grad", "v_grad", "u_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[0.5, 1.5, 2.0], [2.5, 1.0, 0.75]], dtype=np.float32).astype(storage_dtype)
    z_np = np.array([[2.0, 4.0, 5.0], [3.0, 6.0, 7.0]], dtype=np.float32).astype(storage_dtype)
    w_np = np.array([[4.0, 9.0, 16.0], [25.0, 36.0, 49.0]], dtype=np.float32).astype(storage_dtype)
    v_np = np.array([[0.1, 0.2, -0.3], [0.4, -0.2, 0.0]], dtype=np.float32).astype(storage_dtype)
    u_np = np.array([[1.5, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    z_ref = z_np.astype(np.float32)
    w_ref = w_np.astype(np.float32)
    v_ref = v_np.astype(np.float32)
    u_ref = u_np.astype(np.float32)

    expected = {
        "x_grad": (y_ref + 1.0 / z_ref).astype(storage_dtype),
        "y_grad": x_ref.astype(storage_dtype),
        "z_grad": (-(x_ref) / (z_ref * z_ref)).astype(storage_dtype),
        "w_grad": (-(1.0 / (2.0 * np.sqrt(w_ref)))).astype(storage_dtype),
        "v_grad": np.exp(v_ref).astype(storage_dtype),
        "u_grad": (1.0 / u_ref).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        "z": _host_to_gpu(z_np, dtype, stream),
        "w": _host_to_gpu(w_np, dtype, stream),
        "v": _host_to_gpu(v_np, dtype, stream),
        "u": _host_to_gpu(u_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    for name in bwd_eq.output_names():
        out_gpu = stamped.output(name)
        out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
        out_host.copy_from_async(out_gpu, stream)
        stream.synchronize()
        got = out_host.numpy().copy()
        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
def test_compile_backward_rejects_reduce_sum_with_squeeze():
    x = ex.input("x")
    loss = ex.reduce_sum(x, axis=1, squeeze=True)
    fwd_eq = ex.compile(loss, device_num=0)

    with pytest.raises(RuntimeError, match="squeeze=False"):
        fwd_eq.compile_backward(["x"])


@pytest.mark.cuda
def test_compile_backward_rejects_unsupported_op():
    x = ex.input("x")
    y = ex.input("y")
    loss = ex.reduce_sum(ex.min(x, y), axis=[0, 1], squeeze=False)
    fwd_eq = ex.compile(loss, device_num=0)

    with pytest.raises(RuntimeError, match="does not yet support backward for op MIN"):
        fwd_eq.compile_backward(["x"])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_more_pointwise_ops_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    d = ex.input("d")
    p = ex.input("p")
    q = ex.input("q")

    loss = ex.reduce_sum(ex.exp2(a) + ex.exp10(b) + ex.log2(c) + ex.log10(d) + (p**q), axis=[0, 1], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b", "c", "d", "p", "q"])
    assert bwd_eq.output_names() == ["a_grad", "b_grad", "c_grad", "d_grad", "p_grad", "q_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    a_np = np.array([[0.0, 1.0, -0.5], [1.5, -1.0, 0.25]], dtype=np.float32).astype(storage_dtype)
    b_np = np.array([[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]], dtype=np.float32).astype(storage_dtype)
    c_np = np.array([[1.5, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    d_np = np.array([[1.25, 2.5, 4.0], [8.0, 3.5, 6.5]], dtype=np.float32).astype(storage_dtype)
    p_np = np.array([[1.5, 2.0, 3.0], [4.0, 2.5, 5.0]], dtype=np.float32).astype(storage_dtype)
    q_np = np.array([[2.0, 1.5, 0.5], [3.0, 2.5, 1.25]], dtype=np.float32).astype(storage_dtype)

    a_ref = a_np.astype(np.float32)
    b_ref = b_np.astype(np.float32)
    c_ref = c_np.astype(np.float32)
    d_ref = d_np.astype(np.float32)
    p_ref = p_np.astype(np.float32)
    q_ref = q_np.astype(np.float32)

    p_pow_q = np.power(p_ref, q_ref)
    expected = {
        "a_grad": (np.log(2.0) * np.exp2(a_ref)).astype(storage_dtype),
        "b_grad": (np.log(10.0) * np.power(10.0, b_ref)).astype(storage_dtype),
        "c_grad": (1.0 / (c_ref * np.log(2.0))).astype(storage_dtype),
        "d_grad": (1.0 / (d_ref * np.log(10.0))).astype(storage_dtype),
        "p_grad": (q_ref * np.power(p_ref, q_ref - 1.0)).astype(storage_dtype),
        "q_grad": (p_pow_q * np.log(p_ref)).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        "d": _host_to_gpu(d_np, dtype, stream),
        "p": _host_to_gpu(p_np, dtype, stream),
        "q": _host_to_gpu(q_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    for name in bwd_eq.output_names():
        out_gpu = stamped.output(name)
        out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
        out_host.copy_from_async(out_gpu, stream)
        stream.synchronize()
        got = out_host.numpy().copy()
        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    upstream_name = "__grad_output"

    out = (x * y) + ex.exp(x) - ex.ln(y)

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x", "y"], upstream_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad", "y_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[2.5, 3.5, 4.0], [1.75, 2.25, 2.75]], dtype=np.float32).astype(storage_dtype)
    grad_np = np.array([[0.5, -1.0, 0.25], [1.5, -0.75, 2.0]], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    grad_ref = grad_np.astype(np.float32)

    expected = {
        "x_grad": (grad_ref * (y_ref + np.exp(x_ref))).astype(storage_dtype),
        "y_grad": (grad_ref * (x_ref - (1.0 / y_ref))).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    for name in bwd_eq.output_names():
        out_gpu = stamped.output(name)
        out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
        out_host.copy_from_async(out_gpu, stream)
        stream.synchronize()
        got = out_host.numpy().copy()
        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
def test_compile_backward_rejects_explicit_upstream_name_collision():
    x = ex.input("x")
    y = ex.input("y")
    out = x * y
    fwd_eq = ex.compile(out, device_num=0)

    with pytest.raises(RuntimeError, match="collides with an existing forward input"):
        fwd_eq.compile_backward(["x"], upstream_input_name="x")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_broadcast_unbroadcast_same_rank_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    loss = ex.reduce_sum(x * y, axis=[0, 1], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x", "y"])

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[0.5, 1.5, -2.0]], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)

    expected = {
        "x_grad": np.broadcast_to(y_ref, x_ref.shape).astype(storage_dtype),
        "y_grad": np.sum(x_ref, axis=0, keepdims=True).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    for name in bwd_eq.output_names():
        out_gpu = stamped.output(name)
        out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
        out_host.copy_from_async(out_gpu, stream)
        stream.synchronize()
        got = out_host.numpy().copy()
        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_broadcast_unbroadcast_leading_axis_squeeze_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    loss = ex.reduce_sum(x + y, axis=[0, 1], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x", "y"])

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32).astype(storage_dtype)

    expected = {
        "x_grad": np.full_like(x_np, fill_value=2.0, dtype=storage_dtype),
        "y_grad": np.ones_like(y_np, dtype=storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    for name in bwd_eq.output_names():
        out_gpu = stamped.output(name)
        out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
        out_host.copy_from_async(out_gpu, stream)
        stream.synchronize()
        got = out_host.numpy().copy()
        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)
