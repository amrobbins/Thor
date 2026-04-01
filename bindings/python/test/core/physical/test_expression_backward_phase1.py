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
    bwd_eq = fwd_eq.compile_backward(["x", "y"], error_input_name=upstream_name)
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
        fwd_eq.compile_backward(["x"], error_input_name="x")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_mean_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_mean(x, axis=1, squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
                    dtype=np.float32).astype(storage_dtype)

    expected = {
        "x_grad": np.full_like(x_np, fill_value=(1.0 / 3.0), dtype=storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
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
def test_compile_backward_reduce_mean_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.reduce_mean(x, axis=1, squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
                    dtype=np.float32).astype(storage_dtype)
    grad_np = np.array([[[2.0, 3.0]], [[4.0, 5.0]]], dtype=np.float32).astype(storage_dtype)

    expected = {
        "x_grad": (np.broadcast_to(grad_np.astype(np.float32) / 3.0, x_np.shape)).astype(storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unsqueeze_forward_and_output_shape(dtype: thor.DataType):
    x = ex.input("x")
    out = ex.unsqueeze(x, axis=[0, 2])

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    expected = x_np.reshape((1, 2, 1, 3))

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    assert eq.output_shape(inputs_gpu) == [1, 2, 1, 3]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output(eq.output_names()[0])
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_squeeze_forward_and_output_shape(dtype: thor.DataType):
    x = ex.input("x")
    out = ex.squeeze(x, axis=[0, 2])

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32).astype(storage_dtype)
    x_np = x_np.reshape((1, 2, 1, 3))
    expected = x_np.reshape((2, 3))

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    assert eq.output_shape(inputs_gpu) == [2, 3]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output(eq.output_names()[0])
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_unsqueeze_identity_numerical(dtype: thor.DataType):
    x = ex.input("x")
    loss = ex.reduce_sum(ex.unsqueeze(x, axis=[0, 2]), axis=[0, 1, 2, 3], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    expected = {
        "x_grad": np.ones_like(x_np, dtype=storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_squeeze_identity_numerical(dtype: thor.DataType):
    x = ex.input("x")
    loss = ex.reduce_sum(ex.squeeze(x, axis=[0, 2]), axis=[0, 1], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32).astype(storage_dtype)
    x_np = x_np.reshape((1, 2, 1, 3))
    expected = {
        "x_grad": np.ones_like(x_np, dtype=storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_sum_with_squeeze_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_sum(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = {
        "x_grad": np.ones_like(x_np, dtype=storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_mean_with_squeeze_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_mean(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = {
        "x_grad": np.full_like(x_np, fill_value=(1.0 / 3.0), dtype=storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_norm2_with_squeeze_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_norm2(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    norms = np.sqrt(np.sum(x_ref * x_ref, axis=1, keepdims=True))
    expected = {
        "x_grad": (x_ref / norms).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
def test_compile_backward_rejects_multi_output_without_explicit_upstreams():
    x = ex.input("x")
    y = ex.input("y")
    outs = ex.outputs({
        "prod": x * y,
        "row_sum": ex.reduce_sum(x, axis=1, squeeze=[1])
    })
    fwd_eq = ex.compile(outs, device_num=0)

    with pytest.raises(RuntimeError, match="requires explicit upstream"):
        fwd_eq.compile_backward(["x", "y"])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_multi_output_explicit_upstreams_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "prod": x * y,
        "row_sum": ex.reduce_sum(x, axis=1, squeeze=[1]),
    })

    fwd_eq = ex.compile(outs, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x", "y"],
        feature_output_name_to_error_input_name={
            "prod": "__grad_prod",
            "row_sum": "__grad_row_sum",
        },
    )
    assert bwd_eq.output_names() == ["x_grad", "y_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[0.5, 1.5, -2.0], [2.0, -1.0, 0.25]], dtype=np.float32).astype(storage_dtype)
    grad_prod_np = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.0]], dtype=np.float32).astype(storage_dtype)
    grad_row_sum_np = np.array([2.0, -3.0], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    grad_prod_ref = grad_prod_np.astype(np.float32)
    grad_row_sum_ref = grad_row_sum_np.astype(np.float32)

    expected = {
        "x_grad": (grad_prod_ref * y_ref + grad_row_sum_ref[:, None]).astype(storage_dtype),
        "y_grad": (grad_prod_ref * x_ref).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        "__grad_prod": _host_to_gpu(grad_prod_np, dtype, stream),
        "__grad_row_sum": _host_to_gpu(grad_row_sum_np, dtype, stream),
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
def test_unsqueeze_forward_negative_axes_and_output_shape(dtype: thor.DataType):
    x = ex.input("x")
    with pytest.raises(RuntimeError, match="bad_cast"):
        out = ex.unsqueeze(x, axis=[-1, 0])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_squeeze_forward_all_singletons_output_shape(dtype: thor.DataType):
    x = ex.input("x")
    out = ex.squeeze(x)

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.arange(1, 7, dtype=np.float32).reshape((1, 2, 1, 3, 1)).astype(storage_dtype)
    expected = x_np.reshape((2, 3))

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    assert eq.output_shape(inputs_gpu) == [2, 3]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output(eq.output_names()[0])
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_squeeze_all_singletons_identity_numerical(dtype: thor.DataType):
    x = ex.input("x")
    loss = ex.reduce_sum(ex.squeeze(x), axis=[0, 1], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.arange(1, 7, dtype=np.float32).reshape((1, 2, 1, 3, 1)).astype(storage_dtype)
    expected = {
        "x_grad": np.ones_like(x_np, dtype=storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_repeated_shape_path_accumulates_numerical(dtype: thor.DataType):
    x = ex.input("x")

    path1 = ex.reduce_sum(ex.unsqueeze(x, axis=[0, 2]), axis=[0, 1, 2, 3], squeeze=False)

    # Break shape-path CSE intentionally while preserving the same math.
    x2 = x + ex.constant_scalar(0.0)
    path2 = ex.reduce_sum(ex.squeeze(ex.unsqueeze(x2, axis=[0, 2]), axis=[0, 2]), axis=[0, 1], squeeze=False)

    loss = path1 + path2

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    expected = {
        "x_grad": np.full_like(x_np, fill_value=2.0, dtype=storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_repeated_shape_path_accumulates_numerical_2(dtype: thor.DataType):
    x = ex.input("x")

    path1 = ex.reduce_sum(ex.unsqueeze(x, axis=[0, 2]), axis=[0, 1, 2, 3], squeeze=False)
    path2 = ex.reduce_sum(ex.squeeze(ex.unsqueeze(x, axis=[0, 2]), axis=[0, 2]), axis=[0, 1], squeeze=False)
    loss = path1 + path2

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    expected = {
        "x_grad": np.full_like(x_np, fill_value=2.0, dtype=storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_sum_with_squeeze_and_broadcast_operand_y_grad_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    loss = ex.reduce_sum(x * y, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["y"])
    assert bwd_eq.output_names() == ["y_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)
    y_np = np.array([[0.5, -1.5]], dtype=np.float32).astype(storage_dtype)

    expected = {
        "y_grad": np.sum(x_np.astype(np.float32), axis=(0, 1), keepdims=False).reshape((1, 2)).astype(storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("y_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["y_grad"].shape
    _assert_close(got, expected["y_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_scalar_constant_through_shape_paths_additive_numerical(dtype: thor.DataType):
    x = ex.input("x")

    # Path 1: standard unsqueeze -> reduce path.
    path1 = ex.reduce_sum(ex.unsqueeze(x, axis=[0, 2]), axis=[0, 1, 2, 3], squeeze=False)

    # Path 2: shape-heavy branch with scalar constants intentionally threaded through it.
    # This keeps the math identical to sum(x), but forces scalar leaves to coexist with
    # squeeze/unsqueeze structure in the staged backward graph.
    scalar_zero = ex.constant_scalar(0.0)
    scalar_one = ex.constant_scalar(1.0)

    x2 = (x + scalar_zero) * scalar_one
    path2 = ex.reduce_sum(
        ex.squeeze(ex.unsqueeze(x2, axis=[0, 2]), axis=[0, 2]),
        axis=[0, 1],
        squeeze=False,
    )

    loss = path1 + path2

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    expected = {
        "x_grad": np.full_like(x_np, fill_value=2.0, dtype=storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_scalar_constant_through_shape_paths_multiplicative_numerical(dtype: thor.DataType):
    x = ex.input("x")

    # Force scalar constants into multiple shape-heavy branches that should still
    # reduce to a simple multiple of x in backward.
    scalar_zero = ex.constant_scalar(0.0)
    scalar_one = ex.constant_scalar(1.0)
    scalar_two = ex.constant_scalar(2.0)

    branch1 = ex.reduce_sum(
        ex.unsqueeze((x * scalar_one) + scalar_zero, axis=[0, 2]),
        axis=[0, 1, 2, 3],
        squeeze=False,
    )

    branch2 = ex.reduce_sum(
        ex.squeeze(ex.unsqueeze((x + scalar_zero) * scalar_one, axis=[0, 2]), axis=[0, 2]),
        axis=[0, 1],
        squeeze=False,
    )

    # branch3 contributes an additional factor of 2 through the same kind of shape path.
    branch3 = ex.reduce_sum(
        ex.squeeze(ex.unsqueeze(x * scalar_two, axis=[0, 2]), axis=[0, 2]),
        axis=[0, 1],
        squeeze=False,
    )

    loss = branch1 + branch2 + branch3

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    expected = {
        "x_grad": np.full_like(x_np, fill_value=4.0, dtype=storage_dtype)
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream)
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reused_primal_accumulates_nontrivial_branches_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_sum((x * x) + ex.exp(x), axis=[0, 1], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=np.float32).astype(storage_dtype)
    x_ref = x_np.astype(np.float32)

    expected = {
        "x_grad": (2.0 * x_ref + np.exp(x_ref)).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_sum_with_squeeze_explicit_upstream_broadcast_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.reduce_sum(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array([[2.0, -1.5], [0.25, 3.0]], dtype=np.float32).astype(storage_dtype)

    expected = {
        "x_grad": np.broadcast_to(grad_np.astype(np.float32)[:, None, :], x_np.shape).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_multi_output_explicit_upstreams_mixed_constant_and_nonconstant_x_grad_numerical(
        dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "prod": x * y,
        "total_x": ex.reduce_sum(x, axis=[0, 1], squeeze=False),
    })

    fwd_eq = ex.compile(outs, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x"],
        feature_output_name_to_error_input_name={
            "prod": "__grad_prod",
            "total_x": "__grad_total_x",
        },
    )
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[0.5, -2.0, 1.5], [3.0, 0.25, -1.0]], dtype=np.float32).astype(storage_dtype)
    grad_prod_np = np.array([[2.0, -0.5, 1.25], [0.0, 3.0, -2.0]], dtype=np.float32).astype(storage_dtype)
    grad_total_x_np = np.array([[1.75]], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    grad_prod_ref = grad_prod_np.astype(np.float32)
    grad_total_x_ref = grad_total_x_np.astype(np.float32)

    expected = {
        "x_grad":
            (grad_prod_ref * y_ref + np.full_like(x_ref, fill_value=grad_total_x_ref.item())).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        "__grad_prod": _host_to_gpu(grad_prod_np, dtype, stream),
        "__grad_total_x": _host_to_gpu(grad_total_x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_min_numerical(dtype: thor.DataType):
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

    expected = {
        "x_grad": np.array(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_min_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.reduce_min(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [
            [[1.0, 9.0], [0.0, 3.0], [5.0, 7.0]],
            [[4.0, 2.0], [6.0, 8.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [[2.0, -1.5], [0.25, 3.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.zeros_like(x_np, dtype=storage_dtype)
    expected[0, 1, 0] = 2.0
    expected[0, 1, 1] = -1.5
    expected[1, 2, 0] = 0.25
    expected[1, 2, 1] = 3.0

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_max_with_squeeze_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_max(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [
            [[1.0, 9.0], [0.0, 3.0], [5.0, 7.0]],
            [[4.0, 2.0], [6.0, 8.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.zeros_like(x_np, dtype=storage_dtype)
    expected[0, 2, 0] = 1.0
    expected[0, 0, 1] = 1.0
    expected[1, 1, 0] = 1.0
    expected[1, 1, 1] = 1.0

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_min_reduce_max_multi_output_explicit_upstreams_numerical(dtype: thor.DataType):
    x = ex.input("x")

    outs = ex.outputs(
        {
            "row_min": ex.reduce_min(x, axis=1, squeeze=[1]),
            "row_max": ex.reduce_max(x, axis=1, squeeze=[1]),
        })

    fwd_eq = ex.compile(outs, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x"],
        feature_output_name_to_error_input_name={
            "row_min": "__grad_row_min",
            "row_max": "__grad_row_max",
        },
    )
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[3.0, 1.0, 4.0], [2.0, -5.0, 0.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_row_min_np = np.array([2.0, -3.0], dtype=np.float32).astype(storage_dtype)
    grad_row_max_np = np.array([0.5, 1.25], dtype=np.float32).astype(storage_dtype)

    expected = np.array(
        [[0.0, 2.0, 0.5], [1.25, -3.0, 0.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "__grad_row_min": _host_to_gpu(grad_row_min_np, dtype, stream),
        "__grad_row_max": _host_to_gpu(grad_row_max_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_max_all_axes_scalar_output_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.reduce_max(x, axis=[0, 1], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[3.0, 1.0, 4.0], [2.0, -5.0, 0.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array([[1.75]], dtype=np.float32).astype(storage_dtype)

    expected = np.zeros_like(x_np, dtype=storage_dtype)
    expected[0, 2] = 1.75

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_elementwise_min_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    upstream_name = "__grad_output"

    out = ex.min(x, y)

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x", "y"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad", "y_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array(
        [[1.0, 5.0, 2.0], [4.0, 3.0, 7.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    y_np = np.array(
        [[2.0, 5.0, 1.0], [4.0, 6.0, 7.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [[1.0, -2.0, 3.0], [0.5, -1.5, 2.5]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    grad_ref = grad_np.astype(np.float32)

    x_mask = np.where(x_ref < y_ref, 1.0, np.where(x_ref > y_ref, 0.0, 0.5))
    y_mask = np.where(x_ref < y_ref, 0.0, np.where(x_ref > y_ref, 1.0, 0.5))

    expected = {
        "x_grad": (grad_ref * x_mask).astype(storage_dtype),
        "y_grad": (grad_ref * y_mask).astype(storage_dtype),
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
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_elementwise_max_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    upstream_name = "__grad_output"

    out = ex.max(x, y)

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x", "y"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad", "y_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array(
        [[1.0, 5.0, 2.0], [4.0, 3.0, 7.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    y_np = np.array(
        [[2.0, 5.0, 1.0], [4.0, 6.0, 7.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [[1.0, -2.0, 3.0], [0.5, -1.5, 2.5]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    grad_ref = grad_np.astype(np.float32)

    x_mask = np.where(x_ref > y_ref, 1.0, np.where(x_ref < y_ref, 0.0, 0.5))
    y_mask = np.where(x_ref > y_ref, 0.0, np.where(x_ref < y_ref, 1.0, 0.5))

    expected = {
        "x_grad": (grad_ref * x_mask).astype(storage_dtype),
        "y_grad": (grad_ref * y_mask).astype(storage_dtype),
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
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_elementwise_min_max_multi_output_explicit_upstreams_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "mn": ex.min(x, y),
        "mx": ex.max(x, y),
    })

    fwd_eq = ex.compile(outs, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x", "y"],
        feature_output_name_to_error_input_name={
            "mn": "__grad_mn",
            "mx": "__grad_mx",
        },
    )
    assert bwd_eq.output_names() == ["x_grad", "y_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array(
        [[1.0, 5.0, 2.0], [4.0, 3.0, 7.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    y_np = np.array(
        [[2.0, 5.0, 1.0], [4.0, 6.0, 7.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_mn_np = np.array(
        [[1.0, -2.0, 3.0], [0.5, -1.5, 2.5]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_mx_np = np.array(
        [[-4.0, 1.25, 0.75], [2.0, -3.5, 1.5]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    grad_mn_ref = grad_mn_np.astype(np.float32)
    grad_mx_ref = grad_mx_np.astype(np.float32)

    min_x_mask = np.where(x_ref < y_ref, 1.0, np.where(x_ref > y_ref, 0.0, 0.5))
    min_y_mask = np.where(x_ref < y_ref, 0.0, np.where(x_ref > y_ref, 1.0, 0.5))
    max_x_mask = np.where(x_ref > y_ref, 1.0, np.where(x_ref < y_ref, 0.0, 0.5))
    max_y_mask = np.where(x_ref > y_ref, 0.0, np.where(x_ref < y_ref, 1.0, 0.5))

    expected = {
        "x_grad": (grad_mn_ref * min_x_mask + grad_mx_ref * max_x_mask).astype(storage_dtype),
        "y_grad": (grad_mn_ref * min_y_mask + grad_mx_ref * max_y_mask).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        "__grad_mn": _host_to_gpu(grad_mn_np, dtype, stream),
        "__grad_mx": _host_to_gpu(grad_mx_np, dtype, stream),
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
def test_compile_backward_reduce_prod_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_prod(x, axis=1, squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    prods = np.prod(x_ref, axis=1, keepdims=True)
    expected = {
        "x_grad": (prods / x_ref).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_prod_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.reduce_prod(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [1.5, -2.0],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    grad_ref = grad_np.astype(np.float32)
    prods = np.prod(x_ref, axis=1, keepdims=True)
    expected = {
        "x_grad": ((grad_ref[:, None] * prods) / x_ref).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_norm1_numerical(dtype: thor.DataType):
    x = ex.input("x")

    loss = ex.reduce_norm1(x, axis=1, squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[2.0, -3.0, 5.0], [-7.0, 11.0, -13.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    expected = {
        "x_grad": np.sign(x_ref).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_norm1_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.reduce_norm1(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[2.0, -3.0, 5.0], [-7.0, 11.0, -13.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [1.25, -0.5],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    grad_ref = grad_np.astype(np.float32)
    expected = {
        "x_grad": (grad_ref[:, None] * np.sign(x_ref)).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_abs_numerical(dtype: thor.DataType):
    x = ex.input("x")
    out = ex.abs(x)

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[-2.0, 3.5, -4.25], [7.0, -1.5, 2.25]],
        dtype=np.float32,
    ).astype(storage_dtype)
    expected = np.abs(x_np.astype(np.float32)).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, dtype, stream)

    stamped = eq.stamp({
        "x": x_gpu
    }, stream)
    stamped.run()

    out_gpu = stamped.output()
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_abs_numerical(dtype: thor.DataType):
    x = ex.input("x")
    loss = ex.abs(x)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[-2.0, 3.5, -4.25], [7.0, -1.5, 2.25]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = {
        "x_grad": np.sign(x_np.astype(np.float32)).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_abs_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.abs(x)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[-2.0, 3.5, -4.25], [7.0, -1.5, 2.25]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [[1.25, -2.0, 0.75], [-0.5, 3.0, -4.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = {
        "x_grad": (grad_np.astype(np.float32) * np.sign(x_np.astype(np.float32))).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_abs_zero_rule_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    out = ex.abs(x)

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array(
        [[-2.0, 0.0, 3.0], [0.0, -4.5, 5.5]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [[1.25, -2.0, 0.75], [-0.5, 3.0, -4.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    grad_ref = grad_np.astype(np.float32)

    safe_sign = np.where(x_ref > 0.0, 1.0, np.where(x_ref < 0.0, -1.0, 0.0))
    expected = {
        "x_grad": (grad_ref * safe_sign).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_reduce_norm1_zero_rule_explicit_upstream_numerical(dtype: thor.DataType):
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.reduce_norm1(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array(
        [[-2.0, 0.0, 3.0], [0.0, -4.5, 5.5]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [1.5, -2.0],
        dtype=np.float32,
    ).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    grad_ref = grad_np.astype(np.float32)

    safe_sign = np.where(x_ref > 0.0, 1.0, np.where(x_ref < 0.0, -1.0, 0.0))
    expected = {
        "x_grad": (grad_ref[:, None] * safe_sign).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_accumulate_grad_outputs_run_distinguishes_overwrite_and_accumulate_numerical(
        dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    upstream_name = "__grad_output"

    out = (x * y) + ex.exp(x) - ex.ln(y)

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq_overwrite = fwd_eq.compile_backward(
        ["x", "y"],
        error_input_name=upstream_name,
        accumulate_grad_outputs=False,
    )
    bwd_eq_accumulate = fwd_eq.compile_backward(
        ["x", "y"],
        error_input_name=upstream_name,
        accumulate_grad_outputs=True,
    )

    assert bwd_eq_overwrite.output_names() == ["x_grad", "y_grad"]
    assert bwd_eq_accumulate.output_names() == ["x_grad", "y_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[2.5, 3.5, 4.0], [1.75, 2.25, 2.75]], dtype=np.float32).astype(storage_dtype)
    grad_np = np.array([[0.5, -1.0, 0.25], [1.5, -0.75, 2.0]], dtype=np.float32).astype(storage_dtype)

    prefill_x_grad_np = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32).astype(storage_dtype)
    prefill_y_grad_np = np.array([[-3.0, -2.0, -1.0], [1.0, 2.0, 3.0]], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    grad_ref = grad_np.astype(np.float32)
    prefill_x_ref = prefill_x_grad_np.astype(np.float32)
    prefill_y_ref = prefill_y_grad_np.astype(np.float32)

    computed = {
        "x_grad": grad_ref * (y_ref + np.exp(x_ref)),
        "y_grad": grad_ref * (x_ref - (1.0 / y_ref)),
    }

    expected_overwrite = {
        "x_grad": computed["x_grad"].astype(storage_dtype),
        "y_grad": computed["y_grad"].astype(storage_dtype),
    }

    expected_accumulate = {
        "x_grad": (prefill_x_ref + computed["x_grad"]).astype(storage_dtype),
        "y_grad": (prefill_y_ref + computed["y_grad"]).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    overwrite_outputs_gpu = {
        "x_grad": _host_to_gpu(prefill_x_grad_np, dtype, stream),
        "y_grad": _host_to_gpu(prefill_y_grad_np, dtype, stream),
    }
    accumulate_outputs_gpu = {
        "x_grad": _host_to_gpu(prefill_x_grad_np, dtype, stream),
        "y_grad": _host_to_gpu(prefill_y_grad_np, dtype, stream),
    }

    bwd_eq_overwrite.run(inputs_gpu, overwrite_outputs_gpu, stream)
    bwd_eq_accumulate.run(inputs_gpu, accumulate_outputs_gpu, stream)

    for name in bwd_eq_overwrite.output_names():
        out_host = _cpu_tensor(list(overwrite_outputs_gpu[name].dimensions), dtype)
        out_host.copy_from_async(overwrite_outputs_gpu[name], stream)
        stream.synchronize()
        got = out_host.numpy().copy()
        assert got.shape == expected_overwrite[name].shape
        _assert_close(got, expected_overwrite[name], dtype)

    for name in bwd_eq_accumulate.output_names():
        out_host = _cpu_tensor(list(accumulate_outputs_gpu[name].dimensions), dtype)
        out_host.copy_from_async(accumulate_outputs_gpu[name], stream)
        stream.synchronize()
        got = out_host.numpy().copy()
        assert got.shape == expected_accumulate[name].shape
        _assert_close(got, expected_accumulate[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_multi_output_accumulate_grad_outputs_run_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")

    outs = ex.outputs({
        "prod": x * y,
        "row_sum": ex.reduce_sum(x, axis=1, squeeze=[1]),
    })

    fwd_eq = ex.compile(outs, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x", "y"],
        feature_output_name_to_error_input_name={
            "prod": "__grad_prod",
            "row_sum": "__grad_row_sum",
        },
        accumulate_grad_outputs=True,
    )
    assert bwd_eq.output_names() == ["x_grad", "y_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[0.5, 1.5, -2.0], [2.0, -1.0, 0.25]], dtype=np.float32).astype(storage_dtype)
    grad_prod_np = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.0]], dtype=np.float32).astype(storage_dtype)
    grad_row_sum_np = np.array([2.0, -3.0], dtype=np.float32).astype(storage_dtype)

    prefill_x_grad_np = np.array([[5.0, 4.0, 3.0], [2.0, 1.0, 0.0]], dtype=np.float32).astype(storage_dtype)
    prefill_y_grad_np = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    grad_prod_ref = grad_prod_np.astype(np.float32)
    grad_row_sum_ref = grad_row_sum_np.astype(np.float32)
    prefill_x_ref = prefill_x_grad_np.astype(np.float32)
    prefill_y_ref = prefill_y_grad_np.astype(np.float32)

    computed = {
        "x_grad": grad_prod_ref * y_ref + grad_row_sum_ref[:, None],
        "y_grad": grad_prod_ref * x_ref,
    }

    expected = {
        "x_grad": (prefill_x_ref + computed["x_grad"]).astype(storage_dtype),
        "y_grad": (prefill_y_ref + computed["y_grad"]).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        "__grad_prod": _host_to_gpu(grad_prod_np, dtype, stream),
        "__grad_row_sum": _host_to_gpu(grad_row_sum_np, dtype, stream),
    }
    outputs_gpu = {
        "x_grad": _host_to_gpu(prefill_x_grad_np, dtype, stream),
        "y_grad": _host_to_gpu(prefill_y_grad_np, dtype, stream),
    }

    bwd_eq.run(inputs_gpu, outputs_gpu, stream)

    for name in bwd_eq.output_names():
        out_host = _cpu_tensor(list(outputs_gpu[name].dimensions), dtype)
        out_host.copy_from_async(outputs_gpu[name], stream)
        stream.synchronize()
        got = out_host.numpy().copy()
        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_accumulate_grad_outputs_stamp_requires_outputs(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    upstream_name = "__grad_output"

    out = (x * y) + ex.exp(x)
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x", "y"],
        error_input_name=upstream_name,
        accumulate_grad_outputs=True,
    )

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[0.5, 1.5], [2.0, 2.5]], dtype=np.float32).astype(storage_dtype)
    grad_np = np.array([[1.0, -0.5], [0.25, 2.0]], dtype=np.float32).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    with pytest.raises(RuntimeError) as excinfo:
        bwd_eq.stamp(inputs_gpu, stream)

    msg = str(excinfo.value).lower()
    assert "accumulate" in msg or "gradient output" in msg or "caller-provided" in msg


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_accumulate_grad_outputs_stamp_uses_provided_accumulators_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    upstream_name = "__grad_output"

    out = (x * y) + ex.exp(x) - ex.ln(y)

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x", "y"],
        error_input_name=upstream_name,
        accumulate_grad_outputs=True,
    )

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32).astype(storage_dtype)
    y_np = np.array([[2.5, 3.5, 4.0], [1.75, 2.25, 2.75]], dtype=np.float32).astype(storage_dtype)
    grad_np = np.array([[0.5, -1.0, 0.25], [1.5, -0.75, 2.0]], dtype=np.float32).astype(storage_dtype)

    prefill_x_grad_np = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32).astype(storage_dtype)
    prefill_y_grad_np = np.array([[-3.0, -2.0, -1.0], [1.0, 2.0, 3.0]], dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)
    grad_ref = grad_np.astype(np.float32)
    prefill_x_ref = prefill_x_grad_np.astype(np.float32)
    prefill_y_ref = prefill_y_grad_np.astype(np.float32)

    expected = {
        "x_grad": (prefill_x_ref + grad_ref * (y_ref + np.exp(x_ref))).astype(storage_dtype),
        "y_grad": (prefill_y_ref + grad_ref * (x_ref - (1.0 / y_ref))).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }
    outputs_gpu = {
        "x_grad": _host_to_gpu(prefill_x_grad_np, dtype, stream),
        "y_grad": _host_to_gpu(prefill_y_grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, outputs_gpu, stream)
    stamped.run()

    for name in bwd_eq.output_names():
        out_host = _cpu_tensor(list(outputs_gpu[name].dimensions), dtype)
        out_host.copy_from_async(outputs_gpu[name], stream)
        stream.synchronize()
        got = out_host.numpy().copy()

        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
def test_compile_backward_accumulate_grad_outputs_stamp_rejects_wrong_shape():
    dtype = thor.DataType.fp32
    x = ex.input("x")
    y = ex.input("y")
    upstream_name = "__grad_output"

    out = x * y
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x", "y"],
        error_input_name=upstream_name,
        accumulate_grad_outputs=True,
    )

    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y_np = np.array([[0.5, 1.5, -2.0], [2.0, -1.0, 0.25]], dtype=np.float32)
    grad_np = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    wrong_x_grad_np = np.zeros((2, 4), dtype=np.float32)  # incompatible with x shape (2, 3)
    correct_y_grad_np = np.zeros_like(y_np, dtype=np.float32)

    outputs_gpu = {
        "x_grad": _host_to_gpu(wrong_x_grad_np, dtype, stream),
        "y_grad": _host_to_gpu(correct_y_grad_np, dtype, stream),
    }

    with pytest.raises(RuntimeError) as excinfo:
        bwd_eq.stamp(inputs_gpu, outputs_gpu, stream)

    msg = str(excinfo.value).lower()
    assert "dimension" in msg or "incompatible" in msg or "shape" in msg


@pytest.mark.cuda
def test_compile_backward_accumulate_grad_outputs_stamp_rejects_wrong_dtype():
    input_dtype = thor.DataType.fp16
    wrong_output_dtype = thor.DataType.fp32

    x = ex.input("x")
    y = ex.input("y")
    upstream_name = "__grad_output"

    out = x + y
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x", "y"],
        error_input_name=upstream_name,
        accumulate_grad_outputs=True,
    )

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    y_np = np.array([[0.5, 1.5], [2.0, 2.5]], dtype=np.float16)
    grad_np = np.array([[1.0, -0.5], [0.25, 2.0]], dtype=np.float16)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, input_dtype, stream),
        "y": _host_to_gpu(y_np, input_dtype, stream),
        upstream_name: _host_to_gpu(grad_np, input_dtype, stream),
    }

    wrong_x_grad_np = np.zeros_like(x_np, dtype=np.float32)
    correct_y_grad_np = np.zeros_like(y_np, dtype=np.float16)

    outputs_gpu = {
        "x_grad": _host_to_gpu(wrong_x_grad_np, wrong_output_dtype, stream),
        "y_grad": _host_to_gpu(correct_y_grad_np, input_dtype, stream),
    }

    with pytest.raises(RuntimeError) as excinfo:
        bwd_eq.stamp(inputs_gpu, outputs_gpu, stream)

    msg = str(excinfo.value).lower()
    assert "dtype" in msg


@pytest.mark.cuda
def test_compile_backward_accumulate_grad_outputs_stamp_rejects_wrong_placement():
    dtype = thor.DataType.fp32
    x = ex.input("x")
    y = ex.input("y")
    upstream_name = "__grad_output"

    out = x - y
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(
        ["x", "y"],
        error_input_name=upstream_name,
        accumulate_grad_outputs=True,
    )

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_np = np.array([[0.5, 1.5], [2.0, 2.5]], dtype=np.float32)
    grad_np = np.array([[1.0, -0.5], [0.25, 2.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    cpu_x_grad = _cpu_tensor(list(x_np.shape), dtype)
    cpu_x_grad.numpy()[...] = 0
    gpu_y_grad = _host_to_gpu(np.zeros_like(y_np, dtype=np.float32), dtype, stream)

    outputs = {
        "x_grad": cpu_x_grad,
        "y_grad": gpu_y_grad,
    }

    with pytest.raises(RuntimeError) as excinfo:
        bwd_eq.stamp(inputs_gpu, outputs, stream)

    msg = str(excinfo.value).lower()
    assert "placement" in msg


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_runtime_scalar_forward_run_numerical(dtype: thor.DataType):
    x = ex.input("x")
    step = ex.runtime_scalar("step")

    out = (x * step) + ex.constant_scalar(1.0)
    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.5]], dtype=np.float32).astype(storage_dtype)
    step_value = 0.25

    expected = ((x_np.astype(np.float32) * step_value) + 1.0).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, dtype, stream)
    out_gpu = _gpu_tensor(list(x_np.shape), dtype, gpu_num=0)

    eq.run({
        "x": x_gpu
    }, {
        "step": step_value
    }, out_gpu, stream)

    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_runtime_scalar_as_type_fp16_casts_before_execution_numerical():
    x = ex.input("x", thor.DataType.fp32)
    step = ex.runtime_scalar("step", thor.DataType.fp16)

    # If the runtime scalar is cast to fp16 as it enters the graph, the fp32 output
    # should contain the fp16-rounded value, not the original fp32 host value.
    out = x + step
    eq = ex.compile(out, device_num=0)

    x_np = np.zeros((2, 3), dtype=np.float32)
    step_value = 0.3333
    expected_scalar = np.float32(np.float16(step_value))
    expected = np.full_like(x_np, fill_value=expected_scalar, dtype=np.float32)

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, thor.DataType.fp32, stream)
    out_gpu = _gpu_tensor(list(x_np.shape), thor.DataType.fp32, gpu_num=0)

    eq.run({
        "x": x_gpu
    }, {
        "step": step_value
    }, out_gpu, stream)

    out_host = _cpu_tensor(list(out_gpu.dimensions), thor.DataType.fp32)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_runtime_scalar_multi_output_run_numerical(dtype: thor.DataType):
    x = ex.input("x")
    step = ex.runtime_scalar("step")

    outs = ex.outputs({
        "scaled": x * step,
        "shifted": x + step,
    })

    eq = ex.compile(outs, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, -2.0, 3.5], [4.0, 5.0, -6.0]], dtype=np.float32).astype(storage_dtype)
    step_value = -1.5

    x_ref = x_np.astype(np.float32)
    expected = {
        "scaled": (x_ref * step_value).astype(storage_dtype),
        "shifted": (x_ref + step_value).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, dtype, stream)
    outputs_gpu = {
        "scaled": _gpu_tensor(list(x_np.shape), dtype, gpu_num=0),
        "shifted": _gpu_tensor(list(x_np.shape), dtype, gpu_num=0),
    }

    eq.run({
        "x": x_gpu
    }, {
        "step": step_value
    }, outputs_gpu, stream)

    for name in eq.output_names():
        out_host = _cpu_tensor(list(outputs_gpu[name].dimensions), dtype)
        out_host.copy_from_async(outputs_gpu[name], stream)
        stream.synchronize()
        got = out_host.numpy().copy()

        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_runtime_scalar_through_reduction_numerical(dtype: thor.DataType):
    x = ex.input("x")
    step = ex.runtime_scalar("step")

    loss = ex.reduce_sum(x * step, axis=[0, 1], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"])
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.5]], dtype=np.float32).astype(storage_dtype)
    step_value = -0.75

    expected = {
        "x_grad": np.full_like(x_np, fill_value=step_value, dtype=storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, {
        "step": step_value
    }, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected["x_grad"].shape
    _assert_close(got, expected["x_grad"], dtype)


@pytest.mark.cuda
def test_runtime_scalar_missing_input_rejected():
    dtype = thor.DataType.fp32

    x = ex.input("x")
    step = ex.runtime_scalar("step")
    out = x * step
    eq = ex.compile(out, device_num=0)

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    with pytest.raises(RuntimeError, match="Missing required fused equation runtime scalar input: step"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
def test_runtime_scalar_unexpected_input_rejected():
    dtype = thor.DataType.fp32

    x = ex.input("x")
    step = ex.runtime_scalar("step")
    out = x * step
    eq = ex.compile(out, device_num=0)

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    with pytest.raises(RuntimeError, match="Unexpected runtime scalar input sent to fused equation: extra"):
        eq.stamp(inputs_gpu, {
            "step": 2.0,
            "extra": 7.0
        }, stream)


@pytest.mark.cuda
def test_compile_backward_rejects_runtime_scalar_wrt():
    x = ex.input("x")
    step = ex.runtime_scalar("step")
    out = x * step

    fwd_eq = ex.compile(out, device_num=0)

    with pytest.raises(RuntimeError, match="compileBackward only supports gradients with respect to tensor inputs"):
        fwd_eq.compile_backward(["x", "step"])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_runtime_scalar_forward_run_twice_updates_value_large_flat_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    step = ex.runtime_scalar("step")

    out = x + (y * step)
    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    n = 8192

    x_np = np.linspace(-4.0, 4.0, n, dtype=np.float32).astype(storage_dtype)
    y_np = np.linspace(1.0, 3.0, n, dtype=np.float32).astype(storage_dtype)

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
    }
    out_gpu = _gpu_tensor([n], dtype, gpu_num=0)

    step_value_1 = 0.25
    expected_1 = (x_ref + (y_ref * step_value_1)).astype(storage_dtype)

    eq.run(inputs_gpu, {
        "step": step_value_1
    }, out_gpu, stream)

    out_host = _cpu_tensor([n], dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got_1 = out_host.numpy().copy()

    assert got_1.shape == expected_1.shape
    _assert_close(got_1, expected_1, dtype)

    step_value_2 = -1.75
    expected_2 = (x_ref + (y_ref * step_value_2)).astype(storage_dtype)

    eq.run(inputs_gpu, {
        "step": step_value_2
    }, out_gpu, stream)

    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got_2 = out_host.numpy().copy()

    assert got_2.shape == expected_2.shape
    _assert_close(got_2, expected_2, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_runtime_scalar_multi_output_large_flat_shared_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    step = ex.runtime_scalar("step")

    outs = ex.outputs({
        "affine": (x * step) + y,
        "shifted": y - step,
        "mixed": (x + y) * step,
    })

    eq = ex.compile(outs, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    n = 4096

    x_np = np.linspace(-2.0, 2.0, n, dtype=np.float32).astype(storage_dtype)
    y_np = np.linspace(0.5, 4.5, n, dtype=np.float32).astype(storage_dtype)
    step_value = 1.25

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)

    expected = {
        "affine": ((x_ref * step_value) + y_ref).astype(storage_dtype),
        "shifted": (y_ref - step_value).astype(storage_dtype),
        "mixed": ((x_ref + y_ref) * step_value).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
    }
    outputs_gpu = {
        "affine": _gpu_tensor([n], dtype, gpu_num=0),
        "shifted": _gpu_tensor([n], dtype, gpu_num=0),
        "mixed": _gpu_tensor([n], dtype, gpu_num=0),
    }

    eq.run(inputs_gpu, {
        "step": step_value
    }, outputs_gpu, stream)

    for name in eq.output_names():
        out_host = _cpu_tensor([n], dtype)
        out_host.copy_from_async(outputs_gpu[name], stream)
        stream.synchronize()
        got = out_host.numpy().copy()

        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_runtime_scalar_specialized_broadcast_multi_output_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    step = ex.runtime_scalar("step")

    outs = ex.outputs({
        "sum_scaled": x + (y * step),
        "prod_shifted": (x * y) + step,
    })

    eq = ex.compile(outs, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.arange(32 * 1 * 128, dtype=np.float32).reshape((32, 1, 128))
    x_np = (x_np / 97.0 - 1.5).astype(storage_dtype)

    y_np = np.arange(1 * 16 * 128, dtype=np.float32).reshape((1, 16, 128))
    y_np = (y_np / 53.0 + 0.25).astype(storage_dtype)

    step_value = -0.75

    x_ref = x_np.astype(np.float32)
    y_ref = y_np.astype(np.float32)

    expected = {
        "sum_scaled": (x_ref + (y_ref * step_value)).astype(storage_dtype),
        "prod_shifted": ((x_ref * y_ref) + step_value).astype(storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
    }
    outputs_gpu = {
        "sum_scaled": _gpu_tensor([32, 16, 128], dtype, gpu_num=0),
        "prod_shifted": _gpu_tensor([32, 16, 128], dtype, gpu_num=0),
    }

    eq.run(inputs_gpu, {
        "step": step_value
    }, outputs_gpu, stream)

    for name in eq.output_names():
        out_host = _cpu_tensor([32, 16, 128], dtype)
        out_host.copy_from_async(outputs_gpu[name], stream)
        stream.synchronize()
        got = out_host.numpy().copy()

        assert got.shape == expected[name].shape
        _assert_close(got, expected[name], dtype)


@pytest.mark.cuda
def test_runtime_scalar_as_type_fp16_casts_before_execution_large_flat_numerical():
    x = ex.input("x", thor.DataType.fp32)
    step = ex.runtime_scalar("step", thor.DataType.fp16)

    out = (x * ex.constant_scalar(2.0)) + step
    eq = ex.compile(out, device_num=0)

    n = 4096
    x_np = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    step_value = 0.3333

    step_cast = np.float32(np.float16(step_value))
    expected = ((x_np * 2.0) + step_cast).astype(np.float32)

    stream = Stream(gpu_num=0)
    x_gpu = _host_to_gpu(x_np, thor.DataType.fp32, stream)
    out_gpu = _gpu_tensor([n], thor.DataType.fp32, gpu_num=0)

    eq.run({
        "x": x_gpu
    }, {
        "step": step_value
    }, out_gpu, stream)

    out_host = _cpu_tensor([n], thor.DataType.fp32)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_compile_backward_runtime_scalar_broadcast_unbroadcast_numerical(dtype: thor.DataType):
    x = ex.input("x")
    y = ex.input("y")
    step = ex.runtime_scalar("step")

    loss = ex.reduce_sum((x + y) * step, axis=[0, 1, 2], squeeze=False)

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x", "y"])
    assert bwd_eq.output_names() == ["x_grad", "y_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)

    x_np = np.arange(32 * 1 * 64, dtype=np.float32).reshape((32, 1, 64))
    x_np = (x_np / 37.0 - 2.0).astype(storage_dtype)

    y_np = np.arange(1 * 16 * 64, dtype=np.float32).reshape((1, 16, 64))
    y_np = (y_np / 19.0 + 0.5).astype(storage_dtype)

    step_value = 0.75

    expected = {
        "x_grad": np.full_like(x_np, fill_value=(16.0 * step_value), dtype=storage_dtype),
        "y_grad": np.full_like(y_np, fill_value=(32.0 * step_value), dtype=storage_dtype),
    }

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "y": _host_to_gpu(y_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, {
        "step": step_value
    }, stream)
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
def test_reduce_sum_bf16_input_adapts_and_runs_numerical():
    dtype = thor.DataType.bf16
    x = ex.input("x")
    out = ex.reduce_sum(x, axis=1, squeeze=False)

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.sum(x_np.astype(np.float32), axis=1, keepdims=True).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output()
    out_host = _cpu_tensor(list(out_gpu.dimensions), thor.DataType.fp32)
    print(out_gpu)
    print(out_host)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        got.astype(np.float32),
        expected.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.cuda
def test_reduce_mean_bf16_input_with_bf16_compute_request_normalizes_and_runs_numerical():
    dtype = thor.DataType.bf16
    x = ex.input("x")

    # This explicitly requests a reduction compute dtype that should now be normalized
    # by the core policy rather than rejected.
    out = ex.reduce_mean(x, axis=1, squeeze=False, compute_dtype=thor.DataType.bf16)

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.mean(x_np.astype(np.float32), axis=1, keepdims=True).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output()
    out_host = _cpu_tensor(list(out_gpu.dimensions), thor.DataType.fp32)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        got.astype(np.float32),
        expected.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.cuda
def test_compile_backward_reduce_sum_bf16_input_explicit_upstream_numerical():
    dtype = thor.DataType.bf16
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.reduce_sum(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [2.0, -1.5],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.broadcast_to(grad_np.astype(np.float32)[:, None], x_np.shape).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), thor.DataType.fp32)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        got.astype(np.float32),
        expected.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.cuda
def test_compile_backward_reduce_max_bf16_input_explicit_upstream_numerical():
    dtype = thor.DataType.bf16
    x = ex.input("x")
    upstream_name = "__grad_output"

    loss = ex.reduce_max(x, axis=1, squeeze=[1])

    fwd_eq = ex.compile(loss, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [
            [[1.0, 9.0], [0.0, 3.0], [5.0, 7.0]],
            [[4.0, 2.0], [6.0, 8.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)
    grad_np = np.array(
        [[2.0, -1.5], [0.25, 3.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.zeros_like(x_np, dtype=storage_dtype)
    expected[0, 2, 0] = np.array(2.0, dtype=storage_dtype)
    expected[0, 0, 1] = np.array(-1.5, dtype=storage_dtype)
    expected[1, 1, 0] = np.array(0.25, dtype=storage_dtype)
    expected[1, 1, 1] = np.array(3.0, dtype=storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output("x_grad")
    out_host = _cpu_tensor(list(out_gpu.dimensions), thor.DataType.fp32)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        got.astype(np.float32),
        expected.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.cuda
def test_reduce_sum_fp8_e4m3_input_adapts_and_runs_numerical():
    dtype = thor.DataType.fp8_e4m3
    x = ex.input("x")
    out = ex.reduce_sum(x, axis=1, squeeze=False)

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.sum(x_np.astype(np.float32), axis=1, keepdims=True).astype(storage_dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output()
    out_host = _cpu_tensor(list(out_gpu.dimensions), thor.DataType.fp32)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        got.astype(np.float32),
        expected.astype(np.float32),
        rtol=2e-1,
        atol=2e-1,
    )


@pytest.mark.cuda
def test_reduce_argmin_bf16_input_adapts_and_runs_numerical():
    input_dtype = thor.DataType.bf16
    output_dtype = thor.DataType.uint32

    x = ex.input("x")
    out = ex.argmin(x, axis=1, squeeze=[1])

    eq = ex.compile(out, device_num=0)

    storage_dtype = _numpy_storage_dtype(input_dtype)
    x_np = np.array(
        [
            [[1.0, 9.0], [0.0, 3.0], [5.0, 7.0]],
            [[4.0, 2.0], [6.0, 8.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    ).astype(storage_dtype)

    expected = np.argmin(x_np.astype(np.float32), axis=1).astype(np.uint32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, input_dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    out_gpu = stamped.output(eq.output_names()[0])
    out_host = _cpu_tensor(list(out_gpu.dimensions), output_dtype)
    out_host.copy_from_async(out_gpu, stream)
    stream.synchronize()
    got = out_host.numpy().copy()

    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)
