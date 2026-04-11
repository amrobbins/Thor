import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

MATMUL_DTYPES = [
    thor.DataType.fp16,
    # thor.DataType.bf16,
    thor.DataType.fp32,
]


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _gpu_tensor(shape: list[int], dtype: thor.DataType, gpu_num: int = 0) -> PhysicalTensor:
    placement = Placement(DeviceType.gpu, gpu_num)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _cast_reference_to_storage_dtype(values: np.ndarray, dtype: thor.DataType) -> np.ndarray:
    return values.astype(np.float32).astype(_numpy_storage_dtype(dtype))


def _assert_close(got: np.ndarray, expected: np.ndarray, dtype: thor.DataType):
    got32 = got.astype(np.float32)
    expected32 = expected.astype(np.float32)

    if dtype == thor.DataType.fp32:
        np.testing.assert_allclose(got32, expected32, rtol=1e-4, atol=1e-5)
    elif dtype == thor.DataType.fp16:
        np.testing.assert_allclose(got32, expected32, rtol=5e-2, atol=5e-2)
    elif dtype == thor.DataType.bf16:
        np.testing.assert_allclose(got32, expected32, rtol=7e-2, atol=7e-2)
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
    host = _cpu_tensor(list(tensor.dimensions), dtype)
    host.copy_from_async(tensor, stream)
    stream.synchronize()
    return host.numpy().copy()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_matmul_dunder_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    eq = ex.compile(a @ b, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(storage_dtype)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
                    dtype=np.float32).astype(storage_dtype)

    expected = a_np.astype(np.float32) @ b_np.astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_matmul_transpose_followed_by_pointwise_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    expr = ex.exp(ex.matmul(a, b, transpose_a=True, transpose_b=True) * 0.25 + 0.5)
    eq = ex.compile(expr, device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 2.0]], dtype=np.float32).astype(storage_dtype)
    b_np = np.array([[1.5, -0.5, 2.0], [0.25, 1.0, -1.0], [2.5, -2.0, 0.75], [-1.5, 0.5, 1.25]],
                    dtype=np.float32).astype(storage_dtype)

    mm = a_np.astype(np.float32).T @ b_np.astype(np.float32).T
    expected = np.exp(mm * 0.25 + 0.5)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_gemm_alpha_beta_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = 1.25
    beta = -0.5
    eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(storage_dtype)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
                    dtype=np.float32).astype(storage_dtype)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32).astype(storage_dtype)

    expected = alpha * (a_np.astype(np.float32) @ b_np.astype(np.float32)) + beta * c_np.astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    assert eq.output_shape(inputs_gpu) == [2, 4]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, dtype), dtype)


@pytest.mark.cuda
def test_gemm_transpose_ab_and_preallocated_output_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = 0.75
    beta = 1.5

    expr = ex.gemm(
        a,
        b,
        c,
        alpha=alpha,
        beta=beta,
        transpose_a=True,
        transpose_b=True,
    )
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 2.0]], dtype=np.float32)  # 3x2
    b_np = np.array(
        [[1.5, -0.5, 2.0], [0.25, 1.0, -1.0], [2.5, -2.0, 0.75], [-1.5, 0.5, 1.25]], dtype=np.float32)  # 4x3
    c_np = np.array([[0.2, -1.0, 0.5, 2.0], [-0.75, 0.25, 1.5, -0.5]], dtype=np.float32)  # 2x4

    expected = alpha * (a_np.T @ b_np.T) + beta * c_np

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }
    preallocated_output = _gpu_tensor([2, 4], dtype)

    stamped = eq.stamp(inputs_gpu, stream, preallocated_output=preallocated_output)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_matmul_output_dtype_override_and_preallocated_output_numerical():
    input_dtype = thor.DataType.fp16
    output_dtype = thor.DataType.fp16

    a = ex.input("a")
    b = ex.input("b")
    expr = ex.matmul(a, b, compute_dtype=thor.DataType.fp32, output_dtype=output_dtype)
    eq = ex.compile(expr, device_num=0)

    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(_numpy_storage_dtype(input_dtype))
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
                    dtype=np.float32).astype(_numpy_storage_dtype(input_dtype))
    expected = a_np.astype(np.float32) @ b_np.astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, input_dtype, stream),
        "b": _host_to_gpu(b_np, input_dtype, stream),
    }
    preallocated_output = _gpu_tensor([2, 4], output_dtype)

    stamped = eq.stamp(inputs_gpu, stream, preallocated_output=preallocated_output)
    stamped.run()

    got = _copy_to_host(stamped.output(), output_dtype, stream)
    _assert_close(got, expected, output_dtype)


@pytest.mark.cuda
def test_matmul_rejects_incompatible_dimensions_at_stamp_time():
    a = ex.input("a")
    b = ex.input("b")
    eq = ex.compile(ex.matmul(a, b), device_num=0)

    dtype = thor.DataType.fp32
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(np.ones((2, 3), dtype=np.float32), dtype, stream),
        "b": _host_to_gpu(np.ones((5, 4), dtype=np.float32), dtype, stream),
    }

    with pytest.raises(RuntimeError, match="incompatible matrix dimensions"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
def test_gemm_rejects_incompatible_addend_dimensions_at_stamp_time():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    eq = ex.compile(ex.gemm(a, b, c), device_num=0)

    dtype = thor.DataType.fp32
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(np.ones((2, 3), dtype=np.float32), dtype, stream),
        "b": _host_to_gpu(np.ones((3, 4), dtype=np.float32), dtype, stream),
        "c": _host_to_gpu(np.ones((4, 2), dtype=np.float32), dtype, stream),
    }

    with pytest.raises(RuntimeError, match="addend tensor dimensions are incompatible"):
        eq.stamp(inputs_gpu, stream)


def test_matmul_rdunder():
    a = ex.input("a")
    b = ex.input("b")
    expr1 = a.__rmatmul__(b)
    expr2 = ex.matmul(b, a)
    eq1 = ex.compile(expr1, device_num=0)
    eq2 = ex.compile(expr2, device_num=0)
    assert eq1 is not None
    assert eq2 is not None


def test_matmul_idunder():
    a = ex.input("a")
    b = ex.input("b")
    c = a.__imatmul__(b)
    eq = ex.compile(c, device_num=0)
    assert eq is not None
