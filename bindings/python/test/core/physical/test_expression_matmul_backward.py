import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

MATMUL_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.fp32,
]


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
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
def test_compile_backward_pointwise_over_transposed_matmul_numerical(dtype: thor.DataType):
    upstream_name = "__grad_output"

    a = ex.input("a")
    b = ex.input("b")
    expr = ex.exp(ex.matmul(a, b, transpose_a=True, transpose_b=True) * 0.25 + 0.5)

    fwd_eq = ex.compile(expr, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["a_grad", "b_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 2.0]], dtype=np.float32).astype(storage_dtype)  # 3x2
    b_np = np.array([[1.5, -0.5, 2.0], [0.25, 1.0, -1.0], [2.5, -2.0, 0.75], [-1.5, 0.5, 1.25]],
                    dtype=np.float32).astype(storage_dtype)  # 4x3
    upstream_np = np.array([[1.0, -0.5, 0.25, 2.0], [-1.0, 0.75, 1.5, -0.25]], dtype=np.float32).astype(storage_dtype)

    a_ref = a_np.astype(np.float32)
    b_ref = b_np.astype(np.float32)
    upstream_ref = upstream_np.astype(np.float32)

    mm = a_ref.T @ b_ref.T
    out = np.exp(mm * 0.25 + 0.5)
    grad_mm = upstream_ref * out * 0.25

    expected_a_grad = b_ref.T @ grad_mm.T
    expected_b_grad = grad_mm.T @ a_ref.T

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(upstream_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b = _copy_to_host(stamped.output("b_grad"), dtype, stream)

    _assert_close(got_a, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
def test_compile_backward_gemm_alpha_beta_numerical(dtype: thor.DataType):
    upstream_name = "__grad_output"

    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = 1.25
    beta = -0.5

    fwd_eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b", "c"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["a_grad", "b_grad", "c_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]], dtype=np.float32).astype(storage_dtype)
    b_np = np.array([[2.0, -1.0, 0.0, 1.0], [0.5, 3.0, -2.0, 0.25], [-1.5, 2.0, 4.0, -0.5]],
                    dtype=np.float32).astype(storage_dtype)
    c_np = np.array([[0.25, -1.0, 2.0, 0.5], [1.25, 0.75, -0.5, 3.0]], dtype=np.float32).astype(storage_dtype)
    upstream_np = np.array([[1.0, -0.25, 0.5, 2.0], [-1.5, 0.75, 1.25, -0.5]], dtype=np.float32).astype(storage_dtype)

    a_ref = a_np.astype(np.float32)
    b_ref = b_np.astype(np.float32)
    upstream_ref = upstream_np.astype(np.float32)

    expected_a_grad = alpha * (upstream_ref @ b_ref.T)
    expected_b_grad = alpha * (a_ref.T @ upstream_ref)
    expected_c_grad = beta * upstream_ref

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        upstream_name: _host_to_gpu(upstream_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b = _copy_to_host(stamped.output("b_grad"), dtype, stream)
    got_c = _copy_to_host(stamped.output("c_grad"), dtype, stream)

    _assert_close(got_a, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)
    _assert_close(got_c, _cast_reference_to_storage_dtype(expected_c_grad, dtype), dtype)
