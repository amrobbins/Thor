import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

MATMUL_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.fp32,
]

BACKWARD_DTYPES = [
    thor.DataType.fp16,
    thor.DataType.fp32,
]

TRANSPOSE_CASES = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
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


def _make_device_scalar(value: float, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    arr = np.array([value], dtype=np.float32)
    return _host_to_gpu(arr, thor.DataType.fp32, stream, gpu_num=gpu_num)


def _write_device_scalar(buffer: PhysicalTensor, value: float, stream: Stream):
    host = _cpu_tensor([1], thor.DataType.fp32)
    host.numpy()[0] = np.float32(value)
    buffer.copy_from_async(host, stream)


def _tensor_scalar_inputs(**bindings: PhysicalTensor) -> dict[str, object]:
    return {
        name: thor.physical.TensorScalarBinding(buffer, 0, thor.DataType.fp32) for name, buffer in bindings.items()
    }


def _matmul_case_arrays(transpose_a: bool,
                        transpose_b: bool,
                        dtype: thor.DataType,
                        m: int = 2,
                        k: int = 3,
                        n: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)

    a_np = np.arange(1, 1 + np.prod(a_shape), dtype=np.float32).reshape(a_shape) / 4.0 - 1.0
    b_np = np.arange(1, 1 + np.prod(b_shape), dtype=np.float32).reshape(b_shape) / 5.0 - 0.75
    c_np = np.arange(1, 1 + (m * n), dtype=np.float32).reshape((m, n)) / 6.0 - 0.5
    grad_np = np.arange(1, 1 + (m * n), dtype=np.float32).reshape((m, n)) / 7.0 - 0.5

    storage_dtype = _numpy_storage_dtype(dtype)
    return (
        a_np.astype(storage_dtype),
        b_np.astype(storage_dtype),
        c_np.astype(storage_dtype),
        grad_np.astype(storage_dtype),
    )


def _matmul_reference(a: np.ndarray, b: np.ndarray, transpose_a: bool = False, transpose_b: bool = False) -> np.ndarray:
    a_ref = a.astype(np.float32)
    b_ref = b.astype(np.float32)
    a_eff = a_ref.T if transpose_a else a_ref
    b_eff = b_ref.T if transpose_b else b_ref
    return a_eff @ b_eff


def _gemm_reference(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
        alpha: float = 1.0,
        beta: float = 1.0) -> np.ndarray:
    return alpha * _matmul_reference(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b) + beta * c.astype(np.float32)


def _matmul_backward_reference(
        a: np.ndarray,
        b: np.ndarray,
        grad: np.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
        alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    a_ref = a.astype(np.float32)
    b_ref = b.astype(np.float32)
    grad_ref = grad.astype(np.float32)

    a_eff = a_ref.T if transpose_a else a_ref
    b_eff = b_ref.T if transpose_b else b_ref

    da_eff = alpha * (grad_ref @ b_eff.T)
    db_eff = alpha * (a_eff.T @ grad_ref)

    da = da_eff.T if transpose_a else da_eff
    db = db_eff.T if transpose_b else db_eff
    return da, db


def _gemm_backward_reference(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        grad: np.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
        alpha: float = 1.0,
        beta: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    da, db = _matmul_backward_reference(a, b, grad, transpose_a=transpose_a, transpose_b=transpose_b, alpha=alpha)
    dc = beta * grad.astype(np.float32)
    return da, db, dc


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", MATMUL_DTYPES)
@pytest.mark.parametrize("transpose_a,transpose_b", TRANSPOSE_CASES)
def test_matmul_forward_all_transpose_combinations_numerical(
        dtype: thor.DataType, transpose_a: bool, transpose_b: bool):
    a = ex.input("a")
    b = ex.input("b")
    eq = ex.compile(ex.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b), device_num=0)

    a_np, b_np, _, _ = _matmul_case_arrays(transpose_a, transpose_b, dtype)
    expected = _matmul_reference(a_np, b_np, transpose_a=transpose_a, transpose_b=transpose_b)

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
@pytest.mark.parametrize("transpose_a,transpose_b", TRANSPOSE_CASES)
def test_gemm_forward_explicit_all_transpose_combinations_numerical(
        dtype: thor.DataType, transpose_a: bool, transpose_b: bool):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = 1.25
    beta = -0.5
    eq = ex.compile(
        ex.gemm(a, b, c, alpha=alpha, beta=beta, transpose_a=transpose_a, transpose_b=transpose_b),
        device_num=0,
    )

    a_np, b_np, c_np, _ = _matmul_case_arrays(transpose_a, transpose_b, dtype)
    expected = _gemm_reference(
        a_np, b_np, c_np, transpose_a=transpose_a, transpose_b=transpose_b, alpha=alpha, beta=beta)

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
@pytest.mark.parametrize("transpose_a,transpose_b", TRANSPOSE_CASES)
def test_operator_lowered_gemm_forward_constant_scales_all_transpose_combinations_numerical(
        transpose_a: bool, transpose_b: bool):
    dtype = thor.DataType.fp32
    alpha = 0.75
    beta = -1.25

    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    expr = ex.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b) * alpha + c * beta
    eq = ex.compile(expr, device_num=0)

    a_np, b_np, c_np, _ = _matmul_case_arrays(transpose_a, transpose_b, dtype)
    expected = _gemm_reference(
        a_np, b_np, c_np, transpose_a=transpose_a, transpose_b=transpose_b, alpha=alpha, beta=beta)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), dtype, stream)
    _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_forward_runtime_scalars_explicit_api_reused_across_runs_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.runtime_scalar("alpha")
    beta = ex.runtime_scalar("beta")
    eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)

    a_np, b_np, c_np, _ = _matmul_case_arrays(False, False, dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)

    for alpha_v, beta_v in [(0.75, -1.25), (-0.5, 0.25), (1.5, 0.0)]:
        stamped.run({
            "alpha": alpha_v,
            "beta": beta_v
        })
        got = _copy_to_host(stamped.output(), dtype, stream)
        expected = _gemm_reference(a_np, b_np, c_np, alpha=alpha_v, beta=beta_v)
        _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_operator_lowered_gemm_forward_runtime_scalars_reused_across_runs_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.runtime_scalar("alpha")
    beta = ex.runtime_scalar("beta")
    eq = ex.compile((a @ b) * alpha + c * beta, device_num=0)

    a_np, b_np, c_np, _ = _matmul_case_arrays(False, False, dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)

    for alpha_v, beta_v in [(0.75, -1.25), (-0.5, 0.25), (0.0, 2.0)]:
        stamped.run({
            "alpha": alpha_v,
            "beta": beta_v
        })
        got = _copy_to_host(stamped.output(), dtype, stream)
        expected = _gemm_reference(a_np, b_np, c_np, alpha=alpha_v, beta=beta_v)
        _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_forward_tensor_runtime_scalars_explicit_api_reused_across_runs_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.tensor_runtime_scalar("alpha")
    beta = ex.tensor_runtime_scalar("beta")
    eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)

    a_np, b_np, c_np, _ = _matmul_case_arrays(False, False, dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    alpha_buffer = _make_device_scalar(0.75, stream)
    beta_buffer = _make_device_scalar(-1.25, stream)
    tensor_scalar_inputs = _tensor_scalar_inputs(alpha=alpha_buffer, beta=beta_buffer)

    stamped = eq.stamp(inputs_gpu, stream, tensor_scalar_inputs=tensor_scalar_inputs)

    for alpha_v, beta_v in [(0.75, -1.25), (-0.5, 0.25), (1.5, 0.0)]:
        _write_device_scalar(alpha_buffer, alpha_v, stream)
        _write_device_scalar(beta_buffer, beta_v, stream)
        stamped.run()
        got = _copy_to_host(stamped.output(), dtype, stream)
        expected = _gemm_reference(a_np, b_np, c_np, alpha=alpha_v, beta=beta_v)
        _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_forward_tensor_runtime_scalars_with_extra_host_scaling_reused_across_runs_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.tensor_runtime_scalar("alpha") * 0.5
    beta = ex.tensor_runtime_scalar("beta") * -2.0
    eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)

    a_np, b_np, c_np, _ = _matmul_case_arrays(False, False, dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    alpha_buffer = _make_device_scalar(1.5, stream)
    beta_buffer = _make_device_scalar(0.625, stream)
    tensor_scalar_inputs = _tensor_scalar_inputs(alpha=alpha_buffer, beta=beta_buffer)

    stamped = eq.stamp(inputs_gpu, stream, tensor_scalar_inputs=tensor_scalar_inputs)

    for alpha_v, beta_v in [(1.5, 0.625), (-1.0, -0.25), (2.0, 0.0)]:
        _write_device_scalar(alpha_buffer, alpha_v, stream)
        _write_device_scalar(beta_buffer, beta_v, stream)
        stamped.run()
        got = _copy_to_host(stamped.output(), dtype, stream)
        expected = _gemm_reference(a_np, b_np, c_np, alpha=alpha_v * 0.5, beta=beta_v * -2.0)
        _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_forward_mixed_tensor_runtime_scalar_and_host_runtime_scalar_reused_across_runs_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = ex.tensor_runtime_scalar("alpha")
    beta = ex.runtime_scalar("beta")
    eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)

    a_np, b_np, c_np, _ = _matmul_case_arrays(False, False, dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }

    alpha_buffer = _make_device_scalar(0.75, stream)
    tensor_scalar_inputs = _tensor_scalar_inputs(alpha=alpha_buffer)

    stamped = eq.stamp(inputs_gpu, stream, tensor_scalar_inputs=tensor_scalar_inputs)

    for alpha_v, beta_v in [(0.75, -1.25), (-0.5, 0.25), (1.25, 0.0)]:
        _write_device_scalar(alpha_buffer, alpha_v, stream)
        stamped.run({
            "beta": beta_v
        })
        got = _copy_to_host(stamped.output(), dtype, stream)
        expected = _gemm_reference(a_np, b_np, c_np, alpha=alpha_v, beta=beta_v)
        _assert_close(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_forward_preallocated_output_reuse_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    alpha = 0.75
    beta = 1.5
    eq = ex.compile(
        ex.gemm(a, b, c, alpha=alpha, beta=beta, transpose_a=True, transpose_b=True),
        device_num=0,
    )

    a_np, b_np, c_np, _ = _matmul_case_arrays(True, True, dtype)
    expected = _gemm_reference(a_np, b_np, c_np, transpose_a=True, transpose_b=True, alpha=alpha, beta=beta)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
    }
    preallocated_output = _gpu_tensor([2, 4], dtype)

    stamped = eq.stamp(inputs_gpu, stream, preallocated_output=preallocated_output)
    stamped.run()
    got1 = _copy_to_host(preallocated_output, dtype, stream)
    _assert_close(got1, expected, dtype)

    stamped.run()
    got2 = _copy_to_host(preallocated_output, dtype, stream)
    _assert_close(got2, expected, dtype)


@pytest.mark.cuda
def test_matmul_forward_output_dtype_override_and_preallocated_output_numerical():
    input_dtype = thor.DataType.fp16
    output_dtype = thor.DataType.fp16

    a = ex.input("a")
    b = ex.input("b")
    eq = ex.compile(ex.matmul(a, b, compute_dtype=thor.DataType.fp32, output_dtype=output_dtype), device_num=0)

    a_np, b_np, _, _ = _matmul_case_arrays(False, False, input_dtype)
    expected = _matmul_reference(a_np, b_np)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, input_dtype, stream),
        "b": _host_to_gpu(b_np, input_dtype, stream),
    }
    preallocated_output = _gpu_tensor([2, 4], output_dtype)

    stamped = eq.stamp(inputs_gpu, stream, preallocated_output=preallocated_output)
    stamped.run()

    got = _copy_to_host(preallocated_output, output_dtype, stream)
    _assert_close(got, _cast_reference_to_storage_dtype(expected, output_dtype), output_dtype)


@pytest.mark.cuda
def test_matmul_forward_rejects_incompatible_dimensions_at_stamp_time():
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
def test_gemm_forward_rejects_incompatible_addend_dimensions_at_stamp_time():
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

    with pytest.raises(RuntimeError, match="addend|dimensions|incompatible"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
def test_gemm_forward_transpose_c_rejects_in_staged_path():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    eq = ex.compile(ex.gemm(a, b, c, transpose_c=True), device_num=0)

    dtype = thor.DataType.fp32
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(np.ones((2, 3), dtype=np.float32), dtype, stream),
        "b": _host_to_gpu(np.ones((3, 4), dtype=np.float32), dtype, stream),
        "c": _host_to_gpu(np.ones((4, 2), dtype=np.float32), dtype, stream),
    }

    with pytest.raises(RuntimeError, match="transpose_aux|transposeC|transpose_c"):
        eq.stamp(inputs_gpu, stream)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("transpose_a,transpose_b", TRANSPOSE_CASES)
def test_matmul_backward_all_transpose_combinations_numerical(
        dtype: thor.DataType, transpose_a: bool, transpose_b: bool):
    a = ex.input("a")
    b = ex.input("b")
    upstream_name = "__grad_output"

    fwd_eq = ex.compile(ex.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["a_grad", "b_grad"]

    a_np, b_np, _, grad_np = _matmul_case_arrays(transpose_a, transpose_b, dtype)
    expected_a_grad, expected_b_grad = _matmul_backward_reference(
        a_np,
        b_np,
        grad_np,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)

    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b_grad, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_matmul_backward_pointwise_composition_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    upstream_name = "__grad_output"

    out = ex.exp(ex.matmul(a, b, transpose_a=True, transpose_b=False) * 0.25 + 0.5)
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b"], error_input_name=upstream_name)

    a_np, b_np, _, grad_np = _matmul_case_arrays(True, False, dtype)
    mm = _matmul_reference(a_np, b_np, transpose_a=True, transpose_b=False)
    local_grad = grad_np.astype(np.float32) * np.exp(mm * 0.25 + 0.5) * 0.25
    expected_a_grad, expected_b_grad = _matmul_backward_reference(
        a_np,
        b_np,
        local_grad,
        transpose_a=True,
        transpose_b=False,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)

    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b_grad, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_matmul_backward_requested_subset_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    upstream_name = "__grad_output"

    fwd_eq = ex.compile(ex.matmul(a, b, transpose_a=False, transpose_b=True), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["a_grad"]

    a_np, b_np, _, grad_np = _matmul_case_arrays(False, True, dtype)
    expected_a_grad, _ = _matmul_backward_reference(a_np, b_np, grad_np, transpose_a=False, transpose_b=True)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_matmul_backward_accumulate_grad_outputs_stamp_uses_provided_accumulators_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    upstream_name = "__grad_output"

    fwd_eq = ex.compile(ex.matmul(a, b), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b"], error_input_name=upstream_name, accumulate_grad_outputs=True)

    a_np, b_np, _, grad_np = _matmul_case_arrays(False, False, dtype)
    computed_a_grad, computed_b_grad = _matmul_backward_reference(a_np, b_np, grad_np)

    prefill_a_grad_np = np.array([[5.0, 4.0, 3.0], [2.0, 1.0, 0.0]],
                                 dtype=np.float32).astype(_numpy_storage_dtype(dtype))
    prefill_b_grad_np = np.array(
        [[-1.0, -2.0, -3.0, -4.0], [1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 0.25, -0.25]],
        dtype=np.float32,
    ).astype(_numpy_storage_dtype(dtype))

    expected_a = _cast_reference_to_storage_dtype(prefill_a_grad_np.astype(np.float32) + computed_a_grad, dtype)
    expected_b = _cast_reference_to_storage_dtype(prefill_b_grad_np.astype(np.float32) + computed_b_grad, dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }
    outputs_gpu = {
        "a_grad": _host_to_gpu(prefill_a_grad_np, dtype, stream),
        "b_grad": _host_to_gpu(prefill_b_grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream=stream, preallocated_outputs=outputs_gpu)
    stamped.run()

    got_a_grad = _copy_to_host(outputs_gpu["a_grad"], dtype, stream)
    got_b_grad = _copy_to_host(outputs_gpu["b_grad"], dtype, stream)

    _assert_close(got_a_grad, expected_a, dtype)
    _assert_close(got_b_grad, expected_b, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("transpose_a,transpose_b", TRANSPOSE_CASES)
def test_gemm_backward_explicit_all_transpose_combinations_numerical(
        dtype: thor.DataType, transpose_a: bool, transpose_b: bool):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    upstream_name = "__grad_output"
    alpha = 1.25
    beta = -0.5

    fwd_eq = ex.compile(
        ex.gemm(a, b, c, alpha=alpha, beta=beta, transpose_a=transpose_a, transpose_b=transpose_b),
        device_num=0,
    )
    bwd_eq = fwd_eq.compile_backward(["a", "b", "c"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["a_grad", "b_grad", "c_grad"]

    a_np, b_np, c_np, grad_np = _matmul_case_arrays(transpose_a, transpose_b, dtype)
    expected_a_grad, expected_b_grad, expected_c_grad = _gemm_backward_reference(
        a_np,
        b_np,
        c_np,
        grad_np,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        alpha=alpha,
        beta=beta,
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)
    got_c_grad = _copy_to_host(stamped.output("c_grad"), dtype, stream)

    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b_grad, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)
    _assert_close(got_c_grad, _cast_reference_to_storage_dtype(expected_c_grad, dtype), dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_gemm_backward_accumulate_grad_outputs_stamp_uses_provided_accumulators_numerical(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    upstream_name = "__grad_output"
    alpha = 0.75
    beta = -1.25

    fwd_eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b", "c"], error_input_name=upstream_name, accumulate_grad_outputs=True)

    a_np, b_np, c_np, grad_np = _matmul_case_arrays(False, False, dtype)
    computed_a_grad, computed_b_grad, computed_c_grad = _gemm_backward_reference(
        a_np, b_np, c_np, grad_np, alpha=alpha, beta=beta)

    prefill_a_grad_np = np.array([[1.5, -0.5, 0.25], [0.75, -1.25, 0.5]],
                                 dtype=np.float32).astype(_numpy_storage_dtype(dtype))
    prefill_b_grad_np = np.array(
        [[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8], [0.9, -1.0, 1.1, -1.2]],
        dtype=np.float32).astype(_numpy_storage_dtype(dtype))
    prefill_c_grad_np = np.array([[2.0, -1.0, 0.5, -0.25], [1.25, -0.75, 0.25, -0.125]],
                                 dtype=np.float32).astype(_numpy_storage_dtype(dtype))

    expected_a = _cast_reference_to_storage_dtype(prefill_a_grad_np.astype(np.float32) + computed_a_grad, dtype)
    expected_b = _cast_reference_to_storage_dtype(prefill_b_grad_np.astype(np.float32) + computed_b_grad, dtype)
    expected_c = _cast_reference_to_storage_dtype(prefill_c_grad_np.astype(np.float32) + computed_c_grad, dtype)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }
    outputs_gpu = {
        "a_grad": _host_to_gpu(prefill_a_grad_np, dtype, stream),
        "b_grad": _host_to_gpu(prefill_b_grad_np, dtype, stream),
        "c_grad": _host_to_gpu(prefill_c_grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream=stream, preallocated_outputs=outputs_gpu)
    stamped.run()

    got_a_grad = _copy_to_host(outputs_gpu["a_grad"], dtype, stream)
    got_b_grad = _copy_to_host(outputs_gpu["b_grad"], dtype, stream)
    got_c_grad = _copy_to_host(outputs_gpu["c_grad"], dtype, stream)

    _assert_close(got_a_grad, expected_a, dtype)
    _assert_close(got_b_grad, expected_b, dtype)
    _assert_close(got_c_grad, expected_c, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_gemm_backward_beta_zero_aux_grad_is_zero(dtype: thor.DataType):
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    upstream_name = "__grad_output"

    fwd_eq = ex.compile(ex.gemm(a, b, c, alpha=0.75, beta=0.0), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["c"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["c_grad"]

    a_np, b_np, c_np, grad_np = _matmul_case_arrays(False, False, dtype)
    expected_c_grad = np.zeros_like(c_np.astype(np.float32))

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_c_grad = _copy_to_host(stamped.output("c_grad"), dtype, stream)
    _assert_close(got_c_grad, _cast_reference_to_storage_dtype(expected_c_grad, dtype), dtype)


@pytest.mark.cuda
def test_gemm_backward_runtime_scalars_reused_across_runs_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    upstream_name = "__grad_output"
    alpha = ex.runtime_scalar("alpha")
    beta = ex.runtime_scalar("beta")

    fwd_eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b", "c"], error_input_name=upstream_name)

    a_np, b_np, c_np, grad_np = _matmul_case_arrays(False, False, dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)

    for alpha_v, beta_v in [(0.75, -1.25), (-0.5, 0.25), (1.5, 0.0)]:
        stamped.run({
            "alpha": alpha_v,
            "beta": beta_v
        })
        got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
        got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)
        got_c_grad = _copy_to_host(stamped.output("c_grad"), dtype, stream)

        expected_a_grad, expected_b_grad, expected_c_grad = _gemm_backward_reference(
            a_np, b_np, c_np, grad_np, alpha=alpha_v, beta=beta_v)
        _assert_close(got_a_grad, expected_a_grad, dtype)
        _assert_close(got_b_grad, expected_b_grad, dtype)
        _assert_close(got_c_grad, expected_c_grad, dtype)


@pytest.mark.cuda
def test_gemm_backward_tensor_runtime_scalars_with_extra_host_scaling_reused_across_runs_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    upstream_name = "__grad_output"
    alpha = ex.tensor_runtime_scalar("alpha") * 0.5
    beta = ex.tensor_runtime_scalar("beta") * -2.0

    fwd_eq = ex.compile(ex.gemm(a, b, c, alpha=alpha, beta=beta), device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b", "c"], error_input_name=upstream_name)

    a_np, b_np, c_np, grad_np = _matmul_case_arrays(False, False, dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    alpha_buffer = _make_device_scalar(1.5, stream)
    beta_buffer = _make_device_scalar(0.625, stream)
    tensor_scalar_inputs = _tensor_scalar_inputs(alpha=alpha_buffer, beta=beta_buffer)

    stamped = bwd_eq.stamp(inputs_gpu, stream, tensor_scalar_inputs=tensor_scalar_inputs)

    for alpha_v, beta_v in [(1.5, 0.625), (-1.0, -0.25), (2.0, 0.0)]:
        _write_device_scalar(alpha_buffer, alpha_v, stream)
        _write_device_scalar(beta_buffer, beta_v, stream)
        stamped.run()

        got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
        got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)
        got_c_grad = _copy_to_host(stamped.output("c_grad"), dtype, stream)

        expected_a_grad, expected_b_grad, expected_c_grad = _gemm_backward_reference(
            a_np,
            b_np,
            c_np,
            grad_np,
            alpha=alpha_v * 0.5,
            beta=beta_v * -2.0,
        )
        _assert_close(got_a_grad, expected_a_grad, dtype)
        _assert_close(got_b_grad, expected_b_grad, dtype)
        _assert_close(got_c_grad, expected_c_grad, dtype)


@pytest.mark.cuda
def test_operator_lowered_gemm_backward_runtime_scalars_reused_across_runs_numerical():
    dtype = thor.DataType.fp32
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    upstream_name = "__grad_output"
    alpha = ex.runtime_scalar("alpha")
    beta = ex.runtime_scalar("beta")

    fwd_eq = ex.compile((a @ b) * alpha + c * beta, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b", "c"], error_input_name=upstream_name)

    a_np, b_np, c_np, grad_np = _matmul_case_arrays(False, False, dtype)
    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        "c": _host_to_gpu(c_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)

    for alpha_v, beta_v in [(0.75, -1.25), (-0.5, 0.25), (1.5, 0.0)]:
        stamped.run({
            "alpha": alpha_v,
            "beta": beta_v
        })

        got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
        got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)
        got_c_grad = _copy_to_host(stamped.output("c_grad"), dtype, stream)

        expected_a_grad, expected_b_grad, expected_c_grad = _gemm_backward_reference(
            a_np, b_np, c_np, grad_np, alpha=alpha_v, beta=beta_v)
        _assert_close(got_a_grad, expected_a_grad, dtype)
        _assert_close(got_b_grad, expected_b_grad, dtype)
        _assert_close(got_c_grad, expected_c_grad, dtype)


@pytest.mark.cuda
def test_gemm_backward_rejects_transpose_c():
    a = ex.input("a")
    b = ex.input("b")
    c = ex.input("c")
    fwd_eq = ex.compile(ex.gemm(a, b, c, transpose_c=True), device_num=0)

    with pytest.raises(RuntimeError, match="transpose_aux|transposeC|transpose_c"):
        fwd_eq.compile_backward(["c"], error_input_name="__grad_output")


@pytest.mark.cuda
def test_matmul_backward_large_fp16_likely_workspace_numerical():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"

    m = 512
    k = 768
    n = 640

    a = ex.input("a")
    b = ex.input("b")
    out = ex.exp((a @ b) * 0.125 - 0.25)

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["a", "b"], error_input_name=upstream_name)

    a_ref = (((np.arange(m * k, dtype=np.float32).reshape(m, k) % 23.0) - 11.0) / 64.0)
    b_ref = (((np.arange(k * n, dtype=np.float32).reshape(k, n) % 19.0) - 9.0) / 64.0)
    upstream_ref = (((np.arange(m * n, dtype=np.float32).reshape(m, n) % 17.0) - 8.0) / 32.0)

    storage_dtype = _numpy_storage_dtype(dtype)
    a_np = a_ref.astype(storage_dtype)
    b_np = b_ref.astype(storage_dtype)
    grad_np = upstream_ref.astype(storage_dtype)

    mm = a_ref @ b_ref
    local_grad = upstream_ref * np.exp(mm * 0.125 - 0.25) * 0.125
    expected_a_grad = local_grad @ b_ref.T
    expected_b_grad = a_ref.T @ local_grad

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "a": _host_to_gpu(a_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_a_grad = _copy_to_host(stamped.output("a_grad"), dtype, stream)
    got_b_grad = _copy_to_host(stamped.output("b_grad"), dtype, stream)

    _assert_close(got_a_grad, _cast_reference_to_storage_dtype(expected_a_grad, dtype), dtype)
    _assert_close(got_b_grad, _cast_reference_to_storage_dtype(expected_b_grad, dtype), dtype)
